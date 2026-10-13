import argparse
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return normalized.type_as(x) * self.weight


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)


class TransformerBlock(nn.Module):
    def __init__(self, args, hidden_dim):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args.dim, hidden_dim)
        self.attention_norm = RMSNorm(args.dim, args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, args.norm_eps)


class CachedTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_dim = args.hidden_dim
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * args.dim) / 3)
            hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        self.hidden_dim = hidden_dim
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList(TransformerBlock(args, hidden_dim) for _ in range(args.n_layers))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        self.tok_embeddings.weight = self.output.weight

        head_dim = args.dim // args.n_heads
        freqs = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(args.max_seq_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("freqs_cos", torch.cos(angles), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(angles), persistent=False)

    def _rotate(self, x, cos, sin):
        pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        real, imag = pairs.unbind(-1)
        cos = cos.view(1, 1, 1, -1)
        sin = sin.view(1, 1, 1, -1)
        rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
        return rotated.flatten(-2).type_as(x)

    def new_cache(self, device):
        shape = (
            self.args.n_layers,
            self.args.max_seq_len,
            self.args.n_kv_heads or self.args.n_heads,
            self.args.dim // self.args.n_heads,
        )
        return torch.empty(shape, device=device), torch.empty(shape, device=device)

    def step(self, token_id, position, key_cache, value_cache):
        x = self.tok_embeddings(torch.tensor([[token_id]], dtype=torch.long, device=key_cache.device))
        cos = self.freqs_cos[position]
        sin = self.freqs_sin[position]
        scale = self.args.dim // self.args.n_heads

        for layer_id, layer in enumerate(self.layers):
            att = layer.attention
            normalized = layer.attention_norm(x)
            q = att.wq(normalized).view(1, 1, att.n_heads, att.head_dim)
            k = att.wk(normalized).view(1, 1, att.n_kv_heads, att.head_dim)
            v = att.wv(normalized).view(1, 1, att.n_kv_heads, att.head_dim)
            q = self._rotate(q, cos, sin).transpose(1, 2)
            k = self._rotate(k, cos, sin)

            key_cache[layer_id, position].copy_(k[0, 0])
            value_cache[layer_id, position].copy_(v[0, 0])
            keys = key_cache[layer_id, : position + 1].permute(1, 0, 2)
            values = value_cache[layer_id, : position + 1].permute(1, 0, 2)
            keys = keys.repeat_interleave(att.n_rep, dim=0).unsqueeze(0)
            values = values.repeat_interleave(att.n_rep, dim=0).unsqueeze(0)
            scores = torch.matmul(q, keys.transpose(-2, -1)) / (scale**0.5)
            weights = torch.softmax(scores.float(), dim=-1).type_as(q)
            attended = torch.matmul(weights, values).transpose(1, 2).contiguous().view(1, 1, -1)

            x = x + att.wo(attended)
            normalized = layer.ffn_norm(x)
            x = x + layer.feed_forward.w2(F.silu(layer.feed_forward.w1(normalized)) * layer.feed_forward.w3(normalized))

        return self.output(self.norm(x))[0, 0]


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class XorShift64:
    mask = (1 << 64) - 1

    def __init__(self, seed):
        self.state = seed & self.mask

    def random_f32(self):
        self.state ^= self.state >> 12
        self.state ^= (self.state << 25) & self.mask
        self.state ^= self.state >> 27
        value = self.state * 0x2545F4914F6CDD1D & self.mask
        random_u32 = (value >> 32) & 0xFFFFFFFF
        return (random_u32 >> 8) / 16777216.0


def sample_token(logits, temperature, top_p, rng):
    if temperature == 0:
        return int(torch.argmax(logits).item())

    probabilities = F.softmax(logits.float() / temperature, dim=-1)
    coin = rng.random_f32()
    if top_p <= 0 or top_p >= 1:
        cumulative = 0.0
        for token_id, probability in enumerate(probabilities.tolist()):
            cumulative += probability
            if coin < cumulative:
                return token_id
        return probabilities.numel() - 1

    sorted_probabilities, sorted_ids = torch.sort(probabilities, descending=True)
    cutoff = (1.0 - top_p) / (probabilities.numel() - 1)
    keep = sorted_probabilities >= cutoff
    candidate_probabilities = sorted_probabilities[keep].tolist()
    candidate_ids = sorted_ids[keep].tolist()

    cumulative_probability = 0.0
    last_index = len(candidate_probabilities) - 1
    for index, probability in enumerate(candidate_probabilities):
        cumulative_probability += probability
        if cumulative_probability > top_p:
            last_index = index
            break

    target = coin * cumulative_probability
    cumulative = 0.0
    for index in range(last_index + 1):
        cumulative += candidate_probabilities[index]
        if target < cumulative:
            return candidate_ids[index]
    return candidate_ids[last_index]


def penalize_repeated_tokens(logits, generated, repetition_penalty):
    if repetition_penalty == 1.0 or not generated:
        return logits
    adjusted = logits.clone()
    token_ids = torch.tensor(sorted(set(generated)), dtype=torch.long, device=logits.device)
    repeated_logits = adjusted[token_ids]
    adjusted[token_ids] = torch.where(
        repeated_logits < 0,
        repeated_logits * repetition_penalty,
        repeated_logits / repetition_penalty,
    )
    return adjusted


def bias_stop_tokens(logits, stop_token, stop_token_bias):
    if stop_token is None or stop_token_bias == 0:
        return logits
    stop_tokens = stop_token if isinstance(stop_token, (set, tuple, list)) else (stop_token,)
    adjusted = logits.clone()
    for stop_id in stop_tokens:
        adjusted[stop_id] += stop_token_bias
    return adjusted


def block_repeated_ngrams(logits, generated, ngram_size):
    if ngram_size == 0 or not generated:
        return logits
    if ngram_size == 1:
        banned = set(generated)
    elif len(generated) < ngram_size - 1:
        return logits
    else:
        prefix = generated[-(ngram_size - 1) :]
        banned = {
            generated[index + ngram_size - 1]
            for index in range(len(generated) - ngram_size + 1)
            if generated[index : index + ngram_size - 1] == prefix
        }
    if not banned:
        return logits
    adjusted = logits.clone()
    banned_ids = torch.tensor(sorted(banned), dtype=torch.long, device=logits.device)
    adjusted[banned_ids] = -torch.inf
    if not torch.isfinite(adjusted).any():
        return logits
    return adjusted


def generate(
    model,
    prompt_ids,
    new_tokens,
    device,
    stop_token=None,
    temperature=0.0,
    top_p=0.9,
    seed=1,
    repetition_penalty=1.0,
    min_new_tokens=0,
    stop_token_bias=0.0,
    stop_token_bias_after=None,
    no_repeat_ngram_size=0,
):
    bias_after = min_new_tokens if stop_token_bias_after is None else stop_token_bias_after
    if (
        repetition_penalty <= 0
        or not math.isfinite(stop_token_bias)
        or min_new_tokens < 0
        or min_new_tokens > new_tokens
        or bias_after < 0
        or no_repeat_ngram_size < 0
    ):
        raise ValueError("repetition_penalty must be positive, stop_token_bias finite, min_new_tokens within the limit, and bias/ngram settings nonnegative")
    key_cache, value_cache = model.new_cache(device)
    rng = XorShift64(seed)
    synchronize(device)
    start = time.perf_counter()
    logits = None
    for position, token_id in enumerate(prompt_ids):
        logits = model.step(token_id, position, key_cache, value_cache)
    synchronize(device)
    prefill_seconds = time.perf_counter() - start

    generated = []
    stopped = False
    stop_reason = None
    start = time.perf_counter()
    for index in range(new_tokens):
        sample_logits = penalize_repeated_tokens(logits, generated, repetition_penalty)
        sample_logits = block_repeated_ngrams(sample_logits, generated, no_repeat_ngram_size)
        if stop_token is not None:
            stop_tokens = stop_token if isinstance(stop_token, (set, tuple, list)) else (stop_token,)
            if index < min_new_tokens:
                sample_logits = sample_logits.clone()
                for stop_id in stop_tokens:
                    sample_logits[stop_id] = -torch.inf
            elif index >= bias_after:
                sample_logits = bias_stop_tokens(sample_logits, stop_tokens, stop_token_bias)
        token_id = sample_token(sample_logits, temperature, top_p, rng)
        if stop_token is not None and (
            token_id in stop_token if isinstance(stop_token, (set, tuple, list)) else token_id == stop_token
        ):
            stopped = True
            stop_reason = token_id
            break
        generated.append(token_id)
        if index + 1 < new_tokens:
            position = len(prompt_ids) + index
            logits = model.step(token_id, position, key_cache, value_cache)
    synchronize(device)
    decode_seconds = time.perf_counter() - start
    return generated, prefill_seconds, decode_seconds, stopped, stop_reason


def verify_reference(model, state_dict, model_args, prompt_ids, device, reference_repo):
    sys.path.insert(0, str(reference_repo))
    from model import ModelArgs, Transformer

    reference = Transformer(ModelArgs(**model_args))
    reference.load_state_dict(state_dict, strict=True)
    reference.eval().to(device)
    tokens = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        expected = reference(tokens)[0, -1]
        # The reference model has no KV-cache path; compare its final prompt logits to cached prefill.
        key_cache, value_cache = model.new_cache(device)
        actual = None
        for position, token_id in enumerate(prompt_ids):
            actual = model.step(token_id, position, key_cache, value_cache)
        max_error = (expected - actual).abs().max().item()
        if not torch.allclose(expected, actual, rtol=2e-4, atol=2e-4):
            raise RuntimeError(f"cached logits do not match upstream reference on {device}: max error {max_error}")
    del reference
    return max_error


def benchmark(
    model,
    prompt_ids,
    new_tokens,
    runs,
    device,
    threads,
    stop_token,
    temperature,
    top_p,
    seed,
    repetition_penalty,
    min_new_tokens,
    stop_token_bias,
    stop_token_bias_after,
    no_repeat_ngram_size=0,
):
    if threads is not None:
        torch.set_num_threads(threads)
    with torch.inference_mode():
        warmup_tokens = min(16, new_tokens)
        generate(
            model,
            prompt_ids,
            warmup_tokens,
            device,
            stop_token,
            temperature,
            top_p,
            seed,
            repetition_penalty,
            min(min_new_tokens, warmup_tokens),
            stop_token_bias,
            stop_token_bias_after,
            no_repeat_ngram_size,
        )
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        samples = [
            generate(
                model,
                prompt_ids,
                new_tokens,
                device,
                stop_token,
                temperature,
                top_p,
                seed,
                repetition_penalty,
                min_new_tokens,
                stop_token_bias,
                stop_token_bias_after,
                no_repeat_ngram_size,
            )
            for _ in range(runs)
        ]
        synchronize(device)

    prefill = statistics.median(item[1] for item in samples)
    decode = statistics.median(item[2] for item in samples)
    generated_count = int(statistics.median(len(item[0]) for item in samples))
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
    return {
        "prefill_ms": prefill * 1000,
        "decode_ms": decode * 1000,
        "decode_tok_s": generated_count / decode,
        "end_to_end_tok_s": (len(prompt_ids) + generated_count) / (prefill + decode),
        "peak_cuda_mb": peak_mb,
        "generated_count": generated_count,
        "generated": samples[-1][0],
        "stopped": samples[-1][3],
        "stop_token_id": samples[-1][4],
    }


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="FP32 CPU/CUDA benchmark for TinyLLMStory-zh")
    parser.add_argument("--checkpoint", type=Path, default=root / "ckpt.pt")
    parser.add_argument("--tokenizer", type=Path, default=root / "tok4096.model")
    parser.add_argument("--reference-repo", type=Path, help="Optional chinese-llama2.c checkout for logit validation")
    parser.add_argument("--prompt", default="从前，有一只叫小白的兔子。")
    parser.add_argument(
        "--prompt-style",
        choices=("raw", "story"),
        default="raw",
        help="Encode the prompt as entered, or wrap it in the trained story-opening template",
    )
    parser.add_argument("--new-tokens", type=int, default=96)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cpu-threads", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    stop_group = parser.add_mutually_exclusive_group()
    stop_group.add_argument("--stop-on-bos", action="store_true", help="Stop on the legacy BOS sequence delimiter")
    stop_group.add_argument("--stop-on-eos", action="store_true", help="Stop on EOS, with BOS as a legacy fallback")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 selects greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1, help="Seed for the llama2.c-compatible xorshift sampler")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Penalty for tokens already generated; 1 disables it")
    parser.add_argument("--min-new-tokens", type=int, default=0, help="Suppress EOS/BOS stops before this continuation length")
    parser.add_argument("--stop-token-bias", type=float, default=0.0, help="Add to EOS/BOS logits after the bias threshold; 0 disables it")
    parser.add_argument("--stop-token-bias-after", type=int, help="Start applying a nonzero stop-token bias after this many tokens; defaults to min-new-tokens")
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0, help="Ban continuation tokens that would repeat this n-gram size; 0 disables it")
    args = parser.parse_args()
    if (
        args.new_tokens < 1
        or args.runs < 1
        or any(count < 1 for count in args.cpu_threads)
        or args.min_new_tokens < 0
        or args.min_new_tokens > args.new_tokens
        or args.repetition_penalty <= 0
        or not math.isfinite(args.stop_token_bias)
        or (args.stop_token_bias_after is not None and args.stop_token_bias_after < 0)
        or args.no_repeat_ngram_size < 0
    ):
        parser.error(
            "new-tokens, runs, and cpu-threads must be positive; min-new-tokens must fit the token cap; "
            "repetition-penalty must be positive; stop-token-bias must be finite; bias threshold and no-repeat-ngram-size must be nonnegative"
        )
    if args.temperature < 0 or not 0 <= args.top_p <= 1 or args.seed <= 0:
        parser.error("temperature must be nonnegative, top-p must be in [0, 1], and seed must be positive")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model_args = checkpoint["model_args"]
    defaults = {
        "hidden_dim": None,
        "multiple_of": 256,
        "norm_eps": 1e-5,
        "n_kv_heads": None,
        "max_seq_len": 2048,
        "dropout": 0.0,
    }
    runtime_args = SimpleNamespace(**{**defaults, **model_args})
    state_dict = checkpoint["model"]
    tokenizer = SentencePieceProcessor(model_file=str(args.tokenizer))
    prompt = args.prompt if args.prompt_style == "raw" else f"故事开头：{args.prompt.strip()}\n续写："
    prompt_ids = [tokenizer.bos_id(), *tokenizer.encode(prompt)]
    if len(prompt_ids) + args.new_tokens > model_args["max_seq_len"]:
        raise ValueError("prompt plus generated tokens exceeds model max_seq_len")

    parameter_count = sum(tensor.numel() for tensor in state_dict.values())
    if torch.equal(state_dict["tok_embeddings.weight"], state_dict["output.weight"]):
        parameter_count -= state_dict["output.weight"].numel()
    hidden_dim = runtime_args.hidden_dim
    if hidden_dim is None:
        hidden_dim = int(2 * (4 * model_args["dim"]) / 3)
        multiple = runtime_args.multiple_of
        hidden_dim = multiple * ((hidden_dim + multiple - 1) // multiple)
    sampler = "greedy" if args.temperature == 0 else f"temperature={args.temperature:g},top_p={args.top_p:g}"
    sampler += (
        f",repetition_penalty={args.repetition_penalty:g},min_new_tokens={args.min_new_tokens}"
        f",stop_token_bias={args.stop_token_bias:g}"
        f",stop_token_bias_after={args.stop_token_bias_after if args.stop_token_bias_after is not None else args.min_new_tokens}"
        f",no_repeat_ngram_size={args.no_repeat_ngram_size}"
    )
    print(f"torch={torch.__version__}; cuda_available={torch.cuda.is_available()}; dtype=fp32; sampler={sampler}+KV-cache")
    print(f"cpu={platform.processor() or platform.machine()}; cuda_version={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(
        f"model=dim{model_args['dim']}-layers{model_args['n_layers']}-heads{model_args['n_heads']}-"
        f"kv_heads{model_args.get('n_kv_heads', model_args['n_heads'])}-hidden{hidden_dim} "
        f"parameters={parameter_count:,}; prompt_style={args.prompt_style}; prompt_tokens={len(prompt_ids)}; "
        f"token_limit={args.new_tokens}; runs={args.runs}"
    )
    if args.reference_repo:
        for name in args.devices:
            if name == "cuda" and not torch.cuda.is_available():
                continue
            device = torch.device(name)
            model = CachedTransformer(runtime_args).eval().to(device)
            model.load_state_dict(state_dict, strict=True)
            error = verify_reference(model, state_dict, model_args, prompt_ids, device, args.reference_repo)
            print(f"reference_check device={name} max_abs_logit_error={error:.8g}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    results = []
    for name in args.devices:
        if name == "cuda" and not torch.cuda.is_available():
            print("cuda=unavailable; skipped")
            continue
        device = torch.device(name)
        model = CachedTransformer(runtime_args).eval().to(device)
        model.load_state_dict(state_dict, strict=True)
        thread_counts = args.cpu_threads if device.type == "cpu" else [1]
        for threads in thread_counts:
            if args.stop_on_eos:
                stop_token = {tokenizer.eos_id(), tokenizer.bos_id()}
                stop_name = "EOS/BOS"
            elif args.stop_on_bos:
                stop_token = tokenizer.bos_id()
                stop_name = "BOS"
            else:
                stop_token = None
                stop_name = "limit"
            stats = benchmark(
                model,
                prompt_ids,
                args.new_tokens,
                args.runs,
                device,
                threads,
                stop_token,
                args.temperature,
                args.top_p,
                args.seed,
                args.repetition_penalty,
                args.min_new_tokens,
                args.stop_token_bias,
                args.stop_token_bias_after,
                args.no_repeat_ngram_size,
            )
            row = {"device": name, "threads": threads if device.type == "cpu" else "n/a", **stats}
            results.append(row)
            memory = f" peak_cuda={stats['peak_cuda_mb']:.1f} MiB" if stats["peak_cuda_mb"] is not None else ""
            print(
                f"device={name} threads={row['threads']} prefill={stats['prefill_ms']:.2f} ms "
                f"decode={stats['decode_tok_s']:.2f} tok/s end_to_end={stats['end_to_end_tok_s']:.2f} tok/s "
                f"generated={stats['generated_count']}/{args.new_tokens} tokens "
                f"stop={stop_name if stats['stopped'] else 'limit'}{memory}"
            )
            print(f"story_continuation={tokenizer.decode(stats['generated'])}")
        del model
    cpu_rows = [row for row in results if row["device"] == "cpu"]
    cuda_rows = [row for row in results if row["device"] == "cuda"]
    if cpu_rows and cuda_rows:
        best_cpu = max(cpu_rows, key=lambda row: row["decode_tok_s"])
        print(f"cuda_speedup_vs_best_cpu={cuda_rows[0]['decode_tok_s'] / best_cpu['decode_tok_s']:.2f}x")


if __name__ == "__main__":
    main()
