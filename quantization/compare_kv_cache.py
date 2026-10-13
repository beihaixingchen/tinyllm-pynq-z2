import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor

from q6w_format import load_dequantized_state, sha256_file


class FP16KVTransformer:
    """Q6W model wrapper storing K/V in FP16 and computing attention in FP32."""

    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.hidden_dim = model.hidden_dim
        self.n_layers = model.args.n_layers

    def __getattr__(self, name):
        return getattr(self.model, name)

    def new_cache(self, device):
        shape = (
            self.args.n_layers,
            self.args.max_seq_len,
            self.args.n_kv_heads or self.args.n_heads,
            self.args.dim // self.args.n_heads,
        )
        return (
            torch.empty(shape, device=device, dtype=torch.float16),
            torch.empty(shape, device=device, dtype=torch.float16),
        )

    def step(self, token_id, position, key_cache, value_cache):
        model = self.model
        args = self.args
        x = model.tok_embeddings(torch.tensor([[token_id]], dtype=torch.long, device=key_cache.device))
        cos = model.freqs_cos[position]
        sin = model.freqs_sin[position]
        scale = args.dim // args.n_heads

        for layer_id, layer in enumerate(model.layers):
            att = layer.attention
            normalized = layer.attention_norm(x)
            q = att.wq(normalized).view(1, 1, att.n_heads, att.head_dim)
            k = att.wk(normalized).view(1, 1, att.n_kv_heads, att.head_dim)
            v = att.wv(normalized).view(1, 1, att.n_kv_heads, att.head_dim)
            q = model._rotate(q, cos, sin).transpose(1, 2)
            k = model._rotate(k, cos, sin)

            key_cache[layer_id, position].copy_(k[0, 0])
            value_cache[layer_id, position].copy_(v[0, 0])
            keys = key_cache[layer_id, : position + 1].float().permute(1, 0, 2)
            values = value_cache[layer_id, : position + 1].float().permute(1, 0, 2)
            keys = keys.repeat_interleave(att.n_rep, dim=0).unsqueeze(0)
            values = values.repeat_interleave(att.n_rep, dim=0).unsqueeze(0)
            scores = torch.matmul(q, keys.transpose(-2, -1)) / (scale**0.5)
            weights = torch.softmax(scores.float(), dim=-1).type_as(q)
            attended = torch.matmul(weights, values).transpose(1, 2).contiguous().view(1, 1, -1)

            x = x + att.wo(attended)
            normalized = layer.ffn_norm(x)
            x = x + layer.feed_forward.w2(
                F.silu(layer.feed_forward.w1(normalized)) * layer.feed_forward.w3(normalized)
            )

        return model.output(model.norm(x))[0, 0]


def load_prompts(path, limit):
    prompts = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            opening = item["prompt"].splitlines()[0]
            if opening in seen:
                continue
            seen.add(opening)
            prompts.append(item)
            if len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def compare_long_contexts(fp32_model, fp16_model, prompts, tokenizer, lengths, device):
    token_stream = []
    for item in prompts:
        token_stream.append(tokenizer.bos_id())
        token_stream.extend(tokenizer.encode(item["prompt"] + item.get("reference", "")))
        token_stream.append(tokenizer.eos_id())
        if len(token_stream) >= max(lengths):
            break
    if len(token_stream) < max(lengths):
        raise ValueError("Held-out prompt/reference text is too short for the requested context probes")

    probes = {}
    for length in lengths:
        caches = [fp32_model.new_cache(device), fp16_model.new_cache(device)]
        max_abs = 0.0
        squared_error = 0.0
        value_count = 0
        cosine_sum = 0.0
        top1_matches = 0
        with torch.inference_mode():
            for position, token_id in enumerate(token_stream[:length]):
                left = fp32_model.step(token_id, position, *caches[0]).float()
                right = fp16_model.step(token_id, position, *caches[1]).float()
                difference = left - right
                max_abs = max(max_abs, float(difference.abs().max().item()))
                squared_error += float(difference.square().sum().item())
                value_count += difference.numel()
                cosine_sum += float(F.cosine_similarity(left[None], right[None]).item())
                top1_matches += int(left.argmax().item() == right.argmax().item())
        probes[str(length)] = {
            "positions": length,
            "max_abs_over_all_positions": max_abs,
            "rmse_over_all_logits": math.sqrt(squared_error / value_count),
            "mean_position_cosine": cosine_sum / length,
            "top1_matches": top1_matches,
        }
    return probes


def main():
    repo_root = Path(__file__).resolve().parents[1]
    training_root = repo_root / "training"
    parser = argparse.ArgumentParser(description="Compare FP32 and FP16 KV storage on fixed Q6W weights")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=training_root / "checkpoints" / "q6w_qat_lr1e6_step25000.pt",
    )
    parser.add_argument("--binary", type=Path, default=repo_root / "quantization/artifacts/tinyllamas_zh_story_q6w_qat_lr1e6.bin")
    parser.add_argument("--manifest", type=Path, default=repo_root / "quantization/artifacts/tinyllamas_zh_story_q6w_qat_lr1e6.json")
    parser.add_argument("--tokenizer", type=Path, default=repo_root / "tok4096.model")
    parser.add_argument("--prompts", type=Path, default=training_root / "data/story_pairs_masked_v2/eval_prompts.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-limit", type=int, default=128)
    parser.add_argument("--probe-context-lengths", type=int, nargs="+", default=[128, 256, 511])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--min-new-tokens", type=int, default=32)
    parser.add_argument("--stop-token-bias", type=float, default=2.0)
    parser.add_argument("--stop-token-bias-after", type=int, default=96)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()
    if (
        args.prompt_limit < 1
        or any(length < 1 for length in args.probe_context_lengths)
        or args.max_new_tokens < 1
        or args.min_new_tokens < 0
        or args.min_new_tokens > args.max_new_tokens
        or args.no_repeat_ngram_size < 0
        or args.repetition_penalty <= 0
        or args.cpu_threads < 1
        or not math.isfinite(args.stop_token_bias)
    ):
        parser.error("prompt/token limits, repetition penalty, bias and thread count are invalid")
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError("Refusing to overwrite an existing KV comparison output")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    q6_state, manifest = load_dequantized_state(args.binary, args.manifest, checkpoint["model"])
    if sha256_file(args.checkpoint) != manifest["source_checkpoint_sha256"]:
        raise ValueError("The Q6W artifact was exported from a different checkpoint")

    sys.path.insert(0, str(repo_root / "benchmarks"))
    import torch_benchmark as benchmark

    defaults = {"hidden_dim": None, "multiple_of": 256, "norm_eps": 1e-5, "n_kv_heads": None, "max_seq_len": 2048, "dropout": 0.0}
    model_args = SimpleNamespace(**{**defaults, **checkpoint["model_args"]})
    if any(length > model_args.max_seq_len for length in args.probe_context_lengths):
        parser.error("probe-context-lengths cannot exceed the model context length")
    fp32_model = benchmark.CachedTransformer(model_args).eval()
    fp32_model.load_state_dict(q6_state, strict=True)
    fp16_model = FP16KVTransformer(benchmark.CachedTransformer(model_args).eval())
    fp16_model.model.load_state_dict(q6_state, strict=True)

    device = torch.device("cpu")
    fp32_model.to(device)
    fp16_model.to(device)
    torch.set_num_threads(args.cpu_threads)
    tokenizer = SentencePieceProcessor(model_file=str(args.tokenizer))
    stop_tokens = {tokenizer.eos_id(), tokenizer.bos_id()}
    prompts = load_prompts(args.prompts, args.prompt_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    with torch.inference_mode(), args.output.open("w", encoding="utf-8", newline="\n") as target:
        for index, item in enumerate(prompts, start=1):
            prompt = item["prompt"]
            prompt_ids = [tokenizer.bos_id(), *tokenizer.encode(prompt)]
            token_limit = min(args.max_new_tokens, model_args.max_seq_len - len(prompt_ids))
            if token_limit < 1:
                continue
            caches = [fp32_model.new_cache(device), fp16_model.new_cache(device)]
            logits = [None, None]
            for position, token_id in enumerate(prompt_ids):
                logits[0] = fp32_model.step(token_id, position, *caches[0])
                logits[1] = fp16_model.step(token_id, position, *caches[1])
            left, right = logits[0].float(), logits[1].float()
            fp32_generated, _, _, _, fp32_stop = benchmark.generate(
                fp32_model, prompt_ids, token_limit, device, stop_tokens, args.temperature, args.top_p,
                args.seed, args.repetition_penalty, min(args.min_new_tokens, token_limit),
                args.stop_token_bias, args.stop_token_bias_after, args.no_repeat_ngram_size,
            )
            fp16_generated, _, _, _, fp16_stop = benchmark.generate(
                fp16_model, prompt_ids, token_limit, device, stop_tokens, args.temperature, args.top_p,
                args.seed, args.repetition_penalty, min(args.min_new_tokens, token_limit),
                args.stop_token_bias, args.stop_token_bias_after, args.no_repeat_ngram_size,
            )
            common = 0
            for a, b in zip(fp32_generated, fp16_generated):
                if a != b:
                    break
                common += 1
            row = {
                "index": index,
                "prompt": prompt,
                "fp32_kv_generated": tokenizer.decode(fp32_generated),
                "fp16_kv_generated": tokenizer.decode(fp16_generated),
                "fp32_kv_tokens": len(fp32_generated),
                "fp16_kv_tokens": len(fp16_generated),
                "fp32_kv_stop_id": fp32_stop,
                "fp16_kv_stop_id": fp16_stop,
                "common_prefix_tokens": common,
                "token_sequences_equal": fp32_generated == fp16_generated,
                "logits": {
                    "max_abs": float((left - right).abs().max().item()),
                    "rmse": float(torch.sqrt(torch.mean((left - right).square())).item()),
                    "cosine": float(F.cosine_similarity(left[None], right[None]).item()),
                    "top1_match": bool(left.argmax().item() == right.argmax().item()),
                },
            }
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if index % 16 == 0 or index == len(prompts):
                print(f"evaluated {index}/{len(prompts)} prompts", flush=True)

    summary = {
        "checkpoint": args.checkpoint.name,
        "source_step": checkpoint["iter_num"],
        "q6w_binary_sha256": manifest["binary_sha256"],
        "kv_storage": {"baseline": "FP32", "comparison": "FP16", "attention_compute": "FP32"},
        "prompts": len(rows),
        "logits": {
            "max_abs_mean": sum(row["logits"]["max_abs"] for row in rows) / len(rows),
            "rmse_mean": sum(row["logits"]["rmse"] for row in rows) / len(rows),
            "cosine_mean": sum(row["logits"]["cosine"] for row in rows) / len(rows),
            "top1_agreements": sum(row["logits"]["top1_match"] for row in rows),
        },
        "generation": {
            "identical_sequences": sum(row["token_sequences_equal"] for row in rows),
            "mean_common_prefix_tokens": sum(row["common_prefix_tokens"] for row in rows) / len(rows),
            "mean_tokens": {
                dtype: sum(row[f"{dtype}_kv_tokens"] for row in rows) / len(rows)
                for dtype in ("fp32", "fp16")
            },
            "stop_counts": {
                dtype: {
                    "eos": sum(row[f"{dtype}_kv_stop_id"] == tokenizer.eos_id() for row in rows),
                    "bos": sum(row[f"{dtype}_kv_stop_id"] == tokenizer.bos_id() for row in rows),
                    "limit": sum(row[f"{dtype}_kv_stop_id"] is None for row in rows),
                }
                for dtype in ("fp32", "fp16")
            },
        },
        "long_context_probes": compare_long_contexts(
            fp32_model,
            fp16_model,
            prompts,
            tokenizer,
            sorted(set(args.probe_context_lengths)),
            device,
        ),
        "output": args.output.name,
    }
    with args.output.with_suffix(".summary.json").open("x", encoding="utf-8", newline="\n") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
