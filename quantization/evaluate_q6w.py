import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sentencepiece import SentencePieceProcessor

from q6w_format import load_dequantized_state, sha256_file


def load_jsonl_prompts(path, limit):
    result = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            item = json.loads(line)
            prompt = item["prompt"]
            opening = prompt.splitlines()[0]
            if opening in seen:
                continue
            seen.add(opening)
            result.append(item)
            if len(result) >= limit:
                break
    if not result:
        raise ValueError(f"No prompts found in {path}")
    return result


def load_model_args(model_args):
    defaults = {
        "hidden_dim": None,
        "multiple_of": 256,
        "norm_eps": 1e-5,
        "n_kv_heads": None,
        "max_seq_len": 2048,
        "dropout": 0.0,
    }
    from types import SimpleNamespace

    return SimpleNamespace(**{**defaults, **model_args})


def compare_logits(benchmark, fp32_model, q6_model, prompt_ids, device):
    fp32_cache = fp32_model.new_cache(device)
    q6_cache = q6_model.new_cache(device)
    fp32_logits = q6_logits = None
    with torch.inference_mode():
        for position, token_id in enumerate(prompt_ids):
            fp32_logits = fp32_model.step(token_id, position, *fp32_cache)
            q6_logits = q6_model.step(token_id, position, *q6_cache)
    a = fp32_logits.float()
    b = q6_logits.float()
    return {
        "max_abs": float((a - b).abs().max().item()),
        "rmse": float(torch.sqrt(torch.mean((a - b).square())).item()),
        "cosine": float(F.cosine_similarity(a[None], b[None]).item()),
        "fp32_top1": int(a.argmax().item()),
        "q6_top1": int(b.argmax().item()),
        "top1_match": bool(a.argmax().item() == b.argmax().item()),
    }


def evaluate_masked_loss(args, checkpoint, q6_state):
    if args.loss_batches <= 0:
        return None
    training_root = args.training_root.resolve()
    if not (training_root / "model.py").is_file() or not (training_root / "tinystories.py").is_file():
        raise FileNotFoundError(f"Training source files not found in {training_root}")
    os.environ["TINYLLAMA_DATA_DIR"] = str(args.data_dir.resolve())
    sys.path.insert(0, str(training_root))
    from model import ModelArgs, Transformer
    from tinystories import Task

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.cpu_threads)
    fp32_model = Transformer(ModelArgs(**checkpoint["model_args"]))
    q6_model = Transformer(ModelArgs(**checkpoint["model_args"]))
    fp32_model.load_state_dict(checkpoint["model"], strict=True)
    q6_model.load_state_dict(q6_state, strict=True)
    fp32_model.to(device).eval()
    q6_model.to(device).eval()
    batches = Task.iter_batches(
        split="val",
        batch_size=args.batch_size,
        max_seq_len=checkpoint["model_args"]["max_seq_len"],
        vocab_size=checkpoint["model_args"]["vocab_size"],
        vocab_source="custom",
        mask_loss=True,
        device=str(device),
        num_workers=0,
    )
    fp32_losses = []
    q6_losses = []
    with torch.inference_mode():
        for _ in range(args.loss_batches):
            x, y = next(batches)
            fp32_model(x, y)
            q6_model(x, y)
            fp32_losses.append(fp32_model.last_loss.float().item())
            q6_losses.append(q6_model.last_loss.float().item())
    return {
        "device": str(device),
        "batches": args.loss_batches,
        "batch_size": args.batch_size,
        "fp32_masked_loss": sum(fp32_losses) / len(fp32_losses),
        "q6w_masked_loss": sum(q6_losses) / len(q6_losses),
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    training_default = repo_root / "training"
    parser = argparse.ArgumentParser(description="Compare Q6W dequantized reference against its FP32 source")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=repo_root / "tok4096.model")
    parser.add_argument("--prompts", type=Path, default=training_default / "data/story_pairs_masked_v2/eval_prompts.jsonl")
    parser.add_argument("--training-root", type=Path, default=training_default)
    parser.add_argument("--data-dir", type=Path, default=training_default / "data/story_pairs_masked_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-limit", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--loss-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.65)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--min-new-tokens", type=int, default=32)
    parser.add_argument("--stop-token-bias", type=float, default=2.0)
    parser.add_argument("--stop-token-bias-after", type=int, default=96)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.prompt_limit < 1 or args.max_new_tokens < 1 or args.cpu_threads < 1:
        parser.error("prompt-limit, max-new-tokens, and cpu-threads must be positive")
    if args.min_new_tokens < 0 or args.min_new_tokens > args.max_new_tokens:
        parser.error("min-new-tokens must be within the token cap")
    if args.loss_batches < 0 or args.batch_size < 1 or args.no_repeat_ngram_size < 0:
        parser.error("loss-batches and no-repeat-ngram-size must be nonnegative; batch-size must be positive")
    if args.repetition_penalty <= 0 or not math.isfinite(args.stop_token_bias):
        parser.error("repetition-penalty must be positive and stop-token-bias finite")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    q6_state, manifest = load_dequantized_state(args.binary, args.manifest, checkpoint["model"])
    if sha256_file(args.checkpoint) != manifest["source_checkpoint_sha256"]:
        raise ValueError("The Q6W artifact was exported from a different checkpoint")
    sys.path.insert(0, str(repo_root / "benchmarks"))
    import torch_benchmark as benchmark

    model_args = load_model_args(checkpoint["model_args"])
    fp32_model = benchmark.CachedTransformer(model_args).eval()
    q6_model = benchmark.CachedTransformer(model_args).eval()
    fp32_model.load_state_dict(checkpoint["model"], strict=True)
    q6_model.load_state_dict(q6_state, strict=True)
    device = torch.device("cpu")
    fp32_model.to(device)
    q6_model.to(device)
    torch.set_num_threads(args.cpu_threads)
    tokenizer = SentencePieceProcessor(model_file=str(args.tokenizer))
    stop_tokens = {tokenizer.eos_id(), tokenizer.bos_id()}
    prompts = load_jsonl_prompts(args.prompts, args.prompt_limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with args.output.open("w", encoding="utf-8", newline="\n") as target:
        for index, item in enumerate(prompts, start=1):
            prompt = item["prompt"]
            prompt_ids = [tokenizer.bos_id(), *tokenizer.encode(prompt)]
            token_limit = min(args.max_new_tokens, checkpoint["model_args"]["max_seq_len"] - len(prompt_ids))
            if token_limit < 1:
                continue
            logits = compare_logits(benchmark, fp32_model, q6_model, prompt_ids, device)
            fp32_generated, _, _, _, fp32_stop = benchmark.generate(
                fp32_model,
                prompt_ids,
                token_limit,
                device,
                stop_tokens,
                args.temperature,
                args.top_p,
                args.seed,
                args.repetition_penalty,
                min(args.min_new_tokens, token_limit),
                args.stop_token_bias,
                args.stop_token_bias_after,
                args.no_repeat_ngram_size,
            )
            q6_generated, _, _, _, q6_stop = benchmark.generate(
                q6_model,
                prompt_ids,
                token_limit,
                device,
                stop_tokens,
                args.temperature,
                args.top_p,
                args.seed,
                args.repetition_penalty,
                min(args.min_new_tokens, token_limit),
                args.stop_token_bias,
                args.stop_token_bias_after,
                args.no_repeat_ngram_size,
            )
            common = 0
            for left, right in zip(fp32_generated, q6_generated):
                if left != right:
                    break
                common += 1
            row = {
                "index": index,
                "prompt": prompt,
                "reference": item.get("reference", ""),
                "fp32_generated": tokenizer.decode(fp32_generated),
                "q6w_generated": tokenizer.decode(q6_generated),
                "fp32_tokens": len(fp32_generated),
                "q6w_tokens": len(q6_generated),
                "fp32_stop_id": fp32_stop,
                "q6w_stop_id": q6_stop,
                "common_prefix_tokens": common,
                "token_sequences_equal": fp32_generated == q6_generated,
                "logits": logits,
            }
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            if index % 16 == 0 or index == len(prompts):
                print(f"evaluated {index}/{len(prompts)} prompts", flush=True)

    loss = evaluate_masked_loss(args, checkpoint, q6_state)
    stop_counts = {}
    for label in ("fp32", "q6w"):
        stop_counts[label] = {
            "eos": sum(row[f"{label}_stop_id"] == tokenizer.eos_id() for row in rows),
            "bos": sum(row[f"{label}_stop_id"] == tokenizer.bos_id() for row in rows),
            "limit": sum(row[f"{label}_stop_id"] is None for row in rows),
        }
    summary = {
        "source_checkpoint": args.checkpoint.name,
        "source_step": checkpoint["iter_num"],
        "q6_binary": args.binary.name,
        "q6_binary_sha256": manifest["binary_sha256"],
        "prompts": len(rows),
        "sampler": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "min_new_tokens": args.min_new_tokens,
            "stop_token_bias": args.stop_token_bias,
            "stop_token_bias_after": args.stop_token_bias_after,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "logits": {
            "max_abs_mean": sum(row["logits"]["max_abs"] for row in rows) / len(rows),
            "rmse_mean": sum(row["logits"]["rmse"] for row in rows) / len(rows),
            "cosine_mean": sum(row["logits"]["cosine"] for row in rows) / len(rows),
            "top1_agreements": sum(row["logits"]["top1_match"] for row in rows),
        },
        "generation": {
            "mean_tokens": {
                label: sum(row[f"{label}_tokens"] for row in rows) / len(rows)
                for label in ("fp32", "q6w")
            },
            "stop_counts": stop_counts,
            "identical_sequences": sum(row["token_sequences_equal"] for row in rows),
            "mean_common_prefix_tokens": sum(row["common_prefix_tokens"] for row in rows) / len(rows),
        },
        "masked_validation_loss": loss,
        "paired_output": args.output.name,
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8", newline="\n") as target:
        json.dump(summary, target, ensure_ascii=False, indent=2)
        target.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
