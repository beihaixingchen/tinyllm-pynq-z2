import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Prepare a Q6 TinyLLM PYNQ end-to-end request and PyTorch golden")
    default_model_repo = Path(os.environ.get("TINYLLM_MODEL_REPO", project_dir))
    parser.add_argument("--model-repo", type=Path, default=default_model_repo,
                        help="model package root (defaults to this project)")
    parser.add_argument("--weights", type=Path,
                        help="Q6W weights (defaults to this repository's QAT artifact)")
    parser.add_argument("--manifest", type=Path,
                        help="Q6W manifest (defaults to this repository's QAT manifest)")
    parser.add_argument("--checkpoint", type=Path,
                        help="source PyTorch checkpoint; defaults to the path recorded in the manifest")
    parser.add_argument("--tokenizer", type=Path,
                        help="SentencePiece model (defaults to model-repo/tok4096.model)")
    parser.add_argument("--out-dir", type=Path, default=project_dir / "build" / "e2e")
    parser.add_argument(
        "--linear-mode", choices=(
            "ps", "ffn", "all", "batched", "selected", "cached", "cached_qkv",
            "cached_vocab", "vocab_bram",
        ), default="ffn"
    )
    parser.add_argument("--bitstream", type=Path, default=project_dir / "build" / "output" / "q6w_gemv.bit")
    parser.add_argument("--hwh", type=Path, default=project_dir / "build" / "output" / "q6w_gemv.hwh")
    parser.add_argument("--prompt", default="从前，有一只叫小白的兔子。")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--min-new-tokens", type=int, default=4)
    args = parser.parse_args()

    model_repo = args.model_repo.resolve()
    quant_dir = model_repo / "quantization"
    args.weights = (args.weights or quant_dir / "artifacts" / "tinyllamas_zh_story_q6w_qat_lr1e6.bin").resolve()
    args.manifest = (args.manifest or quant_dir / "artifacts" / "tinyllamas_zh_story_q6w_qat_lr1e6.json").resolve()
    args.checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    args.tokenizer = (args.tokenizer or model_repo / "tok4096.model").resolve()
    import numpy as np
    import torch
    from sentencepiece import SentencePieceProcessor

    sys.path.insert(0, str(quant_dir))
    sys.path.insert(0, str(model_repo / "benchmarks"))
    from q6w_format import load_dequantized_state, sha256_file
    from torch_benchmark import CachedTransformer

    if args.max_new_tokens < 1 or not 0 <= args.min_new_tokens <= args.max_new_tokens:
        raise ValueError("Token limits must satisfy max_new_tokens >= 1 and 0 <= min_new_tokens <= max_new_tokens")
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        manifest_checkpoint = Path(manifest["source_checkpoint"])
        candidates = [
            manifest_checkpoint if manifest_checkpoint.is_absolute() else model_repo / manifest_checkpoint,
            model_repo / "training" / "checkpoints" / manifest_checkpoint.name,
            model_repo / "ckpt.pt",
        ]
        checkpoint_path = next(
            (
                candidate
                for candidate in candidates
                if candidate.is_file()
                and sha256_file(candidate) == manifest["source_checkpoint_sha256"]
            ),
            None,
        )
        if checkpoint_path is None:
            checkpoint_dir = model_repo / "training" / "checkpoints"
            if checkpoint_dir.is_dir():
                checkpoint_path = next(
                    (
                        candidate
                        for candidate in checkpoint_dir.glob("*.pt")
                        if sha256_file(candidate) == manifest["source_checkpoint_sha256"]
                    ),
                    None,
                )
    if checkpoint_path is None:
        raise FileNotFoundError(
            "Source checkpoint not found or SHA-256 mismatched; pass --checkpoint"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Source checkpoint not found: {}".format(checkpoint_path))
    if sha256_file(checkpoint_path) != manifest["source_checkpoint_sha256"]:
        raise ValueError("Q6 source checkpoint SHA-256 does not match its manifest")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["iter_num"]) != int(manifest["source_step"]):
        raise ValueError("Q6 source checkpoint step does not match the manifest")

    q6_state, loaded_manifest = load_dequantized_state(args.weights, args.manifest, checkpoint["model"])
    model_args = dict(checkpoint["model_args"])
    defaults = {
        "hidden_dim": None,
        "multiple_of": 256,
        "norm_eps": 1e-5,
        "n_kv_heads": None,
        "max_seq_len": 2048,
        "dropout": 0.0,
    }
    runtime_args = type("RuntimeArgs", (), {**defaults, **model_args})()
    runtime_args.max_seq_len = int(loaded_manifest["model_args"]["max_seq_len"])
    model = CachedTransformer(runtime_args).eval()
    model.load_state_dict(q6_state, strict=True)

    tokenizer = SentencePieceProcessor(model_file=str(args.tokenizer))
    prompt = "故事开头：{}\n续写：".format(args.prompt.strip())
    prompt_ids = [int(tokenizer.bos_id())] + [int(value) for value in tokenizer.encode(prompt)]
    if not prompt_ids or len(prompt_ids) + args.max_new_tokens > int(runtime_args.max_seq_len):
        raise ValueError("Prompt plus requested generation exceeds the model context")

    stop_ids = sorted({int(tokenizer.bos_id()), int(tokenizer.eos_id())})
    key_cache, value_cache = model.new_cache(torch.device("cpu"))
    with torch.inference_mode():
        for position, token_id in enumerate(prompt_ids):
            logits = model.step(token_id, position, key_cache, value_cache)

        reference_logits = []
        sampled_ids = []
        generated_ids = []
        stopped_on = None
        for index in range(args.max_new_tokens):
            row = logits.detach().cpu().numpy().astype("<f4", copy=False)
            reference_logits.append(row)
            scores = logits.clone()
            if index < args.min_new_tokens:
                for token_id in stop_ids:
                    scores[token_id] = -float("inf")
            token_id = int(torch.argmax(scores).item())
            sampled_ids.append(token_id)
            if index >= args.min_new_tokens and token_id in stop_ids:
                stopped_on = token_id
                break
            generated_ids.append(token_id)
            if index + 1 < args.max_new_tokens:
                logits = model.step(token_id, len(prompt_ids) + index, key_cache, value_cache)

    reference_matrix = np.stack(reference_logits).astype("<f4", copy=False)
    (args.out_dir / "reference_logits.bin").write_bytes(reference_matrix.tobytes())

    request = {
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "prompt_token_count": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "linear_mode": args.linear_mode,
        "stop_token_ids": stop_ids,
        "reference_sampled_ids": sampled_ids,
        "reference_generated_ids": generated_ids,
        "reference_step_count": len(sampled_ids),
        "stopped_on": stopped_on,
        "model": {
            "dim": int(runtime_args.dim),
            "layers": int(runtime_args.n_layers),
            "heads": int(runtime_args.n_heads),
            "kv_heads": int(runtime_args.n_kv_heads),
            "vocab_size": int(runtime_args.vocab_size),
            "max_seq_len": int(runtime_args.max_seq_len),
            "norm_eps": float(runtime_args.norm_eps),
            "source_step": int(loaded_manifest["source_step"]),
            "weights_sha256": loaded_manifest["binary_sha256"],
        },
    }
    (args.out_dir / "request.json").write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    shutil.copy2(args.weights, args.out_dir / "model_q6w.bin")
    shutil.copy2(args.manifest, args.out_dir / "manifest.json")
    if args.linear_mode != "ps":
        shutil.copy2(args.bitstream, args.out_dir / "q6w_gemv.bit")
        shutil.copy2(args.hwh, args.out_dir / "q6w_gemv.hwh")

    print(json.dumps({
        "package": str(args.out_dir),
        "prompt": prompt,
        "prompt_ids": prompt_ids,
        "reference_sampled_ids": sampled_ids,
        "reference_generated_ids": generated_ids,
        "decoded_reference": tokenizer.decode(generated_ids),
        "reference_steps": len(sampled_ids),
        "reference_logits_bytes": reference_matrix.nbytes,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
