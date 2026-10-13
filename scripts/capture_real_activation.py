import argparse
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sentencepiece import SentencePieceProcessor


TENSOR_NAME = "layers.0.feed_forward.w1.weight"


def write_hex(path, values, width):
    with path.open("w", encoding="ascii", newline="\n") as target:
        for value in values:
            target.write("{:0{}X}\n".format(int(value) & ((1 << (width * 4)) - 1), width))


def error_stats(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    error = actual - reference
    reference_rms = float(np.sqrt(np.mean(reference * reference)))
    actual_norm = float(np.linalg.norm(actual))
    reference_norm = float(np.linalg.norm(reference))
    cosine = float(np.dot(actual, reference) / (actual_norm * reference_norm)) if actual_norm and reference_norm else 1.0
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs": float(np.max(np.abs(error))),
        "reference_rms": reference_rms,
        "relative_rmse": float(np.sqrt(np.mean(error * error)) / reference_rms) if reference_rms else 0.0,
        "cosine": cosine,
    }


def main():
    project_dir = Path(__file__).resolve().parents[1]
    workspace = project_dir.parent.parent
    tiny_repo = workspace / "TinyLLMStory-zh"
    q6_dir = tiny_repo / "quantization"
    sys.path.insert(0, str(q6_dir))
    sys.path.insert(0, str(tiny_repo / "benchmarks"))
    from q6w_format import sha256_file, unpack_signed_int6
    from torch_benchmark import CachedTransformer

    parser = argparse.ArgumentParser(description="Capture one real TinyLLM layer-0 FFN activation for the Q6W GEMV RTL")
    parser.add_argument("--weights", type=Path, default=q6_dir / "artifacts" / "tinyllamas_zh_story_q6w_qat_lr1e6.bin")
    parser.add_argument("--manifest", type=Path, default=q6_dir / "artifacts" / "tinyllamas_zh_story_q6w_qat_lr1e6.json")
    parser.add_argument("--tokenizer", type=Path, default=tiny_repo / "tok4096.model")
    parser.add_argument("--out-dir", type=Path, default=project_dir / "build" / "real_layer0")
    parser.add_argument("--prompt", default="从前，有一只叫小白的兔子。")
    parser.add_argument("--prompt-style", choices=("raw", "story"), default="story")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint_path = Path(manifest["source_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError("Q6 source checkpoint is missing: {}".format(checkpoint_path))
    checkpoint_sha256 = sha256_file(checkpoint_path)
    if checkpoint_sha256 != manifest["source_checkpoint_sha256"]:
        raise ValueError("Q6 source checkpoint SHA-256 does not match its manifest")
    weights_sha256 = sha256_file(args.weights)
    if weights_sha256 != manifest["binary_sha256"]:
        raise ValueError("Q6 weight file SHA-256 does not match its manifest")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if int(checkpoint["iter_num"]) != int(manifest["source_step"]):
        raise ValueError("Q6 source checkpoint step does not match its manifest")
    model_args = checkpoint["model_args"]
    defaults = {"hidden_dim": None, "multiple_of": 256, "norm_eps": 1e-5, "n_kv_heads": None, "max_seq_len": 2048, "dropout": 0.0}
    runtime_args = SimpleNamespace(**{**defaults, **model_args})
    model = CachedTransformer(runtime_args).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    tokenizer = SentencePieceProcessor(model_file=str(args.tokenizer))
    prompt = args.prompt if args.prompt_style == "raw" else "故事开头：{}\n续写：".format(args.prompt.strip())
    prompt_ids = [tokenizer.bos_id(), *tokenizer.encode(prompt)]
    if not prompt_ids:
        raise ValueError("The prompt must produce at least one token")

    captured = []

    def capture_w1_input(_module, inputs):
        captured.append(inputs[0].detach().to(device="cpu", dtype=torch.float32).reshape(-1).clone())

    hook = model.layers[0].feed_forward.w1.register_forward_pre_hook(capture_w1_input)
    key_cache, value_cache = model.new_cache(torch.device("cpu"))
    with torch.inference_mode():
        for position, token_id in enumerate(prompt_ids):
            model.step(int(token_id), position, key_cache, value_cache)
    hook.remove()
    if len(captured) != len(prompt_ids):
        raise RuntimeError("Expected one layer-0 FFN input per prompt token")
    activation = captured[-1].numpy().astype(np.float32, copy=False)

    entry = next(item for item in manifest["tensors"] if item["name"] == TENSOR_NAME)
    rows, columns = entry["shape"]
    if (rows, columns) != (352, 128) or activation.size != columns:
        raise ValueError("Unexpected layer-0 w1 or captured activation shape")
    blob = args.weights.read_bytes()
    packed = blob[entry["data_offset"] : entry["data_offset"] + entry["data_bytes"]]
    row_scales = np.frombuffer(
        blob[entry["scale_offset"] : entry["scale_offset"] + entry["scale_bytes"]], dtype="<f4"
    ).copy()
    q_weights = unpack_signed_int6(packed, rows, columns)
    if row_scales.size != rows:
        raise ValueError("Q6 row scale array has an invalid length")

    activation_scale = float(np.max(np.abs(activation)) / 127.0)
    if activation_scale == 0.0:
        activation_scale = 1.0
    activation_q = np.clip(np.rint(activation / activation_scale), -127, 127).astype(np.int8)
    accumulators64 = q_weights.astype(np.int64) @ activation_q.astype(np.int64)
    if np.any(accumulators64 < -(1 << 31)) or np.any(accumulators64 >= (1 << 31)):
        raise OverflowError("GEMV accumulator does not fit signed int32")
    accumulators = accumulators64.astype("<i4")

    q6_dequantized = q_weights.astype(np.float32) * row_scales[:, None]
    q6_reference = q6_dequantized @ activation
    fp32_weight = checkpoint["model"][TENSOR_NAME].detach().cpu().numpy().astype(np.float32, copy=False)
    fp32_reference = fp32_weight @ activation
    scaled_hardware = accumulators.astype(np.float32) * row_scales * np.float32(activation_scale)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_bytes = activation_q.tobytes() + packed
    (args.out_dir / "input.bin").write_bytes(input_bytes)
    (args.out_dir / "expected.bin").write_bytes(accumulators.tobytes())
    (args.out_dir / "row_scales.bin").write_bytes(row_scales.astype("<f4", copy=False).tobytes())
    (args.out_dir / "activation_scale.bin").write_bytes(struct.pack("<f", activation_scale))
    (args.out_dir / "q6_reference.bin").write_bytes(q6_reference.astype("<f4").tobytes())
    (args.out_dir / "fp32_reference.bin").write_bytes(fp32_reference.astype("<f4").tobytes())
    (args.out_dir / "activation_fp32.bin").write_bytes(activation.astype("<f4", copy=False).tobytes())
    write_hex(args.out_dir / "input.hex", input_bytes, 2)
    write_hex(args.out_dir / "expected.hex", accumulators, 8)

    report = {
        "tensor": TENSOR_NAME,
        "shape": [rows, columns],
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_sha256,
        "q6_weight_sha256": weights_sha256,
        "prompt_style": args.prompt_style,
        "prompt": prompt,
        "prompt_token_count": len(prompt_ids),
        "captured_position": len(prompt_ids) - 1,
        "captured_token_id": int(prompt_ids[-1]),
        "activation_scale": activation_scale,
        "activation_fp32_range": [float(activation.min()), float(activation.max())],
        "activation_int8_range": [int(activation_q.min()), int(activation_q.max())],
        "accumulator_range": [int(accumulators.min()), int(accumulators.max())],
        "input_bytes": len(input_bytes),
        "metrics": {
            "activation_quantized_q6_vs_q6_fp32": error_stats(scaled_hardware, q6_reference),
            "q6_dequantized_vs_checkpoint_fp32": error_stats(q6_reference, fp32_reference),
            "combined_hw_reconstruction_vs_checkpoint_fp32": error_stats(scaled_hardware, fp32_reference),
        },
        "note": "RTL outputs raw int32 accumulators; row and activation scales are applied on the host for this validation.",
    }
    (args.out_dir / "real_vector_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
