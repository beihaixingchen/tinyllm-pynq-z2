import argparse
import sys
from pathlib import Path

import torch


workspace = Path(__file__).resolve().parent
project_root = workspace.parent
sys.path.insert(0, str(project_root / "quantization"))

from q6w_fake_quant import fake_quantize_q6w, install_q6w_fake_quant
from q6w_format import QUANT_KEYS, load_dequantized_state
from model import ModelArgs, Transformer


def main():
    parser = argparse.ArgumentParser(description="Check Q6W fake-quant equivalence and STE gradients")
    parser.add_argument(
        "--checkpoint", type=Path,
        default=workspace / "checkpoints" / "masked_v2_step23900.pt",
    )
    parser.add_argument(
        "--binary", type=Path,
        default=project_root / "quantization" / "artifacts" / "tinyllamas_zh_story_q6w.bin",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=project_root / "quantization" / "artifacts" / "tinyllamas_zh_story_q6w.json",
    )
    args = parser.parse_args()
    source_checkpoint = args.checkpoint
    q6_binary = args.binary
    q6_manifest = args.manifest
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    dequantized, _ = load_dequantized_state(q6_binary, q6_manifest, state)

    for key in QUANT_KEYS:
        actual = fake_quantize_q6w(state[key])
        if not torch.equal(actual, dequantized[key]):
            raise AssertionError(f"Fake quantization differs from exported Q6W weights: {key}")

    sample = torch.nn.Parameter(torch.tensor([[0.0, 0.5, -1.0, 2.0]], dtype=torch.float32))
    fake_quantize_q6w(sample).sum().backward()
    if not torch.equal(sample.grad, torch.ones_like(sample)):
        raise AssertionError("STE gradient is not identity")

    regular = Transformer(ModelArgs(**checkpoint["model_args"]))
    regular.load_state_dict(dequantized, strict=True)
    regular.eval()
    quant_aware = Transformer(ModelArgs(**checkpoint["model_args"]))
    quant_aware.load_state_dict(state, strict=True)
    keys_before = set(quant_aware.state_dict())
    counts = install_q6w_fake_quant(quant_aware)
    quant_aware.eval()
    if set(quant_aware.state_dict()) != keys_before:
        raise AssertionError("Fake quantization changed checkpoint state-dict keys")

    tokens = torch.tensor([[1, 24, 87, 412, 2]], dtype=torch.long)
    with torch.inference_mode():
        expected = regular(tokens)
        actual = quant_aware(tokens)
    if not torch.equal(actual, expected):
        error = (actual - expected).abs().max().item()
        raise AssertionError(f"Fake-quant model differs from dequantized Q6W model: max_abs={error}")

    print(
        f"fake-quant tensor equality={len(QUANT_KEYS)}/{len(QUANT_KEYS)}; "
        f"STE identity=pass; module counts={counts}; state_dict keys=unchanged; model logits=exact"
    )


if __name__ == "__main__":
    main()
