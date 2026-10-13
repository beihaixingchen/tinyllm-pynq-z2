import argparse
import json
from pathlib import Path

import numpy as np


def validate_split(root, manifest, split, filename, block_size):
    tokens = np.memmap(root / "tok4096" / filename, dtype=np.uint16, mode="r")
    masks = np.memmap(root / "tok4096_mask" / filename, dtype=np.uint8, mode="r")
    if len(tokens) != len(masks):
        raise ValueError(f"token/mask length mismatch for {split}: {len(tokens)} != {len(masks)}")

    expected_examples = manifest["counts"][f"{split}_examples"]
    expected_supervised = manifest["counts"][f"{split}_supervised_tokens"]
    bos_count = eos_count = supervised_count = upward = downward = 0
    previous_mask = 0
    for start in range(0, len(tokens), block_size):
        end = min(start + block_size, len(tokens))
        token_block = tokens[start:end]
        mask_block = masks[start:end]
        if np.any(mask_block > 1):
            raise ValueError(f"mask contains values other than 0/1 in {split}")
        bos_positions = token_block == manifest["bos_id"]
        eos_positions = token_block == manifest["eos_id"]
        if np.any(mask_block[bos_positions] != 0):
            raise ValueError(f"BOS targets must be masked in {split}")
        if np.any(mask_block[eos_positions] != 1):
            raise ValueError(f"EOS targets must be supervised in {split}")

        prior = np.empty_like(mask_block)
        prior[0] = previous_mask
        prior[1:] = mask_block[:-1]
        upward += int(np.count_nonzero((prior == 0) & (mask_block == 1)))
        downward += int(np.count_nonzero((prior == 1) & (mask_block == 0)))
        bos_count += int(np.count_nonzero(bos_positions))
        eos_count += int(np.count_nonzero(eos_positions))
        supervised_count += int(np.count_nonzero(mask_block))
        previous_mask = int(mask_block[-1])

    if bos_count != expected_examples or eos_count != expected_examples:
        raise ValueError(f"BOS/EOS/example counts do not match in {split}")
    if supervised_count != expected_supervised:
        raise ValueError(f"supervised token count does not match in {split}")
    if upward != expected_examples or downward != expected_examples - 1:
        raise ValueError(f"loss mask is not one contiguous continuation suffix per story in {split}")

    return {
        "tokens": len(tokens),
        "examples": bos_count,
        "supervised_tokens": supervised_count,
        "mask_up_transitions": upward,
        "mask_down_transitions": downward,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate aligned token and continuation-loss-mask bins")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=16_000_000)
    args = parser.parse_args()
    if args.block_size < 1:
        parser.error("block-size must be positive")

    manifest = json.loads((args.data_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise ValueError("refusing to validate an incomplete dataset")
    result = {
        "train": validate_split(args.data_dir, manifest, "train", "data01_train.bin", args.block_size),
        "validation": validate_split(args.data_dir, manifest, "validation", "data00_val.bin", args.block_size),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
