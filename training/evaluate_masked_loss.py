import argparse
import os
from contextlib import nullcontext

import torch


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint on the masked story validation set.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/story_pairs_masked_v2")
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--eval-iters", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--cpu-threads", type=int, default=1)
    args = parser.parse_args()

    os.environ["TINYLLAMA_DATA_DIR"] = args.data_dir
    torch.set_num_threads(args.cpu_threads)

    from model import ModelArgs, Transformer
    from tinystories import Task

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and args.dtype == "float16":
        raise ValueError("CPU evaluation requires --dtype=float32")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = Transformer(ModelArgs(**checkpoint["model_args"]))
    state = checkpoint["model"]
    for key in list(state):
        if key.startswith("_orig_mod."):
            state[key[len("_orig_mod.") :]] = state.pop(key)
    model.load_state_dict(state)
    model.to(device).eval()

    batches = Task.iter_batches(
        split=args.split,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        vocab_size=checkpoint["model_args"]["vocab_size"],
        vocab_source="custom",
        mask_loss=True,
        device=device,
        num_workers=0,
    )
    autocast = (
        torch.amp.autocast(device_type="cuda", dtype=dtype)
        if device == "cuda" and args.dtype == "float16"
        else nullcontext()
    )

    losses = []
    with torch.inference_mode():
        for _ in range(args.eval_iters):
            x, y = next(batches)
            with autocast:
                model(x, y)
                losses.append(model.last_loss.float().item())

    print(
        f"checkpoint={args.checkpoint} step={checkpoint['iter_num']} "
        f"split={args.split} batches={len(losses)} device={device} dtype={args.dtype} "
        f"masked_loss={sum(losses) / len(losses):.6f}"
    )


if __name__ == "__main__":
    main()
