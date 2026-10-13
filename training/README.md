# Q6W training and reproduction

This directory contains the training code used to produce the Q6-aware model.
Most vendored training sources retain their MIT attribution in `LICENSE`.
`tokenizer.py` preserves Meta's separate Llama 2 Community License notice; see
[`NOTICE.md`](NOTICE.md) and [`LLAMA2_LICENSE.txt`](LLAMA2_LICENSE.txt).

The dependency pins are in `requirements.txt`. The recorded run used Python
3.8.16 and PyTorch 2.0.0+cu117; select a CUDA-enabled PyTorch wheel compatible
with the local GPU/driver before running the FP16 training recipe.

## Published checkpoints

| File | Step | SHA-256 | Purpose |
| --- | ---: | --- | --- |
| `checkpoints/masked_v2_step23900.pt` | 23,900 | `8baea4008389992aa5d2d3888235a29193ceb8f16540b3a68770025952fdf43f` | FP32 starting point for QAT; corresponds to the pre-QAT Q6W baseline |
| `checkpoints/q6w_qat_lr1e6_step25000.pt` | 25,000 | `280d6155e3a309d5e32e72a7ddd9c87261bc11fbb7d4f75a8822206ef74ca169` | FP32 master checkpoint after Q6W-aware fine-tuning |

The `.pt` files are PyTorch training checkpoints, not deployment files. The
corresponding packed Q6W files and manifests are in
`../quantization/artifacts/`.

## Dataset

The training corpus is not included. It was prepared from
[TinyStoriesChinese](https://huggingface.co/datasets/adam89/TinyStoriesChinese),
which is derived from TinyStories; follow both datasets' terms. Download or
prepare the source JSONL shards separately. The code below uses the included
`tok4096.model`, filters stories, creates a deterministic story-level split,
selects an interior sentence boundary and writes continuation-only loss masks.

From this directory, download the source shards:

```powershell
python tinystories.py download
```

Build and validate the masked dataset (the output directory must not already
exist):

```powershell
python prepare_random_prefix_data.py `
  --input-dir data/TinyStories_all_data `
  --output-dir data/story_pairs_masked_v2 `
  --tokenizer ..\tok4096.model `
  --seed 20260927
python validate_random_prefix_data.py --data-dir data/story_pairs_masked_v2
```

The original QAT run used 3,540,308 examples (3,363,122 train / 177,186
validation). Rebuilding from a changed upstream dataset revision can produce a
different corpus; the source data and generated token/mask files are purposely
not distributed here.

## Re-run QAT

The training code expects the resume checkpoint at `<out_dir>/ckpt.pt`. Make a
fresh output directory and copy the published step-23,900 checkpoint into it:

```powershell
New-Item -ItemType Directory -Path out_q6w_qat_lr1e6
Copy-Item checkpoints\masked_v2_step23900.pt out_q6w_qat_lr1e6\ckpt.pt
$env:TINYLLAMA_DATA_DIR = "data/story_pairs_masked_v2"
python train.py `
  --out_dir=out_q6w_qat_lr1e6 `
  --init_from=resume --max_iters=25099 `
  --eval_interval=100 --log_interval=50 --eval_iters=50 `
  --learning_rate=1e-6 --weight_decay=0.0 --decay_lr=False `
  --reset_optimizer=True --reset_best_val_loss=True `
  --early_stop_patience=5 --early_stop_min_delta=0.0001 `
  --qat_q6w=True --mask_loss=True --wandb_log=False `
  --batch_size=128 --gradient_accumulation_steps=1 --max_seq_len=512 `
  --vocab_source=custom --vocab_size=4096 `
  --dim=128 --n_layers=5 --n_heads=4 --n_kv_heads=2 --multiple_of=32 `
  --device=cuda --dtype=float16 --compile=False
```

`max_iters` is the absolute step cap in this trainer. Starting from step 23,900,
`25099` permits up to 1,200 updates; validation and patience can stop earlier.
The original run used CUDA FP16 with GradScaler, a fresh AdamW state, constant
learning rate, and selected step 25,000 as the best checkpoint.

Export a new deployment artifact under a new name so the published files remain
unchanged:

```powershell
python ..\quantization\quantize_q6w.py `
  --checkpoint out_q6w_qat_lr1e6\ckpt.pt `
  --output ..\quantization\artifacts\tinyllamas_zh_story_q6w_qat_retrained.bin `
  --manifest ..\quantization\artifacts\tinyllamas_zh_story_q6w_qat_retrained.json
```

For the original 50-batch masked-loss and fixed-prompt comparison, run
`quantization/evaluate_q6w.py` with the selected checkpoint, Q6W binary,
manifest, generated `eval_prompts.jsonl`, and this directory as `--training-root`.
The generated dataset is required for masked-loss evaluation.

## What the QAT result means

Fake quantization is applied to embedding/Linear weights on each training
forward pass; FP32 master weights are updated with an identity straight-through
estimator. RMSNorm parameters remain FP32. It is weight-only QAT, not integer
activation/KV training. On the original fixed validation set, Q6W masked loss
improved from 2.203846 to 2.193370 (0.48%); EOS/cap counts changed from 121/7 to
122/6. This supports compensation for the measured weight-quantization error,
not a broad semantic-quality claim.
