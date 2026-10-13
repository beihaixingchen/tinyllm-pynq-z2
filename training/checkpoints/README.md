# Checkpoints

- `masked_v2_step23900.pt`: FP32 pre-QAT starting checkpoint.
- `q6w_qat_lr1e6_step25000.pt`: FP32 master/optimizer checkpoint after Q6W-aware
  fine-tuning.

These files are for PyTorch training and evaluation. The PYNQ runtime consumes
the packed signed-int6 `.bin` files under `quantization/artifacts/`, together
with their JSON manifests.

SHA-256:

| File | SHA-256 |
| --- | --- |
| `masked_v2_step23900.pt` | `8baea4008389992aa5d2d3888235a29193ceb8f16540b3a68770025952fdf43f` |
| `q6w_qat_lr1e6_step25000.pt` | `280d6155e3a309d5e32e72a7ddd9c87261bc11fbb7d4f75a8822206ef74ca169` |
