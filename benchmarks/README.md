# Reference benchmark

`torch_benchmark.py` is the cached autoregressive PyTorch reference used to
prepare golden requests and compare the Q6W accelerator output. It is vendored
from the TinyLLMStory-zh model project; see `quantization/LICENSE` for the
upstream MIT notice and `docs/Q6_QUANTIZATION.md` for model provenance.
