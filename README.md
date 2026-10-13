# TinyLLM on PYNQ-Z2

Run a small Q6 language model on a PYNQ-Z2 board with the autoregressive model
loop implemented as one programmable-logic (PL) accelerator. The ARM processor
prepares the request and starts the accelerator; tokenization and text decoding
remain software responsibilities.

This repository contains the RTL/HLS design, Vivado build scripts, host-side
reference preparation, and PYNQ runtime. Model weights, checkpoints, and the
SentencePiece tokenizer are **not** included.

## Measured performance

The fastest board result confirmed so far is **203.66 generated tokens/s** on a
PYNQ-Z2 with a 20-token prompt and 128 generated tokens. It is the median of five
runs (0.62850 s per request); all five runs matched every generated token and
all 128 per-step logit argmaxes against the Q6 PyTorch reference.

| Configuration | Generated tokens/s | Request | Validation |
| --- | ---: | --- | --- |
| Autonomous PL, W1/W3 projections parallel | **203.66** | 20 prompt + 128 generated | 5/5 exact token IDs and argmaxes |
| Tuned PS-only reference | 27.98 | Same 20 + 128 request, including prefill | Exact token IDs and argmaxes |

The PL result is about **7.3x** the PS-only end-to-end generated-token rate for
this model and request. It is a measured throughput comparison, not a claim of
10x speedup. Host tokenization, SSH/file transfer, model preparation, and
one-time overlay loading are excluded. The result returns logits for validation.

The measured overlay used a 127 MHz PL/HP clock with the PS FCLK at 125 MHz.
Vivado 2018.2 routed setup and hold slack were +0.020 ns and +0.014 ns,
respectively, with no failing endpoints. Results are specific to the TinyLLM Q6
model and the PYNQ-Z2 / XC7Z020.

The source also contains a K/V-parallel experimental configuration. It passed
HLS C simulation and Vivado timing signoff, but has **not** completed a board
benchmark; do not attribute the 203.66 tokens/s result to that experiment.

## Architecture

- The PL performs prompt prefill and the complete greedy decode loop, including
  embeddings, all transformer layers, attention, KV-cache updates, final norm,
  vocabulary projection, argmax, and stop-token handling.
- The model image and KV cache reside in PS DDR and are accessed through AXI HP
  ports. The model is not stored entirely in BRAM.
- The benchmark model is a 1.45M-parameter, five-layer Llama-style Chinese story
  model: dimension 128, four query heads, two KV heads, vocabulary 4096, and
  maximum context 512.
- The validated speed profile runs independent W1 and W3 FFN projections in
  parallel on separate memory ports. A later optional experiment also
  parallelizes K/V projections and uses a precomputed RoPE table.

## Requirements

- Vivado and Vivado HLS 2018.2, including the PYNQ-Z2 board files.
- PYNQ-Z2 with the PYNQ 2.5 image for the measured runtime.
- A host Python environment with NumPy, PyTorch, SentencePiece, and Paramiko.
- The companion model repository and its Q6W artifact/checkpoint. These files
  are separate from this code repository and are not redistributed here:
  [ModenCn/tinyllamas-zh](https://github.com/ModenCn/tinyllamas-zh).

Install host dependencies in an environment that can also import the companion
repository's `q6w_format.py` and `CachedTransformer` reference implementation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-host.txt
```

## Build and run the validated profile

Place this repository and the companion model repository in a shared workspace:

```text
workspace/
  LLM-PYNQ/q6w_gemv/
  TinyLLMStory-zh/
```

From the project root, prepare a fixed 20-token Chinese prompt and 128-token Q6
PyTorch golden request. Pass `--model-repo` if the companion checkout is
elsewhere; alternatively set `TINYLLM_MODEL_REPO`.

```powershell
$modelRepo = (Resolve-Path "..\..\TinyLLMStory-zh").Path
python scripts\prepare_e2e.py --model-repo $modelRepo --linear-mode ps `
  --max-new-tokens 128 --min-new-tokens 4 `
  --out-dir build\autonomous_hls\e2e128
python scripts\generate_hls_layout.py `
  --manifest build\autonomous_hls\e2e128\manifest.json `
  --weights build\autonomous_hls\e2e128\model_q6w.bin `
  --model-output build\autonomous_hls\e2e128\autonomous_model.bin `
  --output rtl\autonomous\model_layout.h
python scripts\prepare_hls_testdata.py --package-dir build\autonomous_hls\e2e128
```

Build/export the HLS IP. Set `AUTONOMOUS_PYTHON` to the Python executable from
the environment used above, and set `$vivadoHls` to your Vivado 2018.2 install:

```powershell
$env:AUTONOMOUS_HLS_PROJECT = "tinyllm_pynq_par"
$env:AUTONOMOUS_PACKAGE_DIR = (Resolve-Path "build\autonomous_hls\e2e128").Path
$env:AUTONOMOUS_PYTHON = (Get-Command python).Source
$env:AUTONOMOUS_PERFORMANCE_BUILD = "1"
$env:AUTONOMOUS_LEGACY_KV_MIRROR = "1"
$env:AUTONOMOUS_PARALLEL_FFN = "1"
$vivadoHls = "C:\Xilinx\Vivado\2018.2\bin\vivado_hls.bat"
& $vivadoHls -f scripts\build_autonomous_hls.tcl
```

The HLS script runs C simulation, synthesis, and IP export. Build the PYNQ-Z2
overlay with PS FCLK 125 MHz and PL/HP 127 MHz. The publish guard refuses to
copy a bitstream if routed timing does not pass setup and hold signoff.

```powershell
$env:AUTONOMOUS_OVERLAY_TAG = "par127"
$env:AUTONOMOUS_PL_MHZ = "125"
$env:AUTONOMOUS_CORE_MHZ = "127"
$env:AUTONOMOUS_MEMORY_FAST_CLOCK = "1"
$env:AUTONOMOUS_PHYS_OPT_DIRECTIVE = "AggressiveExplore"
$vivado = "C:\Xilinx\Vivado\2018.2\bin\vivado.bat"
& $vivado -mode batch -source scripts\build_autonomous_overlay.tcl
```

The overlay is emitted under
`build/autonomous_overlay_par127/output/`. For the board run, configure `PYNQ_HOST`
or pass `--host`; the deploy script prompts for the board password unless
`PYNQ_PASSWORD` is set in the local shell. Do not commit credentials.

```powershell
$env:PYNQ_HOST = "<your-pynq-z2-address>"
python scripts\deploy_autonomous.py `
  --package-dir build\autonomous_hls\e2e128 `
  --overlay-dir build\autonomous_overlay_par127\output `
  --remote-dir /home/xilinx/tinyllm-pynq/par127 `
  --clock-mhz 125 --runs 5 --dump-logits --performance-overlay `
  --legacy-kv-mirror --dual-ffn-w3
```

The board script loads the overlay at runtime. It does not rewrite the SD boot
partition or require a reboot.

### Optional K/V-parallel experiment

To build the newer, not-yet-board-benchmarked K/V-parallel variant, add these
HLS environment variables before running `build_autonomous_hls.tcl`:

```powershell
$env:AUTONOMOUS_PRECOMPUTED_ROPE = "1"
$env:AUTONOMOUS_PARALLEL_KV = "1"
```

The parallel K/V build requires `AUTONOMOUS_PARALLEL_FFN=1` and
`AUTONOMOUS_LEGACY_KV_MIRROR=1`. Its board runner uses `--parallel-kv`; benchmark
it independently before comparing or reporting its throughput.

## Repository layout

- `rtl/autonomous/`: full-model HLS accelerator and C-simulation testbench.
- `rtl/*.sv`, `sim/`: streaming Q6 GEMV RTL experiments and testbenches.
- `scripts/build_autonomous_*.tcl`: HLS and Vivado overlay builds.
- `scripts/prepare_*.py`: model layout, golden request, and simulation inputs.
- `scripts/deploy_*.py`, `autonomous_board.py`, `board_e2e.py`: deployment and
  board-side execution/validation.
- `build/`: generated projects, models, bitstreams, test vectors, and reports;
  intentionally ignored by Git.

## License and attribution

This repository's source code and RTL are licensed under the MIT License. The
model, checkpoint, tokenizer, and training data are not included and retain
their own terms. The reference model implementation and Q6W format are from
[ModenCn/tinyllamas-zh](https://github.com/ModenCn/tinyllamas-zh); see that
project for its license and the model/data provenance.
