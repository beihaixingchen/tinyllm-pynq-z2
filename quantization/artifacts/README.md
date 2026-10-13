# Q6W model artifacts

The default deployment file is `tinyllamas_zh_story_q6w_qat_lr1e6.bin` with its
same-stem JSON manifest. `tinyllamas_zh_story_q6w.bin` is the pre-QAT baseline
for controlled comparison. The manifest is part of the format: it defines the
tensor order, dimensions, scales, and byte offsets.

Both weight files are 1,130,816 bytes. Verify them with:

```powershell
Get-FileHash -Algorithm SHA256 *.bin
```

QAT SHA-256:
`ddeaa72c5923f2c445a7543cf53dc13dffc674f53f49f822c0653ef6735c4c90`

Baseline SHA-256:
`4edc130ef84cbb4450667a3e74ad5f8403608b3f90247256c006ec3604c7a0bf`

The tokenizer is stored at the repository root as `tok4096.model`. Model and
tokenizer notices are in `quantization/LICENSE`. Training source and checkpoints
are documented in `training/README.md`.
