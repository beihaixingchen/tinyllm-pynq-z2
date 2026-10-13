import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np
import torch


MAGIC = b"TLQ6ROW\0"
VERSION = 1
HEADER = struct.Struct("<8sIIII")
ALIGNMENT = 64
QMIN = -31
QMAX = 31
QUANT_KEYS = ["tok_embeddings.weight"]
NORM_KEYS = []
for _layer in range(5):
    QUANT_KEYS.extend(
        [
            f"layers.{_layer}.attention.wq.weight",
            f"layers.{_layer}.attention.wk.weight",
            f"layers.{_layer}.attention.wv.weight",
            f"layers.{_layer}.attention.wo.weight",
            f"layers.{_layer}.feed_forward.w1.weight",
            f"layers.{_layer}.feed_forward.w2.weight",
            f"layers.{_layer}.feed_forward.w3.weight",
        ]
    )
    NORM_KEYS.extend(
        [
            f"layers.{_layer}.attention_norm.weight",
            f"layers.{_layer}.ffn_norm.weight",
        ]
    )
NORM_KEYS.append("norm.weight")


def align_up(value, alignment=ALIGNMENT):
    return (value + alignment - 1) // alignment * alignment


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pack_signed_int6(values):
    """Pack a 2D signed-int6 tensor row-wise, four 6-bit values per 3 bytes."""
    values = np.asarray(values, dtype=np.int16)
    if values.ndim != 2 or values.shape[1] % 4:
        raise ValueError("Q6 rows must be 2D and have a width divisible by four")
    if np.any(values < QMIN) or np.any(values > QMAX):
        raise ValueError(f"Q6 values must be in [{QMIN}, {QMAX}]")
    unsigned = np.bitwise_and(values, 0x3F).astype(np.uint8).reshape(values.shape[0], -1, 4)
    packed = np.empty((values.shape[0], unsigned.shape[1], 3), dtype=np.uint8)
    packed[:, :, 0] = unsigned[:, :, 0] | ((unsigned[:, :, 1] & 0x03) << 6)
    packed[:, :, 1] = (unsigned[:, :, 1] >> 2) | ((unsigned[:, :, 2] & 0x0F) << 4)
    packed[:, :, 2] = (unsigned[:, :, 2] >> 4) | (unsigned[:, :, 3] << 2)
    return packed.tobytes(order="C")


def unpack_signed_int6(data, rows, columns):
    if columns % 4:
        raise ValueError("Q6 row width must be divisible by four")
    expected = rows * columns * 6 // 8
    if len(data) != expected:
        raise ValueError(f"Packed Q6 length mismatch: expected {expected}, got {len(data)}")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(rows, columns // 4, 3)
    unsigned = np.empty((rows, columns // 4, 4), dtype=np.uint8)
    unsigned[:, :, 0] = packed[:, :, 0] & 0x3F
    unsigned[:, :, 1] = (packed[:, :, 0] >> 6) | ((packed[:, :, 1] & 0x0F) << 2)
    unsigned[:, :, 2] = (packed[:, :, 1] >> 4) | ((packed[:, :, 2] & 0x03) << 4)
    unsigned[:, :, 3] = packed[:, :, 2] >> 2
    signed = unsigned.astype(np.int8)
    signed[unsigned >= 32] -= 64
    return signed.reshape(rows, columns)


def quantize_rows(tensor):
    values = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if values.ndim != 2:
        raise ValueError("Per-output-row quantization requires a matrix")
    maxima = values.abs().amax(dim=1)
    scales = torch.where(maxima > 0, maxima / QMAX, torch.ones_like(maxima))
    quantized = torch.round(values / scales[:, None]).clamp(QMIN, QMAX).to(torch.int8)
    reconstructed = quantized.float() * scales[:, None]
    error = reconstructed - values
    stats = {
        "max_abs_error": float(error.abs().max().item()),
        "mean_abs_error": float(error.abs().mean().item()),
        "rmse": float(torch.sqrt(torch.mean(error.square())).item()),
    }
    return quantized.numpy(), scales.numpy().astype("<f4", copy=False), stats


def _write_aligned(target, data):
    target.seek(align_up(target.tell()))
    offset = target.tell()
    target.write(data)
    return offset


def export_q6w(checkpoint_path, output_path, manifest_path):
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing Q6 artifact or manifest")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint["model"]
    if not torch.equal(state["tok_embeddings.weight"], state["output.weight"]):
        raise ValueError("This Q6W exporter expects tied input embeddings/output weights")
    expected = set(QUANT_KEYS + NORM_KEYS + ["output.weight"])
    if set(state) != expected:
        raise ValueError("Checkpoint tensor names do not match the documented TinyLLMStory-zh architecture")
    for key in QUANT_KEYS:
        if state[key].ndim != 2 or state[key].shape[1] % 4:
            raise ValueError(f"Unsupported quantized tensor shape for {key}: {tuple(state[key].shape)}")
        if not torch.isfinite(state[key]).all():
            raise ValueError(f"Non-finite values found in {key}")
    for key in NORM_KEYS:
        if not torch.isfinite(state[key]).all():
            raise ValueError(f"Non-finite values found in {key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as target:
        target.write(HEADER.pack(MAGIC, VERSION, len(QUANT_KEYS) + len(NORM_KEYS), 6, 1))
        target.write(b"\0" * (align_up(target.tell()) - target.tell()))

        for key in QUANT_KEYS:
            tensor = state[key]
            quantized, scales, stats = quantize_rows(tensor)
            packed = pack_signed_int6(quantized)
            unpacked = unpack_signed_int6(packed, tensor.shape[0], tensor.shape[1])
            if not np.array_equal(unpacked, quantized):
                raise AssertionError(f"Q6 pack/unpack self-check failed for {key}")
            scale_bytes = scales.tobytes(order="C")
            scale_offset = _write_aligned(target, scale_bytes)
            data_offset = _write_aligned(target, packed)
            entries.append(
                {
                    "name": key,
                    "shape": list(tensor.shape),
                    "dtype": "signed_int6_twos_complement",
                    "layout": "row_major; each output row has one symmetric scale",
                    "qmin": QMIN,
                    "qmax": QMAX,
                    "scale_dtype": "little_endian_float32",
                    "scale_shape": [tensor.shape[0]],
                    "scale_offset": scale_offset,
                    "scale_bytes": len(scale_bytes),
                    "data_offset": data_offset,
                    "data_bytes": len(packed),
                    "values": int(tensor.numel()),
                    "stats": stats,
                }
            )

        for key in NORM_KEYS:
            tensor = state[key].detach().to(device="cpu", dtype=torch.float32).contiguous()
            data = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
            data_offset = _write_aligned(target, data)
            entries.append(
                {
                    "name": key,
                    "shape": list(tensor.shape),
                    "dtype": "little_endian_float32",
                    "layout": "row_major",
                    "data_offset": data_offset,
                    "data_bytes": len(data),
                    "values": int(tensor.numel()),
                }
            )

    repository_root = Path(__file__).resolve().parents[1]
    try:
        checkpoint_reference = checkpoint_path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        checkpoint_reference = checkpoint_path.name

    manifest = {
        "format": "TinyLLMStory-zh Q6W rowwise v1",
        "magic_ascii": MAGIC.rstrip(b"\0").decode("ascii"),
        "version": VERSION,
        "quantization": {
            "kind": "weight-only symmetric per-output-row int6",
            "integer_range": [QMIN, QMAX],
            "scale": "max(abs(row))/31; all-zero rows use 1.0",
            "rounding": "nearest via torch.round (ties-to-even)",
            "packing": "four signed 6-bit two's-complement values packed low-bit-first into 3 bytes",
            "activations": "FP32 reference; activation quantization is not included",
            "norm_parameters": "FP32",
            "tied_output": "output.weight aliases tok_embeddings.weight; stored once",
        },
        "model_args": checkpoint["model_args"],
        "source_checkpoint": checkpoint_reference,
        "source_step": int(checkpoint["iter_num"]),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "unique_parameter_count": sum(tensor.numel() for key, tensor in state.items() if key != "output.weight"),
        "q6_value_count": sum(int(state[key].numel()) for key in QUANT_KEYS),
        "fp32_norm_value_count": sum(int(state[key].numel()) for key in NORM_KEYS),
        "tensor_count": len(entries),
        "header": {
            "struct": "<8sIIII",
            "bytes": HEADER.size,
            "fields": ["magic[8]", "version:uint32", "tensor_count:uint32", "weight_bits:uint32", "scale_dtype:uint32 (1=FP32)"],
            "payload_alignment_bytes": ALIGNMENT,
        },
        "binary_file": output_path.name,
        "binary_bytes": output_path.stat().st_size,
        "binary_sha256": sha256_file(output_path),
        "fp32_unique_weight_bytes": sum(tensor.numel() for key, tensor in state.items() if key != "output.weight") * 4,
        "tensors": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("x", encoding="utf-8", newline="\n") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return manifest


def load_dequantized_state(binary_path, manifest_path, checkpoint_state):
    binary_path = Path(binary_path)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(binary_path) != manifest["binary_sha256"]:
        raise ValueError("Q6 binary SHA-256 does not match its manifest")
    with binary_path.open("rb") as source:
        header_bytes = source.read(HEADER.size)
    magic, version, tensor_count, bits, scale_dtype = HEADER.unpack(header_bytes)
    if (magic, version, tensor_count, bits, scale_dtype) != (MAGIC, VERSION, manifest["tensor_count"], 6, 1):
        raise ValueError("Q6 binary header does not match the supported format")

    result = {key: value.detach().to(device="cpu", dtype=torch.float32).clone() for key, value in checkpoint_state.items()}
    with binary_path.open("rb") as source:
        for entry in manifest["tensors"]:
            shape = tuple(entry["shape"])
            source.seek(entry["data_offset"])
            data = source.read(entry["data_bytes"])
            if len(data) != entry["data_bytes"]:
                raise ValueError(f"Truncated Q6 data for {entry['name']}")
            if entry["dtype"] == "signed_int6_twos_complement":
                rows, columns = shape
                source.seek(entry["scale_offset"])
                scale_data = source.read(entry["scale_bytes"])
                scales = np.frombuffer(scale_data, dtype="<f4")
                if len(scales) != rows or entry["scale_bytes"] != rows * 4:
                    raise ValueError(f"Invalid scale array for {entry['name']}")
                quantized = unpack_signed_int6(data, rows, columns)
                restored = quantized.astype(np.float32) * scales[:, None]
                result[entry["name"]] = torch.from_numpy(restored.copy())
            elif entry["dtype"] == "little_endian_float32":
                values = np.frombuffer(data, dtype="<f4")
                if values.size != math.prod(shape):
                    raise ValueError(f"Invalid FP32 tensor size for {entry['name']}")
                result[entry["name"]] = torch.from_numpy(values.copy().reshape(shape))
            else:
                raise ValueError(f"Unsupported tensor dtype {entry['dtype']}")
    result["output.weight"] = result["tok_embeddings.weight"]
    return result, manifest
