import argparse
import hashlib
import json
import math
import struct
from pathlib import Path


MAGIC = b"TLQ6ROW\0"
HEADER = struct.Struct("<8sIIII")
TENSOR_NAME = "layers.0.feed_forward.w1.weight"


def decode_row(data):
    values = []
    for offset in range(0, len(data), 3):
        byte0, byte1, byte2 = data[offset : offset + 3]
        packed = (
            byte0 & 0x3F,
            (byte0 >> 6) | ((byte1 & 0x0F) << 2),
            (byte1 >> 4) | ((byte2 & 0x03) << 4),
            byte2 >> 2,
        )
        values.extend(value - 64 if value & 0x20 else value for value in packed)
    return values


def write_hex(path, values, width):
    with path.open("w", encoding="ascii", newline="\n") as target:
        for value in values:
            target.write("{:0{}X}\n".format(value & ((1 << (width * 4)) - 1), width))


def main():
    repo_dir = Path(__file__).resolve().parents[1]
    workspace = Path(__file__).resolve().parents[3]
    artifacts = workspace / "TinyLLMStory-zh" / "quantization" / "artifacts"

    parser = argparse.ArgumentParser(description="Prepare Q6W GEMV simulation vectors")
    parser.add_argument("--weights", type=Path, default=artifacts / "tinyllamas_zh_story_q6w_qat_lr1e6.bin")
    parser.add_argument("--manifest", type=Path, default=artifacts / "tinyllamas_zh_story_q6w_qat_lr1e6.json")
    parser.add_argument("--out-dir", type=Path, default=repo_dir / "build" / "sim")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    blob = args.weights.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != manifest["binary_sha256"]:
        raise ValueError("Q6 weight file SHA-256 does not match the manifest")

    magic, version, tensor_count, weight_bits, scale_dtype = HEADER.unpack_from(blob)
    expected_header = (MAGIC, manifest["version"], manifest["tensor_count"], 6, 1)
    if (magic, version, tensor_count, weight_bits, scale_dtype) != expected_header:
        raise ValueError("Unsupported or inconsistent Q6W header")

    entry = next(item for item in manifest["tensors"] if item["name"] == TENSOR_NAME)
    rows, columns = entry["shape"]
    if [rows, columns] != [352, 128]:
        raise ValueError("This first RTL test expects layers.0.feed_forward.w1 [352, 128]")
    row_bytes = columns * 6 // 8
    if entry["data_bytes"] != rows * row_bytes or columns % 4:
        raise ValueError("Unsupported Q6 row packing")

    packed_weights = blob[entry["data_offset"] : entry["data_offset"] + entry["data_bytes"]]
    scales = struct.unpack_from("<{}f".format(rows), blob, entry["scale_offset"])
    if len(packed_weights) != rows * row_bytes or len(scales) != rows:
        raise ValueError("Truncated Q6W tensor or row-scale array")

    activation = [math.sin(index * 0.173) + 0.25 * math.cos(index * 0.071) for index in range(columns)]
    activation_max = max(abs(value) for value in activation)
    activation_scale = activation_max / 127.0
    activation_q = [max(-127, min(127, round(value / activation_scale))) for value in activation]

    accumulators = []
    reference = []
    reconstructed = []
    for row in range(rows):
        start = row * row_bytes
        quantized = decode_row(packed_weights[start : start + row_bytes])
        if len(quantized) != columns or any(value < -31 or value > 31 for value in quantized):
            raise ValueError("Invalid signed-int6 row in Q6W tensor")
        accumulators.append(sum(weight * value for weight, value in zip(quantized, activation_q)))
        reference.append(sum(weight * scales[row] * value for weight, value in zip(quantized, activation)))
        reconstructed.append(accumulators[-1] * scales[row] * activation_scale)

    errors = [actual - expected for actual, expected in zip(reconstructed, reference)]
    rmse = math.sqrt(sum(error * error for error in errors) / rows)
    max_abs_error = max(abs(error) for error in errors)
    dot = sum(actual * expected for actual, expected in zip(reconstructed, reference))
    norm = math.sqrt(
        sum(actual * actual for actual in reconstructed)
        * sum(expected * expected for expected in reference)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    input_bytes = [value & 0xFF for value in activation_q] + list(packed_weights)
    (args.out_dir / "input.bin").write_bytes(bytes(input_bytes))
    (args.out_dir / "expected.bin").write_bytes(
        struct.pack("<{}i".format(rows), *accumulators)
    )
    write_hex(args.out_dir / "input.hex", input_bytes, 2)
    write_hex(args.out_dir / "expected.hex", accumulators, 8)
    report = {
        "tensor": TENSOR_NAME,
        "shape": [rows, columns],
        "input_bytes": len(input_bytes),
        "weight_sha256": digest,
        "activation_scale": activation_scale,
        "activation_int8_range": [min(activation_q), max(activation_q)],
        "accumulator_range": [min(accumulators), max(accumulators)],
        "int8_activation_vs_q6_fp32": {
            "rmse": rmse,
            "max_abs_error": max_abs_error,
            "cosine": dot / norm if norm else 1.0,
        },
    }
    (args.out_dir / "vector_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
