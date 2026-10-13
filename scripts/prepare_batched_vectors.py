import argparse
import json
import struct
from pathlib import Path

import numpy as np


ARTIFACT_NAME = "tinyllamas_zh_story_q6w_qat_lr1e6"
COUNT_HEADER = struct.Struct("<H")
MATRIX_HEADER = struct.Struct("<HH")


def unpack_signed_int6(data, rows, columns):
    packed = np.frombuffer(data, dtype=np.uint8).reshape(rows, columns // 4, 3)
    values = np.empty((rows, columns // 4, 4), dtype=np.uint8)
    values[:, :, 0] = packed[:, :, 0] & 0x3F
    values[:, :, 1] = (packed[:, :, 0] >> 6) | ((packed[:, :, 1] & 0x0F) << 2)
    values[:, :, 2] = (packed[:, :, 1] >> 4) | ((packed[:, :, 2] & 0x03) << 4)
    values[:, :, 3] = packed[:, :, 2] >> 2
    signed = values.reshape(rows, columns).astype(np.int16)
    signed[signed >= 32] -= 64
    return signed.astype(np.int8)


def main():
    project_dir = Path(__file__).resolve().parents[1]
    artifact_dir = project_dir / "quantization" / "artifacts"
    manifest = json.loads(
        (artifact_dir / (ARTIFACT_NAME + ".json")).read_text(encoding="utf-8")
    )
    blob = (artifact_dir / (ARTIFACT_NAME + ".bin")).read_bytes()
    entries = {item["name"]: item for item in manifest["tensors"]}

    parser = argparse.ArgumentParser(description="Prepare actual-Q6 batched GEMV RTL vectors")
    parser.add_argument("--case", choices=("linears", "qkv", "vocab"), required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.case == "linears":
        names = [
            "layers.0.attention.wq.weight",
            "layers.0.attention.wk.weight",
            "layers.0.attention.wv.weight",
            "layers.0.attention.wo.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w3.weight",
            "layers.0.feed_forward.w2.weight",
        ]
    elif args.case == "qkv":
        names = [
            "layers.0.attention.wq.weight",
            "layers.0.attention.wk.weight",
            "layers.0.attention.wv.weight",
        ]
    else:
        names = ["tok_embeddings.weight"]

    packet = bytearray(COUNT_HEADER.pack(len(names)))
    expected_values = []
    matrices = []
    for index, name in enumerate(names):
        entry = entries[name]
        if entry["dtype"] != "signed_int6_twos_complement":
            raise ValueError("Expected a Q6 matrix: {}".format(name))
        rows, columns = (int(value) for value in entry["shape"])
        if columns > 352 or columns % 4 or rows > 4096:
            raise ValueError("Matrix is outside batched GEMV limits: {}".format(name))
        start = int(entry["data_offset"])
        packed = blob[start : start + int(entry["data_bytes"])]
        qweights = unpack_signed_int6(packed, rows, columns)
        rng = np.random.RandomState(20260928 + index * 7919 + rows + columns)
        activation = rng.randint(-32767, 32768, size=columns).astype("<i2")
        accumulators = np.matmul(qweights.astype(np.int32), activation.astype(np.int32))
        packet.extend(MATRIX_HEADER.pack(columns, rows))
        packet.extend(activation.tobytes())
        packet.extend(packed)
        expected_values.extend(int(value) for value in accumulators)
        matrices.append({
            "name": name,
            "rows": rows,
            "columns": columns,
            "weight_bytes": len(packed),
            "accumulator_min": int(accumulators.min()),
            "accumulator_max": int(accumulators.max()),
        })

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input.bin").write_bytes(packet)
    (out_dir / "expected.bin").write_bytes(np.asarray(expected_values, dtype="<i4").tobytes())
    (out_dir / "input.hex").write_text(
        "".join("{:02x}\n".format(value) for value in packet), encoding="ascii"
    )
    (out_dir / "expected.hex").write_text(
        "".join("{:08x}\n".format(value & 0xFFFFFFFF) for value in expected_values),
        encoding="ascii",
    )
    report = {
        "case": args.case,
        "batch_count": len(names),
        "input_bytes": len(packet),
        "output_bytes": len(expected_values) * 4,
        "output_values": len(expected_values),
        "q6_sha256": manifest["binary_sha256"],
        "matrices": matrices,
    }
    (out_dir / "vector_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
