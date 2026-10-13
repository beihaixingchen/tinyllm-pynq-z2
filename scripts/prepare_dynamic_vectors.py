import argparse
import json
import struct
from pathlib import Path

import numpy as np


HEADER = struct.Struct("<HH")


def unpack_signed_int6(data, rows, columns):
    packed = np.frombuffer(data, dtype=np.uint8).reshape(rows, columns // 4, 3)
    unsigned = np.empty((rows, columns // 4, 4), dtype=np.uint8)
    unsigned[:, :, 0] = packed[:, :, 0] & 0x3F
    unsigned[:, :, 1] = (packed[:, :, 0] >> 6) | ((packed[:, :, 1] & 0x0F) << 2)
    unsigned[:, :, 2] = (packed[:, :, 1] >> 4) | ((packed[:, :, 2] & 0x03) << 4)
    unsigned[:, :, 3] = packed[:, :, 2] >> 2
    signed = unsigned.reshape(rows, columns).astype(np.int16)
    signed[signed >= 32] -= 64
    return signed.astype(np.int8)


def main():
    workspace = Path(__file__).resolve().parents[3]
    tiny_repo = workspace / "TinyLLMStory-zh"
    artifact_dir = tiny_repo / "quantization" / "artifacts"
    parser = argparse.ArgumentParser(description="Prepare a real-Q6 dynamic GEMV RTL test vector")
    parser.add_argument("--matrix", default="layers.0.feed_forward.w1.weight")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = artifact_dir / "tinyllamas_zh_story_q6w_qat_lr1e6.json"
    weight_path = artifact_dir / "tinyllamas_zh_story_q6w_qat_lr1e6.bin"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next((item for item in manifest["tensors"] if item["name"] == args.matrix), None)
    if entry is None or entry["dtype"] != "signed_int6_twos_complement":
        raise ValueError("Matrix is not a Q6 tensor in the artifact manifest")
    rows, columns = (int(value) for value in entry["shape"])
    if columns > 352 or rows > 352 or columns % 4:
        raise ValueError("Test matrix exceeds the dynamic GEMV limits")

    blob = weight_path.read_bytes()
    start = int(entry["data_offset"])
    packed = blob[start : start + int(entry["data_bytes"])]
    qweights = unpack_signed_int6(packed, rows, columns)
    activation = np.random.RandomState(20260928 + rows + columns).randint(-127, 128, size=columns).astype(np.int8)
    expected = np.matmul(qweights.astype(np.int32), activation.astype(np.int32))
    packet = HEADER.pack(columns, rows) + activation.tobytes() + packed

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "input.hex").write_text("".join("{:02x}\n".format(byte) for byte in packet), encoding="ascii")
    (out_dir / "expected.hex").write_text(
        "".join("{:08x}\n".format(int(value) & 0xFFFFFFFF) for value in expected), encoding="ascii"
    )
    report = {
        "matrix": args.matrix,
        "rows": rows,
        "columns": columns,
        "input_bytes": len(packet),
        "weight_bytes": len(packed),
        "output_bytes": expected.nbytes,
        "accumulator_min": int(expected.min()),
        "accumulator_max": int(expected.max()),
        "q6_sha256": manifest["binary_sha256"],
    }
    (out_dir / "vector_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
