import argparse
import json
import struct
from pathlib import Path

import numpy as np


ARTIFACT_NAME = "tinyllamas_zh_story_q6w_qat_lr1e6"
LOAD_COUNT = struct.Struct("<H")
LOAD_HEADER = struct.Struct("<HH")
COMPUTE_HEADER = struct.Struct("<HH")


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


def write_hex(path, data):
    path.write_text("".join("{:02x}\n".format(value) for value in data), encoding="ascii")


def main():
    project_dir = Path(__file__).resolve().parents[1]
    artifact_dir = project_dir / "quantization" / "artifacts"
    manifest = json.loads(
        (artifact_dir / (ARTIFACT_NAME + ".json")).read_text(encoding="utf-8")
    )
    blob = (artifact_dir / (ARTIFACT_NAME + ".bin")).read_bytes()
    entries = {item["name"]: item for item in manifest["tensors"]}

    out_dir = project_dir / "build" / "cached_vectors"
    args_parser = argparse.ArgumentParser(description="Prepare PL weight-cache vectors")
    args_parser.add_argument("--out-dir", type=Path, default=out_dir)
    args_parser.add_argument("--include-qkv", action="store_true")
    args = args_parser.parse_args()

    entries_by_slot = {}
    for layer in range(5):
        prefix = "layers.{}.feed_forward.".format(layer)
        for suffix in ("w1", "w3"):
            name = prefix + suffix + ".weight"
            entry = entries[name]
            rows, columns = (int(value) for value in entry["shape"])
            if (rows, columns) != (352, 128):
                raise ValueError("Unexpected cached matrix shape for {}".format(name))
            start = int(entry["data_offset"])
            packed = blob[start : start + int(entry["data_bytes"])]
            entries_by_slot[layer * 2 + (suffix == "w3")] = (name, rows, columns, packed)

    if args.include_qkv:
        for layer in range(5):
            prefix = "layers.{}.attention.".format(layer)
            qkv_names = [prefix + name + ".weight" for name in ("wq", "wk", "wv")]
            qkv_entries = [entries[name] for name in qkv_names]
            qkv_shapes = [tuple(int(value) for value in entry["shape"]) for entry in qkv_entries]
            if qkv_shapes != [(128, 128), (64, 128), (64, 128)]:
                raise ValueError("Unexpected QKV shapes for layer {}".format(layer))
            qkv_packed = b"".join(
                blob[int(entry["data_offset"]):
                     int(entry["data_offset"]) + int(entry["data_bytes"])]
                for entry in qkv_entries
            )
            entries_by_slot[10 + layer] = (
                "layers.{}.attention.qkv.weight".format(layer), 256, 128, qkv_packed
            )

    operation_count = 15 if args.include_qkv else 10
    output_rows = 4800 if args.include_qkv else 3520
    load_packet = bytearray(LOAD_COUNT.pack(operation_count))
    matrices = []
    for slot in range(operation_count):
        name, rows, columns, packed = entries_by_slot[slot]
        if len(packed) != rows * columns * 6 // 8:
            raise ValueError("Unexpected packed data size for {}".format(name))
        load_packet.extend(LOAD_HEADER.pack(0, slot))
        load_packet.extend(packed)
        matrices.append({"slot": slot, "name": name, "rows": rows, "columns": columns})

    compute_packet = bytearray(LOAD_COUNT.pack(operation_count))
    expected = []
    rng = np.random.RandomState(20260928)
    activation = None
    for slot in range(operation_count):
        name, rows, columns, packed = entries_by_slot[slot]
        reuse_activation = slot < 10 and (slot % 2) == 1
        if not reuse_activation:
            activation = rng.randint(-32767, 32768, size=columns).astype("<i2")
        qweights = unpack_signed_int6(packed, rows, columns)
        accumulators = np.matmul(qweights.astype(np.int32), activation.astype(np.int32))
        compute_packet.extend(COMPUTE_HEADER.pack(0xFFFE if reuse_activation else 0xFFFF, slot))
        if not reuse_activation:
            compute_packet.extend(activation.tobytes())
        expected.extend(int(value) for value in accumulators)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "load.bin").write_bytes(load_packet)
    (out_dir / "compute.bin").write_bytes(compute_packet)
    (out_dir / "expected.bin").write_bytes(np.asarray(expected, dtype="<i4").tobytes())
    write_hex(out_dir / "load.hex", load_packet)
    write_hex(out_dir / "compute.hex", compute_packet)
    (out_dir / "expected.hex").write_text(
        "".join("{:08x}\n".format(value & 0xFFFFFFFF) for value in expected),
        encoding="ascii",
    )
    report = {
        "q6_sha256": manifest["binary_sha256"],
        "load_bytes": len(load_packet),
        "compute_bytes": len(compute_packet),
        "output_rows": len(expected),
        "qkv_included": args.include_qkv,
        "activation_min": int(activation.min()),
        "activation_max": int(activation.max()),
        "accumulator_min": min(expected),
        "accumulator_max": max(expected),
        "matrices": matrices,
    }
    (out_dir / "vector_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
