import argparse
import json
import math
import struct
from pathlib import Path


def align(value, boundary=64):
    return (value + boundary - 1) & ~(boundary - 1)


def unpack_q6(data, count):
    values = bytearray(count)
    for index in range(0, count, 4):
        offset = (index // 4) * 3
        b0, b1, b2 = data[offset:offset + 3]
        packed = (
            b0 & 0x3f,
            ((b0 >> 6) | ((b1 & 0x0f) << 2)) & 0x3f,
            ((b1 >> 4) | ((b2 & 0x03) << 4)) & 0x3f,
            (b2 >> 2) & 0x3f,
        )
        for lane, value in enumerate(packed):
            if value & 0x20:
                value -= 64
            values[index + lane] = value & 0xff
    return values


def float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def make_rope_table(model_args):
    dim = int(model_args["dim"])
    heads = int(model_args["n_heads"])
    max_seq = int(model_args["max_seq_len"])
    head_dim = dim // heads
    if dim % heads or head_dim % 2:
        raise ValueError("RoPE table requires an even head dimension")

    ratio = float32(math.pow(10000.0, -2.0 / head_dim))
    table = bytearray(max_seq * head_dim * 4)
    frequency = float32(1.0)
    for pair in range(head_dim // 2):
        for position in range(max_seq):
            angle = float32(float32(position) * frequency)
            struct.pack_into(
                "<ff", table, (position * head_dim + pair * 2) * 4,
                float32(math.cos(angle)), float32(math.sin(angle)),
            )
        frequency = float32(frequency * ratio)
    return table


def main():
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build the aligned HLS model image from a Q6W manifest")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=project_dir / "rtl" / "autonomous" / "model_layout.h",
    )
    parser.add_argument(
        "--model-output", type=Path,
        default=project_dir / "build" / "autonomous_hls" / "autonomous_model.bin",
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "TinyLLMStory-zh Q6W rowwise v1":
        raise ValueError("Unsupported Q6 model format")
    if manifest["quantization"].get("packing") != (
        "four signed 6-bit two's-complement values packed low-bit-first into 3 bytes"
    ):
        raise ValueError("Unsupported Q6 packing")

    original = args.weights.read_bytes()
    if len(original) != manifest["binary_bytes"]:
        raise ValueError("Q6 binary size does not match its manifest")
    model = bytearray()
    offsets = []
    token_embedding_tensor = None
    for tensor in manifest["tensors"]:
        if tensor["dtype"] == "signed_int6_twos_complement":
            scale_offset = align(len(model))
            model.extend(b"\0" * (scale_offset - len(model)))
            scale_bytes = tensor["scale_bytes"]
            model.extend(original[tensor["scale_offset"]:tensor["scale_offset"] + scale_bytes])
            data_offset = align(len(model))
            model.extend(b"\0" * (data_offset - len(model)))
            packed = original[tensor["data_offset"]:tensor["data_offset"] + tensor["data_bytes"]]
            model.extend(unpack_q6(packed, tensor["values"]))
            offsets.append((scale_offset, data_offset))
            if tensor["name"] == "tok_embeddings.weight":
                token_embedding_tensor = tensor
        else:
            data_offset = align(len(model))
            model.extend(b"\0" * (data_offset - len(model)))
            data_bytes = tensor["data_bytes"]
            model.extend(original[tensor["data_offset"]:tensor["data_offset"] + data_bytes])
            offsets.append((None, data_offset))
    model.extend(b"\0" * (align(len(model)) - len(model)))
    lines = [
        "#ifndef TINYLLM_MODEL_LAYOUT_H",
        "#define TINYLLM_MODEL_LAYOUT_H",
        "",
        "#define MODEL_BINARY_BYTES {}u".format(len(model)),
    ]
    for key, macro in (
        ("dim", "MODEL_DIM"), ("n_layers", "MODEL_LAYERS"),
        ("n_heads", "MODEL_HEADS"), ("n_kv_heads", "MODEL_KV_HEADS"),
        ("vocab_size", "MODEL_VOCAB_SIZE"), ("max_seq_len", "MODEL_MAX_SEQ_LEN"),
    ):
        lines.append("#define {} {}u".format(macro, manifest["model_args"][key]))
    lines.extend(["", "#define MODEL_NORM_EPS 1.0e-5f", ""])

    for tensor, (scale_offset, data_offset) in zip(manifest["tensors"], offsets):
        macro = "TENSOR_" + "_".join(
            part.upper() for part in tensor["name"].replace(".weight", "").split(".")
        )
        if scale_offset is not None:
            lines.append("#define {}_SCALE_OFFSET {}u".format(macro, scale_offset))
        lines.append("#define {}_DATA_OFFSET {}u".format(macro, data_offset))
        shape = tensor["shape"]
        if len(shape) == 2:
            lines.append("#define {}_ROWS {}u".format(macro, shape[0]))
            lines.append("#define {}_COLS {}u".format(macro, shape[1]))

    if token_embedding_tensor is None:
        raise ValueError("Model manifest has no tok_embeddings.weight tensor")
    stream_offset = align(len(model))
    model.extend(b"\0" * (stream_offset - len(model)))
    rows, columns = token_embedding_tensor["shape"]
    scale_start = token_embedding_tensor["scale_offset"]
    scale_bytes = token_embedding_tensor["scale_bytes"]
    scales = original[scale_start:scale_start + scale_bytes]
    data_start = token_embedding_tensor["data_offset"]
    packed_data = original[data_start:data_start + token_embedding_tensor["data_bytes"]]
    q8_data = unpack_q6(packed_data, token_embedding_tensor["values"])
    if len(scales) != rows * 4 or len(q8_data) != rows * columns:
        raise ValueError("tok_embeddings.weight row data is inconsistent with its shape")
    for row in range(rows):
        model.extend(scales[row * 4:(row + 1) * 4])
        model.extend(b"\0" * 4)
        model.extend(q8_data[row * columns:(row + 1) * columns])

    matrix_streams = []
    for tensor in manifest["tensors"]:
        if (tensor["dtype"] != "signed_int6_twos_complement"
                or len(tensor["shape"]) != 2
                or tensor["name"] == "tok_embeddings.weight"):
            continue
        matrix_rows, matrix_columns = tensor["shape"]
        if matrix_columns % 8:
            raise ValueError("Matrix stream columns must be divisible by 8")
        scale_bytes = original[
            tensor["scale_offset"]:tensor["scale_offset"] + tensor["scale_bytes"]
        ]
        packed = original[
            tensor["data_offset"]:tensor["data_offset"] + tensor["data_bytes"]
        ]
        matrix_q8 = unpack_q6(packed, tensor["values"])
        if len(scale_bytes) != matrix_rows * 4 or len(matrix_q8) != matrix_rows * matrix_columns:
            raise ValueError("{} row data is inconsistent with its shape".format(tensor["name"]))
        matrix_stream_offset = align(len(model))
        model.extend(b"\0" * (matrix_stream_offset - len(model)))
        for row in range(matrix_rows):
            model.extend(scale_bytes[row * 4:(row + 1) * 4])
            model.extend(b"\0" * 4)
            start = row * matrix_columns
            model.extend(matrix_q8[start:start + matrix_columns])
        macro = "TENSOR_" + "_".join(
            part.upper() for part in tensor["name"].replace(".weight", "").split(".")
        )
        matrix_streams.append((macro, matrix_stream_offset, 1 + matrix_columns // 8))

    lines.extend([
        "#define TENSOR_TOK_EMBEDDINGS_STREAM_OFFSET {}u".format(stream_offset),
        "#define TENSOR_TOK_EMBEDDINGS_STREAM_ROW_WORDS {}u".format(1 + columns // 8),
    ])
    for macro, matrix_stream_offset, row_words in matrix_streams:
        lines.append("#define {}_STREAM_OFFSET {}u".format(macro, matrix_stream_offset))
        lines.append("#define {}_STREAM_ROW_WORDS {}u".format(macro, row_words))
    model.extend(b"\0" * (align(len(model)) - len(model)))
    lines[3] = "#define MODEL_BINARY_BYTES {}u".format(len(model))
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_bytes(model)
    rope_table = make_rope_table(manifest["model_args"])
    (args.model_output.parent / "rope_table.bin").write_bytes(rope_table)
    lines.extend(["", "#endif", ""])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="ascii")
    print(json.dumps({
        "q6_source_bytes": len(original),
        "autonomous_int8_model_bytes": len(model),
        "rope_table_bytes": len(rope_table),
        "tensor_count": len(offsets),
        "layout": str(args.output.resolve()),
        "model": str(args.model_output.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
