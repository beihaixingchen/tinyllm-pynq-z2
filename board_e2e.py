import argparse
import ctypes
import hashlib
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
from pynq import Overlay, allocate
from pynq.ps import Clocks


ROWS = 352
COLUMNS = 128
INPUT_BYTES = COLUMNS + ROWS * COLUMNS * 6 // 8
HEADER = struct.Struct("<8sIIII")
LINEAR_HEADER = struct.Struct("<HH")
MAX_DMA_TRANSFER_BYTES = (1 << 16) - 1


class TimingProfile:
    def __init__(self):
        self.values = {}

    def add(self, name, seconds):
        entry = self.values.setdefault(name, [0.0, 0])
        entry[0] += seconds
        entry[1] += 1

    def snapshot(self):
        return {name: [value[0], value[1]] for name, value in self.values.items()}

    @staticmethod
    def delta(before, after):
        result = {}
        for name, value in after.items():
            previous = before.get(name, [0.0, 0])
            result[name] = [value[0] - previous[0], value[1] - previous[1]]
        return result


def unpack_signed_int6(data, rows, columns):
    if columns % 4:
        raise ValueError("Q6 row width must be divisible by four")
    packed = np.frombuffer(data, dtype=np.uint8)
    expected_bytes = rows * columns * 6 // 8
    if packed.size != expected_bytes:
        raise ValueError("Invalid packed Q6 matrix length")
    packed = packed.reshape(rows, columns // 4, 3)
    values = np.empty((rows, columns // 4, 4), dtype=np.uint8)
    values[:, :, 0] = packed[:, :, 0] & 0x3F
    values[:, :, 1] = (packed[:, :, 0] >> 6) | ((packed[:, :, 1] & 0x0F) << 2)
    values[:, :, 2] = (packed[:, :, 1] >> 4) | ((packed[:, :, 2] & 0x03) << 4)
    values[:, :, 3] = packed[:, :, 2] >> 2
    signed = values.reshape(rows, columns).astype(np.int16)
    signed[signed >= 32] -= 64
    return signed.astype(np.int8)


def error_stats(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    delta = actual - reference
    ref_rms = float(np.sqrt(np.mean(reference * reference)))
    actual_norm = float(np.linalg.norm(actual))
    ref_norm = float(np.linalg.norm(reference))
    cosine = float(np.dot(actual, reference) / (actual_norm * ref_norm)) if actual_norm and ref_norm else 1.0
    return {
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "max_abs": float(np.max(np.abs(delta))),
        "relative_rmse": float(np.sqrt(np.mean(delta * delta)) / ref_rms) if ref_rms else 0.0,
        "cosine": cosine,
        "argmax_match": int(np.argmax(actual)) == int(np.argmax(reference)),
    }


class Q6Model:
    def __init__(self, package_dir, linear_mode="ffn"):
        self.package_dir = package_dir
        self.linear_mode = linear_mode
        manifest = json.loads((package_dir / "manifest.json").read_text(encoding="utf-8"))
        blob = (package_dir / "model_q6w.bin").read_bytes()
        actual_hash = hashlib.sha256(blob).hexdigest()
        if actual_hash != manifest["binary_sha256"]:
            raise ValueError("Q6 binary SHA-256 does not match manifest")
        magic, version, count, bits, scale_type = HEADER.unpack_from(blob)
        if (magic, version, count, bits, scale_type) != (b"TLQ6ROW\0", 1, manifest["tensor_count"], 6, 1):
            raise ValueError("Unsupported Q6 binary header")

        self.weights = {}
        self.row_scales = {}
        self.packed_weights = {}
        self.shapes = {}
        for entry in manifest["tensors"]:
            name = entry["name"]
            shape = tuple(int(size) for size in entry["shape"])
            if entry["dtype"] == "signed_int6_twos_complement":
                rows, columns = shape
                self.shapes[name] = shape
                if rows * columns * 6 // 8 != entry["data_bytes"] or rows * 4 != entry["scale_bytes"]:
                    raise ValueError("Invalid Q6 tensor lengths for {}".format(name))
                packed = memoryview(blob)[entry["data_offset"] : entry["data_offset"] + entry["data_bytes"]]
                scales = np.frombuffer(
                    blob, dtype="<f4", count=rows, offset=entry["scale_offset"]
                ).copy()
                self.row_scales[name] = scales
                self.packed_weights[name] = packed
                selected_pl_matrix = name.endswith(
                    (".feed_forward.w1.weight", ".feed_forward.w3.weight")
                )
                attention_pl_matrix = (
                    linear_mode == "cached_qkv"
                    and name.endswith((".attention.wq.weight", ".attention.wk.weight",
                                       ".attention.wv.weight"))
                )
                accelerated_matrix = (
                    linear_mode in ("selected", "cached", "cached_qkv", "cached_vocab") and selected_pl_matrix
                ) or attention_pl_matrix or (
                    linear_mode == "vocab_bram" and name == "tok_embeddings.weight"
                )
                needs_arm_matrix = linear_mode not in ("all", "batched") and not accelerated_matrix
                needs_float_matrix = needs_arm_matrix or name == "tok_embeddings.weight"
                if needs_float_matrix:
                    quantized = unpack_signed_int6(packed, rows, columns)
                    self.weights[name] = quantized.astype(np.float32) * scales[:, None]
            elif entry["dtype"] == "little_endian_float32":
                values = np.frombuffer(
                    blob, dtype="<f4", count=int(np.prod(shape)), offset=entry["data_offset"]
                ).copy()
                self.weights[name] = values.reshape(shape)
            else:
                raise ValueError("Unsupported dtype {}".format(entry["dtype"]))

        self.weights["output.weight"] = self.weights["tok_embeddings.weight"]
        self.args = manifest["model_args"]
        if (self.args["dim"], self.args["n_layers"], self.args["n_heads"], self.args["n_kv_heads"], self.args["vocab_size"]) != (128, 5, 4, 2, 4096):
            raise ValueError("Q6 model dimensions do not match this board runtime")


class PLGEMV:
    def __init__(self, package_dir, dynamic=False, batch_mode=False, cache_mode=False):
        start = time.perf_counter()
        self.overlay = Overlay(str(package_dir / "q6w_gemv.bit"))
        self.dma = self.overlay.axi_dma_0
        self.dma_clock_mhz = float(Clocks.fclk0_mhz)
        core_clock = getattr(Clocks, "fclk1_mhz", None)
        self.fpga_clock_mhz = float(core_clock) if core_clock else self.dma_clock_mhz
        self.dynamic = dynamic
        self.batch_mode = batch_mode
        self.cache_mode = cache_mode
        self.batch_templates = {}
        self.cached_templates = {}
        self.weight_cache_load_seconds = 0.0
        self.weight_cache_bytes = 0
        self.cached_matrix_count = 0
        if dynamic:
            self.sources = {}
            self.destinations = {}
        else:
            self.source = allocate(shape=(INPUT_BYTES,), dtype=np.uint8)
            self.destination = allocate(shape=(ROWS,), dtype=np.int32)
        self.calls = 0
        self.seconds = 0.0
        self.profile = TimingProfile()
        self.load_seconds = time.perf_counter() - start

    def _dynamic_buffer(self, pool, count, dtype):
        buffer = pool.get(count)
        if buffer is None:
            buffer = allocate(shape=(count,), dtype=dtype)
            pool[count] = buffer
        return buffer

    def _transfer(self, source, destination):
        if source.nbytes > MAX_DMA_TRANSFER_BYTES or destination.nbytes > MAX_DMA_TRANSFER_BYTES:
            raise ValueError("DMA transfer exceeds the configured 16-bit BTT limit")
        start = time.perf_counter()
        self.dma.recvchannel.transfer(destination)
        self.dma.sendchannel.transfer(source)
        self.dma.sendchannel.wait()
        self.dma.recvchannel.wait()
        elapsed = time.perf_counter() - start
        self.calls += 1
        self.seconds += elapsed
        self.profile.add("dma_transfer_and_wait", elapsed)
        return np.asarray(destination, dtype=np.int32), elapsed

    def run(self, activation, packed_weights, row_scales):
        activation = np.asarray(activation, dtype=np.float32)
        peak = float(np.max(np.abs(activation)))
        activation_scale = peak / 127.0 if peak else 1.0
        activation_q = np.clip(np.rint(activation / activation_scale), -127, 127).astype(np.int8)
        self.source[:COLUMNS] = activation_q.view(np.uint8)
        self.source[COLUMNS:] = np.frombuffer(packed_weights, dtype=np.uint8)

        accumulators, _elapsed = self._transfer(self.source, self.destination)
        return accumulators.astype(np.float32) * row_scales * np.float32(activation_scale)

    def run_matrix(self, q6, name, activation):
        if not self.dynamic:
            raise RuntimeError("The currently loaded overlay does not accept dynamic GEMV packets")
        if self.batch_mode:
            return self.run_batch(q6, [(name, activation)])[0]
        storage_name = "tok_embeddings.weight" if name == "output.weight" else name
        rows, columns = q6.shapes[storage_name]
        row_scales = q6.row_scales[storage_name]
        packed_weights = q6.packed_weights[storage_name]
        if columns > 352 or columns % 4 or rows < 1:
            raise ValueError("Unsupported Q6 matrix shape {} for {}".format((rows, columns), name))
        activation = np.asarray(activation, dtype=np.float32).reshape(-1)
        if activation.size != columns:
            raise ValueError("Activation width does not match {}".format(name))
        peak = float(np.max(np.abs(activation)))
        activation_scale = peak / 127.0 if peak else 1.0
        activation_q = np.clip(np.rint(activation / activation_scale), -127, 127).astype(np.int8)
        row_bytes = columns * 6 // 8
        result = np.empty(rows, dtype=np.float32)

        for row_start in range(0, rows, ROWS):
            row_end = min(row_start + ROWS, rows)
            tile_rows = row_end - row_start
            weights_start = row_start * row_bytes
            weights_end = row_end * row_bytes
            weights_tile = packed_weights[weights_start:weights_end]
            input_count = LINEAR_HEADER.size + columns + len(weights_tile)
            source = self._dynamic_buffer(self.sources, input_count, np.uint8)
            destination = self._dynamic_buffer(self.destinations, tile_rows, np.int32)
            source[:LINEAR_HEADER.size] = np.frombuffer(
                LINEAR_HEADER.pack(columns, tile_rows), dtype=np.uint8
            )
            source[LINEAR_HEADER.size : LINEAR_HEADER.size + columns] = activation_q.view(np.uint8)
            source[LINEAR_HEADER.size + columns :] = np.frombuffer(weights_tile, dtype=np.uint8)
            accumulators, _elapsed = self._transfer(source, destination)
            result[row_start:row_end] = (
                accumulators.astype(np.float32)
                * row_scales[row_start:row_end]
                * np.float32(activation_scale)
            )
        return result

    def run_batch(self, q6, matrices):
        if not self.batch_mode:
            raise RuntimeError("The currently loaded overlay does not accept batched GEMV packets")
        if not matrices or len(matrices) > 16:
            raise ValueError("A GEMV batch must contain between 1 and 16 matrices")

        key = tuple(name for name, _activation in matrices)
        template = self.batch_templates.get(key)
        if template is None:
            packet = bytearray(struct.pack("<H", len(matrices)))
            metadata = []
            total_rows = 0
            for name, _activation in matrices:
                storage_name = "tok_embeddings.weight" if name == "output.weight" else name
                if storage_name not in q6.packed_weights:
                    raise ValueError("No packed Q6 tensor for {}".format(name))
                rows, columns = q6.shapes[storage_name]
                if columns > 352 or columns % 4 or rows < 1 or rows > 4096:
                    raise ValueError("Unsupported Q6 matrix shape {} for {}".format((rows, columns), name))
                packet.extend(LINEAR_HEADER.pack(columns, rows))
                activation_offset = len(packet)
                packet.extend(bytes(columns * 2))
                packet.extend(q6.packed_weights[storage_name])
                metadata.append({
                    "name": name,
                    "columns": columns,
                    "rows": rows,
                    "row_scales": q6.row_scales[storage_name],
                    "activation_offset": activation_offset,
                    "activation_scale": np.float32(1.0),
                })
                total_rows += rows

            if len(packet) > MAX_DMA_TRANSFER_BYTES:
                raise ValueError("Batched GEMV packet exceeds the configured AXI DMA transfer limit")
            source = allocate(shape=(len(packet),), dtype=np.uint8)
            source[:] = np.frombuffer(packet, dtype=np.uint8)
            template = {"source": source, "metadata": metadata, "total_rows": total_rows}
            self.batch_templates[key] = template

        source = template["source"]
        metadata = template["metadata"]
        if len(metadata) != len(matrices):
            raise ValueError("GEMV matrix template does not match this batch")
        for (name, activation), entry in zip(matrices, metadata):
            if name != entry["name"]:
                raise ValueError("GEMV matrix order differs from its cached packet template")
            activation = np.asarray(activation, dtype=np.float32).reshape(-1)
            if activation.size != entry["columns"]:
                raise ValueError("Activation width does not match {}".format(name))
            peak = float(np.max(np.abs(activation)))
            activation_scale = peak / 32767.0 if peak else 1.0
            activation_q = np.clip(np.rint(activation / activation_scale), -32767, 32767).astype("<i2")
            start = entry["activation_offset"]
            source[start : start + entry["columns"] * 2] = activation_q.view(np.uint8)
            entry["activation_scale"] = np.float32(activation_scale)

        destination = self._dynamic_buffer(self.destinations, template["total_rows"], np.int32)
        accumulators, _elapsed = self._transfer(source, destination)

        results = []
        row_start = 0
        for entry in metadata:
            rows = entry["rows"]
            row_end = row_start + rows
            results.append(
                accumulators[row_start:row_end].astype(np.float32)
                * entry["row_scales"]
                * entry["activation_scale"]
            )
            row_start = row_end
        return results

    @staticmethod
    def _cache_slot(name):
        parts = name.split(".")
        if len(parts) != 5 or parts[0] != "layers" or parts[2] != "feed_forward" or parts[4] != "weight":
            raise ValueError("Only transformer FFN matrices can use the BRAM cache")
        layer = int(parts[1])
        if not 0 <= layer < 5:
            raise ValueError("FFN layer is outside the BRAM cache")
        if parts[3] == "w1":
            return layer * 2
        if parts[3] == "w3":
            return layer * 2 + 1
        raise ValueError("Only W1/W3 matrices can use the BRAM cache")

    def load_weight_cache(self, q6):
        if not self.cache_mode:
            raise RuntimeError("The loaded overlay does not have the selected weight cache")
        entries = []
        if q6.linear_mode == "vocab_bram":
            name = "tok_embeddings.weight"
            if q6.shapes[name] != (4096, 128):
                raise ValueError("Unexpected cached Q6 vocabulary shape")
            entries.append((0, name, q6.packed_weights[name]))
        else:
            for layer in range(5):
                prefix = "layers.{}.feed_forward.".format(layer)
                for suffix in ("w1", "w3"):
                    name = prefix + suffix + ".weight"
                    if q6.shapes[name] != (352, 128):
                        raise ValueError("Unexpected cached Q6 matrix shape for {}".format(name))
                    entries.append((self._cache_slot(name), name, q6.packed_weights[name]))

        if q6.linear_mode == "cached_qkv":
            self.cached_qkv_scales = {}
            for layer in range(5):
                prefix = "layers.{}.attention.".format(layer)
                qkv_names = [prefix + name + ".weight" for name in ("wq", "wk", "wv")]
                expected_shapes = ((128, 128), (64, 128), (64, 128))
                for name, expected in zip(qkv_names, expected_shapes):
                    if q6.shapes[name] != expected:
                        raise ValueError("Unexpected cached Q6 matrix shape for {}".format(name))
                qkv_packed = b"".join(bytes(q6.packed_weights[name]) for name in qkv_names)
                entries.append((10 + layer, "layers.{}.attention.qkv.weight".format(layer), qkv_packed))
                self.cached_qkv_scales[layer] = np.concatenate(
                    [q6.row_scales[name] for name in qkv_names]
                )

        start = time.perf_counter()
        total_bytes = 0
        for slot, _name, packed in entries:
            packet = bytearray(struct.pack("<H", 1))
            packet.extend(LINEAR_HEADER.pack(0, slot))
            packet.extend(packed)
            if len(packet) > MAX_DMA_TRANSFER_BYTES:
                raise ValueError("Cached matrix exceeds the configured AXI DMA transfer limit")
            source = allocate(shape=(len(packet),), dtype=np.uint8)
            try:
                source[:] = np.frombuffer(packet, dtype=np.uint8)
                self.dma.sendchannel.transfer(source)
                self.dma.sendchannel.wait()
            finally:
                source.freebuffer()
            total_bytes += len(packet)
        self.weight_cache_load_seconds = time.perf_counter() - start
        self.weight_cache_bytes = total_bytes
        self.cached_matrix_count = len(entries)

    def run_cached_batch(self, q6, matrices):
        if not self.cache_mode:
            raise RuntimeError("The loaded overlay does not have the selected weight cache")
        if not matrices or len(matrices) > 16:
            raise ValueError("A cached GEMV batch must contain between 1 and 16 matrices")

        operations = []
        for name, activation in matrices:
            rows, columns = q6.shapes[name]
            operations.append((name, self._cache_slot(name), rows, columns,
                               q6.row_scales[name], activation))
        return self.run_cached_operations(operations)

    def run_vocab_cache(self, q6, activation):
        name = "tok_embeddings.weight"
        rows, columns = q6.shapes[name]
        if (rows, columns) != (4096, 128):
            raise ValueError("Unexpected Q6 vocabulary shape")
        return self.run_cached_operations([
            ("output.weight", 0, rows, columns, q6.row_scales[name], activation)
        ])[0]

    def run_cached_operations(self, operations):
        reuse_flags = tuple(
            index > 0 and operations[index][5] is operations[index - 1][5]
            for index in range(len(operations))
        )
        key = (
            tuple((name, slot, rows, columns) for name, slot, rows, columns, _scales, _activation in operations),
            reuse_flags,
        )
        template = self.cached_templates.get(key)
        if template is None:
            packet = bytearray(struct.pack("<H", len(operations)))
            metadata = []
            total_rows = 0
            for index, (name, slot, rows, columns, row_scales, _activation) in enumerate(operations):
                if columns != 128 or rows < 1:
                    raise ValueError("Unsupported cached matrix shape for {}".format(name))
                reuse_activation = reuse_flags[index]
                packet.extend(LINEAR_HEADER.pack(0xFFFE if reuse_activation else 0xFFFF, slot))
                if reuse_activation:
                    activation_offset = None
                else:
                    activation_offset = len(packet)
                    packet.extend(bytes(columns * 2))
                metadata.append({
                    "name": name,
                    "slot": slot,
                    "columns": columns,
                    "rows": rows,
                    "row_scales": row_scales,
                    "activation_offset": activation_offset,
                    "reuse_activation": reuse_activation,
                    "activation_scale": np.float32(1.0),
                })
                total_rows += rows

            source = allocate(shape=(len(packet),), dtype=np.uint8)
            source[:] = np.frombuffer(packet, dtype=np.uint8)
            template = {"source": source, "metadata": metadata, "total_rows": total_rows}
            self.cached_templates[key] = template

        source = template["source"]
        metadata = template["metadata"]
        quantize_start = time.perf_counter()
        quantized_activations = {}
        for operation, entry in zip(operations, metadata):
            name, _slot, _rows, _columns, _scales, activation = operation
            if (name, operation[1], operation[2], operation[3]) != (
                entry["name"], entry["slot"], entry["rows"], entry["columns"]
            ):
                raise ValueError("Cached matrix order differs from its packet template")
            activation_key = id(activation)
            activation = np.asarray(activation, dtype=np.float32).reshape(-1)
            if activation.size != entry["columns"]:
                raise ValueError("Activation width does not match {}".format(name))
            quantized = quantized_activations.get(activation_key)
            if entry["reuse_activation"]:
                if quantized is None:
                    raise ValueError("Reused activation was not quantized by the preceding operation")
                activation_scale = quantized[1]
            elif quantized is None:
                start = entry["activation_offset"]
                end = start + entry["columns"] * 2
                kernel_start = time.perf_counter()
                peak = float(np.max(np.abs(activation)))
                activation_scale = peak / 32767.0 if peak else 1.0
                activation_q = np.clip(
                    np.rint(activation / activation_scale), -32767, 32767
                ).astype(np.int16)
                self.profile.add("activation_quantizer_kernel", time.perf_counter() - kernel_start)
                packet_start = time.perf_counter()
                source[start:end] = activation_q.view(np.uint8)
                self.profile.add("activation_packet_copy", time.perf_counter() - packet_start)
                quantized = (activation_q, np.float32(activation_scale))
                quantized_activations[activation_key] = quantized
            else:
                activation_q, activation_scale = quantized
                start = entry["activation_offset"]
                end = start + entry["columns"] * 2
                packet_start = time.perf_counter()
                source[start:end] = activation_q.view(np.uint8)
                self.profile.add("activation_packet_copy", time.perf_counter() - packet_start)
            entry["activation_scale"] = activation_scale
        self.profile.add("activation_quantize_and_packetize", time.perf_counter() - quantize_start)

        destination = self._dynamic_buffer(self.destinations, template["total_rows"], np.int32)
        accumulators, _elapsed = self._transfer(source, destination)
        scale_start = time.perf_counter()
        results = []
        row_start = 0
        for entry in metadata:
            row_end = row_start + entry["rows"]
            results.append(
                accumulators[row_start:row_end].astype(np.float32)
                * entry["row_scales"]
                * entry["activation_scale"]
            )
            row_start = row_end
        self.profile.add("scale_reconstruction", time.perf_counter() - scale_start)
        return results

    def run_cached_qkv(self, q6, names, activation):
        layers = {int(name.split(".")[1]) for name in names}
        if len(layers) != 1 or len(names) != 3:
            raise ValueError("A cached QKV operation must contain one layer's Wq/Wk/Wv")
        layer = layers.pop()
        order = ("wq", "wk", "wv")
        if [name.split(".")[3] for name in names] != list(order):
            raise ValueError("Cached QKV matrices must be ordered Wq, Wk, Wv")
        shapes = [q6.shapes[name] for name in names]
        if shapes != [(128, 128), (64, 128), (64, 128)]:
            raise ValueError("Unexpected QKV shapes for cached PL GEMV")
        combined = self.run_cached_operations([
            (
                "layers.{}.attention.qkv.weight".format(layer), 10 + layer,
                256, 128, self.cached_qkv_scales[layer], activation,
            )
        ])[0]
        return [combined[:128], combined[128:192], combined[192:]]

    def close(self):
        if self.dynamic:
            for buffer in self.sources.values():
                buffer.freebuffer()
            for buffer in self.destinations.values():
                buffer.freebuffer()
            for template in self.batch_templates.values():
                template["source"].freebuffer()
            for template in self.cached_templates.values():
                template["source"].freebuffer()
        else:
            self.source.freebuffer()
            self.destination.freebuffer()


class NeonSGEMV:
    def __init__(self, library_path, paired_rows=False, four_rows=False):
        self.library = ctypes.CDLL(str(library_path))
        self.kernel = (
            self.library.arm_neon_sgemv_f32_fourrows
            if four_rows else self.library.arm_neon_sgemv_f32_pairrows
            if paired_rows else self.library.arm_neon_sgemv_f32
        )
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.kernel.argtypes = [float_pointer, float_pointer, float_pointer,
                                ctypes.c_int, ctypes.c_int]
        self.kernel.restype = None

    def __call__(self, matrix, vector):
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        vector = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[1] != vector.size:
            raise ValueError("NEON SGEMV matrix and vector shapes do not match")
        output = np.empty(matrix.shape[0], dtype=np.float32)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.kernel(
            matrix.ctypes.data_as(float_pointer),
            vector.ctypes.data_as(float_pointer),
            output.ctypes.data_as(float_pointer),
            matrix.shape[0],
            matrix.shape[1],
        )
        return output


class NeonQ6VocabSGEMV:
    def __init__(self, library_path, packed_weights, row_scales, shape, reference):
        self.library = ctypes.CDLL(str(library_path))
        self.kernel = self.library.arm_neon_q6_vocab_sgemv_f32
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.kernel.argtypes = [
            ctypes.POINTER(ctypes.c_int8), float_pointer,
            ctypes.POINTER(ctypes.c_int16), ctypes.c_float, float_pointer,
            ctypes.c_int, ctypes.c_int,
        ]
        self.kernel.restype = None
        self.rows, self.columns = shape
        weights = unpack_signed_int6(packed_weights, self.rows, self.columns)
        self.weights = np.ascontiguousarray(weights)
        self.row_scales = np.ascontiguousarray(row_scales, dtype=np.float32)
        self._self_test(reference)

    def _quantize_activation(self, vector):
        max_abs = np.max(np.abs(vector)) if vector.size else np.float32(0.0)
        scale = np.float32(max_abs / np.float32(32767.0)) if max_abs else np.float32(1.0)
        quantized = np.clip(np.rint(vector / scale), -32767, 32767).astype(np.int16)
        return np.ascontiguousarray(quantized), scale

    def _run(self, vector):
        quantized, activation_scale = self._quantize_activation(vector)
        output = np.empty(self.rows, dtype=np.float32)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.kernel(
            self.weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            self.row_scales.ctypes.data_as(float_pointer),
            quantized.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_float(float(activation_scale)),
            output.ctypes.data_as(float_pointer), self.rows, self.columns,
        )
        return output

    def _self_test(self, reference):
        vector = np.linspace(-1.25, 0.875, self.columns, dtype=np.float32)
        expected = np.asarray(np.dot(reference, vector), dtype=np.float32)
        actual = self._run(vector)
        if not np.allclose(actual, expected, rtol=2e-4, atol=2e-4):
            raise RuntimeError("Q6 int8 vocabulary GEMV self-test failed")

    def __call__(self, matrix, vector):
        vector = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
        if matrix.shape != (self.rows, self.columns) or vector.size != self.columns:
            raise ValueError("Q6 vocabulary GEMV matrix and vector shapes do not match")
        return self._run(vector)


class NeonGQAAttention:
    def __init__(self, library_path):
        self.library = ctypes.CDLL(str(library_path))
        self.kernel = self.library.arm_neon_gqa_attention_f32
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.kernel.argtypes = [
            float_pointer, float_pointer, float_pointer, float_pointer,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.kernel.restype = None

    def __call__(self, queries, keys, values, groups_per_kv):
        queries = np.ascontiguousarray(queries, dtype=np.float32)
        keys = np.ascontiguousarray(keys, dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        if keys.ndim != 3 or values.shape != keys.shape:
            raise ValueError("NEON attention expects matching [sequence, kv_heads, head_dim] caches")
        sequence, kv_heads, head_dim = keys.shape
        if queries.shape != (kv_heads * groups_per_kv, head_dim):
            raise ValueError("NEON attention query shape does not match the KV cache")
        output = np.empty_like(queries)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.kernel(
            queries.ctypes.data_as(float_pointer),
            keys.ctypes.data_as(float_pointer),
            values.ctypes.data_as(float_pointer),
            output.ctypes.data_as(float_pointer),
            sequence, kv_heads, groups_per_kv, head_dim,
        )
        return output


class NeonRMSNorm:
    def __init__(self, library_path):
        self.library = ctypes.CDLL(str(library_path))
        self.kernel = self.library.arm_neon_rmsnorm_f32
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.kernel.argtypes = [float_pointer, float_pointer, float_pointer,
                                ctypes.c_int, ctypes.c_float]
        self.kernel.restype = None
        self._self_test()

    def _self_test(self):
        vector = np.linspace(-1.25, 0.875, 128, dtype=np.float32)
        weight = np.linspace(0.5, 1.5, 128, dtype=np.float32)
        epsilon = 1e-5
        mean_square = np.mean(vector * vector, dtype=np.float32)
        inv_rms = np.float32(1.0 / np.sqrt(np.float32(mean_square + np.float32(epsilon))))
        expected = np.asarray(vector * inv_rms * weight, dtype=np.float32)
        actual = self(vector, weight, epsilon)
        if not np.allclose(actual, expected, rtol=2e-6, atol=2e-6):
            raise RuntimeError("NEON RMSNorm self-test failed")

    def __call__(self, vector, weight, epsilon):
        vector = np.asarray(vector, dtype=np.float32)
        weight = np.asarray(weight, dtype=np.float32)
        if vector.ndim != 1 or not vector.flags.c_contiguous:
            vector = np.ascontiguousarray(vector).reshape(-1)
        if weight.ndim != 1 or weight.size != vector.size:
            raise ValueError("NEON RMSNorm vector and weight shapes do not match")
        weight = np.ascontiguousarray(weight)
        output = np.empty(vector.size, dtype=np.float32)
        float_pointer = ctypes.POINTER(ctypes.c_float)
        self.kernel(
            vector.ctypes.data_as(float_pointer),
            weight.ctypes.data_as(float_pointer),
            output.ctypes.data_as(float_pointer),
            vector.size,
            ctypes.c_float(epsilon),
        )
        return output

class NoPLGEMV:
    def __init__(self):
        self.profile = TimingProfile()
        self.calls = 0
        self.seconds = 0.0
        self.load_seconds = 0.0
        self.weight_cache_load_seconds = 0.0
        self.weight_cache_bytes = 0
        self.cached_matrix_count = 0
        self.fpga_clock_mhz = None
        self.dma_clock_mhz = None

    @staticmethod
    def close():
        return None


class HybridTransformer:
    def __init__(self, q6, gemv, linear_mode="ffn", neon_vocab=None,
                 neon_attention=None, neon_rmsnorm=None, fuse_projections=False):
        self.q6 = q6
        self.gemv = gemv
        self.linear_mode = linear_mode
        self.neon_vocab = neon_vocab
        self.neon_attention = neon_attention
        self.neon_rmsnorm = neon_rmsnorm
        self.fuse_projections = fuse_projections
        self.profile = TimingProfile()
        self.args = q6.args
        self.dim = int(self.args["dim"])
        self.head_dim = self.dim // int(self.args["n_heads"])
        self.layers = int(self.args["n_layers"])
        self.kv_heads = int(self.args["n_kv_heads"])
        self.n_heads = int(self.args["n_heads"])
        self.max_seq_len = int(self.args["max_seq_len"])
        self.norm_eps = float(self.args.get("norm_eps", 1e-5))
        self.freqs = np.float32(1.0) / np.power(
            np.float32(10000.0), np.arange(0, self.head_dim, 2, dtype=np.float32) / np.float32(self.head_dim)
        )
        positions = np.arange(self.max_seq_len, dtype=np.float32)[:, None]
        angles = positions * self.freqs[None, :]
        self.rope_cos = np.cos(angles).astype(np.float32)
        self.rope_sin = np.sin(angles).astype(np.float32)
        self.fused_qkv_weights = {}
        self.fused_ffn_weights = {}
        if fuse_projections:
            if linear_mode != "ps":
                raise ValueError("Fused CPU projections are supported only in PS mode")
            for layer_id in range(self.layers):
                prefix = "layers.{}.".format(layer_id)
                attention = prefix + "attention."
                feed_forward = prefix + "feed_forward."
                qkv_names = [attention + "w{}.weight".format(name) for name in ("q", "k", "v")]
                ffn_names = [feed_forward + name + ".weight" for name in ("w1", "w3")]
                self.fused_qkv_weights[layer_id] = np.ascontiguousarray(
                    np.concatenate([q6.weights[name] for name in qkv_names], axis=0)
                )
                self.fused_ffn_weights[layer_id] = np.ascontiguousarray(
                    np.concatenate([q6.weights[name] for name in ffn_names], axis=0)
                )

    def rms_norm(self, vector, weight):
        if self.neon_rmsnorm is not None:
            return self.neon_rmsnorm(vector, weight, self.norm_eps)
        mean_square = np.mean(vector * vector, dtype=np.float32)
        inv_rms = np.float32(1.0 / np.sqrt(np.float32(mean_square + np.float32(self.norm_eps))))
        return np.asarray(vector * inv_rms * weight, dtype=np.float32)

    def rotate(self, vector, position):
        cos = self.rope_cos[position]
        sin = self.rope_sin[position]
        pairs = vector.reshape(vector.shape[0], self.head_dim // 2, 2)
        result = np.empty_like(pairs)
        result[:, :, 0] = pairs[:, :, 0] * cos - pairs[:, :, 1] * sin
        result[:, :, 1] = pairs[:, :, 0] * sin + pairs[:, :, 1] * cos
        return result.reshape(vector.shape).astype(np.float32, copy=False)

    def matvec(self, name, vector):
        start = time.perf_counter()
        if self.linear_mode == "all":
            result = self.gemv.run_matrix(self.q6, name, vector)
            location = "PL"
        elif self.linear_mode == "batched":
            result = self.gemv.run_batch(self.q6, [(name, vector)])[0]
            location = "PL"
        elif self.linear_mode in ("cached", "cached_qkv", "cached_vocab") and self.is_ffn_matrix(name):
            result = self.gemv.run_cached_batch(self.q6, [(name, vector)])[0]
            location = "PL"
        elif self.linear_mode == "vocab_bram" and name == "output.weight":
            result = self.gemv.run_vocab_cache(self.q6, vector)
            location = "PL_BRAM"
        elif self.linear_mode == "cached_vocab" and name == "output.weight":
            result = self.gemv.run_batch(self.q6, [(name, vector)])[0]
            location = "PL"
        elif self.linear_mode == "selected" and self.is_pl_matrix(name):
            result = self.gemv.run_matrix(self.q6, name, vector)
            location = "PL"
        elif self.neon_vocab is not None and name == "output.weight":
            result = self.neon_vocab(self.q6.weights[name], vector)
            location = "ARM_NEON"
        else:
            result = np.asarray(np.dot(self.q6.weights[name], vector), dtype=np.float32)
            location = "ARM"
        self.profile.add("linear_{}.{}".format(location, name), time.perf_counter() - start)
        return result

    @staticmethod
    def is_pl_matrix(name):
        return name.endswith((".feed_forward.w1.weight", ".feed_forward.w3.weight"))

    @staticmethod
    def is_ffn_matrix(name):
        return name.endswith((".feed_forward.w1.weight", ".feed_forward.w3.weight"))

    def matvec_group(self, names, vector):
        start = time.perf_counter()
        if self.fuse_projections and self.linear_mode == "ps":
            first, second = names[0], names[1] if len(names) > 1 else ""
            if len(names) == 3 and all(
                ".attention.w{}.weight".format(index) in name
                for name, index in zip(names, ("q", "k", "v"))
            ):
                layer_id = int(first.split(".")[1])
                matrix = self.fused_qkv_weights[layer_id]
                joined = np.asarray(np.dot(matrix, vector), dtype=np.float32)
                rows = [self.q6.weights[name].shape[0] for name in names]
                result = [joined[:rows[0]], joined[rows[0]:rows[0] + rows[1]], joined[rows[0] + rows[1]:]]
                self.profile.add("linear_ARM_fused_qkv", time.perf_counter() - start)
                return result
            if len(names) == 2 and all(self.is_ffn_matrix(name) for name in names):
                layer_id = int(first.split(".")[1])
                matrix = self.fused_ffn_weights[layer_id]
                joined = np.asarray(np.dot(matrix, vector), dtype=np.float32)
                rows = self.q6.weights[first].shape[0]
                result = [joined[:rows], joined[rows:]]
                self.profile.add("linear_ARM_fused_ffn", time.perf_counter() - start)
                return result
        if self.linear_mode == "batched":
            result = self.gemv.run_batch(self.q6, [(name, vector) for name in names])
            self.profile.add("linear_PL_group." + "+".join(names), time.perf_counter() - start)
            return result
        if self.linear_mode == "cached_qkv" and len(names) == 3 and all(
            ".attention.w{}.weight".format(index) in name for name, index in zip(names, ("q", "k", "v"))
        ):
            result = self.gemv.run_cached_qkv(self.q6, names, vector)
            self.profile.add("linear_PL_group." + "+".join(names), time.perf_counter() - start)
            return result
        if self.linear_mode in ("selected", "cached", "cached_qkv", "cached_vocab") and all(self.is_ffn_matrix(name) for name in names):
            if self.linear_mode in ("cached", "cached_qkv", "cached_vocab"):
                result = self.gemv.run_cached_batch(self.q6, [(name, vector) for name in names])
            else:
                result = self.gemv.run_batch(self.q6, [(name, vector) for name in names])
            self.profile.add("linear_PL_group." + "+".join(names), time.perf_counter() - start)
            return result
        return [self.matvec(name, vector) for name in names]

    def gemv_ffn(self, name, vector):
        if self.linear_mode in ("all", "selected", "vocab_bram"):
            return self.matvec(name, vector)
        if self.linear_mode == "ps":
            return self.matvec(name, vector)
        return self.gemv.run(vector, self.q6.packed_weights[name], self.q6.row_scales[name])

    def new_cache(self):
        shape = (self.layers, self.max_seq_len, self.kv_heads, self.head_dim)
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)

    def step(self, token_id, position, key_cache, value_cache):
        step_start = time.perf_counter()
        if not 0 <= position < self.max_seq_len:
            raise ValueError("Position exceeds model context")
        stage_start = time.perf_counter()
        x = self.q6.weights["tok_embeddings.weight"][token_id].copy()
        self.profile.add("embedding_lookup", time.perf_counter() - stage_start)
        for layer_id in range(self.layers):
            prefix = "layers.{}.".format(layer_id)
            att_prefix = prefix + "attention."
            stage_start = time.perf_counter()
            normalized = self.rms_norm(x, self.q6.weights[prefix + "attention_norm.weight"])
            self.profile.add("attention_rms_norm", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            q, k, v = self.matvec_group(
                [att_prefix + "wq.weight", att_prefix + "wk.weight", att_prefix + "wv.weight"],
                normalized,
            )
            self.profile.add("attention_qkv_projection", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            q = q.reshape(self.n_heads, self.head_dim)
            k = k.reshape(self.kv_heads, self.head_dim)
            v = v.reshape(self.kv_heads, self.head_dim)
            q = self.rotate(q, position)
            k = self.rotate(k, position)
            key_cache[layer_id, position] = k
            value_cache[layer_id, position] = v
            self.profile.add("rope_and_kv_write", time.perf_counter() - stage_start)

            stage_start = time.perf_counter()
            sequence = position + 1
            groups_per_kv = self.n_heads // self.kv_heads
            if self.neon_attention is not None:
                attended = self.neon_attention(
                    q, key_cache[layer_id, :sequence], value_cache[layer_id, :sequence],
                    groups_per_kv,
                )
            else:
                keys = key_cache[layer_id, :sequence].transpose(1, 0, 2)
                values = value_cache[layer_id, :sequence].transpose(1, 0, 2)
                queries = q.reshape(self.kv_heads, groups_per_kv, self.head_dim).transpose(0, 2, 1)
                scores = np.matmul(keys, queries).astype(np.float32, copy=False)
                scores /= np.float32(math.sqrt(self.head_dim))
                scores -= np.max(scores, axis=1, keepdims=True)
                probabilities = np.exp(scores).astype(np.float32, copy=False)
                probabilities /= np.sum(probabilities, axis=1, keepdims=True, dtype=np.float32)
                attended = np.matmul(probabilities.transpose(0, 2, 1), values)
                attended = attended.reshape(self.n_heads, self.head_dim).astype(np.float32, copy=False)
            self.profile.add("attention_scores_softmax_value", time.perf_counter() - stage_start)

            attended = attended.reshape(self.dim)
            stage_start = time.perf_counter()
            attention_update = self.matvec(att_prefix + "wo.weight", attended)
            x = np.asarray(x + attention_update, dtype=np.float32)
            self.profile.add("attention_output_and_residual", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            normalized = self.rms_norm(x, self.q6.weights[prefix + "ffn_norm.weight"])
            self.profile.add("ffn_rms_norm", time.perf_counter() - stage_start)
            ffn_prefix = prefix + "feed_forward."
            stage_start = time.perf_counter()
            if (self.linear_mode in ("batched", "selected", "cached", "cached_qkv", "cached_vocab")
                    or self.fuse_projections):
                gate, up = self.matvec_group(
                    [ffn_prefix + "w1.weight", ffn_prefix + "w3.weight"], normalized
                )
            else:
                gate = self.gemv_ffn(ffn_prefix + "w1.weight", normalized)
                up = self.gemv_ffn(ffn_prefix + "w3.weight", normalized)
            self.profile.add("ffn_w1_w3_projection", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            silu = gate / (np.float32(1.0) + np.exp(-gate))
            hidden = np.asarray(silu * up, dtype=np.float32)
            self.profile.add("ffn_swiglu", time.perf_counter() - stage_start)
            stage_start = time.perf_counter()
            x = np.asarray(x + self.matvec(ffn_prefix + "w2.weight", hidden), dtype=np.float32)
            self.profile.add("ffn_w2_and_residual", time.perf_counter() - stage_start)

        stage_start = time.perf_counter()
        normalized = self.rms_norm(x, self.q6.weights["norm.weight"])
        self.profile.add("final_rms_norm", time.perf_counter() - stage_start)
        stage_start = time.perf_counter()
        logits = self.matvec("output.weight", normalized)
        self.profile.add("vocabulary_projection", time.perf_counter() - stage_start)
        self.profile.add("model_step_total", time.perf_counter() - step_start)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Run the Q6 TinyLLM decode loop on PS and/or PL")
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path)
    vocab_group = parser.add_mutually_exclusive_group()
    vocab_group.add_argument("--neon-vocab", action="store_true")
    vocab_group.add_argument("--neon-vocab-pairrows", action="store_true")
    vocab_group.add_argument("--neon-vocab-fourrows", action="store_true")
    vocab_group.add_argument("--neon-vocab-q6-int8", action="store_true")
    parser.add_argument("--neon-attention", action="store_true")
    parser.add_argument("--neon-rmsnorm", action="store_true")
    parser.add_argument("--fuse-projections", action="store_true")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0:
        raise ValueError("runs must be positive and warmup must be non-negative")

    package_dir = args.app_dir.resolve()
    request = json.loads((package_dir / "request.json").read_text(encoding="utf-8"))
    reference_logits = np.frombuffer(
        (package_dir / "reference_logits.bin").read_bytes(), dtype="<f4"
    ).copy().reshape(int(request["reference_step_count"]), int(request["model"]["vocab_size"]))
    reference_sampled = [int(value) for value in request["reference_sampled_ids"]]

    q6_start = time.perf_counter()
    linear_mode = request.get("linear_mode", "ffn")
    if linear_mode not in (
        "ps", "ffn", "all", "batched", "selected", "cached", "cached_qkv",
        "cached_vocab", "vocab_bram",
    ):
        raise ValueError("request linear_mode is unsupported")
    neon_vocab_requested = (
        args.neon_vocab or args.neon_vocab_pairrows or args.neon_vocab_fourrows
        or args.neon_vocab_q6_int8
    )
    neon_requested = neon_vocab_requested or args.neon_attention or args.neon_rmsnorm
    if neon_requested and linear_mode not in ("ps", "cached", "cached_qkv", "vocab_bram"):
        raise ValueError("NEON kernels are currently supported only with ps or cached linear mode")
    if args.fuse_projections and linear_mode != "ps":
        raise ValueError("Fused CPU projections are supported only in PS mode")
    q6 = Q6Model(package_dir, linear_mode=linear_mode)
    q6_load_seconds = time.perf_counter() - q6_start
    if linear_mode == "ps":
        gemv = NoPLGEMV()
    else:
        dynamic = linear_mode in (
            "all", "batched", "selected", "cached", "cached_qkv", "cached_vocab", "vocab_bram"
        )
        gemv = PLGEMV(
            package_dir,
            dynamic=dynamic,
            batch_mode=(linear_mode in (
                "batched", "selected", "cached", "cached_qkv", "cached_vocab", "vocab_bram"
            )),
            cache_mode=(linear_mode in ("cached", "cached_qkv", "cached_vocab", "vocab_bram")),
        )
        if linear_mode in ("cached", "cached_qkv", "cached_vocab", "vocab_bram"):
            gemv.load_weight_cache(q6)
    neon_library = package_dir / "neon_kernels.so"
    neon_vocab_prepare_start = time.perf_counter()
    if args.neon_vocab_q6_int8:
        neon_vocab = NeonQ6VocabSGEMV(
            neon_library, q6.packed_weights["tok_embeddings.weight"],
            q6.row_scales["tok_embeddings.weight"],
            q6.shapes["tok_embeddings.weight"], q6.weights["tok_embeddings.weight"],
        )
    elif neon_vocab_requested:
        neon_vocab = NeonSGEMV(
            neon_library, paired_rows=args.neon_vocab_pairrows,
            four_rows=args.neon_vocab_fourrows,
        )
    else:
        neon_vocab = None
    neon_vocab_prepare_seconds = time.perf_counter() - neon_vocab_prepare_start
    neon_attention = NeonGQAAttention(neon_library) if args.neon_attention else None
    neon_rmsnorm = NeonRMSNorm(neon_library) if args.neon_rmsnorm else None
    model = HybridTransformer(
        q6, gemv, linear_mode=linear_mode, neon_vocab=neon_vocab,
        neon_attention=neon_attention, neon_rmsnorm=neon_rmsnorm,
        fuse_projections=args.fuse_projections,
    )
    generation_profile = TimingProfile()

    def profile_snapshots():
        return model.profile.snapshot(), gemv.profile.snapshot(), generation_profile.snapshot()

    def profile_delta(before, after):
        model_before, gemv_before, generation_before = before
        model_after, gemv_after, generation_after = after
        result = {}
        for source, initial, final in (
            ("model", model_before, model_after),
            ("pl_runtime", gemv_before, gemv_after),
            ("generation", generation_before, generation_after),
        ):
            for name, values in TimingProfile.delta(initial, final).items():
                result[source + "." + name] = values
        return result

    def generate():
        key_cache, value_cache = model.new_cache()
        profile_before = profile_snapshots()
        prefill_calls_start = gemv.calls
        prefill_pl_start = gemv.seconds
        start = time.perf_counter()
        logits = None
        for position, token_id in enumerate(request["prompt_ids"]):
            logits = model.step(int(token_id), position, key_cache, value_cache)
        prefill_seconds = time.perf_counter() - start
        prefill_gemv_calls = gemv.calls - prefill_calls_start
        prefill_gemv_seconds = gemv.seconds - prefill_pl_start
        profile_after_prefill = profile_snapshots()

        generated = []
        sampled = []
        logit_rows = []
        comparison = []
        decode_calls_start = gemv.calls
        decode_pl_start = gemv.seconds
        decode_start = time.perf_counter()
        profile_before_decode = profile_snapshots()
        first_sample_seconds = None
        stop_ids = set(int(value) for value in request["stop_token_ids"])
        for index in range(int(request["max_new_tokens"])):
            if index < int(request["reference_step_count"]):
                prefix_matches = generated == request["reference_generated_ids"][:index]
                reference = reference_logits[index]
                logit_rows.append(np.asarray(logits, dtype="<f4").copy())
                if prefix_matches:
                    comparison.append(error_stats(logits, reference))
                else:
                    comparison.append(None)

            scores = logits.copy()
            if index < int(request["min_new_tokens"]):
                scores[list(stop_ids)] = -np.inf
            sample_start = time.perf_counter()
            token_id = int(np.argmax(scores))
            generation_profile.add("greedy_argmax", time.perf_counter() - sample_start)
            if first_sample_seconds is None:
                first_sample_seconds = time.perf_counter() - sample_start
            sampled.append(token_id)
            if token_id in stop_ids and index >= int(request["min_new_tokens"]):
                break
            generated.append(token_id)
            if index + 1 < int(request["max_new_tokens"]):
                logits = model.step(token_id, len(request["prompt_ids"]) + index, key_cache, value_cache)
        decode_seconds = time.perf_counter() - decode_start
        profile_after_decode = profile_snapshots()
        return {
            "generated_ids": generated,
            "sampled_ids": sampled,
            "logits": logit_rows,
            "logit_comparison": comparison,
            "prefill_seconds": prefill_seconds,
            "prefill_gemv_calls": prefill_gemv_calls,
            "prefill_gemv_seconds": prefill_gemv_seconds,
            "decode_seconds": decode_seconds,
            "decode_gemv_calls": gemv.calls - decode_calls_start,
            "decode_gemv_seconds": gemv.seconds - decode_pl_start,
            "first_sample_seconds": first_sample_seconds or 0.0,
            "profile_prefill": profile_delta(profile_before, profile_after_prefill),
            "profile_decode": profile_delta(profile_before_decode, profile_after_decode),
        }

    try:
        for _ in range(args.warmup):
            warmup_result = generate()
            if warmup_result["sampled_ids"] != reference_sampled:
                break

        measured = []
        for _ in range(args.runs):
            before_calls = gemv.calls
            before_pl_seconds = gemv.seconds
            result = generate()
            result["pl_gemv_calls"] = gemv.calls - before_calls
            result["pl_gemv_seconds"] = gemv.seconds - before_pl_seconds
            measured.append(result)

        def summarize_profiles(phase, denominator):
            field = "profile_" + phase
            names = sorted(set(name for item in measured for name in item[field]))
            summary = {}
            for name in names:
                seconds = [item[field].get(name, [0.0, 0])[0] for item in measured]
                calls = [item[field].get(name, [0.0, 0])[1] for item in measured]
                median_seconds = float(np.median(seconds))
                summary[name] = {
                    "median_seconds": median_seconds,
                    "median_calls": int(np.median(calls)),
                    "percent_of_phase_wall": 100.0 * median_seconds / denominator if denominator else 0.0,
                }
            return summary

        median_prefill = float(np.median([item["prefill_seconds"] for item in measured]))
        median_decode = float(np.median([item["decode_seconds"] for item in measured]))
        median_tokens = float(np.median([len(item["generated_ids"]) for item in measured]))
        median_pl = float(np.median([item["pl_gemv_seconds"] for item in measured]))
        median_prefill_pl = float(np.median([item["prefill_gemv_seconds"] for item in measured]))
        median_decode_pl = float(np.median([item["decode_gemv_seconds"] for item in measured]))
        comparable_metrics = [item for item in measured[-1]["logit_comparison"] if item is not None]
        final = measured[-1]
        prefix_comparisons = [item for item in final["logit_comparison"] if item is not None]
        report = {
            "prompt": request["prompt"],
            "prompt_ids": request["prompt_ids"],
            "reference_sampled_ids": reference_sampled,
            "generated_ids": final["generated_ids"],
            "sampled_ids": final["sampled_ids"],
            "tokens_match_reference": final["sampled_ids"] == reference_sampled,
            "all_measured_runs_match_reference": all(item["sampled_ids"] == reference_sampled for item in measured),
            "matching_token_prefix": next((i for i, pair in enumerate(zip(final["sampled_ids"], reference_sampled)) if pair[0] != pair[1]), min(len(final["sampled_ids"]), len(reference_sampled))),
            "logit_comparison_steps": len(prefix_comparisons),
            "logit_metrics": prefix_comparisons,
            "logit_metrics_summary": {
                "mean_relative_rmse": float(np.mean([item["relative_rmse"] for item in comparable_metrics])) if comparable_metrics else None,
                "max_relative_rmse": float(np.max([item["relative_rmse"] for item in comparable_metrics])) if comparable_metrics else None,
                "minimum_cosine": float(np.min([item["cosine"] for item in comparable_metrics])) if comparable_metrics else None,
                "all_argmax_match": all(item["argmax_match"] for item in comparable_metrics),
            },
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "timing_seconds_median": {
                "q6_weight_unpack": q6_load_seconds,
                "neon_vocab_prepare": neon_vocab_prepare_seconds,
                "overlay_load": gemv.load_seconds,
                "pl_weight_cache_load": gemv.weight_cache_load_seconds,
                "prefill": median_prefill,
                "ttft": median_prefill + float(np.median([item["first_sample_seconds"] for item in measured])),
                "decode": median_decode,
                "end_to_end_model": median_prefill + median_decode,
                "pl_gemv_dma_total": median_pl,
                "pl_gemv_dma_prefill": median_prefill_pl,
                "pl_gemv_dma_decode": median_decode_pl,
            },
            "operator_profile_prefill": summarize_profiles("prefill", median_prefill),
            "operator_profile_decode": summarize_profiles("decode", median_decode),
            "throughput_tokens_per_second": {
                "prefill": len(request["prompt_ids"]) / median_prefill if median_prefill else 0.0,
                "decode": median_tokens / median_decode if median_decode else 0.0,
                "end_to_end_model": median_tokens / (median_prefill + median_decode) if median_prefill + median_decode else 0.0,
            },
            "pl_gemv_calls_per_run": int(np.median([item["pl_gemv_calls"] for item in measured])),
            "linear_mode": linear_mode,
            "fpga_clock_mhz": gemv.fpga_clock_mhz,
            "fpga_core_clock_mhz": gemv.fpga_clock_mhz,
            "fpga_dma_clock_mhz": gemv.dma_clock_mhz,
            "arm_vocab_kernel": (
                "ARMv7 NEON pairrows" if args.neon_vocab_pairrows
                else "ARMv7 NEON Q6 int8 weights / int16 activations" if args.neon_vocab_q6_int8
                else "ARMv7 NEON fourrows" if args.neon_vocab_fourrows
                else "ARMv7 NEON" if args.neon_vocab else "NumPy/ATLAS"
            ),
            "arm_attention_kernel": "ARMv7 NEON" if args.neon_attention else "NumPy/ATLAS",
            "arm_rmsnorm_kernel": "ARMv7 NEON" if neon_rmsnorm else "NumPy/ATLAS",
            "fused_cpu_projections": bool(args.fuse_projections),
            "execution_device": "PS only; overlay not loaded" if linear_mode == "ps" else "PS+PL hybrid",
            "accelerated_linears": (
                "none; all dense Q6 linears execute on PS"
                if linear_mode == "ps"
                else "all Q6 dense linears including attention Q/K/V/O, FFN W1/W2/W3, and tiled vocabulary projection"
                if linear_mode in ("all", "batched")
                else "Q6 vocabulary matrix cached in BRAM; all other dense linears execute on PS"
                if linear_mode == "vocab_bram"
                else "FFN W1/W3 and attention Q/K/V stored in PL BRAM and reused"
                if linear_mode == "cached_qkv"
                else "FFN W1/W3 cached in BRAM; vocabulary projection streamed from PS DDR"
                if linear_mode == "cached_vocab"
                else "FFN W1/W3 stored in PL BRAM and reused across decode steps"
                if linear_mode == "cached"
                else "FFN W1/W3 batched in PL; attention projections, W2, and vocabulary projection on ARM"
                if linear_mode == "selected"
                else "W1 and W3 in all five transformer layers"
            ),
            "gemv_batching": (
                "not used"
                if linear_mode == "ps"
                else "Q/K/V and W1/W3 share one DMA transfer per layer; vocabulary projection uses one 4096-row transfer"
                if linear_mode == "batched"
                else "One 4096-row cached vocabulary GEMV; only a 128-element int16 activation is sent per token"
                if linear_mode == "vocab_bram"
                else "Cached FFN W1/W3 and attention QKV weights; only int16 activations are sent at runtime"
                if linear_mode == "cached_qkv"
                else "W1/W3 use BRAM cache; 4096-row vocabulary matrix is sent per token"
                if linear_mode == "cached_vocab"
                else "W1/W3 use BRAM cache slots; only int16 activations are sent during each decode step"
                if linear_mode == "cached"
                else "W1/W3 share one DMA transfer per layer"
                if linear_mode == "selected"
                else "one DMA transfer per matrix operation"
            ),
            "remaining_work_on_arm": (
                "all model operations"
                if linear_mode == "ps"
                else "embedding lookup, RMSNorm, RoPE/GQA/KV attention, SwiGLU, scale reconstruction and greedy argmax"
                if linear_mode in ("all", "batched")
                else "embedding lookup, attention Q/K/V/O and math, FFN W1/W2/W3, RMSNorm, RoPE, SwiGLU, scales, and argmax"
                if linear_mode == "vocab_bram"
                else "embedding, attention Wo/math/KV, W2, vocabulary projection, RMSNorm, RoPE, SwiGLU, scales, and argmax"
                if linear_mode == "cached_qkv"
                else "embedding, attention projections/math/KV, W2, RMSNorm, RoPE, SwiGLU, scales, and argmax"
                if linear_mode == "cached_vocab"
                else "embedding, attention Q/K/V/O, W2, vocabulary projection, RMSNorm, RoPE/GQA/KV, SwiGLU, scales, and argmax"
            ),
            "pl_weight_cache": {
                "bytes": gemv.weight_cache_bytes,
                "matrices": gemv.cached_matrix_count,
                "storage": "one Q6 4096x128 vocabulary matrix in BRAM" if linear_mode == "vocab_bram"
                else "Q6 W1/W3 plus attention QKV in BRAM" if linear_mode == "cached_qkv"
                else "10 Q6 W1/W3 matrices in BRAM; vocabulary weights streamed from PS DDR" if linear_mode == "cached_vocab"
                else "10 Q6 W1/W3 matrices in BRAM" if linear_mode == "cached" else "none",
            },
            "note": "Tokenization is performed on the host; PL returns raw int32 GEMV accumulators and ARM applies row/activation scales.",
        }
        output_path = args.output.resolve() if args.output else package_dir / "board_result.json"
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        logits_path = output_path.with_name("board_logits.bin")
        rows = final["logits"]
        if rows:
            np.asarray(rows, dtype="<f4").reshape(-1).tofile(str(logits_path))
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    finally:
        gemv.close()


if __name__ == "__main__":
    main()
