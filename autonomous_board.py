import argparse
import json
import os
import struct
import time
from pathlib import Path

import numpy as np


BASE_MODEL_BYTES = 1492224
KV_BYTES = 1310720
MAX_SEQ = 512
HEAD_DIM = 32
ROPE_TABLE_BYTES = MAX_SEQ * HEAD_DIM * 4
MAX_NEW = 128
VOCAB = 4096
OUTPUT_TOKEN_BASE = MAX_SEQ
RESULT_COUNT_INDEX = OUTPUT_TOKEN_BASE + MAX_NEW
STOPPED_TOKEN_INDEX = RESULT_COUNT_INDEX + 1
LOGITS_BASE = STOPPED_TOKEN_INDEX + 1
TRACE_RECORD_WORDS = 12
TRACE_STAGE_COUNT = 80
TRACE_WORDS = TRACE_RECORD_WORDS * TRACE_STAGE_COUNT
TRACE_BASE = LOGITS_BASE + MAX_NEW * VOCAB
IO_WORDS = TRACE_BASE + TRACE_WORDS
VOCAB_STREAM_HALF_ROWS = VOCAB // 2
VOCAB_STREAM_ROW_BYTES = 136
VOCAB_STREAM_HALF_BYTES = VOCAB_STREAM_HALF_ROWS * VOCAB_STREAM_ROW_BYTES
VOCAB_STREAM_HALF_OFFSET = BASE_MODEL_BYTES + VOCAB_STREAM_HALF_BYTES
AUXILIARY_WORDS = VOCAB_STREAM_HALF_BYTES // 8
IO_ALLOC_WORDS = IO_WORDS
MODEL_PROBE_OFFSETS = (
    0, 4, 16380, 16384, 16388, 16396,
    540672, 540676, 541180, 541184, 541188, 541196,
    1486592, 1486596, 1487104, 1487108, 1491712, BASE_MODEL_BYTES - 4,
)
WQ_DATA_OFFSET = 541184
WQ_SCALE_OFFSET = 540672
REG_WORKSPACE = 0x10
REG_IO = 0x18
REG_PROMPT_LENGTH = 0x20
REG_MAX_NEW = 0x28
REG_MIN_NEW = 0x30
REG_STOP0 = 0x38
REG_STOP1 = 0x40
REG_DUMP_LOGITS = 0x48
REG_VOCAB_STREAM = 0x50
REG_KV_CACHE_ALIAS = 0x58
REG_W3_WORKSPACE = 0x58
REG_CONTROL = 0x00


def allocate(shape, dtype):
    try:
        from pynq import allocate
        return allocate(shape=shape, dtype=dtype)
    except ImportError:
        from pynq import Xlnk
        return Xlnk().cma_array(shape=shape, dtype=dtype)


def physical_address(buffer):
    for name in ("device_address", "physical_address"):
        value = getattr(buffer, name, None)
        if value is not None:
            return int(value)
    raise RuntimeError("PYNQ CMA buffer does not expose a physical address")


def flush(buffer):
    method = getattr(buffer, "flush", None)
    if method:
        method()


def invalidate(buffer):
    method = getattr(buffer, "invalidate", None)
    if method:
        method()


def read_prompt(path):
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError("hls_prompt.bin is truncated")
    prompt_length, max_new, min_new, stop0, stop1 = struct.unpack_from("<5i", data)
    expected_bytes = 20 + prompt_length * 4
    if len(data) != expected_bytes:
        raise ValueError("hls_prompt.bin has an invalid token payload")
    prompt_ids = list(struct.unpack_from("<{}i".format(prompt_length), data, 20))
    return prompt_ids, max_new, min_new, stop0, stop1


def trace_stage_labels():
    labels = ["embedding"]
    per_layer = (
        "attention_norm", "q_projection", "k_projection", "v_projection",
        "q_rope", "k_rope", "attention_output", "wo_projection",
        "attention_residual", "ffn_norm", "w1_gate", "w3_up",
        "swiglu", "w2_projection", "ffn_residual",
    )
    for layer in range(5):
        labels.extend("layer{}_{}".format(layer, name) for name in per_layer)
    labels.extend(("final_norm", "vocab_logits"))
    return labels


def buffer_address(buffer):
    address = physical_address(buffer)
    if address < 0 or address > 0xffffffff:
        raise RuntimeError("This HLS IP exposes 32-bit DDR pointers; CMA address is out of range")
    return address


def run_once(mmio, workspace, io, vocab_stream, kv_cache_alias_addr,
             prompt, max_new, min_new, stop0, stop1, timeout,
             allow_invalid=False, debug_values=None, dump_logits=False,
             dual_ffn_w3=False):
    io[:IO_WORDS] = 0
    io[:len(prompt)] = prompt
    if debug_values is not None:
        io[0] = debug_values[0]
        io[1] = debug_values[1]
    flush(workspace)
    flush(io)
    flush(vocab_stream)
    workspace_addr = buffer_address(workspace)
    io_addr = buffer_address(io)
    vocab_stream_addr = buffer_address(vocab_stream)
    if kv_cache_alias_addr is not None and kv_cache_alias_addr > 0xffffffff:
        raise RuntimeError("workspace KV pointer is outside the HLS 32-bit address range")
    mmio.write(REG_WORKSPACE, workspace_addr)
    mmio.write(REG_IO, io_addr)
    mmio.write(REG_PROMPT_LENGTH, len(prompt))
    mmio.write(REG_MAX_NEW, max_new)
    mmio.write(REG_MIN_NEW, min_new)
    mmio.write(REG_STOP0, stop0)
    mmio.write(REG_STOP1, stop1)
    mmio.write(REG_DUMP_LOGITS, int(dump_logits))
    mmio.write(REG_VOCAB_STREAM, vocab_stream_addr)
    if kv_cache_alias_addr is not None:
        mmio.write(REG_KV_CACHE_ALIAS, kv_cache_alias_addr)
    if dual_ffn_w3:
        mmio.write(REG_W3_WORKSPACE, workspace_addr)
    kv_address = "none" if kv_cache_alias_addr is None else "0x{:08x}".format(kv_cache_alias_addr)
    print("PL launch: workspace=0x{:08x} io=0x{:08x} vocab=0x{:08x} kv={} prompt={} max_new={}".format(
        workspace_addr, io_addr, vocab_stream_addr, kv_address, len(prompt), max_new
    ), flush=True)
    mmio.write(REG_CONTROL, 1)

    start = time.perf_counter()
    deadline = start + timeout
    while True:
        control = mmio.read(REG_CONTROL)
        if control & 0x2:
            break
        if time.perf_counter() >= deadline:
            offsets = (0x00, 0x04, 0x08, 0x0c, 0x10, 0x14, 0x18, 0x1c,
                       0x20, 0x24, 0x28, 0x2c, 0x30, 0x34, 0x38, 0x3c, 0x40, 0x44)
            registers = {"0x{:02x}".format(offset): "0x{:08x}".format(mmio.read(offset))
                         for offset in offsets}
            invalidate(io)
            progress = {
                "result_count": int(io[RESULT_COUNT_INDEX]),
                "first_output_id": int(io[OUTPUT_TOKEN_BASE]),
                "last_output_id": int(io[OUTPUT_TOKEN_BASE + max_new - 1]),
            }
            raise TimeoutError("PL inference timed out; registers={} io={}".format(registers, progress))
        time.sleep(0.001)
    elapsed = time.perf_counter() - start
    invalidate(io)
    count = int(io[RESULT_COUNT_INDEX])
    stopped_on = int(io[STOPPED_TOKEN_INDEX])
    if allow_invalid and count == -1:
        return elapsed, [], -1, np.empty((0,), dtype=np.float32)
    if allow_invalid and count == -2:
        values = io[LOGITS_BASE:LOGITS_BASE + len(MODEL_PROBE_OFFSETS)]
        return elapsed, [], -2, values.view(np.uint32).copy()
    if allow_invalid and count == -3:
        return elapsed, [], -3, io[TRACE_BASE:TRACE_BASE + TRACE_WORDS].copy()
    if count < 0 or count > max_new:
        raise RuntimeError("PL returned invalid generated-token count {}".format(count))
    token_ids = [int(value) for value in io[OUTPUT_TOKEN_BASE:OUTPUT_TOKEN_BASE + count]]
    logits = (io.view(np.float32)[LOGITS_BASE:LOGITS_BASE + count * VOCAB].copy()
              if dump_logits else np.empty((0,), dtype=np.float32))
    return elapsed, token_ids, stopped_on, logits


def main():
    parser = argparse.ArgumentParser(description="Run full autoregressive TinyLLM inference on the PYNQ PL")
    parser.add_argument("--app-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--clock-mhz", type=float)
    parser.add_argument("--bus-probe", action="store_true")
    parser.add_argument("--memory-probe", action="store_true")
    parser.add_argument("--step-trace", action="store_true")
    parser.add_argument("--smoke-one", action="store_true")
    parser.add_argument("--smoke-prefix", type=int)
    parser.add_argument("--dump-logits", action="store_true",
                        help="copy every vocabulary logit vector back to DDR for numerical validation")
    parser.add_argument("--performance-overlay", action="store_true",
                        help="mark an overlay built without trace-debug hardware")
    parser.add_argument("--legacy-kv-mirror", action="store_true",
                        help="use the legacy auxiliary-buffer KV mirror interface")
    parser.add_argument("--dual-ffn-w3", action="store_true",
                        help="pass the W3 workspace base on the separate gmem3 master")
    parser.add_argument("--precomputed-rope", action="store_true",
                        help="load the precomputed RoPE table from the workspace tail")
    parser.add_argument("--parallel-kv", action="store_true",
                        help="run key and value projections in parallel on gmem2/gmem3")
    args = parser.parse_args()
    if args.parallel_kv:
        if not args.legacy_kv_mirror:
            raise ValueError("--parallel-kv requires --legacy-kv-mirror")
        args.dual_ffn_w3 = True
        args.precomputed_rope = True
    if args.runs < 1 or args.runs > 10:
        raise ValueError("runs must be in 1..10")
    if args.performance_overlay and args.step_trace:
        raise ValueError("--step-trace is unavailable in a performance overlay")
    if args.dual_ffn_w3 and not args.legacy_kv_mirror:
        raise ValueError("--dual-ffn-w3 requires --legacy-kv-mirror")
    if sum((args.bus_probe, args.memory_probe, args.step_trace, args.smoke_one,
            args.smoke_prefix is not None)) > 1:
        raise ValueError("use either --smoke-one or --smoke-prefix")
    app_dir = args.app_dir.resolve()

    request = json.loads((app_dir / "request.json").read_text(encoding="utf-8"))
    prompt, max_new, min_new, stop0, stop1 = read_prompt(app_dir / "hls_prompt.bin")
    if len(prompt) != len(request["prompt_ids"]) or prompt != request["prompt_ids"]:
        raise ValueError("HLS prompt IDs differ from request.json")
    smoke_mode = args.smoke_one or args.smoke_prefix is not None
    probe_mode = args.bus_probe or args.memory_probe or args.step_trace
    if args.bus_probe:
        prompt, max_new, min_new = [], 1, 0
    elif args.memory_probe:
        prompt, max_new, min_new = [], 0, 0
    elif args.step_trace:
        trace_token = prompt[0]
        prompt, max_new, min_new = [], -3, 0
    elif args.smoke_one:
        max_new, min_new = 1, 0
    elif args.smoke_prefix is not None:
        if args.smoke_prefix < 1 or args.smoke_prefix > len(prompt):
            raise ValueError("smoke-prefix must be within the prepared prompt length")
        prompt, max_new, min_new = prompt[:args.smoke_prefix], 1, 0
    model_path = app_dir / "autonomous_model.bin"
    model_data = model_path.read_bytes()
    model_bytes = len(model_data)
    if model_bytes < VOCAB_STREAM_HALF_OFFSET + VOCAB_STREAM_HALF_BYTES or model_bytes % 8:
        raise ValueError("autonomous model image has an invalid size: {} bytes".format(model_bytes))
    rope_table_data = None
    if args.precomputed_rope:
        rope_table_data = (app_dir / "rope_table.bin").read_bytes()
        if len(rope_table_data) != ROPE_TABLE_BYTES:
            raise ValueError("rope_table.bin has an invalid size: {} bytes".format(len(rope_table_data)))
    workspace_bytes = model_bytes + KV_BYTES + (ROPE_TABLE_BYTES if args.precomputed_rope else 0)

    workspace = allocate((workspace_bytes // 8,), np.uint64)
    io = allocate((IO_ALLOC_WORDS,), np.int32)
    if args.parallel_kv:
        auxiliary_words = (model_bytes + KV_BYTES) // 8
    else:
        auxiliary_words = AUXILIARY_WORDS + (KV_BYTES // 8 if args.legacy_kv_mirror else 0)
    vocab_stream = allocate((auxiliary_words,), np.uint64)
    workspace[:] = 0
    vocab_stream[:] = 0
    workspace.view(np.uint8)[:model_bytes] = np.frombuffer(model_data, dtype=np.uint8)
    if rope_table_data is not None:
        start = model_bytes + KV_BYTES
        workspace.view(np.uint8)[start:start + ROPE_TABLE_BYTES] = np.frombuffer(
            rope_table_data, dtype=np.uint8,
        )
    if args.parallel_kv:
        vocab_stream.view(np.uint8)[:model_bytes] = np.frombuffer(model_data, dtype=np.uint8)
    else:
        vocab_stream.view(np.uint8)[:VOCAB_STREAM_HALF_BYTES] = np.frombuffer(
            model_data, dtype=np.uint8, count=VOCAB_STREAM_HALF_BYTES,
            offset=VOCAB_STREAM_HALF_OFFSET,
        )
    kv_cache_alias_addr = (None if args.legacy_kv_mirror
                           else buffer_address(workspace) + model_bytes)

    from pynq import MMIO, Overlay
    from pynq.ps import Clocks
    if args.clock_mhz is not None:
        Clocks.fclk0_mhz = args.clock_mhz
    actual_clock_mhz = float(Clocks.fclk0_mhz)
    bit_path = app_dir / "tinyllm_autonomous.bit"
    overlay = Overlay(str(bit_path), download=True)
    ip_info = overlay.ip_dict.get("tinyllm_autonomous_0")
    if ip_info is None:
        raise RuntimeError("HWH does not describe tinyllm_autonomous_0; available IPs: {}".format(
            sorted(overlay.ip_dict.keys())
        ))
    base = ip_info.get("phys_addr", ip_info.get("base_addr"))
    if base is None:
        raise RuntimeError("HWH has no control-register base address: {}".format(ip_info))
    mmio = MMIO(int(base), 0x10000)

    reference_ids = [int(value) for value in request["reference_generated_ids"]]
    with (app_dir / "reference_logits.bin").open("rb") as reference_file:
        reference = np.fromfile(reference_file, dtype="<f4")
    if reference.size % VOCAB:
        raise ValueError("reference logits file is not a whole number of vocabulary rows")
    reference = reference.reshape((-1, VOCAB))

    run_seconds = []
    final = None
    for run in range(args.runs):
        workspace[model_bytes // 8:(model_bytes + KV_BYTES) // 8] = 0
        if args.parallel_kv:
            vocab_stream[model_bytes // 8:] = 0
        elif args.legacy_kv_mirror:
            vocab_stream[AUXILIARY_WORDS:] = 0
        elapsed, token_ids, stopped_on, logits = run_once(
            mmio, workspace, io, vocab_stream, kv_cache_alias_addr,
            prompt, max_new, min_new, stop0, stop1,
            args.timeout, allow_invalid=probe_mode,
            debug_values=(trace_token, 0) if args.step_trace else None,
            dump_logits=args.dump_logits,
            dual_ffn_w3=args.dual_ffn_w3,
        )
        run_seconds.append(elapsed)
        if args.memory_probe:
            actual_words = [int(value) for value in logits]
            expected_words = [struct.unpack_from("<I", model_data, offset)[0]
                              for offset in MODEL_PROBE_OFFSETS]
            final = {
                "run": run + 1,
                "elapsed_seconds": elapsed,
                "probe_completed": stopped_on == -2,
                "probe_offsets": list(MODEL_PROBE_OFFSETS),
                "actual_words": actual_words,
                "expected_words": expected_words,
                "all_model_probe_words_match": actual_words == expected_words,
            }
            if not final["probe_completed"] or not final["all_model_probe_words_match"]:
                break
            continue
        if args.step_trace:
            labels = trace_stage_labels()
            words = logits.reshape((TRACE_STAGE_COUNT, TRACE_RECORD_WORDS))
            records = []
            for row in words:
                stage = int(row[0])
                if int(row[1]) <= 0 or stage < 0 or stage >= len(labels):
                    break
                values = row[3:10].view(np.float32)
                records.append({
                    "stage": labels[stage],
                    "length": int(row[1]),
                    "finite_count": int(row[2]),
                    "min": float(values[0]),
                    "max": float(values[1]),
                    "samples": [float(value) for value in values[2:6]],
                    "last": float(values[6]),
                    "first_nonfinite_index": int(row[10]),
                })
            normalized = io[:128].view(np.float32).astype(np.float32, copy=True)
            q_weights = np.frombuffer(model_data, dtype=np.int8, count=128,
                                      offset=WQ_DATA_OFFSET).astype(np.float32)
            q_scale = struct.unpack_from("<f", model_data, WQ_SCALE_OFFSET)[0]
            input_maximum = np.max(np.abs(normalized))
            activation_scale = np.float32(input_maximum / np.float32(32767.0))
            quant_multiplier = np.float32(np.float32(32767.0) / input_maximum)
            scaled = np.asarray(normalized * quant_multiplier, dtype=np.float32)
            quantized = np.where(
                scaled >= 0.0,
                np.floor(scaled + np.float32(0.5)),
                np.ceil(scaled - np.float32(0.5)),
            ).astype(np.int16)
            quantized = np.clip(quantized.astype(np.int32), -32767, 32767).astype(np.int16)
            dequantized = quantized.astype(np.float32) * activation_scale
            partials = [
                int(np.dot(q_weights[index::4].astype(np.int32),
                           quantized[index::4].astype(np.int32)))
                for index in range(4)
            ]
            integer_sum = (partials[0] + partials[1]) + (partials[2] + partials[3])
            expected_q0 = np.float32(
                np.float32(np.float32(integer_sum) * activation_scale) * np.float32(q_scale)
            )
            metadata = [int(value) & 0xffffffff
                        for value in logits[78 * TRACE_RECORD_WORDS:80 * TRACE_RECORD_WORDS]]
            expected_metadata = [
                WQ_DATA_OFFSET, WQ_SCALE_OFFSET,
                struct.unpack_from("<I", model_data, WQ_DATA_OFFSET)[0],
                struct.unpack_from("<I", model_data, WQ_DATA_OFFSET + 4)[0],
                struct.unpack_from("<I", model_data, WQ_SCALE_OFFSET)[0],
                struct.unpack_from("<I", model_data, WQ_SCALE_OFFSET + 4)[0],
            ]
            q_record = records[2]
            q0_actual = q_record["samples"][0]
            first_logits = io.view(np.float32)[LOGITS_BASE:LOGITS_BASE + VOCAB].copy()
            final = {
                "run": run + 1,
                "elapsed_seconds": elapsed,
                "probe_completed": stopped_on == -3,
                "token": int(trace_token),
                "position": 0,
                "first_logit_finite_count": int(np.count_nonzero(np.isfinite(first_logits))),
                "first_logit_argmax": int(np.argmax(first_logits)),
                "wq_metadata_words": metadata,
                "wq_expected_metadata_words": expected_metadata,
                "wq_metadata_matches": metadata[:6] == expected_metadata,
                "wq_inner_word_addresses": metadata[6:8],
                "wq_inner_packed_words": metadata[8:10],
                "wq_vector": np.asarray(metadata[10:14], dtype=np.uint32).view(np.float32).tolist(),
                "wq_inner_scale_word": metadata[14],
                "wq_scale_inside_matvec": float(np.asarray(metadata[15:16], dtype=np.uint32).view(np.float32)[0]),
                "wq_partials": np.asarray(metadata[16:20], dtype=np.uint32).view(np.float32).tolist(),
                "wq_result_inside_matvec": float(np.asarray(metadata[20:21], dtype=np.uint32).view(np.float32)[0]),
                "wq_trace_signature": metadata[21],
                "wq_row0_expected": float(expected_q0),
                "wq_row0_actual": q0_actual,
                "stages": records,
            }
            with (app_dir / "board_trace.bin").open("wb") as trace_file:
                logits.tofile(trace_file)
            with (app_dir / "board_trace_logits.bin").open("wb") as logits_file:
                first_logits.astype("<f4", copy=False).tofile(logits_file)
            continue
        if args.bus_probe:
            final = {
                "run": run + 1,
                "elapsed_seconds": elapsed,
                "probe_completed": (
                    not token_ids and int(io[RESULT_COUNT_INDEX]) == -1
                    if args.bus_probe else len(token_ids) == 1 and int(io[RESULT_COUNT_INDEX]) == 1
                ),
                "invalid_request_status": int(io[RESULT_COUNT_INDEX]),
                "generated_ids": token_ids,
            }
            if not final["probe_completed"]:
                break
            continue
        if smoke_mode:
            actual_logits = logits[:VOCAB]
            expected_logits = reference[0]
            actual_argmax = (int(np.argmax(actual_logits)) if actual_logits.size else
                             (token_ids[0] if len(token_ids) == 1 else None))
            final = {
                "run": run + 1,
                "elapsed_seconds": elapsed,
                "generated_ids": token_ids,
                "generated_token_count": len(token_ids),
                "first_token_matches_reference": bool(args.smoke_one and
                    len(token_ids) == 1 and token_ids[0] == reference_ids[0]
                ),
                "first_logit_argmax": actual_argmax,
                "first_logit_expected_argmax": int(np.argmax(expected_logits)),
                "first_logit_argmax_matches_reference": bool(
                    args.smoke_one and len(token_ids) == 1 and actual_argmax is not None
                    and actual_argmax == np.argmax(expected_logits)
                ),
                "first_logit_min": float(np.min(actual_logits)) if actual_logits.size else None,
                "first_logit_max": float(np.max(actual_logits)) if actual_logits.size else None,
                "probe_completed": len(token_ids) == 1 and int(io[RESULT_COUNT_INDEX]) == 1,
                "prompt_prefix_tokens": len(prompt),
            }
            continue
        steps = min(len(token_ids), len(reference_ids), len(reference))
        expected_logits = reference[:steps]
        if logits.size:
            actual_logits = logits[:steps * VOCAB].reshape((steps, VOCAB))
            argmax_matches = int(np.count_nonzero(
                np.argmax(actual_logits, axis=1) == np.argmax(expected_logits, axis=1)
            ))
            delta = actual_logits.astype(np.float64) - expected_logits.astype(np.float64)
            ref64 = expected_logits.astype(np.float64)
            relative_rmse = float(np.sqrt(np.sum(delta * delta) / max(np.sum(ref64 * ref64), 1e-30)))
            max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
        else:
            argmax_matches = None
            relative_rmse = None
            max_abs_error = None
        tokens_match = token_ids == reference_ids[:len(token_ids)] and len(token_ids) == len(reference_ids)
        final = {
            "run": run + 1,
            "elapsed_seconds": elapsed,
            "generated_tokens": len(token_ids),
            "decode_tokens_per_second": len(token_ids) / elapsed if elapsed else 0.0,
            "tokens_match_reference": tokens_match,
            "logit_argmax_matches": argmax_matches,
            "logits_compared": bool(logits.size),
            "reference_steps_compared": steps,
            "relative_logit_rmse": relative_rmse,
            "max_absolute_logit_error": max_abs_error,
            "stopped_on_token": stopped_on,
            "generated_ids": token_ids,
        }
        if not tokens_match or (argmax_matches is not None and argmax_matches != steps):
            break

    report = {
        "mode": ("model_DDR_probe" if args.memory_probe else
                 "single_model_step_trace" if args.step_trace else
                 "AXI_bus_probe" if args.bus_probe else
                 "one_step_PL_smoke" if smoke_mode else "autonomous_full_model_PL"),
        "clock_mhz": actual_clock_mhz,
        "requested_clock_mhz": args.clock_mhz,
        "runs": run_seconds,
        "median_seconds": float(np.median(run_seconds)),
        "request_prompt_tokens": len(prompt),
        "requested_new_tokens": max_new,
        "logits_output_enabled": args.dump_logits,
        "all_runs_match_golden": (
            None if probe_mode or args.smoke_prefix is not None else bool(
                final and final.get("tokens_match_reference", final.get("first_token_matches_reference", False))
                and (final.get("logit_argmax_matches") is None
                     or final.get("logit_argmax_matches") == final.get("reference_steps_compared", 1))
                and (final.get("first_logit_argmax_matches") is None
                     or final.get("first_logit_argmax_matches"))
            )
        ),
        "last_run": final,
        "model_workspace_bytes": workspace_bytes,
        "io_buffer_words": len(io),
        "bitstream": str(bit_path),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if (args.dump_logits and not args.bus_probe and final is not None
            and final.get("reference_steps_compared", 0)):
        final_logits = logits[:final["reference_steps_compared"] * VOCAB]
        with args.output.with_name("board_logits.bin").open("wb") as logits_file:
            final_logits.astype("<f4", copy=False).tofile(logits_file)
    elif args.dump_logits and smoke_mode and logits.size:
        with args.output.with_name("board_logits.bin").open("wb") as logits_file:
            logits.astype("<f4", copy=False).tofile(logits_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    del io
    del workspace
    del vocab_stream
    if probe_mode and (not final["probe_completed"] or
                       (args.memory_probe and not final["all_model_probe_words_match"])):
        raise SystemExit(2)
    if not probe_mode and not smoke_mode and not report["all_runs_match_golden"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
