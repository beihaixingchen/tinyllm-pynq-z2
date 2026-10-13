import argparse
import time
from pathlib import Path

import numpy as np
from pynq import Overlay, allocate


OUTPUTS = 352
RUNS = 20


def error_stats(actual, reference):
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    error = actual - reference
    reference_rms = float(np.sqrt(np.mean(reference * reference)))
    actual_norm = float(np.linalg.norm(actual))
    reference_norm = float(np.linalg.norm(reference))
    cosine = float(np.dot(actual, reference) / (actual_norm * reference_norm)) if actual_norm and reference_norm else 1.0
    return {
        "rmse": float(np.sqrt(np.mean(error * error))),
        "max_abs": float(np.max(np.abs(error))),
        "reference_rms": reference_rms,
        "relative_rmse": float(np.sqrt(np.mean(error * error)) / reference_rms) if reference_rms else 0.0,
        "cosine": cosine,
    }


def main():
    parser = argparse.ArgumentParser(description="Validate a real TinyLLM activation against the Q6W GEMV overlay")
    parser.add_argument("vector_dir", type=Path, help="Directory containing a captured real activation and references")
    args = parser.parse_args()

    vector_dir = args.vector_dir.resolve()
    input_data = np.fromfile(str(vector_dir / "input.bin"), dtype=np.uint8)
    expected = np.fromfile(str(vector_dir / "expected.bin"), dtype="<i4")
    row_scales = np.fromfile(str(vector_dir / "row_scales.bin"), dtype="<f4")
    activation_scale_data = np.fromfile(str(vector_dir / "activation_scale.bin"), dtype="<f4")
    q6_reference = np.fromfile(str(vector_dir / "q6_reference.bin"), dtype="<f4")
    fp32_reference = np.fromfile(str(vector_dir / "fp32_reference.bin"), dtype="<f4")
    if expected.size != OUTPUTS or row_scales.size != OUTPUTS or q6_reference.size != OUTPUTS or fp32_reference.size != OUTPUTS:
        raise ValueError("Expected exactly {} output and row-scale values".format(OUTPUTS))
    if activation_scale_data.size != 1 or input_data.size != 33920:
        raise ValueError("Invalid activation scale or input packet length")

    app_dir = Path(__file__).resolve().parent
    overlay = Overlay(str(app_dir / "q6w_gemv.bit"))
    dma = overlay.axi_dma_0
    source = allocate(shape=(input_data.size,), dtype=np.uint8)
    destination = allocate(shape=(OUTPUTS,), dtype=np.int32)
    source[:] = input_data

    def run_once():
        dma.recvchannel.transfer(destination)
        dma.sendchannel.transfer(source)
        dma.sendchannel.wait()
        dma.recvchannel.wait()

    try:
        run_once()
        actual = np.asarray(destination, dtype=np.int32).copy()
        if not np.array_equal(actual, expected):
            mismatch = int(np.flatnonzero(actual != expected)[0])
            raise AssertionError("raw accumulator row {} expected {}, got {}".format(
                mismatch, int(expected[mismatch]), int(actual[mismatch])
            ))

        times = []
        for _ in range(RUNS):
            start = time.perf_counter()
            run_once()
            times.append(time.perf_counter() - start)
            actual = np.asarray(destination, dtype=np.int32).copy()
            if not np.array_equal(actual, expected):
                mismatch = int(np.flatnonzero(actual != expected)[0])
                raise AssertionError("raw accumulator row {} expected {}, got {}".format(
                    mismatch, int(expected[mismatch]), int(actual[mismatch])
                ))

        activation_scale = float(activation_scale_data[0])
        scaled_output = actual.astype(np.float64) * row_scales.astype(np.float64) * activation_scale
        print("PASS: {} DMA runs matched all {} exact raw accumulators".format(RUNS, OUTPUTS))
        print("Input bytes: {}; output bytes: {}".format(input_data.size, destination.nbytes))
        print("DMA wall time: median {:.6f} s, min {:.6f} s, max {:.6f} s".format(
            float(np.median(times)), min(times), max(times)
        ))
        for name, reference in (
            ("Q6 dequantized FP32 layer", q6_reference),
            ("source checkpoint FP32 layer", fp32_reference),
        ):
            stats = error_stats(scaled_output, reference)
            print("{}: RMSE={:.8g}, max_abs={:.8g}, relative_RMSE={:.6%}, cosine={:.8f}".format(
                name, stats["rmse"], stats["max_abs"], stats["relative_rmse"], stats["cosine"]
            ))
        q6_vs_fp32 = error_stats(q6_reference, fp32_reference)
        print("Q6 weight-only error vs source FP32: RMSE={:.8g}, relative_RMSE={:.6%}, cosine={:.8f}".format(
            q6_vs_fp32["rmse"], q6_vs_fp32["relative_rmse"], q6_vs_fp32["cosine"]
        ))
    finally:
        source.freebuffer()
        destination.freebuffer()


if __name__ == "__main__":
    main()
