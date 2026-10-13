import time

import numpy as np
from pynq import Overlay, allocate


INPUT_PATH = "input.bin"
EXPECTED_PATH = "expected.bin"
OUTPUTS = 352
RUNS = 20

overlay = Overlay("q6w_gemv.bit")
dma = overlay.axi_dma_0
input_data = open(INPUT_PATH, "rb").read()
expected = np.fromfile(EXPECTED_PATH, dtype="<i4")
if expected.size != OUTPUTS:
    raise ValueError("expected.bin must contain {} int32 values".format(OUTPUTS))

source = allocate(shape=(len(input_data),), dtype=np.uint8)
destination = allocate(shape=(OUTPUTS,), dtype=np.int32)
source[:] = np.frombuffer(input_data, dtype=np.uint8)

def run_once():
    dma.recvchannel.transfer(destination)
    dma.sendchannel.transfer(source)
    dma.sendchannel.wait()
    dma.recvchannel.wait()


def check_result():
    actual = np.asarray(destination, dtype=np.int32)
    if not np.array_equal(actual, expected):
        mismatch = int(np.flatnonzero(actual != expected)[0])
        raise AssertionError(
            "row {} expected {}, got {}".format(
                mismatch, int(expected[mismatch]), int(actual[mismatch])
            )
        )


run_once()
check_result()
times = []
for _ in range(RUNS):
    start = time.perf_counter()
    run_once()
    times.append(time.perf_counter() - start)
    check_result()

median = float(np.median(times))
minimum = min(times)
maximum = max(times)

print("PASS: {} DMA runs matched all {} exact outputs".format(RUNS, OUTPUTS))
print("Input bytes: {}; output bytes: {}".format(len(input_data), destination.nbytes))
print("DMA wall time: median {:.6f} s, min {:.6f} s, max {:.6f} s".format(
    median, minimum, maximum
))
print("Median input throughput: {:.2f} MiB/s".format(
    len(input_data) / median / (1024.0 * 1024.0)
))
print("Clock: 100 MHz; matrix rows/s (not tokens/s): {:.1f}".format(
    OUTPUTS / median
))

source.freebuffer()
destination.freebuffer()
