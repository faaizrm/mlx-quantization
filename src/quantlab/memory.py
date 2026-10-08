"""MLX allocation peak and sampled process RSS are distinct measurements"""

import threading
from contextlib import contextmanager

import mlx.core as mx
import psutil

from quantlab.model import MEMORY_LIMIT


def check_memory() -> None:
    rss = psutil.Process().memory_info().rss
    if rss > MEMORY_LIMIT:
        raise MemoryError(f"Process RSS {rss:,} exceeds {MEMORY_LIMIT:,} byte budget")


@contextmanager
def measure_memory():
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    process = psutil.Process()
    measurement = {"peak_rss_bytes": process.memory_info().rss}
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.02):
            measurement["peak_rss_bytes"] = max(
                measurement["peak_rss_bytes"], process.memory_info().rss
            )

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield measurement
        mx.synchronize()
    finally:
        stopped.set()
        thread.join()
        measurement["peak_rss_bytes"] = max(
            measurement["peak_rss_bytes"], process.memory_info().rss
        )
        measurement["peak_mlx_bytes"] = mx.get_peak_memory()
    if measurement["peak_rss_bytes"] > MEMORY_LIMIT:
        raise MemoryError(f"Measured RSS exceeded budget: {measurement}")
