"""Isolated prefill and greedy decode timings with fresh KV caches"""

import statistics
import time

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from quantlab.memory import check_memory, measure_memory


def benchmark(
    model, prompt: list[int], *, decode_tokens: int = 128, warmups: int = 2, repeats: int = 5
) -> dict:
    if decode_tokens < 2 or repeats < 1 or warmups < 0:
        raise ValueError("Invalid benchmark settings")
    samples = []
    for i in range(warmups + repeats):
        cache = make_prompt_cache(model)
        inputs = mx.array(prompt)[None, :]
        mx.eval(inputs)
        with measure_memory() as memory:
            start = time.perf_counter()
            logits = model(inputs, cache=cache)
            token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
            mx.eval(token)
            prefill_seconds = time.perf_counter() - start
            del logits
            start = time.perf_counter()
            # prefill yields the first generated token; remaining tokens are decode steps
            for _ in range(decode_tokens - 1):
                logits = model(token, cache=cache)
                token = mx.argmax(logits[:, -1, :], axis=-1)[:, None]
                mx.eval(token)
                del logits
            decode_seconds = time.perf_counter() - start
        if i >= warmups:
            samples.append(
                {
                    "prefill_tokens_per_second": len(prompt) / prefill_seconds,
                    "decode_tokens_per_second": (decode_tokens - 1) / decode_seconds,
                    **memory,
                }
            )
        del token, cache, inputs
        check_memory()
    result = {
        "samples": samples,
        "decode_steps": decode_tokens - 1,
    }
    for metric in samples[0]:
        values = [s[metric] for s in samples]
        result[metric] = {
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
        }
    return result
