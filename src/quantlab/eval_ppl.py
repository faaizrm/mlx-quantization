"""Teacher-forced token NLL in float32; no cache reuse between chunks"""

import math
import time

import mlx.core as mx
import mlx.nn as nn

from quantlab.memory import check_memory


def perplexity(model, chunks: list) -> dict:
    if not chunks:
        raise ValueError("Perplexity requires at least one chunk")
    total_nll = 0.0
    total_tokens = 0
    start = time.perf_counter()
    for index, chunk in enumerate(chunks):
        if len(chunk) < 2:
            raise ValueError("A chunk needs an input and a target")
        tokens = mx.array(chunk)[None, :]
        logits = model(tokens[:, :-1]).astype(mx.float32)
        nll = nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="sum")
        total_nll += float(nll.item())
        total_tokens += len(chunk) - 1
        del tokens, logits, nll
        check_memory()
        if (index + 1) % 16 == 0:
            print(
                f"  PPL {index + 1}/{len(chunks)} chunks ({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    mean_nll = total_nll / total_tokens
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "tokens": total_tokens,
        "chunks": len(chunks),
        "seconds": time.perf_counter() - start,
    }
