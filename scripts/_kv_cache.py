"""Independent KV-cache extension; preserve the completed weight-quantization protocol"""

import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from quantlab.results import ROOT, digest

CONTEXTS = (512, 2048, 8192)
CACHE_BITS = (16, 8, 4)
SEED = 20261007
OUTPUT_TOKENS = 128


def anchors(token_count: int, count: int, *, seed: int = SEED) -> list[int]:
    """Nonoverlapping maximal windows with shared target positions at every context length"""
    stride = max(CONTEXTS) + OUTPUT_TOKENS
    available = token_count // stride
    if available < count:
        raise ValueError("Insufficient tokens for disjoint long-context windows")
    return [i * stride + max(CONTEXTS) for i in random.Random(seed).sample(range(available), count)]


def windows(selected: dict) -> tuple[dict, dict]:
    manifest = json.loads((ROOT / "data" / "splits.json").read_text())[
        selected["context"]["data"]["manifest_id"]
    ]
    streams, splits = {}, {}
    for split, count in (("calibration", 3), ("test", 8)):
        original = manifest["splits"][split]
        cache_id = digest({**manifest["settings"], "source_split": original["source_split"]})[:16]
        path = ROOT / "data" / f"tokens-{cache_id}.npy"
        tokens = np.load(path)
        sha = hashlib.sha256(tokens.tobytes()).hexdigest()
        if sha != original["tokens_sha256"]:
            raise RuntimeError("Token stream differs from the frozen core manifest")
        streams[split] = tokens
        splits[split] = {
            "anchors": anchors(len(tokens), count),
            "tokens_sha256": sha,
            "token_count": len(tokens),
            "source_split": original["source_split"],
        }
    result = {
        "core_manifest_id": selected["context"]["data"]["manifest_id"],
        "core_manifest_sha256": digest(manifest),
        "context_lengths": list(CONTEXTS),
        "output_tokens": OUTPUT_TOKENS,
        "seed": SEED,
        "splits": splits,
    }
    path = ROOT / "data" / "kv-cache-splits.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise RuntimeError("Refusing to overwrite a changed KV-cache split manifest")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return streams, result


def make_cache(model, bits: int) -> list:
    from mlx_lm.models.cache import KVCache, QuantizedKVCache

    if bits not in CACHE_BITS:
        raise ValueError(f"Unsupported KV precision {bits}")
    return [
        KVCache() if bits == 16 else QuantizedKVCache(group_size=64, bits=bits)
        for _ in model.layers
    ]


def prefill(model, prompt: list[int], cache: list, step_size: int = 512):
    """Materialize cached states in bounded chunks; compute logits only for the last token"""
    import mlx.core as mx

    if not prompt:
        raise ValueError("Prompt cannot be empty")
    for start in range(0, len(prompt) - 1, step_size):
        stop = min(start + step_size, len(prompt) - 1)
        model(mx.array(prompt[start:stop])[None], cache=cache)
        mx.eval([c.state for c in cache])
    return model(mx.array(prompt[-1:])[None], cache=cache)[:, -1, :]


def score_continuation(model, prompt: list[int], targets: list[int], bits: int) -> dict:
    import mlx.core as mx
    import mlx.nn as nn

    from quantlab.memory import check_memory, measure_memory

    if not targets:
        raise ValueError("Continuation cannot be empty")
    cache = make_cache(model, bits)
    start = time.perf_counter()
    with measure_memory() as memory:
        logits = prefill(model, prompt, cache)
        nlls = []
        for i, target in enumerate(targets):
            nll = nn.losses.cross_entropy(logits.astype(mx.float32), mx.array([target]))
            nlls.append(float(nll.item()))
            if i + 1 < len(targets):
                logits = model(mx.array([[target]]), cache=cache)[:, -1, :]
        cache_bytes = sum(c.nbytes for c in cache)
    check_memory()
    return {
        "mean_nll": statistics_mean(nlls),
        "perplexity": math.exp(statistics_mean(nlls)),
        "token_nlls": nlls,
        "scored_tokens": len(nlls),
        "cache_bytes": cache_bytes,
        "cache_offset": cache[0].offset,
        "seconds": time.perf_counter() - start,
        **memory,
    }


def statistics_mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


def benchmark_once(model, prompt: list[int], bits: int, generated: int = OUTPUT_TOKENS) -> dict:
    import mlx.core as mx

    from quantlab.memory import check_memory, measure_memory

    if generated < 2:
        raise ValueError("Need at least two generated tokens")
    cache = make_cache(model, bits)
    with measure_memory() as memory:
        start = time.perf_counter()
        logits = prefill(model, prompt, cache)
        token = mx.argmax(logits, axis=-1)[:, None]
        mx.eval(token)
        ttft = time.perf_counter() - start
        del logits
        prefill_cache_bytes = sum(c.nbytes for c in cache)
        tokens = [int(token.item())]
        start = time.perf_counter()
        for _ in range(generated - 1):
            logits = model(token, cache=cache)[:, -1, :]
            token = mx.argmax(logits, axis=-1)[:, None]
            mx.eval(token)
            tokens.append(int(token.item()))
            del logits
        decode_seconds = time.perf_counter() - start
        cache_bytes = sum(c.nbytes for c in cache)
    check_memory()
    return {
        "ttft_seconds": ttft,
        "prefill_tokens_per_second": len(prompt) / ttft,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": (generated - 1) / decode_seconds,
        "end_to_end_seconds": ttft + decode_seconds,
        "prefill_cache_bytes": prefill_cache_bytes,
        "cache_bytes": cache_bytes,
        "cache_offset": cache[0].offset,
        "tokens": tokens,
        **memory,
    }


def extension_hash() -> str:
    paths = [Path(__file__), ROOT / "scripts" / "run_kv_cache.py"]
    return hashlib.sha256(b"".join(p.read_bytes() for p in paths)).hexdigest()
