"""Calibration-only, exact-storage projection allocation and paired loss evaluation"""

import hashlib
import math
import random
import time
from functools import reduce

from quantlab.config import MODULES, QuantConfig, Rule
from quantlab.results import ROOT

SEED = 20261007
BUDGET_BLOCKS = (8, 16)


def projection_path(block: int, module: str) -> str:
    if block < 0 or module not in MODULES:
        raise ValueError("Invalid block/projection")
    kind = "self_attn" if module in MODULES[:4] else "mlp"
    return f"model.layers.{block}.{kind}.{module}"


def isolated(block: int, module: str) -> QuantConfig:
    return QuantConfig(f"pair-{block:02d}-{module}", (Rule(projection_path(block, module), 4),))


def promoted(paths: list[str], name: str) -> QuantConfig:
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate projection")
    return QuantConfig(
        name, (Rule("model.layers.*", 4),) + tuple(Rule(p, 8) for p in sorted(paths))
    )


def exact_allocation(costs: dict[str, int], scores: dict[str, float], budget: int) -> list[str]:
    """Maximize the positive calibration proxy while spending exactly the byte budget"""
    if not costs or set(costs) != set(scores):
        raise ValueError("Every projection needs an exact cost and calibration score")
    if budget <= 0 or any(not isinstance(c, int) or c <= 0 for c in costs.values()):
        raise ValueError("Costs and budget must be positive integers")
    if any(not math.isfinite(v) for v in scores.values()):
        raise ValueError("Scores must be finite")
    unit = reduce(math.gcd, costs.values())
    if budget % unit:
        raise ValueError("Budget cannot be filled exactly")
    capacity = budget // unit
    best = [None] * (capacity + 1)
    best[0] = (0.0, ())
    for path in sorted(costs):
        cost, benefit = costs[path] // unit, max(0.0, scores[path])
        # descending capacities make this 0/1, never unbounded knapsack
        for available in range(capacity, cost - 1, -1):
            previous = best[available - cost]
            if previous is None:
                continue
            value = previous[0] + benefit
            if best[available] is None or value > best[available][0]:
                best[available] = (value, (*previous[1], path))
    if best[capacity] is None:
        raise ValueError("Budget cannot be filled exactly")
    return list(best[capacity][1])


def matched_random(paths: list[str], costs: dict[str, int], seed: int) -> list[str]:
    """Shuffle block identities within module types, preserving storage exactly"""
    rng = random.Random(seed)
    chosen = []
    for module in MODULES:
        candidates = sorted(p for p in costs if p.endswith("." + module))
        count = sum(p.endswith("." + module) for p in paths)
        if candidates and len({costs[p] for p in candidates}) != 1:
            raise ValueError("Module-matched controls require equal costs across blocks")
        chosen.extend(rng.sample(candidates, count))
    if sum(costs[p] for p in chosen) != sum(costs[p] for p in paths):
        raise ValueError("Control storage mismatch")
    return sorted(chosen)


def extension_hash() -> str:
    names = ("_fine_grained.py", "run_fine_grained.py", "_kv_cache.py")
    return hashlib.sha256(b"".join((ROOT / "scripts" / p).read_bytes() for p in names)).hexdigest()


def score_chunks(model, chunks: list) -> dict:
    """Match core float32 token NLL, retaining paired chunk losses for uncertainty"""
    import mlx.core as mx
    import mlx.nn as nn

    from quantlab.memory import check_memory

    if not chunks or len({len(c) for c in chunks}) != 1 or len(chunks[0]) < 2:
        raise ValueError("Need nonempty, equal-length chunks with targets")
    start = time.perf_counter()
    losses = []
    target_count = len(chunks[0]) - 1
    for i, chunk in enumerate(chunks):
        tokens = mx.array(chunk)[None, :]
        logits = model(tokens[:, :-1]).astype(mx.float32)
        loss = float(nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="sum").item())
        del tokens, logits
        if not math.isfinite(loss):
            raise RuntimeError("Nonfinite loss; investigate before continuing")
        losses.append(loss / target_count)
        check_memory()
        if (i + 1) % 32 == 0:
            print(f"  quality {i + 1}/{len(chunks)}", flush=True)
    mean = sum(losses) / len(losses)
    return {
        "perplexity": math.exp(mean),
        "mean_nll": mean,
        "per_chunk_mean_nll": losses,
        "chunks": len(chunks),
        "tokens": len(chunks) * target_count,
        "seconds": time.perf_counter() - start,
    }
