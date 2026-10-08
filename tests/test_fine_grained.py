"""Guard exact budgets, calibration-only allocation, and paired loss correctness"""

import itertools
import runpy
import sys

import mlx.core as mx
import numpy as np
import pytest

from quantlab.config import MODULES
from quantlab.eval_ppl import perplexity
from quantlab.results import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from _fine_grained import (  # noqa: E402
    exact_allocation,
    isolated,
    matched_random,
    projection_path,
    promoted,
    score_chunks,
)


def test_exact_knapsack_matches_exhaustive_optimum_and_never_reuses_items():
    costs = {"a": 2, "b": 3, "c": 4, "d": 5, "e": 3}
    scores = {"a": 3.0, "b": -1.0, "c": 3.0, "d": 6.0, "e": 4.0}
    for budget in (5, 7, 8, 10, 17):
        candidates = [
            subset
            for n in range(6)
            for subset in itertools.combinations(costs, n)
            if sum(costs[p] for p in subset) == budget
        ]
        expected = max(sum(max(scores[p], 0) for p in subset) for subset in candidates)
        result = exact_allocation(costs, scores, budget)
        assert len(result) == len(set(result))
        assert sum(costs[p] for p in result) == budget
        assert sum(max(scores[p], 0) for p in result) == expected
        assert result == exact_allocation(dict(reversed(list(costs.items()))), scores, budget)
    with pytest.raises(ValueError, match="exactly"):
        exact_allocation({"x": 4}, {"x": 1.0}, 6)
    with pytest.raises(ValueError, match="finite"):
        exact_allocation({"x": 4}, {"x": float("nan")}, 4)


def test_random_controls_preserve_module_counts_and_actual_cost():
    costs = {projection_path(b, m): (i + 1) * 128 for b in range(28) for i, m in enumerate(MODULES)}
    paths = [projection_path(b, m) for b in (0, 1, 3) for m in MODULES[:5]]
    for seed in (0, 1, 2):
        result = matched_random(paths, costs, seed)
        assert len(set(result)) == len(paths)
        assert sum(costs[p] for p in result) == sum(costs[p] for p in paths)
        for m in MODULES:
            assert sum(p.endswith(m) for p in result) == sum(p.endswith(m) for p in paths)
        assert result == matched_random(paths, costs, seed)


def test_exact_projection_targeting_and_preserved_embeddings():
    path = projection_path(1, "down_proj")
    config = isolated(1, "down_proj")
    assert config.options(path)["bits"] == 4
    assert config.options(projection_path(10, "down_proj")) is False
    assert config.options(projection_path(1, "up_proj")) is False
    allocation = promoted([path], "test")
    assert allocation.options(path)["bits"] == 8
    assert allocation.options(projection_path(1, "up_proj"))["bits"] == 4
    assert allocation.options("model.embed_tokens") is False
    with pytest.raises(ValueError, match="Duplicate"):
        promoted([path, path], "bad")


def test_paired_chunk_losses_match_core_scoring():
    class Model:
        def __call__(self, tokens):
            return mx.broadcast_to(mx.log(mx.array([0.25, 0.75])), (*tokens.shape, 2))

    chunks = [np.array([0, 1, 1]), np.array([1, 0, 1])]
    result = score_chunks(Model(), chunks)
    reference = perplexity(Model(), chunks)
    assert result["mean_nll"] == reference["mean_nll"]
    assert result["perplexity"] == reference["perplexity"]
    assert result["tokens"] == reference["tokens"] == 4
    assert np.mean(result["per_chunk_mean_nll"]) == result["mean_nll"]


def test_frozen_selection_rejects_changes_and_preserves_original(tmp_path):
    freeze = runpy.run_path(str(ROOT / "scripts" / "run_fine_grained.py"))["freeze"]
    path = tmp_path / "selection.json"
    value = {"selection_split": "calibration", "allocations": ["a"]}
    freeze(path, value)
    before = path.read_bytes()
    freeze(path, value)
    with pytest.raises(RuntimeError, match="Frozen manifest mismatch"):
        freeze(path, {**value, "allocations": ["b"]})
    assert path.read_bytes() == before
