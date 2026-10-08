"""Check continuation alignment, cache storage accounting, and immutable paired windows"""

import runpy

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.cache import KVCache, QuantizedKVCache
from mlx_lm.models.qwen2 import Model, ModelArgs

from quantlab.results import ROOT

extension = runpy.run_path(str(ROOT / "scripts" / "_kv_cache.py"))


def cache_model():
    mx.random.seed(34)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=2,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=1,
        )
    )
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    model.eval()
    return model


def test_continuation_scores_correct_targets_after_prefill():
    model = cache_model()
    prompt, targets = [3, 5, 8, 12], [19, 20, 22]
    result = extension["score_continuation"](model, prompt, targets, 16)
    logits = model(mx.array([prompt + targets[:-1]]))[0, len(prompt) - 1 :, :].astype(mx.float32)
    expected = nn.losses.cross_entropy(logits, mx.array(targets), reduction="none").tolist()
    assert result["token_nlls"] == pytest.approx(expected, abs=0.002)
    assert result["cache_offset"] == len(prompt) + len(targets) - 1
    assert result["scored_tokens"] == len(targets)


@pytest.mark.parametrize("bits", [16, 8, 4])
def test_cache_bytes_include_affine_metadata_and_allocated_capacity(bits):
    cache = KVCache() if bits == 16 else QuantizedKVCache(group_size=64, bits=bits)
    array = mx.zeros((1, 1, 17, 64), dtype=mx.float16)
    cache.update_and_fetch(array, array)
    mx.eval(cache.state)
    assert cache.offset == 17
    expected = 2 * 256 * 64 * (2 if bits == 16 else (bits + 0.5) / 8)
    assert cache.nbytes == expected


@pytest.mark.parametrize("bits", [8, 4])
def test_quantized_cache_runs_full_attention_and_is_deterministic(bits):
    model = cache_model()
    first = extension["score_continuation"](model, [1, 2, 3], [4, 5, 6], bits)
    second = extension["score_continuation"](model, [1, 2, 3], [4, 5, 6], bits)
    assert first["token_nlls"] == second["token_nlls"]
    assert first["cache_offset"] == 5
    assert (
        first["cache_bytes"]
        < extension["score_continuation"](model, [1, 2, 3], [4, 5, 6], 16)["cache_bytes"]
    )


def test_long_context_windows_are_fixed_disjoint_and_target_aligned():
    points = extension["anchors"](8320 * 12, 8)
    assert points == extension["anchors"](8320 * 12, 8)
    assert len(set(points)) == 8
    assert all(b - a >= 8320 for a, b in zip(sorted(points), sorted(points)[1:]))
    assert all(0 <= p - 8192 and p + 128 <= 8320 * 12 for p in points)
    with pytest.raises(ValueError):
        extension["anchors"](8192, 8)
