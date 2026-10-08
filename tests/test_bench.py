"""Validate timing accounting and fresh caches on a tiny model"""

import math

from test_core import tiny_model

from quantlab.bench import benchmark


def test_benchmark_excludes_warmups_and_resets_caches():
    base = tiny_model()

    class ObservedModel:
        layers = base.layers

        def __init__(self):
            self.prompt_offsets = []
            self.calls = 0

        def __call__(self, inputs, cache):
            self.calls += 1
            if inputs.shape[1] > 1:
                self.prompt_offsets.append([entry.offset for entry in cache])
            return base(inputs, cache=cache)

    model = ObservedModel()
    result = benchmark(model, [1, 2, 3], decode_tokens=4, warmups=1, repeats=2)
    assert model.calls == 3 * 4
    assert model.prompt_offsets == [[0, 0]] * 3
    assert result["decode_steps"] == 3
    assert len(result["samples"]) == 2
    for metric in ("prefill_tokens_per_second", "decode_tokens_per_second"):
        assert all(
            math.isfinite(sample[metric]) and sample[metric] > 0 for sample in result["samples"]
        )
        assert result[metric]["min"] <= result[metric]["median"] <= result[metric]["max"]
