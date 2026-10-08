import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models.qwen2 import Model, ModelArgs

from quantlab.config import QuantConfig, layer_only, mixed, module_only, uniform
from quantlab.data import chunk_indices
from quantlab.eval_arc import continuation_score
from quantlab.eval_ppl import perplexity
from quantlab.model import apply_config, parameter_bytes
from quantlab.results import ResultStore, read_records
from quantlab.runner import pareto


def tiny_model():
    mx.random.seed(42)
    model = Model(
        ModelArgs(
            model_type="qwen2",
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
        )
    )
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    model.eval()
    return model


def test_predicates_preserve_embeddings_and_override_blocks():
    config = mixed([1], "mixed-test")
    assert config.options("model.embed_tokens") is False
    assert config.options("lm_head") is False
    assert config.options("model.layers.0.self_attn.q_proj")["bits"] == 4
    assert config.options("model.layers.1.self_attn.q_proj")["bits"] == 8
    assert config.options("model.layers.1.input_layernorm") is False
    assert layer_only(1).options("model.layers.10.mlp.up_proj") is False
    assert module_only("q_proj").options("model.layers.2.self_attn.k_proj") is False
    assert QuantConfig.from_dict(config.to_dict()) == config


def test_size_includes_quantization_metadata_and_linear_bias():
    linear = nn.Linear(64, 32, bias=True)
    linear.set_dtype(mx.float16)
    assert parameter_bytes(linear) == (64 * 32 + 32) * 2
    quantized = linear.to_quantized(group_size=32, bits=4)
    # packed weights + two fp16 group arrays + the original fp16 linear bias
    assert parameter_bytes(quantized) == 64 * 32 // 2 + 2 * (64 // 32) * 32 * 2 + 32 * 2


def test_perplexity_determinism():
    model = tiny_model()
    chunks = [np.arange(17, dtype=np.int32), np.arange(3, 20, dtype=np.int32)]
    first = perplexity(model, chunks)
    second = perplexity(model, chunks)
    assert first["perplexity"] == second["perplexity"]
    assert first["tokens"] == 32


def test_ppl_has_correct_shift_and_token_weighting():
    class ConstantModel:
        def __call__(self, inputs):
            return mx.broadcast_to(mx.log(mx.array([0.25, 0.75])), (*inputs.shape, 2))

    result = perplexity(ConstantModel(), [[0, 1, 1], [1, 0]])
    expected = math.exp(-(2 * math.log(0.75) + math.log(0.25)) / 3)
    assert result["perplexity"] == pytest.approx(expected)


def test_rebuilding_quantization_is_identical_and_stacking_rejected():
    first = apply_config(tiny_model(), uniform(4))
    second = apply_config(tiny_model(), uniform(4))
    a, b = tree_flatten(first.parameters()), tree_flatten(second.parameters())
    assert [name for name, _ in a] == [name for name, _ in b]
    for (_, x), (_, y) in zip(a, b, strict=True):
        assert bool(mx.array_equal(x, y).item())
    assert first.model.embed_tokens.weight.dtype == mx.float16
    with pytest.raises(ValueError, match="stacked"):
        apply_config(first, uniform(8))


def test_chunks_deterministic_unique_and_fail_if_insufficient():
    values = chunk_indices(1000, 16, 20, 42)
    assert values == chunk_indices(1000, 16, 20, 42)
    assert len(set(values)) == 20
    assert all(i * 16 + 17 <= 1000 for i in values)
    with pytest.raises(ValueError):
        chunk_indices(100, 16, 20, 42)


def test_resume_identity_distinguishes_model_and_data(tmp_path):
    path = tmp_path / "results.jsonl"
    config = uniform(4).to_dict()
    store = ResultStore(path, {"model": "tiny", "data": "a"})
    store.add(config, "perplexity", 5.0, "calibration", {"chunks": 2})
    store.add(config, "perplexity", 5.0, "calibration", {"chunks": 2})
    assert len(read_records(path)) == 1
    assert store.has(config, "perplexity", "calibration", {"chunks": 2})
    assert not store.has(config, "perplexity", "test", {"chunks": 2})
    assert not ResultStore(path, {"model": "tiny", "data": "b"}).has(
        config, "perplexity", "calibration", {"chunks": 2}
    )


def test_corrupted_results_fail_visibly(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"partial":')
    with pytest.raises(ValueError, match="Invalid JSONL"):
        read_records(path)


def test_pareto_retains_ties_without_using_test_scores():
    rows = [
        {"bytes": 10, "ppl": 10},
        {"bytes": 20, "ppl": 5},
        {"bytes": 20, "ppl": 10},
        {"bytes": 10, "ppl": 10},
    ]
    assert pareto(rows) == [rows[0], rows[1], rows[3]]


def test_arc_scores_answer_only_and_normalizes_length():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return [0, 0, 0] if text == "prompt" else [1, 1]

    class ConstantModel:
        def __call__(self, inputs):
            return mx.broadcast_to(mx.log(mx.array([0.25, 0.75])), (*inputs.shape, 2))

    assert continuation_score(ConstantModel(), Tokenizer(), "prompt", "answer") == pytest.approx(
        math.log(0.75)
    )
