"""Load every configuration from the same original fp16 checkpoint"""

import gc
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm import load

from quantlab.config import QuantConfig

MEMORY_LIMIT = 10_000_000_000


def configure_memory() -> None:
    # leave room for Python, tokenization, and other apps outside the Metal allocator
    mx.set_memory_limit(8_000_000_000)
    mx.set_cache_limit(256_000_000)


def apply_config(model: nn.Module, config: QuantConfig) -> nn.Module:
    if any(isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules()):
        raise ValueError("Refusing stacked quantization; supply a fresh fp16 model")
    nn.quantize(
        model,
        class_predicate=lambda path, module: (
            config.options(path) if isinstance(module, nn.Linear) else False
        ),
    )
    mx.eval(model.parameters())
    return model


def load_config(snapshot: Path, config: QuantConfig):
    configure_memory()
    model, tokenizer = load(str(snapshot), lazy=True)
    if not model.args.tie_word_embeddings:
        raise ValueError(
            "This protocol requires tied embeddings; define a new protocol for other models"
        )
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    apply_config(model, config)
    model.eval()
    return model, tokenizer


def parameter_bytes(model: nn.Module) -> int:
    seen = set()
    total = 0
    for _, array in tree_flatten(model.parameters()):
        if id(array) not in seen:
            seen.add(id(array))
            total += array.nbytes
    return total


def clear_memory() -> None:
    gc.collect()
    mx.clear_cache()
