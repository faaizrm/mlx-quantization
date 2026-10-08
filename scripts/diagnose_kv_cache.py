"""Check native quantized attention against a dequantized-cache reference on calibration"""

import argparse
import hashlib
import json
import math
from pathlib import Path

from quantlab.data import DEFAULT_MODEL
from quantlab.results import ROOT, ResultStore, versions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    import _kv_cache as kv
    import mlx.core as mx
    from mlx_lm.models.base import quantized_scaled_dot_product_attention
    from mlx_lm.models.cache import QuantizedKVCache

    from quantlab.config import QuantConfig
    from quantlab.data import model_snapshot
    from quantlab.model import load_config
    from quantlab.results import experiment_lock

    class DenseReference:
        """Retain quantized storage but route explicitly dequantized K/V through float attention"""

        def __init__(self):
            self.inner = QuantizedKVCache(group_size=64, bits=4)

        @property
        def offset(self):
            return self.inner.offset

        @property
        def state(self):
            return self.inner.state

        @property
        def nbytes(self):
            return self.inner.nbytes

        def make_mask(self, *args, **kwargs):
            return self.inner.make_mask(*args, **kwargs)

        def update_and_fetch(self, keys, values):
            k, v = self.inner.update_and_fetch(keys, values)
            return mx.dequantize(*k, group_size=64, bits=4), mx.dequantize(
                *v, group_size=64, bits=4
            )

    with experiment_lock():
        selected = json.loads((ROOT / "data" / "selected.json").read_text())
        if args.model != selected["context"]["model"]:
            raise RuntimeError("Select this model before diagnostics")
        if versions() != selected["context"]["versions"]:
            raise RuntimeError("Restore the recorded environment")
        streams, manifest = kv.windows(selected)
        snapshot, _ = model_snapshot(args.model)
        model, _ = load_config(snapshot, QuantConfig.from_dict(selected["configs"][0]))
        anchor = manifest["splits"]["calibration"]["anchors"][0]
        prompt = streams["calibration"][anchor - 512 : anchor].tolist()
        targets = streams["calibration"][anchor : anchor + (16 if args.quick else 128)].tolist()
        store = ResultStore(
            ROOT / "results" / "kv-cache-diagnostics.jsonl",
            {
                **selected["context"],
                "protocol": "kv-cache-diagnostic-v1",
                "kv_implementation_sha256": kv.extension_hash(),
                "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
        )
        settings = {
            "prompt_sha256": hashlib.sha256(
                streams["calibration"][anchor - 512 : anchor].tobytes()
            ).hexdigest(),
            "anchor": anchor,
            "scored_tokens": len(targets),
            "quick": args.quick,
        }
        factory = kv.make_cache
        for name, bits in [
            ("native-fp16", 16),
            ("native-int8", 8),
            ("native-int4", 4),
            ("dequantized-int4-reference", 4),
        ]:
            try:
                if name == "dequantized-int4-reference":
                    kv.make_cache = lambda model, bits: [DenseReference() for _ in model.layers]
                result = kv.score_continuation(model, prompt, targets, bits)
                store.add(
                    {"name": name},
                    "continuation_mean_nll",
                    result["mean_nll"],
                    "calibration",
                    settings,
                    sample=result,
                )
                print(name, json.dumps(result), flush=True)
            finally:
                kv.make_cache = factory
        caches = factory(model, 16)
        mx.eval(kv.prefill(model, prompt, caches))
        for layer in (0, 1, 27):
            keys, values = caches[layer].keys_and_values()
            quant_k = mx.quantize(keys, group_size=64, bits=4)
            quant_v = mx.quantize(values, group_size=64, bits=4)
            dense_k = mx.dequantize(*quant_k, group_size=64, bits=4)
            dense_v = mx.dequantize(*quant_v, group_size=64, bits=4)
            mx.random.seed(20261007)
            query = mx.random.normal((1, model.args.num_attention_heads, 1, keys.shape[-1])).astype(
                mx.float16
            )
            # quantized attention scales its query in place; use independent reference storage
            reference_query = mx.array(query.tolist(), dtype=query.dtype)
            native = quantized_scaled_dot_product_attention(
                query,
                quant_k,
                quant_v,
                scale=1 / math.sqrt(keys.shape[-1]),
                mask=None,
                group_size=64,
                bits=4,
            )
            reference = mx.fast.scaled_dot_product_attention(
                reference_query, dense_k, dense_v, scale=1 / math.sqrt(keys.shape[-1]), mask=None
            )

            def relative_rms(a, b):
                return float(
                    mx.sqrt(
                        mx.mean((a.astype(mx.float32) - b.astype(mx.float32)) ** 2)
                        / mx.mean(b.astype(mx.float32) ** 2)
                    ).item()
                )

            errors = {
                "key_quantization_relative_rms": relative_rms(dense_k, keys),
                "value_quantization_relative_rms": relative_rms(dense_v, values),
                "native_vs_dequantized_attention_relative_rms": relative_rms(native, reference),
            }
            store.add(
                {"name": f"layer-{layer}"},
                "attention_relative_rms",
                errors["native_vs_dequantized_attention_relative_rms"],
                "calibration",
                settings,
                errors=errors,
            )
            print("attention check", layer, errors, flush=True)


if __name__ == "__main__":
    main()
