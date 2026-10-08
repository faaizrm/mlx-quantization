"""Measure long-context cache storage, continuation quality, and shuffled speed repeats"""

import argparse
import hashlib
import json
import math
import random

from _kv_cache import (
    CACHE_BITS,
    CONTEXTS,
    OUTPUT_TOKENS,
    SEED,
    benchmark_once,
    extension_hash,
    score_continuation,
    windows,
)

from quantlab.config import QuantConfig
from quantlab.data import DEFAULT_MODEL, model_snapshot, wikitext_chunks
from quantlab.results import (
    ROOT,
    ResultStore,
    experiment_lock,
    hardware_metadata,
    read_records,
    versions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-speed", action="store_true")
    args = parser.parse_args()
    with experiment_lock():
        run(args)


def run(args) -> None:
    from transformers import AutoTokenizer

    from quantlab.memory import measure_memory
    from quantlab.model import clear_memory, load_config, parameter_bytes

    selected = json.loads((ROOT / "data" / "selected.json").read_text())
    source = b"".join(p.read_bytes() for p in sorted((ROOT / "src" / "quantlab").glob("*.py")))
    if selected["selection_split"] != "calibration" or selected["context"]["model"] != args.model:
        raise RuntimeError("Need a calibration-selected configuration for this model")
    if hashlib.sha256(source).hexdigest() != selected["context"]["implementation_sha256"]:
        raise RuntimeError("Core implementation changed since model selection")
    if versions() != selected["context"]["versions"]:
        raise RuntimeError("Restore pinned core library versions before extending this study")
    snapshot, revision = model_snapshot(args.model)
    if revision != selected["context"]["model_revision"]:
        raise RuntimeError("Model revision changed")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
    _, core_data = wikitext_chunks(tokenizer, args.model)
    if core_data != selected["context"]["data"]:
        raise RuntimeError("Core data changed")
    streams, manifest = windows(selected)
    context = {
        **selected["context"],
        "versions": versions(),
        "protocol": "quantlab-kv-cache-v1",
        "hardware": hardware_metadata(),
        "kv_implementation_sha256": extension_hash(),
        "kv_data": manifest,
    }
    path = ROOT / "results" / ("kv-cache-smoke.jsonl" if args.quick else "kv-cache.jsonl")
    store = ResultStore(path, context)
    lengths = (CONTEXTS[0], CONTEXTS[-1]) if args.quick else CONTEXTS
    precisions = (16, 4) if args.quick else CACHE_BITS
    count, generated = (2, 16) if args.quick else (8, OUTPUT_TOKENS)
    speed_prompts, warmups, repeats = (1, 1, 2) if args.quick else (3, 2, 5)
    target_cfg = QuantConfig.from_dict(selected["configs"][0])

    def config(bits):
        return {
            "name": f"kv{bits}",
            "cache_bits": bits,
            "group_size": 64 if bits != 16 else None,
            "quantize_from_token": 0,
            "target": target_cfg.to_dict(),
        }

    def settings(length, index, split):
        anchor = manifest["splits"][split]["anchors"][index]
        return {
            "quick": args.quick,
            "context_tokens": length,
            "window_index": index,
            "anchor": anchor,
            "output_tokens": generated,
            "prefill_step_size": 512,
            "prompt_sha256": hashlib.sha256(
                streams[split][anchor - length : anchor].tobytes()
            ).hexdigest(),
            "target_sha256": hashlib.sha256(
                streams[split][anchor : anchor + generated].tobytes()
            ).hexdigest(),
        }

    if not args.quick:
        pilots = read_records(ROOT / "results" / "kv-cache-smoke.jsonl")
        bench = [
            r
            for r in pilots
            if r["metric"] == "kv_benchmark" and all(r.get(k) == v for k, v in context.items())
        ]
        quality = [
            r
            for r in pilots
            if r["metric"] == "continuation_mean_nll"
            and all(r.get(k) == v for k, v in context.items())
        ]
        if len(quality) != 8 or (not args.no_speed and len(bench) != 8):
            raise RuntimeError("Complete the matching --quick pilot for the requested stages")
        quality_time = max(r["sample"]["seconds"] for r in quality) * generated / 16
        estimate = 9 * count * quality_time / 60
        if not args.no_speed:
            worst = max(r["sample"]["ttft_seconds"] + 127 / r["value"] for r in bench)
            estimate += 9 * speed_prompts * (warmups + repeats) * worst / 60
        print(
            f"Conservative pilot runtime estimate: {estimate:.1f} minutes before resume skips",
            flush=True,
        )
    with measure_memory() as load_memory:
        model, _ = load_config(snapshot, target_cfg)
    print(f"Target {target_cfg.name}; parameter bytes {parameter_bytes(model)}", flush=True)
    try:
        for length in lengths:
            for bits in precisions:
                for index in range(count):
                    opts = settings(length, index, "test")
                    opts["scoring"] = "sequential-teacher-forced-float32-cross-entropy"
                    if store.has(config(bits), "continuation_mean_nll", "test", opts):
                        continue
                    anchor = opts["anchor"]
                    result = score_continuation(
                        model,
                        streams["test"][anchor - length : anchor].tolist(),
                        streams["test"][anchor : anchor + generated].tolist(),
                        bits,
                    )
                    if not math.isfinite(result["mean_nll"]) or result["perplexity"] < 1:
                        raise RuntimeError("Suspicious continuation loss; investigate")
                    # severe int4 loss was reproduced with a dequantized reference on calibration
                    # preserve this negative outcome; do not censor or change the fixed grid
                    result["high_loss_warning"] = result["perplexity"] >= 1000
                    store.add(
                        config(bits),
                        "continuation_mean_nll",
                        result["mean_nll"],
                        "test",
                        opts,
                        sample=result,
                        load_memory=load_memory,
                    )
                    print(
                        f"QUALITY L={length} kv{bits} window={index}: "
                        f"PPL {result['perplexity']:.3f}",
                        flush=True,
                    )
        if args.no_speed:
            return
        jobs = [
            (length, bits, index)
            for length in lengths
            for bits in precisions
            for index in range(speed_prompts)
        ]
        measured_jobs = [(job, repeat) for repeat in range(repeats) for job in jobs]

        def opts_for(job, repeat):
            length, bits, index = job
            return {
                **settings(length, index, "calibration"),
                "repeat": repeat,
                "warmups": warmups,
                "repeats": repeats,
                "order_seed": SEED,
                "stop_at_eos": False,
                "timing": "chunked-prefill-first-output-then-generated-minus-one-decode-steps",
            }

        if all(
            store.has(config(job[1]), "kv_benchmark", "benchmark", opts_for(job, repeat))
            for job, repeat in measured_jobs
        ):
            print("All benchmark samples already recorded", flush=True)
            return
        references = {}
        for row in store.records:
            if (
                row["metric"] == "kv_benchmark"
                and all(row.get(k) == v for k, v in context.items())
                and row["settings"]["quick"] == args.quick
            ):
                references[
                    (
                        row["settings"]["context_tokens"],
                        row["config"]["cache_bits"],
                        row["settings"]["window_index"],
                    )
                ] = row["sample"]["tokens"]
        for repeat in range(-warmups, repeats):
            order = jobs.copy()
            random.Random(SEED + repeat + warmups).shuffle(order)
            for position, job in enumerate(order):
                length, bits, index = job
                opts = opts_for(job, repeat)
                if repeat >= 0 and store.has(config(bits), "kv_benchmark", "benchmark", opts):
                    continue
                anchor = opts["anchor"]
                result = benchmark_once(
                    model,
                    streams["calibration"][anchor - length : anchor].tolist(),
                    bits,
                    generated,
                )
                if job in references and references[job] != result["tokens"]:
                    raise RuntimeError(
                        f"Nondeterministic greedy output within cache configuration {job}"
                    )
                references[job] = result["tokens"]
                if repeat >= 0:
                    store.add(
                        config(bits),
                        "kv_benchmark",
                        result["decode_tokens_per_second"],
                        "benchmark",
                        opts,
                        sample=result,
                        order_position=position,
                    )
                print(
                    f"BENCH repeat={repeat} L={length} kv{bits} prompt={index}: "
                    f"{result['decode_tokens_per_second']:.2f} tok/s",
                    flush=True,
                )
        print(f"Complete: {path}", flush=True)
    finally:
        del model
        clear_memory()


if __name__ == "__main__":
    main()
