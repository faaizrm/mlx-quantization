"""Measure projection sensitivity, exact-budget allocations, and decode performance"""

import argparse
import hashlib
import json
import random
import time

from _fine_grained import (
    BUDGET_BLOCKS,
    SEED,
    exact_allocation,
    extension_hash,
    isolated,
    matched_random,
    projection_path,
    promoted,
    score_chunks,
)
from _kv_cache import benchmark_once

from quantlab.config import MODULES, QuantConfig, mixed, uniform
from quantlab.data import DEFAULT_MODEL, model_snapshot, wikitext_chunks
from quantlab.results import (
    ROOT,
    ResultStore,
    digest,
    experiment_lock,
    hardware_metadata,
    read_records,
    versions,
)


def freeze(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise RuntimeError(f"Frozen manifest mismatch: {path}; investigate, do not overwrite")
    else:
        with path.open("x") as f:
            f.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--stage", choices=("all", "map", "quality", "speed"), default="all")
    args = parser.parse_args()
    with experiment_lock():
        run(args)


def run(args):
    import mlx.nn as nn
    from transformers import AutoTokenizer

    from quantlab.memory import measure_memory
    from quantlab.model import clear_memory, load_config, parameter_bytes

    selected = json.loads((ROOT / "data" / "selected.json").read_text())
    source = b"".join(p.read_bytes() for p in sorted((ROOT / "src" / "quantlab").glob("*.py")))
    if selected["selection_split"] != "calibration" or args.model != selected["context"]["model"]:
        raise RuntimeError("Need the original model's calibration-selected core study")
    if hashlib.sha256(source).hexdigest() != selected["context"]["implementation_sha256"]:
        raise RuntimeError("Original scientific source changed")
    if versions() != selected["context"]["versions"]:
        raise RuntimeError("Restore pinned versions")
    snapshot, revision = model_snapshot(args.model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
    chunks, data = wikitext_chunks(tokenizer, args.model)
    if revision != selected["context"]["model_revision"] or data != selected["context"]["data"]:
        raise RuntimeError("Core model/data identity changed")
    metadata = json.loads((snapshot / "config.json").read_text())
    if metadata["num_hidden_layers"] != 28:
        raise RuntimeError("This extension is prespecified for 28 blocks")
    manifest = {
        "core_data": data,
        "calibration_positions": list(range(32)),
        "calibration_subset_sha256": hashlib.sha256(
            b"".join(c.tobytes() for c in chunks["calibration"][:32])
        ).hexdigest(),
        "selection_split": "calibration",
        "budget_blocks": list(BUDGET_BLOCKS),
        "random_seeds": [0, 1, 2],
    }
    freeze(ROOT / "data" / "fine-grained-splits.json", manifest)
    context = {
        **selected["context"],
        "protocol": "quantlab-fine-grained-v1",
        "hardware": hardware_metadata(),
        "fine_implementation_sha256": extension_hash(),
        "fine_data": manifest,
        "quick": args.quick,
    }
    path = ROOT / "results" / ("fine-grained-smoke.jsonl" if args.quick else "fine-grained.jsonl")
    store = ResultStore(path, context)
    marker = QuantConfig("run").to_dict()

    def row(config, metric, split, settings):
        key = store.key(config.to_dict(), metric, split, settings)
        return next((r for r in store.records if r["key"] == key), None)

    if not args.quick:
        pilot_context = {**store.context, "quick": True}
        pilots = [
            r
            for r in read_records(ROOT / "results" / "fine-grained-smoke.jsonl")
            if all(r.get(k) == v for k, v in pilot_context.items())
        ]
        if not any(r["metric"] == "pilot_complete" for r in pilots):
            raise RuntimeError("Complete the matching --quick pilot first")
        seconds = max(
            r["sample"]["seconds"] / r["sample"]["chunks"]
            for r in pilots
            if r["metric"] == "fine_quality"
        )
        loads = max(r.get("load_seconds", 0) for r in pilots)
        estimate = (seconds * (197 * 32 + 14 * 128 + 13 * 256) + loads * 220) / 60
        print(
            f"Pilot-derived quality/load estimate: {estimate:.1f} minutes "
            "before resume skips; timing extra",
            flush=True,
        )

    catalog_cfg = QuantConfig("catalog")
    catalog_row = row(catalog_cfg, "projection_catalog", "none", {})
    if catalog_row is None:
        catalogs = {}
        for bits in (4, 8):
            with measure_memory() as memory:
                model, _ = load_config(snapshot, uniform(bits))
                catalogs[str(bits)] = {
                    "parameter_bytes": parameter_bytes(model),
                    "projections": {
                        p: parameter_bytes(m)
                        for p, m in model.named_modules()
                        if isinstance(m, nn.QuantizedLinear)
                    },
                    "memory": memory,
                }
            del model
            clear_memory()
        a, b = catalogs["4"], catalogs["8"]
        expected = {projection_path(i, m) for i in range(28) for m in MODULES}
        if set(a["projections"]) != expected or set(b["projections"]) != expected:
            raise RuntimeError("Unexpected quantized module inventory")
        costs = {p: b["projections"][p] - a["projections"][p] for p in sorted(expected)}
        if sum(costs.values()) != b["parameter_bytes"] - a["parameter_bytes"]:
            raise RuntimeError("Promotion costs do not account for model storage")
        catalog_row = store.add(
            catalog_cfg.to_dict(),
            "projection_catalog",
            196,
            "none",
            {},
            catalogs=catalogs,
            costs=costs,
        )
    costs = catalog_row["costs"]
    base_bytes = catalog_row["catalogs"]["4"]["parameter_bytes"]

    def expected_bytes(config):
        if not config.rules or config.name.startswith("pair-"):
            return None
        return base_bytes + sum(costs[p] for p in costs if config.options(p)["bits"] == 8)

    def evaluate(config, split, count):
        settings = {
            "chunk_count": count,
            "chunk_size": 512,
            "scoring": "next-token-nll-float32",
            "positions": "manifest-prefix",
        }
        previous = row(config, "fine_quality", split, settings)
        if previous is not None:
            return previous
        print(f"RUN {config.name} {split} {count}", flush=True)
        start = time.perf_counter()
        with measure_memory() as load_memory:
            model, _ = load_config(snapshot, config)
        load_seconds = time.perf_counter() - start
        try:
            size = parameter_bytes(model)
            expected = expected_bytes(config)
            if expected is not None and size != expected:
                raise RuntimeError(f"Actual size {size} != catalog prediction {expected}")
            with measure_memory() as memory:
                sample = score_chunks(model, chunks[split][:count])
            result = store.add(
                config.to_dict(),
                "fine_quality",
                sample["perplexity"],
                split,
                settings,
                sample=sample,
                parameter_bytes=size,
                memory=memory,
                load_memory=load_memory,
                load_seconds=load_seconds,
            )
            print(f"DONE {config.name} PPL={sample['perplexity']:.6f} bytes={size}", flush=True)
            return result
        finally:
            del model
            clear_memory()

    map_count = 2 if args.quick else 32
    pairs = (
        [(1, "q_proj"), (1, "down_proj")]
        if args.quick
        else [(i, m) for i in range(28) for m in MODULES]
    )
    random.Random(SEED).shuffle(pairs)
    if args.stage in ("all", "map"):
        baseline = evaluate(QuantConfig("fp16"), "calibration", map_count)
        for block, module in pairs:
            result = evaluate(isolated(block, module), "calibration", map_count)
            if result["value"] > 10 * baseline["value"]:
                raise RuntimeError("Large sensitivity loss; investigate recorded result")
        if args.stage == "map":
            return

    if args.quick:
        configs = [uniform(4), promoted([projection_path(1, "down_proj")], "projection-pilot")]
        speed_configs = configs
    else:
        map_settings = {
            "chunk_count": 32,
            "chunk_size": 512,
            "scoring": "next-token-nll-float32",
            "positions": "manifest-prefix",
        }
        baseline = row(QuantConfig("fp16"), "fine_quality", "calibration", map_settings)
        values = {
            projection_path(i, m): row(isolated(i, m), "fine_quality", "calibration", map_settings)
            for i, m in pairs
        }
        if baseline is None or any(r is None for r in values.values()):
            raise RuntimeError("Complete all 196 sensitivity pairs before allocating")
        scores = {
            p: r["sample"]["mean_nll"] - baseline["sample"]["mean_nll"] for p, r in values.items()
        }
        # investigate apparent improvements on full calibration before proceeding
        negative = min(scores, key=lambda p: (scores[p], p))
        if scores[negative] < 0 and args.stage in ("all", "quality"):
            reference = evaluate(QuantConfig("fp16"), "calibration", 128)
            reviewed = evaluate(
                QuantConfig.from_dict(values[negative]["config"]), "calibration", 128
            )
            full_delta = reviewed["sample"]["mean_nll"] - reference["sample"]["mean_nll"]
            print(
                f"NEGATIVE-DELTA REVIEW {negative}: subset delta={scores[negative]:.8f}; "
                f"full-cal delta={full_delta:.8f}",
                flush=True,
            )
        allocations = []
        for k in BUDGET_BLOCKS:
            block_cfg = mixed(selected["ranking"][:k], f"fine-block-{k}")
            budget = sum(costs[p] for p in costs if block_cfg.options(p)["bits"] == 8)
            paths = exact_allocation(costs, scores, budget)
            group = [promoted(paths, f"fine-projection-{k}"), block_cfg]
            group += [
                promoted(matched_random(paths, costs, seed), f"fine-random-{k}-seed-{seed}")
                for seed in (0, 1, 2)
            ]
            for config in group:
                if expected_bytes(config) != base_bytes + budget:
                    raise RuntimeError("Allocation budgets are not equal")
                allocations.append(
                    {
                        "config": config.to_dict(),
                        "budget_blocks": k,
                        "parameter_bytes": base_bytes + budget,
                    }
                )
        frozen = {
            "context": store.context,
            "selection_split": "calibration",
            "calibration_record_keys": sorted(r["key"] for r in values.values())
            + [baseline["key"]],
            "catalog_key": catalog_row["key"],
            "core_selection_sha256": digest(selected),
            "allocations": allocations,
        }
        freeze(ROOT / "data" / "fine-grained-selection.json", frozen)
        configs = [QuantConfig("fp16"), uniform(4), uniform(8)] + [
            QuantConfig.from_dict(a["config"]) for a in allocations
        ]
        speed_configs = [c for c in configs if c.name != "fp16" and "random" not in c.name]

    if args.stage in ("all", "quality"):
        # finish calibration for every frozen candidate before opening the test split
        for split, count in (("calibration", 128), ("test", 256)):
            for config in configs:
                evaluate(config, split, 2 if args.quick else count)
        if args.stage == "quality":
            return

    if args.stage in ("all", "speed"):
        prompt_count, warmups, repeats, generated = (1, 1, 2, 16) if args.quick else (3, 2, 5, 128)

        random.Random(SEED).shuffle(speed_configs)
        for config in speed_configs:

            def settings(prompt_id, repeat):
                return {
                    "prompt_id": prompt_id,
                    "prompt_sha256": hashlib.sha256(
                        chunks["calibration"][prompt_id][:-1].tobytes()
                    ).hexdigest(),
                    "prompt_tokens": 512,
                    "generated_tokens": generated,
                    "cache_bits": 16,
                    "warmups": warmups,
                    "repeat": repeat,
                    "repeats": repeats,
                }

            cases = [(p, r) for p in range(prompt_count) for r in range(repeats)]
            pending = [
                (p, r)
                for p, r in cases
                if row(config, "fine_benchmark", "benchmark", settings(p, r)) is None
            ]
            if not pending:
                continue
            with measure_memory():
                model, _ = load_config(snapshot, config)
            try:
                for prompt_id in sorted({p for p, _ in pending}):
                    for _ in range(warmups):
                        benchmark_once(
                            model, chunks["calibration"][prompt_id][:-1].tolist(), 16, generated
                        )
                random.Random(SEED).shuffle(pending)
                for prompt_id, repeat in pending:
                    sample = benchmark_once(
                        model, chunks["calibration"][prompt_id][:-1].tolist(), 16, generated
                    )
                    for other_repeat in range(repeats):
                        previous = row(
                            config, "fine_benchmark", "benchmark", settings(prompt_id, other_repeat)
                        )
                        if previous and previous["sample"]["tokens"] != sample["tokens"]:
                            raise RuntimeError("Greedy output tokens changed across repeats")
                    store.add(
                        config.to_dict(),
                        "fine_benchmark",
                        sample["decode_tokens_per_second"],
                        "benchmark",
                        settings(prompt_id, repeat),
                        sample=sample,
                    )
                    print(
                        f"TIMING {config.name} prompt={prompt_id} repeat={repeat} "
                        f"{sample['decode_tokens_per_second']:.2f} tokens/s",
                        flush=True,
                    )
            finally:
                del model
                clear_memory()
    if args.quick and args.stage == "all":
        store.add(marker, "pilot_complete", 1, "none", {})


if __name__ == "__main__":
    main()
