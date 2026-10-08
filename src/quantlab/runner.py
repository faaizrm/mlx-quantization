"""Resumable experiment stages, with calibration-only mixed-config selection"""

import argparse
import hashlib
import json
import random
import time

from quantlab.config import MODULES, QuantConfig, baselines, layer_only, mixed, module_only, uniform
from quantlab.data import DEFAULT_MODEL, arc_questions, model_snapshot, wikitext_chunks
from quantlab.results import ROOT, ResultStore, experiment_lock, hardware_metadata


def pareto(rows: list[dict]) -> list[dict]:
    """Minimize actual parameter bytes and calibration perplexity; retain ties"""
    return [
        r
        for r in rows
        if not any(
            o["bytes"] <= r["bytes"]
            and o["ppl"] <= r["ppl"]
            and (o["bytes"] < r["bytes"] or o["ppl"] < r["ppl"])
            for o in rows
        )
    ]


def parser(stage: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Run {stage} experiments")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--quick", action="store_true", help="Smoke test only; writes smoke.jsonl")
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--calibration-chunks", type=int, default=128)
    p.add_argument("--test-chunks", type=int, default=256)
    p.add_argument("--only", nargs="+", help="Run only these named configurations")
    p.add_argument("--no-speed", action="store_true")
    return p


def main(stage: str) -> None:
    args = parser(stage).parse_args()
    with experiment_lock():
        run(stage, args)


def run(stage: str, args) -> None:
    # import MLX after command parsing so --help works without GPU access
    import mlx.core as mx
    from transformers import AutoTokenizer

    from quantlab.bench import benchmark
    from quantlab.eval_arc import arc_accuracy
    from quantlab.eval_ppl import perplexity
    from quantlab.memory import measure_memory
    from quantlab.model import clear_memory, load_config, parameter_bytes

    snapshot, revision = model_snapshot(args.model)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
    chunks, data_settings = wikitext_chunks(
        tokenizer,
        args.model,
        chunk_size=args.chunk_size,
        calibration_count=args.calibration_chunks,
        test_count=args.test_chunks,
        quick=args.quick,
    )
    model_metadata = json.loads((snapshot / "config.json").read_text())
    layers = model_metadata["num_hidden_layers"]
    source = b"".join(p.read_bytes() for p in sorted((ROOT / "src" / "quantlab").glob("*.py")))
    context = {
        "model": args.model,
        "model_revision": revision,
        "data": data_settings,
        "implementation_sha256": hashlib.sha256(source).hexdigest(),
        "hardware": hardware_metadata(),
    }
    store = ResultStore(
        ROOT / "results" / ("smoke.jsonl" if args.quick else "measurements.jsonl"), context
    )
    ppl_settings = {
        "chunk_size": args.chunk_size,
        "scoring": "next-token-nll-float32",
        "context_reset": "each_chunk",
        "special_tokens": False,
    }
    speed_settings = {
        "prompt_tokens": 512,
        "generated_tokens": 128,
        "warmups": 2,
        "repeats": 5,
        "sampler": "greedy",
        "stop_at_eos": False,
        "prompt_sha256": hashlib.sha256(chunks["calibration"][0][:512].tobytes()).hexdigest(),
    }
    if args.quick:
        speed_settings.update(generated_tokens=16, warmups=1, repeats=2)

    def evaluate(config: QuantConfig, split: str, speed: bool = False, arc=None) -> None:
        cfg = config.to_dict()
        todo_ppl = arc is None and not store.has(cfg, "perplexity", split, ppl_settings)
        todo_size = not store.has(cfg, "parameter_bytes", "none", {})
        todo_speed = speed and not store.has(
            cfg, "decode_tokens_per_second", "benchmark", speed_settings
        )
        todo_arc = arc is not None and not store.has(cfg, "arc_accuracy", "test", arc[1])
        if not any((todo_ppl, todo_size, todo_speed, todo_arc)):
            print(f"SKIP {config.name} {split}: measurements already exist", flush=True)
            return
        print(f"RUN {config.name} {split}", flush=True)
        start = time.perf_counter()
        with measure_memory() as load_memory:
            model, model_tokenizer = load_config(snapshot, config)
        try:
            if todo_size:
                store.add(
                    cfg,
                    "parameter_bytes",
                    parameter_bytes(model),
                    "none",
                    {},
                    load_memory=load_memory,
                )
            if todo_ppl:
                with measure_memory() as memory:
                    result = perplexity(model, chunks[split])
                store.add(
                    cfg,
                    "perplexity",
                    result["perplexity"],
                    split,
                    ppl_settings,
                    details=result,
                    memory=memory,
                )
                print(f"  perplexity={result['perplexity']:.6f}; memory={memory}", flush=True)
                if not 1 < result["perplexity"] < 1000:
                    raise RuntimeError("Suspicious perplexity; investigate before continuing")
            if todo_speed:
                if args.chunk_size != 512:
                    raise ValueError("The speed protocol requires a 512-token prompt")
                result = benchmark(
                    model,
                    chunks["calibration"][0][:-1].tolist(),
                    decode_tokens=speed_settings["generated_tokens"],
                    warmups=speed_settings["warmups"],
                    repeats=speed_settings["repeats"],
                )
                for metric in (
                    "prefill_tokens_per_second",
                    "peak_mlx_bytes",
                    "peak_rss_bytes",
                    "decode_tokens_per_second",
                ):
                    store.add(
                        cfg,
                        metric,
                        result[metric]["median"],
                        "benchmark",
                        speed_settings,
                        details=result,
                    )
            if todo_arc:
                with measure_memory() as memory:
                    result = arc_accuracy(model, model_tokenizer, arc[0])
                store.add(
                    cfg,
                    "arc_accuracy",
                    result["accuracy"],
                    "test",
                    arc[1],
                    details=result,
                    memory=memory,
                )
        finally:
            del model, model_tokenizer
            clear_memory()
        print(f"DONE {config.name}: {time.perf_counter() - start:.1f}s", flush=True)

    def relevant(metric: str, split: str) -> list[dict]:
        return [
            r
            for r in store.records
            if r["metric"] == metric
            and r["split"] == split
            and all(r.get(k) == v for k, v in store.context.items())
        ]

    def selected_configs() -> list[QuantConfig]:
        path = ROOT / "data" / ("selected-smoke.json" if args.quick else "selected.json")
        if not path.exists():
            raise RuntimeError("Run the mixed sweep before ARC selection")
        selection = json.loads(path.read_text())
        if selection["context"] != store.context:
            raise RuntimeError(
                "Selected configs belong to another experiment identity; rerun mixed selection"
            )
        return [QuantConfig.from_dict(c) for c in selection["configs"]]

    if stage == "baselines":
        configs = [QuantConfig("fp16"), uniform(4)] if args.quick else baselines()
        split = "test"
    elif stage == "sensitivity":
        configs = [QuantConfig("fp16")]
        configs += [layer_only(i) for i in range(2 if args.quick else layers)]
        configs += [module_only(m) for m in (("q_proj", "gate_proj") if args.quick else MODULES)]
        split = "calibration"
    elif stage == "mixed":
        calibration = {
            r["config"]["name"]: r["value"] for r in relevant("perplexity", "calibration")
        }
        expected = range(2 if args.quick else layers)
        if any(layer_only(i).name not in calibration for i in expected):
            raise RuntimeError("Complete layer sensitivity on the same calibration settings first")
        ranking = sorted(expected, key=lambda i: (-calibration[layer_only(i).name], i))
        configs = [QuantConfig("fp16"), uniform(8), uniform(4)]
        for k in (2,) if args.quick else (2, 4, 6, 8, 12, 16):
            configs.append(mixed(ranking[:k], f"mixed-top-{k}"))
            for seed in (0,) if args.quick else (0, 1, 2):
                blocks = random.Random(seed).sample(list(range(layers)), k)
                configs.append(mixed(blocks, f"mixed-random-{k}-seed-{seed}"))
        split = "calibration"
    elif stage == "arc":
        chosen = selected_configs()
        # choose mixed configs by calibration only, without inspecting test accuracy/PPL
        configs = baselines() + chosen
        questions = arc_questions(5 if args.quick else 500)
        seen = set()
        for config in configs:
            if config.fingerprint in seen or (args.only and config.name not in args.only):
                continue
            seen.add(config.fingerprint)
            evaluate(config, "test", arc=questions)
        return
    else:
        raise ValueError(stage)

    if args.only:
        configs = [c for c in configs if c.name in args.only]
        if not configs:
            raise ValueError("No matching configurations")
    start = time.perf_counter()
    for i, config in enumerate(configs):
        evaluate(config, split, speed=stage == "baselines" and not args.no_speed)
        if i == 0:
            estimate = (time.perf_counter() - start) * (len(configs) - 1) / 60
            print(
                f"Pilot-based remaining runtime estimate: {estimate:.1f} min; "
                "cached results and quantization change this estimate",
                flush=True,
            )

    if stage == "mixed" and not args.only:
        sizes = {r["config"]["name"]: r["value"] for r in relevant("parameter_bytes", "none")}
        values = {r["config"]["name"]: r["value"] for r in relevant("perplexity", "calibration")}
        rows = [
            {"config": c.to_dict(), "bytes": sizes[c.name], "ppl": values[c.name]} for c in configs
        ]
        frontier = pareto(rows)
        # preserve random controls for matched-k comparisons, even when dominated
        test_configs = [QuantConfig.from_dict(r["config"]) for r in frontier]
        test_configs += [c for c in configs if "random" in c.name]
        guided = sorted(
            (r for r in rows if r["config"]["name"].startswith("mixed-top")),
            key=lambda r: r["ppl"],
        )
        selection = {
            "context": store.context,
            "selection_split": "calibration",
            "ranking": ranking,
            "frontier": frontier,
            "configs": [r["config"] for r in guided[:3]],
        }
        target = ROOT / "data" / ("selected-smoke.json" if args.quick else "selected.json")
        target.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n")
        test_configs += [QuantConfig.from_dict(r["config"]) for r in guided[:3]]
        seen = set()
        for config in test_configs:
            if config.name not in seen:
                evaluate(config, "test", speed=not args.no_speed and "random" not in config.name)
                seen.add(config.name)
    mx.clear_cache()
