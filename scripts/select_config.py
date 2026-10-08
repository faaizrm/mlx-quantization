"""Choose a measured configuration under storage/speed constraints using calibration only"""

import argparse
import json
import math
from pathlib import Path

from quantlab.data import DEFAULT_MODEL
from quantlab.results import ROOT, digest, read_records


def choose(records: list[dict], max_model_mb: float, min_decode_tps: float = 0) -> dict:
    """Filter a single experiment identity, then minimize calibration perplexity"""
    if not math.isfinite(max_model_mb) or max_model_mb <= 0:
        raise ValueError("Model storage budget must be positive and finite")
    if not math.isfinite(min_decode_tps) or min_decode_tps < 0:
        raise ValueError("Minimum decode rate must be nonnegative and finite")
    if not records:
        raise ValueError("No measured records available")
    latest = records[-1]
    identity = (
        "model",
        "model_revision",
        "implementation_sha256",
        "data",
        "hardware",
        "versions",
        "protocol",
    )
    context = {k: latest[k] for k in identity}
    records = [r for r in records if all(r[k] == v for k, v in context.items())]
    groups = {}
    for row in records:
        if row["config"]["name"].startswith(("layer-", "module-")):
            continue
        group = groups.setdefault(digest(row["config"]), {"config": row["config"]})
        group[(row["metric"], row["split"])] = row
    candidates = []
    for group in groups.values():
        size = group.get(("parameter_bytes", "none"))
        calibration = group.get(("perplexity", "calibration"))
        speed = group.get(("decode_tokens_per_second", "benchmark"))
        if size is None or calibration is None or size["value"] > max_model_mb * 1e6:
            continue
        if min_decode_tps and (speed is None or speed["value"] < min_decode_tps):
            continue
        candidates.append((calibration["value"], size["value"], group["config"]["name"], group))
    if not candidates:
        raise ValueError("No measured configuration meets these constraints")
    _, _, _, chosen = min(candidates, key=lambda item: item[:3])
    metrics = {}
    source_keys = {}
    for label, key in [
        ("model_bytes", ("parameter_bytes", "none")),
        ("calibration_ppl", ("perplexity", "calibration")),
        ("decode_tokens_per_second", ("decode_tokens_per_second", "benchmark")),
        ("test_ppl", ("perplexity", "test")),
        ("arc_accuracy", ("arc_accuracy", "test")),
    ]:
        if key in chosen:
            metrics[label] = chosen[key]["value"]
            source_keys[label] = chosen[key]["key"]
    return {
        "config": chosen["config"],
        "selection_split": "calibration",
        "selection_rule": "minimum calibration perplexity; ties use storage then name",
        "constraints": {
            "max_model_mb": max_model_mb,
            "min_decode_tokens_per_second": min_decode_tps,
        },
        "context": context,
        "metrics": metrics,
        "source_keys": source_keys,
        "qualifying_candidates": len(candidates),
        "limitations": (
            "Model storage is not total process memory. Speed is the recorded 512-token "
            "prompt benchmark, not a deployment guarantee. Test and ARC values are "
            "descriptive and do not affect selection."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--quick", action="store_true", help="Use smoke records only; never a final recommendation"
    )
    parser.add_argument("--max-model-mb", type=float, required=True)
    parser.add_argument("--min-decode-tps", type=float, default=0)
    parser.add_argument("--output", type=Path, help="Optional new JSON config-selection file")
    args = parser.parse_args()
    records = read_records(
        ROOT / "results" / ("smoke.jsonl" if args.quick else "measurements.jsonl")
    )
    try:
        choice = choose(
            [r for r in records if r["model"] == args.model], args.max_model_mb, args.min_decode_tps
        )
    except ValueError as exc:
        parser.error(str(exc))
    choice["smoke_only"] = args.quick
    text = json.dumps(choice, indent=2, sort_keys=True) + "\n"
    if args.output:
        # export choices without replacing earlier selections or append-only measurements
        with args.output.open("x") as output:
            output.write(text)
    print(text, end="")


if __name__ == "__main__":
    main()
