"""Audit completed measurements and record paired ARC uncertainty from saved predictions"""

import hashlib
import math
from pathlib import Path

import numpy as np
from _report import load_report

from quantlab.results import ROOT, ResultStore


def main() -> None:
    args, rows, _ = load_report(__doc__)
    context_keys = ("model", "model_revision", "implementation_sha256", "hardware", "data")
    context = {key: rows[-1][key] for key in context_keys}
    store = ResultStore(
        ROOT / "results" / ("analysis-smoke.jsonl" if args.quick else "analysis.jsonl"), context
    )
    arc = {r["config"]["name"]: r for r in rows if r["metric"] == "arc_accuracy"}
    if "fp16" not in arc:
        raise SystemExit("Run ARC evaluation before paired analysis")
    baseline = arc["fp16"]
    baseline_predictions = baseline["details"]["predictions"]
    baseline_ids = [(p["id"], p["gold"]) for p in baseline_predictions]
    baseline_correct = np.array([p["prediction"] == p["gold"] for p in baseline_predictions])
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name, record in arc.items():
        predictions = record["details"]["predictions"]
        assert [(p["id"], p["gold"]) for p in predictions] == baseline_ids
        correct = np.array([p["prediction"] == p["gold"] for p in predictions])
        assert float(correct.mean()) == record["value"]
        assert int(correct.sum()) == record["details"]["correct"]
        assert len(predictions) == record["details"]["questions"]
        delta = correct.astype(int) - baseline_correct.astype(int)
        wins, losses = int((delta == 1).sum()), int((delta == -1).sum())
        discordant = wins + losses
        # conditional exact McNemar test: discordant outcomes follow Binomial(n, 1/2)
        p_value = (
            min(
                1.0,
                2
                * sum(math.comb(discordant, i) for i in range(min(wins, losses) + 1))
                / 2**discordant,
            )
            if discordant
            else 1.0
        )
        resamples = (
            np.random.default_rng(2026)
            .choice(delta, (10000, len(delta)), replace=True)
            .mean(axis=1)
        )
        interval = np.quantile(resamples, [0.025, 0.975]).tolist()
        settings = {
            "reference": "fp16",
            "resamples": 10000,
            "seed": 2026,
            "interval": "paired-percentile-bootstrap-95",
            "analysis_sha256": code_hash,
            "source_keys": [baseline["key"], record["key"]],
        }
        store.add(
            record["config"],
            "arc_accuracy_difference_vs_fp16",
            float(delta.mean()),
            "test",
            settings,
            ci95=interval,
            wins=wins,
            losses=losses,
            discordant=discordant,
            exact_mcnemar_p=p_value,
            interpretation="Exploratory paired comparison; no multiplicity adjustment",
        )
        print(
            f"{name}: difference={delta.mean():+.3f}, 95% paired interval={interval}, "
            f"wins/losses={wins}/{losses}"
        )

    for row in rows:
        if row["metric"] == "perplexity":
            expected = row["data"][f"{row['split']}_count"]
            assert row["details"]["chunks"] == expected
            assert row["details"]["tokens"] == expected * row["data"]["chunk_size"]
            assert math.isclose(math.exp(row["details"]["mean_nll"]), row["value"])
        if row["split"] == "benchmark":
            assert len(row["details"]["samples"]) == row["settings"]["repeats"]
            sample_values = [s[row["metric"]] for s in row["details"]["samples"]]
            assert float(np.median(sample_values)) == row["value"]
    print(f"Audited {len(rows)} measurement records; paired ARC analysis recorded in {store.path}")


if __name__ == "__main__":
    main()
