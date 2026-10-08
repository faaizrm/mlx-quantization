"""Audit exact storage budgets, paired quality, and timing protocols"""

import argparse
import hashlib
import json
import math
import os
import statistics

import numpy as np
from _fine_grained import BUDGET_BLOCKS, SEED, exact_allocation, isolated, projection_path
from _report import report_path, timing_protocol

from quantlab.config import MODULES
from quantlab.data import DEFAULT_MODEL
from quantlab.results import ROOT, ResultStore, digest, read_records

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


def paired_interval(first, second, repeats=10000):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not len(a) or not np.isfinite(a - b).all():
        raise ValueError("Paired losses need equal, finite, nonempty vectors")
    delta = a - b
    draws = np.random.default_rng(SEED).integers(0, len(delta), (repeats, len(delta)))
    low, high = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
    return {"mean": float(delta.mean()), "low": float(low), "high": float(high)}


def audit_quality(row):
    sample, settings = row["sample"], row["settings"]
    losses = sample["per_chunk_mean_nll"]
    if len(losses) != settings["chunk_count"] or sample["chunks"] != len(losses):
        raise ValueError("Chunk count mismatch")
    if sample["tokens"] != 512 * len(losses) or not np.isfinite(losses).all():
        raise ValueError("Invalid token count/loss")
    mean = sum(losses) / len(losses)
    if not math.isclose(mean, sample["mean_nll"], abs_tol=1e-12, rel_tol=0):
        raise ValueError("Mean NLL mismatch")
    if not math.isclose(math.exp(mean), row["value"], rel_tol=1e-12):
        raise ValueError("Perplexity mismatch")
    for field in ("memory", "load_memory"):
        if any(row[field][k] > 10_000_000_000 for k in ("peak_mlx_bytes", "peak_rss_bytes")):
            raise ValueError("Recorded memory exceeded budget")


def audit_speed(rows, names, *, prompts=3, repeats=5, generated=128):
    expected = {(n, p, r) for n in names for p in range(prompts) for r in range(repeats)}
    indexed = {
        (r["config"]["name"], r["settings"]["prompt_id"], r["settings"]["repeat"]): r for r in rows
    }
    if set(indexed) != expected or len(rows) != len(expected):
        raise ValueError("Timing cohort is incomplete or has duplicates")
    cohorts = {digest(timing_protocol(r["settings"])) for r in rows}
    if len(cohorts) != 1:
        raise ValueError("Cannot combine timing protocols")
    for (name, prompt, _), row in indexed.items():
        sample, settings = row["sample"], row["settings"]
        reference = indexed[(name, prompt, 0)]
        first_config = indexed[(names[0], prompt, 0)]
        if sample["tokens"] != reference["sample"]["tokens"] or len(sample["tokens"]) != generated:
            raise ValueError("Greedy tokens disagree")
        if settings["prompt_sha256"] != reference["settings"]["prompt_sha256"]:
            raise ValueError("Prompt changed across repeats")
        if settings["prompt_sha256"] != first_config["settings"]["prompt_sha256"]:
            raise ValueError("Prompt changed across configurations")
        if settings["generated_tokens"] != generated or settings["repeats"] != repeats:
            raise ValueError("Timing settings mismatch")
        if settings["prompt_tokens"] != 512 or settings["cache_bits"] != 16:
            raise ValueError("Unexpected prompt/cache protocol")
        if settings["warmups"] != (1 if generated == 16 else 2):
            raise ValueError("Unexpected warmup count")
        if not all(
            math.isfinite(sample[k]) and sample[k] > 0
            for k in ("ttft_seconds", "decode_seconds", "decode_tokens_per_second")
        ):
            raise ValueError("Invalid timing measurement")
        if row["value"] != sample["decode_tokens_per_second"]:
            raise ValueError("Timing sample/value mismatch")
        if not math.isclose(
            row["value"], (generated - 1) / sample["decode_seconds"], rel_tol=1e-12
        ):
            raise ValueError("Timing rate mismatch")
        if any(sample[k] > 10_000_000_000 for k in ("peak_mlx_bytes", "peak_rss_bytes")):
            raise ValueError("Timing memory exceeded budget")
    return indexed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()
    source = ROOT / "results" / ("fine-grained-smoke.jsonl" if args.quick else "fine-grained.jsonl")
    rows = [r for r in read_records(source) if r["model"] == args.model]
    if not rows:
        print("Projection measurements not present; optional report skipped")
        return
    identity = (
        "model",
        "model_revision",
        "implementation_sha256",
        "data",
        "versions",
        "protocol",
        "hardware",
        "fine_implementation_sha256",
        "fine_data",
        "quick",
    )
    context = {k: rows[-1][k] for k in identity}
    rows = [r for r in rows if all(r.get(k) == v for k, v in context.items())]
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate result keys")
    for row in rows:
        expected = digest(
            {
                **context,
                "config": row["config"],
                "metric": row["metric"],
                "split": row["split"],
                "settings": row["settings"],
            }
        )
        if row["key"] != expected:
            raise ValueError("Result identity key does not match provenance")
        if row["metric"] == "fine_quality":
            audit_quality(row)
    timing = [r for r in rows if r["metric"] == "fine_benchmark"]
    cohort = timing_protocol(timing[-1]["settings"]) if timing else {}
    timing = [r for r in timing if timing_protocol(r["settings"]) == cohort]
    if args.quick:
        if not any(r["metric"] == "pilot_complete" for r in rows):
            raise ValueError("Pilot incomplete")
        audit_speed(timing, ["int4-g64", "projection-pilot"], prompts=1, repeats=2, generated=16)
        print(f"Projection pilot passed: {len(rows)} records, quality/storage and four timings")
        return

    def pending(message):
        if not args.allow_pending:
            raise RuntimeError(message)
        print(message + "; final report not yet generated")

    selection_path = ROOT / "data" / "fine-grained-selection.json"
    if not selection_path.exists():
        pending("Projection sensitivity/selection pending")
        return
    selection = json.loads(selection_path.read_text())
    if selection["context"] != context or selection["selection_split"] != "calibration":
        raise ValueError("Frozen selection belongs to another identity/split")
    quality = {
        (r["config"]["name"], r["split"], r["settings"]["chunk_count"]): r
        for r in rows
        if r["metric"] == "fine_quality"
    }
    catalog = next(r for r in rows if r["metric"] == "projection_catalog")
    base_bytes, costs = catalog["catalogs"]["4"]["parameter_bytes"], catalog["costs"]
    baseline = quality[("fp16", "calibration", 32)]
    pair_rows = {
        projection_path(i, m): quality[(isolated(i, m).name, "calibration", 32)]
        for i in range(28)
        for m in MODULES
    }
    scores = {
        p: r["sample"]["mean_nll"] - baseline["sample"]["mean_nll"] for p, r in pair_rows.items()
    }
    expected_keys = sorted(r["key"] for r in pair_rows.values()) + [baseline["key"]]
    if (
        selection["calibration_record_keys"] != expected_keys
        or selection["catalog_key"] != catalog["key"]
    ):
        raise ValueError("Selection inputs do not match measured calibration/catalog")
    configs = ["fp16", "int4-g64", "int8-g64"] + [
        a["config"]["name"] for a in selection["allocations"]
    ]
    missing = [
        (n, s, c)
        for n in configs
        for s, c in (("calibration", 128), ("test", 256))
        if (n, s, c) not in quality
    ]
    if missing:
        pending(f"Projection quality pending: {len(missing)} configuration/split evaluations")
        return
    for allocation in selection["allocations"]:
        name = allocation["config"]["name"]
        for split, count in (("calibration", 128), ("test", 256)):
            r = quality[(name, split, count)]
            if (
                r["config"] != allocation["config"]
                or r["parameter_bytes"] != allocation["parameter_bytes"]
            ):
                raise ValueError("Measured allocation/storage differs from frozen manifest")
        if name.startswith("fine-projection-"):
            expected = exact_allocation(costs, scores, allocation["parameter_bytes"] - base_bytes)
            actual = [
                rule["pattern"] for rule in allocation["config"]["rules"] if rule["bits"] == 8
            ]
            if sorted(expected) != sorted(actual):
                raise ValueError("Allocation does not reproduce from calibration alone")
    for k in BUDGET_BLOCKS:
        group = [a for a in selection["allocations"] if a["budget_blocks"] == k]
        if len(group) != 5 or len({a["parameter_bytes"] for a in group}) != 1:
            raise ValueError("Matched allocations do not share an exact budget")
        guided = next(a for a in group if a["config"]["name"] == f"fine-projection-{k}")
        counts = {
            m: sum(r["pattern"].endswith("." + m) for r in guided["config"]["rules"][1:])
            for m in MODULES
        }
        for allocation in group:
            if "random" in allocation["config"]["name"]:
                for m in MODULES:
                    count = sum(
                        r["pattern"].endswith("." + m) for r in allocation["config"]["rules"][1:]
                    )
                    if count != counts[m]:
                        raise ValueError("Random control module counts differ from guided")
    speed_names = [n for n in configs if n != "fp16" and "random" not in n]
    complete_speed = len(timing) == len(speed_names) * 15
    if not complete_speed and not args.allow_pending:
        raise RuntimeError("Projection timings are pending")
    indexed = audit_speed(timing, speed_names) if complete_speed else {}
    for row in timing:
        if row["config"] != quality[(row["config"]["name"], "calibration", 128)]["config"]:
            raise ValueError("Timing configuration differs from quality configuration")

    def losses(name, split="test", count=256):
        return quality[(name, split, count)]["sample"]["per_chunk_mean_nll"]

    comparisons = {
        str(k): paired_interval(losses(f"fine-projection-{k}"), losses(f"fine-block-{k}"))
        for k in BUDGET_BLOCKS
    }
    negative = min(scores, key=lambda p: (scores[p], p))
    diagnostic = None
    if scores[negative] < 0:
        name = pair_rows[negative]["config"]["name"]
        original = pair_rows[negative]["sample"]["per_chunk_mean_nll"]
        if losses(name, "calibration", 128)[:32] != original:
            raise ValueError("Fresh diagnostic reconstruction changed original chunk losses")
        if losses("fp16", "calibration", 128)[:32] != baseline["sample"]["per_chunk_mean_nll"]:
            raise ValueError("Fresh fp16 reconstruction changed original chunk losses")
        diagnostic = {
            "path": negative,
            "subset": paired_interval(original, baseline["sample"]["per_chunk_mean_nll"]),
            "full": paired_interval(
                losses(name, "calibration", 128), losses("fp16", "calibration", 128)
            ),
            "remaining_96": paired_interval(
                losses(name, "calibration", 128)[32:], losses("fp16", "calibration", 128)[32:]
            ),
            "prefix_reproduced_exactly": True,
        }

    fp16_cal_comparison = paired_interval(
        losses("fine-projection-16", "calibration", 128), losses("fp16", "calibration", 128)
    )
    fp16_test_comparison = paired_interval(losses("fine-projection-16"), losses("fp16"))
    fp16_test = quality[("fp16", "test", 256)]["value"]
    int4_test = quality[("int4-g64", "test", 256)]["value"]
    improvements = {}
    for k in BUDGET_BLOCKS:
        projected = quality[(f"fine-projection-{k}", "test", 256)]
        block = quality[(f"fine-block-{k}", "test", 256)]
        improvements[str(k)] = {
            "ppl_gap_recovery_fraction": (int4_test - projected["value"]) / (int4_test - fp16_test),
            "matched_block_ppl_reduction_fraction": 1 - projected["value"] / block["value"],
            "storage_over_int4_fraction": projected["parameter_bytes"] / base_bytes - 1,
        }

    analyses = ResultStore(ROOT / "results" / "fine-grained-analysis.jsonl", context)
    audit = {
        "source_keys": sorted(r["key"] for r in rows if r["metric"] != "fine_benchmark")
        + sorted(r["key"] for r in timing),
        "timing_complete": complete_speed,
        "cohort": cohort,
        "paired_test_comparisons": comparisons,
        "negative_delta_review": diagnostic,
        "projection16_vs_fp16_calibration": fp16_cal_comparison,
        "projection16_vs_fp16_test": fp16_test_comparison,
        "improvements": improvements,
    }
    analyses.add(
        {"name": "projection-audit", "rules": []},
        "audit",
        len(audit["source_keys"]),
        "analysis",
        {
            "input_digest": digest(audit),
            "analysis_sha256": hashlib.sha256(
                (ROOT / "scripts" / "report_fine_grained.py").read_bytes()
            ).hexdigest(),
        },
        **audit,
    )
    lines = [
        "# Projection-level precision allocation",
        "",
        "All 196 block × projection pairs use the first 32 fixed calibration chunks.",
        "Allocations were frozen from calibration alone and evaluated on 128 calibration",
        "and 256 held-out chunks. Group size is 64; tied embeddings remain fp16.",
        "",
        "| Configuration | Actual model MB | Calibration PPL | Test PPL |",
        "|---|---:|---:|---:|",
    ]
    for name in configs:
        cal, test = quality[(name, "calibration", 128)], quality[(name, "test", 256)]
        lines.append(
            f"| {name} | {test['parameter_bytes'] / 1e6:.2f} | "
            f"{cal['value']:.4f} | {test['value']:.4f} |"
        )
    lines += [
        "",
        "## Matched storage and paired uncertainty",
        "",
        "Negative delta favors projection allocation. Intervals resample paired test chunks",
        "10,000 times (seed 20261007); these are exploratory comparisons on one corpus.",
        "",
        "| Block-equivalent budget | Projection − block mean NLL | "
        "95% paired bootstrap interval | Beats random controls on test |",
        "|---|---:|---:|---:|",
    ]
    for k in BUDGET_BLOCKS:
        delta = comparisons[str(k)]
        wins = sum(
            quality[(f"fine-projection-{k}", "test", 256)]["value"]
            < quality[(f"fine-random-{k}-seed-{s}", "test", 256)]["value"]
            for s in (0, 1, 2)
        )
        lines.append(
            f"| {k} | {delta['mean']:+.6f} | "
            f"[{delta['low']:+.6f}, {delta['high']:+.6f}] | {wins}/3 |"
        )
    lines += [
        "",
        "The exact knapsack maximizes clipped positive isolated delta NLL, an additive proxy",
        "for promotion benefit. Isolated fp16-to-int4 sensitivity need not predict int4-to-int8",
        "restoration in a quantized network. Negative cells are preserved, but contribute zero",
        "to that allocation objective. Random controls preserve promoted counts per module type",
        "and actual storage. Neither the subset size nor budgets were tuned on these test results.",
    ]
    if diagnostic:
        d = diagnostic["full"]
        held = diagnostic["remaining_96"]
        lines += [
            "",
            "## Apparent-improvement diagnostic",
            "",
            f"Most negative subset cell: `{negative}`, delta mean NLL {scores[negative]:+.6f}.",
            f"On all 128 calibration chunks: {d['mean']:+.6f}, "
            f"paired 95% interval [{d['low']:+.6f}, {d['high']:+.6f}].",
            "A fresh checkpoint reconstruction reproduced the original 32 chunk losses exactly.",
            "Exact projection targeting is covered by tests. This localized estimate is not",
            "evidence that quantization generally improves quality; it was chosen as the most",
            "negative of 196 cells and has selection bias.",
            f"Excluding those 32 selection chunks, the remaining 96 have delta {held['mean']:+.6f}",
            f"with paired interval [{held['low']:+.6f}, {held['high']:+.6f}], spanning zero.",
            "",
            "Projection-16's tiny apparent calibration improvement over fp16 also has a",
            f"paired interval spanning zero: [{fp16_cal_comparison['low']:+.6f}, "
            f"{fp16_cal_comparison['high']:+.6f}]. On test it is worse than fp16:",
            f"delta mean NLL {fp16_test_comparison['mean']:+.6f}, paired interval "
            f"[{fp16_test_comparison['low']:+.6f}, {fp16_test_comparison['high']:+.6f}].",
            "No configuration was revised after viewing test outcomes.",
        ]
    lines += ["", "## Timing", ""]
    summaries = {}
    if complete_speed:
        lines += [
            "Three fixed 512-token prompts, 128 greedy outputs, two warmups and five repeats",
            "per prompt/configuration. Table values are medians of the three prompt medians.",
            "Configuration order was shuffled once; prompt/repeat order was shuffled within",
            "each configuration. Configurations were not interleaved between repeats, so",
            "session drift may affect comparisons. Overlapping ranges do not support a speed win.",
            "TTFT ends when the first GPU token is ready; decode times the next 127 tokens",
            "including scalar transfers. Repeat outputs match within every config/prompt.",
            "Compare within this experiment: the core study uses a different timing boundary.",
            "Thermal/background load remains uncontrolled.",
            "",
            "| Configuration | Decode tokens/s | TTFT seconds | "
            "Peak MLX GB | Sampled peak RSS GB |",
            "|---|---:|---:|---:|---:|",
        ]
        for name in speed_names:
            samples = [indexed[(name, p, r)]["sample"] for p in range(3) for r in range(5)]
            per_prompt = {
                metric: [
                    statistics.median(indexed[(name, p, r)]["sample"][metric] for r in range(5))
                    for p in range(3)
                ]
                for metric in ("decode_tokens_per_second", "ttft_seconds")
            }
            summaries[name] = per_prompt
            lines.append(
                f"| {name} | {statistics.median(per_prompt['decode_tokens_per_second']):.2f} | "
                f"{statistics.median(per_prompt['ttft_seconds']):.3f} | "
                f"{max(s['peak_mlx_bytes'] for s in samples) / 1e9:.3f} | "
                f"{max(s['peak_rss_bytes'] for s in samples) / 1e9:.3f} |"
            )
        lines += [
            "",
            "### Per-prompt repeat ranges",
            "",
            "Each cell is the median [minimum, maximum] of five measured repetitions.",
            "Ranges describe observed variation, not confidence intervals.",
            "",
            "| Configuration | Prompt | Decode tokens/s | TTFT seconds |",
            "|---|---:|---:|---:|",
        ]
        for name in speed_names:
            for prompt in range(3):
                values = []
                for metric in ("decode_tokens_per_second", "ttft_seconds"):
                    samples = [
                        indexed[(name, prompt, repeat)]["sample"][metric] for repeat in range(5)
                    ]
                    values.append(
                        f"{statistics.median(samples):.3f} [{min(samples):.3f}, {max(samples):.3f}]"
                    )
                lines.append(f"| {name} | {prompt} | {values[0]} | {values[1]} |")
    else:
        lines += [
            "**Pending:** the complete timing cohort has not finished. No timing claim is made."
        ]
    lines += [
        "",
        "## Provenance and scope",
        "",
        f"Raw records: `{source.relative_to(ROOT)}`; "
        "selection: `data/fine-grained-selection.json`.",
        f"Scientific extension SHA-256: `{context['fine_implementation_sha256']}`.",
        f"Audit: {len(pair_rows)} sensitivity pairs, 26 full configuration/split evaluations,",
        f"and {len(timing)} timing samples in the selected cohort. JSONL retains prior cohorts.",
        "Load/evaluation memory is checked separately against the 10 GB cap. MLX allocation",
        "and process RSS are distinct counters and must not be added. Model bytes include",
        "affine metadata and unquantized arrays; they are not peak runtime memory.",
        "",
        "This follow-up reuses the core model and data. The allocation map uses only 32",
        "calibration chunks, versus 128 for the original block ranking. No new ARC evaluation",
        "or out-of-domain generalization is claimed. Reproduction commands are in README.md.",
        "",
    ]
    report_path("projection.json").write_text(
        json.dumps(
            {
                "context": context,
                "audit": audit,
                "test": {
                    name: {k: quality[(name, "test", 256)][k] for k in ("value", "parameter_bytes")}
                    for name in configs
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    report_path("projection.md").write_text("\n".join(lines))
    plot_results(scores, quality, summaries)
    print(
        f"Audited {len(audit['source_keys'])} projection records; timing complete={complete_speed}"
    )


def plot_results(scores, quality, speed):
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10,
            "svg.hashsalt": "quantlab",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "svg"):
            path = ROOT / "figures" / f"{name}.{suffix}"
            fig.savefig(
                path, bbox_inches="tight", metadata={"Date": None} if suffix == "svg" else None
            )
            if suffix == "svg":
                path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
        plt.close(fig)

    grid = np.array([[scores[projection_path(i, m)] for m in MODULES] for i in range(28)])
    limit = float(np.abs(grid).max())
    fig, ax = plt.subplots(figsize=(8, 9))
    shown = ax.imshow(
        grid,
        aspect="auto",
        cmap="RdBu_r",
        norm=SymLogNorm(linthresh=0.0001, vmin=-limit, vmax=limit),
    )
    ax.set(
        xticks=range(7),
        xticklabels=[m.removesuffix("_proj") for m in MODULES],
        yticks=range(28),
        ylabel="Transformer block",
        xlabel="Projection",
        title="Isolated int4 sensitivity on 32 calibration chunks",
    )
    fig.colorbar(shown, ax=ax, label="Mean NLL increase over fp16 (symmetric log scale)")
    save(fig, "fine_grained_sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, split, count in zip(axes, ("calibration", "test"), (128, 256), strict=True):
        for method, color, marker in (("projection", "#bd4935", "D"), ("block", "#3278a0", "o")):
            rows = [quality[(f"fine-{method}-{k}", split, count)] for k in BUDGET_BLOCKS]
            ax.plot(
                [r["parameter_bytes"] / 1e6 for r in rows],
                [r["value"] for r in rows],
                marker=marker,
                color=color,
                label=method,
            )
        for k in BUDGET_BLOCKS:
            rows = [quality[(f"fine-random-{k}-seed-{s}", split, count)] for s in (0, 1, 2)]
            ax.scatter(
                [r["parameter_bytes"] / 1e6 for r in rows],
                [r["value"] for r in rows],
                marker="x",
                color="0.5",
                label="matched random" if k == 8 else None,
            )
        for name, style in (("int4-g64", "--"), ("int8-g64", ":")):
            ax.axhline(
                quality[(name, split, count)]["value"], linestyle=style, color="0.4", label=name
            )
        ax.set(
            title=f"{split.capitalize()} ({count} chunks)",
            xlabel="Actual model storage (MB)",
            ylabel="Perplexity (lower is better)",
        )
        values = [
            quality[(f"fine-{method}-{k}", split, count)]["value"]
            for method in ("projection", "block")
            for k in BUDGET_BLOCKS
        ]
        values += [quality[(name, split, count)]["value"] for name in ("int4-g64", "int8-g64")]
        margin = (max(values) - min(values)) * 0.08
        ax.set_ylim(min(values) - margin, max(values) + margin)
        ax.legend(fontsize=8)
    save(fig, "fine_grained_allocation")
    if speed:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        labels = [n.removeprefix("fine-") for n in speed]
        for ax, metric, label in zip(
            axes,
            ("decode_tokens_per_second", "ttft_seconds"),
            ("Decode tokens/s", "TTFT seconds"),
            strict=True,
        ):
            for i, name in enumerate(speed):
                vals = speed[name][metric]
                ax.bar(i, statistics.median(vals), color="#3278a0", alpha=0.7)
                ax.scatter([i - 0.13, i, i + 0.13], vals, color="#1b263b", s=18)
            ax.set(
                xticks=range(len(labels)),
                xticklabels=labels,
                ylabel=label,
                title="Bars: median; dots: three prompt medians",
            )
            ax.tick_params(axis="x", rotation=30)
        save(fig, "fine_grained_timing")


if __name__ == "__main__":
    main()
