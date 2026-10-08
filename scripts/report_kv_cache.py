"""Audit and report the prespecified long-context KV-cache experiment"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from _kv_cache import CACHE_BITS, CONTEXTS, SEED
from _report import quality_context, report_path, timing_protocol

from quantlab.data import DEFAULT_MODEL
from quantlab.results import ROOT, ResultStore, read_records

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--quality-only", action="store_true", help="Report without timings")
    parser.add_argument("--allow-pending-speed", action="store_true")
    args = parser.parse_args()
    path = ROOT / "results" / ("kv-cache-smoke.jsonl" if args.quick else "kv-cache.jsonl")
    rows = [r for r in read_records(path) if r["model"] == args.model]
    if not rows:
        print("No KV-cache extension measurements; optional report skipped")
        return
    identity = (
        "model",
        "model_revision",
        "implementation_sha256",
        "kv_implementation_sha256",
        "kv_data",
        "hardware",
        "versions",
        "protocol",
        "data",
    )
    context = {k: rows[-1][k] for k in identity}
    quality_identity = quality_context(context)
    timing = [r for r in rows if r["metric"] == "kv_benchmark"]
    cohort = timing_protocol(timing[-1]["settings"]) if timing else {}
    rows = [
        r
        for r in rows
        if all(
            r.get(k) == v
            for k, v in (
                quality_identity if r["metric"] == "continuation_mean_nll" else context
            ).items()
        )
        and (r["metric"] == "continuation_mean_nll" or timing_protocol(r["settings"]) == cohort)
    ]
    lengths = (CONTEXTS[0], CONTEXTS[-1]) if args.quick else CONTEXTS
    precisions = (16, 4) if args.quick else CACHE_BITS
    count, outputs, prompt_count, repeats = (2, 16, 1, 2) if args.quick else (8, 128, 3, 5)
    quality, speed = {}, {}
    for row in rows:
        key = (row["settings"]["context_tokens"], row["config"]["cache_bits"])
        target = quality if row["metric"] == "continuation_mean_nll" else speed
        target.setdefault(key, []).append(row)
    planned = {(length, bits) for length in lengths for bits in precisions}
    timing_complete = set(speed) == planned and all(
        len(v) == prompt_count * repeats for v in speed.values()
    )
    if args.allow_pending_speed and not timing_complete:
        args.quality_only = True
        print("Speed measurements incomplete; explicitly reporting quality/storage only")
    if set(quality) != planned or (not args.quality_only and set(speed) != planned):
        raise RuntimeError("Complete the prespecified grid before reporting")
    summaries = {}
    analyses = ResultStore(
        ROOT
        / "results"
        / ("kv-cache-analysis-smoke.jsonl" if args.quick else "kv-cache-analysis.jsonl"),
        context,
    )
    analysis_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    draws = np.random.default_rng(SEED).integers(0, count, size=(10000, count))
    for length, bits in sorted(planned):
        q = sorted(quality[(length, bits)], key=lambda r: r["settings"]["window_index"])
        baseline = sorted(quality[(length, 16)], key=lambda r: r["settings"]["window_index"])
        b = [] if args.quality_only else speed[(length, bits)]
        if [r["settings"]["window_index"] for r in q] != list(range(count)):
            raise RuntimeError("Missing or duplicated quality windows")
        actual_repeats = {(r["settings"]["window_index"], r["settings"]["repeat"]) for r in b}
        expected_repeats = {(i, j) for i in range(prompt_count) for j in range(repeats)}
        if not args.quality_only and (
            actual_repeats != expected_repeats or len(b) != prompt_count * repeats
        ):
            raise RuntimeError("Missing or duplicated speed repetitions")
        for row, base in zip(q, baseline, strict=True):
            assert row["settings"]["target_sha256"] == base["settings"]["target_sha256"]
            assert row["sample"]["scored_tokens"] == outputs
            assert len(row["sample"]["token_nlls"]) == outputs
            assert math.isclose(math.fsum(row["sample"]["token_nlls"]) / outputs, row["value"])
            assert row["sample"]["cache_offset"] == length + outputs - 1
        for row in b:
            assert len(row["sample"]["tokens"]) == outputs
            assert row["sample"]["cache_offset"] == length + outputs - 1
            assert math.isclose((outputs - 1) / row["sample"]["decode_seconds"], row["value"])
        losses = np.array([r["value"] for r in q])
        base_losses = np.array([r["value"] for r in baseline])
        ppl = float(np.exp(losses.mean()))
        difference = ppl - float(np.exp(base_losses.mean()))
        boot_difference = np.exp(losses[draws].mean(1)) - np.exp(base_losses[draws].mean(1))
        interval = np.quantile(boot_difference, [0.025, 0.975]).tolist()
        settings = {
            "context_tokens": length,
            "scored_tokens": outputs * count,
            "quick": args.quick,
            "bootstrap_unit": "paired-window",
            "resamples": 10000,
            "seed": SEED,
            "analysis_sha256": analysis_hash,
            "quality_source_implementation_sha256": quality_identity["kv_implementation_sha256"],
            "timing_protocol": cohort if not args.quality_only else None,
            "source_keys": sorted({r["key"] for r in q + baseline + b}),
        }

        def values(metric):
            return np.array([r["sample"][metric] for r in (b or q)])

        summary = {
            "ppl": ppl,
            "ppl_difference": difference,
            "ppl_difference_ci95": interval,
            "relative_ppl_change_percent": 100
            * (math.exp(float((losses - base_losses).mean())) - 1),
            "decode_median": float(np.median(values("decode_tokens_per_second"))) if b else None,
            "decode_min": float(values("decode_tokens_per_second").min()) if b else None,
            "decode_max": float(values("decode_tokens_per_second").max()) if b else None,
            "ttft_median": float(np.median(values("ttft_seconds"))) if b else None,
            "ttft_min": float(values("ttft_seconds").min()) if b else None,
            "ttft_max": float(values("ttft_seconds").max()) if b else None,
            "cache_bytes": int(np.median(values("cache_bytes"))),
            "peak_mlx_bytes": int(values("peak_mlx_bytes").max()),
            "peak_rss_bytes": int(values("peak_rss_bytes").max()),
        }
        assert len(set(values("cache_bytes"))) == 1
        assert all(
            r["sample"]["peak_rss_bytes"] < 10_000_000_000
            and r["sample"]["peak_mlx_bytes"] < 10_000_000_000
            for r in q + b
        )
        analyses.add(
            q[0]["config"], "continuation_perplexity", ppl, "test", settings, summary=summary
        )
        summaries[(length, bits)] = summary
    lines = [
        "# Long-context KV-cache quantization" + (" — SMOKE ONLY" if args.quick else ""),
        "",
        "The target weights are fixed at the calibration-selected mixed-top-16 configuration.",
        "Every context length scores the same held-out targets; no test-based selection occurs.",
        "",
        "| Context | Cache | Allocated cache MB | Conditional PPL | "
        "Difference vs fp16 [95% paired interval] | Decode tok/s [min, max] | "
        "TTFT seconds [min, max] | Peak MLX GB | Peak RSS GB |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for length in lengths:
        for bits in precisions:
            s = summaries[(length, bits)]
            low, high = s["ppl_difference_ci95"]
            decode = (
                f"{s['decode_median']:.2f} [{s['decode_min']:.2f}, {s['decode_max']:.2f}]"
                if s["decode_median"] is not None
                else ("omitted" if timing_complete else "pending")
            )
            ttft = (
                f"{s['ttft_median']:.3f} [{s['ttft_min']:.3f}, {s['ttft_max']:.3f}]"
                if s["ttft_median"] is not None
                else ("omitted" if timing_complete else "pending")
            )
            lines.append(
                f"| {length} | {'fp16' if bits == 16 else 'int' + str(bits)} | "
                f"{s['cache_bytes'] / 1e6:.2f} | {s['ppl']:.4f} | "
                f"{s['ppl_difference']:+.4f} [{low:+.4f}, {high:+.4f}] | "
                f"{decode} | {ttft} | "
                f"{s['peak_mlx_bytes'] / 1e9:.3f} | {s['peak_rss_bytes'] / 1e9:.3f} |"
            )
    lines += [
        "",
        "## Method and limits",
        "",
        "The grid was fixed before measurement. Cache modes are fp16,",
        "affine int8, and affine int4; quantized group size is 64. Every position is",
        "quantized from the start, with no eviction. The original target weights and",
        "model/data revisions remain fixed. The installed APIs and source were inspected.",
        "",
        f"Quality uses {count} disjoint test windows and {outputs} target tokens per window.",
        "These are conditional continuation perplexities, not the original core test PPL.",
        "Shorter contexts use suffixes of the same longest prompt. Each recorded window",
        "retains per-token NLL; mean token NLL is exponentiated once across all windows.",
        "Paired bootstrap intervals resample whole windows 10,000 times, with fixed seed.",
        "They are exploratory, unadjusted for multiple comparisons, and based on a small",
        "single-domain sample. Overlap with the core test stream is not independent replication.",
        "",
        f"Speed uses {prompt_count} fixed validation prompts, {1 if args.quick else 2} warmups,",
        f"and {repeats} measured repeats per case. A seeded shuffle changes case order in",
        "every repeat block. The table pools prompts/repeats and reports median [min, max];",
        "ranges include prompt and timing variability. Fresh caches are used on every run.",
        "Hardware temperature and background activity are not controlled.",
        "",
        "TTFT includes chunked prefill and the first greedy output. The prefill chunk cap",
        "is 512; intermediate logits are not materialized. Decode covers all remaining",
        "outputs with GPU evaluation and token transfer. EOS is ignored to hold work fixed.",
        "TTFT ends at first-token GPU completion; the first scalar read and cache accounting",
        "sit between timed phases. These intervals exclude tokenization and request serving.",
        "Compare timings within this extension; prefill differs from the core benchmark.",
        "",
        "Allocated cache bytes include scales, offsets, and unused capacity in 256-token",
        "allocation blocks. Memory peaks take the largest observed value across timing runs.",
        "Cache bytes, MLX allocator peaks, and sampled RSS are distinct and must not be added.",
        "Quantized attention uses a different kernel path than fp16 fast attention, so",
        "memory compression need not improve speed. Cache quantization can change outputs.",
        "",
        "![Cache memory and latency](figures/kv_cache_tradeoffs.png)",
        "",
        "![Paired continuation-quality differences](figures/kv_cache_quality.png)",
        "",
        f"Raw records: `{path.relative_to(ROOT)}`.",
        f"Timing runner SHA-256: `{context['kv_implementation_sha256']}`.",
        f"Quality source SHA-256: `{quality_identity['kv_implementation_sha256']}`.",
        "Recorded quality and timing source identities are linked in data/cache-sources.json.",
        "",
    ]
    output = ROOT / "figures" / ("smoke" if args.quick else "")
    output.mkdir(exist_ok=True, parents=True)
    table = report_path("cache.md", quick=args.quick)
    if args.quality_only:
        table = report_path("cache-quality.md", quick=args.quick)
        lines[2:2] = [
            "Quality/storage companion; completed timings are in RESULTS.md."
            if timing_complete
            else "Quality and cache storage are complete. Speed/TTFT are pending.",
            "Peak columns come from quality runs. Timing columns are intentionally omitted."
            if timing_complete
            else "Peak columns come from quality runs; timing is still planned.",
            "",
        ]
        lines = [
            line.replace("figures/kv_cache_tradeoffs.png", "figures/kv_cache_storage.png").replace(
                "Memory peaks take the largest observed value across timing runs.",
                "Memory peaks take the largest observed value across quality runs.",
            )
            for line in lines
        ]
    report_path(
        "cache-quality.json" if args.quality_only else "cache.json", quick=args.quick
    ).write_text(
        json.dumps(
            {
                "context": context,
                "target": q[0]["config"]["target"]["name"],
                "summaries": [
                    {"context_tokens": length, "cache_bits": bits, **summary}
                    for (length, bits), summary in sorted(summaries.items())
                ],
            },
            sort_keys=True,
        )
        + "\n"
    )
    table.write_text("\n".join(lines))
    plt.rcParams.update(
        {
            "svg.hashsalt": "quantlab",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    colors = {16: "#1b263b", 8: "#3278a0", 4: "#bd4935"}
    labels = {16: "fp16 cache", 8: "int8 cache", 4: "int4 cache (quality failure)"}

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "svg"):
            p = output / f"{name}.{ext}"
            fig.savefig(
                p, bbox_inches="tight", metadata={"Date": None} if ext == "svg" else None, dpi=180
            )
            if ext == "svg":
                p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines()) + "\n")
        plt.close(fig)

    fig, axes = plt.subplots(
        1, 1 if args.quality_only else 3, figsize=(6 if args.quality_only else 13, 4)
    )
    if not args.quality_only:
        fig.suptitle(f"medians across {prompt_count} prompts × {repeats} repeats")
    for ax, metric, scale, title, ylabel in zip(
        np.atleast_1d(axes),
        ("cache_bytes",) if args.quality_only else ("cache_bytes", "ttft_median", "decode_median"),
        (1e6,) if args.quality_only else (1e6, 1, 1),
        ("Allocated KV storage",)
        if args.quality_only
        else ("Allocated KV storage", "Time to first token", "Greedy decode"),
        ("Cache MB (decimal)",)
        if args.quality_only
        else ("Cache MB (decimal)", "Median seconds", "Median tokens/s"),
        strict=True,
    ):
        for bits in precisions:
            ax.plot(
                lengths,
                [summaries[(length, bits)][metric] / scale for length in lengths],
                "o-",
                color=colors[bits],
                label=labels[bits],
            )
        ax.set(
            xscale="log",
            xticks=lengths,
            xticklabels=[str(v) for v in lengths],
            xlabel="Context tokens",
            ylabel=ylabel,
            title=title,
        )
        ax.minorticks_off()
        ax.grid(alpha=0.2)
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=8)
    save(fig, "kv_cache_storage" if args.quality_only else "kv_cache_tradeoffs")
    quantized = [b for b in precisions if b != 16]
    fig, axes = plt.subplots(1, len(quantized), figsize=(6 * len(quantized), 4))
    for ax, bits in zip(np.atleast_1d(axes), quantized, strict=True):
        centers = np.array([summaries[(length, bits)]["ppl_difference"] for length in lengths])
        bounds = np.array([summaries[(length, bits)]["ppl_difference_ci95"] for length in lengths])
        # percentile intervals need not contain the point estimate: draw endpoints directly
        x = np.arange(len(lengths))
        ax.vlines(x, bounds[:, 0], bounds[:, 1], color=colors[bits])
        ax.plot(x, centers, "o", color=colors[bits], label=labels[bits])
        ax.axhline(0, color="0.4", ls="--", lw=0.8)
        ax.set(
            xticks=np.arange(len(lengths)),
            xticklabels=[str(v) for v in lengths],
            xlabel="Context tokens",
            ylabel="Conditional PPL difference vs fp16 cache",
            title=f"{labels[bits]}: paired bootstrap 95% intervals",
        )
        ax.grid(alpha=0.2)
    save(fig, "kv_cache_quality")
    audited_count = sum(map(len, quality.values())) + (
        0 if args.quality_only else sum(map(len, speed.values()))
    )
    print(f"Audited {audited_count} raw records; report: {table}")
    for length in lengths:
        base = summaries[(length, 16)]
        for bits in precisions:
            s = summaries[(length, bits)]
            ratio = (
                f"{s['decode_median'] / base['decode_median']:.3f}"
                if s["decode_median"]
                else ("omitted" if timing_complete else "pending")
            )
            print(
                f"L={length} kv{bits}: cache saving "
                f"{100 * (1 - s['cache_bytes'] / base['cache_bytes']):.2f}%, "
                f"decode ratio {ratio}, "
                f"PPL change {s['relative_ppl_change_percent']:+.3f}%"
            )


if __name__ == "__main__":
    main()
