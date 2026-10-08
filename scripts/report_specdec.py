"""Regenerate speculative-decoding figures and tables from recorded measurements"""

import argparse
import json
import os
import statistics

from _report import report_path

from quantlab.data import DEFAULT_MODEL
from quantlab.results import ROOT, read_records

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    source = ROOT / "results" / ("specdec-smoke.jsonl" if args.quick else "specdec.jsonl")
    rows = [r for r in read_records(source) if r["model"] == args.model]
    if not rows:
        print("Speculative-decoding results are not present; optional report skipped")
        return
    keys = (
        "model_revision",
        "draft_revision",
        "implementation_sha256",
        "specdec_implementation_sha256",
        "data",
        "hardware",
        "versions",
        "protocol",
        "tokenizer_sha256",
    )
    rows = [r for r in rows if all(r[k] == rows[-1][k] for k in keys)]
    indexed = {(r["config"]["name"], r["settings"]["prompt_index"]): r for r in rows}
    modes = ["no-draft", "draft-fp16", "draft-int4"]
    prompts = list(range(1 if args.quick else 3))
    if set(indexed) != {(mode, prompt) for mode in modes for prompt in prompts}:
        raise RuntimeError(
            "Complete all planned prompts/modes before reporting speculative decoding"
        )
    acceptance = {}
    for (mode, prompt), row in indexed.items():
        samples = row["samples"]
        assert len(samples) == row["settings"]["repeats"]
        assert row["value"] == statistics.median(s["decode_tokens_per_second"] for s in samples)
        reference = indexed[("no-draft", prompt)]["samples"][0]["tokens"]
        for sample in samples:
            assert sample["tokens"] == reference
            assert len(reference) == row["settings"]["generated_tokens"]
            assert sample["accepted_tokens"] == sum(sample["accepted_flags"])
            assert sample["accepted_tokens"] <= sample["proposed_tokens"]
        proposals = sum(s["proposed_tokens"] for s in samples)
        accepted = sum(s["accepted_tokens"] for s in samples)
        acceptance[(mode, prompt)] = accepted / proposals if proposals else 0.0
    lines = [
        "# Speculative decoding" + (" — SMOKE ONLY" if args.quick else ""),
        "",
        f"Target: `{rows[-1]['config']['target']['name']}` on `{args.model}`. "
        f"Draft: `{rows[-1]['draft_model']}`. All tokens match the no-draft reference.",
        "",
        "| Prompt | Mode | Decode tok/s [min, max] | Speedup | Acceptance | "
        "Combined model MB | Peak MLX GB | Sampled peak RSS GB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for prompt in prompts:
        for mode in modes:
            row = indexed[(mode, prompt)]
            speedup = row["value"] / indexed[("no-draft", prompt)]["value"]
            rate = f"{acceptance[(mode, prompt)] * 100:.1f}%" if mode != "no-draft" else "—"
            lines.append(
                f"| {prompt} | {mode} | {row['value']:.2f} "
                f"[{row['minimum']:.2f}, {row['maximum']:.2f}] | "
                f"{speedup:.3f}× | {rate} | {sum(row['parameter_bytes'].values()) / 1e6:.2f} | "
                f"{max(s['peak_mlx_bytes'] for s in row['samples']) / 1e9:.3f} | "
                f"{max(s['peak_rss_bytes'] for s in row['samples']) / 1e9:.3f} |"
            )
    lines += [
        "",
        "## Protocol and interpretation",
        "",
        "The target was chosen solely by calibration perplexity in the core study. The first",
        "three saved calibration windows supply distinct 512-token prompts. Target and draft",
        "tokenizer.json files are byte-identical. Each mode uses fresh caches, greedy outputs,",
        "two draft proposals per round, and fixed output length, ignoring EOS. Both target and",
        "draft caches are prefilled on the first 511 prompt tokens outside the decode timer.",
        "",
        "The timer covers 128 generated outputs beginning with the last prompt-token query,",
        "including generator cleanup and GPU synchronization. Each prompt/mode has two warmups",
        "and five timed repetitions. The built-in no-draft generator performs asynchronous",
        "lookahead; its extra work is included. Compare speedups within this table: these are",
        "library-generator timings, not the core custom loop's 127 post-first-token steps.",
        "",
        "Acceptance is accepted draft outputs divided by all actual draft forward proposals,",
        "including rejected suffixes. The counter starts after prefill. It is distinct from the",
        "fraction of final output supplied by the draft. Raw counts, flags, tokens, times,",
        "and memory samples are saved in the JSONL. All warmup and measured outputs were",
        "checked against no-draft tokens. Peak columns take the maximum across measured repeats.",
        "",
        "Three prompts from one domain provide a small workload sample. Fixed mode order and",
        "uncontrolled temperature limit timing conclusions. No draft-length tuning was performed.",
        "Figure whiskers divide each mode's observed timing range by the no-draft median;",
        "they are descriptive ranges and do not propagate uncertainty in that denominator.",
        "",
        f"Source: `{source.relative_to(ROOT)}`.",
        f"Benchmark code SHA-256: `{rows[-1]['specdec_implementation_sha256']}`.",
        "",
    ]
    if args.quick:
        lines.insert(2, "Pilot override: one prompt, 16 outputs, one warmup, two repetitions.\n")
    output = ROOT / "figures" / ("smoke" if args.quick else "")
    output.mkdir(parents=True, exist_ok=True)
    table = report_path("speculative.md", quick=args.quick)
    report_path("speculative.json", quick=args.quick).write_text(
        json.dumps(
            {
                "target": rows[-1]["config"]["target"]["name"],
                "draft": rows[-1]["draft_model"],
            },
            sort_keys=True,
        )
        + "\n"
    )
    table.write_text("\n".join(lines))
    plt.rcParams.update(
        {"svg.hashsalt": "quantlab", "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=150)
    colors = ["#1b263b", "#3278a0", "#bd4935"]
    x = np.arange(len(prompts))
    for i, mode in enumerate(modes):
        medians = np.array([indexed[(mode, p)]["value"] for p in prompts])
        low = np.array([indexed[(mode, p)]["minimum"] for p in prompts])
        high = np.array([indexed[(mode, p)]["maximum"] for p in prompts])
        reference = np.array([indexed[("no-draft", p)]["value"] for p in prompts])
        axes[0].bar(
            x + (i - 1) * 0.25,
            medians / reference,
            width=0.23,
            color=colors[i],
            label=mode,
            yerr=np.array([medians - low, high - medians]) / reference,
            capsize=3,
        )
        if mode != "no-draft":
            axes[1].bar(
                x + (i - 1.5) * 0.3,
                [100 * acceptance[(mode, p)] for p in prompts],
                width=0.28,
                color=colors[i],
                label=mode,
            )
    for ax in axes:
        ax.set_xticks(x, [f"Prompt {p}" for p in prompts])
        ax.legend(frameon=False, fontsize=8)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].axhline(1.0, color="0.4", ls="--", lw=0.8)
    axes[0].set(ylabel="Decode speedup vs no draft (×)", title="Speedup (whiskers: timing range)")
    axes[1].set(
        ylabel="Accepted / proposed draft tokens (%)", ylim=(0, 100), title="Draft acceptance"
    )
    fig.tight_layout()
    for extension in ("png", "svg"):
        path = output / f"specdec.{extension}"
        fig.savefig(
            path, bbox_inches="tight", metadata={"Date": None} if extension == "svg" else None
        )
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    plt.close(fig)
    print(f"Audited {len(rows)} speculative measurements; report: {table}")
    print(
        json.dumps(
            {
                mode: {
                    "speedup_range": [
                        min(
                            indexed[(mode, p)]["value"] / indexed[("no-draft", p)]["value"]
                            for p in prompts
                        ),
                        max(
                            indexed[(mode, p)]["value"] / indexed[("no-draft", p)]["value"]
                            for p in prompts
                        ),
                    ],
                    "acceptance_range": [
                        min(acceptance[(mode, p)] for p in prompts),
                        max(acceptance[(mode, p)] for p in prompts),
                    ],
                }
                for mode in modes[1:]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
