"""Create a measured-results table without selecting configurations on test data"""

from _report import load_report, metric_values, report_path

from quantlab.results import ROOT, read_records


def main() -> None:
    args, records, output = load_report(__doc__)
    sizes = metric_values(records, "parameter_bytes", "none")
    metrics = {
        "test_ppl": metric_values(records, "perplexity", "test"),
        "cal_ppl": metric_values(records, "perplexity", "calibration"),
        "decode": metric_values(records, "decode_tokens_per_second", "benchmark"),
        "prefill": metric_values(records, "prefill_tokens_per_second", "benchmark"),
        "rss": metric_values(records, "peak_rss_bytes", "benchmark"),
        "mlx": metric_values(records, "peak_mlx_bytes", "benchmark"),
        "arc": metric_values(records, "arc_accuracy", "test"),
    }
    names = sorted(n for n in sizes if not n.startswith(("layer-", "module-")))
    lines = [
        "# Measured results" + (" — SMOKE ONLY" if args.quick else ""),
        "",
        f"Model: `{args.model}`. One protocol identity; missing entries mean not measured.",
        "",
        "| Config | Size MB | Calibration PPL | Test PPL | Prefill tok/s | "
        "Decode tok/s | Peak MLX GB | Peak RSS GB | ARC accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(metric, name, scale=1):
        return f"{metrics[metric][name] / scale:.4f}" if name in metrics[metric] else "—"

    for name in names:
        cells = [
            name,
            f"{sizes[name] / 1e6:.2f}",
            cell("cal_ppl", name),
            cell("test_ppl", name),
            cell("prefill", name),
            cell("decode", name),
            cell("mlx", name, 1e9),
            cell("rss", name, 1e9),
            cell("arc", name),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Speed and RSS entries are medians across measured repetitions; min/max and raw",
        "samples are retained in JSONL. Test PPL is not used to choose configurations.",
        "",
        f"Implementation: `{records[-1]['implementation_sha256']}`.",
        "",
    ]
    benchmark_rows = {
        r["config"]["name"]: r for r in records if r["metric"] == "decode_tokens_per_second"
    }
    lines += [
        "## Repeated benchmark measurements",
        "",
        "Each cell is median [minimum, maximum] across measured repetitions.",
        "",
        "| Config | Prefill tok/s | Decode tok/s | Peak MLX GB | Sampled peak RSS GB |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, record in sorted(benchmark_rows.items()):

        def spread(metric: str, scale: float = 1) -> str:
            values = record["details"][metric]
            return (
                f"{values['median'] / scale:.2f} "
                f"[{values['min'] / scale:.2f}, {values['max'] / scale:.2f}]"
            )

        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    spread("prefill_tokens_per_second"),
                    spread("decode_tokens_per_second"),
                    spread("peak_mlx_bytes", 1e9),
                    spread("peak_rss_bytes", 1e9),
                ]
            )
            + " |"
        )
    analysis_path = ROOT / "results" / ("analysis-smoke.jsonl" if args.quick else "analysis.jsonl")
    analyses = {
        r["config"]["name"]: r
        for r in read_records(analysis_path)
        if all(
            r[k] == records[-1][k]
            for k in ("model", "model_revision", "implementation_sha256", "data", "hardware")
        )
        and set(r["settings"]["source_keys"]).issubset({row["key"] for row in records})
    }
    if analyses:
        lines += [
            "",
            "## Paired ARC uncertainty",
            "",
            "Accuracy differences versus fp16, in percentage points. Paired percentile",
            "bootstrap intervals use 10,000 question resamples (seed 2026). Exact McNemar",
            "p-values are exploratory and are not adjusted for multiple comparisons.",
            "",
            "| Config | Difference pp | 95% paired interval pp | Wins / losses | Exact p |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, record in sorted(analyses.items()):
            if name == "fp16":
                continue
            low, high = record["ci95"]
            lines.append(
                f"| {name} | {record['value'] * 100:+.1f} | "
                f"[{low * 100:+.1f}, {high * 100:+.1f}] | "
                f"{record['wins']} / {record['losses']} | {record['exact_mcnemar_p']:.3f} |"
            )
    target = report_path("core.md", quick=args.quick)
    target.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Table generated at {target}")


if __name__ == "__main__":
    main()
