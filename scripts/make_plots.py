"""Regenerate figures from a single recorded experiment identity"""

import os

from quantlab.results import ROOT

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from _report import load_report, metric_values


def main() -> None:
    args, records, output = load_report(__doc__)
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 180,
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "quantlab",
        }
    )
    calibration = metric_values(records, "perplexity", "calibration")
    suffix = " (SMOKE ONLY)" if args.quick else ""

    def save(fig, name: str):
        fig.tight_layout()
        for extension in ("png", "svg"):
            path = output / f"{name}.{extension}"
            fig.savefig(
                path, bbox_inches="tight", metadata={"Date": None} if extension == "svg" else None
            )
            if extension == "svg":
                path.write_text(
                    "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
                )
        plt.close(fig)

    if "fp16" in calibration:
        for prefix, name, title in (
            ("layer-", "layer_sensitivity", "Block sensitivity"),
            ("module-", "module_sensitivity", "Module-type sensitivity"),
        ):
            values = sorted((k, v) for k, v in calibration.items() if k.startswith(prefix))
            if values:
                fig, ax = plt.subplots(figsize=(10, 4))
                labels = [k.removeprefix(prefix).removesuffix("-int4") for k, _ in values]
                deltas = [v - calibration["fp16"] for _, v in values]
                ax.bar(labels, deltas, color=["#3278a0" if d >= 0 else "#d18439" for d in deltas])
                ax.axhline(0, color="0.4", lw=0.8)
                ax.set(ylabel="Calibration perplexity increase over fp16", title=title + suffix)
                save(fig, name)

    sizes = metric_values(records, "parameter_bytes", "none")
    test = metric_values(records, "perplexity", "test")
    if test:
        fig, ax = plt.subplots(figsize=(8, 5))
        groups = {
            "fp16": ("#1b263b", "s"),
            "uniform": ("#3278a0", "o"),
            "guided mixed": ("#bd4935", "D"),
            "random control": ("#aaa", "x"),
        }
        used = set()
        for name, ppl in sorted(test.items()):
            if name not in sizes:
                continue
            group = (
                "random control"
                if "random" in name
                else "guided mixed"
                if name.startswith("mixed")
                else "fp16"
                if name == "fp16"
                else "uniform"
            )
            color, marker = groups[group]
            ax.scatter(
                sizes[name] / 1e6,
                ppl,
                c=color,
                marker=marker,
                label=group if group not in used else None,
                s=55,
            )
            used.add(group)
            if "random" not in name:
                ax.annotate(
                    name,
                    (sizes[name] / 1e6, ppl),
                    xytext=(-4, -14) if name == "int8-g64" else (4, 5),
                    textcoords="offset points",
                    fontsize=8,
                    ha="right" if name == "int8-g64" else "left",
                )
        ax.set(
            xlabel="Actual parameter storage (MB, decimal)",
            ylabel="Held-out test perplexity",
            title="Quality versus model storage" + suffix,
        )
        ax.margins(y=0.10)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        save(fig, "pareto")

    decode = metric_values(records, "decode_tokens_per_second", "benchmark")
    memory = metric_values(records, "peak_mlx_bytes", "benchmark")
    if decode:
        names = sorted(decode, key=decode.__getitem__)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].barh(names, [decode[n] for n in names], color="#3278a0")
        axes[0].set(xlabel="Median decode tokens/s", title="Greedy decode" + suffix)
        axes[1].barh(names, [memory[n] / 1e9 for n in names], color="#bd4935")
        axes[1].set(xlabel="Median peak MLX allocation (GB)", title="MLX memory" + suffix)
        save(fig, "speed_memory")

    if not args.quick and any(n.startswith("mixed-random") for n in calibration):
        import numpy as np

        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
        for ax, values, title in zip(
            axes, (calibration, test), ("Calibration", "Held-out test"), strict=True
        ):
            ks = [
                k
                for k in (2, 4, 6, 8, 12, 16)
                if all(f"mixed-random-{k}-seed-{s}" in values for s in range(3))
            ]
            if not ks:
                continue
            controls = np.array(
                [[values[f"mixed-random-{k}-seed-{s}"] for s in range(3)] for k in ks]
            )
            ax.fill_between(
                ks, controls.min(1), controls.max(1), color="0.85", label="Random range (3 seeds)"
            )
            ax.plot(ks, controls.mean(1), "x--", color="0.45", label="Random mean")
            guided_ks = [k for k in ks if f"mixed-top-{k}" in values]
            ax.plot(
                guided_ks,
                [values[f"mixed-top-{k}"] for k in guided_ks],
                "D-",
                color="#bd4935",
                label="Sensitivity guided",
            )
            ax.set(xlabel="Blocks promoted to 8-bit", ylabel="Perplexity", title=title + suffix)
            ax.set_xticks(ks)
            ax.grid(alpha=0.2)
        axes[0].legend(frameon=False, fontsize=8)
        save(fig, "random_controls")
    print(f"Figures generated in {output}")


if __name__ == "__main__":
    main()
