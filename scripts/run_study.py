"""Run the core quality, sensitivity, and speed benchmarks in order"""

import argparse
import subprocess
import sys

from quantlab.data import DEFAULT_MODEL
from quantlab.results import ROOT, read_records


def check_baselines(model: str, quick: bool) -> None:
    path = ROOT / "results" / ("smoke.jsonl" if quick else "measurements.jsonl")
    records = [r for r in read_records(path) if r["model"] == model]
    latest = records[-1]
    values = {
        r["config"]["name"]: r["value"]
        for r in records
        if r["metric"] == "perplexity"
        and r["split"] == "test"
        and r["implementation_sha256"] == latest["implementation_sha256"]
        and r["data"] == latest["data"]
    }
    for name, value in values.items():
        if name.startswith("int4") and value < values["fp16"]:
            raise RuntimeError(f"Investigate {name} beating fp16 before continuing")
        if name.startswith("int") and value > 10 * values["fp16"]:
            raise RuntimeError(f"Investigate large quality degradation for {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-speed", action="store_true")
    args = parser.parse_args()
    common = ["--model", args.model] + (["--quick"] if args.quick else [])

    def run(script, *flags):
        print(f"RUN {script}", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(ROOT / "scripts" / script), *common, *flags],
            cwd=ROOT,
            check=True,
        )

    run("run_baselines.py", "--no-speed")
    check_baselines(args.model, args.quick)
    run("run_sensitivity.py", "--no-speed")
    run("run_mixed_sweep.py", "--no-speed")
    run("run_arc.py", "--no-speed")
    if not args.no_speed:
        run("run_baselines.py")
        run("run_mixed_sweep.py")
    run("analyze_results.py")
    run("make_plots.py")
    run("make_tables.py")


if __name__ == "__main__":
    main()
