"""Audit each experiment and consolidate the measured results"""

import argparse
import json
import subprocess
import sys

from _report import report_path

from quantlab.data import DEFAULT_MODEL
from quantlab.results import ROOT


def tables(text: str) -> list[list[list[str]]]:
    """Read generated tables, rejecting ragged or malformed intermediates"""
    result, current = [], []
    for line in [*text.splitlines(), ""]:
        if line.startswith("| ") or line.startswith("|---"):
            current.append([cell.strip() for cell in line.strip("|").split("|")])
        elif current:
            if len(current) < 2 or any(len(row) != len(current[0]) for row in current):
                raise ValueError("Malformed generated result table")
            if any(not cell or set(cell) - {"-", ":", " "} for cell in current[1]):
                raise ValueError("Missing table separator")
            result.append([current[0], *current[2:]])
            current = []
    return result


def markdown(headers: list[str], rows: list[list[str]]) -> str:
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Result columns do not match headers")
    return "\n".join(
        "| " + " | ".join(row) + " |" for row in [headers, ["---"] * len(headers), *rows]
    )


def generate(model: str) -> str:
    # an optional reporter must not leave an old table that looks freshly audited
    names = ("core.md", "projection.md", "cache.md", "cache-quality.md", "speculative.md")
    for name in (*names, "projection.json", "cache.json", "cache-quality.json", "speculative.json"):
        report_path(name).unlink(missing_ok=True)
    commands = (
        ("analyze_results.py",),
        ("make_plots.py",),
        ("make_tables.py",),
        ("report_specdec.py",),
        ("report_kv_cache.py",),
        ("report_kv_cache.py", "--quality-only"),
        ("report_fine_grained.py",),
    )
    for script, *flags in commands:
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--model", model, *flags],
            cwd=ROOT,
            check=True,
        )
    reports = {name: tables(report_path(name).read_text()) for name in names}
    fine = {row[0]: row for row in reports["projection.md"][0][1:]}
    timing = {row[0]: row for row in reports["projection.md"][2][1:]}
    projection = json.loads(report_path("projection.json").read_text())
    cache = json.loads(report_path("cache.json").read_text())
    speculative = json.loads(report_path("speculative.json").read_text())
    audit, context = projection["audit"], projection["context"]
    if not audit["timing_complete"]:
        raise ValueError("Projection timing cohort is incomplete")
    raw = projection["test"]
    hardware = context["hardware"]
    improvement = audit["improvements"]["16"]
    saving = 100 * (
        1 - raw["fine-projection-16"]["parameter_bytes"] / raw["fp16"]["parameter_bytes"]
    )
    labels = {
        "fp16": "fp16",
        "int4-g64": "Uniform int4",
        "int8-g64": "Uniform int8",
        "fine-block-8": "Block allocation (8)",
        "fine-projection-8": "Projection allocation (8)",
        "fine-block-16": "Block allocation (16)",
        "fine-projection-16": "Projection allocation (16)",
    }
    table = markdown(
        ["Configuration", "Model MB", "Test perplexity", "Decode tokens/s"],
        [
            [label, fine[name][1], fine[name][3], timing.get(name, [name, "—"])[1]]
            for name, label in labels.items()
        ],
    )
    wins = [int(row[3].split("/")[0]) for row in reports["projection.md"][1][1:]]
    controls = (
        "Projection allocation beat all three matched random controls at both budgets."
        if wins == [3, 3]
        else f"Projection allocation beat {wins[0]}/3 and {wins[1]}/3 random controls."
    )
    summary = {(r["context_tokens"], r["cache_bits"]): r for r in cache["summaries"]}
    lengths = (512, 2048, 8192)
    if set(summary) != {(length, bits) for length in lengths for bits in (16, 8, 4)}:
        raise ValueError("Missing audited cache summaries")
    cache_saving = 100 * (
        1 - summary[(8192, 8)]["cache_bytes"] / summary[(8192, 16)]["cache_bytes"]
    )
    changes = [summary[(length, 8)]["relative_ppl_change_percent"] for length in lengths]
    slowdown = [
        100 * (1 - summary[(n, 8)]["decode_median"] / summary[(n, 16)]["decode_median"])
        for n in lengths
    ]
    draft_rows = [r for r in reports["speculative.md"][0][1:] if r[1] == "draft-int4"]
    speedups = [float(row[3].removesuffix("×")) for row in draft_rows]
    int4_ppl = [summary[(n, 4)]["ppl"] for n in lengths]
    projection_speed = float(timing["fine-projection-16"][1])
    block_speed = float(timing["fine-block-16"][1])
    return f"""# Mixed-precision quantization on an M3 Pro

I wanted to see whether choosing *which* weights stay at higher precision could
recover most of a model's quality without restoring its full storage footprint.
These experiments use `{model}`, MLX, and an {hardware["chip"]}
with {hardware["unified_memory_bytes"] / 2**30:.0f} GiB unified memory.

## Choosing where to spend the bits

Uniform int4 saves space, but some weights are more sensitive to quantization than
others. The first approach promoted entire transformer blocks to int8. A finer
approach profiled all **196 layer/projection pairs**, then used a 0/1 knapsack to
choose which projections to promote under an exact storage budget. The remaining
projections stayed at 4 bits; tied embeddings stayed at fp16.

Configuration selection used WikiText-2 validation only: 128 calibration chunks,
with the first 32 used for the projection map. Final quality was measured on 256
held-out test chunks of 512 scored tokens each. The original baselines and selected
whole-block configurations also ran on 500 ARC-Easy questions.

## Better quality at the same size

Projection allocation recovered **{100 * improvement["ppl_gap_recovery_fraction"]:.1f}%
of the fp16-to-int4 perplexity gap** with **{saving:.1f}% less model storage than fp16**.
At {fine["fine-projection-16"][1]} MB, test perplexity fell from
{fine["fine-block-16"][3]} with whole-block promotion to {fine["fine-projection-16"][3]}.
Lower perplexity is better; fp16 still had the best score.

{table}

Both budgets match the storage cost of promoting 8 or 16 whole blocks, with group
size 64. Model MB includes quantization metadata and unquantized weights; it isn't
peak runtime memory. Fp16 timing wasn't repeated in this comparison.

{controls}
Paired bootstrap intervals also supported the quality improvement over whole-block
allocation. Decode speed was similar: {projection_speed:.1f} versus {block_speed:.1f} tokens/s
at the larger budget, with overlapping repeat ranges. The gain here was quality at a fixed size.

![Quality at matched storage budgets](figures/fine_grained_allocation.png)

## Smaller caches and a draft model

**KV-cache quantization.** With `{cache["target"]}` weights fixed, the cache tests
covered 512-, 2,048-, and 8,192-token contexts with eight paired continuation windows.
Int8 saved **{cache_saving:.1f}% of cache storage**, with
{min(changes):.2f}–{max(changes):.2f}% higher continuation perplexity, but decoding was
{min(slowdown):.1f}–{max(slowdown):.1f}% slower. Int4 cache failed badly here:
perplexity reached {min(int4_ppl):,.0f}–{max(int4_ppl):,.0f}. An independently dequantized reference
reproduced the failure on calibration data. A smaller cache didn't guarantee lower
total peak memory or faster attention.

**Speculative decoding.** Adding a quantized 0.5B draft model to
`{speculative["target"]}` gave **{min(speedups):.2f}–{max(speedups):.2f}× decode
speedup across three prompts**.
Every greedy output token matched the target-only baseline. The fp16 draft was less
consistent and sometimes slower than no draft.

## Benchmark details and limits

Timing used two warmups and five measured repeats. Projection tests
used three 512-token prompts and 128 generated tokens (90 samples); the cache grid
had 135 timing samples. First-token GPU latency and the next 127 decode steps were
measured separately. The speculative generator used a different timing boundary,
so speed comparisons stay within each experiment. Memory records include actual
array storage, MLX peak allocation, and process RSS.

The projection allocations haven't been tested on ARC-Easy or other corpora.
Thermals and background activity weren't controlled. These results describe one
model on one machine, and the allocation relies on a small calibration sample.
The [raw measurements](results/) and fixed [splits and configurations](data/) are
included; `make report` rebuilds this write-up and the detailed tables locally.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    result = generate(args.model)
    target = ROOT / "RESULTS.md"
    temporary = target.with_suffix(".md.tmp")
    temporary.write_text(result)
    temporary.replace(target)
    print(f"Consolidated results: {target}")


if __name__ == "__main__":
    main()
