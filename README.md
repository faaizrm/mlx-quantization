# MLX Mixed-Precision Quantization

I built this project to explore how much of a language model's quality can be kept
when most of its weights are stored at 4 bits. It uses MLX to profile and benchmark
Qwen2.5-1.5B-Instruct on Apple Silicon, then keeps the most sensitive projections at
8 bits under a fixed storage budget.

## Overview

The best mixed-precision configuration recovered **95.1% of the fp16-to-int4
perplexity gap** while using **48.9% less model storage than fp16**. The experiments
also cover KV-cache quantization and speculative decoding. [RESULTS.md](RESULTS.md)
walks through the approach, benchmarks, and tradeoffs.

## Models and weights

The main model is [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
Speculative decoding uses [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
as the draft model. Both checkpoints come directly from Qwen's official Hugging Face
repositories. The code downloads the original Safetensors, casts weights to fp16,
and applies quantization locally with MLX.

`make benchmark` downloads both models as needed, using the revisions pinned in
[data/sources.json](data/sources.json). To prepare the main model and datasets
separately, run `python scripts/prepare_data.py` after setup. Downloads are cached
under `.cache/huggingface/` unless `HF_HOME` is set.

## Setup

Requires an Apple Silicon Mac with Metal access and Python 3.12+. The recorded runs
used Python 3.13.5, MLX 0.32.3, and an M3 Pro with 18 GB unified memory.

```bash
git clone https://github.com/faaizrm/mlx-quantization.git
cd mlx-quantization
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps
make check
```

## Reproduce the results

The measurements are included. To rebuild the report and plots without downloading
a model or rerunning the experiments:

```bash
make report
```

<details>
<summary>Run the benchmarks from scratch</summary>

In a fresh clone, archive the included measurements and selections once, then run
the pipeline:

```bash
mkdir -p .cache
mkdir .cache/reference
mv results/*.jsonl .cache/reference/
mv data/selected.json data/fine-grained-selection.json .cache/reference/
make benchmark
```

This downloads the pinned model and dataset, runs a small check before each full
experiment, and rebuilds the report. Expect several hours, and avoid other heavy
workloads when measuring speed. If interrupted, rerun `make benchmark` to resume; don't
repeat the archive step. Calibration and test splits stay fixed in `data/`.

</details>

## Project structure

- `src/` — model loading, quantization, evaluation, and timing.
- `scripts/` — experiment runners and report generation.
- `data/` — fixed splits and precision configurations.
- `results/` — raw measurements; `figures/` — the main comparison plot.
- `tests/` — validation for the evaluation and reporting code.

`make report` also generates detailed tables and additional plots locally. To build
a wheel, install `uv` and run `make build`.
