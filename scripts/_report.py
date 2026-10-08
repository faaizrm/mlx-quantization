"""Reporting helpers for consistent experiment identities"""

import argparse
import json
from pathlib import Path

from quantlab.data import DEFAULT_MODEL
from quantlab.results import ROOT, read_records


def load_report(description: str):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    path = ROOT / "results" / ("smoke.jsonl" if args.quick else "measurements.jsonl")
    records = [r for r in read_records(path) if r.get("model") == args.model]
    if not records:
        raise SystemExit(f"No measurements in {path}; run experiments first")
    latest = records[-1]
    identity = (
        "model_revision",
        "implementation_sha256",
        "data",
        "versions",
        "protocol",
        "hardware",
    )
    records = [r for r in records if all(r.get(k) == latest.get(k) for k in identity)]
    output = ROOT / "figures" / ("smoke" if args.quick else "")
    output.mkdir(parents=True, exist_ok=True)
    return args, records, output


def metric_values(records: list[dict], metric: str, split: str) -> dict:
    return {
        r["config"]["name"]: r["value"]
        for r in records
        if r["metric"] == metric and r["split"] == split
    }


def report_path(name: str, *, quick: bool = False) -> Path:
    """Keep detailed intermediate reports out of the published documentation"""
    directory = ROOT / ".cache" / "reports" / ("smoke" if quick else "full")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def quality_context(context: dict, *, root: Path = ROOT) -> dict:
    """Resolve the separately recorded quality source for the bundled cache measurements"""
    path = root / "data" / "cache-sources.json"
    sources = json.loads(path.read_text()) if path.exists() else {}
    source = context["kv_implementation_sha256"]
    return {**context, "kv_implementation_sha256": sources.get(source, source)}


def timing_protocol(settings: dict) -> dict:
    """Separate repeat and prompt coordinates from shared benchmark settings"""
    varying = {
        "prompt_id",
        "prompt_sha256",
        "repeat",
        "context_tokens",
        "window_index",
        "anchor",
        "target_sha256",
    }
    return {k: v for k, v in settings.items() if k not in varying}
