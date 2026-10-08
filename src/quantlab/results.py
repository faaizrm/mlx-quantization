"""Append-only measurements with explicit experiment identity and provenance"""

import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "quantlab-v1"


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str:
        p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else "uncommitted"

    return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}


def versions() -> dict:
    return {
        name: importlib.metadata.version(name)
        for name in ("mlx", "mlx-lm", "datasets", "transformers", "numpy")
    }


def hardware_metadata() -> dict:
    def sysctl(key: str) -> str:
        return subprocess.run(
            ["sysctl", "-n", key], check=True, capture_output=True, text=True
        ).stdout.strip()

    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "unified_memory_bytes": int(sysctl("hw.memsize")),
        "cpu_cores": int(sysctl("hw.ncpu")),
        "macos": platform.mac_ver()[0],
    }


def read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{i}; preserve and repair before resuming"
                ) from exc
    return records


class ResultStore:
    def __init__(self, path: Path, context: dict):
        self.path = path
        self.context = {"protocol": PROTOCOL, "versions": versions(), **context}
        self.records = read_records(path)
        self.keys = {r["key"] for r in self.records}

    def key(self, config: dict, metric: str, split: str, settings: dict) -> str:
        return digest(
            {
                **self.context,
                "config": config,
                "metric": metric,
                "split": split,
                "settings": settings,
            }
        )

    def has(self, config: dict, metric: str, split: str, settings: dict) -> bool:
        return self.key(config, metric, split, settings) in self.keys

    def add(
        self, config: dict, metric: str, value: float, split: str, settings: dict, **details
    ) -> dict:
        if not math.isfinite(value):
            raise ValueError(f"Non-finite measurement: {metric}={value}")
        key = self.key(config, metric, split, settings)
        record = {
            **self.context,
            **git_state(),
            "key": key,
            "timestamp": datetime.now(UTC).isoformat(),
            "platform": platform.platform(),
            "config": config,
            "metric": metric,
            "value": value,
            "split": split,
            "settings": settings,
            **details,
        }
        if key not in self.keys:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.records.append(record)
            self.keys.add(key)
        return record


@contextmanager
def experiment_lock():
    """The kernel releases this lock after a crash; the file alone is not a live lock"""
    with (ROOT / ".run.lock").open("a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another experiment is running") from exc
        f.seek(0)
        f.truncate()
        f.write(str(os.getpid()))
        f.flush()
        yield
