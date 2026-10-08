"""Pinned sources and immutable, token-level calibration/test manifests"""

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

from quantlab.results import ROOT, digest

os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def pinned_revision(repo: str, kind: str = "model") -> str:
    from huggingface_hub import HfApi

    path = ROOT / "data" / "sources.json"
    sources = json.loads(path.read_text()) if path.exists() else {}
    key = f"{kind}:{repo}"
    if key not in sources:
        info = HfApi().repo_info(repo, repo_type=kind)
        sources[key] = info.sha
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sources, indent=2, sort_keys=True) + "\n")
    return sources[key]


def model_snapshot(model_id: str) -> tuple[Path, str]:
    from huggingface_hub import snapshot_download

    revision = pinned_revision(model_id)
    path = snapshot_download(
        model_id,
        revision=revision,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
    )
    return Path(path), revision


def chunk_indices(total_tokens: int, chunk_size: int, count: int, seed: int) -> list[int]:
    available = (total_tokens - 1) // chunk_size
    if available < count:
        raise ValueError(f"Need {count} chunks; corpus has only {available}")
    return random.Random(seed).sample(range(available), count)


def wikitext_chunks(
    tokenizer,
    model_id: str,
    *,
    chunk_size: int = 512,
    calibration_count: int = 128,
    test_count: int = 256,
    seed: int = 2026,
    quick: bool = False,
) -> tuple[dict, dict]:
    from datasets import load_dataset

    repo = "Salesforce/wikitext"
    revision = pinned_revision(repo, "dataset")
    settings = {
        "dataset": repo,
        "revision": revision,
        "dataset_config": "wikitext-2-raw-v1",
        "model": model_id,
        "tokenizer_revision": pinned_revision(model_id),
        "chunk_size": chunk_size,
        "calibration_count": calibration_count,
        "test_count": test_count,
        "seed": seed,
        "join": "\\n\\n",
        "add_special_tokens": False,
    }
    manifest_path = ROOT / "data" / "splits.json"
    manifests = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    key = digest(settings)
    previous = manifests.get(key)
    manifest = {"settings": settings, "splits": {}}
    chunks = {}
    for split, source_split, count in (
        ("calibration", "validation", calibration_count),
        ("test", "test", test_count),
    ):
        cache = (
            ROOT / "data" / f"tokens-{digest({**settings, 'source_split': source_split})[:16]}.npy"
        )
        if cache.exists():
            tokens = np.load(cache)
        else:
            dataset = load_dataset(repo, "wikitext-2-raw-v1", split=source_split, revision=revision)
            text = "\n\n".join(dataset["text"])
            tokens = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
            np.save(cache, tokens)
        indices = chunk_indices(len(tokens), chunk_size, count, seed)
        manifest["splits"][split] = {
            "source_split": source_split,
            "chunk_indices": indices,
            "token_count": len(tokens),
            "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        }
        chosen = indices[:2] if quick else indices
        # 512 inputs plus the next token give exactly 512 scored targets per chunk
        chunks[split] = [tokens[i * chunk_size : i * chunk_size + chunk_size + 1] for i in chosen]
    if previous is not None and previous != manifest:
        raise RuntimeError(
            "Saved split manifest disagrees with tokenized data; investigate before proceeding"
        )
    if previous is None:
        manifests[key] = manifest
        manifest_path.write_text(json.dumps(manifests, indent=2, sort_keys=True) + "\n")
    return chunks, {
        "manifest_id": key,
        "manifest_sha256": digest(manifest),
        "quick": quick,
        "chunk_size": chunk_size,
        "calibration_count": len(chunks["calibration"]),
        "test_count": len(chunks["test"]),
    }


def arc_questions(count: int = 500, seed: int = 2026) -> tuple[list[dict], dict]:
    from datasets import load_dataset

    repo = "allenai/ai2_arc"
    revision = pinned_revision(repo, "dataset")
    dataset = load_dataset(repo, "ARC-Easy", split="test", revision=revision)
    indices = random.Random(seed).sample(range(len(dataset)), min(count, len(dataset)))
    questions = [dataset[i] for i in indices]
    manifest = {
        "dataset": repo,
        "revision": revision,
        "split": "test",
        "seed": seed,
        "indices": indices,
        "ids": [q["id"] for q in questions],
    }
    path = ROOT / "data" / f"arc-{count}-{seed}.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise RuntimeError("ARC manifest changed")
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return questions, {
        "manifest_sha256": digest(manifest),
        "count": len(questions),
        "scoring": "mean-answer-token-log-likelihood-v1",
    }
