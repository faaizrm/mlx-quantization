"""Reject misleading aggregate metrics, mismatched protocols, and unstable generation"""

import copy
import math
import sys

import pytest

from quantlab.results import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from report_fine_grained import audit_quality, audit_speed, paired_interval  # noqa: E402


def quality_row():
    return {
        "settings": {"chunk_count": 2},
        "value": math.exp(1.5),
        "sample": {"per_chunk_mean_nll": [1.0, 2.0], "chunks": 2, "tokens": 1024, "mean_nll": 1.5},
        "memory": {"peak_mlx_bytes": 10, "peak_rss_bytes": 20},
        "load_memory": {"peak_mlx_bytes": 20, "peak_rss_bytes": 30},
    }


def timing_rows():
    row = {
        "config": {"name": "model"},
        "value": 15.0,
        "settings": {
            "prompt_id": 0,
            "repeat": 0,
            "prompt_sha256": "fixed",
            "prompt_tokens": 512,
            "cache_bits": 16,
            "warmups": 1,
            "generated_tokens": 16,
            "repeats": 2,
        },
        "sample": {
            "tokens": list(range(16)),
            "decode_seconds": 1.0,
            "decode_tokens_per_second": 15.0,
            "ttft_seconds": 0.5,
            "peak_mlx_bytes": 100,
            "peak_rss_bytes": 200,
        },
    }
    other = copy.deepcopy(row)
    other["settings"]["repeat"] = 1
    return [row, other]


def test_report_recomputes_loss_and_rejects_bad_count_or_memory():
    row = quality_row()
    audit_quality(row)
    row["value"] *= 2
    with pytest.raises(ValueError, match="Perplexity"):
        audit_quality(row)
    row = quality_row()
    row["sample"]["tokens"] -= 1
    with pytest.raises(ValueError, match="token count"):
        audit_quality(row)
    row = quality_row()
    row["memory"]["peak_mlx_bytes"] = 10_000_000_001
    with pytest.raises(ValueError, match="memory"):
        audit_quality(row)


def test_report_rejects_partial_cohort_and_protocol_changes():
    rows = timing_rows()
    audit_speed(rows, ["model"], prompts=1, repeats=2, generated=16)
    with pytest.raises(ValueError, match="incomplete"):
        audit_speed(rows[:1], ["model"], prompts=1, repeats=2, generated=16)
    rows[1]["settings"]["warmups"] = 2
    with pytest.raises(ValueError, match="combine"):
        audit_speed(rows, ["model"], prompts=1, repeats=2, generated=16)


def test_report_rejects_token_disagreement():
    rows = timing_rows()
    rows[1]["sample"]["tokens"][0] = 999
    with pytest.raises(ValueError, match="tokens disagree"):
        audit_speed(rows, ["model"], prompts=1, repeats=2, generated=16)


def test_paired_interval_cancels_shared_chunk_difficulty():
    result = paired_interval([100.0, 2.0, 10.0], [101.0, 3.0, 11.0])
    assert result == {"mean": -1.0, "low": -1.0, "high": -1.0}
    with pytest.raises(ValueError, match="equal"):
        paired_interval([1.0], [1.0, 2.0])
