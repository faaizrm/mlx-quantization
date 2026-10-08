"""A budget selector must never choose using the held-out score or mixed provenance"""

import runpy

import pytest

from quantlab.results import ROOT

choose = runpy.run_path(str(ROOT / "scripts" / "select_config.py"))["choose"]


def rows(name, calibration, test, size=100, speed=50, hardware="a"):
    context = dict.fromkeys(
        ("model", "model_revision", "implementation_sha256", "data", "versions", "protocol"),
        "fixed",
    )
    context["hardware"] = hardware
    return [
        {
            **context,
            "config": {"name": name},
            "metric": metric,
            "split": split,
            "value": value,
            "key": f"{name}-{metric}-{split}-{hardware}",
        }
        for metric, split, value in [
            ("parameter_bytes", "none", size),
            ("perplexity", "calibration", calibration),
            ("perplexity", "test", test),
            ("decode_tokens_per_second", "benchmark", speed),
        ]
    ]


def test_selection_uses_calibration_even_when_test_order_is_reversed():
    result = choose(rows("a", 3, 9) + rows("b", 4, 2), 1, 40)
    assert result["config"]["name"] == "a"
    assert result["metrics"]["test_ppl"] == 9
    assert result["selection_split"] == "calibration"


def test_selection_enforces_both_constraints_and_avoids_old_hardware():
    data = (
        rows("a", 1, 1, size=2_000_000)
        + rows("b", 2, 2, speed=20)
        + rows("old", 1, 1, hardware="old")
        + rows("c", 3, 3)
    )
    result = choose(data, 1, 30)
    assert result["config"]["name"] == "c"
    assert result["qualifying_candidates"] == 1
    with pytest.raises(ValueError, match="No measured"):
        choose(data, 1, 100)
    with pytest.raises(ValueError, match="positive"):
        choose(data, float("nan"))
