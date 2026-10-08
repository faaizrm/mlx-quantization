"""Keep measurement sources and timing protocols distinct"""

import json
import sys

from quantlab.results import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
from _report import quality_context, timing_protocol  # noqa: E402


def test_recorded_quality_source_preserves_other_identities(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/cache-sources.json").write_text(json.dumps({"timing": "quality"}))
    context = {
        "kv_implementation_sha256": "timing",
        "model": "fixed",
        "data": {"split": "test"},
        "hardware": {"chip": "test"},
        "versions": {"mlx": "pinned"},
    }
    assert quality_context(context, root=tmp_path) == {
        **context,
        "kv_implementation_sha256": "quality",
    }
    assert context["kv_implementation_sha256"] == "timing"
    context["kv_implementation_sha256"] = "new-run"
    assert quality_context(context, root=tmp_path) == context


def test_timing_protocol_preserves_all_shared_settings():
    first = {"prompt_id": 0, "repeat": 0, "prompt_sha256": "a", "warmups": 2, "repeats": 5}
    second = {**first, "prompt_id": 1, "repeat": 4, "prompt_sha256": "b"}
    assert timing_protocol(first) == timing_protocol(second)
    assert timing_protocol(first) != timing_protocol({**second, "warmups": 0})
    assert timing_protocol(first) != timing_protocol({**second, "backend": "different"})
