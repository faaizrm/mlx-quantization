"""Exercise proposal accounting, rejection, and target-token preservation on a tiny model"""

import runpy

import mlx.core as mx
import pytest
from test_core import tiny_model

from quantlab.results import ROOT

decode_once = runpy.run_path(str(ROOT / "scripts" / "run_specdec.py"))["decode_once"]


@pytest.mark.parametrize("reject", [False, True])
def test_speculation_preserves_tokens_and_counts_all_proposals(reject):
    model = tiny_model()
    prompt = [1, 2, 3]
    baseline = decode_once(model, None, prompt, 5, 2)
    draft = tiny_model()
    if reject:
        wrong_token = next(i for i in range(128) if i not in baseline["tokens"])
        underlying = draft

        class WrongDraft:
            layers = underlying.layers

            def __call__(self, inputs, cache):
                underlying(inputs, cache=cache)
                logits = mx.where(mx.arange(128) == wrong_token, 0.0, -100.0)
                return mx.broadcast_to(logits, (*inputs.shape, 128))

        draft = WrongDraft()
    result = decode_once(model, draft, prompt, 5, 2)
    assert result["tokens"] == baseline["tokens"]
    if reject:
        assert result["accepted_tokens"] == 0
        assert result["proposed_tokens"] == 9  # 2+2+2+2+1, including rejected suffixes
        assert result["acceptance_rate"] == 0
    else:
        assert result["accepted_tokens"] == result["proposed_tokens"] == 4
        assert result["acceptance_rate"] == 1
        assert result["draft_output_fraction"] == 0.8
