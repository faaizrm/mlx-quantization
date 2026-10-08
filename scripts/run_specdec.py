"""Benchmark MLX speculative decoding with a calibration-selected mixed target"""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from quantlab.config import QuantConfig, uniform
from quantlab.data import DEFAULT_MODEL, model_snapshot, wikitext_chunks
from quantlab.results import ROOT, ResultStore, experiment_lock, hardware_metadata, read_records

DRAFT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


class CountedDraft:
    """Count actual draft forward calls after prefill: one proposal per call"""

    def __init__(self, model):
        self.model = model
        self.proposals = 0

    def __call__(self, *args, **kwargs):
        self.proposals += 1
        return self.model(*args, **kwargs)


def decode_once(model, draft, prompt: list[int], generated: int, draft_tokens: int) -> dict:
    import mlx.core as mx
    from mlx_lm.generate import generate_step, speculative_generate_step
    from mlx_lm.models.cache import make_prompt_cache

    from quantlab.memory import measure_memory

    stream = mx.default_stream(mx.gpu)
    tokens, accepted = [], []
    counted = CountedDraft(draft) if draft is not None else None
    with measure_memory() as memory:
        prefill_start = time.perf_counter()
        caches = []
        for network in (model, draft) if draft is not None else (model,):
            cache = make_prompt_cache(network)
            network(mx.array(prompt[:-1])[None], cache=cache)
            mx.eval([c.state for c in cache])
            caches.extend(cache)
        mx.synchronize(stream)
        prefill_seconds = time.perf_counter() - prefill_start
        last = mx.array(prompt[-1:])
        mx.eval(last)
        start = time.perf_counter()
        if draft is None:
            generator = generate_step(
                last, model, stream=stream, prompt_cache=caches, max_tokens=generated
            )
        else:
            generator = speculative_generate_step(
                last,
                model,
                counted,
                stream=stream,
                prompt_cache=caches,
                max_tokens=generated,
                num_draft_tokens=draft_tokens,
            )
        try:
            for item in generator:
                tokens.append(int(item[0]))
                accepted.append(bool(item[2]) if draft is not None else False)
        finally:
            generator.close()
        mx.synchronize(stream)
        seconds = time.perf_counter() - start
    if len(tokens) != generated:
        raise RuntimeError(f"Expected {generated} output tokens, got {len(tokens)}")
    proposals = counted.proposals if counted else 0
    if sum(accepted) > proposals:
        raise RuntimeError("More accepted tokens than draft proposals")
    return {
        "decode_tokens_per_second": generated / seconds,
        "decode_seconds": seconds,
        "prefill_seconds": prefill_seconds,
        "tokens": tokens,
        "accepted_flags": accepted,
        "accepted_tokens": sum(accepted),
        "proposed_tokens": proposals,
        "acceptance_rate": sum(accepted) / proposals if proposals else None,
        "draft_output_fraction": sum(accepted) / generated,
        **memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    with experiment_lock():
        run(args)


def run(args) -> None:
    from transformers import AutoTokenizer

    from quantlab.memory import check_memory, measure_memory
    from quantlab.model import clear_memory, load_config, parameter_bytes

    selected = json.loads((ROOT / "data" / "selected.json").read_text())
    source = b"".join(p.read_bytes() for p in sorted((ROOT / "src" / "quantlab").glob("*.py")))
    if selected["context"]["model"] != args.model or selected["selection_split"] != "calibration":
        raise RuntimeError("Run calibration selection for this model before speculative decoding")
    if selected["context"]["implementation_sha256"] != hashlib.sha256(source).hexdigest():
        raise RuntimeError("Core scientific implementation changed since calibration selection")
    target_cfg = QuantConfig.from_dict(selected["configs"][0])
    snapshot, revision = model_snapshot(args.model)
    if revision != selected["context"]["model_revision"]:
        raise RuntimeError("Target model revision changed since calibration selection")
    draft_snapshot, draft_revision = model_snapshot(DRAFT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
    # exact serialized tokenizer equality checks vocabulary IDs, merges, and preprocessing
    tokenizer_bytes = (snapshot / "tokenizer.json").read_bytes()
    if tokenizer_bytes != (draft_snapshot / "tokenizer.json").read_bytes():
        raise RuntimeError("Target and draft tokenizers differ; investigate before proceeding")
    chunks, data = wikitext_chunks(tokenizer, args.model)
    if data != selected["context"]["data"]:
        raise RuntimeError("Calibration manifest changed since selection")
    context = {
        **selected["context"],
        "protocol": "quantlab-specdec-v1",
        "hardware": hardware_metadata(),
        "draft_model": DRAFT_MODEL,
        "draft_revision": draft_revision,
        "specdec_implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tokenizer_bytes).hexdigest(),
    }
    path = ROOT / "results" / ("specdec-smoke.jsonl" if args.quick else "specdec.jsonl")
    store = ResultStore(path, context)
    generated, warmups, repeats = (16, 1, 2) if args.quick else (128, 2, 5)
    prompt_count = 1 if args.quick else 3
    modes = [("no-draft", None), ("draft-fp16", QuantConfig("fp16")), ("draft-int4", uniform(4))]
    if not args.quick:
        pilots = [
            r
            for r in read_records(ROOT / "results" / "specdec-smoke.jsonl")
            if all(r.get(k) == v for k, v in context.items())
        ]
        if len({r["config"]["name"] for r in pilots}) < len(modes):
            raise RuntimeError("Run --quick for all three modes before the full benchmark")
        slowest = min(r["value"] for r in pilots)
        estimate = len(modes) * prompt_count * (warmups + repeats) * generated / slowest / 60
        print(
            f"Pilot-derived decode estimate: {estimate:.1f} minutes, plus prefill and loads",
            flush=True,
        )
    references = {}
    for mode, draft_cfg in modes:
        config = {
            "name": mode,
            "target": target_cfg.to_dict(),
            "draft": draft_cfg.to_dict() if draft_cfg else None,
        }
        settings_list = [
            {
                "quick": args.quick,
                "prompt_index": i,
                "prompt_tokens": 512,
                "prompt_ids": chunks["calibration"][i][:-1].tolist(),
                "generated_tokens": generated,
                "draft_tokens_per_round": 2,
                "warmups": warmups,
                "repeats": repeats,
                "sampler": "greedy",
                "stop_at_eos": False,
                "timing": "mlx-lm-low-level-generator-after-511-token-prefill",
                "acceptance_definition": "accepted draft outputs / all draft forward proposals",
            }
            for i in range(prompt_count)
        ]
        todo = [
            s
            for s in settings_list
            if not store.has(config, "decode_tokens_per_second", "benchmark", s)
        ]
        if todo:
            print(f"LOAD {mode}", flush=True)
            with measure_memory() as load_memory:
                model, _ = load_config(snapshot, target_cfg)
                draft = load_config(draft_snapshot, draft_cfg)[0] if draft_cfg else None
            sizes = {
                "target_bytes": parameter_bytes(model),
                "draft_bytes": parameter_bytes(draft) if draft is not None else 0,
            }
            try:
                for settings in todo:
                    index = settings["prompt_index"]
                    print(f"RUN {mode} prompt {index}", flush=True)
                    samples = []
                    for repeat in range(warmups + repeats):
                        sample = decode_once(model, draft, settings["prompt_ids"], generated, 2)
                        check_memory()
                        key = (index, generated)
                        if mode == "no-draft" and key not in references:
                            references[key] = sample["tokens"]
                        if sample["tokens"] != references[key]:
                            raise RuntimeError(
                                f"Greedy token mismatch for {mode}, prompt {index}; investigate"
                            )
                        if repeat >= warmups:
                            samples.append(sample)
                    rates = [s["decode_tokens_per_second"] for s in samples]
                    record = store.add(
                        config,
                        "decode_tokens_per_second",
                        statistics.median(rates),
                        "benchmark",
                        settings,
                        samples=samples,
                        minimum=min(rates),
                        maximum=max(rates),
                        load_memory=load_memory,
                        parameter_bytes=sizes,
                        output_matches_no_draft=True,
                    )
                    print(f"DONE {mode} prompt {index}: {record['value']:.2f} tok/s", flush=True)
            finally:
                del model, draft
                clear_memory()
        # restore reference tokens when a previous baseline completed before interruption
        if mode == "no-draft":
            for settings in settings_list:
                key = store.key(config, "decode_tokens_per_second", "benchmark", settings)
                record = next(r for r in store.records if r["key"] == key)
                references[(settings["prompt_index"], generated)] = record["samples"][0]["tokens"]
    print(f"Complete: {path}", flush=True)


if __name__ == "__main__":
    main()
