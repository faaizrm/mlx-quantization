import argparse

from quantlab.config import QuantConfig
from quantlab.data import DEFAULT_MODEL, model_snapshot
from quantlab.results import ROOT, ResultStore, experiment_lock


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a deterministic fp16 generation check")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    from quantlab.memory import measure_memory
    from quantlab.model import clear_memory, load_config

    with experiment_lock():
        snapshot, revision = model_snapshot(args.model)
        config = QuantConfig("fp16")
        with measure_memory() as memory:
            model, tokenizer = load_config(snapshot, config)
            prompt = tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": (
                            "Explain in two sentences why quantization can make "
                            "a language model use less memory."
                        ),
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            text = generate(
                model, tokenizer, prompt=prompt, max_tokens=80, sampler=make_sampler(temp=0)
            )
        store = ResultStore(
            ROOT / "results" / "generation.jsonl", {"model": args.model, "model_revision": revision}
        )
        store.add(
            config.to_dict(),
            "generation_completed",
            1,
            "smoke",
            {"max_tokens": 80, "temperature": 0},
            text=text,
            prompt=prompt,
            memory=memory,
        )
        print(text, flush=True)
        del model
        clear_memory()


if __name__ == "__main__":
    main()
