"""Download pinned public weights and datasets into the project cache"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quantlab.data import DEFAULT_MODEL, arc_questions, model_snapshot, wikitext_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    snapshot, revision = model_snapshot(args.model)
    print(f"Model snapshot: {snapshot} ({revision})", flush=True)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
    _, settings = wikitext_chunks(tokenizer, args.model, quick=args.quick)
    print(f"WikiText prepared: {settings}", flush=True)
    _, settings = arc_questions(count=5 if args.quick else 500)
    print(f"ARC prepared: {settings}", flush=True)


if __name__ == "__main__":
    main()
