"""Zero-shot ARC-Easy, scored on answer continuation tokens only"""

import mlx.core as mx
import mlx.nn as nn

from quantlab.memory import check_memory


def continuation_score(model, tokenizer, prompt: str, answer: str) -> float:
    # separate tokenization fixes the answer boundary and is identical for every config
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    suffix = tokenizer.encode(" " + answer, add_special_tokens=False)
    if not prefix or not suffix:
        raise ValueError("Empty prompt or answer tokens")
    tokens = mx.array(prefix + suffix)[None, :]
    logits = model(tokens[:, :-1])[:, len(prefix) - 1 :, :].astype(mx.float32)
    nll = nn.losses.cross_entropy(logits, mx.array(suffix)[None, :], reduction="mean")
    return -float(nll.item())


def arc_accuracy(model, tokenizer, questions: list[dict]) -> dict:
    if not questions:
        raise ValueError("No ARC questions")
    correct = 0
    predictions = []
    for index, question in enumerate(questions):
        prompt = f"Question: {question['question']}\nAnswer:"
        scores = [
            continuation_score(model, tokenizer, prompt, answer)
            for answer in question["choices"]["text"]
        ]
        winner = max(range(len(scores)), key=scores.__getitem__)
        label = question["choices"]["label"][winner]
        correct += label == question["answerKey"]
        predictions.append(
            {
                "id": question["id"],
                "prediction": label,
                "gold": question["answerKey"],
                "scores": scores,
            }
        )
        check_memory()
        if (index + 1) % 25 == 0:
            print(f"  ARC {index + 1}/{len(questions)}", flush=True)
    return {
        "accuracy": correct / len(questions),
        "correct": correct,
        "questions": len(questions),
        "predictions": predictions,
    }
