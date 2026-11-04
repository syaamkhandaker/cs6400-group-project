"""QASPER answer scoring, following the official ``qasper_evaluator.py`` (v0.3).
Every annotation becomes one reference with a type: ``extractive`` (spans
joined by ", "), ``abstractive`` (free-form), ``boolean`` ("Yes"/"No") or
``none`` ("Unanswerable"). A prediction's Answer-F1 is its best token F1 over
the references, and that best reference's type is the one it is counted under,
as in the official script.

Two decisions the proposal asks for are scored on top:
* yes/no exact match, over questions whose first annotation is boolean
* the unanswerable decision, over all questions (gold: first annotation is ``none``)
"""

from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np

ANSWER_TYPES = ("extractive", "abstractive", "boolean", "none")
_PUNCT = set(string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    """F1 of the shared words between two normalized answers."""
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    common = sum((Counter(pred) & Counter(gold)).values())
    if common == 0:
        return 0.0
    p, r = common / len(pred), common / len(gold)
    return 2 * p * r / (p + r)


def qasper_references(qa: dict) -> list[dict[str, str]]:
    """``[{answer, type}]`` for every annotation of one QASPER question."""
    refs = []
    for entry in qa["answers"]:
        a = entry["answer"]
        # Same check order as the official evaluator.
        if a.get("unanswerable"):
            refs.append({"answer": "Unanswerable", "type": "none"})
        elif a.get("extractive_spans"):
            refs.append({"answer": ", ".join(a["extractive_spans"]), "type": "extractive"})
        elif a.get("free_form_answer"):
            refs.append({"answer": a["free_form_answer"], "type": "abstractive"})
        elif a.get("yes_no") is not None:
            refs.append({"answer": "Yes" if a["yes_no"] else "No", "type": "boolean"})
    return refs


def predicts_unanswerable(prediction: str) -> bool:
    """True if the answer starts with "Unanswerable"."""
    return normalize_answer(prediction).startswith("unanswerable")


def yes_no(prediction: str) -> str | None:
    """Return "yes" or "no" if the answer starts with it, else None."""
    words = normalize_answer(prediction).split()
    return words[0] if words and words[0] in ("yes", "no") else None


def score_prediction(prediction: str, references: list[dict[str, str]]) -> dict:
    """Answer-F1 and its type (official), plus the first-annotation decisions."""
    # The best-matching reference gives the F1 and the type it is counted under.
    f1s = [(token_f1(prediction, ref["answer"]), ref["type"]) for ref in references]
    best_f1, best_type = sorted(f1s, key=lambda x: x[0], reverse=True)[0]
    # The first annotation decides the yes/no and unanswerable labels.
    first = references[0]
    out = {
        "f1": best_f1,
        "f1_type": best_type,
        "first_type": first["type"],
        "gold_unanswerable": first["type"] == "none",
        "pred_unanswerable": predicts_unanswerable(prediction),
    }
    if first["type"] == "boolean":
        out["yes_no_correct"] = float(yes_no(prediction) == first["answer"].lower())
    return out


def aggregate_qa(scored: list[dict]) -> dict:
    """Mean Answer-F1 overall and per type, yes/no EM, and unanswerable-decision accuracy."""
    out: dict = {"n": len(scored), "answer_f1": float(np.mean([s["f1"] for s in scored])) if scored else 0.0}
    for t in ANSWER_TYPES:
        vals = [s["f1"] for s in scored if s["f1_type"] == t]
        out[f"f1_{t}"] = float(np.mean(vals)) if vals else None
        out[f"n_{t}"] = len(vals)
    yn = [s["yes_no_correct"] for s in scored if "yes_no_correct" in s]
    out["yes_no_em"] = float(np.mean(yn)) if yn else None
    out["n_yes_no"] = len(yn)
    out["unanswerable_acc"] = (
        float(np.mean([s["gold_unanswerable"] == s["pred_unanswerable"] for s in scored])) if scored else None
    )
    # cites_gold is set by run_qa.py and is None when no gold paragraph was matched.
    cited = [s["cites_gold"] for s in scored if s.get("cites_gold") is not None]
    out["cites_gold_rate"] = float(np.mean(cited)) if cited else None
    out["n_cites_gold"] = len(cited)
    return out
