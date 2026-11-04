"""Build evaluation queries with gold relevance from the raw benchmark files.

QASPER
    Gold evidence strings are matched to paragraph *units* of the cached corpus
    (see ``rag.retrieval.corpus``). A paragraph's text is the join of its chunk
    bodies. Matching tries a whitespace/case-normalised exact match first, then
    token F1 >= ``tau`` (needed because split windows are re-decoded by the
    tokenizer). Gold = union of evidence over all non-unanswerable annotations.
    Excluded (and counted in ``setup["stats"]``): questions that are only
    unanswerable, have no evidence, or whose evidence could not be matched.
    "FLOAT SELECTED" evidence (figures/tables) is dropped: it is not in the
    paragraph corpus.

SciFact
    Standard BEIR qrels. ``train.tsv`` queries form the tuning fold and
    ``test.tsv`` queries the reporting fold.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rag.retrieval.corpus import Corpus

_SPLIT_FILES = {"train": "train", "validation": "dev", "dev": "dev", "test": "test"}
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass
class EvalQuery:
    qid: str
    text: str
    doc_id: str | None  # None = search the whole corpus
    relevant: dict[str, float]  # unit_id -> gain
    fold: str  # "tune" or "test"
    meta: dict = field(default_factory=dict)


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _tokens(text: str) -> Counter:
    return Counter(_TOKEN.findall(text.lower()))


def _f1(a: Counter, b: Counter) -> float:
    overlap = sum((a & b).values())
    if not overlap:
        return 0.0
    p, r = overlap / sum(a.values()), overlap / sum(b.values())
    return 2 * p * r / (p + r)


class _EvidenceMatcher:
    def __init__(self, units: dict[str, str], tau: float):
        self.tau = tau
        self.exact = {_norm(text): unit for unit, text in units.items()}
        self.tokens = {unit: _tokens(text) for unit, text in units.items()}

    def match(self, evidence: str) -> str | None:
        hit = self.exact.get(_norm(evidence))
        if hit is not None:
            return hit
        target = _tokens(evidence)
        best_unit, best = None, 0.0
        for unit, toks in self.tokens.items():
            score = _f1(target, toks)
            if score > best:
                best_unit, best = unit, score
        return best_unit if best >= self.tau else None


def _paragraph_units(corpus: Corpus, doc_ids: list[str]) -> dict[str, dict[str, str]]:
    wanted = set(doc_ids)
    bodies: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for chunk, unit in zip(corpus.chunks, corpus.unit_ids):
        if chunk.doc_id in wanted:
            bodies[chunk.doc_id][unit].append(chunk.body)
    return {d: {u: " ".join(parts) for u, parts in units.items()} for d, units in bodies.items()}


def build_qasper_queries(
    data_dir: Path,
    corpus: Corpus,
    split: str = "validation",
    scope: str = "paper",
    num_papers: int | None = None,
    seed: int = 13,
    tune_frac: float = 0.3,
    tau: float = 0.9,
) -> tuple[list[EvalQuery], dict]:
    path = Path(data_dir) / "qasper" / f"qasper-{_SPLIT_FILES[split]}-v0.3.json"
    raw = json.loads(path.read_text())

    cached = set(corpus.doc_rows)
    available = sorted(p for p in raw if p in cached)
    if not available:
        raise SystemExit(
            f"None of the {len(raw)} papers in {path.name} have chunks in the cache. "
            "Check that Chunk.doc_id equals the QASPER paper id and that --split matches the cache."
        )

    # Seeded paper-level sample, then a paper-level tune/test partition (no leakage across folds).
    rng = random.Random(seed)
    chosen = rng.sample(available, min(num_papers, len(available)) if num_papers else len(available))
    n_tune = int(round(len(chosen) * tune_frac))
    fold_of = {p: ("tune" if i < n_tune else "test") for i, p in enumerate(chosen)}

    units = _paragraph_units(corpus, chosen)
    stats: Counter = Counter()
    queries: list[EvalQuery] = []
    for pid in chosen:
        matcher = _EvidenceMatcher(units.get(pid, {}), tau)
        for qa in raw[pid]["qas"]:
            stats["questions_seen"] += 1
            evidence: set[str] = set()
            answerable = False
            for ans in qa["answers"]:
                a = ans["answer"]
                if a.get("unanswerable"):
                    continue
                answerable = True
                evidence.update(e for e in a.get("evidence", []) if e.strip())
            if not answerable:
                stats["excluded_unanswerable"] += 1
                continue
            floats = {e for e in evidence if e.startswith("FLOAT SELECTED")}
            stats["float_evidence_dropped"] += len(floats)
            text_ev = evidence - floats
            if not text_ev:
                stats["excluded_no_text_evidence"] += 1
                continue
            gold = {m for m in (matcher.match(e) for e in text_ev) if m is not None}
            stats["evidence_strings"] += len(text_ev)
            stats["evidence_matched"] += sum(1 for e in text_ev if matcher.match(e) is not None)
            if not gold:
                stats["excluded_unmatched_evidence"] += 1
                continue
            queries.append(
                EvalQuery(
                    qid=qa["question_id"],
                    text=qa["question"],
                    doc_id=pid if scope == "paper" else None,
                    relevant={u: 1.0 for u in gold},
                    fold=fold_of[pid],
                    meta={"paper_id": pid, "n_gold_units": len(gold)},
                )
            )
            stats["questions_kept"] += 1

    setup = {
        "dataset": "qasper",
        "split": split,
        "scope": scope,
        "seed": seed,
        "num_papers_available": len(available),
        "papers_in_json_without_chunks": len(raw) - len(available),
        "num_papers_sampled": len(chosen),
        "tune_frac": tune_frac,
        "evidence_match_tau": tau,
        "tune_papers": [p for p in chosen if fold_of[p] == "tune"],
        "test_papers": [p for p in chosen if fold_of[p] == "test"],
        "stats": dict(stats),
        "evidence_match_rate": stats["evidence_matched"] / max(stats["evidence_strings"], 1),
    }
    return queries, setup


def build_scifact_queries(data_dir: Path, corpus: Corpus) -> tuple[list[EvalQuery], dict]:
    root = Path(data_dir) / "scifact"
    texts: dict[str, str] = {}
    for line in (root / "queries.jsonl").read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            texts[str(d["_id"])] = d["text"]

    cached_docs = set(corpus.doc_rows)
    queries: list[EvalQuery] = []
    stats: Counter = Counter()
    for fold, fname in (("tune", "train"), ("test", "test")):
        rel: dict[str, dict[str, float]] = defaultdict(dict)
        with (root / "qrels" / f"{fname}.tsv").open() as handle:
            for line in handle:
                qid, did, score = line.rstrip("\n").split("\t")[:3]
                if qid == "query-id":
                    continue
                if float(score) <= 0:
                    continue
                if did not in cached_docs:
                    stats["qrel_docs_missing_from_cache"] += 1
                    continue
                rel[qid][did] = float(score)
        for qid, relevant in rel.items():
            if qid in texts:
                queries.append(EvalQuery(qid, texts[qid], None, relevant, fold))
                stats[f"queries_{fold}"] += 1

    if stats["qrel_docs_missing_from_cache"]:
        print(
            f"WARNING: {stats['qrel_docs_missing_from_cache']} qrel docs are not in the cache "
            "(cache built with --limit, or doc_id differs from the corpus _id)."
        )
    setup = {"dataset": "scifact", "scope": "global", "stats": dict(stats)}
    return queries, setup