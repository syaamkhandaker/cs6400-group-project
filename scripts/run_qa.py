#!/usr/bin/env python3
"""End-to-end QA on a fixed QASPER subset: retrieve, answer with citations, score.
Every method retrieves the top k paragraphs from the question's own paper
and hands their best chunks to the same Ollama model with the same prompt and
decoding settings, so the retriever is the only thing that changes between rows.
The subset is a seeded sample of ``--num-questions`` questions from the
retrieval test set. The set comes from ``build_qasper_queries`` with the
same ``--seed``/``--tune-frac``/``--num-papers`` as ``run_eval.py``, so alpha
and n, chosen on the tune fold, were not tuned on these questions. Answerable
questions whose evidence is only figures or tables are left out (text-only QA,
as in the proposal). Unanswerable questions stay in. Alpha and the rerank candidate 
count default to the values ``run_eval.py --rerank`` selected, read from ``--retrieval-results``.

Examples::
    python scripts/run_qa.py --split validation --num-questions 200
    python scripts/run_qa.py --methods dense rerank --num-questions 20 --tag smoke
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tqdm import tqdm

from rag.config import REPO_ROOT, load_raw_config
from rag.embeddings import HuggingFaceEmbedder
from rag.eval.metrics import paired_bootstrap
from rag.eval.qa_metrics import aggregate_qa, qasper_references, score_prediction
from rag.eval.tasks import _SPLIT_FILES, _EvidenceMatcher, _paragraph_units, build_qasper_queries
from rag.generation import answer_with_citations
from rag.llm import OllamaLLM
from rag.retrieval.corpus import Corpus
from rag.retrieval.rerank import CrossEncoderReranker, RerankedHybridRetriever
from rag.retrieval.retrievers import BM25Retriever, DenseRetriever, Hit, HybridRetriever, rank_units

METHOD_LABELS = {
    "bm25": "BM25",
    "dense": "Dense (MiniLM)",
    "hybrid": "Hybrid weighted",
    "rerank": "Hybrid + CE rerank",
}
TABLE_COLS = [
    ("Answer-F1", "answer_f1"), ("Extractive", "f1_extractive"), ("Abstractive", "f1_abstractive"),
    ("Yes/No", "f1_boolean"), ("Unans.", "f1_none"), ("Yes/No EM", "yes_no_em"),
    ("Unans. acc", "unanswerable_acc"), ("Gold in ctx", "context_has_gold_rate"), ("Cites gold", "cites_gold_rate"),
]


def main() -> None:
    args = _parse_args()
    config = load_raw_config(args.config)
    alpha, n_candidates = _retrieval_settings(args, config)
    k = args.k or int(config["generation"].get("context_chunks", 5))

    # Same test fold as run_eval.py, then a seeded question sample from it.
    corpus = Corpus.from_config(config, "qasper", args.split)
    raw = json.loads((args.data_dir / "qasper" / f"qasper-{_SPLIT_FILES[args.split]}-v0.3.json").read_text())
    _, setup = build_qasper_queries(
        args.data_dir, corpus, split=args.split, scope="paper", num_papers=args.num_papers,
        seed=args.seed, tune_frac=args.tune_frac, tau=args.tau,
    )
    subset, stats = _sample_questions(raw, setup["test_papers"], args)
    print(f"QA subset: {len(subset)} questions from the test fold. Stats: {dict(stats)}")
    print(f"Methods: {args.methods} | k={k} | alpha={alpha} | rerank n={n_candidates}")

    # One retriever per method; all share the corpus, and only the retriever differs between rows.
    embedder = HuggingFaceEmbedder.from_config(config)
    dense = DenseRetriever(corpus, embedder)
    dense.precompute([qa["question"] for _, qa, _, _ in subset])
    bm25 = BM25Retriever(corpus, k1=args.bm25_k1, b=args.bm25_b)
    hybrid = HybridRetriever(dense, bm25, alpha=alpha, norm=args.norm, pool=args.pool, rrf_k=args.rrf_k)
    retrievers = {"bm25": bm25, "dense": dense, "hybrid": hybrid}
    if "rerank" in args.methods:
        retrievers["rerank"] = RerankedHybridRetriever(hybrid, CrossEncoderReranker.from_config(config), n_candidates)
    llm = OllamaLLM.from_yaml(args.config)
    print(f"LLM: {llm.config.model} (temperature {llm.config.temperature})")

    units = _paragraph_units(corpus, sorted({pid for pid, _, _, _ in subset}))
    row_of = {c.chunk_id: i for i, c in enumerate(corpus.chunks)}  # chunk id -> corpus row
    scored: dict[str, list[dict]] = defaultdict(list)
    seconds: dict[str, dict[str, float]] = defaultdict(lambda: {"retrieve": 0.0, "generate": 0.0})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "qasper" + (f"_{args.tag}" if args.tag else "")
    # One JSON line per question, flushed as we go so a crash keeps the finished questions.
    with (out_dir / f"qa_per_question_{tag}.jsonl").open("w") as handle:
        for pid, qa, refs, text_ev in tqdm(subset, desc="qa"):
            # Gold paragraphs, used only for the "Gold in ctx" and "Cites gold" checks.
            matcher = _EvidenceMatcher(units.get(pid, {}), args.tau)
            gold = {u for u in (matcher.match(e) for e in text_ev) if u is not None}
            scope = corpus.scope(pid)  # search only this question's paper
            record = {"qid": qa["question_id"], "paper_id": pid, "question": qa["question"],
                      "references": refs, "gold_units": sorted(gold), "methods": {}}
            for method in args.methods:
                # Top-k paragraphs (best chunk of each) -> LLM answer with citations -> score.
                t = time.perf_counter()
                ranked = rank_units(scope, retrievers[method].score(qa["question"], scope), corpus.unit_ids, k)
                hits = [Hit(corpus.chunks[row], score) for _, score, row in ranked]
                seconds[method]["retrieve"] += time.perf_counter() - t
                t = time.perf_counter()
                generated = answer_with_citations(qa["question"], hits, llm)
                seconds[method]["generate"] += time.perf_counter() - t

                result = score_prediction(generated.answer, refs)
                cited_units = {corpus.unit_ids[row_of[cid]] for cid in generated.cited_chunk_ids}
                result["context_has_gold"] = bool(gold & {u for u, _, _ in ranked}) if gold else None
                result["cites_gold"] = bool(gold & cited_units) if gold else None
                scored[method].append(result)
                record["methods"][METHOD_LABELS[method]] = {
                    **result,
                    "answer": generated.answer,
                    "raw": generated.raw,
                    "citations": [
                        {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "section": c.section, "text": c.body}
                        for c in (corpus.chunks[row_of[cid]] for cid in generated.cited_chunk_ids)
                    ],
                    "retrieved_chunk_ids": [h.chunk.chunk_id for h in hits],
                }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    # Aggregate per method and write the JSON summary and the markdown table.
    n = len(subset)
    results = {
        "args": {key: (str(v) if isinstance(v, Path) else v) for key, v in vars(args).items()},
        "alpha": alpha,
        "rerank_n": n_candidates,
        "k": k,
        "llm": {"model": llm.config.model, "temperature": llm.config.temperature},
        "subset_stats": dict(stats),
        "metrics": {},
        "latency_ms_per_question": {
            METHOD_LABELS[m]: {phase: 1000 * s / n for phase, s in seconds[m].items()} for m in args.methods
        },
    }
    for method in args.methods:
        metrics = aggregate_qa(scored[method])
        hits = [s["context_has_gold"] for s in scored[method] if s["context_has_gold"] is not None]
        metrics["context_has_gold_rate"] = sum(hits) / len(hits) if hits else None
        results["metrics"][METHOD_LABELS[method]] = metrics
    results["paired_bootstrap_f1"] = f1_bootstrap(
        {METHOD_LABELS[m]: [s["f1"] for s in scored[m]] for m in args.methods}, seed=args.seed
    )
    (out_dir / f"qa_results_{tag}.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / f"qa_table_{tag}.md").write_text(_markdown(results, args, n, k))
    print((out_dir / f"qa_table_{tag}.md").read_text())
    print(f"Wrote QA results to {out_dir}/ (tag {tag})")


def f1_bootstrap(f1_by_label: dict[str, list[float]], seed: int) -> dict:
    """Paired bootstrap of Answer-F1 for every later method minus every earlier one."""
    labels = list(f1_by_label)
    return {
        f"{a} - {b}": paired_bootstrap(f1_by_label[a], f1_by_label[b], seed=seed)
        for i, b in enumerate(labels) for a in labels[i + 1:]
    }


def _retrieval_settings(args, config: dict) -> tuple[float, int]:
    """Alpha and rerank n: command line first, then run_eval.py's results JSON, then the config."""
    alpha, n = args.alpha, args.rerank_n
    path = Path(args.retrieval_results)
    if (alpha is None or n is None) and path.is_file():
        saved = json.loads(path.read_text())
        if alpha is None:
            alpha = float(saved["selected_alpha"])
        if n is None and saved.get("rerank"):
            n = int(saved["rerank"]["selected_n"])
        print(f"Read selected settings from {path}")
    retrieval = config.get("retrieval", {})
    return (
        float(retrieval.get("alpha", 0.5)) if alpha is None else alpha,
        int(retrieval.get("n_candidates", 30)) if n is None else n,
    )


def _sample_questions(raw: dict, test_papers: list[str], args) -> tuple[list, Counter]:
    """Seeded sample of test-fold questions, skipping answerable ones with only figure/table evidence."""
    stats: Counter = Counter()
    pool = []
    for pid in test_papers:
        for qa in raw[pid]["qas"]:
            stats["test_fold_questions"] += 1
            refs = qasper_references(qa)
            answerable = [e["answer"] for e in qa["answers"] if not e["answer"].get("unanswerable")]
            text_ev = {
                ev.strip() for a in answerable for ev in a.get("evidence", [])
                if ev.strip() and not ev.startswith("FLOAT SELECTED")
            }
            if not refs:
                stats["excluded_no_reference"] += 1
            elif answerable and not text_ev:
                stats["excluded_no_text_evidence"] += 1
            else:
                pool.append((pid, qa, refs, sorted(text_ev)))
    stats["eligible"] = len(pool)
    subset = random.Random(args.seed).sample(pool, min(args.num_questions, len(pool)))
    stats.update(f"first_type_{refs[0]['type']}" for _, _, refs, _ in subset)
    return subset, stats


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _markdown(results: dict, args, n: int, k: int) -> str:
    """Results table plus notes on the counts and the bootstrap intervals."""
    lines = [
        f"**QASPER QA** | split: {args.split} (retrieval test fold) | questions: {n} | context: top-{k} paragraphs "
        f"from the question's paper | LLM: {results['llm']['model']} (T={results['llm']['temperature']}) | "
        f"alpha={results['alpha']}, rerank n={results['rerank_n']}",
        "",
        "| Method | " + " | ".join(c[0] for c in TABLE_COLS) + " | Gen. ms/q |",
        "|---|" + "---|" * (len(TABLE_COLS) + 1),
    ]
    for label, m in results["metrics"].items():
        gen = results["latency_ms_per_question"][label]["generate"]
        lines.append(f"| {label} | " + " | ".join(_fmt(m[key]) for _, key in TABLE_COLS) + f" | {gen:.0f} |")
    first = next(iter(results["metrics"].values()))
    lines += [
        "",
        f"Answer-F1 is counted under the type of the best-matching reference (official evaluator): "
        f"extractive {first['n_extractive']}, abstractive {first['n_abstractive']}, yes/no {first['n_boolean']}, "
        f"unanswerable {first['n_none']} for the first row. Yes/No EM uses the {first['n_yes_no']} questions whose "
        "first annotation is yes/no. 'Gold in ctx' and 'Cites gold' are the shares of questions with matched gold "
        "paragraphs whose context, or whose citations, include one.",
    ]
    if results.get("paired_bootstrap_f1"):
        lines += ["", "Paired bootstrap of Answer-F1 (mean diff [95% CI]):", ""]
        for name, v in results["paired_bootstrap_f1"].items():
            lines.append(f"- {name}: {v['mean_diff']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    p.add_argument("--num-questions", type=int, default=200)
    p.add_argument("--num-papers", type=int, default=None, help="Must match run_eval.py to reuse its test fold.")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--tune-frac", type=float, default=0.3)
    p.add_argument("--tau", type=float, default=0.9, help="Token-F1 threshold for matching gold paragraphs.")
    p.add_argument("--methods", nargs="+", choices=tuple(METHOD_LABELS), default=list(METHOD_LABELS))
    p.add_argument("--k", type=int, default=None, help="Paragraphs given to the LLM (default generation.context_chunks).")
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--rerank-n", type=int, default=None)
    p.add_argument("--retrieval-results", default=str(REPO_ROOT / "results" / "results_qasper_paper.json"))
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--tag", default="")
    p.add_argument("--bm25-k1", type=float, default=1.2)
    p.add_argument("--bm25-b", type=float, default=0.75)
    p.add_argument("--norm", choices=("minmax", "zscore"), default="minmax")
    p.add_argument("--pool", type=int, default=100)
    p.add_argument("--rrf-k", type=int, default=60)
    return p.parse_args()


if __name__ == "__main__":
    main()
