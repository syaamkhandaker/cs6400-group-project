#!/usr/bin/env python3
"""Evaluate BM25, dense and hybrid retrieval on QASPER or BEIR-SciFact.

All methods search the same cached chunks. Hybrid alpha is selected on the
tune fold (paper-disjoint for QASPER, SciFact train qrels for SciFact) and
reported on the test fold only.

Examples::

    python scripts/run_eval.py --dataset qasper --split validation --scope paper
    python scripts/run_eval.py --dataset qasper --split validation --scope global --num-papers 100
    python scripts/run_eval.py --dataset scifact
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from tqdm import tqdm

from rag.config import REPO_ROOT, load_raw_config
from rag.embeddings import HuggingFaceEmbedder
from rag.eval.metrics import aggregate, paired_bootstrap, query_metrics
from rag.eval.tasks import build_qasper_queries, build_scifact_queries
from rag.retrieval.corpus import Corpus
from rag.retrieval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    prepare_fusion,
    rank_units,
    rrf_fusion,
    weighted_fusion,
)

TABLE_COLS = [
    ("R@1", "recall@1"), ("R@3", "recall@3"), ("R@5", "recall@5"), ("R@10", "recall@10"),
    ("P@5", "precision@5"), ("MRR@10", "mrr@10"), ("nDCG@10", "ndcg@10"),
]


def alpha_name(a: float) -> str:
    return f"hybrid[a={a:.2f}]"


def main() -> None:
    args = _parse_args()
    ks = tuple(sorted(args.ks))
    kmax = max(ks)
    config = load_raw_config(args.config)
    corpus = (
        Corpus.from_dir(args.dataset, args.cache_dir)
        if args.cache_dir
        else Corpus.from_config(config, args.dataset, args.split)
    )
    print(f"Corpus: {len(corpus.chunks)} chunks, {len(corpus.doc_rows)} docs, "
          f"{len(set(corpus.unit_ids))} units")

    if args.dataset == "qasper":
        queries, setup = build_qasper_queries(
            args.data_dir, corpus, split=args.split or "validation", scope=args.scope,
            num_papers=args.num_papers, seed=args.seed, tune_frac=args.tune_frac, tau=args.tau,
        )
    else:
        queries, setup = build_scifact_queries(args.data_dir, corpus)
        args.scope = "global"
    folds = {f: [i for i, q in enumerate(queries) if q.fold == f] for f in ("tune", "test")}
    print(f"Queries: {len(folds['tune'])} tune, {len(folds['test'])} test. Stats: {setup['stats']}")
    if not folds["test"]:
        raise SystemExit("No test queries after filtering.")

    t0 = time.perf_counter()
    embedder = HuggingFaceEmbedder.from_config(config)
    dense = DenseRetriever(corpus, embedder)
    dense.precompute([q.text for q in queries])
    embed_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    bm25 = BM25Retriever(corpus, field=args.bm25_field, k1=args.bm25_k1, b=args.bm25_b, stem=not args.no_stem)
    bm25_build_s = time.perf_counter() - t0

    alphas = [round(float(a), 4) for a in args.alpha_grid]
    per_query: dict[str, list] = defaultdict(list)
    tops: dict[str, list] = defaultdict(list)
    lat = {"dense": 0.0, "bm25": 0.0}

    for q in tqdm(queries, desc="eval"):
        scope = corpus.scope(q.doc_id)
        t = time.perf_counter(); d = dense.score(q.text, scope); lat["dense"] += time.perf_counter() - t
        t = time.perf_counter(); s = bm25.score(q.text, scope); lat["bm25"] += time.perf_counter() - t
        prep = prepare_fusion(d, s, args.norm, args.pool)
        scored = {"bm25": s, "dense": d, "hybrid_rrf": rrf_fusion(d, s, args.pool, args.rrf_k)}
        for a in alphas:
            scored[alpha_name(a)] = weighted_fusion(prep, a)
        for name, sc in scored.items():
            ranked = rank_units(scope, sc, corpus.unit_ids, kmax)
            per_query[name].append(query_metrics([u for u, _, _ in ranked], q.relevant, ks))
            tops[name].append(ranked[: args.dump_k])

    def fold_mean(name: str, fold: str) -> dict:
        return aggregate([per_query[name][i] for i in folds[fold]])

    # Select alpha on the tune fold only.
    if folds["tune"]:
        tune_curve = {a: fold_mean(alpha_name(a), "tune")[args.select_metric] for a in alphas}
        best_alpha = max(alphas, key=lambda a: (tune_curve[a], -abs(a - 0.5)))
    else:
        print("WARNING: no tune queries; using alpha=0.5.")
        tune_curve, best_alpha = {}, 0.5
    best = alpha_name(best_alpha) if best_alpha in alphas else alpha_name(alphas[0])
    print(f"Selected alpha={best_alpha} on tune fold by {args.select_metric}")

    reported = {
        "BM25": "bm25",
        "Dense (MiniLM)": "dense",
        f"Hybrid weighted (alpha={best_alpha:.2f})": best,
        "Hybrid RRF": "hybrid_rrf",
    }
    test_results = {label: fold_mean(name, "test") for label, name in reported.items()}
    tune_results = {label: fold_mean(name, "tune") for label, name in reported.items()} if folds["tune"] else {}

    boot = {}
    hyb_label = f"Hybrid weighted (alpha={best_alpha:.2f})"
    for base in ("BM25", "Dense (MiniLM)", "Hybrid RRF"):
        for m in ("ndcg@10", "recall@5", f"mrr@{kmax}"):
            a = [per_query[best][i][m] for i in folds["test"]]
            b = [per_query[reported[base]][i][m] for i in folds["test"]]
            boot[f"{hyb_label} - {base} [{m}]"] = paired_bootstrap(a, b, seed=args.seed)

    n_test = len(folds["test"])
    latency = {
        "dense_score_ms_per_query": 1000 * lat["dense"] / len(queries),
        "bm25_score_ms_per_query": 1000 * lat["bm25"] / len(queries),
        "query_embedding_total_s": embed_s,
        "bm25_index_build_s": bm25_build_s,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{args.scope}" + (f"_{args.tag}" if args.tag else "")

    results = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "setup": setup,
        "corpus": {"chunks": len(corpus.chunks), "docs": len(corpus.doc_rows)},
        "n_queries": {"tune": len(folds["tune"]), "test": n_test},
        "selected_alpha": best_alpha,
        "alpha_curve_tune": {str(a): v for a, v in tune_curve.items()},
        "alpha_curve_test_for_reference_only": {
            str(a): fold_mean(alpha_name(a), "test") for a in alphas
        },
        "test": test_results,
        "tune": tune_results,
        "paired_bootstrap_test": boot,
        "latency": latency,
    }
    (out_dir / f"results_{tag}.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / f"table_{tag}.md").write_text(_markdown(results, args, n_test))

    with (out_dir / f"per_query_{tag}.jsonl").open("w") as handle:
        for qi, q in enumerate(queries):
            methods = {}
            for label, name in reported.items():
                methods[label] = {
                    "metrics": per_query[name][qi],
                    "retrieved": [
                        {
                            "rank": r + 1,
                            "unit_id": unit,
                            "is_gold": unit in q.relevant,
                            "score": score,
                            "doc_id": corpus.chunks[row].doc_id,
                            "section": corpus.chunks[row].section,
                            "chunk_id": corpus.chunks[row].chunk_id,
                            "text": corpus.chunks[row].body,
                        }
                        for r, (unit, score, row) in enumerate(tops[name][qi])
                    ],
                }
            handle.write(json.dumps({
                "qid": q.qid, "fold": q.fold, "query": q.text, "doc_id": q.doc_id,
                "gold_units": sorted(q.relevant), "methods": methods,
            }, ensure_ascii=False) + "\n")

    print((out_dir / f"table_{tag}.md").read_text())
    print(f"Wrote results to {out_dir}/ (tag {tag})")


def _markdown(results: dict, args, n_test: int) -> str:
    unit = "paragraph" if args.dataset == "qasper" else "abstract"
    lines = [
        f"**{args.dataset}** | scope: {args.scope} | test queries: {n_test} | "
        f"ranking unit: {unit} (max over chunks) | alpha selected on tune fold: {results['selected_alpha']}",
        "",
        "| Method | " + " | ".join(c[0] for c in TABLE_COLS) + " |",
        "|---|" + "---|" * len(TABLE_COLS),
    ]
    for label, m in results["test"].items():
        lines.append(f"| {label} | " + " | ".join(f"{m[key]:.3f}" for _, key in TABLE_COLS) + " |")
    lines += ["", "Paired bootstrap, hybrid weighted minus baseline (test fold, mean diff [95% CI]):", ""]
    for k, v in results["paired_bootstrap_test"].items():
        lines.append(f"- {k}: {v['mean_diff']:+.3f} [{v['ci_low']:+.3f}, {v['ci_high']:+.3f}]")
    return "\n".join(lines) + "\n"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=("qasper", "scifact"))
    p.add_argument("--split", default=None, help="QASPER split (validation/test/train).")
    p.add_argument("--scope", choices=("paper", "global"), default="paper", help="QASPER only.")
    p.add_argument("--num-papers", type=int, default=None, help="Seeded sample of papers (default: all).")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--tune-frac", type=float, default=0.3, help="Fraction of sampled papers used to pick alpha.")
    p.add_argument("--tau", type=float, default=0.9, help="Token-F1 threshold for fuzzy evidence matching.")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    p.add_argument("--cache-dir", type=Path, default=None, help="Load this cache directory directly.")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--tag", default="")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5, 10])
    p.add_argument("--dump-k", type=int, default=5)
    p.add_argument("--bm25-field", choices=("text", "body"), default="text")
    p.add_argument("--bm25-k1", type=float, default=1.2)
    p.add_argument("--bm25-b", type=float, default=0.75)
    p.add_argument("--no-stem", action="store_true")
    p.add_argument("--norm", choices=("minmax", "zscore"), default="minmax")
    p.add_argument("--pool", type=int, default=100)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--alpha-grid", type=float, nargs="+", default=[i / 10 for i in range(11)])
    p.add_argument("--select-metric", default="ndcg@10")
    return p.parse_args()


if __name__ == "__main__":
    main()