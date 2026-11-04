"""Ranking metrics over ranked unit ids and a graded relevance dict."""

from __future__ import annotations

import math

import numpy as np

DEFAULT_KS = (1, 3, 5, 10)


def query_metrics(ranked: list[str], relevant: dict[str, float], ks=DEFAULT_KS) -> dict[str, float]:
    """Recall/Precision/Hit/nDCG at each k, plus MRR cut at max(ks).

    Precision@k always divides by k, so it is capped at |relevant|/k when a
    query has few relevant units (typical for QASPER).
    """
    rel = {u: g for u, g in relevant.items() if g > 0}
    n_rel = len(rel)
    gains = [rel.get(u, 0.0) for u in ranked]
    ideal = sorted(rel.values(), reverse=True)
    kmax = max(ks)
    out: dict[str, float] = {}
    for k in ks:
        hits = sum(1 for g in gains[:k] if g > 0)
        out[f"recall@{k}"] = hits / n_rel if n_rel else 0.0
        out[f"precision@{k}"] = hits / k
        out[f"hit@{k}"] = float(hits > 0)
        dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains[:k]))
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
        out[f"ndcg@{k}"] = dcg / idcg if idcg > 0 else 0.0
    out[f"mrr@{kmax}"] = next((1.0 / (i + 1) for i, g in enumerate(gains[:kmax]) if g > 0), 0.0)
    return out


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    return {m: float(np.mean([q[m] for q in per_query])) for m in per_query[0]}


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 5000, seed: int = 0) -> dict[str, float]:
    """Mean difference a-b over queries with a 95% percentile bootstrap CI."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi)}