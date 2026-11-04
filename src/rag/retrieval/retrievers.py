"""BM25, dense and hybrid retrievers over one shared ``Corpus``.

Every retriever exposes ``score(query, scope) -> np.ndarray`` (one score per
row in ``scope``) and ``search(query, k, doc_id)`` for chunk-level top-k.
Scores are exact (dense uses a full matrix product, which returns the same
values as ``faiss.IndexFlatIP``), so any subset of chunks can be scored and
fused without approximation.

Hybrid fusion (weighted):

    pool_d, pool_s = top-``pool`` chunks of each retriever within the scope
    nd = normalise(dense, stats of pool_d)      ns = normalise(bm25, stats of pool_s)
    fused = alpha * nd + (1 - alpha) * ns       over pool_d | pool_s only

Min-max values are clipped to [0, 1], so a candidate outside the other
retriever's pool gets that retriever's pool minimum (0), not a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rag.chunking import Chunk
from rag.retrieval.bm25 import BM25Index
from rag.retrieval.corpus import Corpus


@dataclass(frozen=True)
class Hit:
    """Same fields as ``pipeline.Hit`` so ``RetrievalPipeline.answer`` style code can consume it."""

    chunk: Chunk
    score: float


def _top_k_rows(scope: np.ndarray, scores: np.ndarray, k: int) -> list[tuple[int, float]]:
    idx = np.flatnonzero(np.isfinite(scores))
    order = idx[np.argsort(-scores[idx], kind="stable")][:k]
    return [(int(scope[i]), float(scores[i])) for i in order]


def rank_units(scope: np.ndarray, scores: np.ndarray, unit_ids: np.ndarray, k: int) -> list[tuple[str, float, int]]:
    """Collapse chunk scores to units by max score.

    Returns up to ``k`` ``(unit_id, score, best_chunk_row)`` tuples, best first.
    Ties break by corpus order, so results are deterministic.
    """
    idx = np.flatnonzero(np.isfinite(scores))
    order = idx[np.argsort(-scores[idx], kind="stable")]
    out: list[tuple[str, float, int]] = []
    seen: set[str] = set()
    for i in order:
        row = int(scope[i])
        unit = unit_ids[row]
        if unit in seen:
            continue
        seen.add(unit)
        out.append((unit, float(scores[i]), row))
        if len(out) >= k:
            break
    return out


class _Base:
    name = "base"
    corpus: Corpus

    def score(self, query: str, scope: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def search(self, query: str, k: int = 10, doc_id: str | None = None) -> list[Hit]:
        scope = self.corpus.scope(doc_id)
        if len(scope) == 0:
            return []
        scores = self.score(query, scope)
        return [Hit(self.corpus.chunks[row], s) for row, s in _top_k_rows(scope, scores, k)]


class BM25Retriever(_Base):
    name = "bm25"

    def __init__(self, corpus: Corpus, field: str = "text", k1: float = 1.2, b: float = 0.75, stem: bool = True):
        self.corpus = corpus
        self.field = field
        self.index = BM25Index([getattr(c, field) for c in corpus.chunks], k1=k1, b=b, stem=stem)

    def score(self, query: str, scope: np.ndarray) -> np.ndarray:
        return self.index.scores(query, scope)


class DenseRetriever(_Base):
    """Cosine similarity with the cached MiniLM vectors. No reranking, rewriting or expansion."""

    name = "dense"

    def __init__(self, corpus: Corpus, embedder):
        self.corpus = corpus
        self.embedder = embedder
        self._queries: dict[str, np.ndarray] = {}

    def precompute(self, queries: list[str]) -> None:
        """Embed many queries in batches (same model and normalisation as embed_query)."""
        new = [q for q in dict.fromkeys(queries) if q not in self._queries]
        if new:
            for q, vec in zip(new, self.embedder.embed_documents(new)):
                self._queries[q] = vec

    def score(self, query: str, scope: np.ndarray) -> np.ndarray:
        if query not in self._queries:
            self._queries[query] = self.embedder.embed_query(query)
        q = self._queries[query]
        vectors = self.corpus.vectors
        if scope is self.corpus.all_rows:
            return vectors @ q
        return vectors[scope] @ q


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------


def _top_pool(x: np.ndarray, pool: int) -> np.ndarray:
    if len(x) <= pool:
        return np.arange(len(x))
    return np.argpartition(-x, pool - 1)[:pool]


def _normalize(x: np.ndarray, pool_idx: np.ndarray, method: str) -> np.ndarray:
    p = x[pool_idx]
    if method == "minmax":
        lo, hi = float(p.min()), float(p.max())
        if hi <= lo:
            return np.zeros_like(x)
        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    if method == "zscore":
        sd = float(p.std())
        if sd <= 0:
            return np.zeros_like(x)
        return (x - float(p.mean())) / sd
    raise ValueError(f"Unknown normalisation {method!r} (use 'minmax' or 'zscore')")


@dataclass
class FusionInputs:
    nd: np.ndarray
    ns: np.ndarray
    mask: np.ndarray


def prepare_fusion(dense: np.ndarray, sparse: np.ndarray, norm: str = "minmax", pool: int = 100) -> FusionInputs:
    """Normalise once per query so an alpha sweep only does a cheap linear blend."""
    pd, ps = _top_pool(dense, pool), _top_pool(sparse, pool)
    mask = np.zeros(len(dense), dtype=bool)
    mask[pd] = True
    mask[ps] = True
    return FusionInputs(_normalize(dense, pd, norm), _normalize(sparse, ps, norm), mask)


def weighted_fusion(prep: FusionInputs, alpha: float) -> np.ndarray:
    fused = alpha * prep.nd + (1.0 - alpha) * prep.ns
    return np.where(prep.mask, fused, -np.inf)


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(-x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def rrf_fusion(dense: np.ndarray, sparse: np.ndarray, pool: int = 100, k: int = 60, alpha: float = 0.5) -> np.ndarray:
    """Reciprocal rank fusion over the same pools (ablation for weighted fusion)."""
    rd, rs = _ranks(dense), _ranks(sparse)
    in_d, in_s = rd <= pool, rs <= pool
    fused = np.where(in_d, alpha / (k + rd), 0.0) + np.where(in_s, (1.0 - alpha) / (k + rs), 0.0)
    return np.where(in_d | in_s, fused, -np.inf)


class HybridRetriever(_Base):
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        alpha: float = 0.5,
        norm: str = "minmax",
        pool: int = 100,
        method: str = "weighted",
        rrf_k: int = 60,
    ):
        if dense.corpus is not bm25.corpus:
            raise ValueError("Dense and BM25 retrievers must share one Corpus.")
        self.corpus = dense.corpus
        self.dense, self.bm25 = dense, bm25
        self.alpha, self.norm, self.pool, self.method, self.rrf_k = alpha, norm, pool, method, rrf_k
        self.name = f"hybrid_{method}"

    def score(self, query: str, scope: np.ndarray) -> np.ndarray:
        d = self.dense.score(query, scope)
        s = self.bm25.score(query, scope)
        if self.method == "rrf":
            return rrf_fusion(d, s, self.pool, self.rrf_k, self.alpha)
        return weighted_fusion(prepare_fusion(d, s, self.norm, self.pool), self.alpha)