"""Cross-encoder reranking of the fused hybrid candidates.

Candidates follow the proposal: the top ``n`` chunks of weighted fusion and the
top ``n`` chunks of RRF, merged without duplicates. Only those chunks are scored
by the cross-encoder (``cross-encoder/ms-marco-MiniLM-L-6-v2`` by default), which
reads the query and the chunk together. Every other chunk in the scope gets
``-inf``, so ``rank_units`` and ``search`` never return it.
"""

from __future__ import annotations

import numpy as np

from rag.retrieval.corpus import Corpus
from rag.retrieval.retrievers import HybridRetriever, _Base, prepare_fusion, rrf_fusion, weighted_fusion

# Used when neither the caller nor the config names a model.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def top_n(scores: np.ndarray, n: int) -> np.ndarray:
    """Scope positions of the ``n`` best finite scores, best first."""
    idx = np.flatnonzero(np.isfinite(scores))
    return idx[np.argsort(-scores[idx], kind="stable")][:n]


def merge_candidates(*rankings: np.ndarray, n: int) -> np.ndarray:
    """Union of the top-``n`` positions of each ranking, in first-seen order."""
    seen = dict.fromkeys(int(i) for scores in rankings for i in top_n(scores, n))
    return np.fromiter(seen, dtype=np.int64, count=len(seen))


class CrossEncoderReranker:
    """Score (query, chunk) pairs jointly. ``pairs_scored`` counts every pair sent to the model."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 32, max_length: int = 512, field: str = "text"):
        # Imported here so code that only uses top_n/merge_candidates does not load torch.
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self.batch_size = batch_size
        self.field = field  # "text" = chunk with its title/section prefix, "body" = chunk only
        self.model = CrossEncoder(model_name, max_length=max_length)
        self.pairs_scored = 0

    @classmethod
    def from_config(cls, config: dict, model_name: str | None = None) -> CrossEncoderReranker:
        """Model: ``model_name`` if given, else ``rerank.model`` in the config, else DEFAULT_MODEL."""
        rerank = config.get("rerank", {})
        return cls(
            model_name or rerank.get("model", DEFAULT_MODEL),
            batch_size=int(rerank.get("batch_size", 32)),
        )

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """One relevance score per text (an unbounded logit; higher means more relevant)."""
        if not texts:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict(
            [(query, text) for text in texts],
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        self.pairs_scored += len(texts)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def rerank_scores(reranker, query: str, corpus: Corpus, scope: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Cross-encoder score at each candidate position of ``scope``, ``-inf`` everywhere else."""
    out = np.full(len(scope), -np.inf, dtype=np.float32)
    if len(positions):
        # positions index into scope; scope[p] is the chunk's row in the corpus.
        texts = [getattr(corpus.chunks[int(scope[p])], reranker.field) for p in positions]
        out[positions] = reranker.score(query, texts)
    return out


class RerankedHybridRetriever(_Base):
    """Weighted and RRF candidates from ``hybrid``, reordered by the cross-encoder."""

    name = "hybrid_rerank"

    def __init__(self, hybrid: HybridRetriever, reranker, n_candidates: int = 30):
        self.corpus = hybrid.corpus
        self.hybrid = hybrid
        self.reranker = reranker
        self.n_candidates = n_candidates

    def candidates(self, query: str, scope: np.ndarray) -> np.ndarray:
        """Scope positions of weighted-fusion top-n merged with RRF top-n."""
        h = self.hybrid
        d = h.dense.score(query, scope)
        s = h.bm25.score(query, scope)
        weighted = weighted_fusion(prepare_fusion(d, s, h.norm, h.pool), h.alpha)
        return merge_candidates(weighted, rrf_fusion(d, s, h.pool, h.rrf_k), n=self.n_candidates)

    def score(self, query: str, scope: np.ndarray) -> np.ndarray:
        """Cross-encoder scores for the candidates, ``-inf`` for every other chunk."""
        return rerank_scores(self.reranker, query, self.corpus, scope, self.candidates(query, scope))
