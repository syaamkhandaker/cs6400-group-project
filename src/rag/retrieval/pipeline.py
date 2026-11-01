"""Load the embedding cache and return the nearest chunks for a question.

Document vectors come from ``scripts/build_cache.py``. The question is embedded
when ``retrieve`` is called.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from rag.cache import load_embedding_cache
from rag.chunking import Chunk
from rag.config import load_raw_config
from rag.data.papers import resolve_split


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class RetrievalPipeline:
    def __init__(self, embedder, chunks: list[Chunk], vectors: np.ndarray):
        self.embedder = embedder
        self.chunks = chunks
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.vectors.shape[1])
        if len(self.vectors):
            self.index.add(self.vectors)
        rows: dict[str, list[int]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            rows[chunk.doc_id].append(index)
        self.doc_rows = {doc_id: np.asarray(indexes, dtype=np.int64) for doc_id, indexes in rows.items()}

    @classmethod
    def from_cache(cls, dataset: str, split: str, config_path: Path | None = None, embedder=None):
        config = load_raw_config(config_path)
        cached = load_embedding_cache(config, dataset, resolve_split(dataset, split))
        if embedder is None:
            from rag.embeddings import HuggingFaceEmbedder

            embedder = HuggingFaceEmbedder.from_config(config)
        return cls(embedder, cached.chunks, cached.vectors)

    def retrieve(self, question: str, top_k: int = 10, doc_id: str | None = None) -> list[Hit]:
        """Embed ``question`` and return the closest cached chunks.

        Pass ``doc_id`` for QASPER so the search stays inside that paper.
        """
        if top_k <= 0 or self.index.ntotal == 0:
            return []
        query = np.ascontiguousarray(self.embedder.embed_query(question), dtype=np.float32).reshape(1, -1)
        if doc_id is None:
            scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        else:
            rows = self.doc_rows.get(doc_id)
            if rows is None or len(rows) == 0:
                return []
            params = faiss.SearchParameters()
            params.sel = faiss.IDSelectorArray(np.ascontiguousarray(rows, dtype=np.int64))
            scores, indices = self.index.search(query, min(top_k, len(rows)), params=params)
        return [
            Hit(self.chunks[int(row)], float(score))
            for row, score in zip(indices[0], scores[0])
            if row >= 0
        ]

    def answer(self, question: str, llm, *, top_k: int = 5, doc_id: str | None = None) -> tuple[str, list[Hit]]:
        """Retrieve passages, then ask the LLM to answer from those passages only."""
        hits = self.retrieve(question, top_k=top_k, doc_id=doc_id)
        passages = "\n\n".join(f"[{hit.chunk.chunk_id}] {hit.chunk.body}" for hit in hits)
        text = llm.generate(
            "Answer the question using only the passages. Cite the chunk id you used.\n\n"
            f"Passages:\n{passages}\n\nQuestion: {question}"
        )
        return text, hits
