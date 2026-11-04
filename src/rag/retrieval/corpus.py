"""The chunked corpus every retriever searches, loaded from the embedding cache.

All retrievers share one ``Corpus`` so that BM25, dense and hybrid see exactly
the same chunks in the same row order.

Relevance is judged on *units*, not chunks:

* QASPER: a unit is one paragraph (all windows of a split paragraph collapse
  to ``doc:section:paragraph``, i.e. the chunk_id without its trailing part).
* SciFact: a unit is one abstract (``doc_id``).

A unit's score is the maximum score over its chunks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk


def unit_of(chunk: Chunk, dataset: str) -> str:
    if dataset == "qasper":
        return chunk.chunk_id.rsplit(":", 1)[0]
    return chunk.doc_id


@dataclass
class Corpus:
    dataset: str
    chunks: list[Chunk]
    vectors: np.ndarray
    unit_ids: np.ndarray  # object array of str, row-aligned with chunks
    doc_rows: dict[str, np.ndarray]
    all_rows: np.ndarray
    path: Path | None = None

    @classmethod
    def build(cls, dataset: str, chunks: list[Chunk], vectors: np.ndarray, path: Path | None = None) -> Corpus:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        rows: dict[str, list[int]] = {}
        for i, chunk in enumerate(chunks):
            rows.setdefault(chunk.doc_id, []).append(i)
        return cls(
            dataset=dataset,
            chunks=chunks,
            vectors=np.ascontiguousarray(vectors, dtype=np.float32),
            unit_ids=np.array([unit_of(c, dataset) for c in chunks], dtype=object),
            doc_rows={d: np.asarray(r, dtype=np.int64) for d, r in rows.items()},
            all_rows=np.arange(len(chunks), dtype=np.int64),
            path=path,
        )

    @classmethod
    def from_dir(cls, dataset: str, path: Path | str) -> Corpus:
        path = Path(path)
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (path / "chunks.jsonl").read_text().splitlines()
            if line.strip()
        ]
        return cls.build(dataset, chunks, np.load(path / "vectors.npy"), path)

    @classmethod
    def from_config(cls, config: dict, dataset: str, split: str | None) -> Corpus:
        from rag.cache import cache_directory
        from rag.data.papers import resolve_split

        return cls.from_dir(dataset, cache_directory(config, dataset, resolve_split(dataset, split)))

    def scope(self, doc_id: str | None) -> np.ndarray:
        """Row indices a query may search: one paper, or the whole corpus."""
        if doc_id is None:
            return self.all_rows
        return self.doc_rows.get(doc_id, np.zeros(0, dtype=np.int64))