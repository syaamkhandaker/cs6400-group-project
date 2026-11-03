"""Document loader for the two benchmarks this project evaluates.

* QASPER v0.3 — primary. Full NLP papers, plus questions, answers, and gold
  evidence paragraphs in ``rag.data.qasper``.
* BEIR-SciFact — secondary. The abstract corpus (about 5K documents) and the
  300 test queries in ``rag.data.scifact``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paper:
    doc_id: str
    title: str
    sections: tuple[tuple[str, tuple[str, ...]], ...]


def resolve_split(dataset: str, split: str | None) -> str:
    """SciFact document embeddings cover the shared corpus, not a query split."""
    if dataset == "scifact":
        return "corpus"
    if dataset == "qasper":
        return split or "validation"
    raise ValueError(f"Unknown dataset {dataset!r}. Expected 'qasper' or 'scifact'.")


def load_papers(dataset: str, split: str, data_dir: Path) -> list[Paper]:
    """Load the documents that get chunked and embedded."""
    if dataset == "qasper":
        from rag.data.qasper import load_qasper_papers

        return load_qasper_papers(split, data_dir)
    if dataset == "scifact":
        from rag.data.scifact import load_scifact_papers

        return load_scifact_papers(data_dir)
    raise ValueError(f"Unknown dataset {dataset!r}. Expected 'qasper' or 'scifact'.")
