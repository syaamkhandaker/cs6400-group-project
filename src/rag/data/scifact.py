"""BEIR-SciFact abstracts and the 300 test queries.

The corpus is one abstract per document. ``qrels/test.tsv`` lists which
abstracts are relevant to each test query. A repeated document id keeps the
first row.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rag.data.papers import Paper


@dataclass(frozen=True)
class SciFactQuery:
    query_id: str
    text: str
    relevant_doc_ids: tuple[str, ...]


def load_scifact_papers(data_dir: Path) -> list[Paper]:
    papers = []
    seen = set()
    for record in _jsonl(data_dir / "scifact" / "corpus.jsonl"):
        doc_id = str(record.get("_id") or "")
        text = record.get("text") or ""
        if not doc_id or doc_id in seen or not isinstance(text, str) or not text.strip():
            continue
        seen.add(doc_id)
        papers.append(Paper(doc_id, str(record.get("title") or ""), (("Abstract", (text.strip(),)),)))
    return papers


def load_scifact_queries(data_dir: Path, split: str = "test") -> list[SciFactQuery]:
    directory = data_dir / "scifact"
    relevant = _qrels(directory / "qrels" / f"{split}.tsv")
    queries = []
    for record in _jsonl(directory / "queries.jsonl"):
        query_id = str(record.get("_id") or "")
        text = record.get("text") or ""
        if query_id not in relevant or not isinstance(text, str) or not text.strip():
            continue
        queries.append(SciFactQuery(query_id, text.strip(), tuple(relevant[query_id])))
    return queries


def _qrels(path: Path) -> dict[str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run `python scripts/download_data.py --dataset scifact`.")
    relevant = defaultdict(list)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[2].lstrip("-").isdigit() or int(parts[2]) <= 0:
            continue
        if parts[1] not in relevant[parts[0]]:
            relevant[parts[0]].append(parts[1])
    if not relevant:
        raise ValueError(f"No relevance judgments in {path}")
    return relevant


def _jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run `python scripts/download_data.py --dataset scifact`.")
    records = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
