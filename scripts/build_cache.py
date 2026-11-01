#!/usr/bin/env python3
"""Chunk every paper, embed the chunks in large batches, and save the cache.

The output directory is ``cache/embeddings/<config hash>/`` with
``chunks.jsonl`` and ``vectors.npy``. The retrieval pipeline loads that
directory instead of embedding the corpus again.

Examples::

    python scripts/build_cache.py --dataset qasper --split validation
    python scripts/build_cache.py --dataset scifact --limit 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from tqdm import tqdm

from rag.cache import cache_directory, save_embedding_cache
from rag.chunking import Chunker
from rag.config import REPO_ROOT, load_raw_config
from rag.data.papers import load_papers, resolve_split
from rag.embeddings import HuggingFaceEmbedder


def main() -> None:
    args = _parse_args()
    config = load_raw_config(args.config)
    split = resolve_split(args.dataset, args.split)
    dest = cache_directory(config, args.dataset, split)
    if dest.is_dir() and (dest / "vectors.npy").is_file() and not args.force:
        print(f"Cache already exists: {dest}")
        print("Pass --force to rebuild it.")
        return

    papers = load_papers(args.dataset, split, args.data_dir)
    if args.limit is not None:
        papers = papers[: args.limit]
    if not papers:
        raise SystemExit(f"No papers loaded for {args.dataset} ({split}) from {args.data_dir}")

    chunker = Chunker.from_config(config)
    chunks = []
    for paper in tqdm(papers, desc="chunk"):
        chunks.extend(chunker.chunk_paper(paper))
    if not chunks:
        raise SystemExit("Chunking produced no text.")

    batch_size = args.batch_size or int(config.get("cache", {}).get("batch_size", 256))
    embedder = HuggingFaceEmbedder(config["dense"]["model"], batch_size=batch_size)
    texts = [chunk.text for chunk in chunks]
    parts = []
    for start in tqdm(range(0, len(texts), batch_size), desc="embed"):
        parts.append(embedder.embed_documents(texts[start : start + batch_size]))
    vectors = np.vstack(parts)

    cached = save_embedding_cache(config, args.dataset, split, chunks, vectors)
    print(
        f"Saved {len(chunks)} chunks, dim {vectors.shape[1]}, "
        f"key {cached.key} -> {cached.path}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("qasper", "scifact"))
    parser.add_argument("--split", default=None, help="QASPER split. SciFact always uses the corpus.")
    parser.add_argument("--config", type=Path, default=None, help="Defaults to configs/default.yaml")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Embedding batch size. Defaults to cache.batch_size in the config (256).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only chunk the first N papers.")
    parser.add_argument("--force", action="store_true", help="Rebuild an existing cache directory.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
