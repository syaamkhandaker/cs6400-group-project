"""On-disk cache of document chunks and their embedding matrix.

A cache directory is named by a hash of the settings that change the vectors:
dataset, split, chunk size, tokenizer, and embedding model. Inside it:

* ``chunks.jsonl`` — one chunk per line, row-aligned with the matrix
* ``vectors.npy`` — float32 array of shape (n_chunks, dim)
* ``meta.json`` — the key and the config that produced it

The retrieval pipeline loads this directory instead of re-embedding papers.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.chunking import Chunk
from rag.config import REPO_ROOT

FORMAT_VERSION = 1


@dataclass
class EmbeddingCache:
    key: str
    dataset: str
    split: str
    model: str
    chunks: list[Chunk]
    vectors: np.ndarray
    path: Path


def embedding_cache_key(config: dict, dataset: str, split: str) -> str:
    """Hash the inputs that change chunk boundaries or embedding values."""
    payload = {
        "format": FORMAT_VERSION,
        "dataset": dataset,
        "split": split,
        "chunking": {
            "max_tokens": int(config["chunking"]["max_tokens"]),
            "tokenizer": config["chunking"]["tokenizer"],
        },
        "dense": {"model": config["dense"]["model"]},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def cache_directory(config: dict, dataset: str, split: str, root: Path | None = None) -> Path:
    return _cache_root(config, root) / embedding_cache_key(config, dataset, split)


def save_embedding_cache(
    config: dict,
    dataset: str,
    split: str,
    chunks: list[Chunk],
    vectors: np.ndarray,
    root: Path | None = None,
) -> EmbeddingCache:
    """Write chunks and vectors atomically under the config hash."""
    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if len(chunks) != len(vectors):
        raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")

    key = embedding_cache_key(config, dataset, split)
    dest = _cache_root(config, root) / key
    tmp = dest.parent / f".{dest.name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    meta = {
        "key": key,
        "format": FORMAT_VERSION,
        "dataset": dataset,
        "split": split,
        "model": config["dense"]["model"],
        "dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "num_chunks": len(chunks),
        "chunking": {
            "max_tokens": int(config["chunking"]["max_tokens"]),
            "tokenizer": config["chunking"]["tokenizer"],
        },
    }
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    with (tmp / "chunks.jsonl").open("w") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    np.save(tmp / "vectors.npy", vectors)

    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    return EmbeddingCache(key, dataset, split, meta["model"], chunks, vectors, dest)


def load_embedding_cache(
    config: dict,
    dataset: str,
    split: str,
    root: Path | None = None,
) -> EmbeddingCache:
    """Load the cache for this config, or raise if it has not been built."""
    key = embedding_cache_key(config, dataset, split)
    path = _cache_root(config, root) / key
    meta_path = path / "meta.json"
    chunks_path = path / "chunks.jsonl"
    vectors_path = path / "vectors.npy"
    if not meta_path.is_file() or not chunks_path.is_file() or not vectors_path.is_file():
        raise FileNotFoundError(
            f"No embedding cache at {path}. Build it with:\n"
            f"  python scripts/build_cache.py --dataset {dataset} --split {split}"
        )

    meta = json.loads(meta_path.read_text())
    if meta.get("key") != key:
        raise ValueError(f"Cache key mismatch in {path}: meta has {meta.get('key')!r}, expected {key!r}")

    chunks = [
        Chunk.from_dict(json.loads(line))
        for line in chunks_path.read_text().splitlines()
        if line.strip()
    ]
    vectors = np.load(vectors_path)
    if vectors.shape[0] != len(chunks):
        raise ValueError(
            f"Cache at {path} has {len(chunks)} chunks and {vectors.shape[0]} vectors"
        )
    return EmbeddingCache(key, dataset, split, meta["model"], chunks, vectors, path)


def _cache_root(config: dict, root: Path | None) -> Path:
    if root is not None:
        return root
    raw = Path(config.get("cache", {}).get("dir", "cache/embeddings"))
    if not raw.is_absolute():
        raw = REPO_ROOT / raw
    return raw
