"""Cache key, chunking, and vector search without downloading a model."""

from __future__ import annotations

import numpy as np

from rag.cache import embedding_cache_key, load_embedding_cache, save_embedding_cache
from rag.chunking import Chunker
from rag.data.papers import Paper
from rag.retrieval.pipeline import RetrievalPipeline


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join(token_ids)


class FakeEmbedder:
    def embed_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


CONFIG = {
    "chunking": {"max_tokens": 200, "tokenizer": "sentence-transformers/all-MiniLM-L6-v2"},
    "dense": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
    "cache": {"dir": "cache/embeddings", "batch_size": 256},
}


def test_cache_key_changes_with_model():
    other = {
        **CONFIG,
        "dense": {"model": "sentence-transformers/all-MiniLM-L12-v2"},
    }
    assert embedding_cache_key(CONFIG, "qasper", "validation") != embedding_cache_key(
        other, "qasper", "validation"
    )


def test_long_paragraph_is_split_and_short_one_is_kept():
    paper = Paper(
        "p1",
        "Title",
        (("Intro", ("one two three four", "kept")),),
    )
    chunks = Chunker(WhitespaceTokenizer(), max_tokens=2).chunk_paper(paper)
    bodies = [chunk.body for chunk in chunks]
    assert bodies == ["one two", "three four", "kept"]
    assert chunks[-1].body == "kept"
    assert chunks[0].text.startswith("Title [Intro]\n")


def test_cache_roundtrip_and_pipeline_load(tmp_path):
    paper = Paper("p1", "Title", (("Intro", ("alpha", "beta")),))
    chunks = Chunker(WhitespaceTokenizer(), max_tokens=8).chunk_paper(paper)
    vectors = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    saved = save_embedding_cache(CONFIG, "qasper", "validation", chunks, vectors, root=tmp_path)

    loaded = load_embedding_cache(CONFIG, "qasper", "validation", root=tmp_path)
    assert loaded.path == saved.path
    assert loaded.key == saved.key
    assert [chunk.chunk_id for chunk in loaded.chunks] == [chunk.chunk_id for chunk in chunks]
    assert np.allclose(loaded.vectors, vectors)

    pipeline = RetrievalPipeline(FakeEmbedder(), loaded.chunks, loaded.vectors)
    hits = pipeline.retrieve("which chunk?", top_k=1)
    assert hits[0].chunk.body == "alpha"
    scoped = pipeline.retrieve("which chunk?", top_k=1, doc_id="missing")
    assert scoped == []
