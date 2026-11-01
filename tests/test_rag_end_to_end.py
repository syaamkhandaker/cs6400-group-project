"""Retrieve from the SciFact cache and answer one test query with Ollama.

Requires the embedding cache (`python scripts/build_cache.py --dataset scifact`)
and a running Ollama server. If `generation.model` is not pulled, the test
uses the first model from `ollama list`.

    uv run pytest tests/test_rag_end_to_end.py -s
"""

from __future__ import annotations

import json
import os
import urllib.request

import pytest

from rag.config import REPO_ROOT, load_raw_config
from rag.data.scifact import load_scifact_queries
from rag.llm import OllamaLLM, load_config
from rag.retrieval.pipeline import RetrievalPipeline


def test_scifact_retrieve_then_answer(capsys):
    data_dir = REPO_ROOT / "data"
    if not (data_dir / "scifact" / "queries.jsonl").is_file():
        pytest.skip("SciFact is not in data/scifact/")
    try:
        pipeline = RetrievalPipeline.from_cache("scifact", "test")
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    _use_installed_ollama_model()
    llm = OllamaLLM.from_yaml()
    query = load_scifact_queries(data_dir, "test")[0]
    top_k = int(load_raw_config()["generation"].get("context_chunks", 5))
    try:
        text, hits = pipeline.answer(query.text, llm, top_k=top_k)
    except RuntimeError as exc:
        pytest.skip(str(exc))

    assert hits
    assert text.strip()

    with capsys.disabled():
        print(f"\nmodel: {llm.config.model}")
        print(f"question: {query.text}")
        print(f"gold doc ids: {', '.join(query.relevant_doc_ids)}\n")
        for hit in hits:
            print(f"--- {hit.chunk.chunk_id} doc={hit.chunk.doc_id} score={hit.score:.3f} ---")
            print(hit.chunk.body[:400])
            print()
        print("answer:", text.strip())
        print()


def _use_installed_ollama_model() -> None:
    if os.environ.get("OLLAMA_MODEL"):
        return
    base = load_config().base_url
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as response:
            names = [item["name"] for item in json.load(response).get("models", [])]
    except OSError as exc:
        pytest.skip(f"Ollama is not running at {base}: {exc}")
    wanted = load_config().model
    if any(name == wanted or name.startswith(wanted + ":") for name in names):
        return
    if not names:
        pytest.skip("Ollama has no pulled models")
    os.environ["OLLAMA_MODEL"] = names[0]
