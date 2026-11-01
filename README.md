# Hybrid RAG

Dense retrieval over QASPER and BEIR-SciFact, with answers from a local Ollama model.

## Setup

```bash
uv sync
python scripts/download_data.py
ollama serve
ollama pull llama3.2
```

`uv sync` installs the project dependencies. The download script puts QASPER v0.3 in `data/qasper/` and SciFact in `data/scifact/`. If you already have a different model pulled, set `OLLAMA_MODEL` instead of pulling `llama3.2`.

Build the embedding caches once:

```bash
python scripts/build_cache.py --dataset qasper --split validation
python scripts/build_cache.py --dataset scifact
```

## Tests

```bash
uv run pytest tests -q
uv run pytest tests/test_qasper_chunk.py -s
uv run pytest tests/test_rag_end_to_end.py -s
```

The first command is the unit tests. The second prints every chunk of one QASPER paper. The third retrieves SciFact passages and asks Ollama to answer from them. Ollama has to be running for that last test.
