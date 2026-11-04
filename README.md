# Hybrid RAG

Retrieval for QASPER and BEIR-SciFact, comparing a BM25 baseline, a dense (MiniLM) baseline, and a weighted dense + BM25 hybrid. Answers come from a local Ollama model.

## Setup

```bash
uv sync
uv add nltk scipy        # BM25 stemming and sparse scoring (skip if already in pyproject.toml)
python scripts/download_data.py
ollama serve
ollama pull llama3.2
```

`uv sync` installs the project dependencies. The download script puts QASPER v0.3 in `data/qasper/` and SciFact in `data/scifact/`. If you already have a different model pulled, set `OLLAMA_MODEL` instead of pulling `llama3.2`. Ollama is only needed for the end-to-end answer test, not for retrieval evaluation.

Build the embedding caches once:

```bash
python scripts/build_cache.py --dataset qasper --split validation
python scripts/build_cache.py --dataset scifact
```

## Retrieval evaluation

`scripts/run_eval.py` runs BM25, dense, and hybrid retrieval over the same cached chunks and scores them against the gold labels.

```bash
# QASPER: search within each question's paper (default)
python scripts/run_eval.py --dataset qasper --split validation --scope paper

# QASPER: harder setting, all papers pooled
python scripts/run_eval.py --dataset qasper --split validation --scope global

# BEIR-SciFact
python scripts/run_eval.py --dataset scifact
```

Useful options:

| Flag | Default | Purpose |
|---|---|---|
| `--num-papers N` | all | Seeded random sample of QASPER papers |
| `--seed` | 13 | Seed for paper sampling and the bootstrap |
| `--tune-frac` | 0.3 | Fraction of sampled papers used to choose alpha |
| `--alpha-grid` | 0.0 to 1.0 by 0.1 | Alpha values swept for weighted fusion |
| `--norm` | `minmax` | Score normalisation: `minmax` or `zscore` |
| `--pool` | 100 | Candidates taken from each retriever before fusion |
| `--bm25-field` | `text` | `text` (with title/section prefix) or `body` |
| `--tau` | 0.9 | Token-F1 threshold for fuzzy evidence matching |
| `--tag` | none | Suffix for output filenames |
| `--cache-dir` | from config | Load a specific cache directory directly |

### Methods

- **BM25**: Okapi BM25 implemented from scratch (`src/rag/retrieval/bm25.py`). Lowercase alphanumeric tokens, Lucene stopwords, Porter stemming. IDF is computed over the whole cached split, even when a query is limited to one paper.
- **Dense**: cosine similarity between the query and the cached MiniLM chunk vectors. No reranking, query rewriting, or expansion.
- **Hybrid (weighted)**: `alpha * dense + (1 - alpha) * bm25`, each normalised over its top-`pool` candidates. Scores are exact, so a chunk outside the other retriever's pool is clipped to that pool's minimum rather than guessed.
- **Hybrid (RRF)**: reciprocal rank fusion over the same pools, kept as an ablation.

### Setup and metrics

- **Ranking unit.** Chunks are collapsed to a unit by their best score: a paragraph for QASPER, an abstract for SciFact. `k` in the metrics means the number of distinct units.
- **QASPER gold.** Evidence strings are matched to paragraphs (exact match first, then token F1 >= `--tau`). Gold is the union of evidence across answerable annotations. Unanswerable questions, questions without text evidence, and questions whose evidence cannot be matched are excluded and counted in the results JSON. Figure and table evidence is dropped because it is not in the paragraph corpus.
- **Tuning and reporting.** Alpha is chosen on a tune fold and reported on a disjoint test fold. For QASPER the folds are disjoint sets of papers. For SciFact the tune fold is the train qrels and the test fold is the test qrels. Baselines are reported on the test fold.
- **Metrics.** Recall, Precision, Hit and nDCG at k = 1, 3, 5, 10, plus MRR@10. A paired bootstrap gives 95% intervals for hybrid minus each baseline.
- **Precision caveat.** Most QASPER questions have one or two gold paragraphs, so Precision@k is capped at |gold| / k.

### Outputs

Each run writes three files to `results/`, named by dataset and scope (for example `qasper_paper`):

- `results_<tag>.json`: arguments, subset and exclusion counts, evidence match rate, chosen alpha, alpha curves, test and tune metrics, bootstrap intervals, latency.
- `table_<tag>.md`: the test-fold metrics table.
- `per_query_<tag>.jsonl`: for every query, the top retrieved units for each method with source document, section, chunk id, score, a gold flag, and the retrieved text. This is the provenance record and the input for error analysis.

### Sanity checks

- On SciFact, BM25 nDCG@10 should land near the published BEIR value of about 0.665.
- In the results JSON, the alpha = 0.0 and alpha = 1.0 rows of the alpha curve should be close to the BM25 and dense rows.
- Check `evidence_match_rate` and the exclusion counts for QASPER. A low match rate means the paper loader is missing some paragraphs.

## Tests

```bash
uv run pytest tests -q
uv run pytest tests/test_qasper_chunk.py -s
uv run pytest tests/test_rag_end_to_end.py -s
```

The first command is the unit tests. The second prints every chunk of one QASPER paper. The third retrieves SciFact passages and asks Ollama to answer from them. Ollama has to be running for that last test.

## Layout

```
scripts/
  download_data.py     fetch QASPER and SciFact
  build_cache.py       chunk papers, embed, save cache
  run_eval.py          baseline and hybrid retrieval evaluation
src/rag/
  chunking.py, embeddings.py, cache.py, config.py, llm.py
  retrieval/
    pipeline.py        dense retrieval + LLM answer
    corpus.py          shared chunked corpus and ranking units
    bm25.py            BM25 from scratch
    retrievers.py      BM25, dense, hybrid, fusion
  eval/
    metrics.py         ranking metrics and bootstrap
    tasks.py           QASPER and SciFact query/gold builders
```
