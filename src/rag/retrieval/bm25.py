"""Okapi BM25 implemented from scratch on a sparse term-weight matrix.

Document-side BM25 weights are precomputed once:

    w(t, d) = idf(t) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * |d| / avgdl))
    idf(t)  = ln(1 + (N - df + 0.5) / (df + 0.5))        (Lucene/BEIR variant)

so scoring a query is a column-sum over the query's terms. IDF and avgdl are
computed over the *whole* indexed corpus even when a query is restricted to
one paper (``rows``), so the statistics do not depend on the search scope.

Tokenisation (shared by every sparse component): lowercase, ``[a-z0-9]+``
tokens, Lucene's English stopword list, optional Porter stemming.
"""

from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

STOPWORDS = frozenset(
    "a an and are as at be but by for if in into is it no not of on or such "
    "that the their then there these they this to was will with".split()
)
_WORD = re.compile(r"[a-z0-9]+")


def make_tokenizer(stem: bool = True):
    if stem:
        try:
            from nltk.stem import PorterStemmer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Stemming needs nltk (pip install nltk) or pass stem=False.") from exc
        stem_fn = lru_cache(maxsize=500_000)(PorterStemmer().stem)
    else:

        def stem_fn(word: str) -> str:
            return word

    def tokenize(text: str) -> list[str]:
        return [stem_fn(w) for w in _WORD.findall(text.lower()) if w not in STOPWORDS]

    return tokenize


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.2, b: float = 0.75, stem: bool = True):
        self.k1, self.b = k1, b
        self.tokenize = make_tokenizer(stem)
        self.n_docs = len(texts)
        self.vocab: dict[str, int] = {}

        rows: list[int] = []
        cols: list[int] = []
        tfs: list[int] = []
        lengths = np.zeros(self.n_docs, dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = self.tokenize(text)
            lengths[i] = len(tokens)
            for term, count in Counter(tokens).items():
                rows.append(i)
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                tfs.append(count)

        r = np.asarray(rows, dtype=np.int32)
        c = np.asarray(cols, dtype=np.int32)
        tf = np.asarray(tfs, dtype=np.float32)
        self.avgdl = float(lengths.mean()) if self.n_docs else 0.0
        df = np.bincount(c, minlength=len(self.vocab)).astype(np.float32)
        idf = np.log1p((self.n_docs - df + 0.5) / (df + 0.5))
        norm = k1 * (1.0 - b + b * lengths[r] / max(self.avgdl, 1e-9))
        weights = idf[c] * tf * (k1 + 1.0) / (tf + norm)
        self.weights = sp.csc_matrix((weights, (r, c)), shape=(self.n_docs, len(self.vocab)), dtype=np.float32)

    def scores(self, query: str, rows: np.ndarray | None = None) -> np.ndarray:
        """BM25 score for every doc (or for ``rows`` only), in that order."""
        n_out = self.n_docs if rows is None else len(rows)
        term_ids = sorted({self.vocab[t] for t in self.tokenize(query) if t in self.vocab})
        if not term_ids:
            return np.zeros(n_out, dtype=np.float32)
        total = np.asarray(self.weights[:, term_ids].sum(axis=1)).ravel().astype(np.float32)
        return total if rows is None else total[rows]