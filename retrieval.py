"""
retrieval.py
=============
Hybrid retriever over the chunk store: dense-style TF-IDF cosine similarity
+ BM25 lexical scoring, fused with Reciprocal Rank Fusion (RRF). No external
ML dependency (no faiss/chromadb/sentence-transformers) is required to run
this in a restricted sandbox -- everything is implemented on top of numpy so
it is honest about what is actually executing, while remaining a drop-in
`VectorStore` interface: swap `TfidfEmbedder` for a real sentence-transformer
or an API embedding call in production without touching the retriever logic.

Why hybrid, not just one method:
  - TF-IDF/cosine ("vector" leg) captures soft term overlap and is robust to
    small wording differences.
  - BM25 ("lexical" leg) is strong on exact keyword / rare-term matches
    (e.g. named entities, numbers) that TF-IDF cosine can under-weight after
    normalization.
  - RRF fusion combines both rankings without needing to tune a weighted sum,
    and is the same technique production hybrid-search stacks (Elasticsearch,
    Weaviate, Azure AI Search) use.

Metadata-aware filtering (language, category) is supported as a pre-filter
before scoring, so retrieval can be scoped e.g. to the query's language.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Stopwords are excluded from the retrieval index itself (not just from the
# grounding-guard's content-word check). Leaving them in previously let a
# query like "what's 2+2" score a false-positive BM25 match against an
# unrelated passage (e.g. "Inflation is the rate at which...") purely
# because both share "what"/"is"/"the" -- which then fooled the off-topic
# guard into treating the query as in-corpus. Filtering stopwords out of
# the index/query tokens fixes retrieval confidence at the source instead
# of trying to patch it downstream.
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or",
    "it", "this", "that", "for", "with", "as", "by", "be", "at", "from", "which", "its",
    "what", "how", "does", "do", "did", "who", "s", "t", "can", "will", "would", "should",
}

# Very light, dependency-free suffix stripping (not a full Porter stemmer)
# so "boiling"/"boils"/"boiled" or "translate"/"translation" share a token.
# Applied identically to corpus and query text so matching stays symmetric.
_SUFFIXES = ("ational", "ization", "iveness", "ational", "ing", "tion", "sion", "ies", "ed", "es", "s")


def _light_stem(word: str) -> str:
    if len(word) <= 4 or not word.isalpha():
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokenize(text: str, stem: bool = True, drop_stopwords: bool = True) -> list[str]:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [_light_stem(t) for t in toks] if stem else toks


@dataclass
class ScoredChunk:
    chunk: object  # chunking.Chunk
    score: float
    rank: int
    method_scores: dict


class TfidfEmbedder:
    """Minimal TF-IDF vectorizer (fit + transform) using pure numpy."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None

    def fit(self, docs_tokens: list[list[str]]):
        df = Counter()
        for toks in docs_tokens:
            for t in set(toks):
                df[t] += 1
        self.vocab = {t: i for i, t in enumerate(sorted(df))}
        n = len(docs_tokens)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0
        return self

    def transform(self, docs_tokens: list[list[str]]) -> np.ndarray:
        n, v = len(docs_tokens), len(self.vocab)
        mat = np.zeros((n, v), dtype=np.float32)
        for row, toks in enumerate(docs_tokens):
            tf = Counter(toks)
            length = max(len(toks), 1)
            for t, c in tf.items():
                idx = self.vocab.get(t)
                if idx is not None:
                    mat[row, idx] = (c / length) * self.idf[idx]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.doc_tokens: list[list[str]] = []
        self.df: Counter = Counter()
        self.idf: dict[str, float] = {}
        self.avgdl = 0.0
        self.N = 0

    def fit(self, docs_tokens: list[list[str]]):
        self.doc_tokens = docs_tokens
        self.N = len(docs_tokens)
        self.avgdl = sum(len(d) for d in docs_tokens) / max(self.N, 1)
        df = Counter()
        for toks in docs_tokens:
            for t in set(toks):
                df[t] += 1
        self.df = df
        self.idf = {t: math.log((self.N - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
        return self

    def score(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.N, dtype=np.float32)
        q_tf = Counter(query_tokens)
        for i, doc in enumerate(self.doc_tokens):
            dl = len(doc)
            tf = Counter(doc)
            s = 0.0
            for t in q_tf:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                s += idf * (freq * (self.k1 + 1)) / max(denom, 1e-6)
            scores[i] = s
        return scores


class HybridVectorStore:
    """RRF fusion of TF-IDF cosine similarity + BM25, with optional metadata
    pre-filtering (e.g. language). This is the "vector DB" layer of the
    pipeline; in production the TfidfEmbedder swap-out is a real embedding
    model/API and this class would delegate nearest-neighbour search to
    FAISS/Chroma/pgvector, but the interface (index/search) stays the same.
    """

    def __init__(self, rrf_k: int = 60):
        self.chunks: list = []
        self.tokens: list[list[str]] = []
        self.tfidf = TfidfEmbedder()
        self.tfidf_matrix: np.ndarray | None = None
        self.bm25 = BM25()
        self.rrf_k = rrf_k

    def index(self, chunks: list):
        self.chunks = chunks
        self.tokens = [tokenize(c.text) for c in chunks]
        self.tfidf.fit(self.tokens)
        self.tfidf_matrix = self.tfidf.transform(self.tokens)
        self.bm25.fit(self.tokens)
        return self

    def _candidate_indices(self, language: str | None) -> list[int]:
        if not language:
            return list(range(len(self.chunks)))
        return [i for i, c in enumerate(self.chunks) if c.metadata.get("language", "en") == language]

    def search(self, query: str, top_k: int = 5, language: str | None = None) -> list[ScoredChunk]:
        if not self.chunks:
            return []
        q_tokens = tokenize(query)
        candidates = self._candidate_indices(language)
        if not candidates:
            candidates = list(range(len(self.chunks)))  # fall back: no docs in that language

        q_vec = self.tfidf.transform([q_tokens])[0]
        cos_scores = self.tfidf_matrix[candidates] @ q_vec  # cosine (rows already L2-normalized)
        bm25_scores = self.bm25.score(q_tokens)[candidates]

        cos_rank = {idx: r for r, idx in enumerate(np.argsort(-cos_scores))}
        bm25_rank = {idx: r for r, idx in enumerate(np.argsort(-bm25_scores))}

        rrf_scores = {}
        for local_i in range(len(candidates)):
            rrf = 1.0 / (self.rrf_k + cos_rank[local_i] + 1) + 1.0 / (self.rrf_k + bm25_rank[local_i] + 1)
            rrf_scores[local_i] = rrf

        order = sorted(rrf_scores.keys(), key=lambda i: -rrf_scores[i])[:top_k]
        results = []
        for rank, local_i in enumerate(order):
            global_i = candidates[local_i]
            results.append(
                ScoredChunk(
                    chunk=self.chunks[global_i],
                    score=float(rrf_scores[local_i]),
                    rank=rank,
                    method_scores={
                        "tfidf_cosine": float(cos_scores[local_i]),
                        "bm25": float(bm25_scores[local_i]),
                    },
                )
            )
        return results
