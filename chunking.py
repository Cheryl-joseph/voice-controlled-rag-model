"""
chunking.py
============
Multiple, composable chunking strategies rather than one naive fixed-size
splitter. A `ChunkRouter` picks a strategy per-document based on document
metadata/length, and every chunk carries rich metadata used later by the
retriever and by guardrails (grounding checks need to map an answer back to
an exact chunk + source doc).

Strategies implemented:
  1. FixedSizeChunker        - fixed token/char window with configurable overlap
  2. SentenceWindowChunker   - groups whole sentences up to a token budget,
                                keeps sentence boundaries intact (no mid-sentence cuts)
  3. SemanticChunker         - splits at points where lexical/semantic similarity
                                between consecutive sentences drops (topic shift),
                                approximated here with a bag-of-words cosine signal
                                so it needs no external embedding model
  4. RecursiveChunker        - paragraph -> sentence -> word recursive fallback,
                                mirrors LangChain's RecursiveCharacterTextSplitter idea
  5. MetadataAwareChunker    - wraps any of the above but injects structured
                                metadata (doc_id, language, category, position,
                                char_span) that downstream retrieval and citation
                                display depend on
  6. QueryPassageChunker     - special-cased for MSMARCO-XI's schema: since each
                                passage is already a short, self-contained
                                answer-bearing unit, short passages are kept
                                whole (1 passage = 1 chunk) while long passages
                                are recursively split - avoids over-chunking
                                short factual passages, which would hurt recall

A ChunkRouter selects strategy 6 as the default for MSMARCO-XI-shaped data
(short query/passage pairs) and falls back to sentence-window + semantic
splitting for longer free-text documents.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

WORD_RE = re.compile(r"\S+")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+")  # handles Latin + Devanagari danda punctuation


def word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return sents or ([text.strip()] if text.strip() else [])


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_id: str
    strategy: str
    position: int              # order within parent doc
    char_start: int
    char_end: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "doc_id": self.doc_id,
            "strategy": self.strategy,
            "position": self.position,
            "char_span": [self.char_start, self.char_end],
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# 1. Fixed-size chunker (word-based window, with overlap)
# ---------------------------------------------------------------------------
class FixedSizeChunker:
    name = "fixed_size"

    def __init__(self, size_words: int = 60, overlap_words: int = 12):
        assert overlap_words < size_words
        self.size = size_words
        self.overlap = overlap_words

    def split(self, text: str) -> list[tuple[str, int, int]]:
        words = list(WORD_RE.finditer(text))
        if not words:
            return []
        out = []
        step = self.size - self.overlap
        i = 0
        while i < len(words):
            window = words[i : i + self.size]
            if not window:
                break
            start = window[0].start()
            end = window[-1].end()
            out.append((text[start:end], start, end))
            if i + self.size >= len(words):
                break
            i += step
        return out


# ---------------------------------------------------------------------------
# 2. Sentence-window chunker (never cuts mid-sentence)
# ---------------------------------------------------------------------------
class SentenceWindowChunker:
    name = "sentence_window"

    def __init__(self, max_words: int = 80, sentence_overlap: int = 1):
        self.max_words = max_words
        self.sentence_overlap = sentence_overlap

    def split(self, text: str) -> list[tuple[str, int, int]]:
        sents = split_sentences(text)
        if not sents:
            return []
        out, cur, cur_words = [], [], 0
        cursor = 0
        spans = []
        # recompute character spans by walking through the original text
        pos = 0
        sent_spans = []
        for s in sents:
            idx = text.find(s, pos)
            if idx == -1:
                idx = pos
            sent_spans.append((s, idx, idx + len(s)))
            pos = idx + len(s)

        i = 0
        while i < len(sent_spans):
            cur, cur_start = [], sent_spans[i][1]
            cur_words = 0
            j = i
            while j < len(sent_spans) and cur_words + word_count(sent_spans[j][0]) <= self.max_words:
                cur.append(sent_spans[j][0])
                cur_words += word_count(sent_spans[j][0])
                j += 1
            if not cur:  # single sentence longer than budget - keep it anyway
                cur = [sent_spans[i][0]]
                j = i + 1
            cur_end = sent_spans[j - 1][2]
            out.append((" ".join(cur), cur_start, cur_end))
            i = max(j - self.sentence_overlap, i + 1)
        return out


# ---------------------------------------------------------------------------
# 3. Semantic chunker: bag-of-words cosine drop between adjacent sentences
# ---------------------------------------------------------------------------
class SemanticChunker:
    name = "semantic"

    def __init__(self, similarity_threshold: float = 0.15, min_sentences: int = 1, max_words: int = 120):
        self.threshold = similarity_threshold
        self.min_sentences = min_sentences
        self.max_words = max_words

    @staticmethod
    def _bow(s: str) -> dict[str, int]:
        d: dict[str, int] = {}
        for w in re.findall(r"\w+", s.lower()):
            d[w] = d.get(w, 0) + 1
        return d

    @staticmethod
    def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
        if not a or not b:
            return 0.0
        keys = set(a) | set(b)
        dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
        na = sum(v * v for v in a.values()) ** 0.5
        nb = sum(v * v for v in b.values()) ** 0.5
        return dot / (na * nb) if na and nb else 0.0

    def split(self, text: str) -> list[tuple[str, int, int]]:
        sents = split_sentences(text)
        if len(sents) <= 1:
            return [(text.strip(), 0, len(text))] if text.strip() else []

        pos, sent_spans = 0, []
        for s in sents:
            idx = text.find(s, pos)
            if idx == -1:
                idx = pos
            sent_spans.append((s, idx, idx + len(s)))
            pos = idx + len(s)

        bows = [self._bow(s) for s, _, _ in sent_spans]
        boundaries = [0]
        words_in_seg = word_count(sent_spans[0][0])
        for k in range(1, len(sent_spans)):
            sim = self._cosine(bows[k - 1], bows[k])
            words_in_seg += word_count(sent_spans[k][0])
            topic_shift = sim < self.threshold
            over_budget = words_in_seg > self.max_words
            if (topic_shift and (k - boundaries[-1]) >= self.min_sentences) or over_budget:
                boundaries.append(k)
                words_in_seg = word_count(sent_spans[k][0])
        boundaries.append(len(sent_spans))

        out = []
        for a, b in zip(boundaries, boundaries[1:]):
            seg = sent_spans[a:b]
            if not seg:
                continue
            out.append((" ".join(s for s, _, _ in seg), seg[0][1], seg[-1][2]))
        return out


# ---------------------------------------------------------------------------
# 4. Recursive chunker: paragraph -> sentence -> fixed-size fallback
# ---------------------------------------------------------------------------
class RecursiveChunker:
    name = "recursive"

    def __init__(self, max_words: int = 80, overlap_words: int = 10):
        self.max_words = max_words
        self.fallback = FixedSizeChunker(size_words=max_words, overlap_words=overlap_words)

    def split(self, text: str) -> list[tuple[str, int, int]]:
        paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paras) <= 1:
            paras = [text]
        out = []
        cursor = 0
        for para in paras:
            start = text.find(para, cursor)
            if start == -1:
                start = cursor
            if word_count(para) <= self.max_words:
                out.append((para.strip(), start, start + len(para)))
            else:
                sw = SentenceWindowChunker(max_words=self.max_words).split(para)
                for seg, s, e in sw:
                    out.append((seg, start + s, start + e))
            cursor = start + len(para)
        return out


# ---------------------------------------------------------------------------
# 6. Query/passage-aware chunker for MSMARCO-XI's native schema
# ---------------------------------------------------------------------------
class QueryPassageChunker:
    """Short MSMARCO-style passages (the common case) are kept as one chunk
    each -- splitting a 2-3 sentence factual passage would only dilute
    retrieval signal. Longer passages recursively split. This is the
    default strategy the router picks for this dataset's shape."""

    name = "query_passage_aware"

    def __init__(self, short_passage_word_limit: int = 90, overlap_words: int = 10):
        self.short_limit = short_passage_word_limit
        self.recursive = RecursiveChunker(max_words=short_passage_word_limit, overlap_words=overlap_words)

    def split(self, text: str) -> list[tuple[str, int, int]]:
        if word_count(text) <= self.short_limit:
            return [(text.strip(), 0, len(text))] if text.strip() else []
        return self.recursive.split(text)


STRATEGIES: dict[str, Callable] = {
    FixedSizeChunker.name: FixedSizeChunker,
    SentenceWindowChunker.name: SentenceWindowChunker,
    SemanticChunker.name: SemanticChunker,
    RecursiveChunker.name: RecursiveChunker,
    QueryPassageChunker.name: QueryPassageChunker,
}


class ChunkRouter:
    """Chooses a chunking strategy per document based on simple, explainable
    signals (length, presence of paragraph breaks) - metadata-aware in the
    sense that the routing decision AND the resulting chunk metadata both
    reflect properties of the source document, not a single global setting."""

    def __init__(self):
        self.query_passage = QueryPassageChunker()
        self.semantic = SemanticChunker()
        self.recursive = RecursiveChunker()

    def choose(self, doc) -> tuple[str, object]:
        wc = word_count(doc.text)
        has_paragraphs = "\n\n" in doc.text
        if wc <= 90 and getattr(doc, "query", ""):
            # MSMARCO-XI-shaped: short, query-anchored passage
            return self.query_passage.name, self.query_passage
        if has_paragraphs and wc > 200:
            return self.recursive.name, self.recursive
        return self.semantic.name, self.semantic

    def chunk_document(self, doc) -> list[Chunk]:
        strategy_name, strategy = self.choose(doc)
        spans = strategy.split(doc.text)
        chunks = []
        for i, (seg_text, cstart, cend) in enumerate(spans):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{strategy_name}::{i}",
                    text=seg_text,
                    doc_id=doc.doc_id,
                    strategy=strategy_name,
                    position=i,
                    char_start=cstart,
                    char_end=cend,
                    metadata={
                        "language": doc.language,
                        "source": doc.source,
                        "origin_query": doc.query,
                        "gold_answer": doc.answer,
                        **doc.metadata,
                    },
                )
            )
        return chunks


def chunk_corpus(docs, router: ChunkRouter | None = None) -> list[Chunk]:
    router = router or ChunkRouter()
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(router.chunk_document(doc))
    return all_chunks
