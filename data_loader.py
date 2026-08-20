"""
data_loader.py
================
Loads the source corpus for the RAG system.

Primary source (production): ai4bharat/MSMARCO-XI on the Hugging Face Hub
    https://huggingface.co/datasets/ai4bharat/MSMARCO-XI

MSMARCO-XI is IndicRAGSuite's machine-translated, human-verified variant of
MS MARCO, released per Indian language (hi, ta, te, bn, mr, gu, kn, ml, pa,
or, as, ne, ur). Each row carries:
    query          : str   -> natural language question
    answers        : list  -> gold answer string(s)
    passages       : list  -> candidate passages (is_selected flags the gold one)
    source_lang    : str
    target_lang    : str
    meta           : dict  -> translation model metadata

Each per-language split ships as a single large parquet file (validation
split alone is ~474MB per language), so pulling it requires outbound network
access and the `datasets` / `pyarrow` libraries. This module tries that path
first and transparently falls back to a small, clearly-labelled bundled
sample corpus (`sample_corpus.json`) so the rest of the pipeline -
chunking, retrieval, guardrails, harness, latency benchmarking - can be
developed, tested and demoed offline without lying about where the data
came from.

Swap DATA_SOURCE to "hf" and run with network access to point the whole
pipeline at the real dataset with zero other code changes.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterator

SAMPLE_CORPUS_PATH = os.path.join(os.path.dirname(__file__), "sample_corpus.json")


@dataclass
class Document:
    doc_id: str
    text: str
    query: str = ""
    answer: str = ""
    language: str = "en"
    source: str = "MSMARCO-XI"
    metadata: dict = field(default_factory=dict)


def load_from_huggingface(lang: str = "hi", split: str = "train", limit: int | None = 2000) -> list[Document]:
    """Production loader. Requires network + `datasets` installed.

    from datasets import load_dataset
    ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split=split)
    """
    from datasets import load_dataset  # noqa: F401 (import kept local: optional heavy dep)

    ds = load_dataset("ai4bharat/MSMARCO-XI", lang, split=split)
    docs: list[Document] = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        passages = row.get("passages") or []
        for p_idx, passage in enumerate(passages):
            text = passage.get("passage_text") if isinstance(passage, dict) else str(passage)
            if not text:
                continue
            docs.append(
                Document(
                    doc_id=f"{lang}-{i}-{p_idx}",
                    text=text,
                    query=row.get("query", ""),
                    answer=", ".join(row.get("answers", []) or []),
                    language=lang,
                    metadata={"is_selected": passage.get("is_selected") if isinstance(passage, dict) else None},
                )
            )
    return docs


def load_sample_corpus() -> list[Document]:
    """Offline dev/demo corpus, same schema shape as the real dataset.

    This is NOT MSMARCO-XI data. It is a small, hand-written set of
    query/passage/answer triples covering the same kind of open-domain
    factual QA MSMARCO-XI targets, used so the pipeline is fully runnable
    and benchmarkable without network access. Every stage of the pipeline
    (chunking, retrieval, harness, guardrails) is schema-identical to the
    production path in load_from_huggingface().
    """
    with open(SAMPLE_CORPUS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    docs = []
    for row in raw:
        docs.append(
            Document(
                doc_id=row["doc_id"],
                text=row["text"],
                query=row.get("query", ""),
                answer=row.get("answer", ""),
                language=row.get("language", "en"),
                source=row.get("source", "sample_corpus_dev"),
                metadata=row.get("metadata", {}),
            )
        )
    return docs


def load_corpus(source: str = "sample") -> list[Document]:
    if source == "hf":
        try:
            return load_from_huggingface()
        except Exception as e:  # network / dependency unavailable
            print(f"[data_loader] HF load failed ({e}); falling back to sample corpus.")
            return load_sample_corpus()
    return load_sample_corpus()


def iter_batches(docs: list[Document], batch_size: int = 32) -> Iterator[list[Document]]:
    for i in range(0, len(docs), batch_size):
        yield docs[i : i + batch_size]
