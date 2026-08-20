"""
generation.py
==============
Answer generation stage. Two interchangeable backends behind one interface
(`Generator.generate(query, context_chunks) -> GenerationResult`):

  1. ExtractiveGenerator  - zero-dependency, deterministic, sub-millisecond.
                             Picks the sentence(s) in the top retrieved
                             chunk(s) with the highest lexical overlap with
                             the query. This is what the offline benchmark
                             (and the P50/P70/P100 numbers in the README)
                             actually runs, since it needs to be runnable
                             with no network/API key and still be a real,
                             inspectable generation step - not a canned string.

  2. LLMGenerator          - production backend. Sends the query + retrieved
                             chunks to an LLM (Claude, via the Anthropic
                             Messages API) with a strict "answer only from
                             the provided context, say you don't know
                             otherwise" system prompt, structured retries on
                             transient failures, and a hard timeout. This is
                             what the web demo's answer panel calls.

Both return the same GenerationResult shape so the harness and guardrails
don't need to know which backend produced the answer.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

SYSTEM_PROMPT = (
    "You are a grounded question-answering assistant. Answer the user's question using ONLY the "
    "provided context passages. If the context does not contain enough information to answer, say "
    "exactly: \"I don't have enough grounded information in the retrieved context to answer that.\" "
    "Keep answers concise (1-3 sentences). Do not use outside knowledge."
)


@dataclass
class GenerationResult:
    answer: str
    backend: str
    latency_ms: float
    used_chunk_ids: list
    raw: dict | None = None


class ExtractiveGenerator:
    """Deterministic, dependency-free generator: ranks sentences across the
    retrieved chunks by content-word overlap with the query and returns the
    best 1-2 sentences. Serves as the benchmarkable default backend and as a
    grounded fallback if the LLM backend errors out."""

    name = "extractive"
    STOP = {
        "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or", "what",
        "which", "who", "how", "does", "do", "did", "it", "this", "that", "for", "with", "as", "by",
    }

    def _content_words(self, text: str) -> set[str]:
        return {w for w in re.findall(r"\w+", text.lower()) if w not in self.STOP and len(w) > 2}

    def generate(self, query: str, context_chunks: list) -> GenerationResult:
        t0 = time.perf_counter()
        q_words = self._content_words(query)
        best_sent, best_score, best_chunk_id = "", -1.0, None
        for ch in context_chunks:
            for sent in re.split(r"(?<=[.!?])\s+", ch.text):
                s_words = self._content_words(sent)
                if not s_words:
                    continue
                overlap = len(q_words & s_words) / max(len(q_words), 1)
                if overlap > best_score:
                    best_score, best_sent, best_chunk_id = overlap, sent.strip(), ch.chunk_id
        if not best_sent and context_chunks:
            best_sent = context_chunks[0].text.split(".")[0].strip() + "."
            best_chunk_id = context_chunks[0].chunk_id
        latency_ms = (time.perf_counter() - t0) * 1000
        answer = best_sent if best_sent else "I don't have enough grounded information in the retrieved context to answer that."
        return GenerationResult(
            answer=answer,
            backend=self.name,
            latency_ms=latency_ms,
            used_chunk_ids=[best_chunk_id] if best_chunk_id else [],
        )


class LLMGenerator:
    """Production backend calling Claude via the Anthropic Messages API.
    Structured I/O in, structured result out; retried on 429/5xx."""

    name = "llm_claude"

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-6", timeout_s: float = 8.0):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.timeout_s = timeout_s

    def generate(self, query: str, context_chunks: list, max_retries: int = 2) -> GenerationResult:
        if not self.api_key or requests is None:
            raise RuntimeError("ANTHROPIC_API_KEY not set (or `requests` unavailable) for LLMGenerator.")

        context_block = "\n\n".join(f"[{c.chunk_id}] {c.text}" for c in context_chunks)
        user_msg = f"Context:\n{context_block}\n\nQuestion: {query}"

        last_err = None
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "max_tokens": 300,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": user_msg}],
                    },
                    timeout=self.timeout_s,
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    payload = resp.json()
                    text = "".join(b.get("text", "") for b in payload.get("content", []) if b.get("type") == "text")
                    return GenerationResult(
                        answer=text.strip(),
                        backend=self.name,
                        latency_ms=latency_ms,
                        used_chunk_ids=[c.chunk_id for c in context_chunks],
                        raw=payload,
                    )
                if resp.status_code in (429, 500, 502, 503) and attempt < max_retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise RuntimeError(f"Anthropic API failed: {resp.status_code} {resp.text[:200]}")
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    time.sleep(0.3 * (attempt + 1))
                    continue
                raise RuntimeError(f"Anthropic API failed after retries: {last_err}") from last_err
        raise RuntimeError(f"Anthropic API failed after retries: {last_err}")
