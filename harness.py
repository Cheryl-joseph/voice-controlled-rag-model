"""
harness.py
===========
The orchestration layer. This is NOT a single prompt-in/text-out call - it
is a structured pipeline of discrete "tool calls" (STT -> retrieval ->
guardrails -> generation -> grounding check), each with:
  - structured input/output (dataclasses, not free text)
  - per-stage timing (feeds the latency benchmark)
  - retry logic where a stage can transiently fail (STT, LLM generation)
  - explicit error recovery / fallback paths (e.g. LLMGenerator failing
    falls back to ExtractiveGenerator rather than crashing the request)
  - early-exit when a guardrail fails, with a structured refusal response
    instead of a forced answer

PipelineResult is the single structured object returned to callers (the
API layer / the web UI), so the frontend can render per-stage latency,
which guardrail (if any) fired, and the grounded answer + citations in one
shot.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from chunking import ChunkRouter, chunk_corpus
from data_loader import load_corpus
from generation import ExtractiveGenerator, GenerationResult
from guardrails import GuardrailPipeline, GuardrailResult
from retrieval import HybridVectorStore, ScoredChunk
from stt import MockSTT
from tools import ToolRouter


NOT_IN_DATABASE_MSG = "The database does not provide data for the given question."


@dataclass
class StageTiming:
    stage: str
    latency_ms: float


@dataclass
class PipelineResult:
    query_text: str
    status: str  # "answered" | "answered_via_tool" | "refused_unsafe" | "refused_off_topic" | "refused_ungrounded" | "error"
    answer: str | None
    retrieved: list = field(default_factory=list)  # list of {chunk_id, text, score}
    guardrail_trace: list = field(default_factory=list)  # list of GuardrailResult
    stage_timings: list = field(default_factory=list)  # list of StageTiming
    total_latency_ms: float = 0.0
    tool_name: str | None = None
    tool_detail: dict | None = None

    def to_dict(self) -> dict:
        return {
            "query_text": self.query_text,
            "status": self.status,
            "answer": self.answer,
            "retrieved": self.retrieved,
            "guardrail_trace": [g.__dict__ for g in self.guardrail_trace],
            "stage_timings": [t.__dict__ for t in self.stage_timings],
            "total_latency_ms": round(self.total_latency_ms, 3),
            "tool_name": self.tool_name,
            "tool_detail": self.tool_detail,
        }


class RagHarness:
    """Structured orchestrator: builds the index once, then serves queries
    through a fixed stage graph with retries + guardrails + fallbacks."""

    def __init__(self, data_source: str = "sample", top_k: int = 4):
        self.top_k = top_k
        self.router = ChunkRouter()
        self.store = HybridVectorStore()
        self.guardrails = GuardrailPipeline()
        self.generator = ExtractiveGenerator()
        self.stt = MockSTT()
        self.tool_router = ToolRouter()
        self._build_index(data_source)

    def _build_index(self, data_source: str):
        docs = load_corpus(data_source)
        chunks = chunk_corpus(docs, self.router)
        self.store.index(chunks)
        self.chunk_count = len(chunks)
        self.doc_count = len(docs)

    # -- individual stages, each independently timed & retry-capable -------

    def _stage_tool_route(self, query: str, timings: list):
        t0 = time.perf_counter()
        routed = self.tool_router.try_route(query)
        timings.append(StageTiming("tool_router", (time.perf_counter() - t0) * 1000))
        return routed

    def _stage_stt(self, spoken_text: str, timings: list) -> str:
        t0 = time.perf_counter()
        result = self.stt.transcribe_text(spoken_text)
        timings.append(StageTiming("stt", (time.perf_counter() - t0) * 1000))
        return result.text

    def _stage_unsafe_guard(self, query: str, timings: list, trace: list) -> GuardrailResult:
        t0 = time.perf_counter()
        result = self.guardrails.unsafe.check(query)
        timings.append(StageTiming("guardrail_unsafe_input", (time.perf_counter() - t0) * 1000))
        trace.append(result)
        return result

    def _stage_retrieve(self, query: str, timings: list, max_retries: int = 1) -> list[ScoredChunk]:
        last_err = None
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                results = self.store.search(query, top_k=self.top_k)
                timings.append(StageTiming("retrieval", (time.perf_counter() - t0) * 1000))
                return results
            except Exception as e:  # transient failure -> retry
                last_err = e
                timings.append(StageTiming(f"retrieval_error_attempt_{attempt}", (time.perf_counter() - t0) * 1000))
        raise RuntimeError(f"Retrieval failed after retries: {last_err}")

    def _stage_off_topic_guard(self, scored: list, timings: list, trace: list) -> GuardrailResult:
        t0 = time.perf_counter()
        result = self.guardrails.off_topic.check(scored)
        timings.append(StageTiming("guardrail_off_topic", (time.perf_counter() - t0) * 1000))
        trace.append(result)
        return result

    def _stage_generate(self, query: str, chunks: list, timings: list) -> GenerationResult:
        t0 = time.perf_counter()
        try:
            result = self.generator.generate(query, chunks)
        except Exception:
            # error recovery: degrade to a trivially safe extractive fallback
            result = GenerationResult(
                answer=chunks[0].text.split(".")[0].strip() + "." if chunks else "",
                backend="fallback_first_sentence",
                latency_ms=0.0,
                used_chunk_ids=[chunks[0].chunk_id] if chunks else [],
            )
        timings.append(StageTiming("generation", (time.perf_counter() - t0) * 1000))
        return result

    def _stage_grounding_guard(self, answer: str, chunks: list, timings: list, trace: list) -> GuardrailResult:
        t0 = time.perf_counter()
        result = self.guardrails.grounding.check(answer, [c.text for c in chunks])
        timings.append(StageTiming("guardrail_grounding", (time.perf_counter() - t0) * 1000))
        trace.append(result)
        return result

    # -- full pipeline -------------------------------------------------

    def run(self, spoken_text: str) -> PipelineResult:
        t_start = time.perf_counter()
        timings: list[StageTiming] = []
        trace: list[GuardrailResult] = []

        query = self._stage_stt(spoken_text, timings)

        unsafe_result = self._stage_unsafe_guard(query, timings, trace)
        if not unsafe_result.passed:
            return self._finish(query, "refused_unsafe", None, [], trace, timings, t_start)

        # tool routing: some queries (e.g. arithmetic) are correctly outside
        # what a retrieval-grounded pipeline can answer from the corpus --
        # route them to a dedicated tool instead of forcing a RAG answer.
        routed = self._stage_tool_route(query, timings)
        if routed is not None:
            tool_name, tool_result = routed
            if tool_result.handled:
                return self._finish(
                    query, "answered_via_tool", tool_result.output, [], trace, timings, t_start,
                    tool_name=tool_name, tool_detail=tool_result.detail,
                )
            # matched the tool's pattern but it couldn't safely evaluate it
            # (e.g. malformed expression) -> fall through to normal refusal
            # handling below rather than silently ignoring the routing.

        try:
            scored = self._stage_retrieve(query, timings)
        except Exception as e:
            return self._finish(query, "error", f"Retrieval error: {e}", [], trace, timings, t_start)

        off_topic_result = self._stage_off_topic_guard(scored, timings, trace)
        if not off_topic_result.passed:
            return self._finish(query, "refused_off_topic", NOT_IN_DATABASE_MSG, scored, trace, timings, t_start)

        top_chunks = [sc.chunk for sc in scored]
        gen_result = self._stage_generate(query, top_chunks, timings)

        grounding_result = self._stage_grounding_guard(gen_result.answer, top_chunks, timings, trace)
        if not grounding_result.passed:
            return self._finish(query, "refused_ungrounded", NOT_IN_DATABASE_MSG, scored, trace, timings, t_start)

        return self._finish(query, "answered", gen_result.answer, scored, trace, timings, t_start)

    def _finish(self, query, status, answer, scored, trace, timings, t_start, tool_name=None, tool_detail=None) -> PipelineResult:
        total_ms = (time.perf_counter() - t_start) * 1000
        retrieved = [
            {
                "chunk_id": sc.chunk.chunk_id,
                "doc_id": sc.chunk.doc_id,
                "text": sc.chunk.text,
                "strategy": sc.chunk.strategy,
                "score": round(sc.score, 5),
                "method_scores": sc.method_scores,
            }
            for sc in scored
        ]
        return PipelineResult(
            query_text=query,
            status=status,
            answer=answer,
            retrieved=retrieved,
            guardrail_trace=trace,
            stage_timings=timings,
            total_latency_ms=total_ms,
            tool_name=tool_name,
            tool_detail=tool_detail,
        )
