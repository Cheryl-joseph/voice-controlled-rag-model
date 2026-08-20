"""
guardrails.py
==============
Three independent guardrail checks the harness runs at different pipeline
stages. Each returns a GuardrailResult so the harness can decide to abstain
("I don't have grounded information to answer that") instead of forcing an
answer:

  1. UnsafeInputGuard   - runs on the raw transcribed query, before retrieval.
                           Blocks unsafe/inappropriate requests (violence,
                           self-harm, illegal activity, prompt-injection
                           attempts against the system prompt).
  2. OffTopicGuard      - runs after retrieval. If the best retrieval score
                           is below a confidence floor, the query is judged
                           off-topic for this corpus rather than forcing a
                           low-quality answer.
  3. GroundingGuard     - runs after generation. Checks that the generated
                           answer's claims are lexically supported by the
                           retrieved chunks (n-gram overlap check), a cheap
                           but real hallucination check that does not need a
                           second LLM call.

These are intentionally conservative and explainable (regex + lexical
overlap) rather than a black-box classifier, so failures are debuggable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 1. Unsafe / inappropriate input
# ---------------------------------------------------------------------------
UNSAFE_PATTERNS = [
    r"\bhow (?:do|can) i (?:make|build|synthesi[sz]e) (?:a )?(?:bomb|explosive|weapon)\b",
    r"\b(?:kill|murder|assassinate) (?:my|the|a)\b",
    r"\bhow to (?:hurt|harm|kill) myself\b",
    r"\bsuicide method\b",
    r"\bhow to hack (?:into|a)\b",
    r"\bmake (?:meth|methamphetamine|heroin|nerve gas)\b",
    r"\bignore (?:previous|all|prior) instructions\b",
    r"\byou are now\b.{0,30}\bunrestricted\b",
    r"\bdisregard your (?:system prompt|guardrails|instructions)\b",
]
UNSAFE_RE = re.compile("|".join(UNSAFE_PATTERNS), re.IGNORECASE)


@dataclass
class GuardrailResult:
    passed: bool
    stage: str
    reason: str = ""
    detail: dict | None = None


class UnsafeInputGuard:
    stage = "unsafe_input"

    def check(self, query: str) -> GuardrailResult:
        m = UNSAFE_RE.search(query)
        if m:
            return GuardrailResult(
                passed=False,
                stage=self.stage,
                reason="Query matched an unsafe-content / prompt-injection pattern.",
                detail={"matched_span": m.group(0)},
            )
        return GuardrailResult(passed=True, stage=self.stage)


# ---------------------------------------------------------------------------
# 2. Off-topic (low retrieval confidence)
# ---------------------------------------------------------------------------
class OffTopicGuard:
    """Confidence floor on raw BM25 mass (not the RRF-fused rank score,
    which compresses too tightly on small corpora to separate on/off-topic
    queries). BM25's raw score scales with real lexical evidence for the
    query, which is a decent cheap proxy for "is this corpus even about
    what was asked".

    Known limitation, documented rather than hidden: purely lexical
    confidence floors can be fooled by a query that shares a strong keyword
    with an unrelated passage (e.g. asking about "weather" in an unrelated
    sense still lights up a weather passage). Production systems should
    calibrate this floor against a held-out labelled on-topic/off-topic
    query set, and/or add an embedding-similarity floor once real
    embeddings replace TF-IDF.
    """

    stage = "off_topic"

    def __init__(self, min_bm25: float = 2.5):
        self.min_bm25 = min_bm25

    def check(self, scored_chunks: list) -> GuardrailResult:
        if not scored_chunks:
            return GuardrailResult(False, self.stage, "No chunks retrieved for this query.")
        top = scored_chunks[0]
        top_bm25 = top.method_scores.get("bm25", 0.0)
        if top_bm25 < self.min_bm25:
            return GuardrailResult(
                False,
                self.stage,
                "Top retrieval confidence is below threshold; query is likely outside the corpus's coverage.",
                detail={"top_bm25": top_bm25, "threshold": self.min_bm25},
            )
        return GuardrailResult(True, self.stage, detail={"top_bm25": top_bm25})


# ---------------------------------------------------------------------------
# 3. Grounding / hallucination check
# ---------------------------------------------------------------------------
class GroundingGuard:
    stage = "grounding"

    def __init__(self, min_overlap_ratio: float = 0.28):
        self.min_overlap_ratio = min_overlap_ratio

    @staticmethod
    def _content_words(text: str) -> set[str]:
        STOP = {
            "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "on", "and", "or",
            "it", "this", "that", "for", "with", "as", "by", "be", "at", "from", "which", "its",
        }
        return {w for w in re.findall(r"\w+", text.lower()) if w not in STOP and len(w) > 2}

    def check(self, answer: str, context_chunks: list[str]) -> GuardrailResult:
        if not answer.strip():
            return GuardrailResult(False, self.stage, "Empty answer.")
        answer_words = self._content_words(answer)
        context_words = set()
        for c in context_chunks:
            context_words |= self._content_words(c)
        if not answer_words:
            return GuardrailResult(False, self.stage, "Answer contained no checkable content words.")
        overlap = answer_words & context_words
        ratio = len(overlap) / len(answer_words)
        passed = ratio >= self.min_overlap_ratio
        return GuardrailResult(
            passed,
            self.stage,
            reason="" if passed else "Answer is not sufficiently grounded in retrieved context (possible hallucination).",
            detail={"overlap_ratio": round(ratio, 3), "threshold": self.min_overlap_ratio},
        )


class GuardrailPipeline:
    def __init__(self):
        self.unsafe = UnsafeInputGuard()
        self.off_topic = OffTopicGuard()
        self.grounding = GroundingGuard()
