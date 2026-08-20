"""
stt.py
=======
Speech-to-text stage using the ElevenLabs Speech-to-Text API (chosen over
Sarvam here because MSMARCO-XI's per-language demo needs broad language
coverage plus strong English performance; Sarvam is the better pick if the
deployment target is India-only/Indic-first telephony audio - swap the
class below for a SarvamSTT with the same `.transcribe()` interface if so).

Docs: https://elevenlabs.io/docs/api-reference/speech-to-text

This module is written as a real, callable client (structured request,
timeout, retries, error surface) even though this sandbox has no outbound
network / API key to actually hit the endpoint. The harness treats it as one
more tool call with the same retry/error-handling contract as retrieval or
generation.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    latency_ms: float
    raw: dict | None = None


class TranscriptionError(Exception):
    pass


class ElevenLabsSTT:
    def __init__(self, api_key: str | None = None, model_id: str = "scribe_v1", timeout_s: float = 10.0):
        self.api_key = api_key or os.environ.get("ELEVENLABS_API_KEY")
        self.model_id = model_id
        self.timeout_s = timeout_s

    def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm", max_retries: int = 2) -> TranscriptionResult:
        if not self.api_key:
            raise TranscriptionError("ELEVENLABS_API_KEY not set. Provide a key to call the real STT endpoint.")
        if requests is None:
            raise TranscriptionError("`requests` package not available in this environment.")

        headers = {"xi-api-key": self.api_key}
        files = {"file": ("audio", audio_bytes, mime_type)}
        data = {"model_id": self.model_id}

        last_err = None
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = requests.post(
                    ELEVENLABS_STT_URL, headers=headers, files=files, data=data, timeout=self.timeout_s
                )
                latency_ms = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    payload = resp.json()
                    return TranscriptionResult(
                        text=payload.get("text", ""),
                        language=payload.get("language_code", "en"),
                        confidence=payload.get("language_probability", 1.0),
                        latency_ms=latency_ms,
                        raw=payload,
                    )
                if resp.status_code in (429, 500, 502, 503) and attempt < max_retries:
                    time.sleep(0.25 * (attempt + 1))  # backoff, then retry
                    continue
                raise TranscriptionError(f"ElevenLabs STT failed: {resp.status_code} {resp.text[:200]}")
            except Exception as e:  # network error -> retry then raise
                last_err = e
                if attempt < max_retries:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise TranscriptionError(f"ElevenLabs STT request failed after retries: {last_err}") from last_err
        raise TranscriptionError(f"ElevenLabs STT failed after retries: {last_err}")


class MockSTT:
    """Deterministic offline stand-in used by the benchmark/demo harness so
    the rest of the pipeline (chunking/retrieval/guardrails/generation) can
    be exercised and timed without a live STT API key. It simply returns the
    text it was "spoken" with, after a small simulated processing delay
    representative of real STT round-trip latency for short utterances."""

    def transcribe_text(self, text: str, simulated_latency_ms: float = 0.0) -> TranscriptionResult:
        t0 = time.perf_counter()
        if simulated_latency_ms:
            time.sleep(simulated_latency_ms / 1000.0)
        return TranscriptionResult(
            text=text, language="en", confidence=1.0, latency_ms=(time.perf_counter() - t0) * 1000
        )
