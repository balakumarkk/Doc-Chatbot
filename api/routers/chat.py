"""
api/routers/chat.py
-------------------
POST /chat/stream — SSE streaming RAG answer.

Flow:
  1. retrieve()         embed question + query Chroma (distance-filtered)
  2. emit "sources"     frontend can show source panel immediately
  3. build_prompt()     numbered context blocks + grounded system prompt
  4. stream_answer()    Groq API token stream
  5. emit "token"       one event per token
  6. emit "done"        signals stream end

SSE event payload shapes (see api/schemas/chat.py):
  {"type": "sources", "data": [SourceChunk, ...]}
  {"type": "token",   "data": "text fragment"}
  {"type": "done"}
  {"type": "error",   "data": "error message"}

Why fetch + ReadableStream instead of EventSource on the client?
  EventSource only supports GET requests. We need POST to send the
  question body. The SSE wire format is identical either way.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncIterator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from api.config import settings
from api.dependencies import get_collection, get_model
from api.schemas.chat import ChatRequest, SourceChunk

log = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _serialize_sources(chunks: list[dict]) -> list[dict]:
    """Convert raw retriever chunks into SourceChunk-shaped dicts."""
    return [
        SourceChunk(
            index    = i + 1,
            url      = c.get("source_url", ""),
            text     = c.get("text", ""),
            distance = round(c.get("distance", 0.0), 4),
            headings = c.get("headings", {}),
        ).model_dump()
        for i, c in enumerate(chunks)
    ]


def _event(payload: dict) -> dict:
    """Wrap a payload dict as an SSE data field."""
    return {"data": json.dumps(payload, ensure_ascii=False)}


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
@router.post(
    "/stream",
    summary="Stream a RAG answer via SSE",
    response_description="Server-Sent Events stream",
)
async def chat_stream(
    req:        ChatRequest,
    collection  = Depends(get_collection),
    model       = Depends(get_model),
) -> EventSourceResponse:
    """
    Retrieve relevant documentation chunks for the question, then stream
    a grounded LLM answer token-by-token.

    **SSE event sequence:**
    1. `sources` — retrieved chunks (emitted before first token)
    2. `token`   — one event per LLM output token
    3. `done`    — signals end of stream

    The frontend connects with `fetch()` and reads the `ReadableStream`.
    """

    async def _generate() -> AsyncIterator[dict]:
        try:
            from chat.retriever import retrieve
            from chat.llm import build_prompt, stream_answer

            # --- Retrieve ---
            chunks = retrieve(
                req.question,
                top_k             = req.top_k,
                distance_threshold = req.threshold,
                db_path            = settings.vector_db_path,
                collection_name    = settings.collection_name,
            )
            log.info(
                "Retrieved %d chunks for question: %.60s…",
                len(chunks), req.question,
            )

            # --- Emit sources first (frontend shows panel immediately) ---
            yield _event({"type": "sources", "data": _serialize_sources(chunks)})

            # --- Build prompt ---
            messages = build_prompt(req.question, chunks)

            # --- Stream tokens ---
            for token in stream_answer(
                messages,
                model       = req.model,
                temperature = req.temperature,
                max_tokens  = settings.default_max_tokens,
            ):
                yield _event({"type": "token", "data": token})

            # --- 6. Done ---
            yield _event({"type": "done"})

        except Exception as exc:
            log.error("Stream error for question %r: %s", req.question, exc, exc_info=True)
            yield _event({"type": "error", "data": str(exc)})

    return EventSourceResponse(_generate())
