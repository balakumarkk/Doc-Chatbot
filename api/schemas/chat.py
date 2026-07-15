"""
api/schemas/chat.py
-------------------
Pydantic models for the /chat/* endpoints.

Keep all request and response shapes here so they are:
  - Easy to version (add ChatRequestV2 when needed)
  - Auto-documented in FastAPI's /docs Swagger UI
  - Reusable across routers without circular imports

Future schemas to add here:
  - ConversationMessage   (for multi-turn history)
  - FeedbackRequest       (thumbs up/down per answer)
  - HistoryEntry          (stored conversation)
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from api.config import settings


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """Body for POST /chat/stream."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question.",
        examples=["How do agents call tools?"],
    )
    top_k: int = Field(
        default=settings.default_top_k,
        ge=1,
        le=20,
        description="Number of chunks to retrieve from the vector DB.",
    )
    threshold: float = Field(
        default=settings.default_threshold,
        ge=0.0,
        le=1.0,
        description="Distance threshold — chunks beyond this are dropped.",
    )
    model: str = Field(
        default=settings.default_model,
        description="Groq model ID.",
        examples=["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    )
    temperature: float = Field(
        default=settings.default_temperature,
        ge=0.0,
        le=1.0,
        description="Sampling temperature (lower = more factual).",
    )


# ---------------------------------------------------------------------------
# Response pieces
# ---------------------------------------------------------------------------
class SourceChunk(BaseModel):
    """One retrieved chunk sent in the 'sources' SSE event."""

    index:    int            # 1-based position (matches [1], [2] citations)
    url:      str            # Original web URL
    text:     str            # Chunk content (may be long — frontend truncates)
    distance: float          # Cosine distance (lower = more relevant)
    headings: dict[str, str] # e.g. {"h1": "Agents Guide", "h2": "Tool Calling"}


class StreamEvent(BaseModel):
    """Shape of every SSE data payload."""

    type: str  # "sources" | "token" | "done" | "error"
    data: Any = None

    model_config = {"json_schema_extra": {
        "examples": [
            {"type": "sources", "data": [{"index": 1, "url": "https://...", "text": "...", "distance": 0.237, "headings": {}}]},
            {"type": "token",   "data": "Agents call tools by"},
            {"type": "done"},
            {"type": "error",   "data": "GROQ_API_KEY not set"},
        ]
    }}
