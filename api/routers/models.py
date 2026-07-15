"""
api/routers/models.py
---------------------
GET /models — list available Groq chat models.

The frontend uses this to populate the model selector dropdown in the
settings panel so users can switch models without redeploying.

Future extension: add POST /models/set to persist user's model preference.
"""

import logging

from fastapi import APIRouter

from api.config import settings

log = logging.getLogger(__name__)
router = APIRouter(tags=["Models"])

# Models to exclude from the chat model list
_EXCLUDE_KEYWORDS = ("whisper", "tts", "guard", "orpheus", "allam", "compound")


@router.get(
    "/models",
    summary="List available Groq chat models",
    response_description="Sorted list of model IDs suitable for chat",
)
async def list_models() -> dict:
    """
    Fetches the current model list from Groq and filters to chat-capable models.

    Falls back to `[default_model]` if the Groq API is unreachable.
    """
    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        raw = client.models.list()
        chat_models = sorted(
            m.id
            for m in raw.data
            if not any(kw in m.id.lower() for kw in _EXCLUDE_KEYWORDS)
        )
        return {"models": chat_models, "default": settings.default_model}

    except Exception as exc:
        log.warning("Could not fetch Groq model list: %s", exc)
        return {"models": [settings.default_model], "default": settings.default_model}
