"""
api/routers/health.py
---------------------
GET /health — liveness + readiness check.

Returns corpus size so the frontend / deployment platform can confirm
the vector DB is loaded and accessible.
"""

import logging

from fastapi import APIRouter, Depends

from api.config import settings
from api.dependencies import get_collection

log = logging.getLogger(__name__)
router = APIRouter(tags=["System"])


@router.get(
    "/health",
    summary="Liveness + readiness check",
    response_description="API status and vector DB stats",
)
async def health(collection=Depends(get_collection)) -> dict:
    """
    Returns:
    - `status`: always `"ok"` if the server is reachable
    - `collection`: name of the active Chroma collection
    - `document_count`: number of embedded chunks in the DB
    - `default_model`: currently configured Groq model
    """
    try:
        count = collection.count()
    except Exception as exc:
        log.warning("Could not read collection count: %s", exc)
        count = -1

    return {
        "status":         "ok",
        "collection":     settings.collection_name,
        "document_count": count,
        "default_model":  settings.default_model,
    }
