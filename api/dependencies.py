"""
api/dependencies.py
-------------------
FastAPI dependency injection for shared resources.

The embedding model and Chroma collection are expensive to load — they should
be initialised once at startup and reused across all requests. This module
provides FastAPI-injectable functions that return the singleton instances.

Usage in a router:
    from fastapi import Depends
    from api.dependencies import get_collection, get_model

    @router.post("/chat/stream")
    async def chat_stream(
        req: ChatRequest,
        collection = Depends(get_collection),
        model      = Depends(get_model),
    ): ...

Adding new shared resources (e.g. a database connection, a cache):
  - Add a loader function here
  - Inject it in the relevant router via Depends()
"""

from functools import lru_cache

from api.config import settings


@lru_cache(maxsize=1)
def get_model():
    """Return the singleton BGE embedding model (loaded once, reused forever)."""
    from embedder.model import load_model
    return load_model()


@lru_cache(maxsize=1)
def get_collection():
    """Return the singleton Chroma collection (opened once, reused forever)."""
    from embedder.store import get_collection as _get_col
    return _get_col(settings.vector_db_path, settings.collection_name)
