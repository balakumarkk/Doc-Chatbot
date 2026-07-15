"""
api/main.py
-----------
FastAPI application factory.

This is the entry point for uvicorn:
  uvicorn api.main:app --reload

All routers are registered here. Add new routers below as the API grows.
Middleware, exception handlers, and startup/shutdown events also live here.

Planned future routers (add to include_router list below when ready):
  - api.routers.history   → GET/POST /history  (conversation persistence)
  - api.routers.feedback  → POST /feedback      (thumbs up/down per answer)
  - api.routers.auth      → POST /auth/token    (API key / JWT auth)
  - api.routers.admin     → GET /admin/stats    (usage stats)
"""

import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load .env into os.environ FIRST — before any sub-module reads os.getenv().
# chat/llm.py also calls load_dotenv() at import time, but if it's imported
# lazily (inside a request handler), this call here guarantees the key is
# available from the very start of the process.
load_dotenv()

from api.config import settings
from api.routers import chat, health, models

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CORS origins
# ---------------------------------------------------------------------------
_origins = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:3000",   # Next.js / alt dev server
    "http://localhost:8000",   # Same-origin (Swagger UI)
]
if settings.frontend_url:
    _origins.append(settings.frontend_url)

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    application = FastAPI(
        title       = settings.app_title,
        description = settings.app_description,
        version     = settings.app_version,
        docs_url    = "/docs",
        redoc_url   = "/redoc",
    )

    # ── Middleware ────────────────────────────────────────────────────────────
    application.add_middleware(
        CORSMiddleware,
        allow_origins     = _origins,
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # ── Startup: warm up model + DB so first request is fast ─────────────────
    @application.on_event("startup")
    async def _warmup() -> None:
        log.info("Warming up embedding model and vector DB…")
        try:
            from api.dependencies import get_collection, get_model
            get_model()
            get_collection()
            log.info("Warmup complete — API ready.")
        except Exception as exc:
            log.error("Warmup failed (API may be slow on first request): %s", exc)

    # ── Routers ───────────────────────────────────────────────────────────────
    application.include_router(health.router)
    application.include_router(models.router)
    application.include_router(chat.router,  prefix="/chat")

    # Future routers — uncomment when implemented:
    # application.include_router(history.router, prefix="/history")
    # application.include_router(feedback.router, prefix="/feedback")
    # application.include_router(auth.router,     prefix="/auth")
    # application.include_router(admin.router,    prefix="/admin")

    return application


# Module-level app instance — uvicorn imports this
app = create_app()
