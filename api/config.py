"""
api/config.py
-------------
Centralised settings — reads from .env / environment variables.

All API modules import `settings` from here instead of calling os.getenv()
directly. This means:
  - One place to see every config knob
  - Pydantic validates types at startup (wrong value → clear error, not a
    runtime crash)
  - Works locally (.env file) and in production (env vars on Railway/Render)

Add new settings here when adding new features.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Required ─────────────────────────────────────────────────────────────
    groq_api_key: str

    # ── Vector DB ─────────────────────────────────────────────────────────────
    vector_db_path:  str = "vector_db"
    collection_name: str = "docs"

    # ── Chat defaults ─────────────────────────────────────────────────────────
    default_model:       str   = "llama-3.3-70b-versatile"
    default_top_k:       int   = 5
    default_threshold:   float = 0.38
    default_temperature: float = 0.2
    default_max_tokens:  int   = 1024

    # ── CORS ──────────────────────────────────────────────────────────────────
    # Set FRONTEND_URL to your Vercel deployment URL in production
    frontend_url: str = ""

    # ── App metadata ──────────────────────────────────────────────────────────
    app_title:       str = "Doc Chatbot API"
    app_description: str = "RAG chatbot over scraped documentation — Groq + Chroma + BGE"
    app_version:     str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # GROQ_API_KEY and groq_api_key both work
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()


# Convenience alias — most modules just do: from api.config import settings
settings = get_settings()
