"""
api/ — Doc Chatbot FastAPI package.

Entry point:
  uvicorn api.main:app --reload
"""
from .main import app  # noqa: F401 — re-export for uvicorn

__all__ = ["app"]
