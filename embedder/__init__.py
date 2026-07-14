"""
embedder/__init__.py
--------------------
Public API for the embedder package.
"""

from .model import load_model, encode_passages, encode_query
from .store import get_collection, upsert_chunks, query_collection

__all__ = [
    "load_model",
    "encode_passages",
    "encode_query",
    "get_collection",
    "upsert_chunks",
    "query_collection",
]
