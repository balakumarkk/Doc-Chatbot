"""
chat/retriever.py
-----------------
Wraps embedder query into a clean interface for the chat layer.

retrieve(question) -> list[dict]

Each returned chunk:
  { text, source_url, source_file, headings, token_count, distance }

Chunks with distance > DISTANCE_THRESHOLD are dropped — these are
too dissimilar to be useful and would only confuse the LLM.

Calibrated threshold: 0.38
  IN scope  (docs questions):  distances 0.21 – 0.29
  OUT scope (unrelated):       distances 0.42 – 0.49
  Gap of ~0.13 — threshold sits safely in the middle.
"""

from __future__ import annotations

import os

# Singletons — loaded once, reused across all calls
_model = None
_collection = None

DEFAULT_THRESHOLD = 0.38
DEFAULT_TOP_K = 5
DEFAULT_DB_PATH = "vector_db"
DEFAULT_COLLECTION = "docs"


def _get_model():
    global _model
    if _model is None:
        from embedder.model import load_model
        _model = load_model()
    return _model


def _get_collection(db_path: str = DEFAULT_DB_PATH, collection_name: str = DEFAULT_COLLECTION):
    global _collection
    if _collection is None:
        from embedder.store import get_collection
        _collection = get_collection(db_path, collection_name)
    return _collection


def retrieve(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    distance_threshold: float = DEFAULT_THRESHOLD,
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = DEFAULT_COLLECTION,
) -> list[dict]:
    """
    Retrieve the most relevant chunks for *question*.

    Parameters
    ----------
    question           : User's question (raw text).
    top_k              : Max chunks to retrieve before threshold filtering.
    distance_threshold : Drop chunks with distance > this value.
                         Calibrated at 0.38 for this corpus.
    db_path            : Chroma DB directory.
    collection_name    : Chroma collection name.

    Returns
    -------
    List of chunk dicts ordered by relevance (closest first).
    Empty list if no chunks pass the threshold (triggers "I don't know").
    """
    from embedder.store import query_collection

    model      = _get_model()
    collection = _get_collection(db_path, collection_name)

    results = query_collection(question, collection, model, top_k=top_k)

    # Apply distance threshold — drop everything that's too dissimilar
    filtered = [r for r in results if r["distance"] <= distance_threshold]

    return filtered
