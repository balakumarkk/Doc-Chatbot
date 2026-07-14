"""
embedder/store.py
-----------------
Chroma vector DB operations:
  - get_collection  : open / create a persistent Chroma collection
  - upsert_chunks   : store chunk embeddings (idempotent via chunk_id)
  - query_collection: embed a question and retrieve top-K matching chunks
"""

from __future__ import annotations

import json
import os
from typing import Any

import chromadb
from chromadb.config import Settings


# ---------------------------------------------------------------------------
# Collection accessor
# ---------------------------------------------------------------------------
def get_collection(
    db_path: str = "vector_db",
    collection_name: str = "docs",
) -> chromadb.Collection:
    """
    Open (or create) a persistent Chroma collection on disk.

    Parameters
    ----------
    db_path         : Directory where Chroma stores its files.
    collection_name : Name of the collection inside the DB.

    Returns
    -------
    chromadb.Collection — ready for upsert / query.
    """
    os.makedirs(db_path, exist_ok=True)

    client = chromadb.PersistentClient(
        path=db_path,
        settings=Settings(anonymized_telemetry=False),
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )

    return collection


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------
def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection: chromadb.Collection,
) -> int:
    """
    Upsert chunk embeddings into Chroma.

    Uses ``chunk_id`` as the document ID — re-running after new scrapes is safe:
    existing chunks are updated in-place, new ones are inserted.

    Parameters
    ----------
    chunks     : List of chunk dicts (from chunks.jsonl).
    embeddings : Corresponding embedding vectors from encode_passages().
    collection : Chroma collection returned by get_collection().

    Returns
    -------
    Number of records upserted.
    """
    if not chunks:
        return 0

    ids        = [c["chunk_id"]    for c in chunks]
    documents  = [c["text"]        for c in chunks]
    metadatas  = [_build_metadata(c) for c in chunks]

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    return len(ids)


def _build_metadata(chunk: dict) -> dict[str, Any]:
    """
    Flatten chunk fields into a Chroma-compatible metadata dict.

    Chroma metadata values must be str / int / float / bool.
    The ``headings`` dict is JSON-serialised to a string.
    """
    return {
        "source_url":  chunk.get("source_url", ""),
        "source_file": chunk.get("source_file", ""),
        "chunk_index": chunk.get("chunk_index", 0),
        "token_count": chunk.get("token_count", 0),
        "headings":    json.dumps(chunk.get("headings", {})),
    }


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def query_collection(
    question: str,
    collection: chromadb.Collection,
    model=None,
    top_k: int = 5,
) -> list[dict]:
    """
    Embed *question* and retrieve the top-K most relevant chunks.

    Uses encode_query() (with BGE prefix) for the question vector.

    Parameters
    ----------
    question   : User's question (raw text, no prefix needed).
    collection : Chroma collection to search.
    model      : Loaded SentenceTransformer (uses singleton if None).
    top_k      : Number of results to return.

    Returns
    -------
    List of dicts, each containing:
        text, source_url, source_file, headings (dict), token_count, distance
    Ordered from most to least relevant (lowest distance first).
    """
    from .model import encode_query, load_model

    if model is None:
        model = load_model()

    query_vector = encode_query(question, model)

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text":        doc,
            "source_url":  meta.get("source_url", ""),
            "source_file": meta.get("source_file", ""),
            "headings":    json.loads(meta.get("headings", "{}")),
            "token_count": meta.get("token_count", 0),
            "distance":    round(dist, 4),
        })

    return output
