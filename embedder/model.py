"""
embedder/model.py
-----------------
Wraps sentence-transformers for BGE-small-en-v1.5.

BGE models use ASYMMETRIC encoding:
  - encode_passages()  : NO prefix  → use when embedding document chunks for storage
  - encode_query()     : WITH prefix → use when embedding the user's question at search time

Using the same function for both sides degrades retrieval ranking.
"""

from __future__ import annotations

from typing import Optional

# Lazy import so the module can be imported without sentence-transformers
# installed (e.g. for unit tests that mock it).
_model_singleton: Optional[object] = None

# BGE query prefix — only applied to the question, never to document chunks
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
def load_model(model_name: str = DEFAULT_MODEL):
    """
    Load (and cache) the sentence-transformers model.

    The model is downloaded on first call (~130 MB) and cached by
    sentence-transformers in the HuggingFace cache directory.
    Subsequent calls return the cached singleton immediately.

    Parameters
    ----------
    model_name : HuggingFace model identifier.

    Returns
    -------
    SentenceTransformer instance.
    """
    global _model_singleton

    if _model_singleton is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model: {model_name}  (first run downloads ~130 MB)")
        _model_singleton = SentenceTransformer(model_name)
        print("Model ready.")

    return _model_singleton


# ---------------------------------------------------------------------------
# Passage encoding  (document chunks → stored vectors)
# ---------------------------------------------------------------------------
def encode_passages(
    texts: list[str],
    model=None,
    batch_size: int = 64,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Encode document chunks for storage in the vector DB.

    NO prefix is applied — BGE passage encoding uses raw text.

    Parameters
    ----------
    texts        : List of chunk texts to encode.
    model        : Loaded SentenceTransformer (uses singleton if None).
    batch_size   : Encoding batch size.
    show_progress: Show tqdm progress bar.

    Returns
    -------
    List of embedding vectors (each a list of floats).
    """
    if model is None:
        model = load_model()

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,  # cosine similarity works on normalised vectors
        convert_to_numpy=True,
    )
    return embeddings.tolist()


# ---------------------------------------------------------------------------
# Query encoding  (user question → search vector)
# ---------------------------------------------------------------------------
def encode_query(text: str, model=None) -> list[float]:
    """
    Encode a user's question for retrieval.

    The BGE query prefix is prepended automatically.
    This is intentionally separate from encode_passages() to enforce the
    asymmetric encoding required by BGE models.

    Parameters
    ----------
    text  : The user's question (raw, no prefix needed from caller).
    model : Loaded SentenceTransformer (uses singleton if None).

    Returns
    -------
    Single embedding vector (list of floats).
    """
    if model is None:
        model = load_model()

    prefixed = _BGE_QUERY_PREFIX + text.strip()

    embedding = model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embedding.tolist()
