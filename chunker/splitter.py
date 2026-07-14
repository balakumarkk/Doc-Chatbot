"""
chunker/splitter.py
-------------------
Two-pass LangChain chunking pipeline:

  Pass 1 — MarkdownHeaderTextSplitter
      Splits on # / ## / ### headings and attaches heading metadata to every
      child document so the RAG retriever knows which section a chunk belongs to.

  Pass 2 — RecursiveCharacterTextSplitter (tiktoken-aware)
      Splits oversized sections into token-bounded chunks with overlap,
      using cl100k_base as the tokeniser (reasonable approximation for both
      OpenAI and Anthropic models).

Output is a list of ChunkRecord dataclass instances — one per final chunk.
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# LangChain imports (langchain-text-splitters package)
# ---------------------------------------------------------------------------
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# Tiktoken — token counting (cl100k_base ≈ GPT-4 / text-embedding-3-*)
# ---------------------------------------------------------------------------
try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))

except ImportError:  # graceful fallback: rough 4-chars-per-token estimate
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# ChunkRecord
# ---------------------------------------------------------------------------
@dataclass
class ChunkRecord:
    """One chunk of text ready to be embedded and stored."""

    chunk_id: str          # sha256 of (source_file + chunk_index)
    source_file: str       # relative path of the original .md / .txt file
    source_url: str        # original web URL (empty string if unknown)
    chunk_index: int       # 0-based position within the source file
    text: str              # chunk content (clean, stripped)
    token_count: int       # approximate tokens (cl100k_base)
    headings: dict         # e.g. {"h1": "Overview", "h2": "Authentication"}
    extra: dict = field(default_factory=dict)  # reserved for future metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def split_markdown(
    text: str,
    source_file: str = "",
    source_url: str = "",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_tokens: int = 20,
) -> list[ChunkRecord]:
    """
    Split *text* into token-bounded chunks with heading metadata attached.

    Parameters
    ----------
    text            : Raw markdown / plain text content.
    source_file     : Relative path to the original file (stored in metadata).
    source_url      : Original URL the page was scraped from.
    chunk_size      : Maximum tokens per chunk.
    chunk_overlap   : Overlap tokens carried from the previous chunk.
    min_chunk_tokens: Chunks shorter than this are discarded (noise filter).

    Returns
    -------
    List of ChunkRecord objects in document order.
    """

    # ------------------------------------------------------------------
    # Pass 1 — split on Markdown headings
    # ------------------------------------------------------------------
    headers_to_split_on = [
        ("#",   "h1"),
        ("##",  "h2"),
        ("###", "h3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,   # keep headers in chunk text for context
    )

    # If the text has no markdown headings, wrap it so MarkdownHeaderTextSplitter
    # still returns a single document rather than an empty list.
    sections = md_splitter.split_text(text)
    if not sections:
        from langchain_core.documents import Document  # lazy import
        sections = [Document(page_content=text, metadata={})]

    # ------------------------------------------------------------------
    # Pass 2 — token-aware recursive splitter
    # ------------------------------------------------------------------
    char_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[ChunkRecord] = []
    chunk_index = 0

    for section in sections:
        heading_meta = {
            k: v for k, v in section.metadata.items()
            if k in ("h1", "h2", "h3")
        }
        section_text = section.page_content.strip()

        if not section_text:
            continue

        # Split oversized sections further
        sub_docs = char_splitter.create_documents([section_text])

        for sub in sub_docs:
            sub_text = sub.page_content.strip()
            if not sub_text:
                continue

            tok_count = _count_tokens(sub_text)
            if tok_count < min_chunk_tokens:
                continue  # drop very short / noisy chunks

            chunk_id = _make_chunk_id(source_file, chunk_index)

            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    source_file=source_file,
                    source_url=source_url,
                    chunk_index=chunk_index,
                    text=sub_text,
                    token_count=tok_count,
                    headings=heading_meta,
                )
            )
            chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _make_chunk_id(source_file: str, index: int) -> str:
    """Deterministic sha256 ID for a chunk — stable across re-runs."""
    raw = f"{source_file}::{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
