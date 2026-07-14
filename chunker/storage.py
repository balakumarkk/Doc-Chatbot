"""
chunker/storage.py
------------------
Writes ChunkRecord objects to a .jsonl file (one JSON object per line).

Functions
---------
write_chunks_jsonl  — append / overwrite chunks.jsonl
url_for_file        — look up original URL from the scraper manifest
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from .splitter import ChunkRecord


# ---------------------------------------------------------------------------
# manifest helper
# ---------------------------------------------------------------------------
def url_for_file(relative_path: str, manifest_path: str) -> str:
    """
    Return the original URL for *relative_path* by looking it up in the
    scraper's manifest.json.

    Parameters
    ----------
    relative_path : Path stored in the manifest entry, e.g.
                    "clean_text/some_page.md"
    manifest_path : Absolute path to manifest.json

    Returns
    -------
    URL string, or empty string if not found / manifest missing.
    """
    if not os.path.exists(manifest_path):
        return ""

    with open(manifest_path, encoding="utf-8") as f:
        try:
            entries: list[dict] = json.load(f)
        except json.JSONDecodeError:
            return ""

    # Normalise slashes for comparison
    target = relative_path.replace("\\", "/")

    for entry in entries:
        fp = entry.get("filepath", "").replace("\\", "/")
        if fp == target or fp.endswith("/" + target) or target.endswith(fp):
            return entry.get("url", "")

    return ""


# ---------------------------------------------------------------------------
# write_chunks_jsonl
# ---------------------------------------------------------------------------
def write_chunks_jsonl(
    chunks: Iterable[ChunkRecord],
    output_path: str,
    mode: str = "a",
) -> int:
    """
    Write *chunks* to *output_path* in JSON Lines format.

    Each line is a JSON object with all ChunkRecord fields.

    Parameters
    ----------
    chunks      : Iterable of ChunkRecord instances.
    output_path : Destination file path (created if absent).
    mode        : ``'a'`` to append (default), ``'w'`` to overwrite.

    Returns
    -------
    Number of chunks written.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    written = 0
    with open(output_path, mode=mode, encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "chunk_id":    chunk.chunk_id,
                "source_file": chunk.source_file,
                "source_url":  chunk.source_url,
                "chunk_index": chunk.chunk_index,
                "text":        chunk.text,
                "token_count": chunk.token_count,
                "headings":    chunk.headings,
                **chunk.extra,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return written
