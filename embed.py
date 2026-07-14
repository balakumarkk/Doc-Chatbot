#!/usr/bin/env python3
"""
embed.py — Embedder CLI
=======================
Reads chunks.jsonl, encodes each chunk with BAAI/bge-small-en-v1.5,
and upserts the vectors into a local Chroma collection.

Usage
-----
  # Default: reads scraper_config.yaml for all settings
  python embed.py

  # Dry run: load model + encode first batch, do not write to DB
  python embed.py --dry-run

  # Custom paths
  python embed.py --input scraped_docs/chunks.jsonl --db-path vector_db/

  # Smaller batches (useful on low-memory machines)
  python embed.py --batch-size 32
"""

import argparse
import json
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="embed",
        description="Embed chunks.jsonl into a local Chroma vector DB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--input",
        metavar="FILE",
        default=None,
        help="Path to chunks.jsonl (default: from scraper_config.yaml).",
    )
    p.add_argument(
        "--db-path",
        metavar="DIR",
        default=None,
        help="Chroma persistent store directory (default: vector_db/).",
    )
    p.add_argument(
        "--collection",
        metavar="NAME",
        default=None,
        help="Chroma collection name (default: docs).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="INT",
        help="Chunks per encoding batch (default: 64).",
    )
    p.add_argument(
        "--model",
        metavar="NAME",
        default=None,
        help="sentence-transformers model name (default: BAAI/bge-small-en-v1.5).",
    )
    p.add_argument(
        "--config",
        metavar="FILE",
        default="scraper_config.yaml",
        help="Path to scraper_config.yaml (default: ./scraper_config.yaml).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Encode the first batch only. Do not write to the vector DB.",
    )
    return p


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_embedder_cfg(config_path: str) -> dict:
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("embedder", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Chunk reader
# ---------------------------------------------------------------------------
def _read_chunks(input_file: str) -> list[dict]:
    chunks = []
    with open(input_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    from embedder.model import load_model, encode_passages
    from embedder.store import get_collection, upsert_chunks

    # --- Resolve settings (CLI > YAML > defaults) ---
    cfg = _load_embedder_cfg(args.config)

    input_file      = args.input      or cfg.get("input_file",  "scraped_docs/chunks.jsonl")
    db_path         = args.db_path    or cfg.get("db_path",     "vector_db")
    collection_name = args.collection or cfg.get("collection",  "docs")
    batch_size      = args.batch_size or cfg.get("batch_size",  64)
    model_name      = args.model      or cfg.get("model",       "BAAI/bge-small-en-v1.5")

    input_file = os.path.abspath(input_file)
    db_path    = os.path.abspath(db_path)

    # --- Validate input ---
    if not os.path.exists(input_file):
        print(f"[ERROR] chunks.jsonl not found: {input_file}", file=sys.stderr)
        print("        Run `python chunk.py` first to generate it.", file=sys.stderr)
        sys.exit(1)

    # --- Load chunks ---
    print(f"Reading chunks from: {input_file}")
    chunks = _read_chunks(input_file)
    print(f"Chunks loaded      : {len(chunks)}")

    if not chunks:
        print("[WARN] No chunks found. Nothing to embed.", file=sys.stderr)
        sys.exit(0)

    # --- Load model ---
    model = load_model(model_name)

    # --- Dry run: encode first batch only ---
    if args.dry_run:
        print(f"\n[DRY RUN] Encoding first batch of {min(batch_size, len(chunks))} chunks...")
        sample = chunks[:batch_size]
        vecs = encode_passages([c["text"] for c in sample], model=model, batch_size=batch_size, show_progress=False)
        print(f"[DRY RUN] OK — got {len(vecs)} vectors, dim={len(vecs[0])}")
        print("[DRY RUN] No data written to vector DB.")
        return

    # --- Full embed + upsert ---
    print(f"\nEmbedder starting")
    print(f"  Model      : {model_name}  (dim=384)")
    print(f"  Batch size : {batch_size}")
    print(f"  DB path    : {db_path}")
    print(f"  Collection : {collection_name}")
    print("-" * 60)

    collection = get_collection(db_path, collection_name)

    total_upserted = 0
    n_batches = (len(chunks) + batch_size - 1) // batch_size

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(f"  Batch {batch_num}/{n_batches}  ({len(batch)} chunks) — encoding...", end="", flush=True)

        embeddings = encode_passages(
            [c["text"] for c in batch],
            model=model,
            batch_size=batch_size,
            show_progress=False,
        )

        upserted = upsert_chunks(batch, embeddings, collection)
        total_upserted += upserted
        print(f" upserted {upserted}  (total: {total_upserted})")

    # --- Summary ---
    print("-" * 60)
    print(f"Done.")
    print(f"  Total upserted : {total_upserted}")
    print(f"  Collection size: {collection.count()} documents")
    print(f"  Vector DB      : {db_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    run(args)
