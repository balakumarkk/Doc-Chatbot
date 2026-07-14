#!/usr/bin/env python3
"""
chunk.py — Chunker CLI
======================
Reads clean text files produced by the scraper and writes token-bounded chunks
to a JSONL file ready for embedding.

Usage
-----
  # Use defaults from scraper_config.yaml
  python chunk.py

  # Override specific settings
  python chunk.py --input-dir scraped_docs/clean_text \\
                  --output    scraped_docs/chunks.jsonl \\
                  --chunk-size 512 --overlap 64

  # Overwrite existing chunks.jsonl instead of appending
  python chunk.py --overwrite

  # Dry-run: count chunks without writing
  python chunk.py --dry-run
"""

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="chunk",
        description="LangChain-based markdown chunker for RAG pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--input-dir",
        metavar="PATH",
        default=None,
        help="Directory containing scraped .md / .txt files "
             "(default: chunker.input_dir from scraper_config.yaml).",
    )
    p.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Output JSONL file path "
             "(default: chunker.output_file from scraper_config.yaml).",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Max tokens per chunk (default: 512).",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Overlap tokens between consecutive chunks (default: 64).",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Discard chunks shorter than this (default: 20).",
    )
    p.add_argument(
        "--config",
        metavar="FILE",
        default="scraper_config.yaml",
        help="Path to scraper_config.yaml (default: ./scraper_config.yaml).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite chunks.jsonl instead of appending.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Process files and print statistics without writing any output.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-file chunk counts.",
    )

    return p


# ---------------------------------------------------------------------------
# Config loader (reads chunker section of scraper_config.yaml)
# ---------------------------------------------------------------------------
def _load_chunker_cfg(config_path: str) -> dict:
    """Return chunker section from scraper_config.yaml, or empty dict."""
    if not os.path.exists(config_path):
        return {}
    try:
        import yaml  # pyyaml
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data.get("chunker", {})
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    from chunker import split_markdown, write_chunks_jsonl, url_for_file

    # --- Resolve config values (CLI > YAML > hard-coded defaults) ---
    cfg = _load_chunker_cfg(args.config)

    input_dir   = args.input_dir  or cfg.get("input_dir",   "scraped_docs/clean_text")
    output_file = args.output     or cfg.get("output_file", "scraped_docs/chunks.jsonl")
    chunk_size  = args.chunk_size or cfg.get("chunk_size",  512)
    overlap     = args.overlap    or cfg.get("chunk_overlap", 64)
    min_tokens  = args.min_tokens or cfg.get("min_chunk_tokens", 20)

    input_dir   = os.path.abspath(input_dir)
    output_file = os.path.abspath(output_file)

    # Manifest lives one level above clean_text/
    manifest_path = os.path.join(os.path.dirname(input_dir), "manifest.json")

    # --- Discover files ---
    exts = {".md", ".txt"}
    files = sorted(
        p for p in Path(input_dir).rglob("*") if p.suffix in exts
    )

    if not files:
        print(f"[WARN] No .md / .txt files found in: {input_dir}", file=sys.stderr)
        sys.exit(0)

    print(f"Chunker starting")
    print(f"  Input dir  : {input_dir}")
    print(f"  Files found: {len(files)}")
    print(f"  Chunk size : {chunk_size} tokens  |  Overlap: {overlap}  |  Min: {min_tokens}")
    print(f"  Output     : {output_file}" + (" [dry-run]" if args.dry_run else ""))
    print("-" * 60)

    write_mode = "w" if args.overwrite else "a"

    total_files   = 0
    total_chunks  = 0
    skipped_files = 0

    for fpath in files:
        rel_path = str(fpath.relative_to(Path(input_dir).parent)).replace("\\", "/")
        source_url = url_for_file(rel_path, manifest_path)

        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"  [SKIP] Cannot read {fpath.name}: {exc}", file=sys.stderr)
            skipped_files += 1
            continue

        chunks = split_markdown(
            text=text,
            source_file=rel_path,
            source_url=source_url,
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            min_chunk_tokens=min_tokens,
        )

        if args.verbose:
            print(f"  {fpath.name:<60} -> {len(chunks):>4} chunks")

        if not args.dry_run and chunks:
            write_chunks_jsonl(chunks, output_file, mode=write_mode)
            # After first file in append mode keep appending
            write_mode = "a"

        total_files  += 1
        total_chunks += len(chunks)

    # --- Summary ---
    print("-" * 60)
    print(f"Done.")
    print(f"  Files processed : {total_files}  (skipped: {skipped_files})")
    print(f"  Total chunks    : {total_chunks}")
    if not args.dry_run:
        print(f"  Output file     : {output_file}")
    else:
        print("  [dry-run] No file written.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()
    run(args)
