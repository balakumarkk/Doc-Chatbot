#!/usr/bin/env python3
"""
scrape.py — Web Scraper CLI
===========================
Uses trafilatura + requests to extract clean content from websites.

Modes:
  --urls        Scrape a fixed list of URLs passed on the command line
  --url-file    Scrape URLs read from a text file (one per line)
  --seed        Crawl from a seed URL up to --depth levels deep

Usage examples:
  python scrape.py --urls https://docs.python.org/3/library/pathlib.html
  python scrape.py --url-file my_urls.txt --format md --delay 1.5
  python scrape.py --seed https://docs.python.org/3/ --depth 2 --same-domain
"""

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scrape",
        description="Trafilatura-based web content scraper for RAG pipelines.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input mode (mutually exclusive) ---
    input_group = p.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--urls",
        nargs="+",
        metavar="URL",
        help="One or more URLs to scrape.",
    )
    input_group.add_argument(
        "--url-file",
        metavar="FILE",
        help="Path to a text file with one URL per line.",
    )
    input_group.add_argument(
        "--seed",
        metavar="URL",
        help="Seed URL to start a BFS crawl from.",
    )

    # --- Crawl options ---
    p.add_argument(
        "--depth",
        type=int,
        default=1,
        metavar="INT",
        help="BFS crawl depth when using --seed (default: 1).",
    )
    p.add_argument(
        "--same-domain",
        action="store_true",
        default=True,
        help="Restrict crawl to the seed domain (default: True).",
    )
    p.add_argument(
        "--no-same-domain",
        dest="same_domain",
        action="store_false",
        help="Allow crawling across domains.",
    )

    # --- Output options ---
    p.add_argument(
        "--output-dir",
        default="scraped_docs",
        metavar="PATH",
        help="Root output directory (default: ./scraped_docs).",
    )
    p.add_argument(
        "--format",
        choices=["md", "txt"],
        default="md",
        help="Output format for clean text files (default: md).",
    )

    # --- Fetch behaviour ---
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        metavar="SECS",
        help="Seconds to wait between requests (default: 1.0).",
    )
    p.add_argument(
        "--timeout",
        type=int,
        default=20,
        metavar="SECS",
        help="HTTP request timeout in seconds (default: 20).",
    )
    p.add_argument(
        "--no-robots",
        action="store_true",
        default=False,
        help="Skip robots.txt checks (use responsibly!).",
    )
    p.add_argument(
        "--no-raw",
        action="store_true",
        default=False,
        help="Do NOT save raw HTML snapshots (raw HTML is saved by default).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip URLs already present in manifest.json (default: True).",
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Re-scrape all URLs, even if already in manifest.json.",
    )

    # --- Config file ---
    p.add_argument(
        "--config",
        metavar="FILE",
        default="scraper_config.yaml",
        help="Path to YAML config file (default: scraper_config.yaml).",
    )

    return p


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> None:
    # --- Load config (YAML → CLI overrides on top) ---
    from scraper.config import load_config
    cfg = load_config(args.config)
    cfg.apply_cli_overrides(args)

    # Import here so logger is available after CLI parse
    from scraper.logger import setup_logger, get_logger
    from scraper.fetcher import fetch
    from scraper.extractor import extract_content
    from scraper.crawler import crawl, iterate_urls
    from scraper.storage import (
        save_content,
        update_manifest,
        get_scraped_urls,
        ManifestEntry,
        now_iso,
    )

    # --- Resolve paths ---
    output_dir = os.path.abspath(cfg.output.dir)
    clean_text_dir = os.path.join(output_dir, "clean_text")
    raw_html_dir = os.path.join(output_dir, "raw_html")
    manifest_path = os.path.join(output_dir, "manifest.json")

    os.makedirs(clean_text_dir, exist_ok=True)
    if cfg.output.save_raw_html:
        os.makedirs(raw_html_dir, exist_ok=True)

    logger = setup_logger(
        log_dir=output_dir,
        console_level=cfg.logging.console_level,
        file_level=cfg.logging.file_level,
    )
    log = get_logger()
    log.info("Config loaded from: %s", cfg._source)

    # --- Resume: load already-scraped URLs ---
    already_scraped: set[str] = set()
    if cfg.resume.enabled:
        already_scraped = get_scraped_urls(manifest_path)
        if already_scraped:
            log.info("Resume mode: %d URLs already in manifest — will be skipped.", len(already_scraped))

    # --- Build a fetch wrapper using config values ---
    def _fetch(url: str):
        return fetch(
            url,
            timeout=cfg.fetch.timeout,
            delay=cfg.fetch.delay,
            respect_robots=cfg.fetch.respect_robots,
            save_raw=cfg.output.save_raw_html,
            raw_html_dir=raw_html_dir if cfg.output.save_raw_html else None,
            custom_ua=cfg.fetch.user_agent or None,
        )

    def _fetch_html(url: str):
        result = _fetch(url)
        return result.html if result.ok else None

    # --- Determine URL source ---
    if args.urls:
        url_list = args.urls
        source_desc = f"{len(url_list)} URL(s) from command line"
    elif args.url_file:
        url_list = _load_url_file(args.url_file)
        source_desc = f"{len(url_list)} URL(s) from {args.url_file}"
    else:
        url_list = None

    if url_list is not None:
        if cfg.crawl.depth > 0:
            # Crawl mode: each URL is a seed, follow links up to depth
            source_desc += f" [crawl depth={cfg.crawl.depth}]"
            url_html_iter = _multi_seed_crawl(url_list, cfg, _fetch_html, crawl)
        else:
            # Flat fetch: just grab each URL as a single page
            url_html_iter = iterate_urls(url_list, fetch_fn=_fetch_html)
    else:  # --seed
        source_desc = (
            f"crawl from {args.seed} "
            f"(depth={cfg.crawl.depth}, same_domain={cfg.crawl.same_domain_only})"
        )
        url_html_iter = crawl(
            args.seed,
            depth=cfg.crawl.depth,
            same_domain_only=cfg.crawl.same_domain_only,
            path_prefix_depth=cfg.crawl.path_prefix_depth,
            exclude_patterns=cfg.crawl.exclude_patterns,
            fetch_fn=_fetch_html,
        )

    log.info("=" * 70)
    log.info("Scraper starting — %s", source_desc)
    log.info("Output dir : %s", output_dir)
    log.info("Format     : .%s", cfg.output.format)
    log.info("Raw HTML   : %s", raw_html_dir if cfg.output.save_raw_html else "disabled")
    log.info("Delay      : %.1fs  |  Timeout: %ds  |  Robots: %s",
             cfg.fetch.delay, cfg.fetch.timeout,
             "respect" if cfg.fetch.respect_robots else "SKIP")
    log.info("=" * 70)

    # --- Main loop ---
    total = skipped = success = failed = 0

    for url, html in url_html_iter:
        total += 1

        if cfg.resume.enabled and url in already_scraped:
            log.info("SKIP (already scraped): %s", url)
            skipped += 1
            continue

        # Handle empty HTML (fetch failed — robots, timeout, HTTP error)
        if not html:
            failed += 1
            log.warning("FAILED (fetch): %s", url)
            entry = ManifestEntry(
                url=url,
                title="",
                filepath="",
                scrape_date=now_iso(),
                word_count=0,
                status="failed",
                error="Fetch returned no HTML (blocked by robots.txt or HTTP error)",
            )
            update_manifest(entry, manifest_path)
            continue

        result = extract_content(html, url)

        if result is None:
            failed += 1
            entry = ManifestEntry(
                url=url,
                title="",
                filepath="",
                scrape_date=now_iso(),
                word_count=0,
                status="failed",
                error="Extraction returned no content",
            )
            update_manifest(entry, manifest_path)
            continue

        filepath = save_content(result, clean_text_dir, fmt=cfg.output.format)
        rel_path = os.path.relpath(filepath, output_dir)

        entry = ManifestEntry(
            url=url,
            title=result.title,
            filepath=rel_path,
            scrape_date=now_iso(),
            word_count=result.word_count,
            status="success",
        )
        update_manifest(entry, manifest_path)
        success += 1

    # --- Summary ---
    log.info("=" * 70)
    log.info(
        "Done. Total: %d | Success: %d | Failed: %d | Skipped: %d",
        total, success, failed, skipped,
    )
    log.info("Manifest → %s", manifest_path)
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_url_file(path: str) -> list[str]:
    if not os.path.exists(path):
        print(f"[ERROR] URL file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _multi_seed_crawl(urls, cfg, fetch_fn, crawl_fn):
    """
    Crawl each URL in *urls* as an independent seed up to ``cfg.crawl.depth``.

    Yields ``(url, html)`` tuples, deduplicating across seeds so the same
    page is never fetched twice even if multiple seeds link to it.
    """
    from scraper.logger import get_logger
    log = get_logger()

    global_visited: set[str] = set()

    for seed_url in urls:
        log.info("--- Seed: %s (depth=%d) ---", seed_url, cfg.crawl.depth)
        for url, html in crawl_fn(
            seed_url,
            depth=cfg.crawl.depth,
            same_domain_only=cfg.crawl.same_domain_only,
            path_prefix_depth=cfg.crawl.path_prefix_depth,
            exclude_patterns=cfg.crawl.exclude_patterns,
            fetch_fn=fetch_fn,
        ):
            if url in global_visited:
                continue
            global_visited.add(url)
            yield url, html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run(args)
