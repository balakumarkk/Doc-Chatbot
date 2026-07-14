"""
scraper — Trafilatura-based web content extraction module.

Sub-modules:
    fetcher   : HTTP fetching with User-Agent rotation & retries
    extractor : trafilatura content extraction + post-cleaning
    crawler   : BFS link discovery (same-domain, depth-limited)
    storage   : file saving, slug generation, manifest management
    logger    : logging setup (console + file)
"""

__version__ = "1.0.0"
