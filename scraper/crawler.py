"""
scraper/crawler.py
------------------
BFS link discovery: same-domain, depth-limited.
Uses trafilatura's focused_crawler when available, falls back to BeautifulSoup.
"""

import re
from collections import deque
from typing import Iterator, Optional
from urllib.parse import urljoin, urlparse

from .logger import get_logger

log = get_logger()


# ---------------------------------------------------------------------------
# Domain helpers
# ---------------------------------------------------------------------------
def _base_domain(url: str) -> str:
    """Return ``netloc`` (e.g. ``docs.aws.amazon.com``) from a URL."""
    return urlparse(url).netloc.lower()


def _path_scope(url: str, n_segments: int) -> str:
    """
    Return the first *n_segments* path segments of *url* as a prefix string.

    Examples::

        _path_scope("https://docs.aws.amazon.com/AmazonS3/latest/userguide/X.html", 1)
        # → "docs.aws.amazon.com/AmazonS3"

        _path_scope("https://docs.aws.amazon.com/AmazonS3/latest/userguide/X.html", 2)
        # → "docs.aws.amazon.com/AmazonS3/latest"
    """
    p = urlparse(url)
    segments = [s for s in p.path.split("/") if s]  # remove empty parts
    prefix_segs = segments[:n_segments]
    prefix = "/".join(prefix_segs)
    return f"{p.netloc.lower()}/{prefix}" if prefix else p.netloc.lower()


# Non-HTML extensions to skip during crawl
_SKIP_EXTENSIONS = {
    ".pdf", ".md", ".txt", ".zip", ".gz", ".tar",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico",
    ".mp4", ".mp3", ".woff", ".woff2", ".css", ".js",
    ".xml", ".json", ".yaml", ".yml",
}


def _normalise(url: str, base: str) -> Optional[str]:
    """
    Resolve *url* against *base* and return a canonical HTTP(S) URL,
    or None if it should be skipped.

    Skips:
    - Fragment-only links (``#anchor`` with no path change)
    - Non-HTML file extensions (.pdf, .md, images, etc.)
    - mailto: / javascript: pseudo-protocols
    """
    if not url:
        return None
    # Skip pure fragment links (same-page anchors)
    if url.startswith("#"):
        return None
    url = url.split("#")[0].rstrip("/")  # strip fragment + trailing slash
    if not url:
        return None
    if url.startswith(("mailto:", "javascript:", "tel:", "data:")):
        return None
    joined = urljoin(base, url)
    if not joined.startswith(("http://", "https://")):
        return None
    # Skip non-HTML extensions
    path = urlparse(joined).path.lower()
    ext = ""
    if "." in path.split("/")[-1]:
        ext = "." + path.split(".")[-1]
    if ext in _SKIP_EXTENSIONS:
        return None
    return joined


# ---------------------------------------------------------------------------
# Link extraction (BeautifulSoup fallback)
# ---------------------------------------------------------------------------
def _extract_links_bs4(html: str, base_url: str) -> list[str]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        log.warning("beautifulsoup4 not installed — cannot extract links from %s", base_url)
        return []

    soup = BeautifulSoup(html, "lxml")
    links = []
    for tag in soup.find_all("a", href=True):
        norm = _normalise(tag["href"], base_url)
        if norm:
            links.append(norm)
    return links


# ---------------------------------------------------------------------------
# Public crawler
# ---------------------------------------------------------------------------
def crawl(
    seed_url: str,
    depth: int,
    same_domain_only: bool = True,
    path_prefix_depth: int = 0,
    exclude_patterns: list[str] | None = None,
    fetch_fn=None,
) -> Iterator[tuple[str, str]]:
    """
    BFS crawler starting from *seed_url*.

    Args:
        seed_url:           Starting URL.
        depth:              Maximum BFS depth (0 = seed only).
        same_domain_only:   If True, only follow links on the same domain.
        path_prefix_depth:  If > 0, restrict crawl to URLs that share the first
                            N path segments with the seed URL.
                            Example: seed = ``/AmazonS3/latest/userguide/X.html``
                            with ``path_prefix_depth=1`` restricts to ``/AmazonS3/*``.
                            0 = domain-only check (no path restriction).
        exclude_patterns:   List of regex patterns; matching URLs are skipped.
        fetch_fn:           Callable ``(url) -> html_str | None``.

    Yields:
        ``(url, html)`` tuples for every reachable page within *depth*.
    """
    if fetch_fn is None:
        from .fetcher import fetch as _fetch

        def fetch_fn(u: str) -> Optional[str]:
            result = _fetch(u)
            return result.html if result.ok else None

    seed_domain = _base_domain(seed_url)
    # Compute path scope prefix (e.g. "docs.aws.amazon.com/AmazonS3")
    seed_prefix = _path_scope(seed_url, path_prefix_depth) if path_prefix_depth > 0 else None
    if seed_prefix:
        log.info("Path scope restricted to: %s", seed_prefix)
    visited: set[str] = set()
    queue: deque[tuple[str, int]] = deque([(seed_url, 0)])
    _excludes = [re.compile(p, re.IGNORECASE) for p in (exclude_patterns or [])]

    def _canonical(u: str) -> str:
        """Strip query string for dedup (keep path only)."""
        p = urlparse(u)
        return f"{p.scheme}://{p.netloc}{p.path}".rstrip("/")

    while queue:
        url, current_depth = queue.popleft()

        canonical = _canonical(url)
        if canonical in visited:
            continue
        visited.add(canonical)

        log.info("Crawling [depth=%d] %s", current_depth, url)
        html = fetch_fn(url)
        if not html:
            log.warning("No HTML returned for %s — skipping.", url)
            continue

        yield url, html

        if current_depth >= depth:
            continue  # don't extract links beyond max depth

        links = _extract_links_bs4(html, url)
        log.debug("Found %d links on %s", len(links), url)

        for link in links:
            if _canonical(link) in visited:
                continue
            # --- Scope checks ---
            if same_domain_only and _base_domain(link) != seed_domain:
                log.debug("Off-domain link skipped: %s", link)
                continue
            if seed_prefix:
                link_prefix = _path_scope(link, path_prefix_depth)
                if not link_prefix.startswith(seed_prefix):
                    log.debug("Out-of-scope path skipped: %s", link)
                    continue
            if any(rx.search(link) for rx in _excludes):
                log.debug("Excluded by pattern: %s", link)
                continue
            queue.append((link, current_depth + 1))


# ---------------------------------------------------------------------------
# Simple URL-list "crawler" (no link following)
# ---------------------------------------------------------------------------
def iterate_urls(
    urls: list[str],
    fetch_fn=None,
) -> Iterator[tuple[str, str]]:
    """
    Yield ``(url, html)`` for each URL in *urls* without any link following.
    Used in URL-list mode.
    """
    if fetch_fn is None:
        from .fetcher import fetch as _fetch

        def fetch_fn(u: str) -> Optional[str]:
            result = _fetch(u)
            return result.html if result.ok else None

    for url in urls:
        log.info("Fetching %s", url)
        html = fetch_fn(url)
        if html:
            yield url, html
        else:
            # yield empty html so main loop can log to manifest
            yield url, ""
