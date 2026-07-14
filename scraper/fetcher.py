"""
scraper/fetcher.py
------------------
HTTP fetching with User-Agent rotation, retry logic, and optional raw-HTML saving.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.robotparser import RobotFileParser
from urllib.parse import urlparse, urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .logger import get_logger
from .storage import url_to_slug

log = get_logger()

# ---------------------------------------------------------------------------
# User-Agent pool — realistic desktop browser strings
# ---------------------------------------------------------------------------
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]

# Cache of robots.txt parsers per domain
_robots_cache: dict[str, RobotFileParser] = {}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    url: str
    final_url: str
    html: str
    status_code: int
    ok: bool
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_session(retries: int = 3) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=retries,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _random_ua() -> str:
    return random.choice(_USER_AGENTS)


def _get_robots_parser(base_url: str) -> Optional[RobotFileParser]:
    """Fetch and cache the robots.txt for a given base URL."""
    parsed = urlparse(base_url)
    domain_key = f"{parsed.scheme}://{parsed.netloc}"
    if domain_key in _robots_cache:
        return _robots_cache[domain_key]

    robots_url = urljoin(domain_key, "/robots.txt")
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        _robots_cache[domain_key] = rp
        log.debug("Loaded robots.txt from %s", robots_url)
    except Exception as exc:
        log.debug("Could not load robots.txt from %s: %s", robots_url, exc)
        _robots_cache[domain_key] = None
        return None
    return rp


def is_allowed_by_robots(url: str, respect_robots: bool = True) -> bool:
    """Return True if scraping ``url`` is allowed by robots.txt."""
    if not respect_robots:
        return True
    rp = _get_robots_parser(url)
    if rp is None:
        return True  # no robots.txt → assume allowed
    return rp.can_fetch("*", url)


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------
def fetch(
    url: str,
    *,
    timeout: int = 20,
    retries: int = 3,
    delay: float = 0.0,
    respect_robots: bool = True,
    save_raw: bool = True,
    raw_html_dir: Optional[str] = None,
    custom_ua: Optional[str] = None,
) -> FetchResult:
    """
    Fetch a URL and return a :class:`FetchResult`.

    Args:
        url:            Target URL.
        timeout:        Request timeout in seconds.
        retries:        Number of HTTP-level retries.
        delay:          Seconds to sleep *before* the request (rate limiting).
        respect_robots: If True, check robots.txt before fetching.
        save_raw:       If True, write raw HTML to *raw_html_dir*.
        raw_html_dir:   Directory to save raw HTML files.
        custom_ua:      Override the User-Agent string (None → rotate pool).

    Returns:
        :class:`FetchResult`
    """
    if delay > 0:
        log.debug("Sleeping %.2fs before fetching %s", delay, url)
        time.sleep(delay)

    # Robots check
    if not is_allowed_by_robots(url, respect_robots):
        log.warning("robots.txt disallows: %s", url)
        return FetchResult(url=url, final_url=url, html="", status_code=0, ok=False,
                           error="Disallowed by robots.txt")

    session = _build_session(retries)
    headers = {
        "User-Agent": custom_ua if custom_ua else _random_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }

    try:
        resp = session.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        log.error("HTTP error fetching %s: %s", url, exc)
        sc = exc.response.status_code if exc.response is not None else 0
        return FetchResult(url=url, final_url=url, html="", status_code=sc,
                           ok=False, error=str(exc))
    except requests.exceptions.RequestException as exc:
        log.error("Request failed for %s: %s", url, exc)
        return FetchResult(url=url, final_url=url, html="", status_code=0, ok=False, error=str(exc))

    # Decode explicitly as UTF-8 to avoid Windows cp1252 garbling (→, §, etc.)
    encoding = resp.encoding or "utf-8"
    if encoding.lower() in ("iso-8859-1", "latin-1", "windows-1252"):
        encoding = "utf-8"  # most modern sites are UTF-8; override browser-default guess
    html = resp.content.decode(encoding, errors="replace")
    final_url = resp.url
    log.debug("Fetched %s → %d bytes", final_url, len(html))

    # Optionally save raw HTML
    if save_raw and raw_html_dir:
        _save_raw_html(html, url, raw_html_dir)

    return FetchResult(
        url=url,
        final_url=final_url,
        html=html,
        status_code=resp.status_code,
        ok=True,
    )


def _save_raw_html(html: str, url: str, raw_html_dir: str) -> None:
    os.makedirs(raw_html_dir, exist_ok=True)
    slug = url_to_slug(url)
    path = os.path.join(raw_html_dir, f"{slug}.html")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        log.debug("Saved raw HTML → %s", path)
    except OSError as exc:
        log.warning("Could not save raw HTML for %s: %s", url, exc)
