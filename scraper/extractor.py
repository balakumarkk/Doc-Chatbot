"""
scraper/extractor.py
--------------------
Wrap trafilatura to extract clean main content from raw HTML.
Post-cleans output to remove nav remnants, excessive blank lines, and footer noise.
"""

import re
from typing import Optional

import trafilatura

from .logger import get_logger
from .storage import ExtractResult, url_to_slug

log = get_logger()

# ---------------------------------------------------------------------------
# Regex patterns for post-extraction cleaning
# ---------------------------------------------------------------------------
# Lines that look like navigation / footer artifacts (very short + no sentence chars)
_NAV_LINE_RE = re.compile(
    r"^("
    r"(home|back|next|previous|skip|menu|search|login|sign\s*in|sign\s*up|"
    r"copyright|all\s+rights\s+reserved|privacy\s+policy|terms\s+of\s+(use|service)|"
    r"cookie\s+policy|©|\|\s*\|)"
    r")$",
    re.IGNORECASE,
)
# More than 2 consecutive blank lines → collapse
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
# Trailing whitespace on each line
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Public extraction function
# ---------------------------------------------------------------------------
def extract_content(html: str, url: str) -> Optional[ExtractResult]:
    """
    Extract main content from *html* using trafilatura.

    Strategy:
    1. Strict pass: ``favor_precision=True`` (fewer false positives).
    2. Recall pass: ``favor_recall=True`` (catches sparser pages).

    Args:
        html: Raw HTML string.
        url:  Source URL (used for slug generation and logging).

    Returns:
        :class:`ExtractResult` on success, ``None`` on failure.
    """
    # --- Pass 1: strict ---
    metadata = trafilatura.extract_metadata(html, default_url=url)
    text = trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
        favor_precision=True,
        output_format="txt",
    )

    # --- Pass 2: recall fallback ---
    if not text:
        log.debug("Strict extraction failed, retrying with favor_recall for %s", url)
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,
            output_format="txt",
        )

    if not text:
        log.warning("No content extracted from %s", url)
        return None

    # --- Post-cleaning ---
    text = _clean(text)
    if not text.strip():
        log.warning("Content was empty after cleaning for %s", url)
        return None

    title = _get_title(metadata, url)
    word_count = len(text.split())
    slug = url_to_slug(url)
    date = getattr(metadata, "date", None) if metadata else None

    log.info("Extracted %-60s | %-50s | %d words", url[:60], title[:50], word_count)

    return ExtractResult(
        url=url,
        final_url=url,
        title=title,
        text=text,
        date=date,
        word_count=word_count,
        slug=slug,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _get_title(metadata, url: str) -> str:
    """Extract title from metadata, falling back to domain/path."""
    if metadata and metadata.title:
        title = metadata.title.strip()
        # Strip Unicode replacement characters and isolated control chars
        title = re.sub(r"[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+", "", title)
        # Collapse multiple spaces left by stripped chars
        title = re.sub(r"  +", " ", title).strip()
        if title:
            return title
    # Fallback: use the last path segment of the URL
    parts = [p for p in url.split("/") if p and not p.startswith("http")]
    return parts[-1].replace("-", " ").replace("_", " ").title() if parts else "Untitled"


def _clean(text: str) -> str:
    """
    Post-process extracted text:
    - Remove nav / footer artifact lines
    - Strip trailing whitespace per line
    - Collapse excessive blank lines
    """
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Skip very short lines that match known nav patterns
        if stripped and len(stripped) < 60 and _NAV_LINE_RE.match(stripped):
            continue
        # Strip trailing whitespace
        cleaned.append(line.rstrip())

    result = "\n".join(cleaned)
    # Collapse 3+ consecutive blank lines to 2
    result = _MULTI_BLANK_RE.sub("\n\n", result)
    return result.strip()
