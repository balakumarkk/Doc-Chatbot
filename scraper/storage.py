"""
scraper/storage.py
------------------
Slug generation, file saving (.md / .txt), and manifest management.
"""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from .logger import get_logger

log = get_logger()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class ExtractResult:
    """Holds the cleaned output of one scraped page."""
    url: str
    final_url: str
    title: str
    text: str
    date: Optional[str] = None
    word_count: int = 0
    slug: str = ""


@dataclass
class ManifestEntry:
    url: str
    title: str
    filepath: str
    scrape_date: str
    word_count: int
    status: str = "success"
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "filepath": self.filepath,
            "scrape_date": self.scrape_date,
            "word_count": self.word_count,
            "status": self.status,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Slug generation
# ---------------------------------------------------------------------------
def url_to_slug(url: str, max_len: int = 80) -> str:
    """
    Convert a URL to a filesystem-safe slug.

    Examples:
        https://docs.aws.amazon.com/s3/intro/  → ``aws_amazon_com_s3_intro``
        https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API → ``mozilla_org_docs_web_api_fetch_api``
    """
    # Strip scheme
    slug = re.sub(r"^https?://", "", url)
    # Remove www.
    slug = re.sub(r"^www\.", "", slug)
    # Remove trailing slash + query + fragment
    slug = re.sub(r"[?#].*$", "", slug).rstrip("/")
    # Replace non-alphanumeric characters with underscores
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", slug)
    # Remove leading/trailing underscores
    slug = slug.strip("_")
    # Lowercase and truncate
    slug = slug.lower()[:max_len]
    return slug or "unnamed"


# ---------------------------------------------------------------------------
# File saving
# ---------------------------------------------------------------------------
def save_content(result: "ExtractResult", output_dir: str, fmt: str = "md") -> str:
    """
    Save extracted content to disk.

    Args:
        result:     :class:`ExtractResult` with title + text.
        output_dir: Path to the ``clean_text`` directory.
        fmt:        ``"md"`` or ``"txt"``.

    Returns:
        Absolute path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)

    ext = fmt.lstrip(".")
    filename = f"{result.slug}.{ext}"
    filepath = os.path.join(output_dir, filename)

    if ext == "md":
        content = f"# {result.title}\n\n{result.text}\n"
    else:
        content = f"{result.title}\n{'=' * len(result.title)}\n\n{result.text}\n"

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    log.debug("Saved content → %s", filepath)
    return filepath


# ---------------------------------------------------------------------------
# Manifest management
# ---------------------------------------------------------------------------
def load_manifest(manifest_path: str) -> list[dict]:
    """Load the manifest JSON array from disk (returns [] if file missing)."""
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Could not load manifest (%s): %s — starting fresh.", manifest_path, exc)
        return []


def get_scraped_urls(manifest_path: str) -> set[str]:
    """Return the set of URLs already present in manifest.json."""
    return {entry["url"] for entry in load_manifest(manifest_path) if "url" in entry}


def update_manifest(entry: ManifestEntry, manifest_path: str) -> None:
    """
    Append *entry* to the manifest JSON file atomically.

    The file is a JSON array written in pretty-printed format.
    """
    entries = load_manifest(manifest_path)
    # Update in-place if URL already exists (e.g. retry scenario)
    existing_urls = [e.get("url") for e in entries]
    if entry.url in existing_urls:
        idx = existing_urls.index(entry.url)
        entries[idx] = entry.to_dict()
    else:
        entries.append(entry.to_dict())

    tmp_path = manifest_path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, manifest_path)  # atomic rename
        log.debug("Manifest updated → %s (%d entries)", manifest_path, len(entries))
    except OSError as exc:
        log.error("Failed to write manifest: %s", exc)


def now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds")
