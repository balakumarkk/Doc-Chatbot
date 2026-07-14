"""
scraper/config.py
-----------------
Load, validate, and expose scraper configuration.

Priority (highest → lowest):
  1. CLI arguments (argparse Namespace)
  2. scraper_config.yaml  (project root)
  3. Built-in defaults    (defined in this module)

Usage::

    from scraper.config import load_config, ScraperConfig
    cfg = load_config()            # reads scraper_config.yaml if present
    cfg = load_config("my.yaml")   # explicit path
    print(cfg.fetch.delay)         # 1.0
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Default config path (relative to main.py / project root)
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_FILE = "scraper_config.yaml"


# ---------------------------------------------------------------------------
# Dataclasses — one per YAML section
# ---------------------------------------------------------------------------
@dataclass
class OutputConfig:
    dir: str = "scraped_docs"
    format: str = "md"                  # "md" | "txt"
    save_raw_html: bool = True


@dataclass
class FetchConfig:
    delay: float = 1.0
    timeout: int = 20
    retries: int = 3
    respect_robots: bool = True
    user_agent: str = ""                # empty → rotate built-in pool


@dataclass
class CrawlConfig:
    depth: int = 1
    same_domain_only: bool = True
    path_prefix_depth: int = 0       # 0 = domain only; 1 = /FirstSegment; 2 = /First/Second
    exclude_patterns: list[str] = field(default_factory=list)


@dataclass
class ResumeConfig:
    enabled: bool = True


@dataclass
class LoggingConfig:
    console_level: str = "INFO"
    file_level: str = "DEBUG"


@dataclass
class ScraperConfig:
    """Top-level config object returned by :func:`load_config`."""
    output: OutputConfig = field(default_factory=OutputConfig)
    fetch: FetchConfig = field(default_factory=FetchConfig)
    crawl: CrawlConfig = field(default_factory=CrawlConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Path the config was loaded from (informational)
    _source: str = field(default="built-in defaults", repr=False)

    def apply_cli_overrides(self, args: argparse.Namespace) -> "ScraperConfig":
        """
        Overlay CLI argument values on top of the loaded config.
        Only overrides a field when the CLI argument was explicitly set
        (i.e. differs from its argparse default).

        Returns self for chaining.
        """
        # Output
        if _cli_set(args, "output_dir"):
            self.output.dir = args.output_dir
        if _cli_set(args, "format"):
            self.output.format = args.format
        if _cli_set(args, "no_raw"):
            self.output.save_raw_html = not args.no_raw

        # Fetch
        if _cli_set(args, "delay"):
            self.fetch.delay = args.delay
        if _cli_set(args, "timeout"):
            self.fetch.timeout = args.timeout
        if _cli_set(args, "no_robots"):
            self.fetch.respect_robots = not args.no_robots

        # Crawl
        if _cli_set(args, "depth"):
            self.crawl.depth = args.depth
        if _cli_set(args, "same_domain"):
            self.crawl.same_domain_only = args.same_domain

        # Resume
        if _cli_set(args, "resume"):
            self.resume.enabled = args.resume

        return self


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> ScraperConfig:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file.
              Defaults to ``scraper_config.yaml`` in the current directory.
              If the file does not exist, built-in defaults are used silently.

    Returns:
        Populated :class:`ScraperConfig` instance.
    """
    config_path = path or DEFAULT_CONFIG_FILE

    if not os.path.exists(config_path):
        return ScraperConfig(_source="built-in defaults (no config file found)")

    try:
        import yaml
    except ImportError:
        print(
            f"[scraper/config] pyyaml not installed — using built-in defaults. "
            f"Run: pip install pyyaml"
        )
        return ScraperConfig(_source="built-in defaults (pyyaml missing)")

    with open(config_path, encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    cfg = ScraperConfig(_source=os.path.abspath(config_path))

    # --- output ---
    if out := raw.get("output"):
        cfg.output.dir = out.get("dir", cfg.output.dir)
        cfg.output.format = _valid_format(out.get("format", cfg.output.format))
        cfg.output.save_raw_html = bool(out.get("save_raw_html", cfg.output.save_raw_html))

    # --- fetch ---
    if fetch := raw.get("fetch"):
        cfg.fetch.delay = float(fetch.get("delay", cfg.fetch.delay))
        cfg.fetch.timeout = int(fetch.get("timeout", cfg.fetch.timeout))
        cfg.fetch.retries = int(fetch.get("retries", cfg.fetch.retries))
        cfg.fetch.respect_robots = bool(fetch.get("respect_robots", cfg.fetch.respect_robots))
        cfg.fetch.user_agent = str(fetch.get("user_agent", cfg.fetch.user_agent))

    # --- crawl ---
    if crawl := raw.get("crawl"):
        cfg.crawl.depth = int(crawl.get("depth", cfg.crawl.depth))
        cfg.crawl.same_domain_only = bool(crawl.get("same_domain_only", cfg.crawl.same_domain_only))
        cfg.crawl.path_prefix_depth = int(crawl.get("path_prefix_depth", cfg.crawl.path_prefix_depth))
        cfg.crawl.exclude_patterns = list(crawl.get("exclude_patterns", cfg.crawl.exclude_patterns))

    # --- resume ---
    if resume := raw.get("resume"):
        cfg.resume.enabled = bool(resume.get("enabled", cfg.resume.enabled))

    # --- logging ---
    if logging_cfg := raw.get("logging"):
        cfg.logging.console_level = logging_cfg.get("console_level", cfg.logging.console_level).upper()
        cfg.logging.file_level = logging_cfg.get("file_level", cfg.logging.file_level).upper()

    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _valid_format(fmt: str) -> str:
    fmt = str(fmt).lower().lstrip(".")
    if fmt not in ("md", "txt"):
        raise ValueError(f"scraper_config.yaml: output.format must be 'md' or 'txt', got '{fmt}'")
    return fmt


def _cli_set(args: argparse.Namespace, attr: str) -> bool:
    """Return True if the argparse attribute exists and is not None."""
    return hasattr(args, attr) and getattr(args, attr) is not None
