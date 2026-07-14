"""
scraper/logger.py
-----------------
Logging setup: colored console output (INFO) + rotating file handler (DEBUG).
"""

import logging
import os
from logging.handlers import RotatingFileHandler

try:
    import colorama
    colorama.init(autoreset=True)
    _COLORS = {
        "DEBUG": colorama.Fore.CYAN,
        "INFO": colorama.Fore.GREEN,
        "WARNING": colorama.Fore.YELLOW,
        "ERROR": colorama.Fore.RED,
        "CRITICAL": colorama.Fore.MAGENTA,
    }
    _RESET = colorama.Style.RESET_ALL
    _HAS_COLOR = True
except ImportError:
    _HAS_COLOR = False


class _ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI color to console output."""

    FMT = "%(asctime)s  %(levelname)-8s  %(message)s"
    DATE = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if _HAS_COLOR:
            color = _COLORS.get(record.levelname, "")
            return f"{color}{msg}{_RESET}"
        return msg


def setup_logger(
    log_dir: str,
    name: str = "scraper",
    console_level: str = "INFO",
    file_level: str = "DEBUG",
) -> logging.Logger:
    """
    Configure and return the root scraper logger.

    Args:
        log_dir:        Directory where ``scraper.log`` will be written.
        name:           Logger name (default ``"scraper"``).
        console_level:  Log level for console output (default ``"INFO"``).
        file_level:     Log level for file output (default ``"DEBUG"``).

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "scraper.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger  # already configured (e.g. re-imported)

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(
        _ColorFormatter(fmt=_ColorFormatter.FMT, datefmt=_ColorFormatter.DATE)
    )
    logger.addHandler(ch)

    # --- File handler (rotating 5 MB × 3 backups) ---
    fh = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
    fh.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  [%(filename)s:%(lineno)d]  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    logger.addHandler(fh)

    return logger


def get_logger(name: str = "scraper") -> logging.Logger:
    """Return the named logger (must call :func:`setup_logger` first)."""
    return logging.getLogger(name)
