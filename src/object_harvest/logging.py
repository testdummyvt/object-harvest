"""Logging utilities with emoji level prefixes.

Usage:
    from .logging import get_logger
    logger = get_logger(__name__)
"""
from __future__ import annotations

import logging
import sys
from typing import Dict

_LEVEL_EMOJI: Dict[int, str] = {
    logging.DEBUG: "🔍",
    logging.INFO: "🟢",
    logging.WARNING: "⚠️",
    logging.ERROR: "❌",
    logging.CRITICAL: "🔥",
}

class _EmojiFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - simple override
        emoji = _LEVEL_EMOJI.get(record.levelno, "▫️")
        record.msg = f"{emoji} {record.msg}"
        return super().format(record)

def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module-level logger with emoji formatting applied once.

    Idempotent: calling multiple times won't duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = _EmojiFormatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s", "%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
