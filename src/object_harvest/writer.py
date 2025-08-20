"""JSONL writer utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import IO, Any

from .logging import get_logger

logger = get_logger(__name__)

class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self._fh: IO[str] | None = None

    def __enter__(self) -> "JsonlWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def write_obj(self, obj: Any) -> None:
        if not self._fh:
            raise RuntimeError("Writer not opened")
        line = json.dumps(obj, sort_keys=True, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        logger.debug("Wrote JSONL line (%d chars)", len(line))
