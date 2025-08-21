from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict
from threading import Lock


class JSONLWriter:
	"""Legacy: append JSON lines to a single file (unused by current CLI)."""

	def __init__(self, out_path: str) -> None:
		self.out_path = out_path
		os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
		self._lock = Lock()

	def write(self, record: Dict[str, Any]) -> None:
		line = json.dumps(record, ensure_ascii=False)
		with self._lock:
			with open(self.out_path, "a", encoding="utf-8") as f:
				f.write(line + "\n")


class JSONDirWriter:
	"""Write one JSON file per image under a unique run directory."""

	def __init__(self, base_dir: str) -> None:
		self.base_dir = base_dir
		os.makedirs(self.base_dir or ".", exist_ok=True)
		ts = time.strftime("%Y%m%d-%H%M%S")
		short = uuid.uuid4().hex[:8]
		self.run_dir = os.path.join(self.base_dir, f"run-{ts}-{short}")
		os.makedirs(self.run_dir, exist_ok=True)

	def write(self, filename: str, record: Dict[str, Any]) -> str:
		if not filename.endswith(".json"):
			filename = f"{filename}.json"
		path = os.path.join(self.run_dir, filename)
		with open(path, "w", encoding="utf-8") as f:
			json.dump(record, f, ensure_ascii=False)
		return path
