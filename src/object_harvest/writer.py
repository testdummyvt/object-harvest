from __future__ import annotations

import json
import os
from typing import Any, Dict
from threading import Lock


class JSONLWriter:
	def __init__(self, out_path: str) -> None:
		self.out_path = out_path
		os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
		self._lock = Lock()

	def write(self, record: Dict[str, Any]) -> None:
		line = json.dumps(record, ensure_ascii=False)
		with self._lock:
			with open(self.out_path, "a", encoding="utf-8") as f:
				f.write(line + "\n")
