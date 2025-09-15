from __future__ import annotations

import json
import os

from object_harvest.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "load_objects_from_file",
    "parse_objects_arg",
    "load_describe_objects_map",
]


def load_objects_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def parse_objects_arg(arg: str | None) -> list[str]:
    if not arg:
        return []
    candidate = arg.strip()
    try:
        if os.path.exists(candidate) and os.path.isfile(candidate):
            return load_objects_from_file(candidate)
    except Exception:
        pass
    return [s.strip() for s in candidate.split(",") if s.strip()]


def load_describe_objects_map(run_dir: str) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for name in os.listdir(run_dir):
        if not name.lower().endswith((".json", ".ndjson")):
            continue
        stem = os.path.splitext(name)[0]
        try:
            path = os.path.join(run_dir, name)
            if name.lower().endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                objs = data.get("objects") or []
                if isinstance(objs, list):
                    mapping[stem] = [str(o) for o in objs]
            else:
                labels: list[str] = []
                with open(path, "r", encoding="utf-8") as f:
                    for ln in f:
                        ln = ln.strip()
                        if not ln:
                            continue
                        try:
                            obj = json.loads(ln)
                            if isinstance(obj, dict) and obj:
                                labels.extend([str(k) for k in obj.keys()])
                        except Exception:
                            continue
                if labels:
                    mapping[stem] = labels
        except Exception:
            continue
    return mapping
