#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Iterable, TextIO

# Support running from repo without install (src/ layout)
try:
    from object_harvest.logging import get_logger  # type: ignore
except Exception:  # pragma: no cover - fallback for local execution
    import os
    import sys as _sys

    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    SRC = os.path.join(ROOT, "src")
    if SRC not in _sys.path:
        _sys.path.insert(0, SRC)
    from object_harvest.logging import get_logger  # type: ignore

from tqdm import tqdm

logger = get_logger(__name__)


def extract_inner_describe(text: str) -> str:
    """Extract the actual description if the describe value accidentally contains JSON like
    '{"describe": "...", "objects": [...]}'.

    Strategy:
    1) If the string itself is valid JSON object with a 'describe' key, return that.
    2) Else regex search for '"describe"\\s*:\\s*("..."), capture and JSON-decode to handle escapes.
    3) Else, if '"objects"' is present, trim everything after it as a heuristic.
    4) Fallback to the original stripped text.
    """
    s = text.strip()
    if not s:
        return s

    # Step 1: Try to parse the whole string as JSON and get describe
    if s.startswith("{") and '"describe"' in s:
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "describe" in obj:
                v = obj.get("describe")
                if isinstance(v, str):
                    return v.strip()
        except Exception:
            pass

    # Step 2: Regex for JSON substring of describe value
    # Capture balanced JSON string (quotes with possible escapes). We'll capture the outer quotes and decode via json.loads
    m = re.search(r'"describe"\s*:\s*("(?:\\.|[^"\\])*")', s, flags=re.DOTALL)
    if m:
        token = m.group(1)  # includes surrounding quotes
        try:
            return json.loads(token).strip()
        except Exception:
            # drop quotes and unescape minimally
            inner = token.strip('"')
            return inner.replace('\\"', '"').strip()

    # Step 3: If objects present, try to take substring before it
    if '"objects"' in s:
        before = s.split('"objects"', 1)[0]
        # Remove trailing commas/braces if present
        before = before.rsplit(",", 1)[0] if "," in before else before
        # Also strip a leading '"describe":' label if present
        m2 = re.search(r'"describe"\s*:\s*(.*)$', before, flags=re.DOTALL)
        if m2:
            candidate = m2.group(1).strip()
            # If quoted, peel quotes (best-effort)
            if candidate.startswith('"') and '"' in candidate[1:]:
                # take until next unescaped quote
                m3 = re.match(r'"((?:\\.|[^"\\])*)"', candidate)
                if m3:
                    try:
                        return json.loads('"' + m3.group(1) + '"').strip()
                    except Exception:
                        return m3.group(1).replace('\\"', '"').strip()
            return candidate.strip().strip("{}").strip()

    # Step 4: Fallback
    return s


def clean_jsonl_lines(lines: Iterable[str], stats: dict | None = None) -> Iterable[str]:
    for ln in lines:
        line = ln.strip()
        if not line:
            if stats is not None:
                stats["total"] = stats.get("total", 0) + 1
            continue
        try:
            obj = json.loads(line)
        except Exception:
            # passthrough non-json lines
            if stats is not None:
                stats["total"] = stats.get("total", 0) + 1
                stats["malformed"] = stats.get("malformed", 0) + 1
            yield ln
            continue

        if isinstance(obj, dict) and isinstance(obj.get("describe"), str):
            desc = obj["describe"]
            if ('"describe"' in desc) or ('"objects"' in desc):
                obj["describe"] = extract_inner_describe(desc)
                if stats is not None:
                    stats["modified"] = stats.get("modified", 0) + 1
                    stats["total"] = stats.get("total", 0) + 1
                yield json.dumps(obj, ensure_ascii=False) + "\n"
            else:
                if stats is not None:
                    stats["total"] = stats.get("total", 0) + 1
                yield ln if ln.endswith("\n") else ln + "\n"
        else:
            if stats is not None:
                stats["total"] = stats.get("total", 0) + 1
            yield ln if ln.endswith("\n") else ln + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Clean malformed 'describe' values in synthesis JSONL outputs by extracting inner description text."
    )
    ap.add_argument("--input", "-i", required=True, help="Input JSONL file")
    ap.add_argument(
        "--out",
        "-o",
        default="-",
        help="Output file path (default: stdout)",
    )
    args = ap.parse_args(argv)

    logger.info(f"reading input: {args.input}")
    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)
    logger.info(f"loaded {total} lines")

    out_stream: TextIO
    if args.out == "-":
        out_stream = sys.stdout
        close_out = False
    else:
        out_stream = open(args.out, "w", encoding="utf-8")
        close_out = True

    stats: dict = {}
    try:
        for cleaned in tqdm(
            clean_jsonl_lines(lines, stats=stats),
            total=total,
            desc="clean",
            unit="line",
        ):
            out_stream.write(cleaned)
    finally:
        if close_out:
            out_stream.close()

    mod = stats.get("modified", 0)
    malformed = stats.get("malformed", 0)
    logger.info(f"done. total={total}, modified={mod}, malformed={malformed}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
