"""Convert legacy metadata.jsonl 'objects' list-of-dicts format to nested dict schema.

Legacy format (per line):
    {"describe": "...", "objects": [ {"cat": "a cat"}, {"dog": "a dog"} ], ... }

Intermediate (previous) flat dict format (for reference):
    {"describe": "...", "objects": {"cat": "a cat", "dog": "a dog"}, ... }

New nested format:
    {"describe": "...", "objects": {"objects": {"cat": "a cat", "dog": "a dog"}}, ... }

Rules:
- If line lacks 'objects' or it's already in nested format (contains 'objects' key whose value is a dict containing 'objects'), it's passed through unchanged.
- If 'objects' is a list, only dictionary entries with exactly one key are used.
- If 'objects' is a flat dict (previous migration), it will be wrapped under {"objects": <flat>}.
- First occurrence of a key wins; duplicates are ignored when building from list.
- Non-JSON lines or lines that fail to parse are copied verbatim unless --skip-invalid set.

CLI Usage Examples:
    # Convert in place (creates backup metadata.jsonl.bak)
    uv run python scripts/qwen-image/convert_metadata_objects.py --in-place path/to/metadata.jsonl

    # Convert multiple files writing side-by-side with suffix
    uv run python scripts/qwen-image/convert_metadata_objects.py -i meta1.jsonl meta2.jsonl --suffix converted

    # Convert to a specific output file
    uv run python scripts/qwen-image/convert_metadata_objects.py -i meta.jsonl -o meta_new.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from object_harvest.logging import get_logger

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Convert metadata.jsonl objects list -> dict format")
    p.add_argument(
        "-i", "--input", nargs="+", required=True, help="One or more metadata.jsonl files to convert"
    )
    p.add_argument(
        "-o", "--output", help="Single output file (only valid when exactly one input provided)"
    )
    p.add_argument(
        "--suffix",
        default="converted",
        help="Suffix appended before extension when not in-place (default: converted)",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite files in place (creates .bak backup)",
    )
    p.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip invalid JSON lines instead of copying verbatim",
    )
    return p


def convert_objects_field(obj: Dict[str, Any]) -> Dict[str, Any]:
    objs = obj.get("objects")
    # Already nested new format: {'objects': {'objects': {...}}}
    if isinstance(objs, dict) and "objects" in objs and isinstance(objs["objects"], dict):
        return obj  # assume already correct
    # Legacy list format
    if isinstance(objs, list):
        flat: Dict[str, str] = {}
        for entry in objs:
            if isinstance(entry, dict) and len(entry) == 1:
                k, v = next(iter(entry.items()))
                k_s = str(k)
                if k_s not in flat:
                    flat[k_s] = str(v)
        obj["objects"] = {"objects": flat}
        return obj
    # Flat dict from previous migration
    if isinstance(objs, dict):
        obj["objects"] = {"objects": {str(k): str(v) for k, v in objs.items()}}
        return obj
    return obj


def process_file(src: Path, dst: Path, skip_invalid: bool) -> int:
    count_converted = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            stripped = line.rstrip("\n")
            if not stripped.strip():
                fout.write(line)
                continue
            try:
                data = json.loads(stripped)
            except Exception as e:
                if skip_invalid:
                    logger.warning("Skipping invalid JSON line %s:%d (%s)", src, line_no, e)
                    continue
                fout.write(line)
                continue
            if isinstance(data, dict):
                before = data.get("objects")
                data = convert_objects_field(data)
                after = data.get("objects")
                # Count conversions when structure changed from list or flat dict to nested dict
                if before is not after and (isinstance(before, list) or isinstance(before, dict)):
                    count_converted += 1
                json.dump(data, fout, ensure_ascii=False)
                fout.write("\n")
            else:
                # Not a dict, write unchanged
                json.dump(data, fout, ensure_ascii=False)
                fout.write("\n")
    return count_converted


def resolve_destination(src: Path, args: argparse.Namespace) -> Path:
    if args.in_place:
        return src  # temporary write then replace
    if args.output:
        return Path(args.output)
    # derive with suffix
    return src.with_name(src.stem + f".{args.suffix}" + src.suffix)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    inputs = [Path(p) for p in args.input]
    for p in inputs:
        if not p.exists():
            parser.error(f"Input file not found: {p}")

    if args.output and len(inputs) != 1:
        parser.error("--output can only be used with exactly one --input file")

    total_converted = 0

    for src in inputs:
        dst = resolve_destination(src, args)
        if args.in_place:
            backup = src.with_suffix(src.suffix + ".bak")
            tmp = src.with_suffix(src.suffix + ".tmp")
            logger.info("Converting (in-place) %s -> %s (backup: %s)", src, tmp, backup)
            converted = process_file(src, tmp, skip_invalid=args.skip_invalid)
            # move original to backup then tmp to original
            src.replace(backup)
            tmp.replace(src)
        else:
            logger.info("Converting %s -> %s", src, dst)
            converted = process_file(src, dst, skip_invalid=args.skip_invalid)
        total_converted += converted
        logger.info("Converted %d line(s) with list->dict objects in %s", converted, src)

    logger.info("Done. Total lines converted: %d", total_converted)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
