from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterable, Optional

from object_harvest.logging import get_logger
from object_harvest.reader import iter_images
from object_harvest.vlm import VLMClient, describe_and_list
from object_harvest.writer import JSONDirWriter
from object_harvest.utils import RateLimiter
from tqdm import tqdm


logger = get_logger(__name__)


def _safe_stem(image_ref: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(image_ref)
    if parsed.scheme in ("http", "https"):
        stem = os.path.basename(parsed.path) or parsed.netloc
    else:
        stem = os.path.splitext(os.path.basename(image_ref))[0] or "image"
    return stem.replace("/", "_").replace("\\", "_") or "image"


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="object-harvest",
        description="Extract image descriptions and object lists from images using VLMs; writes one JSON per image in a run folder.",
    )
    p.add_argument(
        "--input",
        required=True,
        help="Input folder of images or a text file with paths/URLs",
    )
    p.add_argument(
        "--out", required=True, help="Output folder to store per-image JSON files"
    )
    p.add_argument(
        "--model",
        default=os.getenv("OBJH_MODEL", "qwen/qwen2.5-vl-72b-instruct"),
        help="Model name",
    )
    p.add_argument(
        "--api-base",
        default=os.getenv("OBJH_API_BASE"),
        help="OpenAI-compatible base URL",
    )
    p.add_argument(
        "--max-workers", type=int, default=min(32, (os.cpu_count() or 4) * 5)
    )
    p.add_argument(
        "--rpm",
        type=int,
        default=int(os.getenv("OBJH_RPM", "0")),
        help="Requests per minute throttle (0=unlimited)",
    )
    p.add_argument(
        "--batch", type=int, default=0, help="Optional batch size; 0 processes all"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help=(
            "If set, and --out points to an existing run-* directory, only process images "
            "that do not already have corresponding JSON outputs in that directory."
        ),
    )
    return p.parse_args(list(argv) if argv is not None else None)


def _process_one(client: VLMClient, item: dict, limiter: RateLimiter | None) -> dict:
    path = item.get("path") or item.get("url")
    try:
        if limiter:
            limiter.acquire()
        result = describe_and_list(client, item)
        return {
            "image": path,
            "description": result.get("description"),
            "objects": result.get("objects", []),
        }
    except Exception as e:  # capture errors per image
        logger.error(f"failed processing {path}: {e}")
        return {
            "image": path,
            "description": None,
            "objects": [],
        }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logger.info("starting object-harvest")

    client = VLMClient(model=args.model, base_url=args.api_base)
    items = list(iter_images(args.input, limit=args.batch if args.batch > 0 else None))
    logger.info(f"found {len(items)} images")

    # Determine target run directory. For --resume, prefer the latest existing run-* under --out
    # or use the provided run-* path directly to write into.
    writer_base = args.out
    if args.resume:
        try:
            if os.path.isdir(args.out) and not os.path.basename(args.out).startswith("run-"):
                run_dirs = [
                    os.path.join(args.out, d)
                    for d in os.listdir(args.out)
                    if d.startswith("run-") and os.path.isdir(os.path.join(args.out, d))
                ]
                if run_dirs:
                    # pick most recently modified
                    writer_base = max(run_dirs, key=os.path.getmtime)
                    logger.info(f"resuming: writing into existing run dir {writer_base}")
        except Exception:
            pass

    writer = JSONDirWriter(writer_base)
    existing_stems: set[str] = set()
    if args.resume:
        try:
            for name in os.listdir(writer.run_dir):
                if name.lower().endswith(".json"):
                    existing_stems.add(os.path.splitext(name)[0])
        except FileNotFoundError:
            pass
    limiter = RateLimiter(rpm=args.rpm) if args.rpm and args.rpm > 0 else None

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = []
        for item in items:
            image_ref = item.get("path") or item.get("url") or "image"
            safe = _safe_stem(image_ref)

            if args.resume and safe in existing_stems:
                # Skip already-processed item
                continue

            futures.append(ex.submit(_process_one, client, item, limiter))
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            # derive a safe filename from the image path or URL
            image_ref = rec.get("image") or "image"
            safe = _safe_stem(image_ref)
            writer.write(f"{safe}", rec)

    total_written = (
        len([n for n in os.listdir(writer.run_dir) if n.lower().endswith('.json')])
        if os.path.isdir(writer.run_dir)
        else 0
    )
    logger.info(f"done. outputs under {writer.run_dir}. files present: {total_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
