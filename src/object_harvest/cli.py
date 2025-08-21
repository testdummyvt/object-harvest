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
        default=os.getenv("OPENAI_MODEL", "qwen/qwen2.5-vl-72b-instruct"),
        help="Model name",
    )
    p.add_argument(
        "--api-base",
        default=os.getenv("OPENAI_API_BASE"),
        help="OpenAI-compatible base URL",
    )
    p.add_argument(
        "--max-workers", type=int, default=min(32, (os.cpu_count() or 4) * 5)
    )
    p.add_argument(
        "--rpm",
        type=int,
        default=int(os.getenv("OPENAI_RPM", "0")),
        help="Requests per minute throttle (0=unlimited)",
    )
    p.add_argument(
        "--batch", type=int, default=0, help="Optional batch size; 0 processes all"
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
            "bboxes": {},  # not generated in this phase
        }
    except Exception as e:  # capture errors per image
        logger.error(f"failed processing {path}: {e}")
        return {
            "image": path,
            "description": None,
            "objects": [],
            "bboxes": {},
        }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    logger.info("starting object-harvest")

    client = VLMClient(model=args.model, base_url=args.api_base)
    items = list(iter_images(args.input, limit=args.batch if args.batch > 0 else None))
    logger.info(f"found {len(items)} images")

    writer = JSONDirWriter(args.out)
    limiter = RateLimiter(rpm=args.rpm) if args.rpm and args.rpm > 0 else None

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(_process_one, client, item, limiter) for item in items]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            rec = fut.result()
            # derive a safe filename from the image path or URL
            image_ref = rec.get("image") or "image"
            # use base name for file paths, or sanitized tail for URLs
            from urllib.parse import urlparse

            parsed = urlparse(image_ref)
            if parsed.scheme in ("http", "https"):
                stem = os.path.basename(parsed.path) or parsed.netloc
            else:
                stem = os.path.splitext(os.path.basename(image_ref))[0] or "image"
            safe = stem.replace("/", "_").replace("\\", "_") or "image"
            writer.write(f"{safe}", rec)

    logger.info(f"done. wrote {len(items)} JSON files under {writer.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
