"""Command line interface for object-harvest."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from .logging import get_logger
from .pipeline import process_images
from .schemas import RunConfig

logger = get_logger(__name__)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="object-harvest", description="Extract object lists & boxes with VLM")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", type=Path, help="Directory of images")
    src.add_argument("--list-file", type=Path, help="Text file of image paths")
    src.add_argument("--dataset", type=str, help="HuggingFace dataset name")
    p.add_argument("--dataset-split", default="train")
    p.add_argument("--output", type=Path, default=Path("harvest.jsonl"))
    p.add_argument("--model", required=True, help="Model name (OpenAI-compatible)")
    p.add_argument("--boxes", action="store_true", help="Enable bounding box pass")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--api-base", type=str, default=None)
    p.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY")
    p.add_argument("--max-images", type=int, default=None)
    return p

def parse_env_key(env_name: str) -> str | None:
    return os.environ.get(env_name)

def main(argv: list[str] | None = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    api_key = parse_env_key(args.api_key_env)
    cfg = RunConfig(
        source_dir=args.source,
        list_file=args.list_file,
        dataset=args.dataset,
        dataset_split=args.dataset_split,
        output=args.output,
        model=args.model,
        boxes=args.boxes,
        threads=args.threads,
        api_base=args.api_base,
        api_key=api_key,
        api_key_env=args.api_key_env,
        max_images=args.max_images,
    )
    process_images(cfg)
    return 0

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
