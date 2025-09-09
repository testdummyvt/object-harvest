"""Upload a generated image dataset (metadata.jsonl + data/ folder) to Hugging Face Hub.

Expected directory layout:

dataset_root/
  metadata.jsonl        # lines with at least {"file_name": "data/<image_name>"}
  data/                 # image files referenced by metadata.jsonl

Example:
  uv run python scripts/qwen-image/upload_data.py \
    --dataset-dir ./generated \
    --repo-id your-username/qwen-image-samples \
    --private

Requires: huggingface_hub (install if missing: `uv pip install huggingface_hub`).
Authenticates via HF token from (in order): --token, HF_TOKEN env var, or cached login.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Tuple

from object_harvest.logging import get_logger

logger = get_logger(__name__)

# Optional deps: huggingface_hub and datasets
try:  # pragma: no cover - optional import
    from huggingface_hub import HfFolder
except ImportError:  # pragma: no cover
    HfFolder = None  # type: ignore

try:  # pragma: no cover - optional import
    from datasets import load_dataset, Image as HFImage
except ImportError:  # pragma: no cover
    load_dataset = None  # type: ignore
    HFImage = None  # type: ignore


def validate_dataset_root(root: Path) -> Tuple[Path, Path]:
    """Ensure required files/folders exist; return (metadata_path, data_dir)."""
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found or not a directory: {root}")
    metadata_path = root / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.jsonl missing in {root}")
    data_dir = root / "data"
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(f"data/ folder missing in {root}")
    return metadata_path, data_dir


def iter_metadata(metadata_path: Path) -> Iterable[dict]:
    with metadata_path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                logger.warning("Skipping malformed JSON on line %d", line_num)
                continue
            if not isinstance(obj, dict):
                continue
            yield obj


def check_file_references(metadata_path: Path, data_dir: Path) -> int:
    missing = 0
    for obj in iter_metadata(metadata_path):
        rel = obj.get("file_name")
        if not isinstance(rel, str):
            continue
        p = data_dir.parent / rel
        if not p.exists():
            logger.warning("Missing referenced file: %s", rel)
            missing += 1
    return missing


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upload image dataset (metadata.jsonl + data/) to Hugging Face Hub")
    p.add_argument("--dataset-dir", required=True, help="Path to dataset root containing metadata.jsonl and data/ folder")
    p.add_argument("--repo-id", required=True, help="Target repo id (e.g. username/dataset-name)")
    p.add_argument("--token", default=None, help="Hugging Face token (optional if already logged in or HF_TOKEN set)")
    p.add_argument("--private", action="store_true", help="Create the repo as private")
    p.add_argument("--branch", default="main", help="Target branch to push to (default: main)")
    p.add_argument("--commit-message", default="Add dataset", help="Commit message for the upload")
    p.add_argument("--allow-existing", action="store_true", help="Do not fail if repo already exists")
    p.add_argument("--skip-reference-check", action="store_true", help="Skip verifying that all file_name entries exist")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if load_dataset is None or HFImage is None:
        raise SystemExit(
            "datasets not installed. Install first: uv pip install datasets huggingface_hub"
        )
    if HfFolder is None:
        raise SystemExit(
            "huggingface_hub not installed. Install first: uv pip install huggingface_hub"
        )

    dataset_root = Path(args.dataset_dir).expanduser().resolve()
    metadata_path, data_dir = validate_dataset_root(dataset_root)

    if not args.skip_reference_check:
        missing = check_file_references(metadata_path, data_dir)
        if missing:
            logger.warning("%d missing referenced file(s) detected.", missing)

    token = args.token or os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if not token:
        raise SystemExit("No Hugging Face token provided. Use --token or set HF_TOKEN.")

    logger.info("Loading metadata JSONL into Dataset: %s", metadata_path)
    ds_any = load_dataset("json", data_files={"data": str(metadata_path)})  # returns DatasetDict
    if not isinstance(ds_any, dict) or "data" not in ds_any:
        raise SystemExit("Unexpected dataset structure returned by load_dataset (expected key 'data')")
    raw = ds_any["data"]  # datasets.Dataset
    # Safety: ensure dataset provides map and column_names attributes
    if not hasattr(raw, "map") or not hasattr(raw, "column_names"):
        raise SystemExit("Loaded object is not a datasets.Dataset instance")

    # Add absolute image path and cast to Image feature
    abs_root = dataset_root

    missing_file_name = 0

    def add_image(example):  # type: ignore
        nonlocal missing_file_name
        rel = example.get("file_name")
        if isinstance(rel, str) and rel:
            path = (abs_root / rel).resolve()
            example["image"] = str(path)
        else:
            missing_file_name += 1
            example["image"] = None
        return example

    raw = raw.map(add_image)
    if missing_file_name:
        logger.warning("%d record(s) missing file_name; image set to null", missing_file_name)
    col_names = getattr(raw, "column_names", []) or []
    if "image" not in col_names:
        raise SystemExit("Failed to create image column after mapping.")
    raw = raw.cast_column("image", HFImage())  # type: ignore

    # # Optional: ensure objects column is present even if empty
    # if "objects" not in raw.column_names:
    #     raw = raw.add_column("objects", [None] * len(raw))

    logger.info(
        "Pushing dataset to hub: repo=%s, private=%s, revision=%s", args.repo_id, args.private, args.branch
    )
    raw.push_to_hub(
        args.repo_id,
        private=args.private,
        token=token,
        commit_message=args.commit_message,
        revision=args.branch,
    )
    logger.info("Dataset push complete.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
