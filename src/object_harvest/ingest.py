"""Image ingestion utilities.

Provides generators yielding (image_id, path) pairs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from .logging import get_logger

logger = get_logger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def iter_folder(folder: Path) -> Iterator[tuple[str, Path]]:
    for p in sorted(folder.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p.stem, p


def iter_list_file(list_file: Path) -> Iterator[tuple[str, Path]]:
    for line in list_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        p = Path(line)
        yield p.stem, p


def iter_hf_dataset(
    name: str, split: str = "train"
) -> Iterator[tuple[str, Path]]:  # pragma: no cover - optional
    try:
        from datasets import load_dataset  # type: ignore
    except Exception:  # noqa: BLE001
        logger.warning("datasets library not installed; skipping HF dataset ingestion")
        return
    ds = load_dataset(name, split=split)
    # Heuristic: look for columns with 'image'
    image_col = None
    for c in ds.column_names:
        if "image" in c.lower():
            image_col = c
            break
    if image_col is None:
        logger.warning("No image column found in dataset %s", name)
        return
    for i, row in enumerate(ds):
        img = row[image_col]
        # Save to a temp file path? For now, yield with synthetic ID and underlying PIL image object path not available.
        # Future: support streaming bytes directly.
        tmp_id = f"{name}-{split}-{i}"
        # Write to a temporary file per image (inefficient but simple placeholder)
        try:
            from tempfile import NamedTemporaryFile

            img_path = None
            with NamedTemporaryFile(suffix=".png", delete=False) as nf:
                img.save(nf.name)
                img_path = Path(nf.name)
            yield tmp_id, img_path
        except Exception as e:  # noqa: BLE001
            logger.warning("⚠️ Failed to materialize image %s: %s", tmp_id, e)
            continue


def iter_video_frames(video_path: Path) -> Iterator[tuple[str, Path]]:  # placeholder
    # TODO: implement frame extraction (ffmpeg or cv2). For now, log and yield nothing.
    logger.info("Video frame extraction not implemented yet: %s", video_path)
    if False:
        yield "", Path("")  # pragma: no cover
