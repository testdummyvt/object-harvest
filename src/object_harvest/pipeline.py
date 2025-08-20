"""Processing pipeline orchestrating ingestion, VLM calls, and JSONL writing."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Iterator

from .ingest import iter_folder, iter_hf_dataset, iter_list_file
from .logging import get_logger
from .schemas import ImageRecord, ObjectItem, RunConfig
from .vlm import VLMClient
from .writer import JsonlWriter

logger = get_logger(__name__)

def _iter_images(cfg: RunConfig) -> Iterator[tuple[str, Path]]:
    count = 0
    if cfg.source_dir:
        for tup in iter_folder(cfg.source_dir):
            yield tup
            count += 1
            if cfg.max_images and count >= cfg.max_images:
                return
    if cfg.list_file:
        for tup in iter_list_file(cfg.list_file):
            yield tup
            count += 1
            if cfg.max_images and count >= cfg.max_images:
                return
    if cfg.dataset:
        for tup in iter_hf_dataset(cfg.dataset, cfg.dataset_split):
            yield tup
            count += 1
            if cfg.max_images and count >= cfg.max_images:
                return

def process_images(cfg: RunConfig) -> None:
    logger.info("🚀 Start run model=%s output=%s", cfg.model, cfg.output)
    client = VLMClient(cfg.model, api_base=cfg.api_base, api_key=cfg.api_key)

    with JsonlWriter(cfg.output) as writer:
        with ThreadPoolExecutor(max_workers=cfg.threads) as ex:
            futures = {}
            for image_id, path in _iter_images(cfg):
                futures[ex.submit(_process_single, client, cfg, image_id, path)] = path
            for fut in as_completed(futures):
                rec = fut.result()
                writer.write_obj(rec.model_dump())
    logger.info("📊 Completed processing")

def _process_single(client: VLMClient, cfg: RunConfig, image_id: str, path: Path) -> ImageRecord:
    t0 = time.time()
    objects: list[ObjectItem] = []
    boxes = []
    parse_error = None
    attempts = 0
    try:
        attempts += 1
        objs, err = client.list_objects(path)
        objects = objs
        parse_error = err
        if cfg.boxes and objects:
            # obtain image size for pixel validation
            try:
                from PIL import Image  # type: ignore
                with Image.open(path) as im:
                    width, height = im.size
            except Exception:  # noqa: BLE001
                width = height = None
            size = (width, height) if (width and height) else None
            bxs, b_err = client.list_boxes(
                path,
                objects,
                size=size,
            )
            boxes = bxs
            if b_err:
                parse_error = (parse_error + "; " + b_err) if parse_error else b_err
    except Exception as e:  # noqa: BLE001
        logger.error("Failed processing %s: %s", path, e)
        return ImageRecord(
            image_id=image_id,
            path=str(path),
            model=cfg.model,
            objects=[],
            boxes=[],
            error=str(e),
        )
    t_total = time.time() - t0
    return ImageRecord(
        image_id=image_id,
        path=str(path),
        model=cfg.model,
        objects=objects,
        boxes=boxes,
        t_total=t_total,
        attempts=attempts,
        parse_error=parse_error,
    )
