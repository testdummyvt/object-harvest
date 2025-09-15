from __future__ import annotations

import json

from object_harvest.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "iter_jsonl_input",
    "iter_hf_dataset_input",
]


def iter_jsonl_input(jsonl_path: str, use_desc: bool = False, limit: int | None = None) -> list[dict]:
    items: list[dict] = []
    sel_key = "description" if use_desc else "names"
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for ln in f:
                if limit and len(items) >= limit:
                    break
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                file_name = rec.get("file_name") or rec.get("image")
                objs = rec.get("objects") or {}
                labels: list[str] = []
                if isinstance(objs, dict):
                    vals = objs.get(sel_key)
                    if isinstance(vals, list):
                        labels = [str(x) for x in vals if str(x).strip()]
                if not file_name:
                    continue
                items.append({"path": str(file_name), "det_labels": labels})
    except FileNotFoundError:
        logger.error(f"jsonl not found: {jsonl_path}")
    return items


def iter_hf_dataset_input(dataset_id: str, use_desc: bool = False, limit: int | None = None) -> list[dict]:
    try:
        from datasets import load_dataset, DatasetDict  # type: ignore
    except Exception as e:
        logger.error(f"datasets not available for --hf-dataset: {e}")
        return []
    ds = load_dataset(dataset_id)
    # If ds is a DatasetDict, prefer 'data' split if present, else the first split
    if isinstance(ds, DatasetDict):
        split_key = "data" if "data" in ds else next(iter(ds.keys()))
        sel = ds[split_key]
    else:
        sel = ds
    items: list[dict] = []
    sel_key = "description" if use_desc else "names"
    for i, ex in enumerate(sel):
        if limit and len(items) >= limit:
            break
        file_name = ex.get("file_name")
        img = ex.get("image")
        # Prefer passing PIL image directly if available
        pil_image = None
        try:
            from PIL import Image as _PILImage  # type: ignore
            if img is not None and isinstance(img, _PILImage.Image):
                pil_image = img
        except Exception:
            pass
        if not file_name and not pil_image:
            # Try to extract a path-like ref from datasets Image
            file_name = getattr(img, "filename", None) or getattr(img, "path", None)
            if not file_name:
                file_name = f"example-{i}.jpg"
        labels: list[str] = []
        objs = ex.get("objects") or {}
        if isinstance(objs, dict):
            vals = objs.get(sel_key)
            if isinstance(vals, list):
                labels = [str(x) for x in vals if str(x).strip()]
        item: dict = {"det_labels": labels}
        if pil_image is not None:
            item["image"] = pil_image
        else:
            item["path"] = str(file_name)
        items.append(item)
    return items
