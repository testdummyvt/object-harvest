"""Pydantic models describing objects & run metadata."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, validator

from .logging import get_logger

logger = get_logger(__name__)


class ObjectItem(BaseModel):
    name: str = Field(..., description="Canonical object name")
    confidence: float | None = Field(None, ge=0, le=1)


class BoxItem(BaseModel):
    """Bounding box in pixel xyxy format (x1,y1,x2,y2)."""

    name: str
    x1: float = Field(..., ge=0)
    y1: float = Field(..., ge=0)
    x2: float = Field(..., ge=0)
    y2: float = Field(..., ge=0)
    confidence: float | None = Field(None, ge=0, le=1)

    @validator("x2")
    def _x_order(cls, v: float, values):  # noqa: D401
        if "x1" in values and v <= values["x1"]:
            raise ValueError("x2 must be > x1")
        return v

    @validator("y2")
    def _y_order(cls, v: float, values):  # noqa: D401
        if "y1" in values and v <= values["y1"]:
            raise ValueError("y2 must be > y1")
        return v


class ImageRecord(BaseModel):
    image_id: str
    path: str
    width: int | None = None
    height: int | None = None
    model: str
    objects: list[ObjectItem] = []  # noqa: RUF012 (we later may adjust with model_validate)
    boxes: list[BoxItem] = []
    t_fetch: float | None = None
    t_parse: float | None = None
    t_total: float | None = None
    attempts: int | None = None
    error: str | None = None
    parse_error: str | None = None

    @validator("path")
    def _path_norm(cls, v: str) -> str:  # noqa: D401
        return str(Path(v))


class RunConfig(BaseModel):
    source_dir: Path | None = None
    list_file: Path | None = None
    dataset: str | None = None
    dataset_split: str = "train"
    output: Path
    model: str
    boxes: bool = False
    threads: int = 4
    api_base: str | None = None
    api_key: str | None = None
    api_key_env: str = "OPENAI_API_KEY"
    max_images: int | None = None

    @validator("threads")
    def _threads_positive(cls, v: int) -> int:  # noqa: D401
        if v <= 0:
            raise ValueError("threads must be > 0")
        return v


# Parsing helpers
import json, re  # noqa: E402


def _extract_json_block(text: str) -> str | None:
    if not text:
        return None
    # Strip common markdown fences
    fence = re.compile(r"```(?:json)?(.*?)```", re.DOTALL | re.IGNORECASE)
    m = fence.search(text)
    if m:
        text = m.group(1)
    # Find first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def safe_parse_objects(raw: str) -> tuple[list[ObjectItem], str | None]:
    """Attempt to parse JSON text into list[ObjectItem]. Return (objects, error_message)."""
    block = _extract_json_block(raw) or raw
    try:
        data = json.loads(block)
    except Exception as e:  # noqa: BLE001
        return [], f"json_load_error: {e}"
    if isinstance(data, dict) and "objects" in data:
        data = data["objects"]
    if not isinstance(data, list):
        return [], "data_not_list"
    objs: list[ObjectItem] = []
    for item in data:
        if not isinstance(item, dict):
            logger.warning("⚠️ Skipping non-dict item in objects list: %s", type(item).__name__)
            continue
        name = item.get("name") or item.get("label") or item.get("object")
        if not name:
            logger.warning("⚠️ Skipping item with missing name field: %s", item)
            continue
        conf = item.get("confidence") or item.get("score")
        try:
            objs.append(
                ObjectItem(name=str(name), confidence=float(conf) if conf is not None else None)
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("⚠️ Failed to create ObjectItem from %s: %s", item, e)
            continue
    return objs, None
