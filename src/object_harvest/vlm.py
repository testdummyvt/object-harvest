"""OpenAI-compatible Vision Language Model client abstraction."""
from __future__ import annotations

import base64
import json
import os
from collections.abc import Sequence
from pathlib import Path

from .logging import get_logger
from .schemas import BoxItem, ObjectItem, safe_parse_objects

logger = get_logger(__name__)

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    logger.debug("python-dotenv not available, using system environment variables only")

PROMPT_OBJECTS = """You are an assistant that lists objects present in an image.
Return ONLY JSON: {"objects": [{"name": "...", "confidence": 0.75}, ...]}
Be concise; 1 word names when possible.
"""

PROMPT_BOXES = """You are an assistant that draws bounding boxes for the given objects in an image.
Input will include a JSON array of object names. Return ONLY JSON:
{"boxes": [{"name": "object", "x1": 45, "y1": 60, "x2": 180, "y2": 240, "confidence": 0.8}, ...]}
Coordinates in PIXELS with x1<x2, y1<y2 (xyxy format).
If you only know center (x,y) and width/height (w,h), convert to xyxy as:
  x1 = x, y1 = y, x2 = x + w, y2 = y + h.
"""

class VLMClient:
    def __init__(self, model: str, api_base: str | None = None, api_key: str | None = None):
        self.model = model
        # Use provided values or fall back to environment variables
        self.api_base = api_base or os.getenv("OPENAI_API_BASE")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            logger.warning(
                "No API key found. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter"
            )
        
        try:
            from openai import OpenAI  # type: ignore
        except Exception:  # noqa: BLE001
            self._openai_cls = None
            logger.warning("openai package not installed; VLMClient unavailable for real calls")
        else:
            self._openai_cls = OpenAI

    # --- Utility
    def _image_to_b64(self, image_path: Path) -> str:
        data = image_path.read_bytes()
        return base64.b64encode(data).decode("utf-8")

    def _client(self):
        if not self._openai_cls:
            raise RuntimeError("OpenAI client class not available")
        return self._openai_cls(base_url=self.api_base, api_key=self.api_key)  # type: ignore

    # --- Public methods
    def list_objects(self, image_path: Path) -> tuple[list[ObjectItem], str | None]:
        prompt = PROMPT_OBJECTS
        raw = self._call_api(image_path, prompt)
        objects, err = safe_parse_objects(raw)
        return objects, err

    def list_boxes(
        self,
        image_path: Path,
        object_list: Sequence[ObjectItem],
        size: tuple[int, int] | None = None,
    ) -> tuple[list[BoxItem], str | None]:
        json_objects = json.dumps([o.model_dump() for o in object_list])
        prompt = PROMPT_BOXES + f"\nOBJECTS_JSON={json_objects}\nIf giving pixel boxes use integers; image size may be provided separately."  # size not embedded to reduce token usage
        raw = self._call_api(image_path, prompt)
        # parse boxes
        boxes: list[BoxItem] = []
        block = raw
        try:
            data = json.loads(block)
        except Exception as e:  # noqa: BLE001
            return [], f"json_load_error: {e}"
        if isinstance(data, dict) and "boxes" in data:
            data = data["boxes"]
        if isinstance(data, list):
            for b in data:
                try:
                    # Legacy support: if x,y,w,h given convert to xyxy
                    if {"x", "y", "w", "h"}.issubset(b.keys()) and not {"x1", "y1", "x2", "y2"}.issubset(b.keys()):
                        x1 = float(b.get("x"))
                        y1 = float(b.get("y"))
                        x2 = x1 + float(b.get("w"))
                        y2 = y1 + float(b.get("h"))
                    else:
                        x1 = float(b.get("x1"))
                        y1 = float(b.get("y1"))
                        x2 = float(b.get("x2"))
                        y2 = float(b.get("y2"))
                    # If coordinates look normalized (<=1) and we have size, scale to pixels
                    if size and max(x1, y1, x2, y2) <= 1.01:
                        iw, ih = size
                        x1p = x1 * iw
                        x2p = x2 * iw
                        y1p = y1 * ih
                        y2p = y2 * ih
                    else:
                        x1p, y1p, x2p, y2p = x1, y1, x2, y2
                    boxes.append(
                        BoxItem(
                            name=str(b.get("name")),
                            x1=x1p,
                            y1=y1p,
                            x2=x2p,
                            y2=y2p,
                            confidence=float(b.get("confidence")) if b.get("confidence") is not None else None,
                        )
                    )
                except Exception as e:  # noqa: BLE001
                    logger.error(f"Failed to parse box item: {b} ({e})")
                    continue
        return boxes, None

    # --- Low-level call
    def _call_api(self, image_path: Path, prompt: str) -> str:
        b64 = self._image_to_b64(image_path)
        if not self._openai_cls:
            logger.debug("Mock call_api returning empty JSON due to missing openai package")
            return "{}"
        client = self._client()
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
            temperature=0,
        )
        try:
            return resp.choices[0].message.content or "{}"  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            logger.error("Failed to extract content: %s", e)
            return "{}"
