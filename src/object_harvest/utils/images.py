from __future__ import annotations

import base64
from io import BytesIO
from typing import Any, Dict

from PIL import Image
from urllib.parse import urlparse

__all__ = [
    "load_image_bytes_jpeg",
    "load_image_from_item",
    "image_part_from_item",
]


def load_image_bytes_jpeg(path: str) -> bytes:
    """Open an image and re-encode as high-quality JPEG bytes."""
    with Image.open(path) as im:
        im = im.convert("RGB")
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=95, optimize=True, subsampling=0)
        return buf.getvalue()


def load_image_from_item(item: Dict[str, Any]) -> Image.Image:
    if item.get("image") and isinstance(item.get("image"), Image.Image):
        return item["image"].convert("RGB")
    if item.get("path"):
        return Image.open(item["path"]).convert("RGB")
    if item.get("url"):
        import urllib.request

        parsed = urlparse(item["url"])
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"unsupported url scheme for image load: {parsed.scheme or '(none)'}"
            )

        with urllib.request.urlopen(item["url"]) as resp:
            data = resp.read()
        return Image.open(BytesIO(data)).convert("RGB")
    raise ValueError("item must include 'path' or 'url'")


def image_part_from_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Return an OpenAI chat image_url part from an item with path or url."""
    if item.get("url"):
        parsed = urlparse(item["url"])
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"unsupported url scheme for image part: {parsed.scheme or '(none)'}"
            )
        return {"type": "image_url", "image_url": {"url": item["url"]}}
    if item.get("path"):
        b64 = base64.b64encode(load_image_bytes_jpeg(item["path"]))
        b64str = b64.decode("utf-8")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64str}"},
        }
    raise ValueError("item must include 'path' or 'url'")
