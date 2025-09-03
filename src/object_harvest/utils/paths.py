from __future__ import annotations

import os
from urllib.parse import urlparse

__all__ = ["safe_stem"]


def safe_stem(image_ref: str) -> str:
    """Derive a safe filename stem from a local path or URL.

    - For URLs, use the path basename or the netloc if path empty.
    - For local paths, use the basename without extension.
    - Replace path separators to avoid nested dirs; default to 'image'.
    """
    parsed = urlparse(image_ref)
    if parsed.scheme in ("http", "https"):
        stem = os.path.basename(parsed.path) or parsed.netloc
    else:
        stem = os.path.splitext(os.path.basename(image_ref))[0] or "image"
    return stem.replace("/", "_").replace("\\", "_") or "image"
