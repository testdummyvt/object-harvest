from __future__ import annotations

import os
from typing import Dict, Generator, Iterable, Optional

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in IMAGE_EXTS


def _iter_dir(dir_path: str) -> Generator[Dict, None, None]:
    for root, _dirs, files in os.walk(dir_path):
        for name in files:
            fp = os.path.join(root, name)
            if _is_image(fp):
                yield {"path": fp, "id": os.path.relpath(fp, dir_path)}


def _iter_listfile(list_path: str) -> Generator[Dict, None, None]:
    with open(list_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.strip()
            if not s:
                continue
            if s.startswith("http://") or s.startswith("https://"):
                yield {"url": s, "id": f"url:{i}"}
            else:
                if _is_image(s):
                    yield {"path": s, "id": os.path.basename(s)}


def iter_images(inp: str, limit: Optional[int] = None) -> Iterable[Dict]:
    """Yield canonicalized image items from a folder or list file.

    Each item contains at least one of: `path` or `url`. Always includes `id`.
    """
    count = 0
    if os.path.isdir(inp):
        it = _iter_dir(inp)
    elif os.path.isfile(inp):
        it = _iter_listfile(inp)
    else:
        raise FileNotFoundError(f"input not found: {inp}")

    for item in it:
        yield item
        count += 1
        if limit and count >= limit:
            break
