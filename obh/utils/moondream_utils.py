"""Moondream API utilities for image captioning."""

import os
import threading
import time
from typing import Literal, Optional

import moondream as md
from PIL import Image


# Rate limiting state for Moondream API
_md_last_call_time = 0.0
_md_lock = threading.Lock()

# Valid caption length options
CaptionLength = Literal["short", "normal", "long"]


def setup_moondream_client(
    api_key: Optional[str] = None,
    local: bool = False,
) -> md.vl:
    """Setup Moondream client for image captioning.

    Args:
        api_key: API key for cloud mode. If None, reads from MOONDREAM_API_KEY env var.
        local: If True, use local server at http://localhost:2020/v1

    Returns:
        Configured Moondream vl client.

    Raises:
        ValueError: If cloud mode and no API key provided.
    """
    if local:
        # Local mode uses localhost, no API key needed
        return md.vl(endpoint="http://localhost:2020/v1")

    # Cloud mode requires API key
    api_key = api_key or os.getenv("MOONDREAM_API_KEY")
    if not api_key:
        raise ValueError(
            "API key not provided for Moondream cloud mode. "
            "Set MOONDREAM_API_KEY env var or use --api-key, or use --local for local model."
        )
    return md.vl(api_key=api_key)


def rate_limited_caption(
    client: md.vl,
    image_path: str,
    length: CaptionLength,
    interval: float,
) -> str:
    """Generate a rate-limited image caption using Moondream.

    Args:
        client: Moondream vl client.
        image_path: Path to the image file.
        length: Caption length - "short", "normal", or "long".
        interval: Minimum seconds between calls.

    Returns:
        Generated caption string.
    """
    global _md_last_call_time

    with _md_lock:
        now = time.time()
        elapsed = now - _md_last_call_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _md_last_call_time = time.time()

    # Load and caption the image
    image = Image.open(image_path)
    result = client.caption(image, length=length)
    return result["caption"]
