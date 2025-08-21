from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, cast

from PIL import Image
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)

from object_harvest.logging import get_logger


logger = get_logger(__name__)


PROMPT = """
Produce a single JSON object with the following keys:

"description" — a single, concise paragraph (one line; no hard newlines) that describes the scene with emphasis on all objects present. Do NOT use phrases such as "in this image", "this image is", or any similar meta phrases. Mention each object explicitly and focus on visual details.

"objects" — an array of strings listing every object that appears in the scene (e.g. ["object1", "object2", ...]).

Output only valid JSON (no extra text, no explanation). Example:
{
"description": "A sunlit kitchen counter holds a stainless-steel kettle steaming beside a wooden cutting board scattered with sliced lemons and a serrated knife, a red mug rests near a potted basil plant at the windowsill.",
"objects": ["kettle", "cutting board", "lemons", "knife", "mug", "basil plant", "windowsill"]
}
	""".strip()


def _load_image_bytes(path: str) -> bytes:
    # Ensure consistent encoding; rely on PIL to open and re-encode as PNG
    with Image.open(path) as im:
        im = im.convert("RGB")
        from io import BytesIO

        buf = BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()


@dataclass
class VLMClient:
    model: str
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=self.base_url or os.environ.get("OPENAI_API_BASE"),
        )
        self.provider = (
            "openai"
            if not (self.base_url or os.environ.get("OPENAI_API_BASE"))
            else "custom"
        )


def describe_and_list(client: VLMClient, item: Dict[str, Any]) -> Dict[str, Any]:
    start = time.time()

    parts: list[dict] = [{"type": "text", "text": PROMPT}]
    if item.get("url"):
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": item["url"]},
            }
        )
    elif item.get("path"):
        b64 = base64.b64encode(_load_image_bytes(item["path"])).decode("utf-8")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            }
        )
    else:
        raise ValueError("item must have either 'url' or 'path'")

    # content must be a list of content parts for user messages in 1.x SDK
    user_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        # cast to Any to satisfy strict type checker for content parts
        "content": cast(Any, parts),
    }
    messages: List[ChatCompletionMessageParam] = [user_msg]
    resp = client.client.chat.completions.create(
        model=client.model,
        messages=messages,
        temperature=0.2,
        max_tokens=400,
    )

    content = resp.choices[0].message.content or "{}"
    try:
        import json

        parsed = json.loads(content)
    except Exception as e:
        logger.error(f"JSON parse error: {e}. Raw: {content[:200]}")
        parsed = {"description": None, "objects": []}

    latency_ms = int((time.time() - start) * 1000)
    parsed["latency_ms"] = latency_ms
    return parsed
