from __future__ import annotations
import time
from typing import Any, Dict, List, cast
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionUserMessageParam,
)
from dotenv import load_dotenv

from object_harvest.logging import get_logger
from object_harvest.utils.clients import AIClient
from object_harvest.utils.images import image_part_from_item

logger = get_logger(__name__)
load_dotenv()

PROMPT = """
You are assisting object detection preparation. Suggest relevant objects present in the image and include people if present.

Output format: NDJSON (newline-delimited JSON). Emit one line per object, where each line is a single-key JSON object mapping the object name to a short natural-language description of that object as seen in the image. Do not include any extra text, headers, or code fences.

Examples (format only):
{"bicycle": "a blue road bike leaning against a brick wall"}
{"person": "a man wearing a red jacket walking beside the bike"}
{"backpack": "black backpack with a water bottle in side pocket"}

Rules:
- Keys are object names (lowercase preferred). Include people when present (e.g., "person").
- Values are concise visual descriptions specific to the image.
- Output only NDJSON lines, nothing else.
""".strip()


def describe_and_list(client: AIClient, item: Dict[str, Any]) -> Dict[str, Any]:
    """Deprecated: kept for backward-compat. Now delegates to describe_objects_ndjson and returns parsed lines.

    Returns {"ndjson": str, "objects": [str], "latency_ms": int}
    """
    start = time.time()
    ndjson_text = describe_objects_ndjson(client, item)
    # Best-effort extract object names from NDJSON for compatibility
    objects: list[str] = []
    for line in ndjson_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            import json as _json

            obj = _json.loads(line)
            if isinstance(obj, dict) and obj:
                objects.extend(list(obj.keys()))
        except Exception:
            continue
    latency_ms = int((time.time() - start) * 1000)
    return {"ndjson": ndjson_text, "objects": objects, "latency_ms": latency_ms}


def describe_objects_ndjson(client: AIClient, item: Dict[str, Any]) -> str:
    """Generate NDJSON lines mapping object name -> per-object description, including people when present."""
    parts: list[dict] = [{"type": "text", "text": PROMPT}]
    parts.append(image_part_from_item(item))

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
        temperature=0.01,
        max_tokens=1024,
    )
    content = (resp.choices[0].message.content or "").strip()
    # Normalize to NDJSON: strip code fences and keep only lines that look like JSON objects
    lines: list[str] = []
    for raw in content.splitlines():
        s = raw.strip().strip("`")
        if not s:
            continue
        if not (s.startswith("{") and s.endswith("}")):
            # ignore non-JSON lines
            continue
        lines.append(s)
    if not lines:
        # Fall back to returning raw content even if not recognized
        return content
    return "\n".join(lines)
