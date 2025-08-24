from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from object_harvest.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


PROMPT_TEMPLATE = (
    "You are a concise captioning assistant. Using ONLY the following objects, write a single one-line vivid scene description that naturally includes all of them without listing format: {objects}. "
    "Avoid meta phrases like 'in this image' or 'this picture shows'."
)


@dataclass
class LLMClient:
    model: str
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.client = OpenAI(
            api_key=os.environ.get("OBJH_API_KEY"),
            base_url=self.base_url or os.environ.get("OBJH_API_BASE"),
        )


def synthesize_one_line(objects: List[str], n: int, model: str, base_url: str | None) -> Dict:
    """Generate a one-line description that uses up to N provided objects.

    Returns: { "objects_used": [str], "description": str }
    """
    cleaned = [o.strip() for o in objects if str(o).strip()]
    if not cleaned:
        raise ValueError("No objects provided")
    chosen = cleaned[: n if n > 0 else len(cleaned)]

    prompt = PROMPT_TEMPLATE.format(objects=", ".join(chosen))
    client = LLMClient(model=model, base_url=base_url)

    resp = client.client.chat.completions.create(
        model=client.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=200,
    )
    text = (resp.choices[0].message.content or "").strip().replace("\n", " ")
    return {"objects_used": chosen, "description": text}
