from __future__ import annotations

import os
from dataclasses import dataclass
import json
import random
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI

from object_harvest.logging import get_logger

logger = get_logger(__name__)
load_dotenv()


PROMPT_TEMPLATE = (
    "You are a concise captioning assistant. Using ONLY the following objects, write a single one-line vivid scene description that naturally includes all of them without listing format: {objects}. "
    "Avoid meta phrases like 'in this image' or 'this picture shows'. Then output STRICT JSON with two keys:\n"
    "{{\n"
    "  \"describe\": \"<the one-line description>\",\n"
    "  \"objects\": [{{\"<object_1>\": \"<object_1 phrasing as used within the description>\"}}, {{\"<object_2>\": \"<object_2 phrasing as used within the description>\"}}]\n"
    "}}\n"
    "Rules:\n- Use the exact object names from the provided list as keys.\n- Keep object descriptions concise and consistent with the wording in the main description.\n- Output JSON only (no backticks, no extra text)."
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


def synthesize_one_line(
    objects: List[str],
    n: int,
    model: str,
    base_url: str | None,
    client: "LLMClient | None" = None,
) -> Dict:
    """Generate a one-line description and per-object descriptions for up to N provided objects.

    Returns: { "describe": str, "objects": [ {object: description}, ... ] }
    """
    cleaned = [o.strip() for o in objects if str(o).strip()]
    if not cleaned:
        raise ValueError("No objects provided")
    # Random sample up to N objects; if n <= 0 or n >= len(cleaned), use all
    if n <= 0 or n >= len(cleaned):
        chosen = list(cleaned)
    else:
        chosen = random.sample(cleaned, k=n)

    prompt = PROMPT_TEMPLATE.format(objects=", ".join(chosen))
    llm = client or LLMClient(model=model, base_url=base_url)

    resp = llm.client.chat.completions.create(
        model=llm.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=200,
    )
    raw = (resp.choices[0].message.content or "").strip()
    # Strip possible code fences and attempt JSON parse
    cleaned = raw.strip().strip("`")
    
    try:
        data = json.loads(cleaned)
    except Exception:
        # Fallback: fabricate structure if model didn't return JSON
        text = raw.replace("\n", " ")
        return {
            "describe": text,
            "objects": [{name: name} for name in chosen],
        }

    # Validate and normalize keys
    description = str(data.get("describe", "")).strip().replace("\n", " ")
    obj_list = data.get("objects", [])
    objs: List[Dict[str, str]] = []
    if isinstance(obj_list, list):
        for entry in obj_list:
            if isinstance(entry, dict) and entry:
                k, v = next(iter(entry.items()))
                objs.append({str(k): str(v)})

    # Ensure only requested objects appear as keys; preserve provided order where possible
    chosen_set = set(chosen)
    filtered: List[Dict[str, str]] = []
    seen = set()
    for entry in objs:
        k = next(iter(entry.keys()))
        if k in chosen_set and k not in seen:
            filtered.append(entry)
            seen.add(k)
    # Add any missing objects with a minimal echo description
    for name in chosen:
        if name not in seen:
            filtered.append({name: name})

    return {"describe": description, "objects": filtered}
