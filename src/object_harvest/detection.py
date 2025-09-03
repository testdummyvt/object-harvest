"""Detection backends and VLM detection JSON parsing/validation utilities."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from PIL import Image
from dotenv import load_dotenv

from object_harvest.logging import get_logger
from object_harvest.utils.clients import AIClient
from object_harvest.utils.images import load_image_from_item, image_part_from_item

logger = get_logger(__name__)
load_dotenv()

__all__ = [
    "run_gdino_detection",
    "run_vlm_detection",
    "parse_vlm_detections_json",
]


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _load_image(item: Dict[str, Any]) -> Image.Image:
    # Backward-compat shim; delegate to utils helper
    return load_image_from_item(item)


def run_gdino_detection(
    item: Dict[str, Any],
    labels: Optional[List[str]],
    threshold: float = 0.25,
    hf_model: Optional[str] = None,
    text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run zero-shot object detection using GroundingDINO via transformers pipeline.

    Returns list of {label, score, bbox: {xmin,ymin,xmax,ymax}} with bbox in pixel coordinates.
    """
    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:
        logger.warning(f"transformers not available for gdino: {e}")
        return []

    if not labels and not (text and text.strip()):
        logger.warning(
            "gdino backend requires either non-empty labels or a text description"
        )
        return []

    model_id = hf_model or os.environ.get(
        "OBJH_GDINO_MODEL", "IDEA-Research/grounding-dino-base"
    )
    pipe = pipeline("zero-shot-object-detection", model=model_id)

    image = _load_image(item)
    # Prefer text description if provided; otherwise use candidate labels
    if text and text.strip():
        try:
            outputs = pipe(image, text=text, threshold=threshold)
        except TypeError:
            # Fallback: some implementations expect 'candidate_labels' only; try splitting text to labels
            fallback_labels = (
                [s.strip() for s in text.split(",") if s.strip()] or labels or []
            )
            outputs = pipe(image, candidate_labels=fallback_labels, threshold=threshold)
    else:
        outputs = pipe(image, candidate_labels=labels, threshold=threshold)
    detections: List[Dict[str, Any]] = []
    for det in outputs:
        # transformers returns bbox as dict with xmin/xmax/ymin/ymax
        bbox = det.get("box") or det.get("bbox") or {}
        xmin = _as_float(bbox.get("xmin", bbox.get("x1", 0)))
        ymin = _as_float(bbox.get("ymin", bbox.get("y1", 0)))
        xmax = _as_float(bbox.get("xmax", bbox.get("x2", 0)))
        ymax = _as_float(bbox.get("ymax", bbox.get("y2", 0)))
        detections.append(
            {
                "label": det.get("label", "object"),
                "score": float(det.get("score", 0.0)),
                "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            }
        )
    return detections


_VLM_DET_PROMPT_TEMPLATE = (
    "You are an expert visual grounding system. Given an image and an optional target object list, return detections as pure JSON with a 'detections' array. "
    "Each detection must include: label (string), score (0..1), bbox as an object with pixel coordinates: xmin, ymin, xmax, ymax (numbers). Do NOT normalize. "
    "If a target list is provided, detect only those; otherwise, detect the most salient objects. Output only JSON.\n\n"
    "Targets: {targets}"
)


def run_vlm_detection(
    client: AIClient,
    item: Dict[str, Any],
    labels: Optional[List[str]],
) -> List[Dict[str, Any]]:
    from typing import cast
    from openai.types.chat import (
        ChatCompletionMessageParam,
        ChatCompletionUserMessageParam,
    )

    targets = ", ".join(labels) if labels else "(none)"
    prompt = _VLM_DET_PROMPT_TEMPLATE.format(targets=targets)

    parts: List[dict] = [{"type": "text", "text": prompt}]
    parts.append(image_part_from_item(item))

    user_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": cast(Any, parts),
    }
    messages: List[ChatCompletionMessageParam] = [user_msg]
    resp = client.client.chat.completions.create(
        model=client.model,
        messages=messages,
        temperature=0.01,
        max_tokens=800,
    )
    content = resp.choices[0].message.content or "{}"
    return parse_vlm_detections_json(content)


def _clamp01(v: float) -> float:
    try:
        fv = float(v)
    except Exception:
        return 0.0
    if fv < 0.0:
        return 0.0
    if fv > 1.0:
        return 1.0
    return fv


def parse_vlm_detections_json(content: str) -> List[Dict[str, Any]]:
    """Parse and strictly validate a VLM detection JSON string.

    Rules:
    - Top-level must be an object with key 'detections' as a list.
    - Each detection must include 'label' (str) and 'bbox' (object with numeric xmin,ymin,xmax,ymax) in pixel coordinates.
    - 'score' is optional; defaults to 0.0. Score is clamped to [0,1].
    - Detections failing validation are dropped.
    Returns a list of normalized detections: {label:str, score:float, bbox:{xmin,ymin,xmax,ymax}}.
    """
    try:
        parsed = json.loads(content)
    except Exception as e:
        logger.error(f"VLM detection JSON parse error: {e}. Raw: {content[:200]}")
        return []

    if not isinstance(parsed, dict):
        logger.warning("VLM detection JSON must be an object at top level")
        return []

    dets = parsed.get("detections")
    if not isinstance(dets, list):
        logger.warning("VLM detection JSON must contain 'detections' as a list")
        return []

    normalized: List[Dict[str, Any]] = []
    for i, d in enumerate(dets):
        if not isinstance(d, dict):
            logger.warning(f"skip detection[{i}]: not an object")
            continue
        label = d.get("label")
        if not isinstance(label, str) or not label.strip():
            logger.warning(f"skip detection[{i}]: missing/invalid 'label'")
            continue
        bbox = d.get("bbox")
        if not isinstance(bbox, dict):
            logger.warning(f"skip detection[{i}]: missing/invalid 'bbox'")
            continue
        # Accept numeric-like values (e.g., strings) for robustness; keep pixel coordinates as-is (no normalization)
        try:
            xmin = _as_float(bbox.get("xmin", bbox.get("x1")))
            ymin = _as_float(bbox.get("ymin", bbox.get("y1")))
            xmax = _as_float(bbox.get("xmax", bbox.get("x2")))
            ymax = _as_float(bbox.get("ymax", bbox.get("y2")))
        except Exception:
            logger.warning(f"skip detection[{i}]: non-numeric bbox values")
            continue
        # If any are None after fallback, skip
        if any(v is None for v in [xmin, ymin, xmax, ymax]):
            logger.warning(f"skip detection[{i}]: missing bbox values")
            continue

        score_val = d.get("score", 0.0)
        score = _clamp01(score_val)

        normalized.append(
            {
                "label": label,
                "score": score,
                "bbox": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax},
            }
        )

    return normalized
