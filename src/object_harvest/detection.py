"""Detection utilities and VLM detection JSON parsing/validation helpers.

Currently supports Hugging Face zero-shot object detectors (e.g., GroundingDINO/LLMDet)
via ``transformers``. The VLM-backed detection route has been removed from the CLI,
but the JSON parser remains available for compatibility/tests.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from dotenv import load_dotenv

from object_harvest.logging import get_logger
from object_harvest.utils.images import load_image_from_item

logger = get_logger(__name__)
load_dotenv()

__all__ = [
    "run_gdino_detection",
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


_MODEL_CACHE: dict[str, Tuple[Any, Any, str]] = {}
_MODEL_LOCK: Any = None


def _get_model_lock():
    global _MODEL_LOCK
    if _MODEL_LOCK is None:
        import threading as _threading

        _MODEL_LOCK = _threading.Lock()
    return _MODEL_LOCK


def _load_hf_ovd(model_id: str) -> Tuple[Any, Any, str]:
    """Load and cache an HF zero-shot OD model + processor. Returns (processor, model, device)."""
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
        )
    except Exception as e:
        logger.warning(f"transformers/torch not available for detection: {e}")
        raise

    lock = _get_model_lock()
    with lock:
        cached = _MODEL_CACHE.get(model_id)
        if cached is not None:
            return cached

        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        model = model.to(device)
        model.eval()
        _MODEL_CACHE[model_id] = (processor, model, device)
        return _MODEL_CACHE[model_id]


def run_gdino_detection(
    item: Dict[str, Any],
    labels: Optional[List[str]],
    threshold: float = 0.25,
    hf_model: Optional[str] = None,
    text: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run zero-shot object detection using HF models (GroundingDINO/LLMDet).

    Uses ``AutoProcessor`` and ``AutoModelForZeroShotObjectDetection`` with model-specific
    post-processing if available. Returns list of normalized detections with pixel bboxes.
    """
    model_id = hf_model or os.environ.get(
        "OBJH_GDINO_MODEL", "iSEE-Laboratory/llmdet_large"
    )

    try:
        processor, model, device = _load_hf_ovd(model_id)
    except Exception:
        return []

    if not labels and not (text and str(text).strip()):
        logger.warning("detection requires non-empty --objects or --text")
        return []

    image = _load_image(item)
    width, height = image.width, image.height

    # Build text prompts: list[list[str]] per batch; single image => outer list of one
    prompt_items: List[str] = []
    if labels:
        prompt_items.extend([str(x) for x in labels if str(x).strip()])
    if text and str(text).strip():
        # If labels absent, derive from comma-separated text; else treat as additional hint
        if not prompt_items:
            prompt_items = [s.strip() for s in str(text).split(",") if s.strip()]
        else:
            prompt_items.append(str(text).strip())
    if not prompt_items:
        return []

    try:
        import torch  # type: ignore

        inputs = processor(images=image, text=[prompt_items], return_tensors="pt")
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        # Prefer model-specific grounded post-process if available
        results = None
        if hasattr(processor, "post_process_grounded_object_detection"):
            results = processor.post_process_grounded_object_detection(
                outputs, threshold=threshold, target_sizes=[(height, width)]
            )
        elif hasattr(processor, "post_process_object_detection"):
            results = processor.post_process_object_detection(
                outputs, threshold=threshold, target_sizes=[(height, width)]
            )
        else:
            # As a fallback, try transformers pipeline API (rarely needed)
            try:
                from transformers import pipeline  # type: ignore

                pipe = pipeline("zero-shot-object-detection", model=model_id)
                outputs = pipe(
                    image, candidate_labels=prompt_items, threshold=threshold
                )
                detections: List[Dict[str, Any]] = []
                for det in outputs:
                    bbox = det.get("box") or det.get("bbox") or {}
                    xmin = _as_float(bbox.get("xmin", bbox.get("x1", 0)))
                    ymin = _as_float(bbox.get("ymin", bbox.get("y1", 0)))
                    xmax = _as_float(bbox.get("xmax", bbox.get("x2", 0)))
                    ymax = _as_float(bbox.get("ymax", bbox.get("y2", 0)))
                    detections.append(
                        {
                            "label": det.get("label", "object"),
                            "score": float(det.get("score", 0.0)),
                            "bbox": {
                                "xmin": xmin,
                                "ymin": ymin,
                                "xmax": xmax,
                                "ymax": ymax,
                            },
                        }
                    )
                return detections
            except Exception as e:  # pragma: no cover - best-effort fallback
                logger.warning(f"pipeline fallback failed: {e}")
                return []

        if not results:
            return []
        result = results[0]
        boxes = result.get("boxes") or result.get("boxes_xyxy") or []
        scores = result.get("scores") or []
        labels_out = result.get("labels") or []

        detections: List[Dict[str, Any]] = []
        for box, score, lbl in zip(boxes, scores, labels_out):
            try:
                # box may be tensor; convert to Python list of floats
                if hasattr(box, "tolist"):
                    box_vals = [float(x) for x in box.tolist()]
                else:
                    box_vals = [float(x) for x in list(box)]  # type: ignore[arg-type]
                if len(box_vals) == 4:
                    x1, y1, x2, y2 = box_vals
                else:
                    # Unexpected shape; skip
                    continue
                label_str = lbl if isinstance(lbl, str) else str(lbl)
                if hasattr(score, "item"):
                    score_val = float(score.item())
                else:
                    score_val = float(score)
                detections.append(
                    {
                        "label": label_str,
                        "score": score_val,
                        "bbox": {
                            "xmin": x1,
                            "ymin": y1,
                            "xmax": x2,
                            "ymax": y2,
                        },
                    }
                )
            except Exception:
                continue
        return detections
    except Exception as e:
        logger.error(f"detection failed: {e}")
        return []


## VLM detection removed from CLI; JSON parser retained below


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
