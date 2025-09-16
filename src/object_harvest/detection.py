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

import torch
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)
from datasets import load_dataset

load_dotenv()


__all__ = [
    "run_gdino_detection",
    "parse_vlm_detections_json",
]

OVDMODEL = os.environ.get(
    "OBJH_GDINO_MODEL", "iSEE-Laboratory/llmdet_large"
)

class HFDataLoader:
    def __init__(self, dataset_id: str, use_desc: bool = False, use_obj_desc: bool = False, split: str = "train") -> None:

        self.ds = load_dataset(dataset_id, split=split)
        self.use_desc = use_desc #TODO: Not supported yet
        self.use_obj_desc = use_obj_desc

    def __iter__(self):
        
        sel_key = "description" if self.use_obj_desc else "names"
        count = 0
        for ex in self.ds:
            file_name = ex.get("file_name", None)
            image = ex.get("image", None)
            if image is not None:
                image = image.convert("RGB")
            prompt = []
            for obj in ex.get("objects", {}).get(sel_key, []):
                if isinstance(obj, str) and obj.strip():
                    prompt.append(obj.strip())
            yield {"id": count, "path": str(file_name), "image": image, "prompt": prompt}
            count += 1

    def __len__(self):
        return len(self.ds)

class OVDModel:
    def __init__(self, model_id: str | None = OVDMODEL, device: str = "auto", threshold: float = 0.3) -> None:
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info(f"loaded detector model '{model_id}' on device '{self.device}'")
        self.threshold = threshold

    def __call__(self, image, prompts) -> List[Dict[str, Any]]:
        width, height = image.width, image.height
        inputs = self.processor(images=image, text=[prompts], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs, threshold=self.threshold, target_sizes=[(height, width)]
        )[0]
        
        if not results:
            return []

        boxes = results.get("boxes", [])
        scores = results.get("scores", [])
        labels_out = results.get("labels", [])

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
            except Exception as e:
                logger.exception(f"error processing detection: {e}")
        return detections
