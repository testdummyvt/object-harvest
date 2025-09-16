"""Open-vocabulary detection (OVDet) utilities.

Components:
- ``HFDataLoader``: Iterates a Hugging Face dataset split and extracts prompts
    from ``objects.names`` (default) or ``objects.description`` when requested.
- ``OVDModel``: Wraps a Hugging Face zero-shot object detection model
    (GroundingDINO / LLMDet style) providing device auto-selection (CUDA→MPS→CPU)
    and a callable interface returning normalized detections.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import torch  # type: ignore
from transformers import (  # type: ignore
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
)
from datasets import load_dataset  # type: ignore
from dotenv import load_dotenv
from object_harvest.logging import get_logger

logger = get_logger(__name__)


load_dotenv()


__all__ = ["HFDataLoader", "OVDModel", "OVDMODEL"]

OVDMODEL = os.environ.get("OBJH_GDINO_MODEL", "iSEE-Laboratory/llmdet_large")


class HFDataLoader:
    """Iterate a HF dataset split yielding records for detection.

    Each yielded item: {"id", "path", "image", "prompt"}
    where "prompt" is a list of object labels / descriptions.
    """

    def __init__(
        self,
        dataset_id: str,
        use_desc: bool = False,  # placeholder for future extension
        use_obj_desc: bool = False,
        split: str = "train",
    ) -> None:
        self.ds = load_dataset(dataset_id, split=split)
        self.use_desc = use_desc  # currently unused
        self.use_obj_desc = use_obj_desc

    def __iter__(self):
        sel_key = "description" if self.use_obj_desc else "names"
        for idx, ex in enumerate(self.ds):
            if not isinstance(ex, dict):
                # Some streaming datasets may return other objects; skip safely
                continue
            file_name = ex.get("file_name")
            image = ex.get("image")
            if image is not None and hasattr(image, "convert"):
                try:
                    image = image.convert("RGB")
                except Exception:
                    pass
            prompt: List[str] = []
            objects_block = ex.get("objects") or {}
            if isinstance(objects_block, dict):
                seq = objects_block.get(sel_key) or []
                if isinstance(seq, list):
                    for obj in seq:
                        if isinstance(obj, str) and obj.strip():
                            prompt.append(obj.strip())
            yield {
                "id": idx,
                "path": str(file_name) if file_name else f"example-{idx}.jpg",
                "image": image,
                "prompt": prompt,
            }

    def __len__(self):  # type: ignore[override]
        # Prefer num_rows attribute if available (datasets>=2.x), else len(), else 0
        nr = getattr(self.ds, "num_rows", None)
        if isinstance(nr, int):
            return nr
        try:  # pragma: no cover - defensive
            return len(self.ds)  # type: ignore[arg-type]
        except Exception:
            return 0


class OVDModel:
    """Open-vocabulary detector wrapper.

    Parameters
    ----------
    model_id : str | None
        Hugging Face model id. Falls back to env default.
    device : str
        'auto' chooses CUDA→MPS→CPU; otherwise explicit device string.
    threshold : float
        Score threshold applied in post-processing.
    """

    def __init__(
        self,
        model_id: str | None = OVDMODEL,
        device: str = "auto",
        threshold: float = 0.3,
    ) -> None:
        if device == "auto":
            if torch.cuda.is_available():  # pragma: no cover - device availability
                self.device = "cuda"
            elif (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):  # pragma: no cover
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        mid = model_id or OVDMODEL
        self.processor = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(
            self.device
        )
        self.model.eval()
        logger.info(f"loaded detector model '{mid}' on device '{self.device}'")
        self.threshold = threshold

    def __call__(self, image, prompts: List[str]) -> List[Dict[str, Any]]:
        if not prompts:
            return []
        width, height = image.width, image.height
        inputs = self.processor(images=image, text=[prompts], return_tensors="pt").to(
            self.device
        )
        with torch.no_grad():  # pragma: no cover - inference
            outputs = self.model(**inputs)
        # Prefer grounded post-process when available
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            processed = self.processor.post_process_grounded_object_detection(
                outputs, threshold=self.threshold, target_sizes=[(height, width)]
            )[0]
        else:  # pragma: no cover - fallback path
            processed = self.processor.post_process_object_detection(
                outputs, threshold=self.threshold, target_sizes=[(height, width)]
            )[0]
        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        labels_out = processed.get("text_labels", [])
        detections: List[Dict[str, Any]] = []
        for box, score, lbl in zip(boxes, scores, labels_out):
            try:
                if hasattr(box, "tolist"):
                    x1, y1, x2, y2 = [float(x) for x in box.tolist()]
                else:
                    vals = list(box)  # type: ignore[arg-type]
                    if len(vals) != 4:
                        continue
                    x1, y1, x2, y2 = [float(x) for x in vals]
                score_val = (
                    float(score.item()) if hasattr(score, "item") else float(score)
                )
                label_str = lbl if isinstance(lbl, str) else str(lbl)
                detections.append(
                    {
                        "label": label_str,
                        "score": score_val,
                        "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2},
                    }
                )
            except Exception as e:  # pragma: no cover - skip malformed
                logger.warning(
                    f"skipping malformed detection box/score/label: {box}, {score}, {lbl}: {e}"
                )
                continue
        return detections

    # Legacy VLM detection JSON parser removed as part of cleanup.
