"""object_harvest package

High-level goal: Ingest images, query OpenAI-compatible Vision Language Models (VLMs) to obtain
(a) list of objects, (b) optional bounding boxes, and write structured JSONL lines.

Public entry points kept minimal. Most users interact through the CLI (`object-harvest`).
"""
from .schemas import ObjectItem, BoxItem, ImageRecord, RunConfig  # re-export core models

__all__ = [
    "ObjectItem",
    "BoxItem",
    "ImageRecord",
    "RunConfig",
]
