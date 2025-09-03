from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

__all__ = ["AIClient"]


@dataclass
class AIClient:
    """Unified OpenAI-compatible chat client for text and vision prompts.

    Wraps the OpenAI SDK 1.x client and stores a model name plus optional base_url.
    """

    model: str
    base_url: str | None = None

    def __post_init__(self) -> None:
        self.client = OpenAI(
            api_key=os.environ.get("OBJH_API_KEY"),
            base_url=self.base_url or os.environ.get("OBJH_API_BASE"),
        )
        self.provider = (
            "openai"
            if not (self.base_url or os.environ.get("OBJH_API_BASE"))
            else "custom"
        )
