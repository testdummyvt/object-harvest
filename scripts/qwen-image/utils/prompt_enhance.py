"""Prompt enhancement utility built on the project's AIClient.

This module provides a small helper to improve or rewrite a user-provided text
prompt using a chat-completions model behind the unified ``AIClient``.

Design notes
------------
- Accepts a configurable ``system_prompt`` parameter (defaults to empty string).
- Reuses the shared OpenAI-compatible client via ``AIClient`` for consistent
  auth and base URL handling. If a client isn't supplied, the function will
  construct one from environment variables: ``OBJH_MODEL`` and ``OBJH_API_BASE``.
- Returns the raw enhanced text from the first choice.
"""

from __future__ import annotations

from typing import Optional, List
import os

from dotenv import load_dotenv

from object_harvest.utils.clients import AIClient
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)

load_dotenv()

__all__ = ["enhance_prompt"]


def enhance_prompt(
    user_prompt: str,
    system_prompt: str = "",
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 256,
    client: Optional[AIClient] = None,
) -> str:
    """Enhance or rewrite a text prompt using the configured LLM.

    Parameters
    ----------
    user_prompt:
            The user's original text prompt to be enhanced.
    system_prompt:
            An optional system instruction to steer the enhancement. Defaults to
            an empty string. Leave empty to rely on model defaults. You can update
            this later with specific style or formatting guidance.
    model:
            Optional model name to override the default. If not provided, falls
            back to ``OBJH_MODEL`` from the environment.
    base_url:
            Optional custom API base URL for OpenAI-compatible endpoints. If not
            provided, falls back to ``OBJH_API_BASE`` from the environment.
    temperature:
            Sampling temperature for creativity. Defaults to 0.3.
    max_tokens:
            Maximum tokens to generate for the enhanced prompt.
    client:
            Optional preconstructed ``AIClient`` instance to reuse across calls.

    Returns
    -------
    str
            The enhanced prompt text returned by the model (first choice),
            whitespace-trimmed.
    """
    text = (user_prompt or "").strip()
    if not text:
        raise ValueError("user_prompt must be a non-empty string")

    llm = client or AIClient(
        model=(model or os.getenv("OBJH_MODEL", "")), base_url=base_url
    )
    if not llm.model:
        raise ValueError(
            "Model name is required. Provide `model` or set OBJH_MODEL in the environment."
        )

    messages: List[ChatCompletionMessageParam] = []
    if system_prompt and system_prompt.strip():
        system_msg: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": system_prompt,
        }
        messages.append(system_msg)
    user_msg: ChatCompletionUserMessageParam = {
        "role": "user",
        "content": text,
    }
    messages.append(user_msg)

    resp = llm.client.chat.completions.create(
        model=llm.model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    content = (resp.choices[0].message.content or "").strip()
    # Strip possible code fences if the model added them
    return content.strip("`")
