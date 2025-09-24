import os
import threading
import time
from typing import List, Optional

from openai import OpenAI


_last_call_time = 0.0
_lock = threading.Lock()


def load_objects(objects_file: Optional[str], objects_list: Optional[str]) -> List[str]:
    """Load objects from file or list.

    Args:
        objects_file: Path to text file with one object per line.
        objects_list: Comma-separated list of objects.

    Returns:
        List of object strings.
    """
    objects = []
    if objects_file:
        with open(objects_file, "r") as f:
            objects.extend(line.strip() for line in f if line.strip())
    if objects_list:
        objects.extend(obj.strip() for obj in objects_list.split(",") if obj.strip())
    return objects


def setup_llm_client(base_url: str, api_key: Optional[str]) -> OpenAI:
    """Setup OpenAI client for LLM API.

    Args:
        base_url: Base URL for the API.
        api_key: API key, or None to use env var.

    Returns:
        Configured OpenAI client.
    """
    api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("API key not provided. Set OPENROUTER_API_KEY env var or use --api-key.")
    return OpenAI(base_url=base_url, api_key=api_key)


def rate_limited_call(
    client: OpenAI,
    model: str,
    messages: List[dict[str, str]],
    interval: float,
) -> str:
    """Make a rate-limited LLM call.

    Args:
        client: OpenAI client.
        model: Model name.
        messages: Messages for the chat completion.
        interval: Minimum seconds between calls.

    Returns:
        Response content.
    """
    global _last_call_time
    with _lock:
        now = time.time()
        elapsed = now - _last_call_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _last_call_time = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages,  # type: ignore
        temperature=0.7,  # For creativity
    )
    content = response.choices[0].message.content
    if content is None:
        raise ValueError("No content in response")
    return content.strip()