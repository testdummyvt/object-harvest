"""Concurrency helpers (thread pool + backoff)."""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable, Iterator, TypeVar

from ..logging import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


def run_tasks(
    func: Callable[[T], R], items: Iterable[T], max_workers: int, desc: str | None = None
) -> list[R]:
    results: list[R] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {ex.submit(func, it): it for it in items}
        for fut in as_completed(future_map):
            try:
                results.append(fut.result())
            except Exception as e:  # noqa: BLE001
                logger.warning("⚠️ Task failed, skipping result: %s", e)
                continue
    return results


def backoff_sleep(base: float, attempt: int, max_sleep: float = 10.0) -> None:
    sleep_time = min(max_sleep, base * (2**attempt)) + random.random() * 0.5
    time.sleep(sleep_time)
