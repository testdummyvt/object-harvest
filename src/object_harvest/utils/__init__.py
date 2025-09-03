from __future__ import annotations

import threading
import time
import json

from typing import Any


class RateLimiter:
    """Simple RPM-based limiter shared across threads.

    Uses a sliding window; allows up to `rpm` acquisitions per 60 seconds.
    """

    def __init__(self, rpm: int) -> None:
        self.rpm = max(0, rpm)
        self._lock = threading.Lock()
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        now = time.time()
        with self._lock:
            # drop timestamps older than 60s
            cutoff = now - 60.0
            self._timestamps = [t for t in self._timestamps if t > cutoff]
            if len(self._timestamps) >= self.rpm:
                # sleep until the earliest timestamp falls out of window
                sleep_for = self._timestamps[0] + 60.0 - now
                if sleep_for > 0:
                    time.sleep(sleep_for)
                # after sleep, prune again
                now = time.time()
                cutoff = now - 60.0
                self._timestamps = [t for t in self._timestamps if t > cutoff]
            self._timestamps.append(time.time())


def append_and_maybe_flush_jsonl(
    rec: dict[str, Any],
    fh,
    lock: threading.Lock,
    batch: list[dict[str, Any]],
    batch_size: int,
) -> None:
    """Append a JSON record to an in-memory batch and flush to fh when size is reached.

    Thread-safe: acquires the provided lock during append/flush.
    """
    with lock:
        batch.append(rec)
        if batch_size > 0 and len(batch) >= batch_size:
            for r in batch:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            batch.clear()


def flush_remaining_jsonl(
    fh, lock: threading.Lock, batch: list[dict[str, Any]]
) -> None:
    """Flush any remaining records in the in-memory batch to fh.

    Thread-safe: acquires the provided lock during flush.
    """
    if not batch:
        return
    with lock:
        if batch:
            for r in batch:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            batch.clear()


def update_tqdm_gpm(pbar, start_time: float, successes: int) -> None:
    """Update tqdm bar and display generations-per-minute based on successes and start_time."""
    elapsed_min = max((time.time() - start_time) / 60.0, 1e-6)
    gpm = successes / elapsed_min if elapsed_min > 0 else 0.0
    pbar.update(1)
    pbar.set_postfix(gpm=f"{gpm:.1f}")
