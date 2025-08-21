from __future__ import annotations

import threading
import time


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
