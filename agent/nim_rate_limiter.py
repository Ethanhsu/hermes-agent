"""NVIDIA NIM Rate Limiter — global RPM guard with burst protection.

Enforces safe operating limits for NVIDIA NIM API calls to prevent
HTTP 429 errors. Works as a global gate: every NIM-bound request must
acquire a token before dispatch.

Key parameters (safe defaults, can be overridden via config):
    - global_rpm: 25 (safe operating limit)
    - max_concurrent: 2 (prevent self-DDoS)
    - burst_limit: 5 (single-burst ceiling)
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class NIMRateLimiterConfig:
    """Configuration for NIM rate limiting."""
    global_rpm: int = 25
    max_concurrent_requests: int = 2
    burst_limit: int = 5
    target_latency_ms: int = 1500
    max_rpm_override: int = 30  # Hard ceiling without explicit override

    # Adaptive throttle settings
    adaptive_enabled: bool = True
    throttle_down_factor: float = 0.8  # Reduce to 80% on 429 spike
    throttle_up_factor: float = 1.1    # Increase by 10% on stability
    stability_window_seconds: int = 180  # 3 minutes of stability needed
    min_rpm: int = 5  # Floor — never go below this


class _SlidingWindowCounter:
    """Sliding window counter for rate limiting.

    Divides a time window into sub-windows for finer granularity.
    Combines exact counts in the current sub-window with a weighted
    count from the previous sub-window.
    """

    def __init__(self, window_seconds: float = 60.0, max_tokens: int = 25):
        self.window_seconds = window_seconds
        self.max_tokens = max_tokens
        self.num_subwindows = 10
        self.subwindow_seconds = window_seconds / self.num_subwindows

        self._lock = threading.RLock()
        # Ring buffer of subwindow counts
        self._counts: list[int] = [0] * self.num_subwindows
        self._window_start_times: list[float] = [time.monotonic()] * self.num_subwindows
        self._current_index = 0

    def _advance_if_needed(self) -> None:
        now = time.monotonic()
        current_window_time = self._window_start_times[self._current_index]
        elapsed = now - current_window_time

        if elapsed >= self.subwindow_seconds:
            # Advance to next subwindow
            self._counts[self._current_index] = 0
            self._window_start_times[self._current_index] = now
            self._current_index = (self._current_index + 1) % self.num_subwindows

    def current_count(self) -> float:
        """Get the current sliding window count (weighted estimate)."""
        with self._lock:
            self._advance_if_needed()
            now = time.monotonic()

            # Current subwindow count
            current_count = self._counts[self._current_index]

            # Weighted count from previous subwindow
            prev_index = (self._current_index - 1) % self.num_subwindows
            prev_elapsed = now - self._window_start_times[prev_index]
            prev_weight = max(0.0, 1.0 - prev_elapsed / self.subwindow_seconds)
            prev_count = self._counts[prev_index] * prev_weight

            return current_count + prev_count

    def try_acquire(self) -> bool:
        """Try to acquire a token. Returns True if allowed."""
        with self._lock:
            self._advance_if_needed()
            current = self.current_count()
            if current < self.max_tokens:
                self._counts[self._current_index] += 1
                return True
            return False

    def time_until_available(self) -> float:
        """Estimate seconds until a token becomes available."""
        with self._lock:
            self._advance_if_needed()
            current = self.current_count()
            if current < self.max_tokens:
                return 0.0

            # Calculate tokens needed to free up
            excess = current - self.max_tokens
            # Tokens decay as subwindows expire
            tokens_per_second = self.max_tokens / self.window_seconds
            if tokens_per_second > 0:
                return excess / tokens_per_second
            return self.subwindow_seconds

    def set_max_tokens(self, new_max: int) -> None:
        """Dynamically adjust the max tokens (for adaptive throttling)."""
        with self._lock:
            self.max_tokens = max(1, new_max)


class NIMRateLimiter:
    """Global rate limiter for NVIDIA NIM requests.

    Provides:
    - Sliding window RPM enforcement
    - Burst protection
    - Concurrent request limiting
    - Adaptive throttling
    """

    def __init__(self, config: NIMRateLimiterConfig | None = None):
        self.config = config or NIMRateLimiterConfig()
        self._lock = threading.Lock()

        # Sliding window rate limiter
        self._counter = _SlidingWindowCounter(
            window_seconds=60.0,
            max_tokens=self.config.global_rpm,
        )

        # Burst limiter: track requests in a short window
        self._burst_window_seconds = 2.0
        self._burst_timestamps: list[float] = []
        self._burst_lock = threading.Lock()

        # Concurrent request tracking
        self._concurrent_requests: int = 0
        self._concurrent_sem = threading.Semaphore(
            self.config.max_concurrent_requests
        )

        # Adaptive throttle state
        self._adaptive_rpm = self.config.global_rpm
        self._last_429_time: float = 0.0
        self._last_success_time: float = time.monotonic()
        self._consecutive_429_count: int = 0
        self._stability_start: float = time.monotonic()

        # Metrics
        self._total_requests: int = 0
        self._throttled_requests: int = 0
        self._total_wait_time: float = 0.0

    @property
    def current_rpm(self) -> int:
        """Current effective RPM limit (after adaptive adjustments)."""
        return self._adaptive_rpm

    @property
    def active_concurrent(self) -> int:
        """Approximate number of active concurrent requests."""
        # Semaphore doesn't expose this directly; estimate from state
        return self._concurrent_requests

    def acquire(self, timeout: float = 60.0) -> tuple[bool, float]:
        """Acquire permission to make a NIM request.

        Returns (allowed, wait_time).
        If allowed is False, wait_time indicates how long to wait.
        """
        with self._lock:
            self._total_requests += 1

        # Check burst limit
        with self._burst_lock:
            now = time.monotonic()
            # Clean old timestamps
            self._burst_timestamps = [
                t for t in self._burst_timestamps
                if now - t < self._burst_window_seconds
            ]
            if len(self._burst_timestamps) >= self.config.burst_limit:
                oldest = min(self._burst_timestamps)
                wait = self._burst_window_seconds - (now - oldest) + 0.1
                with self._lock:
                    self._throttled_requests += 1
                    self._total_wait_time += wait
                return False, wait

        # Check sliding window RPM
        wait_time = self._counter.time_until_available()
        if wait_time > 0:
            with self._lock:
                self._throttled_requests += 1
                self._total_wait_time += wait_time
            return False, wait_time

        # Try to acquire from sliding window
        if not self._counter.try_acquire():
            wait_time = self._counter.time_until_available()
            with self._lock:
                self._throttled_requests += 1
                self._total_wait_time += wait_time
            return False, wait_time

        # Record burst timestamp
        with self._burst_lock:
            self._burst_timestamps.append(time.monotonic())

        # Acquire concurrent semaphore (non-blocking)
        acquired = self._concurrent_sem.acquire(timeout=0)
        if not acquired:
            with self._lock:
                self._throttled_requests += 1
            # Release the rate limit token since we can't proceed
            return False, 1.0

        # Track concurrent count
        with self._lock:
            self._concurrent_requests += 1

        # Record success
        with self._lock:
            self._consecutive_429_count = 0
            self._last_success_time = time.monotonic()
            self._stability_start = time.monotonic()

        return True, 0.0

    def release(self) -> None:
        """Release a concurrent request slot."""
        self._concurrent_sem.release()
        with self._lock:
            self._concurrent_requests = max(0, self._concurrent_requests - 1)

    def record_429(self) -> None:
        """Record a 429 response for adaptive throttling."""
        with self._lock:
            self._last_429_time = time.monotonic()
            self._consecutive_429_count += 1
            self._stability_start = time.monotonic()  # Reset stability window

        # Adaptive throttle down
        if self.config.adaptive_enabled:
            self._adaptive_rpm = max(
                self.config.min_rpm,
                int(self._adaptive_rpm * self.config.throttle_down_factor),
            )
            self._counter.set_max_tokens(self._adaptive_rpm)
            logger.info(
                "NIM adaptive throttle: RPM reduced to %d (%d consecutive 429s)",
                self._adaptive_rpm,
                self._consecutive_429_count,
            )

    def record_success(self, latency_ms: float = 0) -> None:
        """Record a successful response for adaptive throttling."""
        with self._lock:
            self._last_success_time = time.monotonic()

        if not self.config.adaptive_enabled:
            return

        # Check stability window
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._stability_start
            # Only consider throttling up if we've been stable
            if elapsed >= self.config.stability_window_seconds:
                if self._adaptive_rpm < self.config.global_rpm:
                    new_rpm = min(
                        self.config.global_rpm,
                        int(self._adaptive_rpm * self.config.throttle_up_factor),
                    )
                    if new_rpm > self._adaptive_rpm:
                        self._adaptive_rpm = new_rpm
                        self._counter.set_max_tokens(self._adaptive_rpm)
                        logger.info(
                            "NIM adaptive throttle: RPM increased to %d (stable for %.0fs)",
                            self._adaptive_rpm,
                            elapsed,
                        )

    def reset_stability(self) -> None:
        """Reset stability window (e.g., after a 429)."""
        with self._lock:
            self._stability_start = time.monotonic()

    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self._lock:
            return {
                "adaptive_rpm": self._adaptive_rpm,
                "config_rpm": self.config.global_rpm,
                "consecutive_429_count": self._consecutive_429_count,
                "last_429_ago": time.monotonic() - self._last_429_time if self._last_429_time else None,
                "last_success_ago": time.monotonic() - self._last_success_time,
                "total_requests": self._total_requests,
                "throttled_requests": self._throttled_requests,
                "avg_wait_time": (
                    self._total_wait_time / self._throttled_requests
                    if self._throttled_requests > 0
                    else 0.0
                ),
            }

    def reset(self) -> None:
        """Reset all state (for testing)."""
        with self._lock:
            self._adaptive_rpm = self.config.global_rpm
            self._consecutive_429_count = 0
            self._total_requests = 0
            self._throttled_requests = 0
            self._total_wait_time = 0.0
        self._counter.set_max_tokens(self.config.global_rpm)


# Global singleton — shared across all agent instances
_nim_limiter: NIMRateLimiter | None = None
_limiter_lock = threading.Lock()


def get_nim_limiter(config: NIMRateLimiterConfig | None = None) -> NIMRateLimiter:
    """Get or create the global NIM rate limiter."""
    global _nim_limiter
    with _limiter_lock:
        if _nim_limiter is None:
            _nim_limiter = NIMRateLimiter(config)
        elif config is not None:
            # Reconfigure if new config provided
            _nim_limiter.config = config
            _nim_limiter._counter.set_max_tokens(config.global_rpm)
        return _nim_limiter


def reset_nim_limiter() -> None:
    """Reset the global limiter (for testing)."""
    global _nim_limiter
    with _limiter_lock:
        if _nim_limiter:
            _nim_limiter.reset()
