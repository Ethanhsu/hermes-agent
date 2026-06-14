"""NVIDIA NIM Guard — unified integration of rate limiter, circuit breaker,
request queue, and metrics for NIM API protection.

This is the main integration point. To activate NIM guards, the agent
checks `is_nim_endpoint()` and routes through this module before
making any API call to NVIDIA NIM.

Usage in run_agent.py:
    from agent.nim_guard import nim_guard

    # Before API call:
    if nim_guard.is_nim_endpoint(base_url):
        allowed, reason = nim_guard.allow_request()
        if not allowed:
            # Wait or handle accordingly
            pass

        nim_guard.before_request()
        try:
            response = make_api_call(...)
            nim_guard.after_success(response.latency_ms)
        except APIError as e:
            if e.status_code == 429:
                nim_guard.after_429()
            else:
                nim_guard.after_failure()
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from agent.nim_rate_limiter import (
    NIMRateLimiterConfig,
    get_nim_limiter,
    reset_nim_limiter,
)
from agent.nim_circuit_breaker import (
    CircuitBreakerConfig,
    get_circuit_breaker,
    reset_circuit_breaker,
)
from agent.nim_request_queue import (
    NIMRequestQueue as RequestQueue,
    Priority,
    RequestQueueConfig,
    get_request_queue,
    reset_request_queue,
)
from agent.nim_metrics import get_metrics, reset_metrics

logger = logging.getLogger(__name__)


def _is_nvidia_nim_url(base_url: str) -> bool:
    """Check if the base URL points to NVIDIA NIM."""
    return (
        "integrate.api.nvidia.com" in (base_url or "")
        or "build.nvidia.com" in (base_url or "")
    )


class NIMGuard:
    """Unified guard for NVIDIA NIM API calls.

    Combines rate limiting, circuit breaking, queuing, and metrics
    into a single interface.
    """

    def __init__(
        self,
        rate_config: NIMRateLimiterConfig | None = None,
        circuit_config: CircuitBreakerConfig | None = None,
        queue_config: RequestQueueConfig | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.rate_limiter = get_nim_limiter(rate_config)
        self.circuit_breaker = get_circuit_breaker(circuit_config)
        self.request_queue = get_request_queue(queue_config)
        self.metrics = get_metrics()

        # Periodic logging
        self._last_log_time: float = time.monotonic()
        self._log_interval: float = 30.0  # Log every 30 seconds

    def is_nim_endpoint(self, base_url: str) -> bool:
        """Check if this endpoint needs NIM guards."""
        return self.enabled and _is_nvidia_nim_url(base_url)

    def allow_request(self) -> tuple[bool, str]:
        """Check if a request is allowed to proceed.

        Returns (allowed, reason).
        """
        if not self.enabled:
            return True, "disabled"

        # Check circuit breaker first
        cb_allowed, cb_reason = self.circuit_breaker.allow_request()
        if not cb_allowed:
            self.metrics.update_circuit_breaker_state(self.circuit_breaker.state.value)
            return False, f"circuit_breaker:{cb_reason}"

        # Check rate limiter
        allowed, wait_time = self.rate_limiter.acquire()
        if not allowed:
            self.metrics.record_throttled()
            return False, f"rate_limited:wait_{wait_time:.1f}s"

        # Check if queue can accept more active requests
        if not self.request_queue.can_dispatch():
            # Need to release the rate limiter token since we can't proceed
            self.rate_limiter.release()
            return False, "queue_full"

        self.metrics.update_circuit_breaker_state(self.circuit_breaker.state.value)
        self.metrics.update_adaptive_rpm(self.rate_limiter.current_rpm)
        self.metrics.update_queue_depth(self.request_queue.depth)

        return True, "allowed"

    def wait_and_acquire(
        self, timeout: float = 60.0, priority: Priority = Priority.NORMAL
    ) -> tuple[bool, str]:
        """Wait until a request slot is available, then acquire.

        Blocks until:
        1. Circuit breaker allows
        2. Rate limiter allows
        3. Queue has a slot

        Returns (acquired, reason).
        """
        if not self.enabled:
            return True, "disabled"

        start = time.monotonic()
        while time.monotonic() - start < timeout:
            # Check circuit breaker
            cb_allowed, cb_reason = self.circuit_breaker.allow_request()
            if not cb_allowed:
                if "cooldown_remaining" in cb_reason:
                    try:
                        remaining = int(cb_reason.split("_")[-1].rstrip("s"))
                        wait = min(remaining + 1, timeout - (time.monotonic() - start))
                    except (ValueError, IndexError):
                        wait = 5.0
                    time.sleep(min(wait, 5.0))
                    continue
                time.sleep(1.0)
                continue

            # Check rate limiter
            allowed, wait_time = self.rate_limiter.acquire()
            if not allowed:
                time.sleep(min(wait_time, 2.0))
                continue

            # Wait for queue slot
            if not self.request_queue.wait_for_slot(
                timeout=max(1.0, timeout - (time.monotonic() - start))
            ):
                # Timeout — release rate limiter token
                self.rate_limiter.release()
                return False, "queue_timeout"

            self.metrics.update_queue_depth(self.request_queue.depth)
            return True, "acquired"

        return False, f"timeout_after_{timeout:.0f}s"

    def before_request(self) -> None:
        """Call before making the actual API request."""
        if not self.enabled:
            return
        self.metrics.record_request()

    def after_success(self, latency_ms: float = 0) -> None:
        """Call after a successful API response."""
        if not self.enabled:
            return
        self.metrics.record_success(latency_ms)
        self.circuit_breaker.record_success()
        self.rate_limiter.record_success(latency_ms)
        self.rate_limiter.release()
        self.request_queue.release_slot()
        self._maybe_log()

    def after_429(self, retry_after: float | None = None) -> None:
        """Call after receiving a 429 response."""
        if not self.enabled:
            return
        self.metrics.record_429()
        self.rate_limiter.record_429()
        self.circuit_breaker.record_failure(is_429=True)
        self.rate_limiter.release()
        self.request_queue.release_slot()
        self._maybe_log()

        # If Retry-After is provided, respect it
        if retry_after and retry_after > 0:
            logger.info(
                "NIM 429: respecting Retry-After=%.1fs", retry_after
            )

    def after_retry(self) -> None:
        """Call when a retry is about to be attempted."""
        if not self.enabled:
            return
        self.metrics.record_retry()

    def after_failure(self, status_code: int | None = None) -> None:
        """Call after a non-429 failure."""
        if not self.enabled:
            return
        self.metrics.record_failure()
        # Only record as circuit breaker failure for 5xx errors
        if status_code and status_code >= 500:
            self.circuit_breaker.record_failure(is_429=False)
        self.rate_limiter.release()
        self.request_queue.release_slot()
        self._maybe_log()

    def _maybe_log(self) -> None:
        """Periodically log metrics summary."""
        now = time.monotonic()
        if now - self._last_log_time >= self._log_interval:
            self._last_log_time = now
            logger.info(self.metrics.log_summary())

    def get_stats(self) -> dict:
        """Get combined statistics from all components."""
        return {
            "rate_limiter": self.rate_limiter.get_stats(),
            "circuit_breaker": self.circuit_breaker.get_stats(),
            "queue": self.request_queue.get_stats(),
            "metrics": self.metrics.snapshot().__dict__,
        }

    def reset(self) -> None:
        """Reset all components (for testing)."""
        reset_nim_limiter()
        reset_circuit_breaker()
        reset_request_queue()
        reset_metrics()
        self.rate_limiter = get_nim_limiter()
        self.circuit_breaker = get_circuit_breaker()
        self.request_queue = get_request_queue()
        self.metrics = get_metrics()


# Global singleton
_guard: NIMGuard | None = None
_guard_lock = threading.Lock()


def get_nim_guard(
    rate_config: NIMRateLimiterConfig | None = None,
    circuit_config: CircuitBreakerConfig | None = None,
    queue_config: RequestQueueConfig | None = None,
    enabled: bool = True,
) -> NIMGuard:
    """Get or create the global NIM guard."""
    global _guard
    with _guard_lock:
        if _guard is None:
            _guard = NIMGuard(
                rate_config=rate_config,
                circuit_config=circuit_config,
                queue_config=queue_config,
                enabled=enabled,
            )
        return _guard


def reset_nim_guard() -> None:
    """Reset the global guard (for testing)."""
    global _guard
    with _guard_lock:
        if _guard:
            _guard.reset()
        _guard = None


# Convenience shorthand for direct use
nim_guard = property(lambda self: get_nim_guard())
