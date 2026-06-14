"""NVIDIA NIM Observability Metrics — request tracking and reporting.

Collects metrics for NIM API calls:
    - nim_requests_total
    - nim_429_total
    - nim_retry_total
    - nim_queue_depth
    - nim_avg_latency
    - nim_circuit_breaker_state

All metrics are thread-safe and written to logs periodically.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of NIM metrics."""
    timestamp: float
    requests_total: int
    requests_429: int
    requests_retry: int
    requests_success: int
    requests_failed: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    queue_depth: int
    circuit_breaker_state: str
    adaptive_rpm: int
    throttle_ratio: float  # throttled / total


class NIMMetrics:
    """Metrics collector for NVIDIA NIM integration.

    Thread-safe, in-memory metrics with exponential decay for latency.
    """

    def __init__(self, window_seconds: float = 300.0):
        self._lock = threading.Lock()
        self._window = window_seconds

        # Counters
        self._requests_total: int = 0
        self._requests_429: int = 0
        self._requests_retry: int = 0
        self._requests_success: int = 0
        self._requests_failed: int = 0
        self._requests_throttled: int = 0

        # Latency tracking (circular buffer)
        self._latencies: deque[tuple[float, float]] = deque()  # (timestamp, ms)
        self._max_latencies: int = 10000

        # Circuit breaker state
        self._cb_state: str = "closed"

        # Adaptive RPM
        self._adaptive_rpm: int = 25

        # Queue depth
        self._queue_depth: int = 0

    def record_request(self) -> None:
        """Record a new request being sent."""
        with self._lock:
            self._requests_total += 1

    def record_success(self, latency_ms: float) -> None:
        """Record a successful response with latency."""
        with self._lock:
            self._requests_success += 1
            now = time.monotonic()
            if len(self._latencies) < self._max_latencies:
                self._latencies.append((now, latency_ms))

    def record_429(self) -> None:
        """Record a 429 rate limit response."""
        with self._lock:
            self._requests_429 += 1
            self._requests_failed += 1

    def record_retry(self) -> None:
        """Record a retry attempt."""
        with self._lock:
            self._requests_retry += 1

    def record_failure(self) -> None:
        """Record a non-429 failure."""
        with self._lock:
            self._requests_failed += 1

    def record_throttled(self) -> None:
        """Record a request that was throttled by the rate limiter."""
        with self._lock:
            self._requests_throttled += 1

    def update_circuit_breaker_state(self, state: str) -> None:
        """Update circuit breaker state."""
        with self._lock:
            self._cb_state = state

    def update_adaptive_rpm(self, rpm: int) -> None:
        """Update current adaptive RPM."""
        with self._lock:
            self._adaptive_rpm = rpm

    def update_queue_depth(self, depth: int) -> None:
        """Update current queue depth."""
        with self._lock:
            self._queue_depth = depth

    def get_latency_percentiles(self) -> tuple[float, float, float]:
        """Get avg, p95, p99 latencies.

        Returns (avg_ms, p95_ms, p99_ms).
        """
        with self._lock:
            # Prune old entries
            cutoff = time.monotonic() - self._window
            while self._latencies and self._latencies[0][0] < cutoff:
                self._latencies.popleft()

            if not self._latencies:
                return (0.0, 0.0, 0.0)

            latencies = sorted(lat for _, lat in self._latencies)
            n = len(latencies)
            avg = sum(latencies) / n
            p95 = latencies[int(n * 0.95)] if n > 1 else latencies[0]
            p99 = latencies[int(n * 0.99)] if n > 1 else latencies[0]
            return (avg, p95, p99)

    def snapshot(self) -> MetricsSnapshot:
        """Take a point-in-time snapshot of all metrics."""
        avg, p95, p99 = self.get_latency_percentiles()
        with self._lock:
            throttle_ratio = (
                self._requests_throttled / self._requests_total
                if self._requests_total > 0
                else 0.0
            )
            return MetricsSnapshot(
                timestamp=time.time(),
                requests_total=self._requests_total,
                requests_429=self._requests_429,
                requests_retry=self._requests_retry,
                requests_success=self._requests_success,
                requests_failed=self._requests_failed,
                avg_latency_ms=avg,
                p95_latency_ms=p95,
                p99_latency_ms=p99,
                queue_depth=self._queue_depth,
                circuit_breaker_state=self._cb_state,
                adaptive_rpm=self._adaptive_rpm,
                throttle_ratio=throttle_ratio,
            )

    def log_summary(self) -> str:
        """Generate a log-friendly summary."""
        snap = self.snapshot()
        lines = [
            "NIM Metrics Summary:",
            f"  Total requests: {snap.requests_total}",
            f"  Success: {snap.requests_success}",
            f"  429 count: {snap.requests_429}",
            f"  Retries: {snap.requests_retry}",
            f"  Failed: {snap.requests_failed}",
            f"  Throttled: {self._requests_throttled}",
            f"  Avg latency: {snap.avg_latency_ms:.0f}ms",
            f"  P95 latency: {snap.p95_latency_ms:.0f}ms",
            f"  P99 latency: {snap.p99_latency_ms:.0f}ms",
            f"  Queue depth: {snap.queue_depth}",
            f"  Circuit breaker: {snap.circuit_breaker_state}",
            f"  Adaptive RPM: {snap.adaptive_rpm}",
            f"  Throttle ratio: {snap.throttle_ratio:.1%}",
        ]
        if snap.requests_total > 0:
            rate_429 = snap.requests_429 / snap.requests_total * 100
            lines.append(f"  429 rate: {rate_429:.1f}%")
            if rate_429 >= 1:
                lines.append("  WARNING: 429 rate exceeds 1% threshold!")
        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all metrics (for testing)."""
        with self._lock:
            self._requests_total = 0
            self._requests_429 = 0
            self._requests_retry = 0
            self._requests_success = 0
            self._requests_failed = 0
            self._requests_throttled = 0
            self._latencies.clear()
            self._cb_state = "closed"
            self._adaptive_rpm = 25
            self._queue_depth = 0


# Global singleton
_metrics: NIMMetrics | None = None
_metrics_lock = threading.Lock()


def get_metrics(window_seconds: float = 300.0) -> NIMMetrics:
    """Get or create the global metrics collector."""
    global _metrics
    with _metrics_lock:
        if _metrics is None:
            _metrics = NIMMetrics(window_seconds)
        return _metrics


def reset_metrics() -> None:
    """Reset the global metrics (for testing)."""
    global _metrics
    with _metrics_lock:
        if _metrics:
            _metrics.reset()
