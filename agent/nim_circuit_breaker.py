"""NVIDIA NIM Circuit Breaker — prevent retry storms.

Monitors consecutive 429 responses from the NIM provider.
When threshold is reached, opens the circuit and pauses all
NIM traffic for a cooldown period. After cooldown, sends a
single probe request to test recovery.

States:
    CLOSED   — Normal operation, requests flow through
    OPEN     — Provider degraded, all requests blocked
    HALF_OPEN — Cooldown expired, testing with probe request
"""

from __future__ import annotations

import enum
import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class CircuitState(enum.Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreakerConfig:
    """Configuration for the circuit breaker."""
    failure_threshold: int = 3       # Consecutive 429s before opening
    cooldown_seconds: float = 60.0   # How long to stay open
    probe_max_retries: int = 3       # Max probe attempts before re-opening


class NIMCircuitBreaker:
    """Circuit breaker for NVIDIA NIM provider.

    Prevents retry storms by temporarily halting all NIM requests
    when the provider shows signs of sustained rate limiting.
    """

    def __init__(self, config: CircuitBreakerConfig | None = None):
        self.config = config or CircuitBreakerConfig()
        self._lock = threading.Lock()

        self._state: CircuitState = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._opened_at: float = 0.0
        self._probe_in_progress: bool = False
        self._probe_event = threading.Event()

        # Metrics
        self._total_trips: int = 0
        self._total_probes: int = 0
        self._total_probe_failures: int = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state (auto-transitions OPEN->HALF_OPEN after cooldown)."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.config.cooldown_seconds:
                    self._state = CircuitState.HALF_OPEN
                    self._probe_in_progress = False
                    logger.info(
                        "NIM circuit breaker: OPEN -> HALF_OPEN "
                        "(cooldown %ds expired)",
                        self.config.cooldown_seconds,
                    )
            return self._state

    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._consecutive_failures

    def allow_request(self) -> tuple[bool, str]:
        """Check if a request is allowed.

        Returns (allowed, reason).
        """
        current_state = self.state  # Auto-transitions if cooldown expired

        if current_state == CircuitState.CLOSED:
            return True, "circuit_closed"

        if current_state == CircuitState.HALF_OPEN:
            # Only one probe request at a time
            with self._lock:
                if not self._probe_in_progress:
                    self._probe_in_progress = True
                    self._probe_event.clear()
                    self._total_probes += 1
                    return True, "probe_request"
            return False, "probe_in_progress"

        # OPEN state
        current_state = self.state  # Re-check for auto-transition
        if current_state == CircuitState.OPEN:
            remaining = self.config.cooldown_seconds - (
                time.monotonic() - self._opened_at
            )
            return False, f"cooldown_remaining_{max(0, int(remaining))}s"

        return False, "circuit_open"

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Probe succeeded — close the circuit
                self._state = CircuitState.CLOSED
                self._consecutive_failures = 0
                self._probe_in_progress = False
                self._probe_event.set()
                logger.info(
                    "NIM circuit breaker: HALF_OPEN -> CLOSED "
                    "(probe succeeded, normal traffic restored)"
                )
            elif self._state == CircuitState.CLOSED:
                self._consecutive_failures = 0

    def record_failure(self, is_429: bool = True) -> None:
        """Record a failed request (429 response)."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — re-open the circuit
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._probe_in_progress = False
                self._probe_event.set()
                self._total_probe_failures += 1
                logger.warning(
                    "NIM circuit breaker: HALF_OPEN -> OPEN "
                    "(probe failed, re-cooling for %.0fs)",
                    self.config.cooldown_seconds,
                )
                return

            if self._state == CircuitState.CLOSED and is_429:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self.config.failure_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()
                    self._total_trips += 1
                    logger.warning(
                        "NIM circuit breaker: CLOSED -> OPEN "
                        "(%d consecutive 429s, cooldown %.0fs)",
                        self._consecutive_failures,
                        self.config.cooldown_seconds,
                    )

    def wait_for_probe(self, timeout: float = 65.0) -> bool:
        """Block until the current probe completes or timeout.

        Returns True if probe completed, False if timeout.
        """
        return self._probe_event.wait(timeout=timeout)

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        current = self.state
        with self._lock:
            return {
                "state": current.value,
                "consecutive_failures": self._consecutive_failures,
                "total_trips": self._total_trips,
                "total_probes": self._total_probes,
                "total_probe_failures": self._total_probe_failures,
                "probe_in_progress": self._probe_in_progress,
                "time_in_current_state": (
                    time.monotonic() - self._opened_at
                    if current in (CircuitState.OPEN, CircuitState.HALF_OPEN)
                    else 0.0
                ),
            }

    def reset(self) -> None:
        """Reset to closed state (for testing)."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._total_trips = 0
            self._total_probes = 0
            self._total_probe_failures = 0
            self._probe_in_progress = False
            self._probe_event.set()


# Global singleton
_circuit_breaker: NIMCircuitBreaker | None = None
_cb_lock = threading.Lock()


def get_circuit_breaker(
    config: CircuitBreakerConfig | None = None,
) -> NIMCircuitBreaker:
    """Get or create the global circuit breaker."""
    global _circuit_breaker
    with _cb_lock:
        if _circuit_breaker is None:
            _circuit_breaker = NIMCircuitBreaker(config)
        elif config is not None:
            _circuit_breaker.config = config
        return _circuit_breaker


def reset_circuit_breaker() -> None:
    """Reset the global circuit breaker (for testing)."""
    global _circuit_breaker
    with _cb_lock:
        if _circuit_breaker:
            _circuit_breaker.reset()
