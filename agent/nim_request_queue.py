"""NVIDIA NIM Request Queue — centralized FIFO queue for NIM API calls.

All NIM-bound requests must pass through this queue before dispatch.
Prevents uncontrolled concurrent access and ensures requests are
processed in order with proper spacing.

Priority levels:
    - high: User-facing requests (chat, tools)
    - normal: Agent planning steps
    - low: Memory operations, background indexing
"""

from __future__ import annotations

import enum
import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    _Callback = Callable[[], Any]
else:
    _Callback = Callable  # type: ignore[misc]

logger = logging.getLogger(__name__)


class Priority(enum.IntEnum):
    HIGH = 0      # User-facing requests
    NORMAL = 1    # Agent planning steps
    LOW = 2       # Memory, background indexing


@dataclass(order=False)
class QueuedRequest:
    """A request waiting in the queue."""
    priority: Priority = field(default=Priority.NORMAL)
    sequence: int = field(default=0)
    callback: _Callback | None = field(default=None, compare=False)
    context: dict | None = field(default=None, compare=False)
    enqueued_at: float = field(default_factory=time.monotonic, compare=False)
    completed: threading.Event = field(default_factory=threading.Event, compare=False)
    result: Any = field(default=None, init=False, compare=False)
    error: Exception | None = field(default=None, init=False, compare=False)

    def __lt__(self, other: "QueuedRequest") -> bool:
        # Priority first, then FIFO within same priority
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.sequence < other.sequence

    def __le__(self, other: "QueuedRequest") -> bool:
        return self == other or self.__lt__(other)

    def __gt__(self, other: "QueuedRequest") -> bool:
        return self.__lt__(other) is False and self != other

    def __ge__(self, other: "QueuedRequest") -> bool:
        return self.__le__(other) or self == other


@dataclass
class RequestQueueConfig:
    """Configuration for the request queue."""
    max_size: int = 100
    strategy: str = "fifo"  # fifo or priority


class NIMRequestQueue:
    """Centralized request queue for NVIDIA NIM API calls.

    Ensures all NIM requests are serialized through a controlled
    pipeline with priority support.
    """

    def __init__(self, config: RequestQueueConfig | None = None):
        self.config = config or RequestQueueConfig()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

        # Priority queue (heapq)
        self._queue: list[QueuedRequest] = []
        self._sequence = 0

        # Active requests tracking
        self._active_count: int = 0
        self._max_active: int = 2  # Max concurrent NIM requests

        # Metrics
        self._total_enqueued: int = 0
        self._total_processed: int = 0
        self._total_rejected: int = 0
        self._total_wait_time: float = 0.0

        # Notifier for when a slot opens
        self._slot_available = threading.Condition(self._lock)

    @property
    def depth(self) -> int:
        """Current queue depth."""
        with self._lock:
            return len(self._queue)

    @property
    def active_count(self) -> int:
        """Number of active (in-flight) requests."""
        with self._lock:
            return self._active_count

    def enqueue(
        self,
        callback: _Callback | None,
        priority: Priority = Priority.NORMAL,
        context: dict | None = None,
        timeout: float = 120.0,
    ) -> QueuedRequest | None:
        """Add a request to the queue.

        Returns the queued request, or None if queue is full.
        """
        with self._not_empty:
            if len(self._queue) >= self.config.max_size:
                self._total_rejected += 1
                logger.warning(
                    "NIM queue full (%d/%d), rejecting request",
                    len(self._queue),
                    self.config.max_size,
                )
                return None

            req = QueuedRequest(
                priority=priority,
                sequence=self._sequence,
                callback=callback,
                context=context,
            )
            self._sequence += 1
            heapq.heappush(self._queue, req)
            self._total_enqueued += 1
            self._not_empty.notify()
            return req

    def dequeue(self, timeout: float = 5.0) -> QueuedRequest | None:
        """Get the next request from the queue.

        Returns None if timeout expires.
        """
        with self._not_empty:
            deadline = time.monotonic() + timeout
            while not self._queue:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=remaining)
            return heapq.heappop(self._queue)

    def complete_request(
        self, req: QueuedRequest, result: Any = None, error: Exception | None = None
    ) -> None:
        """Mark a request as completed."""
        req.result = result
        req.error = error
        req.completed.set()
        with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self._total_processed += 1
            self._slot_available.notify()

    def can_dispatch(self) -> bool:
        """Check if we can dispatch another request."""
        with self._lock:
            return self._active_count < self._max_active

    def start_dispatch(self) -> bool:
        """Reserve a dispatch slot. Returns True if slot acquired."""
        with self._lock:
            if self._active_count < self._max_active:
                self._active_count += 1
                return True
            return False

    def wait_for_slot(self, timeout: float = 60.0) -> bool:
        """Block until a dispatch slot is available.

        Returns True if slot acquired, False if timeout.
        """
        with self._slot_available:
            deadline = time.monotonic() + timeout
            while self._active_count >= self._max_active:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._slot_available.wait(timeout=remaining)
            self._active_count += 1
            return True

    def release_slot(self) -> None:
        """Release a dispatch slot."""
        with self._slot_available:
            self._active_count = max(0, self._active_count - 1)
            self._slot_available.notify()

    def get_stats(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            return {
                "queue_depth": len(self._queue),
                "active_count": self._active_count,
                "max_active": self._max_active,
                "total_enqueued": self._total_enqueued,
                "total_processed": self._total_processed,
                "total_rejected": self._total_rejected,
                "avg_wait_time": (
                    self._total_wait_time / self._total_processed
                    if self._total_processed > 0
                    else 0.0
                ),
            }

    def drain(self, timeout: float = 30.0) -> int:
        """Drain remaining requests and return count.

        Used for graceful shutdown.
        """
        count = 0
        deadline = time.monotonic() + timeout
        with self._not_empty:
            while self._queue and time.monotonic() < deadline:
                req = heapq.heappop(self._queue)
                req.completed.set()
                count += 1
        return count

    def clear(self) -> int:
        """Clear all queued requests. Returns count cleared."""
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            return count


# Global singleton
_queue: NIMRequestQueue | None = None
_queue_lock = threading.Lock()


def get_request_queue(
    config: RequestQueueConfig | None = None,
) -> NIMRequestQueue:
    """Get or create the global request queue."""
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = NIMRequestQueue(config)
        elif config is not None:
            _queue.config = config
        return _queue


def reset_request_queue() -> None:
    """Reset the global queue (for testing)."""
    global _queue
    with _queue_lock:
        _queue = None
