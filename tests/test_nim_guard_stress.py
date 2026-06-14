#!/usr/bin/env python3
"""Stress test for NVIDIA NIM Guard."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.nim_guard import NIMGuard
from agent.nim_rate_limiter import NIMRateLimiterConfig
from agent.nim_circuit_breaker import CircuitBreakerConfig


def test_dry_run():
    guard = NIMGuard(
        rate_config=NIMRateLimiterConfig(
            global_rpm=28,
            burst_limit=5,
            max_concurrent_requests=2,
        ),
        circuit_config=CircuitBreakerConfig(
            failure_threshold=3,
            cooldown_seconds=60.0,
        ),
    )

    total = 45
    allowed = 0
    rejected = 0

    print(f"Simulating {total} rapid requests (28 RPM, burst=5, concurrent=2)...")
    start = time.time()

    for i in range(total):
        ok, reason = guard.allow_request()
        elapsed = time.time() - start
        if ok:
            allowed += 1
            guard.before_request()
            time.sleep(0.05)
            guard.after_success(latency_ms=50)
        else:
            rejected += 1
            time.sleep(0.1)

        status = "OK" if ok else "BLOCKED"
        print(f"  [{i+1:>2}/{total}] {status}: {reason} ({elapsed:.2f}s)")

    total_time = time.time() - start
    print(f"\nResults: {allowed}/{total} allowed ({allowed/total*100:.0f}%) in {total_time:.1f}s")

    # Circuit breaker test
    print("\nCircuit breaker test (consecutive 429s):")
    for i in range(5):
        guard.after_429()
        cb = guard.circuit_breaker
        print(f"  429 #{i+1} -> {cb.state.value}, consecutive={cb.consecutive_failures}")
        if cb.state.value == "open":
            print(f"    Circuit breaker OPENED (cooldown {cb.config.cooldown_seconds}s)")
            break

    # Probe recovery
    print("\nProbe recovery:")
    guard.after_success(latency_ms=100)
    print(f"  After success: {guard.circuit_breaker.state.value}")

    # Stats
    print("\nStats:")
    for k, v in guard.get_stats().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    test_dry_run()
