"""
Small latency benchmarking helpers for read-only robot IO.

Time units are seconds internally. Summary values are also seconds so callers
can decide how to present milliseconds in logs or UI.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencySummary:
    """Latency statistics for one repeated operation, in seconds."""

    count: int
    failures: int
    mean_s: float
    median_s: float
    p95_s: float
    max_s: float
    min_s: float

    @property
    def success_rate(self) -> float:
        total = self.count + self.failures
        if total == 0:
            return 0.0
        return self.count / total


def percentile(sorted_values: list[float], percent: float) -> float:
    """Return an interpolated percentile from already sorted values."""
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be in [0, 100]")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * (percent / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def summarize_latencies(latencies_s: list[float], failures: int = 0) -> LatencySummary:
    """Summarize non-negative latency samples in seconds."""
    if failures < 0:
        raise ValueError("failures must be non-negative")
    if not latencies_s:
        if failures == 0:
            raise ValueError("at least one latency sample or failure is required")
        return LatencySummary(
            count=0,
            failures=failures,
            mean_s=0.0,
            median_s=0.0,
            p95_s=0.0,
            max_s=0.0,
            min_s=0.0,
        )

    for value in latencies_s:
        if value < 0.0:
            raise ValueError("latencies must be non-negative")

    values = sorted(float(value) for value in latencies_s)
    return LatencySummary(
        count=len(values),
        failures=failures,
        mean_s=sum(values) / len(values),
        median_s=percentile(values, 50.0),
        p95_s=percentile(values, 95.0),
        max_s=values[-1],
        min_s=values[0],
    )


def benchmark_operation(
    operation: Callable[[], object],
    samples: int,
    clock: Callable[[], float] = time.monotonic,
    sleep_s: float = 0.0,
    sleeper: Callable[[float], None] = time.sleep,
) -> LatencySummary:
    """Measure repeated operation latency with failure accounting."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if sleep_s < 0.0:
        raise ValueError("sleep_s must be non-negative")

    latencies: list[float] = []
    failures = 0
    for index in range(samples):
        start = clock()
        try:
            operation()
        except Exception:
            failures += 1
        else:
            elapsed = clock() - start
            if elapsed < 0.0:
                raise ValueError("clock moved backwards")
            latencies.append(elapsed)

        if sleep_s > 0.0 and index + 1 < samples:
            sleeper(sleep_s)

    return summarize_latencies(latencies, failures=failures)


def format_latency_summary(summary: LatencySummary) -> str:
    """Format latency statistics in milliseconds for operator logs."""
    return (
        f"count={summary.count} failures={summary.failures} "
        f"success_rate={summary.success_rate:.3f} "
        f"mean={summary.mean_s * 1000.0:.3f}ms "
        f"median={summary.median_s * 1000.0:.3f}ms "
        f"p95={summary.p95_s * 1000.0:.3f}ms "
        f"min={summary.min_s * 1000.0:.3f}ms "
        f"max={summary.max_s * 1000.0:.3f}ms"
    )
