import pytest

from metrics.latency import (
    LatencySummary,
    benchmark_operation,
    format_latency_summary,
    percentile,
    summarize_latencies,
)


def test_percentile_interpolates_sorted_values():
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == pytest.approx(2.5)
    assert percentile([1.0, 2.0, 3.0, 4.0], 95.0) == pytest.approx(3.85)


def test_summarize_latencies_reports_core_stats():
    summary = summarize_latencies([0.003, 0.001, 0.002], failures=1)

    assert summary.count == 3
    assert summary.failures == 1
    assert summary.success_rate == pytest.approx(0.75)
    assert summary.mean_s == pytest.approx(0.002)
    assert summary.median_s == pytest.approx(0.002)
    assert summary.p95_s == pytest.approx(0.0029)
    assert summary.min_s == pytest.approx(0.001)
    assert summary.max_s == pytest.approx(0.003)


def test_summarize_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="at least one"):
        summarize_latencies([])
    with pytest.raises(ValueError, match="non-negative"):
        summarize_latencies([-0.1])
    with pytest.raises(ValueError, match="failures"):
        summarize_latencies([0.1], failures=-1)


def test_benchmark_operation_counts_failures_and_successes():
    times = iter([0.0, 0.01, 0.02, 0.05, 0.06])
    calls = {"count": 0}

    def operation():
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("read failed")

    summary = benchmark_operation(operation, samples=3, clock=lambda: next(times))

    assert summary.count == 2
    assert summary.failures == 1
    assert summary.mean_s == pytest.approx(0.01)
    assert calls["count"] == 3


def test_benchmark_operation_validates_arguments():
    with pytest.raises(ValueError, match="samples"):
        benchmark_operation(lambda: None, samples=0)
    with pytest.raises(ValueError, match="sleep_s"):
        benchmark_operation(lambda: None, samples=1, sleep_s=-0.1)


def test_format_latency_summary_uses_milliseconds():
    summary = LatencySummary(
        count=2,
        failures=0,
        mean_s=0.001,
        median_s=0.002,
        p95_s=0.003,
        min_s=0.0005,
        max_s=0.004,
    )

    text = format_latency_summary(summary)

    assert "count=2" in text
    assert "success_rate=1.000" in text
    assert "mean=1.000ms" in text
    assert "max=4.000ms" in text


def test_benchmark_cli_help_imports_project_modules():
    import subprocess

    result = subprocess.run(
        [
            "python3",
            "scripts/benchmark_real_apex_readonly.py",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "joint-state latency benchmark" in result.stdout
