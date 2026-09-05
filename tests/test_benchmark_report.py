import json

from metrics.benchmark_report import BenchmarkReport
from metrics.latency import summarize_latencies
from metrics.trajectory import TrajectorySummary


def test_benchmark_report_serializes_summaries(tmp_path):
    report = BenchmarkReport.from_summaries(
        "test",
        summarize_latencies([0.001]),
        TrajectorySummary(1, 0.0, 0.0, 0, 0, 0, 0, 0),
        {"lost_hand_events": 2},
    )
    path = tmp_path / "report.json"
    report.write_json(path)
    data = json.loads(path.read_text())
    assert data["name"] == "test"
    assert data["latency"]["count"] == 1
    assert data["perception"]["lost_hand_events"] == 2
