"""Serializable combined benchmark report."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from metrics.latency import LatencySummary
from metrics.trajectory import TrajectorySummary


@dataclass(frozen=True)
class BenchmarkReport:
    name: str
    latency: dict[str, Any]
    trajectory: dict[str, Any]
    perception: dict[str, Any]

    @classmethod
    def from_summaries(cls, name: str, latency: LatencySummary, trajectory: TrajectorySummary, perception: dict[str, Any] | None = None) -> "BenchmarkReport":
        return cls(name=name, latency=asdict(latency), trajectory=asdict(trajectory), perception=perception or {})

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
