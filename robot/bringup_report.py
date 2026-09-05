"""Serializable read-only hardware bring-up report helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BringupReport:
    timestamp: str
    ip: str
    sdk_version: str
    firmware_version: str
    hardware_uid: str
    hand_dir: str
    hardware_error_code: str
    latency: dict[str, Any]
    joint_statistics: list[dict[str, Any]]
    limit_violations: list[dict[str, Any]]
    status: str
    blockers: list[str]

    def write_json(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
