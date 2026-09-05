"""
Human index-finger calibration (M2.3.1).

Calibration data is intentionally decoupled from the mapping
algorithm and serializable to JSON so it can be captured once,
persisted, and reused across runs without editing source code.

All angles are stored in **radians**. Only logs / UI may convert
to degrees.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JointRange:
    """
    A closed angular range [minimum, maximum] in radians.
    """

    minimum: float
    maximum: float

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.minimum)
            and math.isfinite(self.maximum)
        ):
            raise ValueError(
                "JointRange bounds must be finite, got "
                f"({self.minimum}, {self.maximum})"
            )

        if self.minimum >= self.maximum:
            raise ValueError(
                "Invalid JointRange: minimum="
                f"{self.minimum} must be < maximum="
                f"{self.maximum}"
            )

    def width(self) -> float:
        return self.maximum - self.minimum

    def to_dict(self) -> dict[str, float]:
        return {
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JointRange":
        return cls(
            minimum=float(data["minimum"]),
            maximum=float(data["maximum"]),
        )


@dataclass(frozen=True)
class IndexCalibration:
    """
    Human index-finger joint ranges (radians).

    Captured from a real subject; do NOT assume human neutral
    angles are zero. Each range is independently calibrated:

    - ``human_abduction``  : MCP abduction / adduction
    - ``human_mcp_flexion``: MCP flexion / extension
    - ``human_pip_flexion``: PIP flexion
    """

    human_abduction: JointRange
    human_mcp_flexion: JointRange
    human_pip_flexion: JointRange

    def to_dict(self) -> dict[str, Any]:
        return {
            "human_abduction": self.human_abduction.to_dict(),
            "human_mcp_flexion": self.human_mcp_flexion.to_dict(),
            "human_pip_flexion": self.human_pip_flexion.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IndexCalibration":
        return cls(
            human_abduction=JointRange.from_dict(
                data["human_abduction"]
            ),
            human_mcp_flexion=JointRange.from_dict(
                data["human_mcp_flexion"]
            ),
            human_pip_flexion=JointRange.from_dict(
                data["human_pip_flexion"]
            ),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n"
        )

    @classmethod
    def load(cls, path: str | Path) -> "IndexCalibration":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
