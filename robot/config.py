from dataclasses import dataclass, fields
from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RobotConfig:
    ip: str | None = None
    hand_side: str = "right"
    urdf_path: Path = ROOT / "vendor/rysen-sdk/urdf/apex_hand_right.urdf"
    sdk_path: Path = ROOT / "vendor/rysen-sdk/python/rysen_apexhand_sdk.py"
    command_rate: float = 30.0
    max_speed: float = 1.0  # rad/s
    max_accel: float = 5.0  # rad/s^2, enforced by the SDK's trajectory planner
    finger_torque_pct: float = 30.0
    target_timeout: float = 0.5
    feedback_timeout: float = 0.5
    startup_timeout: float = 3.0
    feedback_rate: int = 100
    return_to_neutral_on_exit: bool = True

    def __post_init__(self):
        if not isinstance(self.return_to_neutral_on_exit, bool):
            raise ValueError("return_to_neutral_on_exit must be a boolean")
        if self.hand_side != "right":
            raise ValueError("This controller is exclusively for the RIGHT ApexHand")
        if self.ip is not None and (not isinstance(self.ip, str) or not self.ip.strip()):
            raise ValueError("ip must be a nonempty address or null for dry-run")
        for name in ("command_rate", "max_speed", "max_accel", "finger_torque_pct",
                     "target_timeout", "feedback_timeout", "startup_timeout"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.command_rate > 100 or self.finger_torque_pct > 100:
            raise ValueError("command_rate and finger_torque_pct must not exceed 100")
        if self.target_timeout <= 1 / self.command_rate:
            raise ValueError("target_timeout must exceed one command period")
        if isinstance(self.feedback_rate, bool) or not isinstance(self.feedback_rate, int) or not 1 <= self.feedback_rate <= 1000:
            raise ValueError("feedback_rate must be an integer in [1, 1000]")
        for name in ("urdf_path", "sdk_path"):
            object.__setattr__(self, name, Path(getattr(self, name)).resolve())

    @classmethod
    def from_yaml(cls, path):
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as stream:
            options = yaml.safe_load(stream)
        if not isinstance(options, dict):
            raise ValueError("Robot configuration must be a mapping")
        unknown = set(options) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown robot configuration keys: {sorted(unknown)}")
        for key in ("urdf_path", "sdk_path"):
            if key in options:
                options[key] = path.parent / options[key]
        return cls(**options)
