"""Read-only inventory of capabilities exposed by the Rysen SDK object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SdkCapabilityReport:
    methods: tuple[str, ...]
    read_only_methods: tuple[str, ...]
    calibration_methods: tuple[str, ...]
    position_offset_methods: tuple[str, ...]


READ_ONLY_METHODS = (
    "is_connected",
    "get_version_info",
    "get_hardware_uid",
    "get_parameters",
    "get_hand_dir",
    "get_hardware_error_code",
    "get_joint_states",
    "get_motor_states",
)


def inspect_sdk_capabilities(sdk: Any) -> SdkCapabilityReport:
    methods = tuple(sorted(name for name in dir(sdk) if not name.startswith("_") and callable(getattr(sdk, name, None))))
    read_only = tuple(name for name in READ_ONLY_METHODS if name in methods)
    calibration = tuple(name for name in methods if "calib" in name.lower())
    offsets = tuple(name for name in methods if "offset" in name.lower() or "zero" in name.lower())
    return SdkCapabilityReport(
        methods=methods,
        read_only_methods=read_only,
        calibration_methods=calibration,
        position_offset_methods=offsets,
    )
