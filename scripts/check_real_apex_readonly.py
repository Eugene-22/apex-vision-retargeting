#!/usr/bin/env python3
"""
Read-only Apex Hand SDK bring-up check.

This script is intentionally limited to import, TCP reachability, SDK connect,
read-only status queries, and disconnect. It does not enable fingers, clear
faults, set limits, calibrate sensors, or send motion commands.
"""

from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path
from typing import Any, Iterable

DEFAULT_IP = "192.168.0.103"
CONTROL_PORTS = (5856, 5857)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def vendor_python_path(root: Path | None = None) -> Path:
    base = root if root is not None else project_root()
    return base.parent / "rysen-sdk" / "python"


def ensure_vendor_python_path(root: Path | None = None) -> Path:
    path = vendor_python_path(root)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def import_sdk(root: Path | None = None) -> tuple[Any, Any, Any]:
    ensure_vendor_python_path(root)
    from rysen_apexhand_sdk import ConnectionType, ErrorCode, Rysen

    return Rysen, ConnectionType, ErrorCode


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    return str(value)


def is_ok(value: Any, error_code: Any) -> bool:
    ok = getattr(error_code, "ERROR_CODE_OK", None)
    return value == ok


def check_tcp_port(host: str, port: int, timeout: float) -> tuple[bool, str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True, "open"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        sock.close()


def public_attrs(obj: Any) -> list[tuple[str, Any]]:
    names = [name for name in dir(obj) if not name.startswith("_")]
    attrs: list[tuple[str, Any]] = []
    for name in sorted(names):
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value):
            continue
        attrs.append((name, value))
    return attrs


def sequence_from_attrs(obj: Any, candidate_names: Iterable[str]) -> list[Any]:
    if obj is None:
        return []
    for name in candidate_names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        try:
            return list(value)
        except TypeError:
            continue
    try:
        return list(obj)
    except TypeError:
        return []


def summarize_object(label: str, obj: Any, max_items: int = 8) -> None:
    if obj is None:
        print(f"{label}: None")
        return
    attrs = public_attrs(obj)
    if not attrs:
        print(f"{label}: {obj}")
        return

    instance_attrs = [(name, value) for name, value in attrs if name not in dir(type(obj))]
    fields = instance_attrs if instance_attrs else attrs
    if instance_attrs and len(instance_attrs) == 1:
        print(f"{label}: {instance_attrs[0][1]}")
        return

    print(f"{label}:")
    for name, value in fields[:max_items]:
        print(f"  {name}: {value}")
    if len(fields) > max_items:
        print(f"  ... {len(fields) - max_items} more fields")


def summarize_sequence(label: str, values: list[Any], max_items: int = 4) -> None:
    print(f"{label}: count={len(values)}")
    for index, item in enumerate(values[:max_items]):
        attrs = public_attrs(item)
        if attrs:
            fields = ", ".join(f"{name}={value}" for name, value in attrs[:6])
            print(f"  [{index}] {fields}")
        else:
            print(f"  [{index}] {item}")
    if len(values) > max_items:
        print(f"  ... {len(values) - max_items} more items")


def read_status(sdk: Any, read_tactile: bool) -> None:
    print(f"is_connected: {sdk.is_connected()}")

    for label, method_name in (
        ("version_info", "get_version_info"),
        ("hardware_uid", "get_hardware_uid"),
        ("hand_dir", "get_hand_dir"),
        ("hardware_error_code", "get_hardware_error_code"),
        ("parameters", "get_parameters"),
    ):
        try:
            summarize_object(label, getattr(sdk, method_name)())
        except Exception as exc:
            print(f"{label}: ERROR {type(exc).__name__}: {exc}")

    try:
        joint_states = sdk.get_joint_states()
        summarize_sequence(
            "joint_states",
            sequence_from_attrs(joint_states, ("joint_states", "joints")),
        )
    except Exception as exc:
        print(f"joint_states: ERROR {type(exc).__name__}: {exc}")

    try:
        motor_states = sdk.get_motor_states()
        summarize_sequence(
            "motor_states",
            sequence_from_attrs(motor_states, ("motors", "motor_states")),
        )
    except Exception as exc:
        print(f"motor_states: ERROR {type(exc).__name__}: {exc}")

    if read_tactile:
        try:
            summarize_object("hand_sensor_image", sdk.get_hand_sensor_image())
        except Exception as exc:
            print(f"hand_sensor_image: ERROR {type(exc).__name__}: {exc}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only Rysen Apex Hand SDK check")
    parser.add_argument("--ip", default=DEFAULT_IP, help="Apex Hand IP address")
    parser.add_argument("--timeout", type=float, default=1.0, help="TCP timeout in seconds")
    parser.add_argument(
        "--log-dir",
        default=str(project_root() / "outputs" / "real_apex_logs"),
        help="SDK log directory inside this project",
    )
    parser.add_argument("--skip-tcp", action="store_true", help="Skip TCP port reachability checks")
    parser.add_argument("--require-tcp", action="store_true", help="Exit before SDK connect if TCP checks fail")
    parser.add_argument("--read-tactile", action="store_true", help="Also read tactile sensor image")
    parser.add_argument("--no-connect", action="store_true", help="Only verify SDK import and script wiring")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.timeout <= 0.0:
        print("ERROR: --timeout must be positive", file=sys.stderr)
        return 1

    print("READ-ONLY Apex SDK check: no enable, no move, no fault clear, no calibration")
    vendor_path = ensure_vendor_python_path()
    print(f"vendor_python_path: {vendor_path}")

    try:
        Rysen, ConnectionType, ErrorCode = import_sdk()
    except Exception as exc:
        print(f"SDK import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("SDK import: OK")
    if args.no_connect:
        print("no-connect requested; stopping before network or hardware access")
        return 0

    tcp_ok = True
    if not args.skip_tcp:
        for port in CONTROL_PORTS:
            ok, detail = check_tcp_port(args.ip, port, args.timeout)
            print(f"TCP {args.ip}:{port}: {detail}")
            tcp_ok = tcp_ok and ok
        if args.require_tcp and not tcp_ok:
            print("TCP check failed and --require-tcp was set; stopping before SDK connect")
            return 2
        if not tcp_ok:
            print("TCP check failed; continuing because SDK connect may provide a more specific error")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    sdk = Rysen()
    disconnect_needed = False
    try:
        sdk.set_log_path(str(log_dir))
        sdk.enable_logging(True)
        print(f"SDK log_dir: {log_dir}")

        ret = sdk.connect(args.ip, ConnectionType.CONNECTION_TYPE_ETHERNET)
        print(f"SDK connect: {enum_name(ret)}")
        if not is_ok(ret, ErrorCode):
            return 3
        disconnect_needed = True

        time.sleep(0.2)
        read_status(sdk, args.read_tactile)
        return 0
    finally:
        if disconnect_needed:
            try:
                ret = sdk.disconnect()
                print(f"SDK disconnect: {enum_name(ret)}")
            except Exception as exc:
                print(f"SDK disconnect: ERROR {type(exc).__name__}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
