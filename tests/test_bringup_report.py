import json

from robot.bringup_report import BringupReport


def test_bringup_report_writes_json(tmp_path):
    report = BringupReport(
        timestamp="2026-09-05T00:00:00+00:00",
        ip="192.168.0.103",
        sdk_version="1.5.2",
        firmware_version="3.2.5",
        hardware_uid="uid",
        hand_dir="RIGHT",
        hardware_error_code="ERROR_CODE_OK",
        latency={"p95_s": 0.001},
        joint_statistics=[],
        limit_violations=[],
        status="BLOCKED",
        blockers=["test blocker"],
    )
    path = tmp_path / "report.json"
    report.write_json(path)
    loaded = json.loads(path.read_text())
    assert loaded["status"] == "BLOCKED"
    assert loaded["blockers"] == ["test blocker"]
