from robot.sdk_capabilities import inspect_sdk_capabilities


class FakeSdk:
    def get_joint_states(self):
        return None

    def get_parameters(self):
        return None

    def start_tactile_calibration(self):
        return None

    def move_joint(self):
        return None


def test_capability_report_separates_read_only_and_calibration_methods():
    report = inspect_sdk_capabilities(FakeSdk())
    assert "get_joint_states" in report.read_only_methods
    assert report.calibration_methods == ("start_tactile_calibration",)
    assert report.position_offset_methods == ()
