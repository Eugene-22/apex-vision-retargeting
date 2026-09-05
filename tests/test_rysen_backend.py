from types import SimpleNamespace

import pytest

from robot.apex_joint_names import APEX_ACTIVE_JOINTS, ApexActiveJointTarget
from robot.interface import ApexJointState
from robot.rysen_backend import (
    RealApexHand,
    RealApexHandMvp,
    RysenBackendError,
    SDK_JOINT_ID_ATTRS,
    extract_sdk_joint_positions,
    sdk_joint_id_map,
    vendor_python_path,
)


class FakeErrorCode:
    ERROR_CODE_OK = "ok"
    ERROR_CODE_FAIL = "fail"


class FakeConnectionType:
    CONNECTION_TYPE_ETHERNET = "ethernet"


class FakeJointId:
    pass


for _joint_name, _attr in SDK_JOINT_ID_ATTRS.items():
    setattr(FakeJointId, _attr, _attr)


class FakeSdk:
    def __init__(self, states=None, connect_result="ok", disconnect_result="ok"):
        self.states = states if states is not None else make_sdk_states()
        self.connect_result = connect_result
        self.disconnect_result = disconnect_result
        self.connected = False
        self.connect_calls = []
        self.disconnect_calls = 0
        self.log_path = None
        self.logging_enabled = None

    def connect(self, ip, connection_type):
        self.connect_calls.append((ip, connection_type))
        if self.connect_result == FakeErrorCode.ERROR_CODE_OK:
            self.connected = True
        return self.connect_result

    def disconnect(self):
        self.disconnect_calls += 1
        if self.disconnect_result == FakeErrorCode.ERROR_CODE_OK:
            self.connected = False
        return self.disconnect_result

    def is_connected(self):
        return self.connected

    def get_joint_states(self):
        return SimpleNamespace(joint_states=self.states)

    def set_log_path(self, path):
        self.log_path = path

    def enable_logging(self, enabled):
        self.logging_enabled = enabled


def make_sdk_states(missing=None):
    missing = set(missing or ())
    states = []
    for index, joint_name in enumerate(APEX_ACTIVE_JOINTS):
        if joint_name in missing:
            continue
        attr = SDK_JOINT_ID_ATTRS[joint_name]
        states.append(SimpleNamespace(joint_id=getattr(FakeJointId, attr), position=index + 0.25))
    states.append(SimpleNamespace(joint_id="JOINT_ID_INDEX_J3", position=99.0))
    return states


def make_hand(sdk=None, **kwargs):
    return RealApexHand(
        ip="10.0.0.2",
        sdk=sdk or FakeSdk(),
        connection_type=FakeConnectionType.CONNECTION_TYPE_ETHERNET,
        error_code=FakeErrorCode,
        joint_id_enum=FakeJointId,
        clock=lambda: 42.0,
        **kwargs,
    )


def test_vendor_python_path_is_sibling_of_project_root():
    assert str(vendor_python_path()).endswith("rysen-sdk/python")


def test_sdk_joint_id_map_contains_only_active_joints():
    mapping = sdk_joint_id_map(FakeJointId)

    assert len(mapping) == 16
    assert set(mapping.values()) == set(APEX_ACTIVE_JOINTS)
    assert "JOINT_ID_INDEX_J3" not in mapping


def test_extract_sdk_joint_positions_uses_canonical_order_and_skips_mimic():
    target = extract_sdk_joint_positions(
        SimpleNamespace(joint_states=make_sdk_states()),
        sdk_joint_id_map(FakeJointId),
    )

    assert isinstance(target, ApexActiveJointTarget)
    assert target.values == tuple(index + 0.25 for index in range(16))


def test_extract_sdk_joint_positions_rejects_missing_active_joint():
    with pytest.raises(RysenBackendError, match="right_index_j1"):
        extract_sdk_joint_positions(
            SimpleNamespace(joint_states=make_sdk_states(missing={"right_index_j1"})),
            sdk_joint_id_map(FakeJointId),
        )


def test_real_backend_connects_and_reads_joint_state():
    sdk = FakeSdk()
    hand = make_hand(sdk=sdk)

    state = hand.get_joint_state()

    assert sdk.connect_calls == [("10.0.0.2", "ethernet")]
    assert isinstance(state, ApexJointState)
    assert state.timestamp == 42.0
    assert state.position.values[0] == pytest.approx(0.25)
    hand.close()
    assert sdk.disconnect_calls == 1


def test_real_backend_can_delay_connect():
    sdk = FakeSdk()
    hand = make_hand(sdk=sdk, auto_connect=False)

    assert not hand.is_connected
    hand.connect()
    assert hand.is_connected
    hand.close()


def test_real_backend_rejects_motion_commands():
    hand = make_hand()

    with pytest.raises(NotImplementedError, match="disabled"):
        hand.command_position(ApexActiveJointTarget(values=(0.0,) * 16))


def test_real_backend_raises_on_connect_failure():
    sdk = FakeSdk(connect_result=FakeErrorCode.ERROR_CODE_FAIL)

    with pytest.raises(RysenBackendError, match="SDK connect failed"):
        make_hand(sdk=sdk)


def test_real_backend_raises_on_read_after_close():
    hand = make_hand()
    hand.close()

    with pytest.raises(RysenBackendError, match="closed"):
        hand.get_joint_state()


def test_real_backend_sets_optional_log_dir(tmp_path):
    sdk = FakeSdk()
    log_dir = tmp_path / "sdk_logs"
    hand = make_hand(sdk=sdk, log_dir=log_dir)

    assert sdk.log_path == str(log_dir)
    assert sdk.logging_enabled is True
    hand.close()


class FakeFingerId:
    FINGER_ID_INDEX = "index"

class FakeParam:
    pass

FakeSdk.set_finger_enabled = lambda self, fingers: (setattr(self, "enabled", fingers) or FakeErrorCode.ERROR_CODE_OK)
FakeSdk.set_finger_disabled = lambda self, fingers: (setattr(self, "disabled", fingers) or FakeErrorCode.ERROR_CODE_OK)
FakeSdk.move_joint = lambda self, commands: (setattr(self, "commands", commands) or FakeErrorCode.ERROR_CODE_OK)

def test_mvp_requires_confirmation():
    hand = make_hand()
    with pytest.raises(RysenBackendError, match="confirm_movement"):
        RealApexHandMvp(hand).move_one_joint("right_index_j0", 0.1)
    hand.close()

def test_mvp_sends_one_index_command_and_disables():
    sdk = FakeSdk(); hand = make_hand(sdk=sdk)
    mvp = RealApexHandMvp(hand, confirm_movement=True, finger_id_enum=FakeFingerId, joint_control_param=FakeParam)
    result = mvp.move_one_joint("right_index_j0", 4.30)
    assert len(sdk.commands) == 1
    assert sdk.commands[0].joint_id == FakeJointId.JOINT_ID_INDEX_J0
    assert sdk.enabled == ["index"] and sdk.disabled == ["index"]
    assert result.applied.as_dict()["right_index_j0"] == 4.30
    hand.close()

def test_mvp_rejects_other_and_large_motion():
    hand = make_hand()
    mvp = RealApexHandMvp(hand, confirm_movement=True)
    with pytest.raises(RysenBackendError): mvp.move_one_joint("right_index_j2", 0.1)
    with pytest.raises(RysenBackendError): mvp.move_one_joint("right_index_j0", 99.0)
    hand.close()
