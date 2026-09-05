from pathlib import Path

from scripts import check_real_apex_readonly as check


def test_vendor_python_path_is_sibling_of_project_root():
    root = Path("/workspace/apex-vision-retargeting")
    assert check.vendor_python_path(root) == Path("/workspace/rysen-sdk/python")


def test_sequence_from_named_attribute():
    class Box:
        joint_states = (1, 2, 3)

    assert check.sequence_from_attrs(Box(), ("joint_states",)) == [1, 2, 3]


def test_sequence_from_object_iterable_fallback():
    assert check.sequence_from_attrs((4, 5), ("missing",)) == [4, 5]


def test_sequence_from_non_iterable_returns_empty():
    assert check.sequence_from_attrs(object(), ("missing",)) == []


def test_parse_args_rejects_bad_timeout_in_main():
    assert check.main(["--timeout", "0", "--no-connect"]) == 1


def test_no_connect_import_path_smoke():
    assert check.main(["--no-connect"]) == 0
