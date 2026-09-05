"""
Apex Hand URDF kinematics (M2.4).

Parses the official Apex Hand URDF (vendor data, read-only) and provides
forward kinematics. No robot geometry is hard-coded here: joint origin,
joint axis, joint limits, mimic relationships and the link hierarchy are
all read from the URDF file.

Conventions (per URDF spec):

- ``<origin xyz rpy>`` is the fixed transform from the parent link frame
  to the joint frame, composed as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
- ``<axis xyz>`` is the unit rotation axis expressed in the joint frame.
- For a revolute joint, the parent-to-child transform is

      T = T_origin @ Rot(axis, q)

- A ``<mimic>`` joint is not an actuator: its value is derived as

      q_mimic = multiplier * q_target + offset

Distinction between joint spaces:

- **active joints**    : the 16 revolute, non-mimic joints (actuator space).
- **mechanical joints**: all 21 revolute joints (adds the 5 mimic DIP/IP).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np

from robot.apex_joint_names import INDEX_ACTIVE_JOINTS


# Vendor sibling layout (see CLAUDE.md section 1):
#   ~/projects/{apex-vision-retargeting, apex-hand-urdf}
# In the dev container this is /workspace/{apex-vision-retargeting, apex-hand-urdf}.
DEFAULT_RIGHT_URDF = (
    Path(__file__).resolve().parents[2]
    / "apex-hand-urdf"
    / "apex_hand_right"
    / "apex_hand_right.urdf"
)

INDEX_TIP_LINK = "right_index_tip"


@dataclass(frozen=True)
class JointLimit:
    """Closed revolute joint limit [lower, upper] in radians."""

    lower: float
    upper: float

    def contains(
        self,
        value: float,
        atol: float = 1e-9,
    ) -> bool:
        return self.lower - atol <= value <= self.upper + atol


@dataclass(frozen=True)
class Mimic:
    """URDF ``<mimic>`` element."""

    target: str
    multiplier: float
    offset: float


@dataclass(frozen=True)
class UrdfJoint:
    """One joint from the URDF, plus enough geometry for FK."""

    name: str
    joint_type: str
    parent: str
    child: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: Optional[float]
    upper: Optional[float]
    mimic: Optional[Mimic]

    @property
    def is_revolute(self) -> bool:
        return self.joint_type == "revolute"

    @property
    def is_fixed(self) -> bool:
        return self.joint_type == "fixed"

    @property
    def is_mimic(self) -> bool:
        return self.mimic is not None


def rotation_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    """URDF rpy -> rotation matrix ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``."""
    roll, pitch, yaw = rpy

    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ]
    )

    ry = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ]
    )

    rz = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return rz @ ry @ rx


def rotation_about_axis(
    axis: tuple[float, float, float],
    angle: float,
) -> np.ndarray:
    """Rodrigues rotation about an arbitrary axis by ``angle`` radians."""
    a = np.asarray(axis, dtype=np.float64)

    norm = np.linalg.norm(a)
    if norm < 1e-12:
        return np.eye(3)

    a = a / norm

    k = np.array(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ]
    )

    return (
        np.eye(3)
        + math.sin(angle) * k
        + (1.0 - math.cos(angle)) * (k @ k)
    )


def origin_transform(
    xyz: tuple[float, float, float],
    rpy: tuple[float, float, float],
) -> np.ndarray:
    """Build the 4x4 homogeneous transform from ``xyz`` + ``rpy``."""
    t = np.eye(4)
    t[:3, :3] = rotation_matrix(rpy)
    t[:3, 3] = xyz
    return t


def _joint_transform(joint: UrdfJoint, q: float) -> np.ndarray:
    """Parent-to-child transform of one joint at value ``q``."""
    t = origin_transform(joint.xyz, joint.rpy)

    if joint.is_fixed:
        return t

    r = np.eye(4)
    r[:3, :3] = rotation_about_axis(joint.axis, q)

    return t @ r


class ApexUrdfModel:
    """
    A parsed Apex Hand URDF with forward kinematics.
    """

    def __init__(self, urdf_path: str | Path) -> None:
        self.path = Path(urdf_path)

        if not self.path.exists():
            raise FileNotFoundError(
                f"URDF not found: {self.path}"
            )

        self.joints: dict[str, UrdfJoint] = {}
        self.root_link: str = ""

        self._child_to_joint: dict[str, str] = {}

        self._parse()

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self) -> None:
        tree = ET.parse(self.path)
        root = tree.getroot()

        links: set[str] = set()
        child_links: set[str] = set()

        for element in root.iter("joint"):
            name = element.attrib["name"]
            joint_type = element.attrib["type"]

            parent = element.find("parent").attrib["link"]
            child = element.find("child").attrib["link"]

            origin = element.find("origin")
            if origin is not None:
                xyz = tuple(
                    float(x)
                    for x in origin.attrib.get(
                        "xyz", "0 0 0"
                    ).split()
                )
                rpy = tuple(
                    float(x)
                    for x in origin.attrib.get(
                        "rpy", "0 0 0"
                    ).split()
                )
            else:
                xyz = (0.0, 0.0, 0.0)
                rpy = (0.0, 0.0, 0.0)

            axis_element = element.find("axis")
            if axis_element is not None:
                axis = tuple(
                    float(x)
                    for x in axis_element.attrib["xyz"].split()
                )
            else:
                axis = (0.0, 0.0, 0.0)

            limit_element = element.find("limit")
            lower = (
                float(limit_element.attrib["lower"])
                if limit_element is not None
                else None
            )
            upper = (
                float(limit_element.attrib["upper"])
                if limit_element is not None
                else None
            )

            mimic_element = element.find("mimic")
            mimic = None
            if mimic_element is not None:
                mimic = Mimic(
                    target=mimic_element.attrib["joint"],
                    multiplier=float(
                        mimic_element.attrib.get(
                            "multiplier", "1.0"
                        )
                    ),
                    offset=float(
                        mimic_element.attrib.get("offset", "0")
                    ),
                )

            self.joints[name] = UrdfJoint(
                name=name,
                joint_type=joint_type,
                parent=parent,
                child=child,
                xyz=xyz,
                rpy=rpy,
                axis=axis,
                lower=lower,
                upper=upper,
                mimic=mimic,
            )

            self._child_to_joint[child] = name

            links.add(parent)
            links.add(child)
            child_links.add(child)

        roots = links - child_links
        if len(roots) != 1:
            raise ValueError(
                f"Expected exactly one root link, got {roots}"
            )

        self.root_link = next(iter(roots))

    # ------------------------------------------------------------------
    # Joint spaces
    # ------------------------------------------------------------------

    @property
    def revolute_joints(self) -> list[str]:
        """All revolute joints in URDF order (21 mechanical DoF)."""
        return [
            name
            for name, joint in self.joints.items()
            if joint.is_revolute
        ]

    @property
    def mimic_joints(self) -> list[str]:
        """Revolute joints driven by a ``<mimic>`` (5 passive DoF)."""
        return [
            name
            for name, joint in self.joints.items()
            if joint.is_revolute and joint.is_mimic
        ]

    @property
    def active_joints(self) -> list[str]:
        """Revolute, non-mimic joints (16 active DoF)."""
        return [
            name
            for name, joint in self.joints.items()
            if joint.is_revolute and not joint.is_mimic
        ]

    def joint_limit(self, joint_name: str) -> JointLimit:
        """Return the URDF limit of one revolute joint in radians."""
        joint = self.joints[joint_name]

        if not joint.is_revolute:
            raise ValueError(
                f"Joint '{joint_name}' is not revolute"
            )

        if joint.lower is None or joint.upper is None:
            raise ValueError(
                f"Joint '{joint_name}' has no finite URDF limit"
            )

        return JointLimit(
            lower=joint.lower,
            upper=joint.upper,
        )

    def active_joint_limits(self) -> dict[str, JointLimit]:
        """URDF limits for all active joints, keyed by joint name."""
        return {
            name: self.joint_limit(name)
            for name in self.active_joints
        }

    def validate_active_joint_values(
        self,
        q_active: dict[str, float],
        atol: float = 1e-9,
    ) -> None:
        """Reject missing, mimic, unknown, or out-of-limit active q values."""
        active = set(self.active_joints)

        for name in q_active:
            if name not in self.joints:
                raise ValueError(
                    f"Unknown joint '{name}'"
                )

            joint = self.joints[name]
            if not joint.is_revolute or joint.is_mimic:
                raise ValueError(
                    f"Joint '{name}' is not an active revolute joint"
                )

        missing = active - set(q_active)
        if missing:
            raise ValueError(
                "Missing active joint values: "
                + ", ".join(sorted(missing))
            )

        for name in self.active_joints:
            value = q_active[name]
            limit = self.joint_limit(name)
            if not limit.contains(value, atol=atol):
                raise ValueError(
                    f"Joint '{name}' value {value} rad is outside "
                    f"URDF limit [{limit.lower}, {limit.upper}] rad"
                )

    # ------------------------------------------------------------------
    # Kinematic chain
    # ------------------------------------------------------------------

    def chain_to(self, link_name: str) -> list[str]:
        """Ordered joint names from the root link down to ``link_name``."""
        if link_name == self.root_link:
            return []

        if link_name not in self._child_to_joint:
            raise ValueError(
                f"Link '{link_name}' not found in URDF"
            )

        chain: list[str] = []

        current = link_name
        while current != self.root_link:
            joint_name = self._child_to_joint[current]
            chain.append(joint_name)
            current = self.joints[joint_name].parent

        chain.reverse()
        return chain

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------

    def forward_kinematics(
        self,
        link_name: str,
        q_active: dict[str, float],
    ) -> np.ndarray:
        """
        Pose (4x4) of ``link_name`` given active joint values.

        Mimic joints are resolved automatically from their target.
        """

        q = dict(q_active)

        transform = np.eye(4)

        for joint_name in self.chain_to(link_name):
            joint = self.joints[joint_name]

            if joint.is_fixed:
                qj = 0.0
            elif joint.is_mimic:
                target = joint.mimic.target
                if target not in q:
                    raise ValueError(
                        f"Mimic target '{target}' of '{joint.name}' "
                        "is not present in q_active"
                    )
                qj = (
                    joint.mimic.multiplier * q[target]
                    + joint.mimic.offset
                )
            else:
                if joint.name not in q:
                    raise ValueError(
                        f"Active joint '{joint.name}' "
                        "is not present in q_active"
                    )
                qj = q[joint.name]

            transform = transform @ _joint_transform(joint, qj)

        return transform

    def link_position(
        self,
        link_name: str,
        q_active: dict[str, float],
    ) -> np.ndarray:
        """Translation part (x, y, z) of a link's pose."""
        return self.forward_kinematics(
            link_name, q_active
        )[:3, 3]

    # ------------------------------------------------------------------
    # Index finger convenience
    # ------------------------------------------------------------------

    def index_tip_position(
        self,
        j0: float,
        j1: float,
        j2: float,
    ) -> np.ndarray:
        """Fingertip position of the right index finger (radians)."""
        q = dict(zip(INDEX_ACTIVE_JOINTS, (j0, j1, j2)))
        return self.link_position(INDEX_TIP_LINK, q)
