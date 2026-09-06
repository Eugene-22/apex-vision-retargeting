"""URDF forward kinematics and analytical point Jacobians for ApexHand.

Meshes/inertias are not needed for geometric IK. Fixed/revolute joints and
linear URDF mimic relationships are supported. IK uses independent variables.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from perception.landmarks import validate_side

JOINT_NAMES = tuple(f"f{finger}_joint{joint}" for finger in range(5)
                    for joint in range(5 if finger == 0 else 4))


class ApexHandModel:
    def __init__(self, urdf_path, hand_side="right", tip_offsets=None):
        self.hand_side = validate_side(hand_side)
        self.urdf_path = Path(urdf_path).resolve()
        root = ET.parse(self.urdf_path).getroot()
        self.root_link = f"Palm_link_{hand_side}"
        links = [link.attrib["name"] for link in root.findall("link")]
        if self.root_link not in links or len(set(links)) != len(links):
            raise ValueError("URDF must contain unique links and the requested ApexHand palm")
        joints = root.findall("joint")
        self.joint_names = JOINT_NAMES
        self.num_joints = len(JOINT_NAMES)
        actuated = {j.attrib["name"]: j for j in joints if j.attrib["type"] != "fixed"}
        if set(actuated) != set(JOINT_NAMES):
            raise ValueError("URDF must contain exactly the 21 ApexHand revolute joints")
        self.lower = np.empty(21)
        self.upper = np.empty(21)
        self.velocity_limits = np.empty(21)
        for i, name in enumerate(JOINT_NAMES):
            limit = actuated[name].find("limit")
            if limit is None:
                raise ValueError(f"Missing limits for {name}")
            self.lower[i], self.upper[i], self.velocity_limits[i] = (
                float(limit.attrib[k]) for k in ("lower", "upper", "velocity"))
        if (not np.isfinite([self.lower, self.upper, self.velocity_limits]).all()
                or np.any(self.lower >= self.upper) or np.any(self.velocity_limits <= 0)):
            raise ValueError("Invalid joint limits")

        self.independent_indices = np.array([i for i, name in enumerate(JOINT_NAMES)
                                             if actuated[name].find("mimic") is None])
        self.num_independent_joints = len(self.independent_indices)
        self.expansion = np.zeros((21, self.num_independent_joints))
        self.mimic_offset = np.zeros(21)
        resolved = set(self.independent_indices.tolist())
        for column, i in enumerate(self.independent_indices):
            self.expansion[i, column] = 1
        while len(resolved) < 21:
            progressed = False
            for i, name in enumerate(JOINT_NAMES):
                if i in resolved:
                    continue
                mimic = actuated[name].find("mimic")
                parent_name = mimic.get("joint")
                if parent_name not in JOINT_NAMES:
                    raise ValueError(f"Unknown mimic parent: {parent_name}")
                parent = JOINT_NAMES.index(parent_name)
                if parent not in resolved:
                    continue
                multiplier, offset = float(mimic.get("multiplier", 1)), float(mimic.get("offset", 0))
                if not np.isfinite([multiplier, offset]).all():
                    raise ValueError("Invalid mimic relationship")
                self.expansion[i] = multiplier * self.expansion[parent]
                self.mimic_offset[i] = multiplier * self.mimic_offset[parent] + offset
                resolved.add(i)
                progressed = True
            if not progressed:
                raise ValueError("Cyclic mimic relationships")
        self.independent_lower = self.lower[self.independent_indices].copy()
        self.independent_upper = self.upper[self.independent_indices].copy()
        # Intersect dependent-joint limits with each driving joint's limits.
        for i, row in enumerate(self.expansion):
            columns = np.flatnonzero(row)
            if not len(columns):
                if not self.lower[i] <= self.mimic_offset[i] <= self.upper[i]:
                    raise ValueError("Constant mimic joint violates its limits")
                continue
            column = columns[0]
            bounds = sorted([(self.lower[i] - self.mimic_offset[i]) / row[column],
                             (self.upper[i] - self.mimic_offset[i]) / row[column]])
            self.independent_lower[column] = max(self.independent_lower[column], bounds[0])
            self.independent_upper[column] = min(self.independent_upper[column], bounds[1])
        if np.any(self.independent_lower >= self.independent_upper):
            raise ValueError("Mimic limits have no feasible interval")

        # Topological order is independent of the XML ordering of sensor links.
        self.link_names = [self.root_link]
        self._ancestors = [[]]
        self._edges = []
        pending = list(joints)
        while pending:
            progressed = False
            for joint in pending[:]:
                parent = joint.find("parent").attrib["link"]
                child = joint.find("child").attrib["link"]
                if parent not in self.link_names:
                    continue
                if child in self.link_names or child not in links:
                    raise ValueError("URDF is not a link tree")
                kind = joint.attrib["type"]
                if kind not in ("revolute", "fixed"):
                    raise ValueError("Only fixed and revolute joints are supported")
                origin = joint.find("origin")
                xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
                if xyz.shape != (3,) or rpy.shape != (3,) or not np.isfinite([xyz, rpy]).all():
                    raise ValueError("Invalid joint origin")
                rot = Rotation.from_euler("xyz", rpy).as_matrix()
                parent_id = self.link_names.index(parent)
                joint_id = JOINT_NAMES.index(joint.attrib["name"]) if kind == "revolute" else -1
                axis = np.array([0., 0., 1.])
                if joint_id >= 0 and joint.find("axis") is not None:
                    axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
                    raise ValueError("Invalid revolute joint axis")
                axis /= np.linalg.norm(axis)
                x, y, z = axis
                skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                self._edges.append((parent_id, joint_id, xyz, rot, axis, skew, skew @ skew))
                self.link_names.append(child)
                self._ancestors.append(self._ancestors[parent_id] + ([joint_id] if joint_id >= 0 else []))
                pending.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError("URDF has disconnected links or a cycle")
        if set(self.link_names) != set(links):
            raise ValueError("URDF has disconnected links")

        # Landmark 1 is thumb CMC, 2 MCP, 3 IP; other chains are MCP/PIP/DIP.
        point_links = [self.root_link, "f0_link2", "f0_link3", "f0_link4", "f0_link4"]
        for finger in range(1, 5):
            point_links.extend([f"f{finger}_link1", f"f{finger}_link2",
                                f"f{finger}_link3", f"f{finger}_link3"])
        self._point_links = np.array([self.link_names.index(name) for name in point_links])
        self._offsets = np.zeros((21, 3))
        offsets = np.asarray(tip_offsets if tip_offsets is not None else
                             [[0.032, 0, 0]] + [[0.028, 0, 0]] * 4, dtype=float)
        if (offsets.shape != (5, 3) or not np.isfinite(offsets).all()
                or np.any(np.linalg.norm(offsets, axis=1) < 1e-6)):
            raise ValueError("tip_offsets must have shape (5, 3) with nonzero finite vectors")
        self._offsets[[4, 8, 12, 16, 20]] = offsets
        self._point_mask = np.zeros((21, 21), dtype=bool)
        for i, link_id in enumerate(self._point_links):
            self._point_mask[i, self._ancestors[link_id]] = True
        self.neutral = self.expand(np.clip(np.zeros(self.num_independent_joints),
                                          self.independent_lower, self.independent_upper))
        self.neutral_keypoints = self.keypoints(self.neutral)

    def expand(self, independent_qpos):
        q = np.asarray(independent_qpos, dtype=float)
        if q.shape != (self.num_independent_joints,) or not np.isfinite(q).all():
            raise ValueError("Invalid independent joint vector")
        return self.expansion @ q + self.mimic_offset

    def validate_qpos(self, qpos):
        q = np.asarray(qpos, dtype=float)
        if q.shape != (21,) or not np.isfinite(q).all():
            raise ValueError("qpos must be a finite (21,) array in SDK joint order")
        return q

    def keypoints(self, qpos, with_jacobian=False):
        """21 metric points in URDF palm frame; Jacobian shape (21, 3, 21)."""
        q = self.validate_qpos(qpos)
        rotations = np.empty((len(self.link_names), 3, 3))
        positions = np.empty((len(self.link_names), 3))
        rotations[0], positions[0] = np.eye(3), 0
        axes, origins = np.zeros((21, 3)), np.zeros((21, 3))
        for child, (parent, joint, xyz, rot, axis, skew, skew2) in enumerate(self._edges, 1):
            pre_rotation = rotations[parent] @ rot
            positions[child] = positions[parent] + rotations[parent] @ xyz
            if joint >= 0:
                axes[joint] = pre_rotation @ axis
                origins[joint] = positions[child]
                motion = np.eye(3) + np.sin(q[joint]) * skew + (1 - np.cos(q[joint])) * skew2
                rotations[child] = pre_rotation @ motion
            else:
                rotations[child] = pre_rotation
        points = positions[self._point_links] + np.einsum(
            "nij,nj->ni", rotations[self._point_links], self._offsets)
        if not with_jacobian:
            return points
        jacobian = np.cross(axes[None, :, :], points[:, None, :] - origins[None, :, :])
        jacobian *= self._point_mask[:, :, None]
        return points, jacobian.transpose(0, 2, 1)
