# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared USD joint kinematics helpers for SysID analytical paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .articulation_utils import find_articulation_root

if TYPE_CHECKING:
    from pxr import Gf, Usd, UsdPhysics


@dataclass(frozen=True)
class UsdJointEdge:
    """One USD revolute/prismatic joint edge between two rigid bodies."""

    joint_path: str
    joint_index: int
    joint_type: str
    axis: str
    body0_path: str
    body1_path: str
    local0: Gf.Matrix4d
    local1: Gf.Matrix4d


def collect_usd_joint_edges(stage: Usd.Stage, robot_path: str) -> tuple[list[UsdJointEdge], str]:
    """Collect serial joint edges under an articulation root or its parent scope.

    Args:
        stage: USD stage used by the operation.
        robot_path: Value supplied for ``robot_path``.

    Returns:
        Result produced by the operation.
    """
    from pxr import Sdf

    root_path = find_articulation_root(stage, robot_path) or robot_path
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim or not root_prim.IsValid():
        return [], f"Robot prim path is not valid: {robot_path}"

    edges = _collect_joint_edges_from_prim(root_prim)
    if edges:
        return edges, ""

    search_path = Sdf.Path(root_path).GetParentPath()
    while search_path and search_path != Sdf.Path.absoluteRootPath:
        parent_prim = stage.GetPrimAtPath(str(search_path))
        if parent_prim and parent_prim.IsValid():
            edges = _collect_joint_edges_from_prim(parent_prim, body_root_path=root_path)
            if edges:
                return edges, ""
        search_path = search_path.GetParentPath()

    return [], ""


def usd_joint_local_transform(joint: UsdPhysics.Joint, body_index: int) -> Gf.Matrix4d:
    """Return the authored joint frame transform relative to body0 or body1.

    Args:
        joint: Value supplied for ``joint``.
        body_index: Value supplied for ``body_index``.

    Returns:
        Result produced by the operation.
    """
    from pxr import Gf

    translate_attr = joint.GetLocalPos0Attr() if body_index == 0 else joint.GetLocalPos1Attr()
    rotate_attr = joint.GetLocalRot0Attr() if body_index == 0 else joint.GetLocalRot1Attr()
    translate = translate_attr.Get() if translate_attr else None
    rotate = rotate_attr.Get() if rotate_attr else None
    translate = Gf.Vec3f(0.0, 0.0, 0.0) if translate is None else translate
    rotate = Gf.Quatf(1.0, 0.0, 0.0, 0.0) if rotate is None else rotate
    local = Gf.Matrix4d()
    local.SetTranslate(Gf.Vec3d(translate))
    local.SetRotateOnly(Gf.Quatd(rotate.GetReal(), *rotate.GetImaginary()).GetNormalized())
    return local


def usd_joint_motion_transform(edge: UsdJointEdge, q_value: float) -> Gf.Matrix4d:
    """Return joint motion in the USD joint frame.

    Args:
        edge: Value supplied for ``edge``.
        q_value: Value supplied for ``q_value``.

    Returns:
        Result produced by the operation.
    """
    from pxr import Gf

    axis = axis_vec(edge.axis)
    if edge.joint_type == "prismatic":
        return Gf.Matrix4d(1.0).SetTranslate(
            Gf.Vec3d(axis[0] * float(q_value), axis[1] * float(q_value), axis[2] * float(q_value))
        )
    return Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(axis, float(np.rad2deg(q_value))))


def axis_vec(axis: str) -> Gf.Vec3d:
    """Return the USD basis vector for an axis token.

    Args:
        axis: Value supplied for ``axis``.

    Returns:
        Result produced by the operation.
    """
    from pxr import Gf

    if str(axis).upper() == "Y":
        return Gf.Vec3d(0.0, 1.0, 0.0)
    if str(axis).upper() == "Z":
        return Gf.Vec3d(0.0, 0.0, 1.0)
    return Gf.Vec3d(1.0, 0.0, 0.0)


def child_to_parent_body_transform(edge: UsdJointEdge, q_value: float) -> Gf.Matrix4d:
    """Return a child-body-to-parent-body transform for the edge at q.

    Args:
        edge: Value supplied for ``edge``.
        q_value: Value supplied for ``q_value``.

    Returns:
        Result produced by the operation.
    """
    return edge.local1.GetInverse() * usd_joint_motion_transform(edge, q_value) * edge.local0


def matrix_to_spatial_motion_transform(matrix: Gf.Matrix4d) -> np.ndarray:
    """Spatial motion transform (child frame <- parent frame) from a child-to-parent rigid transform.

    ``matrix`` maps child-body points to parent-body points in USD's row-vector
    convention, so ``arr[:3, :3]`` is the column-convention parent-to-child
    rotation ``E`` and ``arr[3, :3]`` is the child origin position ``t`` in
    parent coordinates. The Plucker transform carrying parent-frame motion
    vectors ``[angular, linear]`` into the child frame is then
    ``[[E, 0], [-E skew(t), E]]`` (Featherstone ``^childX_parent``) -- the
    inverse direction of the input, matching the RNEA ``xup`` convention.

    Args:
        matrix: Value supplied for ``matrix``.

    Returns:
        Result produced by the operation.
    """
    arr = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    rot = arr[:3, :3]
    trans = arr[3, :3]
    out = np.zeros((6, 6), dtype=np.float64)
    out[:3, :3] = rot
    out[3:, :3] = -rot @ _skew(trans)
    out[3:, 3:] = rot
    return out


def _collect_joint_edges_from_prim(root_prim: Usd.Prim, *, body_root_path: str = "") -> list[UsdJointEdge]:
    from pxr import Usd, UsdPhysics

    edges: list[UsdJointEdge] = []
    for prim in Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies()):
        joint_type = ""
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joint_type = "revolute"
            joint_schema = UsdPhysics.RevoluteJoint(prim)
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            joint_type = "prismatic"
            joint_schema = UsdPhysics.PrismaticJoint(prim)
        else:
            continue

        base_joint = UsdPhysics.Joint(prim)
        body0 = base_joint.GetBody0Rel().GetTargets()
        body1 = base_joint.GetBody1Rel().GetTargets()
        if not (body0 and body1):
            continue
        body0_path = str(body0[0])
        body1_path = str(body1[0])
        if body_root_path and not (
            _path_is_under(body0_path, body_root_path) and _path_is_under(body1_path, body_root_path)
        ):
            continue
        axis = joint_schema.GetAxisAttr().Get() or "X"
        edges.append(
            UsdJointEdge(
                joint_path=str(prim.GetPath()),
                joint_index=len(edges),
                joint_type=joint_type,
                axis=str(axis).upper(),
                body0_path=body0_path,
                body1_path=body1_path,
                local0=usd_joint_local_transform(base_joint, 0),
                local1=usd_joint_local_transform(base_joint, 1),
            )
        )
    return edges


def _path_is_under(path: str, root_path: str) -> bool:
    path = str(path).rstrip("/")
    root_path = str(root_path).rstrip("/")
    return path == root_path or path.startswith(f"{root_path}/")


def _skew(v: np.ndarray) -> np.ndarray:
    vec = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -vec[2], vec[1]], [vec[2], 0.0, -vec[0]], [-vec[1], vec[0], 0.0]],
        dtype=np.float64,
    )
