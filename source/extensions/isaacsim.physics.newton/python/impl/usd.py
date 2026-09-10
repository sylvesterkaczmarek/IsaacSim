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

"""USD synchronization utilities for Newton physics simulation.

Stop write-back intentionally supports only one existing, non-animated
``xformOp:transform`` with uniform world scale. Other xform stacks are rejected
before modification. This matches the supplied H1 asset.
"""

from __future__ import annotations

import math

import carb
import newton
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom


class UsdManager:
    """Manager for authoring Newton simulation state to a USD stage.

    Args:
        stage: USD stage to update.
    """

    _matrix_tolerance = 1.0e-9

    def __init__(self, stage: Usd.Stage) -> None:
        self.stage = stage

    def snapshot_body_world_transforms(
        self, model: newton.Model, state: newton.State, scene_scale: float
    ) -> dict[str, Gf.Matrix4d]:
        """Return Newton body world transforms in USD stage units.

        Args:
            model: Newton model that defines the bodies.
            state: Newton state containing the body transforms.
            scene_scale: Scale factor from Newton meters to USD units.

        Returns:
            Mapping from body path to its world transform.
        """
        if state.body_q is None:
            raise RuntimeError("Newton body state is unavailable at Stop.")

        body_transforms = state.body_q.numpy()
        if len(body_transforms) != len(model.body_label):
            raise RuntimeError("Newton body transforms and body labels have different lengths at Stop.")

        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        world_transforms = {}
        for body_path, body_transform in zip(model.body_label, body_transforms):
            prim = self._get_writable_prim(body_path)
            world_scale = _get_world_scale(prim, xform_cache)
            if max(world_scale) - min(world_scale) > self._matrix_tolerance:
                raise RuntimeError(
                    f"Newton Stop pose preservation does not support non-uniform world scale on {body_path}."
                )

            transform = Gf.Transform()
            transform.SetTranslation(Gf.Vec3d(*(float(value) * scene_scale for value in body_transform[:3])))
            transform.SetRotation(
                Gf.Rotation(
                    Gf.Quatd(
                        float(body_transform[6]),
                        Gf.Vec3d(*(float(value) for value in body_transform[3:6])),
                    )
                )
            )
            transform.SetScale(Gf.Vec3d(*world_scale))
            world_transforms[body_path] = transform.GetMatrix()
        return world_transforms

    @staticmethod
    def snapshot_joint_positions(
        model: newton.Model, state: newton.State, scene_scale: float
    ) -> dict[str, tuple[str, float]]:
        """Return supported MuJoCo joint positions in USD units.

        Args:
            model: Newton model that defines the joints.
            state: Newton state to snapshot.
            scene_scale: Scale factor from Newton meters to USD units.

        Returns:
            Mapping from joint path to its JointStateAPI instance name and position.
        """
        if state.joint_q is None:
            raise RuntimeError("Newton joint state is unavailable at Stop.")

        joint_q = state.joint_q.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_types = model.joint_type.numpy()
        if len(joint_types) != len(model.joint_label) or len(joint_q_start) != len(model.joint_label) + 1:
            raise RuntimeError("Newton joint metadata is incomplete at Stop.")

        joint_positions = {}
        unsupported_joints = []
        for joint_index, joint_path in enumerate(model.joint_label):
            joint_type = joint_types[joint_index]
            if joint_type == newton.JointType.FIXED:
                continue
            if joint_type not in (
                newton.JointType.REVOLUTE,
                newton.JointType.PRISMATIC,
            ):
                if joint_q_start[joint_index + 1] > joint_q_start[joint_index]:
                    unsupported_joints.append(f"{joint_path} (type {int(joint_type)})")
                continue

            position_index = int(joint_q_start[joint_index])
            if position_index < 0 or position_index >= len(joint_q):
                raise RuntimeError(f"Newton joint position is unavailable for {joint_path}.")
            position = float(joint_q[position_index])
            if joint_type == newton.JointType.REVOLUTE:
                joint_positions[joint_path] = ("angular", math.degrees(position))
            else:
                joint_positions[joint_path] = ("linear", position * scene_scale)

        if unsupported_joints:
            carb.log_warn(
                "[Newton] Stop pose preservation skipped unsupported joint coordinates: "
                + ", ".join(unsupported_joints)
            )
        return joint_positions

    def write_simulation_state(
        self,
        world_transforms: dict[str, Gf.Matrix4d],
        joint_positions: dict[str, tuple[str, float]],
    ) -> None:
        """Author a stopped Newton pose to USD after validating every target.

        Args:
            world_transforms: Target world transform for each USD body prim.
            joint_positions: JointStateAPI instance name and position for each supported joint.
        """
        edit_layer = self.stage.GetEditTarget().GetLayer()
        if edit_layer is None or not edit_layer.permissionToEdit:
            raise RuntimeError("The current USD edit target is not writable.")

        body_writes = self._prepare_body_writes(world_transforms)
        self._validate_joint_targets(joint_positions)
        with Sdf.ChangeBlock():
            for matrix_op, local_transform in body_writes:
                matrix_op.Set(local_transform, Usd.TimeCode.Default())
            for joint_path, (instance_name, position) in joint_positions.items():
                joint_state = PhysxSchema.JointStateAPI.Apply(self.stage.GetPrimAtPath(joint_path), instance_name)
                joint_state.CreatePositionAttr().Set(position)

    def write_newton_state(self, model: newton.Model, state: newton.State, scene_scale: float) -> None:
        """Snapshot and author the current Newton simulation state."""
        world_transforms = self.snapshot_body_world_transforms(model, state, scene_scale)
        joint_positions = self.snapshot_joint_positions(model, state, scene_scale)
        self.write_simulation_state(world_transforms, joint_positions)

    def _prepare_body_writes(
        self, world_transforms: dict[str, Gf.Matrix4d]
    ) -> list[tuple[UsdGeom.XformOp, Gf.Matrix4d]]:
        """Calculate and validate all local transforms before authoring."""
        xformables = {
            body_path: UsdGeom.Xformable(self._get_writable_prim(body_path)) for body_path in world_transforms
        }
        if any(not xformable for xformable in xformables.values()):
            raise RuntimeError("A Newton body prim is not transformable.")

        future_world_cache = {}
        body_writes = []
        for body_path in sorted(world_transforms, key=lambda path: path.count("/")):
            xformable = xformables[body_path]
            if xformable.GetResetXformStack():
                local_transform = world_transforms[body_path]
            else:
                parent_world = self._compute_future_world(
                    xformable.GetPrim().GetParent(),
                    world_transforms,
                    future_world_cache,
                )
                local_transform = world_transforms[body_path] * parent_world.GetInverse()

            if _matrices_are_close(
                local_transform,
                xformable.GetLocalTransformation(Usd.TimeCode.Default()),
            ):
                continue
            body_writes.append((self._get_matrix_op(xformable), local_transform))
        return body_writes

    def _compute_future_world(
        self,
        prim: Usd.Prim,
        body_worlds: dict[str, Gf.Matrix4d],
        cache: dict[str, Gf.Matrix4d],
    ) -> Gf.Matrix4d:
        """Return a prim world transform after target body poses are applied."""
        if not prim or prim.IsPseudoRoot():
            return Gf.Matrix4d(1.0)

        prim_path = str(prim.GetPath())
        if prim_path in body_worlds:
            return body_worlds[prim_path]
        if prim_path in cache:
            return cache[prim_path]

        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            world = self._compute_future_world(prim.GetParent(), body_worlds, cache)
        else:
            local = xformable.GetLocalTransformation(Usd.TimeCode.Default())
            world = local
            if not xformable.GetResetXformStack():
                world *= self._compute_future_world(prim.GetParent(), body_worlds, cache)
        cache[prim_path] = world
        return world

    @staticmethod
    def _get_matrix_op(xformable: UsdGeom.Xformable) -> UsdGeom.XformOp:
        """Return the existing writable matrix op."""
        body_path = str(xformable.GetPrim().GetPath())
        ops = xformable.GetOrderedXformOps()
        if not (
            len(ops) == 1
            and ops[0].GetOpType() == UsdGeom.XformOp.TypeTransform
            and ops[0].GetName() == "xformOp:transform"
            and not ops[0].IsInverseOp()
            and ops[0].GetNumTimeSamples() == 0
        ):
            raise RuntimeError(f"Unsupported xform-op stack on {body_path}.")
        return ops[0]

    def _validate_joint_targets(self, joint_positions: dict[str, tuple[str, float]]) -> None:
        """Validate all joint targets before authoring."""
        for joint_path, (instance_name, _position) in joint_positions.items():
            prim = self._get_writable_prim(joint_path)
            joint_state = PhysxSchema.JointStateAPI(prim, instance_name)
            if not joint_state and not PhysxSchema.JointStateAPI.CanApply(prim, instance_name):
                raise RuntimeError(f"JointStateAPI cannot be applied to {joint_path}.")

    def _get_writable_prim(self, path: str) -> Usd.Prim:
        """Return an existing non-instance prim."""
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.IsInstanceProxy():
            raise RuntimeError(f"Newton cannot write to prim: {path}.")
        return prim


def _get_world_scale(prim: Usd.Prim, xform_cache: UsdGeom.XformCache) -> tuple[float, float, float]:
    """Return the composed world scale of a USD prim."""
    scale = Gf.Transform(xform_cache.GetLocalToWorldTransform(prim)).GetScale()
    return float(scale[0]), float(scale[1]), float(scale[2])


def _matrices_are_close(first: Gf.Matrix4d, second: Gf.Matrix4d) -> bool:
    """Return whether two matrices are equal within the write tolerance."""
    return all(
        abs(float(first[row][column]) - float(second[row][column])) <= UsdManager._matrix_tolerance
        for row in range(4)
        for column in range(4)
    )
