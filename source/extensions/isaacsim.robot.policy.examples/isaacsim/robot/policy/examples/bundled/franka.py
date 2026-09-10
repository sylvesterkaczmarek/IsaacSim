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

"""Frozen policy spec for the Franka open-drawer manipulation example.

Franka is the one bundled example with an explicit binding, because its Isaac Lab export is not
descriptor-derivable: the gripper is a ``BinaryJointPositionAction`` whose open/close command
vectors are not exported, and the drawer-handle-relative observation is a custom task term the
exporter skips. The artifact ships no IO descriptor and the spec supplies
:func:`franka_open_drawer_binding` instead, which authors the 31-wide observation (9 + 9
default-relative joint positions and velocities, drawer joint position and velocity, the 3-wide
handle-to-end-effector offset, and the previous 8-wide action). Its action binding decodes seven
continuous arm channels and one binary gripper channel that commands both trained finger joints.

Task terms use explicit names and articulation-relative paths, never link indices or suffix
discovery: ``task_joint_pos``/``task_joint_vel`` read the cabinet's ``drawer_top_joint``, and
``task_frame_rel_pos`` is the ``drawer_handle`` frame position minus the ``end_effector`` frame
position, each with its env-config frame-transformer offset applied in its own frame.
:func:`make_franka_task_state_provider` builds the factory the runner consumes.

PhysX and Newton use engine-specific policy artifacts and exported task configurations.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import EllipsisType
from typing import TYPE_CHECKING

import numpy as np

from ..binding import PolicyBinding, Term, quaternion_wxyz_to_rotation_matrix
from ..spec import PolicyArtifact, PolicySpec

if TYPE_CHECKING:
    from isaacsim.core.experimental.prims import Articulation

    from ..env_config import PolicyEnvConfig

__all__ = [
    "get_franka_spec",
    "make_franka_task_state_provider",
]


# ---------------------------------------------------------------------------------------------
# Trained joint order and fixed task references
# ---------------------------------------------------------------------------------------------

#: Franka Panda DOF order of the multiphysics ``franka.usda`` asset, identical on both importers.
FRANKA_JOINT_NAMES: tuple[str, ...] = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
    "panda_finger_joint1",
    "panda_finger_joint2",
)

#: Raw policy action channels: seven arm joints and one shared binary gripper channel.
FRANKA_ACTION_JOINT_NAMES: tuple[str, ...] = FRANKA_JOINT_NAMES[:8]

_DRAWER_JOINT_NAMES = ("drawer_top_joint",)
_DRAWER_FRAME_PATH = "drawer_handle_top"
_DRAWER_FRAME_OFFSET = (0.305, 0.0, 0.01)
_END_EFFECTOR_FRAME_PATH = "panda_hand"
_END_EFFECTOR_FRAME_OFFSET = (0.0, 0.0, 0.1034)


# ---------------------------------------------------------------------------------------------
# Explicit binding hook (the spec's ``binding`` callable)
# ---------------------------------------------------------------------------------------------


def franka_open_drawer_binding(env_config: PolicyEnvConfig) -> PolicyBinding:
    """Build the explicit 31/8 Franka open-drawer binding.

    The hosted export cannot describe the binary gripper action or custom task observation, so
    the term list is authored here in the trained joint order, with the env config's default
    joint state filling the relative-observation and action offsets by name. The deployed
    articulation must expose the trained joint names literally; binding refuses any that
    are missing.

    Args:
        env_config: Parsed env config of the deployed artifact.

    Returns:
        The fully resolved open-drawer policy binding.

    Raises:
        ValueError: If the env config does not cover the trained joints.
    """
    resolved = FRANKA_JOINT_NAMES
    properties = env_config.joint_properties(resolved)
    width = len(resolved)
    default_pos_by_name = dict(zip(resolved, properties.default_positions))
    arm_joints = resolved[:7]
    finger_joints = resolved[7:]

    return PolicyBinding(
        observation_terms=(
            Term(semantic="joint_pos_rel", width=width, joint_names=resolved, offsets=properties.default_positions),
            Term(semantic="joint_vel_rel", width=width, joint_names=resolved, offsets=properties.default_velocities),
            Term(semantic="task_joint_pos", width=1, task_refs=("cabinet",)),
            Term(semantic="task_joint_vel", width=1, task_refs=("cabinet",)),
            Term(semantic="task_frame_rel_pos", width=3, task_refs=("drawer_handle", "end_effector")),
            Term(semantic="last_action", width=len(FRANKA_ACTION_JOINT_NAMES)),
        ),
        action_terms=(
            Term(
                semantic="joint_position",
                width=len(arm_joints),
                joint_names=arm_joints,
                offsets=tuple(default_pos_by_name[name] for name in arm_joints),
                scale=1.0,
            ),
            Term(
                semantic="binary_joint_position",
                width=1,
                joint_names=finger_joints,
                binary_positions=(0.0, 0.04),
            ),
        ),
    )


# ---------------------------------------------------------------------------------------------
# Explicit task resolution (frames and joints; shared by the task state provider)
# ---------------------------------------------------------------------------------------------


def _resolve_relative_frame_path(
    articulation_path: str,
    link_paths: Sequence[str],
    relative_prim_path: str,
    owner: str,
) -> str:
    """Resolve one explicit articulation-relative frame path and verify that it is a link.

    Args:
        articulation_path: Root prim path of the requirement's declared articulation.
        link_paths: Candidate link prim paths.
        relative_prim_path: Explicit path below ``articulation_path``.
        owner: Owning object description used in error messages.

    Returns:
        The absolute frame prim path.

    Raises:
        ValueError: If the reference does not name a link on the declared articulation.
    """
    relative = str(relative_prim_path).strip()
    path = f"{str(articulation_path).rstrip('/')}/{relative}"
    available = {str(candidate).rstrip("/") for candidate in link_paths}
    if path not in available:
        raise ValueError(
            f"{owner}: explicit frame path {path!r} is not a link of articulation {articulation_path!r}; "
            f"available link paths are {sorted(available)}."
        )
    return path


def resolve_task_joint_indices(
    dof_names: Sequence[str], joint_names: Sequence[str], owner: str = "FrankaTaskStateProvider"
) -> tuple[int, ...]:
    """Resolve an articulation-joints requirement's joint names to DOF indices by name.

    Args:
        dof_names: Literal DOF names of the task articulation, in articulation order.
        joint_names: Ordered joint names declared by the requirement.
        owner: Owning object description used in error messages.

    Returns:
        The DOF index of each requested joint, in request order.

    Raises:
        ValueError: If a requested name is not an articulation DOF.
    """
    available = [str(name) for name in dof_names]
    requested = [str(name) for name in joint_names]

    indices: list[int] = []
    for name in requested:
        try:
            index = available.index(name)
        except ValueError:
            raise ValueError(
                f"{owner}: joint {name!r} is not a DOF of the task articulation; available DOFs are {available}."
            ) from None
        indices.append(index)

    return tuple(indices)


def apply_frame_offset(position: np.ndarray, orientation_wxyz: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Apply a frame-local position offset to a world-frame pose in float32.

    Args:
        position: World-frame frame position as a flat three-element vector.
        orientation_wxyz: World-frame frame orientation as a flat wxyz quaternion.
        offset: Frame-local position offset as a flat three-element vector.

    Returns:
        The offset world-frame position as a flat float32 vector.

    """
    quaternion = np.asarray(orientation_wxyz, dtype=np.float32).reshape(-1)
    rotation = quaternion_wxyz_to_rotation_matrix(quaternion)

    result = np.asarray(position, dtype=np.float32).reshape(-1) + rotation @ np.asarray(
        offset, dtype=np.float32
    ).reshape(-1)
    return result.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Task state provider
# ---------------------------------------------------------------------------------------------


def _flat_float32(values: object) -> np.ndarray:
    """Convert a warp or numpy container to a flat float32 numpy vector.

    Args:
        values: Value sequence to coerce.

    Returns:
        The resulting array.
    """
    to_numpy = getattr(values, "numpy", None)
    if callable(to_numpy):
        values = to_numpy()
    return np.asarray(values, dtype=np.float32).reshape(-1)


class FrankaTaskStateProvider:
    """Task state provider for the Franka open-drawer policy's fixed scene references.

    Resolves the cabinet joint and the two articulation-relative frame paths once at
    construction, with no global link-name search. Each control tick reads one typed,
    namespaced context mapping: cabinet joint position/velocity plus drawer-handle and
    end-effector world poses. Frame-local position offsets are applied before subtraction.

    Args:
        robot: Initialized robot articulation whose link paths satisfy robot-side frames.
        cabinet: Initialized cabinet articulation providing task joints and cabinet-side frames.
        env_config: Exported task configuration providing the cabinet reset state.

    Raises:
        ValueError: If an explicit joint/frame reference does not exist on its articulation.
    """

    def __init__(self, robot: Articulation, cabinet: Articulation, env_config: PolicyEnvConfig) -> None:
        from isaacsim.core.experimental.prims import RigidPrim

        self._cabinet = cabinet
        cabinet_position, cabinet_orientation = env_config.scene_entity_root_pose("cabinet")
        if cabinet_position is None or cabinet_orientation is None:
            raise ValueError("Franka policy env config does not define the cabinet initial root pose.")
        default_pos, default_vel = env_config.scene_entity_joint_default_state("cabinet", cabinet.dof_names)
        cabinet.set_default_state(
            positions=[cabinet_position],
            orientations=[cabinet_orientation],
            linear_velocities=[[0.0, 0.0, 0.0]],
            angular_velocities=[[0.0, 0.0, 0.0]],
            dof_positions=list(default_pos),
            dof_velocities=list(default_vel),
        )
        self.reset()

        drawer_indices = resolve_task_joint_indices(
            cabinet.dof_names, _DRAWER_JOINT_NAMES, owner="FrankaTaskStateProvider('cabinet')"
        )
        self._joint_reads: list[tuple[str, Articulation, tuple[int, ...]]] = [("cabinet", cabinet, drawer_indices)]

        frame_specs = (
            ("drawer_handle", cabinet, _DRAWER_FRAME_PATH, _DRAWER_FRAME_OFFSET),
            ("end_effector", robot, _END_EFFECTOR_FRAME_PATH, _END_EFFECTOR_FRAME_OFFSET),
        )
        self._frame_reads: list[tuple[str, RigidPrim, np.ndarray | None]] = []
        for name, source, relative_path, offset in frame_specs:
            path = _resolve_relative_frame_path(
                str(source.paths[0]), source.link_paths[0], relative_path, f"FrankaTaskStateProvider({name!r})"
            )
            self._frame_reads.append((name, RigidPrim(path), np.asarray(offset, dtype=np.float32)))

    def reset(self) -> None:
        """Restore the cabinet's exported default state."""
        from isaacsim.core.simulation_manager import SimulationManager

        self._cabinet.reset_to_default_state()
        simulation_view = SimulationManager.get_physics_simulation_view()
        if simulation_view is not None:
            simulation_view.update_articulations_kinematic()

    def read(self) -> Mapping[str, object]:
        """Read the current task context.

        Returns:
            Mapping from namespaced context keys to float32 numpy vectors.
        """
        context: dict[str, object] = {}
        for name, articulation, indices in self._joint_reads:
            positions = articulation.get_dof_positions(indices=[0], dof_indices=list(indices))
            velocities = articulation.get_dof_velocities(indices=[0], dof_indices=list(indices))
            context[f"{name}.joint_pos"] = _flat_float32(positions)
            context[f"{name}.joint_vel"] = _flat_float32(velocities)

        poses = [(name, prim.get_world_poses(), offset) for name, prim, offset in self._frame_reads]
        for name, (positions, orientations), offset in poses:
            position = _flat_float32(positions)
            orientation = _flat_float32(orientations)
            context[f"{name}.position"] = (
                apply_frame_offset(position, orientation, offset) if offset is not None else position
            )
            context[f"{name}.orientation"] = orientation

        return context


def make_franka_task_state_provider(
    cabinet: Articulation, env_config: PolicyEnvConfig
) -> Callable[[Articulation], FrankaTaskStateProvider]:
    """Build the task-state-provider factory for the generic policy runner.

    Args:
        cabinet: Caller-owned cabinet articulation, valid on the stage before the runner initializes.
        env_config: Exported task config supplying the cabinet default state.

    Returns:
        Factory called by the runner with the spawned robot articulation.
    """

    def factory(robot: Articulation) -> FrankaTaskStateProvider:
        return FrankaTaskStateProvider(robot, cabinet, env_config)

    return factory


# ---------------------------------------------------------------------------------------------
# Bundled spec
# ---------------------------------------------------------------------------------------------


@functools.cache
def _get_default_franka_spec() -> PolicySpec:
    """Build the cached default Franka open-drawer policy spec.

    Returns:
        The resulting :class:`PolicySpec`.
    """
    from isaacsim.storage.native import get_assets_root_path

    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Could not resolve the assets root path; check the Isaac Sim asset configuration.")

    policy_dir = f"{assets_root_path}/Isaac/Samples/Policies/Franka_Policies/Open_Drawer_Policy"
    extension_path = Path(__file__).resolve().parents[5]
    newton_policy_dir = extension_path / "data" / "tests" / "franka_newton_policy"
    return PolicySpec(
        name="franka_open_drawer",
        engines={
            "physx": PolicyArtifact.from_files(
                f"{policy_dir}/policy.pt",
                f"{policy_dir}/env.yaml",
                model_sha256="6b911183a954dc107832d9d31392a1f9b51860ffe9ed55f7f3505e73f4f5ac3a",
            ),
            "newton": PolicyArtifact.from_files(
                str(newton_policy_dir / "policy.pt"),
                str(newton_policy_dir / "env.yaml"),
                model_sha256="8f1a039c89515f5400c0b79b0495f2f003f940b45a59e42b8f9ecbe8478cdc87",
            ),
        },
        usd_path=None,
        # The legacy deployment spawned at the asset-authored pose (origin, identity); pinned
        # here so the runner's env-config fallback can never drift the spawn.
        default_spawn_position=(0.0, 0.0, 0.0),
        default_spawn_orientation=(1.0, 0.0, 0.0, 0.0),
        binding=franka_open_drawer_binding,
    )


def get_franka_spec(
    *,
    engines: Mapping[str, PolicyArtifact] | EllipsisType = ...,
    usd_path: str | Mapping[str, str] | None | EllipsisType = ...,
    name: str | None | EllipsisType = ...,
    default_spawn_position: tuple[float, float, float] | None | EllipsisType = ...,
    default_spawn_orientation: tuple[float, float, float, float] | None | EllipsisType = ...,
    zero_targets_on_initialize: bool | EllipsisType = ...,
    binding: Callable | None | EllipsisType = ...,
) -> PolicySpec:
    """Return the bundled Franka spec; omitted :class:`PolicySpec` fields keep their defaults.

    Args:
        engines: Per-engine artifacts keyed by engine name.
        usd_path: Robot USD path, or an engine-keyed mapping of them.
        name: Diagnostic label used in runner error messages.
        default_spawn_position: World-frame spawn position used when the caller passes none.
        default_spawn_orientation: World-frame WXYZ spawn orientation for the same fallback.
        zero_targets_on_initialize: Reset to the default state and zero all targets on initialize.
        binding: Explicit binding hook used instead of descriptor derivation.

    Returns:
        The resulting :class:`PolicySpec`.
    """
    overrides = {field: value for field, value in locals().items() if value is not ...}
    spec = _get_default_franka_spec()
    return replace(spec, **overrides) if overrides else spec
