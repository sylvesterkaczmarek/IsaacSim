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

"""Stateless motion helpers for scripted teleop tests and data generation.

These functions are not used by the live teleop runtime. They let tests,
examples, and skills feed deterministic pose trajectories into existing teleop
controllers and wait for simulated actors to converge before advancing.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Mapping, Sequence

import numpy as np

Pose = tuple[tuple[float, float, float], tuple[float, float, float, float]]
PoseMap = Mapping[str, Pose]
MotionResult = dict[str, bool | int | float]
ApplyTargetsCallback = Callable[[PoseMap], None]
ReadPosesCallback = Callable[[], PoseMap]
UpdateCallback = Callable[[], Awaitable[None]]


async def wait_for_controllers_running_async(
    controllers: Mapping[str, object],
    update_async: UpdateCallback,
    *,
    max_steps: int = 30,
) -> bool:
    """Wait until every side's teleop motion controller is running.

    Args:
        controllers: Motion controllers keyed by their configured side.
        update_async: Function that advances one simulation step.
        max_steps: Maximum simulation steps to wait after the initial check.

    Returns:
        Whether all controllers reported running within the step limit.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")

    def all_running() -> bool:
        return all(bool(getattr(controller, "is_running")(side)) for side, controller in controllers.items())

    if all_running():
        return True
    for _ in range(max_steps):
        await update_async()
        if all_running():
            return True
    return False


async def set_debug_grasp_async(
    teleop_manager: object,
    sides: Sequence[str],
    close: bool,
    update_async: UpdateCallback,
    *,
    settle_steps: int = 60,
) -> None:
    """Set debug grasp triggers and advance the simulation while they settle.

    Args:
        teleop_manager: Configured teleop manager receiving debug triggers.
        sides: Controller sides whose grasp targets should change.
        close: ``True`` to close and ``False`` to open.
        update_async: Function that advances one simulation step.
        settle_steps: Number of simulation steps after setting the triggers.
    """
    if settle_steps < 0:
        raise ValueError("settle_steps must be nonnegative")
    trigger = 1.0 if close else 0.0
    for side in dict.fromkeys(sides):
        getattr(teleop_manager, "set_debug_trigger")(side, trigger)
    for _ in range(settle_steps):
        await update_async()


def make_pose(
    position: Sequence[float],
    orientation: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
) -> Pose:
    """Validate and normalize a ``(position, wxyz orientation)`` pose tuple.

    Args:
        position: Position with three elements.
        orientation: Quaternion in ``wxyz`` order with four elements.

    Returns:
        Normalized pose tuple.
    """
    position_array = np.asarray(position, dtype=np.float64)
    orientation_array = np.asarray(orientation, dtype=np.float64)
    if position_array.shape != (3,):
        raise ValueError(f"Expected a 3D position, got shape {position_array.shape}")
    if orientation_array.shape != (4,):
        raise ValueError(f"Expected a wxyz quaternion, got shape {orientation_array.shape}")
    norm = float(np.linalg.norm(orientation_array))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Orientation quaternion must have nonzero length")
    return (
        tuple(float(value) for value in position_array),
        tuple(float(value) for value in orientation_array / norm),
    )


def set_debug_markers_world_poses(markers_manager: object, targets: PoseMap) -> None:
    """Set teleop debug markers from world-space target poses.

    The marker manager authors controller markers relative to its tracking-origin
    marker. This helper keeps that coordinate conversion out of scripted tests
    and examples while leaving scene and camera authoring to the core APIs.

    Args:
        markers_manager: Configured teleop markers manager.
        targets: World-space marker poses keyed by marker name.
    """
    import isaacsim.core.experimental.utils.transform as transform_utils
    from isaacsim.core.experimental.prims import XformPrim

    origin_path = markers_manager.MARKER_PATHS.get("origin")
    existing_paths, _ = XformPrim.resolve_paths(origin_path or "")
    if not existing_paths:
        raise RuntimeError("Teleop tracking-origin marker is unavailable")
    origin_positions, origin_orientations = XformPrim(existing_paths[0]).get_world_poses()
    origin_position = np.asarray(origin_positions.numpy(), dtype=np.float64).reshape(-1, 3)[0]
    origin_orientation = np.asarray(origin_orientations.numpy(), dtype=np.float64).reshape(-1, 4)[0]
    inverse_origin_orientation = transform_utils.quaternion_conjugate(origin_orientation)

    for name, target in targets.items():
        position, orientation = make_pose(*target)
        local_position = transform_utils.transform_world_to_local(position, origin_position, origin_orientation).numpy()
        local_orientation = transform_utils.quaternion_multiplication(inverse_origin_orientation, orientation).numpy()
        w, x, y, z = local_orientation
        markers_manager.update_marker_transform(name, tuple(local_position.tolist()), (x, y, z, w))


async def move_debug_markers_to_world_targets_async(
    markers_manager: object,
    controlled_prims: Mapping[str, object],
    targets: PoseMap,
    update_async: UpdateCallback,
    *,
    sample_count: int = 100,
    position_tolerance: float = 0.01,
    orientation_tolerance: float = math.radians(3.0),
    max_steps_per_target: int = 30,
) -> MotionResult:
    """Move teleop debug markers and wait for controlled prims to converge.

    Args:
        markers_manager: Configured teleop markers manager.
        controlled_prims: Experimental-core prim handles keyed like ``targets``.
        targets: Final world-space marker poses.
        update_async: Function that advances one simulation step.
        sample_count: Number of interpolated samples including both endpoints.
        position_tolerance: Required controlled-prim position accuracy in meters.
        orientation_tolerance: Required controlled-prim orientation accuracy in radians.
        max_steps_per_target: Maximum simulation steps allowed for each sample.

    Returns:
        Motion result dictionary.
    """
    if set(controlled_prims) != set(targets):
        raise ValueError("Controlled-prim and target names must match")

    starts: dict[str, Pose] = {}
    for name in targets:
        marker_pose = markers_manager.get_marker_world_pose(name)
        if marker_pose is None:
            raise RuntimeError(f"Teleop debug marker '{name}' is unavailable")
        position, orientation_xyzw = marker_pose
        starts[name] = make_pose(position, (orientation_xyzw[3], *orientation_xyzw[:3]))

    def read_poses() -> dict[str, Pose]:
        measured: dict[str, Pose] = {}
        for name, prim in controlled_prims.items():
            positions, orientations = prim.get_world_poses()
            position = np.asarray(positions.numpy(), dtype=np.float64).reshape(-1, 3)[0]
            orientation = np.asarray(orientations.numpy(), dtype=np.float64).reshape(-1, 4)[0]
            measured[name] = make_pose(position, orientation)
        return measured

    return await move_to_pose_targets_async(
        starts,
        targets,
        lambda poses: set_debug_markers_world_poses(markers_manager, poses),
        read_poses,
        update_async,
        sample_count=sample_count,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        max_steps_per_target=max_steps_per_target,
    )


def interpolate_pose_targets(starts: PoseMap, targets: PoseMap, sample_count: int = 100) -> list[dict[str, Pose]]:
    """Generate synchronized linear-position and quaternion-slerp pose samples.

    Args:
        starts: Initial poses keyed by controlled actor name.
        targets: Final poses keyed by controlled actor name.
        sample_count: Number of samples including both endpoints.

    Returns:
        Synchronized pose samples ordered from start to target.
    """
    if set(starts) != set(targets):
        raise ValueError("Start and target actor names must match")
    if not starts:
        raise ValueError("Pose targets must not be empty")
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2")

    fractions = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    trajectories: dict[str, list[Pose]] = {}
    for name in starts:
        start_position, start_orientation = make_pose(*starts[name])
        target_position, target_orientation = make_pose(*targets[name])
        start_position_array = np.asarray(start_position, dtype=np.float64)
        target_position_array = np.asarray(target_position, dtype=np.float64)
        positions = (
            start_position_array[None, :] + fractions[:, None] * (target_position_array - start_position_array)[None, :]
        )
        start_orientation_array = np.asarray(start_orientation, dtype=np.float64)
        target_orientation_array = np.asarray(target_orientation, dtype=np.float64)
        dot = float(np.dot(start_orientation_array, target_orientation_array))
        if dot < 0.0:
            target_orientation_array = -target_orientation_array
            dot = -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > 1.0 - 1e-8:
            orientations_wxyz = (
                start_orientation_array[None, :]
                + fractions[:, None] * (target_orientation_array - start_orientation_array)[None, :]
            )
            orientations_wxyz /= np.linalg.norm(orientations_wxyz, axis=1, keepdims=True)
        else:
            angle = math.acos(dot)
            inverse_sine = 1.0 / math.sin(angle)
            start_weights = np.sin((1.0 - fractions) * angle) * inverse_sine
            target_weights = np.sin(fractions * angle) * inverse_sine
            orientations_wxyz = (
                start_weights[:, None] * start_orientation_array[None, :]
                + target_weights[:, None] * target_orientation_array[None, :]
            )
        trajectories[name] = [
            make_pose(position, orientation) for position, orientation in zip(positions, orientations_wxyz)
        ]
    return [{name: poses[index] for name, poses in trajectories.items()} for index in range(sample_count)]


def _pose_errors(targets: PoseMap, measured: PoseMap) -> tuple[float, float]:
    missing = set(targets) - set(measured)
    if missing:
        raise RuntimeError(f"Missing measured poses for actors: {', '.join(sorted(missing))}")

    position_errors = []
    orientation_errors = []
    for name, (target_position, target_orientation) in targets.items():
        measured_position, measured_orientation = make_pose(*measured[name])
        position_errors.append(
            float(np.linalg.norm(np.asarray(measured_position) - np.asarray(target_position, dtype=np.float64)))
        )
        dot = abs(
            float(
                np.dot(
                    np.asarray(measured_orientation),
                    np.asarray(target_orientation, dtype=np.float64),
                )
            )
        )
        orientation_errors.append(2.0 * math.acos(float(np.clip(dot, -1.0, 1.0))))
    return max(position_errors, default=0.0), max(orientation_errors, default=0.0)


async def execute_pose_trajectory_async(
    trajectory: Sequence[PoseMap],
    apply_targets: ApplyTargetsCallback,
    read_poses: ReadPosesCallback,
    update_async: UpdateCallback,
    *,
    position_tolerance: float = 0.01,
    orientation_tolerance: float = math.radians(3.0),
    max_steps_per_target: int = 30,
) -> MotionResult:
    """Execute precomputed pose samples and advance after each target converges.

    Args:
        trajectory: Ordered target poses keyed by controlled actor name.
        apply_targets: Function that sends one sample to the controllers.
        read_poses: Function that returns measured actor poses.
        update_async: Function that advances one simulation step.
        position_tolerance: Required position accuracy in meters.
        orientation_tolerance: Required orientation accuracy in radians.
        max_steps_per_target: Maximum simulation steps allowed for each sample.

    Returns:
        Dictionary containing ``reached``, ``completed_samples``,
        ``total_steps``, ``position_error``, and ``orientation_error``.
    """
    if not trajectory:
        raise ValueError("Trajectory must contain at least one target")
    if position_tolerance < 0.0 or orientation_tolerance < 0.0:
        raise ValueError("Trajectory tolerances must be nonnegative")
    if max_steps_per_target < 1:
        raise ValueError("max_steps_per_target must be positive")

    completed_samples = 0
    total_steps = 0
    position_error = math.inf
    orientation_error = math.inf
    for targets in trajectory:
        apply_targets(targets)
        for _ in range(max_steps_per_target):
            await update_async()
            total_steps += 1
            position_error, orientation_error = _pose_errors(targets, read_poses())
            if position_error <= position_tolerance and orientation_error <= orientation_tolerance:
                completed_samples += 1
                break
        else:
            return {
                "reached": False,
                "completed_samples": completed_samples,
                "total_steps": total_steps,
                "position_error": position_error,
                "orientation_error": orientation_error,
            }

    return {
        "reached": True,
        "completed_samples": completed_samples,
        "total_steps": total_steps,
        "position_error": position_error,
        "orientation_error": orientation_error,
    }


async def move_to_pose_targets_async(
    starts: PoseMap,
    targets: PoseMap,
    apply_targets: ApplyTargetsCallback,
    read_poses: ReadPosesCallback,
    update_async: UpdateCallback,
    *,
    sample_count: int = 100,
    position_tolerance: float = 0.01,
    orientation_tolerance: float = math.radians(3.0),
    max_steps_per_target: int = 30,
) -> MotionResult:
    """Interpolate and execute one scripted controller move.

    This is the main convenience function for scripted teleop validation.

    Args:
        starts: Initial poses keyed by controlled actor name.
        targets: Final poses keyed by controlled actor name.
        apply_targets: Function that sends one sample to the controllers.
        read_poses: Function that returns measured actor poses.
        update_async: Function that advances one simulation step.
        sample_count: Number of interpolated samples including both endpoints.
        position_tolerance: Required position accuracy in meters.
        orientation_tolerance: Required orientation accuracy in radians.
        max_steps_per_target: Maximum simulation steps allowed for each sample.

    Returns:
        Motion result dictionary.
    """
    return await execute_pose_trajectory_async(
        interpolate_pose_targets(starts, targets, sample_count),
        apply_targets,
        read_poses,
        update_async,
        position_tolerance=position_tolerance,
        orientation_tolerance=orientation_tolerance,
        max_steps_per_target=max_steps_per_target,
    )
