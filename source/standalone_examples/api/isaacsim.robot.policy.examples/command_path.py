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

"""Commanded-course and traveled-path rendering for the locomotion standalone examples.

The standalones drive their robots open loop with scripted body-frame velocity phases — the
only input the policies take — and render both what was asked for and what happened:

* the commanded course (green line), integrated from the spawn pose as if the robot tracked
  every command exactly, with a yellow waypoint arrow at each phase boundary showing the
  commanded position *and* heading there; and
* the traveled path (red line), recorded from the robot's measured base position, with a red
  arrow marking its measured pose and heading at those same instants.

Nothing in the control path closes the loop on position, so each yellow/red arrow pair is the
deployed policy's accumulated tracking error in both position and heading: a policy that
under-tracks, drifts in yaw, or strafes weakly shows it directly. Keeping the commands open
loop is deliberate — a position controller would absorb exactly the error these examples exist
to show.
"""

import math

_COMMANDED_COLOR = (0.1, 0.8, 0.2)
_WAYPOINT_COLOR = (1.0, 0.85, 0.1)
_TRAVELED_COLOR = (0.9, 0.2, 0.1)
_ARROW_LENGTH = 0.45
_ARROW_RADIUS = 0.09


def _yaw_from_quaternion(quaternion: object) -> float:
    """Extract the world-frame yaw of a WXYZ quaternion.

    Args:
        quaternion: Quaternion.

    Returns:
        The resulting float.
    """
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _author_pose_arrow(prim_path: str, x: float, y: float, yaw: float, height: float, color: tuple) -> None:
    """Author a flat cone marking one pose: positioned at (x, y), pointing along yaw.

    Args:
        prim_path: Prim path.
        x: X.
        y: Y.
        yaw: Yaw.
        height: Height.
        color: Color.
    """
    from isaacsim.core.experimental.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    cone = UsdGeom.Cone.Define(get_current_stage(), prim_path)
    cone.CreateAxisAttr(UsdGeom.Tokens.x)
    cone.CreateHeightAttr(_ARROW_LENGTH)
    cone.CreateRadiusAttr(_ARROW_RADIUS)
    cone.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    xformable = UsdGeom.Xformable(cone.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(x, y, height))
    xformable.AddRotateZOp().Set(math.degrees(yaw))


def integrate_command_phases(
    phases: list[tuple[int, list[float]]],
    start_position: tuple[float, float],
    frame_dt: float,
    start_yaw: float = 0.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float, float]]]:
    """Integrate a command schedule into the world-frame course it describes.

    Args:
        phases: Command schedule as ``(frames, [vx, vy, yaw_rate])`` entries, one command held
            per app frame, matching the standalone's main loop.
        start_position: World-frame ``(x, y)`` the robot starts from.
        frame_dt: Simulated seconds advanced per app frame (the rendering timestep).
        start_yaw: World-frame start yaw in radians.

    Returns:
        Tuple of the per-frame course points and the ``(x, y, yaw)`` pose at each phase boundary.
    """
    x, y = float(start_position[0]), float(start_position[1])
    yaw = float(start_yaw)
    points = [(x, y)]
    boundary_poses = []
    for frames, (forward, lateral, yaw_rate) in phases:
        for _ in range(int(frames)):
            cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
            x += (cos_yaw * forward - sin_yaw * lateral) * frame_dt
            y += (sin_yaw * forward + cos_yaw * lateral) * frame_dt
            yaw += yaw_rate * frame_dt
            points.append((x, y))
        boundary_poses.append((x, y, yaw))
    return points, boundary_poses


def phase_boundary_frames(phases: list[tuple[int, list[float]]]) -> set:
    """Return the frame indices that end each command phase.

    Args:
        phases: Command schedule as ``(frames, [vx, vy, yaw_rate])`` entries.

    Returns:
        The set of zero-based frame indices at which a phase completes.
    """
    boundaries = set()
    cursor = 0
    for frames, _ in phases:
        cursor += int(frames)
        boundaries.add(cursor - 1)
    return boundaries


def author_command_path(
    phases: list[tuple[int, list[float]]],
    start_position: tuple[float, float],
    frame_dt: float,
    prim_path: str = "/World/CommandedCourse",
    start_yaw: float = 0.0,
    height: float = 0.02,
) -> tuple[float, float, float]:
    """Author the commanded course and its per-phase waypoint arrows as stage geometry.

    Args:
        phases: Command schedule as ``(frames, [vx, vy, yaw_rate])`` entries.
        start_position: World-frame ``(x, y)`` the robot spawns at.
        frame_dt: Simulated seconds advanced per app frame (the rendering timestep).
        prim_path: Stage path of the authored course scope.
        start_yaw: World-frame spawn yaw in radians.
        height: Height above the ground plane the course is drawn at.

    Returns:
        The commanded ``(x, y, yaw)`` pose the course ends at under exact tracking.
    """
    from isaacsim.core.experimental.utils.stage import get_current_stage
    from pxr import Gf, UsdGeom

    points, boundary_poses = integrate_command_phases(phases, start_position, frame_dt, start_yaw)

    curve = UsdGeom.BasisCurves.Define(get_current_stage(), f"{prim_path}/course")
    curve.CreateTypeAttr(UsdGeom.Tokens.linear)
    curve.CreateCurveVertexCountsAttr([len(points)])
    curve.CreatePointsAttr([Gf.Vec3f(x, y, height) for x, y in points])
    curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    curve.CreateWidthsAttr([0.03])
    curve.CreateDisplayColorAttr([Gf.Vec3f(*_COMMANDED_COLOR)])

    for index, (x, y, yaw) in enumerate(boundary_poses):
        _author_pose_arrow(f"{prim_path}/waypoint_{index}", x, y, yaw, height, _WAYPOINT_COLOR)

    print(f"Authored commanded course at {prim_path}: {len(points)} points, {len(boundary_poses)} waypoints")
    return boundary_poses[-1]


class TraveledPath:
    """Growing polyline of a robot's measured base positions, drawn beside its commanded course.

    Args:
        prim_path: Stage path of the authored trail scope.
        height: Height above the ground plane the trail is drawn at.
    """

    def __init__(self, prim_path: str = "/World/TraveledPath", height: float = 0.02) -> None:
        self._prim_path = prim_path
        self._height = height
        self._points = []
        self._curve = None
        self._marks = 0

    @property
    def last_pose(self) -> tuple[float, float, float]:
        """The most recently recorded world-frame ``(x, y, yaw)``."""
        return self._last_pose if self._points else (0.0, 0.0, 0.0)

    def record(self, articulation: object, mark: bool = False) -> None:
        """Extend the trail with the articulation's measured pose.

        Args:
            articulation: Live articulation whose measured world pose extends the trail.
            mark: Also author a red pose arrow here (used at the command phase boundaries, so
                each one pairs with the commanded yellow arrow of that phase).
        """
        from isaacsim.core.experimental.utils.stage import get_current_stage
        from pxr import Gf, UsdGeom

        positions, orientations = articulation.get_world_poses()
        position = positions.numpy().reshape(-1)
        x, y = float(position[0]), float(position[1])
        yaw = _yaw_from_quaternion(orientations.numpy().reshape(-1))
        self._points.append((x, y))
        self._last_pose = (x, y, yaw)

        if mark:
            _author_pose_arrow(f"{self._prim_path}/reached_{self._marks}", x, y, yaw, self._height, _TRAVELED_COLOR)
            self._marks += 1

        if len(self._points) < 2:
            return
        if self._curve is None:
            self._curve = UsdGeom.BasisCurves.Define(get_current_stage(), f"{self._prim_path}/trail")
            self._curve.CreateTypeAttr(UsdGeom.Tokens.linear)
            self._curve.SetWidthsInterpolation(UsdGeom.Tokens.constant)
            self._curve.CreateWidthsAttr([0.03])
            self._curve.CreateDisplayColorAttr([Gf.Vec3f(*_TRAVELED_COLOR)])
        self._curve.GetCurveVertexCountsAttr().Set([len(self._points)])
        self._curve.GetPointsAttr().Set([Gf.Vec3f(x, y, self._height) for x, y in self._points])


def report_tracking(commanded_pose: tuple[float, float, float], traveled: TraveledPath) -> None:
    """Print the commanded and measured end poses with the position and heading gaps.

    Args:
        commanded_pose: Commanded ``(x, y, yaw)`` the course ends at.
        traveled: Recorded trail whose last pose is the measured end pose.
    """
    actual_x, actual_y, actual_yaw = traveled.last_pose
    position_gap = math.hypot(commanded_pose[0] - actual_x, commanded_pose[1] - actual_y)
    heading_error = commanded_pose[2] - actual_yaw
    heading_gap = abs(math.atan2(math.sin(heading_error), math.cos(heading_error)))
    print(
        f"Commanded end: ({commanded_pose[0]:.2f}, {commanded_pose[1]:.2f}, "
        f"{math.degrees(commanded_pose[2]):.0f} deg); "
        f"reached: ({actual_x:.2f}, {actual_y:.2f}, {math.degrees(actual_yaw):.0f} deg); "
        f"tracking gap: {position_gap:.2f} m, {math.degrees(heading_gap):.0f} deg"
    )


if __name__ == "__main__":
    print(
        "command_path.py is a support module and does not run a simulation. Run anymal_standalone.py, "
        "go2_standalone.py, h1_standalone.py, or spot_standalone.py instead."
    )
