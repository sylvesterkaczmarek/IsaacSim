# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Teleop UI floating-controller debug mode.

Creates two dynamic rigid-body cube handles, configures the live
``Floating Controller`` panel for left/right tracking, enables
``Debug Mode``, moves the left/right markers, and verifies
the cube handles move toward those markers while the timeline is running.
"""

from __future__ import annotations

import os
import tempfile

import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.timeline
import omni.ui as ui
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
from isaacsim.core.simulation_manager import PhysicsScene
from isaacsim.replicator.teleop.ui.teleop_ui_extension import TeleopUIExtension
from isaacsim.test.utils import MenuUITestCase

WINDOW_TITLE = TeleopUIExtension.WINDOW_NAME
MENU_PATH = f"{TeleopUIExtension.MENU_GROUP}/{TeleopUIExtension.WINDOW_NAME}"

_LEFT_CUBE_PATH = "/World/LeftFloatingCube"
_RIGHT_CUBE_PATH = "/World/RightFloatingCube"
_LEFT_INITIAL_POSITION = (0.0, 0.3, 1.0)
_RIGHT_INITIAL_POSITION = (0.0, -0.3, 1.0)
_LEFT_TARGET_POSITION = (0.45, 0.3, 1.0)
_RIGHT_TARGET_POSITION = (0.45, -0.3, 1.0)
_CUBE_SIZE = 0.15
_CUBE_MASS = 1.0
_IDENTITY_XYZW = (0.0, 0.0, 0.0, 1.0)


def _create_dynamic_cube(
    *,
    path: str,
    position: tuple[float, float, float],
    size: float,
    mass: float,
    collision: bool,
) -> RigidPrim:
    """Create and return a dynamic cube with explicit physical properties.

    Args:
        path: Cube prim path.
        position: Initial world position.
        size: Uniform cube edge length.
        mass: Rigid-body mass.
        collision: Whether to apply collision APIs.

    Returns:
        Experimental rigid-body wrapper for the cube.
    """
    Cube(path, sizes=size, positions=position)
    if collision:
        GeomPrim(path, apply_collision_apis=True)
    return RigidPrim(path, masses=[mass])


class TestTeleopUIFloatingController(MenuUITestCase):
    """Drive the floating controller from debug markers in the live Teleop window."""

    async def tearDown(self) -> None:
        """Stop the timeline and release floating controller fixtures."""
        omni.timeline.get_timeline_interface().stop()
        await super().tearDown()

    def _build_stage(self) -> None:
        """Build a minimal zero-gravity physics scene with two dynamic cube handles."""
        stage_utils.define_prim("/World", "Xform")
        PhysicsScene("/World/PhysicsScene").set_gravity((0.0, 0.0, 0.0))
        _create_dynamic_cube(
            path=_LEFT_CUBE_PATH,
            position=_LEFT_INITIAL_POSITION,
            size=_CUBE_SIZE,
            mass=_CUBE_MASS,
            collision=True,
        )
        _create_dynamic_cube(
            path=_RIGHT_CUBE_PATH,
            position=_RIGHT_INITIAL_POSITION,
            size=_CUBE_SIZE,
            mass=_CUBE_MASS,
            collision=True,
        )

    def _configure_floating_side(self, window, side: str, prim_path: str) -> None:  # noqa: ANN001
        """Configure and enable one side through the live floating panel.

        Args:
            window: Teleop window containing the floating panel.
            side: Controller side to configure.
            prim_path: Rigid-body prim path to track.
        """
        panel = window._floating_panel
        widgets = panel._widgets[side]
        widgets["path"].model.set_value(prim_path)
        widgets["kp"].model.set_value(35.0)
        widgets["kd"].model.set_value(0.0)
        widgets["rkp"].model.set_value(5.0)
        widgets["rkd"].model.set_value(0.0)

        panel._on_path_changed(side)
        self.assertTrue(panel._configured[side], f"{side} floating side should validate")
        panel._on_toggle(side)
        self.assertTrue(panel._desired_enabled[side], f"{side} floating side should be enabled")

    def _runtime_position_x(self, window, side: str) -> float:  # noqa: ANN001
        """Read the live runtime rigid-body x position for a side.

        Args:
            window: Teleop window containing the floating controller.
            side: Controller side to inspect.

        Returns:
            Runtime x position, or negative infinity if the handle is missing.
        """
        controller = window._floating_controller
        handle = controller._left_rigid_prim if side == "left" else controller._right_rigid_prim
        if handle is None:
            return float("-inf")
        positions, _orientations = handle.get_world_poses()
        pos = positions.numpy() if hasattr(positions, "numpy") else np.asarray(positions)
        return float(pos.reshape(-1, 3)[0][0])

    async def test_debug_markers_drive_left_and_right_floating_cubes(self) -> None:
        """Moving debug markers pulls both floating cube handles toward the marker poses."""
        self._build_stage()
        await self.wait_for_stage_loading()
        timeline = omni.timeline.get_timeline_interface()
        window = None

        with tempfile.TemporaryDirectory(prefix="teleop_ui_floating_") as tmp_dir:
            try:
                await self.menu_click_with_retry(MENU_PATH, window_name=WINDOW_TITLE)
                window = ui.Workspace.get_window(WINDOW_TITLE)
                self.assertIsNotNone(window, "Teleop window should exist after opening via the menu")
                window._last_profile_path = os.path.join(tmp_dir, "last_profile.yaml")
                window.visible = True
                window.focus()
                await self.wait_n_frames(5)

                session_panel = window._session_panel
                markers = window._markers_manager
                controller = window._floating_controller

                self.assertIsNotNone(session_panel._debug_tracking_cb, "Debug Mode checkbox should exist")
                session_panel._debug_tracking_cb.model.set_value(True)
                await self.wait_n_frames(5)
                self.assertTrue(window._teleop_manager.debug_tracking_enabled, "Debug mode should be active")
                self.assertIsNotNone(markers.get_marker_world_pose("left"), "Left debug marker should exist")
                self.assertIsNotNone(markers.get_marker_world_pose("right"), "Right debug marker should exist")

                self._configure_floating_side(window, "left", _LEFT_CUBE_PATH)
                self._configure_floating_side(window, "right", _RIGHT_CUBE_PATH)

                timeline.play()
                running = False
                for _ in range(60):
                    if controller.is_running("left") and controller.is_running("right"):
                        running = True
                        break
                    await self.wait_n_frames(1)
                self.assertTrue(running, "Timeline play should activate both floating sides")

                markers.update_marker_transform("left", _LEFT_TARGET_POSITION, _IDENTITY_XYZW)
                markers.update_marker_transform("right", _RIGHT_TARGET_POSITION, _IDENTITY_XYZW)
                await self.wait_n_frames(5)

                handles_ready = False
                for _ in range(60):
                    if controller._left_rigid_prim is not None and controller._right_rigid_prim is not None:
                        handles_ready = True
                        break
                    await self.wait_n_frames(1)
                self.assertTrue(handles_ready, "Floating controller should create runtime rigid handles")

                moved = False
                for _ in range(120):
                    if (
                        self._runtime_position_x(window, "left") > _LEFT_INITIAL_POSITION[0] + 0.01
                        and self._runtime_position_x(window, "right") > _RIGHT_INITIAL_POSITION[0] + 0.01
                    ):
                        moved = True
                        break
                    await self.wait_n_frames(1)
                self.assertTrue(moved, "Both floating cube handles should move toward their debug markers")
            finally:
                timeline.stop()
                await self.wait_n_frames(2)
                if window is not None:
                    window._last_profile_path = os.path.join(tmp_dir, "last_profile.yaml")
                    window.destroy()
