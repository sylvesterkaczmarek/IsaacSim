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

"""Tests for teleop marker lifecycle behavior."""

from unittest.mock import patch

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from isaacsim.replicator.teleop import MarkersManager


class TestMarkersManager(omni.kit.test.AsyncTestCase):
    """Verify marker cleanup with and without an anonymous marker layer."""

    async def setUp(self) -> None:
        """Create a fresh stage and marker manager."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()
        self._manager = MarkersManager()

    async def tearDown(self) -> None:
        """Clear marker state and close the test stage."""
        self._manager.remove_all_markers()
        stage_utils.close_stage()
        await app_utils.update_app_async()
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

    def test_remove_marker_without_anonymous_layer(self) -> None:
        """Fallback cleanup deletes one child or the complete origin hierarchy."""
        stage = stage_utils.get_current_stage()
        origin_path = MarkersManager.MARKER_PATHS["origin"]
        left_path = MarkersManager.MARKER_PATHS["left"]
        right_path = MarkersManager.MARKER_PATHS["right"]
        head_path = MarkersManager.MARKER_PATHS["head"]
        for path in (origin_path, left_path, right_path, head_path):
            stage_utils.define_prim(path, "Xform")

        self.assertIsNone(self._manager.layer)
        self.assertTrue(self._manager.remove_marker("left"))
        self.assertFalse(stage.GetPrimAtPath(left_path).IsValid())
        self.assertTrue(stage.GetPrimAtPath(origin_path).IsValid())
        self.assertTrue(stage.GetPrimAtPath(right_path).IsValid())
        self.assertTrue(self._manager.remove_marker("left"))

        self.assertTrue(self._manager.remove_marker("origin"))
        self.assertFalse(stage.GetPrimAtPath(origin_path).IsValid())
        self.assertFalse(stage.GetPrimAtPath(right_path).IsValid())
        self.assertFalse(stage.GetPrimAtPath(head_path).IsValid())

    def test_ensure_marker_authors_only_to_anonymous_layer(self) -> None:
        """Marker and frame Xforms are isolated from the root layer."""
        stage = stage_utils.get_current_stage()
        origin_path = MarkersManager.MARKER_PATHS["origin"]
        frame_path = f"{origin_path}/{MarkersManager.FRAME_CHILD_NAME}"

        def define_frame(path: str) -> bool:
            return stage_utils.define_prim(path, "Xform").IsValid()

        with patch.object(self._manager, "_add_frame_reference", side_effect=define_frame):
            ok, _message = self._manager.ensure_marker("origin")

        self.assertTrue(ok)
        self.assertTrue(stage.GetPrimAtPath(origin_path).IsValid())
        self.assertTrue(stage.GetPrimAtPath(frame_path).IsValid())
        self.assertIsNotNone(self._manager.layer)
        self.assertIsNone(stage.GetRootLayer().GetPrimAtPath(MarkersManager.MARKERS_SCOPE))

    def test_live_marker_world_pose_is_independent_of_origin_proxy_pose(self) -> None:
        """Finalized controller markers stay registered with the canonical XR transform."""
        stage = stage_utils.get_current_stage()

        def define_frame(path: str) -> bool:
            return stage_utils.define_prim(path, "Xform").IsValid()

        with patch.object(self._manager, "_add_frame_reference", side_effect=define_frame):
            self.assertTrue(self._manager.ensure_marker("left")[0])

        self.assertTrue(
            self._manager.set_origin_world_pose(
                (5.0, 6.0, 0.0),
                (0.0, 0.0, 0.70710678, 0.70710678),
            )
        )
        self._manager.update_marker_world_transforms(
            left_position=(10.0, 20.0, 1.0),
            left_orientation=(0.0, 0.0, 0.0, 1.0),
        )

        position, orientation = self._manager.get_marker_world_pose("left")
        self.assertAlmostEqual(position[0], 10.0, places=5)
        self.assertAlmostEqual(position[1], 20.0, places=5)
        self.assertAlmostEqual(position[2], 1.0, places=5)
        self.assertAlmostEqual(orientation[3], 1.0, places=5)
