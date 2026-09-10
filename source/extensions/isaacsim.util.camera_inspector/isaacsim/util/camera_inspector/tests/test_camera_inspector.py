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

"""Tests for Camera Inspector live updates across stage swaps and camera deletion."""

from __future__ import annotations

import traceback

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from isaacsim.core.experimental.objects import Camera
from isaacsim.util.camera_inspector.extension import Extension

_DEFAULT_PERSP_CAMERA = "/OmniverseKit_Persp"
_INITIAL_CAMERA = "/World/Cam0"
_NEW_CAMERA = "/World/Cam1"


class TestCameraInspector(omni.kit.test.AsyncTestCase):
    """Verify the inspector survives stage swaps and in-scene camera deletion."""

    async def setUp(self) -> None:
        """Create a stage with a camera and open a Camera Inspector instance."""
        self._ext: Extension | None = None
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()
        stage_utils.define_prim("/World", type_name="Xform")
        Camera(_INITIAL_CAMERA)
        await app_utils.update_app_async()

        self._ext = Extension()
        self._ext.on_startup("isaacsim.util.camera_inspector")
        self._ext._window.visible = True
        await app_utils.update_app_async(steps=2)

    async def tearDown(self) -> None:
        """Shut down the inspector instance and close the test stage."""
        if self._ext is not None:
            try:
                if self._ext._window:
                    self._ext._window.visible = False
                self._ext.on_shutdown()
            except Exception:
                pass
            self._ext = None
        stage_utils.close_stage()
        await app_utils.update_app_async()
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

    def _camera_paths(self) -> list[str]:
        """Return prim paths of cameras currently cached by the inspector."""
        return [camera.paths[0] for camera in self._ext._all_cameras]

    def _select_same_path_camera(self) -> None:
        """Select a camera whose path also exists on a default new stage."""
        persp = next((camera for camera in self._ext._all_cameras if camera.paths[0] == _DEFAULT_PERSP_CAMERA), None)
        if persp is not None:
            self._ext._selected_camera = persp
            return
        cam0 = next((camera for camera in self._ext._all_cameras if camera.paths[0] == _INITIAL_CAMERA), None)
        self._ext._selected_camera = cam0
        self.assertIsNotNone(self._ext._selected_camera, "Inspector should have a selected camera after refresh")

    async def test_update_camera_stats_ui_does_not_raise_after_new_stage(self) -> None:
        """Creating a new stage must not raise in the per-frame camera stats callback."""
        ext = self._ext
        ext._on_refresh()
        self.assertTrue(ext._all_cameras, "Initial stage should contain at least one camera")
        self._select_same_path_camera()
        self.assertIsNotNone(ext._camera_state_subscriber)

        raised: list[BaseException] = []
        traces: list[str] = []
        original = Extension._update_camera_stats_ui

        def wrapped(e: object = None) -> None:
            try:
                original(ext, e)
            except Exception as exc:
                raised.append(exc)
                traces.append(traceback.format_exc())

        ext._update_camera_stats_ui = wrapped
        ext._camera_state_subscriber = None
        ext._on_refresh()
        self._select_same_path_camera()

        stage_utils.create_new_stage()
        await app_utils.update_app_async(steps=8)

        if raised:
            self.fail("inspector callback raised during/after New Stage:\n" + "\n".join(traces))

        self.assertIsNone(ext._selected_camera)
        self.assertEqual(ext._all_cameras, [])
        self.assertNotIn(_INITIAL_CAMERA, self._camera_paths())

        ext._on_refresh()
        current_stage = stage_utils.get_current_stage(backend="usd")
        self.assertNotIn(_INITIAL_CAMERA, self._camera_paths())
        for camera in ext._all_cameras:
            self.assertTrue(camera.valid)
            self.assertEqual(camera.prims[0].GetStage(), current_stage)

        stage_utils.define_prim("/World", type_name="Xform")
        Camera(_NEW_CAMERA)
        ext._on_refresh()
        self.assertIn(_NEW_CAMERA, self._camera_paths())
        self.assertIsNotNone(ext._selected_camera)
        self.assertTrue(ext._selected_camera.valid)
        self.assertEqual(ext._selected_camera.prims[0].GetStage(), stage_utils.get_current_stage(backend="usd"))

    async def test_deleting_selected_camera_keeps_remaining_cameras(self) -> None:
        """Deleting the selected camera must rebuild the list instead of emptying the dropdown."""
        ext = self._ext
        Camera(_NEW_CAMERA)
        ext._on_refresh()
        cam0 = next((camera for camera in ext._all_cameras if camera.paths[0] == _INITIAL_CAMERA), None)
        self.assertIsNotNone(cam0, "Initial camera should be listed after refresh")
        ext._selected_camera = cam0
        self.assertIn(_NEW_CAMERA, self._camera_paths())

        raised: list[BaseException] = []
        traces: list[str] = []
        original = Extension._update_camera_stats_ui

        def wrapped(e: object = None) -> None:
            try:
                original(ext, e)
            except Exception as exc:
                raised.append(exc)
                traces.append(traceback.format_exc())

        ext._update_camera_stats_ui = wrapped
        ext._camera_state_subscriber = None
        ext._on_refresh()
        cam0 = next((camera for camera in ext._all_cameras if camera.paths[0] == _INITIAL_CAMERA), None)
        self.assertIsNotNone(cam0)
        ext._selected_camera = cam0

        self.assertTrue(stage_utils.delete_prim(_INITIAL_CAMERA))
        ext._update_camera_stats_ui()
        await app_utils.update_app_async(steps=8)

        if raised:
            self.fail("inspector callback raised after deleting the selected camera:\n" + "\n".join(traces))

        paths = self._camera_paths()
        self.assertNotIn(_INITIAL_CAMERA, paths)
        self.assertIn(_NEW_CAMERA, paths)
        self.assertTrue(paths, "Remaining cameras should still be listed after deleting the selected camera")
        self.assertIsNotNone(ext._selected_camera)
        self.assertNotEqual(ext._selected_camera.paths[0], _INITIAL_CAMERA)
