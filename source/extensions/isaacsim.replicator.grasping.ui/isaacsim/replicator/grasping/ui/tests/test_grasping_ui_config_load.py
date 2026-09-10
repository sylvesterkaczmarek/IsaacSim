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

"""Tests that Grasping UI load preserves per-phase joint drive targets."""

from __future__ import annotations

import tempfile
from pathlib import Path

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
import omni.usd
from isaacsim.replicator.grasping.grasping_manager import GraspingManager
from isaacsim.replicator.grasping.ui.grasping_window import GraspingWindow
from pxr import UsdPhysics

GRIPPER_PATH = "/World/Gripper"
JOINT_OPEN = f"{GRIPPER_PATH}/joints/open_joint"
JOINT_CLOSE = f"{GRIPPER_PATH}/joints/close_joint"


def _define_drive_joint(joint_path: str) -> None:
    """Create a revolute drive joint at the given path.

    Args:
        joint_path: Absolute prim path for the joint.
    """
    joint = UsdPhysics.RevoluteJoint.Define(stage_utils.get_current_stage(), joint_path)
    UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")


class TestGraspingUIConfigLoad(omni.kit.test.AsyncTestCase):
    """Verify configuration load keeps distinct per-phase joint maps."""

    async def setUp(self) -> None:
        """Create a clean stage with a gripper and two drive joints."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()
        stage_utils.define_prim("/World", type_name="Xform")
        stage_utils.define_prim(GRIPPER_PATH, type_name="Xform")
        stage_utils.define_prim(f"{GRIPPER_PATH}/joints", type_name="Xform")
        _define_drive_joint(JOINT_OPEN)
        _define_drive_joint(JOINT_CLOSE)
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        """Close the stage and wait for pending asset loads."""
        stage_utils.close_stage()
        await app_utils.update_app_async()
        # In some cases the test will end before the asset is loaded, in this case wait for assets to load
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await app_utils.update_app_async()

    async def test_load_config_preserves_per_phase_joint_sets(self) -> None:
        """Keep distinct Open/Close joint maps after loading through the Grasping UI."""
        core_mgr = GraspingManager()
        self.assertTrue(core_mgr.set_gripper(GRIPPER_PATH))
        core_mgr.grasp_phases = []
        core_mgr.create_and_add_grasp_phase(name="Open", joint_drive_targets={JOINT_OPEN: 0.0})
        core_mgr.create_and_add_grasp_phase(name="Close", joint_drive_targets={JOINT_CLOSE: 48.0})

        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = str(Path(tmp_dir) / "saved_config.yaml")
            core_mgr.save_config(config_path, overwrite=True)
            self.assertTrue(Path(config_path).is_file())

            window = GraspingWindow(title="Grasping")
            try:
                await app_utils.update_app_async(steps=10)
                window._config_path = config_path
                window._on_load_config()
                await app_utils.update_app_async(steps=40)

                gm = window._grasping_manager
                open_phase = gm.get_grasp_phase_by_name("Open")
                close_phase = gm.get_grasp_phase_by_name("Close")
                self.assertIsNotNone(open_phase)
                self.assertIsNotNone(close_phase)
                self.assertEqual(set(open_phase.joint_drive_targets.keys()), {JOINT_OPEN})
                self.assertEqual(set(close_phase.joint_drive_targets.keys()), {JOINT_CLOSE})
                self.assertEqual(open_phase.get_joint_target(JOINT_OPEN), 0.0)
                self.assertAlmostEqual(close_phase.get_joint_target(JOINT_CLOSE), 48.0)

                included_paths = {joint["path"] for joint in window._joint_ui_data if joint["include"]}
                self.assertIn(JOINT_OPEN, included_paths)
                self.assertIn(JOINT_CLOSE, included_paths)
            finally:
                window.destroy()
                await app_utils.update_app_async(steps=5)
