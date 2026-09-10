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

"""Focused tests for the shared UR10 palletizing state machine."""

from dataclasses import replace

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.examples.manipulation.ur10_palletizing as palletizing
import numpy as np
import omni.kit.test
from isaacsim.robot_motion.examples.manipulation.ur10_palletizing import (
    GripperCommand,
    PalletizerController,
    PalletizerFSM,
    PalletizerObservation,
    Ur10Assets,
    build_scene,
    configure_conveyor,
)
from pxr import Gf, UsdGeom, UsdPhysics


def _observation(**overrides) -> PalletizerObservation:
    observation = PalletizerObservation(
        time=0.0,
        robot_fk_T=np.eye(4),
        robot_home_fk_T=np.eye(4),
        robot_default_config=np.zeros(6),
        gripper_closed=False,
        has_active_bin=True,
        active_bin_attached=False,
        active_bin_needs_flip=False,
        active_bin_grasp_T=np.eye(4),
        active_bin_position=np.zeros(3),
        stack_target=np.zeros(3),
        bin_under_position=None,
        active_bin_at_stack_target=False,
        placement_completes_stack=False,
    )
    return replace(observation, **overrides)


def _advance_to_pick_lift(*, needs_flip: bool) -> PalletizerController:
    controller = PalletizerController()
    common = {
        "active_bin_needs_flip": needs_flip,
        "active_bin_position": np.array([0.0, 0.47, 0.0]),
    }
    controller.step(_observation(time=0.0, **common))
    controller.step(_observation(time=1.0, **common))
    controller.step(_observation(time=1.0, **common))
    controller.step(_observation(time=2.0, **common))
    command = controller.step(_observation(time=2.0, gripper_closed=False, **common))
    if command.gripper is not GripperCommand.CLOSE:
        raise AssertionError("Pick sequence did not request gripper closure before lifting.")
    controller.step(_observation(time=2.0, gripper_closed=True, active_bin_attached=True, **common))
    return controller


class TestUr10Palletizing(omni.kit.test.AsyncTestCase):
    async def test_flip_station_retreat_moves_down_and_away(self) -> None:
        retreat = palletizing.ReleaseFlipStationBin()
        fk_T = np.eye(4)
        fk_T[:3, :3] = np.column_stack(
            (
                np.array([0.0, 0.0, 1.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([-1.0, 0.0, 0.0]),
            )
        )
        observation = _observation(robot_fk_T=fk_T)

        retreat.enter(observation)
        result = retreat.step(observation)

        self.assertIsNotNone(result.command.motion)
        self.assertAlmostEqual(result.command.motion.target_pose.p[2], -0.3)

    async def test_pick_sequence_waits_before_commanding_descent(self) -> None:
        controller = PalletizerController()

        command = controller.step(_observation(time=0.0))
        self.assertIsNone(command.motion)
        self.assertTrue(command.retain_motion)

        command = controller.step(_observation(time=0.99))
        self.assertIsNone(command.motion)
        self.assertTrue(command.retain_motion)

        command = controller.step(_observation(time=1.0))
        self.assertIsNone(command.motion)
        self.assertTrue(command.retain_motion)
        self.assertFalse(command.conveyor_running)

        command = controller.step(_observation(time=1.0))
        self.assertIsNotNone(command.motion)

    async def test_direct_place_pick_lifts_to_measured_clearance_before_traversal(self) -> None:
        controller = _advance_to_pick_lift(needs_flip=False)

        command = controller.step(_observation(time=2.0, gripper_closed=True, active_bin_attached=True))
        self.assertIsNotNone(command.motion)
        np.testing.assert_allclose(command.motion.target_pose.p, [0.0, 0.0, 0.5])

        command = controller.step(_observation(time=3.0, gripper_closed=True, active_bin_attached=True))
        self.assertIsNotNone(command.motion)
        np.testing.assert_allclose(command.motion.target_pose.p, [0.0, 0.0, 0.5])
        self.assertEqual(controller.state, "picking")

    async def test_flip_pick_retains_original_lift_height(self) -> None:
        controller = _advance_to_pick_lift(needs_flip=True)

        command = controller.step(
            _observation(
                time=2.0,
                gripper_closed=True,
                active_bin_attached=True,
                active_bin_needs_flip=True,
            )
        )
        self.assertIsNotNone(command.motion)
        np.testing.assert_allclose(command.motion.target_pose.p, [0.0, 0.0, 0.3])

    async def test_scene_uses_leaf_assets_and_preserves_sdg_paths(self) -> None:
        await stage_utils.create_new_stage_async()
        assets = Ur10Assets()
        build_scene("/World/Ur10Table", assets, use_test_gripper=True)
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()
        configure_conveyor("/World/Ur10Table")

        stage = stage_utils.get_current_stage(backend="usd")
        for prim_path in (
            "/World/Ur10Table/ur10/ee_link",
            "/World/Ur10Table/ur10/ee_link/SurfaceGripper",
            "/World/Ur10Table/pallet/Xform/Mesh_015",
            "/World/Ur10Table/pallet_holder",
            "/World/Ur10Table/conveyor",
        ):
            self.assertTrue(stage.GetPrimAtPath(prim_path).IsValid(), prim_path)

        rollers = stage.GetPrimAtPath("/World/Ur10Table/conveyor/Rollers")
        self.assertTrue(rollers.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertTrue(UsdPhysics.RigidBodyAPI(rollers).GetKinematicEnabledAttr().Get())
        conveyor_node = stage.GetPrimAtPath("/World/Ur10Table/conveyor/ConveyorBeltGraph/ConveyorNode")
        self.assertEqual(conveyor_node.GetAttribute("inputs:direction").Get(), Gf.Vec3f(1.0, 0.0, 0.0))

        context = object.__new__(palletizing.BinStackingContext)
        context.env_path = "/World/Ur10Table"
        context._conveyor_running = None
        context._set_conveyor_running(False)
        conveyor_graph = stage.GetPrimAtPath("/World/Ur10Table/conveyor/ConveyorBeltGraph")
        self.assertEqual(conveyor_graph.GetAttribute("graph:variable:Velocity").Get(), 0.0)
        context._set_conveyor_running(True)
        self.assertAlmostEqual(
            conveyor_graph.GetAttribute("graph:variable:Velocity").Get(),
            palletizing._CONVEYOR_SPEED,
        )

        self.assertIn("/Isaac/Robots_Multiphysics/", assets.ur10_usd)

    async def test_fsm_routes_flipped_bin_back_to_pick(self) -> None:
        fsm = PalletizerFSM()

        self.assertEqual(fsm.step(_observation()).state, "picking")
        self.assertEqual(
            fsm.step(_observation(active_bin_needs_flip=True), sequence_done=True).state,
            "flipping",
        )
        self.assertEqual(fsm.step(_observation(), sequence_done=True).state, "picking")

    async def test_fsm_completes_final_placement(self) -> None:
        fsm = PalletizerFSM()

        fsm.step(_observation())
        self.assertEqual(fsm.step(_observation(), sequence_done=True).state, "placing")
        decision = fsm.step(
            _observation(active_bin_at_stack_target=True, placement_completes_stack=True),
            sequence_done=True,
        )

        self.assertEqual(decision.state, "done")
        self.assertTrue(decision.complete_bin)

    async def test_fsm_recovers_when_active_bin_is_lost(self) -> None:
        fsm = PalletizerFSM()
        fsm.step(_observation())

        decision = fsm.step(_observation(has_active_bin=False))

        self.assertEqual(decision.state, "idle")
        self.assertTrue(decision.state_changed)
