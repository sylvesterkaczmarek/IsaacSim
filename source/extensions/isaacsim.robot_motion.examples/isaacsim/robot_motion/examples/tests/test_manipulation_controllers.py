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

"""Focused tests for the robot-independent manipulation controller boundary."""

from enum import Enum

import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np
import omni.kit.test
import warp as wp
from isaacsim.robot_motion.examples.manipulation import (
    GripperCommand,
    JointGripperController,
    PickPlaceController,
    PickPlacePhase,
    SurfaceGripperController,
    get_robot_config,
)


class _ArmController(mg.BaseController):
    def __init__(self, reset_result: bool = True, return_none: bool = False) -> None:
        self.reset_result = reset_result
        self.return_none = return_none
        self.last_setpoint = None

    def reset(self, estimated_state, setpoint_state, t, **kwargs) -> bool:
        self.last_setpoint = setpoint_state
        return self.reset_result

    def forward(self, estimated_state, setpoint_state, t, **kwargs):
        self.last_setpoint = setpoint_state
        return None if self.return_none else mg.RobotState()


class _GripperController(mg.BaseController):
    def __init__(self, reset_result: bool = True, complete: bool = True) -> None:
        self.reset_result = reset_result
        self.complete = complete

    def reset(self, estimated_state, setpoint_state, t, **kwargs) -> bool:
        return self.reset_result

    def forward(self, estimated_state, setpoint_state, t, **kwargs):
        return mg.RobotState()

    def is_complete(self, estimated_state) -> bool:
        return self.complete


class _GripperStatus(Enum):
    Open = "open"
    Closing = "closing"
    Closed = "closed"


class _SurfaceGripperInterface:
    def __init__(self) -> None:
        self.status = _GripperStatus.Open
        self.open_calls = 0
        self.close_calls = 0

    def get_gripper_status(self, path: str) -> _GripperStatus:
        return self.status

    def open_gripper(self, path: str) -> None:
        self.open_calls += 1

    def close_gripper(self, path: str) -> None:
        self.close_calls += 1


def _sites(entries: dict[str, tuple[float, float, float]]) -> mg.RobotState:
    names = list(entries)
    return mg.RobotState(
        sites=mg.SpatialState.from_name(
            spatial_space=names,
            positions=(names, wp.array(list(entries.values()), dtype=wp.float32, device="cpu")),
            orientations=(
                names,
                wp.array([[1.0, 0.0, 0.0, 0.0]] * len(names), dtype=wp.float32, device="cpu"),
            ),
        )
    )


def _goal(pick=(0.4, 0.0, 0.05), place=(0.0, 0.4, 0.05)) -> mg.RobotState:
    return _sites({"pick": pick, "place": place})


def _controller(
    arm: _ArmController | None = None,
    gripper: _GripperController | None = None,
    *,
    stability_duration: float = 0.1,
    timeout: float = 1.0,
) -> PickPlaceController:
    arm = _ArmController() if arm is None else arm
    gripper = _GripperController() if gripper is None else gripper
    return PickPlaceController(
        arm_controller=arm,
        gripper_open_controller=gripper,
        gripper_close_controller=gripper,
        robot_site_space=["tool"],
        tool_frame="tool",
        grasp_orientation=(1.0, 0.0, 0.0, 0.0),
        phase_timeouts={
            phase: timeout for phase in PickPlacePhase if phase not in {PickPlacePhase.DONE, PickPlacePhase.FAILED}
        },
        stability_duration=stability_duration,
        gripper_dwell=0.0,
    )


def _joint_state(positions: tuple[float, float], velocities: tuple[float, float]) -> mg.RobotState:
    names = ["left", "right"]
    return mg.RobotState(
        joints=mg.JointState.from_name(
            robot_joint_space=names,
            positions=(names, wp.array(positions, dtype=wp.float32, device="cpu")),
            velocities=(names, wp.array(velocities, dtype=wp.float32, device="cpu")),
        )
    )


class TestManipulationControllers(omni.kit.test.AsyncTestCase):
    async def test_robot_configs_use_named_joints_and_explicit_tool_frames(self) -> None:
        franka = get_robot_config("FrAnKa")
        ur10 = get_robot_config("ur10")
        self.assertEqual(dict(franka.default_joint_positions)["panda_finger_joint1"], 0.04)
        self.assertEqual(franka.tool.controller_frame, "panda_hand")
        self.assertEqual(franka.tool.measurement_prim_path, "panda_hand")
        np.testing.assert_allclose(franka.grasp_orientation, (0.0, 1.0, 0.0, 0.0))
        self.assertEqual(franka.gripper.velocity_tolerance, 0.1)
        self.assertAlmostEqual(dict(ur10.default_joint_positions)["shoulder_pan_joint"], -np.pi / 2)
        self.assertEqual(ur10.tool.controller_frame, "tool0")
        self.assertEqual(ur10.tool.measurement_prim_path, "ee_link")
        np.testing.assert_allclose(
            ur10.tool.measurement_to_controller_orientation,
            (0.0, np.sqrt(0.5), 0.0, np.sqrt(0.5)),
            atol=1.0e-7,
        )
        np.testing.assert_allclose(
            ur10.grasp_orientation,
            (0.0, np.sqrt(0.5), 0.0, -np.sqrt(0.5)),
            atol=1.0e-7,
        )
        with self.assertRaises(ValueError):
            get_robot_config("unknown")

    async def test_pick_place_captures_immutable_goals(self) -> None:
        arm = _ArmController()
        controller = _controller(arm, stability_duration=1.0)
        estimated = _sites({"tool": (0.0, 0.0, 0.0)})
        self.assertTrue(controller.reset(estimated, _goal(), 0.0))
        controller.forward(estimated, _goal(pick=(9.0, 9.0, 9.0)), 0.1)
        target = arm.last_setpoint.sites.positions.numpy()[0]
        np.testing.assert_allclose(target, (0.4, 0.0, 0.25), atol=1.0e-6)

    async def test_pick_place_requires_measured_convergence_and_stability(self) -> None:
        controller = _controller()
        far = _sites({"tool": (0.0, 0.0, 0.0)})
        approach = _sites({"tool": (0.4, 0.0, 0.25)})
        self.assertTrue(controller.reset(far, _goal(), 0.0))
        controller.forward(far, _goal(), 0.5)
        self.assertEqual(controller.phase, PickPlacePhase.APPROACH_PICK)
        controller.forward(approach, _goal(), 0.5)
        controller.forward(approach, _goal(), 0.59)
        self.assertEqual(controller.phase, PickPlacePhase.APPROACH_PICK)
        controller.forward(approach, _goal(), 0.61)
        self.assertEqual(controller.phase, PickPlacePhase.DESCEND_PICK)

    async def test_pick_place_timeout_is_failure(self) -> None:
        controller = _controller(timeout=0.2)
        estimated = _sites({"tool": (0.0, 0.0, 0.0)})
        self.assertTrue(controller.reset(estimated, _goal(), 0.0))
        self.assertIsNone(controller.forward(estimated, _goal(), 0.2))
        self.assertTrue(controller.failed)
        self.assertIn("timed out", controller.failure_reason)

    async def test_pick_place_rejects_backward_and_nonfinite_time(self) -> None:
        estimated = _sites({"tool": (0.0, 0.0, 0.0)})
        for invalid_time, message in ((0.9, "backwards"), (np.nan, "finite")):
            with self.subTest(invalid_time=invalid_time):
                controller = _controller()
                self.assertTrue(controller.reset(estimated, _goal(), 1.0))
                self.assertIsNone(controller.forward(estimated, _goal(), invalid_time))
                self.assertTrue(controller.failed)
                self.assertIn(message, controller.failure_reason)

    async def test_pick_place_rejects_invalid_configuration(self) -> None:
        required = {
            "arm_controller": _ArmController(),
            "gripper_open_controller": _GripperController(),
            "gripper_close_controller": _GripperController(),
            "robot_site_space": ["tool"],
            "tool_frame": "tool",
        }
        invalid_options = (
            {"approach_height": -0.1},
            {"lift_height": np.inf},
            {"position_tolerance": np.nan},
            {"position_tolerance": 0.0},
            {"orientation_tolerance": 0.0},
            {"grasp_position_tolerance": 0.0},
            {"gripper_dwell": np.inf},
            {"phase_timeouts": {PickPlacePhase.DONE: 3.0}},
        )
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                PickPlaceController(**required, **options)

    async def test_pick_place_propagates_child_failure(self) -> None:
        estimated = _sites({"tool": (0.0, 0.0, 0.0)})
        reset_failure = _controller(arm=_ArmController(reset_result=False))
        self.assertFalse(reset_failure.reset(estimated, _goal(), 0.0))
        self.assertIn("reset failed", reset_failure.failure_reason)

        forward_failure = _controller(arm=_ArmController(return_none=True))
        self.assertTrue(forward_failure.reset(estimated, _goal(), 0.0))
        self.assertIsNone(forward_failure.forward(estimated, _goal(), 0.1))
        self.assertIn("composition failed", forward_failure.failure_reason)

    async def test_joint_gripper_exact_and_stopped_short_completion(self) -> None:
        controller = JointGripperController(
            robot_joint_space=["left", "right"],
            joint_names=["left", "right"],
            open_positions=[0.04, 0.04],
            closed_positions=[0.0, 0.0],
            command=GripperCommand.CLOSE,
            velocity_tolerance=0.1,
        )
        self.assertTrue(controller.reset(_joint_state((0.04, 0.04), (0.0, 0.0)), None, 0.0))
        self.assertTrue(controller.is_complete(_joint_state((0.0, 0.0), (0.2, 0.2))))
        self.assertFalse(controller.is_complete(_joint_state((0.02, 0.02), (0.2, 0.2))))
        self.assertTrue(controller.is_complete(_joint_state((0.02, 0.02), (0.079, 0.079))))
        self.assertTrue(controller.is_complete(_joint_state((0.02, 0.02), (0.0, 0.0))))
        self.assertFalse(controller.is_complete(_joint_state((0.05, 0.02), (0.0, 0.0))))
        self.assertFalse(controller.is_complete(_joint_state((0.04, 0.04), (0.0, 0.0))))

        names = ["left", "right"]
        position_only = mg.RobotState(
            joints=mg.JointState.from_name(
                robot_joint_space=names,
                positions=(names, wp.array([0.0, 0.0], dtype=wp.float32, device="cpu")),
            )
        )
        self.assertTrue(controller.reset(position_only, None, 0.0))
        self.assertTrue(controller.is_complete(position_only))

    async def test_surface_gripper_uses_exact_status(self) -> None:
        interface = _SurfaceGripperInterface()
        controller = SurfaceGripperController(
            gripper_path="/World/robot/ee_link/SurfaceGripper",
            command=GripperCommand.CLOSE,
            interface=interface,
            status_type=_GripperStatus,
        )
        controller.forward(mg.RobotState(), None, 0.0)
        controller.forward(mg.RobotState(), None, 0.1)
        self.assertEqual(interface.close_calls, 2)
        self.assertFalse(controller.is_complete(mg.RobotState()))
        interface.status = _GripperStatus.Closing
        controller.forward(mg.RobotState(), None, 0.2)
        self.assertEqual(interface.close_calls, 2)
        self.assertFalse(controller.is_complete(mg.RobotState()))
        interface.status = _GripperStatus.Closed
        self.assertTrue(controller.is_complete(mg.RobotState()))
