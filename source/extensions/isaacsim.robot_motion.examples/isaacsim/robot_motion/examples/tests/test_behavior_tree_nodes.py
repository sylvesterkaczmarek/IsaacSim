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

"""Tests for Franka behavior-tree nodes."""

from __future__ import annotations

from unittest import mock

import numpy as np
import omni.kit.test
from isaacsim.robot_motion.examples.extension import (
    BEHAVIOR_TREE_NODE_LIBRARY_NAME,
    LEGACY_BEHAVIOR_TREE_NODE_LIBRARY_NAME,
)
from isaacsim.robot_motion.examples.manipulation import behavior_tree_nodes
from omni.behavior.tree.core import NodeResetReason, NodeStatus, get_factory

_NODE_TYPES = (
    "FrankaReset",
    "FrankaAcquireCube",
    "FrankaRecoverCube",
    "FrankaCalibratePlaceTargets",
    "FrankaMoveToTarget",
    "FrankaSetGripper",
    "FrankaCheckObjectLifted",
    "FrankaCheckObjectAtTarget",
    "FrankaReleaseCenter",
    "FrankaMarkCubePlaced",
    "FrankaCheckStackComplete",
)


def _pose_array(position: list[float]) -> mock.Mock:
    """Create a mocked pose array returned by an experimental prim wrapper."""
    pose = mock.Mock()
    pose.numpy.return_value = np.asarray([position], dtype=np.float32)
    return pose


class TestBehaviorTreeNodeRegistration(omni.kit.test.AsyncTestCase):
    """Verify that Franka behavior-tree nodes register at extension startup."""

    async def test_action_nodes_registered(self) -> None:
        """Check all Franka action node types are available."""
        for library_name in (BEHAVIOR_TREE_NODE_LIBRARY_NAME, LEGACY_BEHAVIOR_TREE_NODE_LIBRARY_NAME):
            weak_library = get_factory().get_node_library(library_name)
            library = weak_library.get() if weak_library is not None else None
            self.assertIsNotNone(library)
            for type_name in _NODE_TYPES:
                with self.subTest(library_name=library_name, type_name=type_name):
                    self.assertIsNotNone(library.get_node_type_handle(type_name))


class TestBehaviorTreeNodes(omni.kit.test.AsyncTestCase):
    """Verify state cleanup and failure handling in Franka behavior-tree nodes."""

    def test_recover_cube_reset_clears_transient_state(self) -> None:
        """Ensure a reset cannot reuse stability data from an earlier recovery."""
        node = behavior_tree_nodes.FrankaRecoverCube()
        node._last_cube_position = np.ones(3)
        node._stable_elapsed = 1.0
        node._warning_elapsed = 2.0
        node._motion_started = True

        node.on_reset(NodeResetReason.EXPLICIT)

        self.assertIsNone(node._last_cube_position)
        self.assertEqual(node._stable_elapsed, 0.0)
        self.assertEqual(node._warning_elapsed, 0.0)
        self.assertFalse(node._motion_started)

    def test_recover_cube_handles_transient_motion_failure(self) -> None:
        """Return failure when recovery target updates encounter transient scene state."""
        node = behavior_tree_nodes.FrankaRecoverCube()
        local = {
            "approach_target": "/World/Targets/left/approach_target",
            "current_cube": "/World/Cube",
            "recovery_key": "recovery_requested:left",
        }
        shared = {"recovery_requested:left": True}
        context = mock.Mock()
        context.get_blackboard.return_value = shared
        context.get_delta_time.return_value = 1.0 / 60.0
        context.get_input.side_effect = {
            "warning_interval": 5.0,
            "movement_tolerance": 0.008,
        }.get
        cube = mock.Mock()
        cube.get_world_poses.return_value = (_pose_array([0.0, 0.0, 0.025]), None)

        with (
            mock.patch.object(node, "_resolve_robot", return_value=mock.Mock()),
            mock.patch.object(node, "get_prim_path", return_value="/World/franka_left"),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
            mock.patch.object(behavior_tree_nodes, "RigidPrim", return_value=cube),
            mock.patch.object(behavior_tree_nodes, "XformPrim", side_effect=RuntimeError("transient prim")),
            mock.patch.object(behavior_tree_nodes.carb, "log_error") as log_error,
        ):
            status = node.on_tick(context)

        self.assertEqual(status, NodeStatus.FAILURE)
        log_error.assert_called_once()

    def test_acquire_cube_prim_failure_does_not_claim_cube(self) -> None:
        """Keep shared reservations unchanged when target preparation fails."""
        node = behavior_tree_nodes.FrankaAcquireCube()
        local = {}
        shared = {}
        context = mock.Mock()
        context.get_blackboard.return_value = shared

        with (
            mock.patch.object(node, "_resolve_robot", return_value=object()),
            mock.patch.object(node, "get_prim_path", return_value="/World/franka_left"),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
            mock.patch.object(behavior_tree_nodes, "RigidPrim", side_effect=RuntimeError("transient prim")),
            mock.patch.object(behavior_tree_nodes.carb, "log_error") as log_error,
        ):
            status = node.on_tick(context)

        self.assertEqual(status, NodeStatus.FAILURE)
        log_error.assert_called_once()
        self.assertNotIn("center_owner", shared)
        self.assertFalse(any(key.startswith("cube_state:") for key in shared))
        self.assertNotIn("current_cube", local)

    def test_acquire_cube_publishes_reservation_and_targets(self) -> None:
        """Publish an atomic reservation and all motion targets after acquisition."""
        node = behavior_tree_nodes.FrankaAcquireCube()
        local = {}
        shared = {}
        context = mock.Mock()
        context.get_blackboard.return_value = shared
        cube_position = [0.1, 0.2, 0.025]
        cube = mock.Mock()
        cube.get_world_poses.return_value = (_pose_array(cube_position), None)
        targets = {}

        def make_target(path: str) -> mock.Mock:
            targets[path] = mock.Mock()
            return targets[path]

        with (
            mock.patch.object(node, "_resolve_robot", return_value=object()),
            mock.patch.object(node, "get_prim_path", return_value="/World/franka_left"),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
            mock.patch.object(behavior_tree_nodes, "RigidPrim", return_value=cube),
            mock.patch.object(behavior_tree_nodes, "XformPrim", side_effect=make_target),
        ):
            status = node.on_tick(context)

        cube_path = "/World/Pyramid/CubeTop"
        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(local["current_cube"], cube_path)
        self.assertEqual(shared[f"cube_state:{cube_path}"], "reserved:left")
        self.assertEqual(shared["center_owner"], "left")
        self.assertFalse(shared["recovery_requested:left"])
        self.assertFalse(shared["stack_complete:left"])
        self.assertEqual(len(targets), 7)
        np.testing.assert_allclose(
            targets["/World/Targets/left/grasp_target"].set_world_poses.call_args.kwargs["positions"],
            [0.1, 0.2, 0.115],
        )
        np.testing.assert_allclose(
            targets["/World/Targets/left/object_goal"].set_world_poses.call_args.kwargs["positions"],
            [0.46, -0.28, 0.025],
        )

    def test_acquire_cube_waits_for_center_lock(self) -> None:
        """Do not select a cube while the other robot owns the shared center."""
        node = behavior_tree_nodes.FrankaAcquireCube()
        local = {}
        shared = {"center_owner": "right"}
        context = mock.Mock()
        context.get_blackboard.return_value = shared

        with (
            mock.patch.object(node, "_resolve_robot", return_value=object()),
            mock.patch.object(node, "get_prim_path", return_value="/World/franka_left"),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
            mock.patch.object(behavior_tree_nodes, "RigidPrim") as rigid_prim,
        ):
            status = node.on_tick(context)

        self.assertEqual(status, NodeStatus.RUNNING)
        rigid_prim.assert_not_called()
        self.assertEqual(shared, {"center_owner": "right"})
        self.assertEqual(local, {})

    def test_move_to_target_checks_default_orientation_convergence(self) -> None:
        """Check convergence when the node supplies the default downward orientation."""
        node = behavior_tree_nodes.FrankaMoveToTarget()
        context = mock.Mock()
        context.get_delta_time.return_value = 1.0 / 60.0
        context.get_input.side_effect = {
            "target_prim": "/World/Target",
            "position_tolerance": 0.01,
            "orientation_tolerance": 0.05,
            "use_target_orientation": False,
            "end_effector_yaw_degrees": 0.0,
        }.get
        target = mock.Mock()
        target.get_world_poses.return_value = (
            _pose_array([0.5, 0.0, 0.3]),
            _pose_array([1.0, 0.0, 0.0, 0.0]),
        )
        robot = mock.Mock()
        robot.get_downward_orientation.return_value = [0.0, 1.0, 0.0, 0.0]
        robot.get_tool_pose.return_value = (
            np.asarray([[0.5, 0.0, 0.3]], dtype=np.float32),
            np.asarray([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
        )

        with (
            mock.patch.object(node, "_resolve_robot", return_value=robot),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value={}),
            mock.patch.object(behavior_tree_nodes, "XformPrim", return_value=target),
        ):
            status = node.on_tick(context)

        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertEqual(robot.move_to_pose.call_args.kwargs["orientation"].shape, (1, 4))
        self.assertTrue(robot.move_to_pose.call_args.kwargs["reset"])

    def test_move_requests_recovery_when_pickup_moves(self) -> None:
        """Preempt pickup motion when the reserved cube leaves its reference pose."""
        node = behavior_tree_nodes.FrankaMoveToTarget()
        local = {
            "current_cube": "/World/Cube",
            "pickup_reference": [0.0, 0.0, 0.0],
            "recovery_key": "recovery_requested:left",
        }
        shared = {"recovery_requested:left": False}
        context = mock.Mock()
        context.get_blackboard.return_value = shared
        context.get_input.side_effect = {
            "target_prim": "/World/Target",
            "position_tolerance": 0.01,
            "orientation_tolerance": 0.05,
            "monitor_pickup_motion": True,
            "monitor_grasp": False,
            "movement_tolerance": 0.008,
        }.get
        cube = mock.Mock()
        cube.get_world_poses.return_value = (_pose_array([0.02, 0.0, 0.0]), None)
        robot = mock.Mock()
        robot.get_tool_pose.return_value = (np.zeros((1, 3)), None)

        with (
            mock.patch.object(node, "_resolve_robot", return_value=robot),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
            mock.patch.object(behavior_tree_nodes, "RigidPrim", return_value=cube),
            mock.patch.object(behavior_tree_nodes, "XformPrim", return_value=mock.Mock()),
        ):
            status = node.on_tick(context)

        self.assertEqual(status, NodeStatus.RUNNING)
        self.assertTrue(shared["recovery_requested:left"])
        robot.move_to_pose.assert_not_called()

    def test_cube_returns_after_nonconverging_motion(self) -> None:
        """Keep retrying a moved target and resume after it returns to a reachable pose."""
        move = behavior_tree_nodes.FrankaMoveToTarget()
        recover = behavior_tree_nodes.FrankaRecoverCube()
        cube_path = "/World/Pyramid/CubeTop"
        local = {
            "approach_target": "/World/Targets/left/approach_target",
            "current_cube": cube_path,
            "grasp_target": "/World/Targets/left/grasp_target",
            "lift_target": "/World/Targets/left/lift_target",
            "pickup_reference": [3.0, 0.0, 0.025],
            "recovery_key": "recovery_requested:left",
        }
        shared = {
            "center_owner": "left",
            f"cube_state:{cube_path}": "reserved:left",
            "recovery_requested:left": False,
        }
        context = mock.Mock()
        context.get_blackboard.return_value = shared
        context.get_delta_time.return_value = 1.0
        context.get_input.side_effect = {
            "target_prim": local["approach_target"],
            "movement_tolerance": 0.008,
            "staging_height": 0.20,
            "position_tolerance": 0.025,
            "orientation_tolerance": 0.05,
            "use_target_orientation": False,
            "monitor_pickup_motion": True,
            "monitor_grasp": False,
            "stable_duration": 0.0,
            "warning_interval": 5.0,
            "end_effector_yaw_degrees": 90.0,
        }.get
        cube = mock.Mock()
        cube.get_world_poses.return_value = (_pose_array([3.0, 0.0, 0.025]), None)
        target = mock.Mock()
        target.get_world_poses.return_value = (
            _pose_array([3.0, 0.0, 0.225]),
            _pose_array([1.0, 0.0, 0.0, 0.0]),
        )
        robot = mock.Mock()
        robot.get_downward_orientation.return_value = [0.0, 1.0, 0.0, 0.0]
        robot.get_tool_pose.return_value = (
            _pose_array([0.5, 0.0, 0.3]).numpy(),
            _pose_array([0.0, 1.0, 0.0, 0.0]).numpy(),
        )

        with (
            mock.patch.object(move, "_resolve_robot", return_value=robot),
            mock.patch.object(recover, "_resolve_robot", return_value=robot),
            mock.patch.object(recover, "get_prim_path", return_value="/World/franka_left"),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
            mock.patch.object(behavior_tree_nodes, "RigidPrim", return_value=cube),
            mock.patch.object(behavior_tree_nodes, "XformPrim", return_value=target),
            mock.patch.object(behavior_tree_nodes.carb, "log_warn") as log_warn,
        ):
            initial_statuses = [move.on_tick(context) for _ in range(5)]
            move.on_reset(NodeResetReason.EXPLICIT)
            recovery_statuses = [recover.on_tick(context) for _ in range(5)]

            cube.get_world_poses.return_value = (_pose_array([0.5, 0.0, 0.025]), None)
            robot.get_tool_pose.return_value = (_pose_array([0.5, 0.0, 0.225]).numpy(), None)
            recovery_complete = recover.on_tick(context)

            target.get_world_poses.return_value = (
                _pose_array([0.5, 0.0, 0.225]),
                _pose_array([1.0, 0.0, 0.0, 0.0]),
            )
            resumed_orientation = np.asarray(robot.move_to_pose.call_args.kwargs["orientation"]).reshape(1, 4)
            robot.get_tool_pose.return_value = (
                _pose_array([0.5, 0.0, 0.225]).numpy(),
                resumed_orientation,
            )
            resumed_status = move.on_tick(context)

        self.assertEqual(initial_statuses, [NodeStatus.RUNNING] * 5)
        self.assertEqual(recovery_statuses, [NodeStatus.RUNNING] * 5)
        self.assertEqual(recovery_complete, NodeStatus.FAILURE)
        self.assertEqual(resumed_status, NodeStatus.SUCCESS)
        self.assertEqual(log_warn.call_count, 2)
        self.assertTrue(all("has not converged" in call.args[0] for call in log_warn.call_args_list))
        self.assertEqual(shared["center_owner"], "left")
        self.assertEqual(shared[f"cube_state:{cube_path}"], "reserved:left")
        self.assertEqual(local["current_cube"], cube_path)
        self.assertFalse(shared["recovery_requested:left"])

    def test_close_gripper_requests_recovery_when_cube_moves(self) -> None:
        """Keep the gripper open and request restaging if the cube moves before closing."""
        node = behavior_tree_nodes.FrankaSetGripper()
        local = {
            "current_cube": "/World/Cube",
            "pickup_reference": [0.0, 0.0, 0.0],
            "recovery_key": "recovery_requested:left",
        }
        shared = {"recovery_requested:left": False}
        context = mock.Mock()
        context.get_blackboard.return_value = shared
        context.get_delta_time.return_value = 1.0 / 60.0
        context.get_input.side_effect = {
            "command": "close",
            "pre_close_settle_duration": 0.25,
            "settle_duration": 0.25,
            "movement_tolerance": 0.008,
        }.get
        cube = mock.Mock()
        cube.get_world_poses.return_value = (_pose_array([0.02, 0.0, 0.0]), None)
        robot = mock.Mock()

        with (
            mock.patch.object(node, "_resolve_robot", return_value=robot),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
            mock.patch.object(behavior_tree_nodes, "RigidPrim", return_value=cube),
        ):
            status = node.on_tick(context)

        self.assertEqual(status, NodeStatus.RUNNING)
        self.assertTrue(shared["recovery_requested:left"])
        robot.close_gripper.assert_not_called()

    def test_close_gripper_waits_at_grasp_pose_before_closing(self) -> None:
        """Hold the grasp pose before closing, then apply the normal post-close settling time."""
        node = behavior_tree_nodes.FrankaSetGripper()
        local = {
            "current_cube": "/World/Cube",
            "pickup_reference": [0.0, 0.0, 0.0],
            "recovery_key": "recovery_requested:left",
        }
        shared = {"recovery_requested:left": False}
        context = mock.Mock()
        context.get_blackboard.return_value = shared
        context.get_delta_time.return_value = 0.125
        context.get_input.side_effect = {
            "command": "close",
            "pre_close_settle_duration": 0.25,
            "settle_duration": 0.25,
            "movement_tolerance": 0.008,
        }.get
        cube = mock.Mock()
        cube.get_world_poses.return_value = (_pose_array([0.0, 0.0, 0.0]), None)
        robot = mock.Mock()

        with (
            mock.patch.object(node, "_resolve_robot", return_value=robot),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
            mock.patch.object(behavior_tree_nodes, "RigidPrim", return_value=cube),
        ):
            first_status = node.on_tick(context)
            second_status = node.on_tick(context)
            close_calls_after_pre_close_settle = robot.close_gripper.call_count
            third_status = node.on_tick(context)

        self.assertEqual(first_status, NodeStatus.RUNNING)
        self.assertEqual(second_status, NodeStatus.RUNNING)
        self.assertEqual(close_calls_after_pre_close_settle, 1)
        self.assertEqual(third_status, NodeStatus.SUCCESS)
        self.assertFalse(shared["recovery_requested:left"])
        self.assertEqual(robot.close_gripper.call_count, 2)

    def test_close_gripper_reset_clears_command_timing(self) -> None:
        """Ensure a reset cannot reuse pre-close or post-command settling time."""
        node = behavior_tree_nodes.FrankaSetGripper()
        node._pre_open_elapsed = 0.25
        node._pre_close_elapsed = 0.25
        node._elapsed = 0.25

        node.on_reset(NodeResetReason.ABORTED)

        self.assertEqual(node._pre_open_elapsed, 0.0)
        self.assertEqual(node._pre_close_elapsed, 0.0)
        self.assertEqual(node._elapsed, 0.0)

    def test_open_gripper_waits_at_drop_pose_before_opening(self) -> None:
        """Hold the drop pose before opening, then apply the normal post-open settling time."""
        node = behavior_tree_nodes.FrankaSetGripper()
        context = mock.Mock()
        context.get_delta_time.return_value = 0.125
        context.get_input.side_effect = {
            "command": "open",
            "pre_open_settle_duration": 0.25,
            "settle_duration": 0.25,
        }.get
        robot = mock.Mock()

        with mock.patch.object(node, "_resolve_robot", return_value=robot):
            first_status = node.on_tick(context)
            second_status = node.on_tick(context)
            open_calls_after_pre_open_settle = robot.open_gripper.call_count
            third_status = node.on_tick(context)

        self.assertEqual(first_status, NodeStatus.RUNNING)
        self.assertEqual(second_status, NodeStatus.RUNNING)
        self.assertEqual(open_calls_after_pre_open_settle, 1)
        self.assertEqual(third_status, NodeStatus.SUCCESS)
        self.assertEqual(robot.open_gripper.call_count, 2)

    def test_close_gripper_handles_empty_pose_result(self) -> None:
        """Return failure when the monitored cube pose view is temporarily empty."""
        node = behavior_tree_nodes.FrankaSetGripper()
        context = mock.Mock()
        context.get_input.side_effect = {
            "command": "close",
            "settle_duration": 0.25,
        }.get
        pose = mock.Mock()
        pose.numpy.return_value = np.empty((0, 3), dtype=np.float32)
        cube = mock.Mock()
        cube.get_world_poses.return_value = (pose, None)

        with (
            mock.patch.object(node, "_resolve_robot", return_value=mock.Mock()),
            mock.patch.object(
                behavior_tree_nodes, "_get_local_blackboard", return_value={"current_cube": "/World/Cube"}
            ),
            mock.patch.object(behavior_tree_nodes, "RigidPrim", return_value=cube),
            mock.patch.object(behavior_tree_nodes.carb, "log_error") as log_error,
        ):
            status = node.on_tick(context)

        self.assertEqual(status, NodeStatus.FAILURE)
        log_error.assert_called_once()

    def test_calibrate_place_targets_handles_prim_failure(self) -> None:
        """Return failure instead of propagating a transient pose-query error."""
        node = behavior_tree_nodes.FrankaCalibratePlaceTargets()
        robot = mock.Mock()
        robot.get_tool_pose.side_effect = RuntimeError("transient prim")
        local = {"current_cube": "/World/Cube", "object_goal": "/World/Goal"}

        with (
            mock.patch.object(node, "_resolve_robot", return_value=robot),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
            mock.patch.object(behavior_tree_nodes.carb, "log_error") as log_error,
        ):
            status = node.on_tick(mock.Mock())

        self.assertEqual(status, NodeStatus.FAILURE)
        log_error.assert_called_once()

    def test_mark_cube_placed_uses_configured_completion_threshold(self) -> None:
        """Publish completion when the configured total placement count is reached."""
        node = behavior_tree_nodes.FrankaMarkCubePlaced()
        local = {
            "completion_key": "stack_complete:left",
            "current_cube": "/World/Pyramid/CubeTop",
            "recovery_key": "recovery_requested:left",
        }
        shared = {"placed_count:left": 1}
        context = mock.Mock()
        context.get_blackboard.return_value = shared
        context.get_input.return_value = 2

        with (
            mock.patch.object(node, "get_prim_path", return_value="/World/franka_left"),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
        ):
            status = node.on_tick(context)

        self.assertEqual(status, NodeStatus.SUCCESS)
        self.assertTrue(shared["stack_complete:left"])

    def test_release_and_mark_cube_publish_transfer_state(self) -> None:
        """Release the center lock and commit the completed placement."""
        cube_path = "/World/Pyramid/CubeTop"
        local = {
            "completion_key": "stack_complete:left",
            "current_cube": cube_path,
            "recovery_key": "recovery_requested:left",
        }
        shared = {
            "center_owner": "left",
            f"cube_state:{cube_path}": "reserved:left",
            "recovery_requested:left": True,
        }

        release = behavior_tree_nodes.FrankaReleaseCenter()
        mark = behavior_tree_nodes.FrankaMarkCubePlaced()

        release_context = mock.Mock()
        release_context.get_blackboard.return_value = shared
        mark_context = mock.Mock()
        mark_context.get_blackboard.return_value = shared
        mark_context.get_input.return_value = 1

        with (
            mock.patch.object(release, "get_prim_path", return_value="/World/franka_left"),
            mock.patch.object(mark, "get_prim_path", return_value="/World/franka_left"),
            mock.patch.object(behavior_tree_nodes, "_get_local_blackboard", return_value=local),
        ):
            release_status = release.on_tick(release_context)
            mark_status = mark.on_tick(mark_context)

        self.assertEqual(release_status, NodeStatus.SUCCESS)
        self.assertEqual(mark_status, NodeStatus.SUCCESS)
        self.assertEqual(shared["center_owner"], "")
        self.assertEqual(shared[f"cube_state:{cube_path}"], "placed:left")
        self.assertEqual(shared["placed_count:left"], 1)
        self.assertTrue(shared["stack_complete:left"])
        self.assertFalse(shared["recovery_requested:left"])
        self.assertEqual(local["current_cube"], "")
