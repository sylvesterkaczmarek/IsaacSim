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

"""Sim-free ``derive_binding`` tests over representative descriptor fixtures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import omni.kit.test
import yaml

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "io_descriptors"


def _load(path: Path) -> dict:
    """Parse one descriptor fixture with the plain safe loader.

    Args:
        path: Path.

    Returns:
        The resulting dict.
    """
    return yaml.safe_load(path.read_text())


def _fixture(name: str) -> dict:
    return _load(_FIXTURE_DIR / name)


class TestDescriptorDerivedBinding(omni.kit.test.AsyncTestCase):
    """Descriptor-primary derivation and binding without a simulation."""

    def test_valid_minimal_round_trip(self) -> None:
        """The minimal anchored-YAML fixture derives, keeps overloads, and binds."""
        from isaacsim.robot.policy.examples.binding import bind_policy, derive_binding

        descriptor = _fixture("valid_minimal.yaml")
        descriptor["actions"][0]["scale"] = [0.25, 0.5]
        descriptor["actions"][0]["clip"] = [[-1.0, 1.0], [-2.0, 2.0]]
        binding = derive_binding(descriptor)
        last_action = binding.observation_terms[-1]
        self.assertEqual(last_action.semantic, "last_action")
        self.assertEqual(last_action.scale, 2.0)
        self.assertEqual(last_action.clip, [-100.0, 100.0])
        self.assertEqual(binding.observation_terms[0].offsets, (0.0, 0.5))
        self.assertEqual(binding.action_terms[0].scale, [0.25, 0.5])
        self.assertEqual(binding.action_terms[0].clip, [[-1.0, 1.0], [-2.0, 2.0]])
        self.assertEqual([term.history_length for term in binding.observation_terms], [1, 1])
        bound = bind_policy(binding, robot_joint_space=("joint_a", "joint_b"))
        self.assertEqual(bound.observation_width, 4)

    def test_multiple_action_terms_share_the_flat_model_output(self) -> None:
        """Ordered action terms consume adjacent slices and combine their named commands."""
        import numpy as np
        from isaacsim.robot.policy.examples.binding import bind_policy, derive_binding

        descriptor = _fixture("valid_minimal.yaml")
        first = descriptor["actions"][0]
        first["shape"] = [1]
        first["joint_names"] = ["joint_a"]
        first["offset"] = [0.0]
        second = {
            **first,
            "name": "second_action",
            "full_path": first["full_path"].replace("JointPositionAction", "JointVelocityAction"),
            "joint_names": ["joint_b"],
            "offset": [1.0],
        }
        descriptor["actions"] = [first, second]

        binding = derive_binding(descriptor)
        bound = bind_policy(binding, robot_joint_space=("joint_a", "joint_b"))
        desired = bound.action(np.array([2.0, 4.0], dtype=np.float32))

        self.assertEqual(len(binding.action_terms), 2)
        self.assertEqual(bound.action_width, 2)
        self.assertEqual(bound.joint_control_modes, {"joint_a": "position", "joint_b": "velocity"})
        self.assertEqual(list(desired.joints.position_names), ["joint_a"])
        self.assertEqual(list(desired.joints.velocity_names), ["joint_b"])
        np.testing.assert_array_equal(
            np.asarray(desired.joints.positions.numpy()).reshape(-1),
            np.array([1.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            np.asarray(desired.joints.velocities.numpy()).reshape(-1),
            np.array([3.0], dtype=np.float32),
        )

    def test_binary_position_action_commands_multiple_joints(self) -> None:
        """One binary model channel selects the same closed/open target for both fingers."""
        import numpy as np
        from isaacsim.robot.policy.examples.binding import PolicyBinding, Term, bind_policy

        binding = PolicyBinding(
            observation_terms=(),
            action_terms=(
                Term(
                    semantic="binary_joint_position",
                    width=1,
                    joint_names=("finger_left", "finger_right"),
                    binary_positions=(0.0, 0.04),
                ),
            ),
        )
        bound = bind_policy(binding, robot_joint_space=("finger_left", "finger_right"))

        closed = bound.action(np.array([-0.1], dtype=np.float32))
        opened = bound.action(np.array([0.0], dtype=np.float32))

        self.assertEqual(bound.action_width, 1)
        self.assertEqual(bound.joint_control_modes, {"finger_left": "position", "finger_right": "position"})
        np.testing.assert_array_equal(closed.joints.positions.numpy(), np.array([0.0, 0.0], dtype=np.float32))
        np.testing.assert_array_equal(opened.joints.positions.numpy(), np.array([0.04, 0.04], dtype=np.float32))

    def test_partial_joint_terms_are_record_local(self) -> None:
        """Observation and action subsets bind from their own records, independent of the sidecar."""
        from isaacsim.robot.policy.examples.binding import bind_policy, derive_binding

        descriptor = _fixture("valid_minimal.yaml")
        descriptor["observations"]["policy"] = descriptor["observations"]["policy"][:1]
        descriptor["actions"][0]["joint_names"] = ["joint_a", "joint_c"]
        descriptor["articulations"]["robot"]["joint_names"] = ["unrelated_sidecar_joint"]
        dof = ("joint_a", "joint_b", "joint_c")
        binding = derive_binding(descriptor)
        observation = binding.observation_terms[0]
        action = binding.action_terms[0]
        self.assertEqual(observation.joint_names, ("joint_a", "joint_b"))
        self.assertEqual(observation.offsets, (0.0, 0.5))
        self.assertEqual(action.joint_names, ("joint_a", "joint_c"))
        bound = bind_policy(binding, robot_joint_space=dof)
        self.assertEqual(bound.observation_width, 2)
        self.assertEqual(bound.action_width, 2)
        self.assertEqual(bound.joint_control_modes, {"joint_a": "position", "joint_c": "position"})

    def test_live_joint_resolution(self) -> None:
        """Binding refuses trained joints missing from the live robot joint space."""
        from isaacsim.robot.policy.examples.binding import bind_policy, derive_binding

        binding = derive_binding(_fixture("valid_minimal.yaml"))
        with self.assertRaises(ValueError):
            bind_policy(binding, robot_joint_space=("joint_a", "joint_c"))

    def test_history_overloads_expand_model_input_width(self) -> None:
        """History overloads expand each term while retaining its per-sample width."""
        from isaacsim.robot.policy.examples.binding import bind_policy, derive_binding

        descriptor = _fixture("valid_minimal.yaml")
        for observation in descriptor["observations"]["policy"]:
            observation["overloads"]["history_length"] = 5

        binding = derive_binding(descriptor)
        bound = bind_policy(binding, robot_joint_space=("joint_a", "joint_b"))

        self.assertEqual([term.history_length for term in binding.observation_terms], [5, 5])
        self.assertEqual([term.width for term in binding.observation_terms], [2, 2])
        self.assertEqual(bound.observation_sample_width, 4)
        self.assertEqual(bound.observation_width, 20)

    def test_term_rejects_nonpositive_history_length(self) -> None:
        """Require normalized history lengths on directly authored terms."""
        from isaacsim.robot.policy.examples.binding import Term

        with self.assertRaisesRegex(ValueError, "history_length must be at least 1"):
            Term(semantic="base_lin_vel", width=3, history_length=0)

    def test_controller_stacks_term_history_oldest_to_newest(self) -> None:
        """Backfill and advance independent term histories once per inference."""
        import numpy as np
        from isaacsim.robot.policy.examples.binding import bind_policy, derive_binding
        from isaacsim.robot.policy.examples.controller import IsaacLabPolicyController

        descriptor = _fixture("valid_minimal.yaml")
        for observation in descriptor["observations"]["policy"]:
            observation["overloads"]["history_length"] = 3
        interface = bind_policy(derive_binding(descriptor), robot_joint_space=("joint_a", "joint_b"))
        interface.observe = MagicMock(
            side_effect=[
                np.array([1.0, 2.0, 3.0, 4.0]),
                np.array([5.0, 6.0, 7.0, 8.0]),
                np.array([9.0, 10.0, 11.0, 12.0]),
            ]
        )
        interface.action = MagicMock(side_effect=lambda action: action)
        model = MagicMock(return_value=np.array([0.25, -0.5], dtype=np.float32))
        controller = IsaacLabPolicyController(model, interface)

        self.assertTrue(controller.reset(None, None, 0.0))
        controller.forward(None, None, 0.0)
        controller.forward(None, None, 0.02)
        self.assertTrue(controller.reset(None, None, 0.0))
        controller.forward(None, None, 0.0)

        expected = [
            [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
            [1.0, 2.0, 1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 3.0, 4.0, 7.0, 8.0],
            [9.0, 10.0, 9.0, 10.0, 9.0, 10.0, 11.0, 12.0, 11.0, 12.0, 11.0, 12.0],
        ]
        for call, expected_observation in zip(model.call_args_list, expected):
            np.testing.assert_array_equal(call.args[0], np.array(expected_observation, dtype=np.float32))

    def test_controller_rejects_wrong_current_frame_observation_width(self) -> None:
        """Reject a bound observation sample that does not match its per-frame width."""
        import numpy as np
        from isaacsim.robot.policy.examples.binding import bind_policy, derive_binding
        from isaacsim.robot.policy.examples.controller import IsaacLabPolicyController

        descriptor = _fixture("valid_minimal.yaml")
        for observation in descriptor["observations"]["policy"]:
            observation["overloads"]["history_length"] = 3
        interface = bind_policy(derive_binding(descriptor), robot_joint_space=("joint_a", "joint_b"))
        interface.observe = MagicMock(return_value=np.array([1.0, 2.0, 3.0], dtype=np.float32))
        model = MagicMock(return_value=np.array([0.25, -0.5], dtype=np.float32))
        controller = IsaacLabPolicyController(model, interface)

        self.assertTrue(controller.reset(None, None, 0.0))
        with self.assertRaises(ValueError) as error:
            controller.forward(None, None, 0.0)

        self.assertIn("shape (3,); expected (4,)", str(error.exception))
        model.assert_not_called()
