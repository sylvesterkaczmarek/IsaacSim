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

"""Regression tests for weighted pose residual construction."""

from __future__ import annotations

import math

import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.residual_engine import (
    ResidualSignalType,
    ResidualSpec,
    RolloutResidualInputs,
    WeightedResidualEngine,
)


def _pose_inputs(sim_quaternion: list[float], measured_quaternion: list[float]) -> RolloutResidualInputs:
    dtype = torch.float64
    zeros = torch.zeros(1, 1, 1, dtype=dtype)
    simulated_pose = torch.tensor([[[0.0, 0.0, 0.0, *sim_quaternion]]], dtype=dtype)
    measured_pose = torch.tensor([[0.0, 0.0, 0.0, *measured_quaternion]], dtype=dtype)
    return RolloutResidualInputs(
        meas_pos=zeros[0],
        meas_vel=zeros[0],
        sim_pos=zeros,
        sim_vel=zeros,
        meas_ee_pose=measured_pose,
        sim_ee_pose=simulated_pose,
    )


class ResidualEnginePoseTests(omni.kit.test.AsyncTestCase):
    """Verify the public residual engine's quaternion error convention."""

    def setUp(self) -> None:
        """Create an engine containing only the pose residual channel."""
        self._engine = WeightedResidualEngine((ResidualSpec(ResidualSignalType.END_EFFECTOR_POSE, 1.0),))

    async def test_identical_pose_has_zero_residual(self) -> None:
        """Return zero translation and rotation error for identical poses."""
        inputs = _pose_inputs([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0])

        residuals, costs = self._engine.compute(inputs)

        torch.testing.assert_close(residuals, torch.zeros(1, 6, dtype=torch.float64))
        torch.testing.assert_close(costs, torch.zeros(1, dtype=torch.float64))

    async def test_half_turn_has_pi_rotation_magnitude(self) -> None:
        """Represent a 180-degree quaternion difference with a length-pi vector."""
        inputs = _pose_inputs([1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

        residuals, costs = self._engine.compute(inputs)

        torch.testing.assert_close(
            torch.linalg.vector_norm(residuals[:, 3:], dim=-1),
            torch.tensor([math.pi], dtype=torch.float64),
        )
        torch.testing.assert_close(costs, torch.tensor([0.5 * math.pi**2], dtype=torch.float64))

    async def test_opposite_quaternion_sign_has_zero_residual(self) -> None:
        """Treat ``q`` and ``-q`` as the same physical orientation."""
        inputs = _pose_inputs([0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0, 1.0])

        residuals, costs = self._engine.compute(inputs)

        torch.testing.assert_close(residuals, torch.zeros(1, 6, dtype=torch.float64))
        torch.testing.assert_close(costs, torch.zeros(1, dtype=torch.float64))

    async def test_zero_norm_quaternion_is_rejected(self) -> None:
        """Reject invalid pose quaternions before constructing a residual."""
        inputs = _pose_inputs([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

        with self.assertRaisesRegex(ValueError, "zero or non-finite norm"):
            self._engine.compute(inputs)

    async def test_sample_weights_must_match_rollout_length(self) -> None:
        """Reject accidental scalar or partial broadcasting over timesteps."""
        engine = WeightedResidualEngine((ResidualSpec(ResidualSignalType.POSITION, 1.0),))
        zeros = torch.zeros(1, 2, 1, dtype=torch.float64)
        for weights in (torch.ones(1), torch.ones(3)):
            with self.subTest(weight_count=weights.numel()):
                inputs = RolloutResidualInputs(
                    meas_pos=zeros[0],
                    meas_vel=zeros[0],
                    sim_pos=zeros,
                    sim_vel=zeros,
                    sample_weights=weights,
                )
                with self.assertRaisesRegex(ValueError, "exactly 2"):
                    engine.compute(inputs)
