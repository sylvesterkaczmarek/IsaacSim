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

# ruff: noqa: ANN205, D102

"""Ordinary-Python smoke test for the differentiable Newton backend."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from isaacsim.robot_setup.sysid.newton_diff_sysid_bridge import (
    NewtonDifferentiableSysIdBridge,
    newton_differentiable_modules_available,
)
from isaacsim.robot_setup.sysid.parameter_types import (
    SysIdParameterEntry,
    SysIdParameterType,
)
from isaacsim.robot_setup.sysid.run_spec import (
    NEWTON_SOLVER_FEATHERSTONE_DIFF,
    NewtonSimulationRunSpec,
)
from isaacsim.robot_setup.sysid.trajectory_csv import TrajectoryDataset

_NEWTON_AVAILABLE, _NEWTON_REASON = newton_differentiable_modules_available()
_REQUIRE_NEWTON = os.environ.get("ISAACSIM_SYSID_REQUIRE_NEWTON_TESTS", "0") == "1"


class TestNewtonDependencyGate(unittest.TestCase):
    """Allow local skips while giving Newton-enabled CI a mandatory gate."""

    def test_required_newton_runtime_is_available(self) -> None:
        if _REQUIRE_NEWTON:
            self.assertTrue(_NEWTON_AVAILABLE, _NEWTON_REASON)


@unittest.skipUnless(_NEWTON_AVAILABLE, _NEWTON_REASON)
class TestStandaloneFeatherstone(unittest.IsolatedAsyncioTestCase):
    """Run a deterministic differentiable rollout without Kit or USD."""

    @staticmethod
    def _builder():
        import newton
        import warp as wp

        builder = newton.ModelBuilder()
        link0 = builder.add_link(
            mass=1.0,
            com=(0.0, 0.0, -0.12),
            inertia=np.diag([0.015, 0.02, 0.025]).tolist(),
            label="link0",
        )
        link1 = builder.add_link(
            mass=0.6,
            com=(0.0, 0.0, -0.08),
            inertia=np.diag([0.008, 0.01, 0.012]).tolist(),
            label="link1",
        )
        joint0 = builder.add_joint_revolute(
            parent=-1,
            child=link0,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            target_ke=5.0,
            target_kd=0.35,
        )
        joint1 = builder.add_joint_revolute(
            parent=link0,
            child=link1,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, 0.0, -0.24), wp.quat_identity()),
            child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            target_ke=3.5,
            target_kd=0.25,
        )
        builder.add_articulation([joint0, joint1], label="standalone_pendulum")
        return builder

    async def test_rollout_torque_clamp_and_backward(self) -> None:
        num_steps = 16
        dt = 0.01
        times = np.arange(num_steps, dtype=np.float64) * dt
        commands_np = 1.5 * np.sin(2.0 * np.pi * 1.25 * times)[:, None] * np.array([[1.0, -0.8]])
        zeros = np.zeros_like(commands_np)
        snapshots = [
            SimpleNamespace(
                stiffness=5.0,
                damping=0.35,
                friction=0.1,
                dynamic_friction=0.1,
                max_force=0.3,
            ),
            SimpleNamespace(
                stiffness=3.5,
                damping=0.25,
                friction=0.08,
                dynamic_friction=0.08,
                max_force=0.3,
            ),
        ]
        bridge = NewtonDifferentiableSysIdBridge(
            robot_prim_path="/standalone_pendulum",
            newton_config=NewtonSimulationRunSpec(
                device="cpu",
                solver=NEWTON_SOLVER_FEATHERSTONE_DIFF,
                featherstone_substeps=2,
                effort_clamp="max_effort",
                cuda_graph_capture=False,
            ),
            joint_baselines=snapshots,
            link_paths=["link0", "link1"],
            num_joints=2,
            num_links=2,
            robot_builder=self._builder(),
        )
        bridge.set_trajectory(
            TrajectoryDataset(
                times=times,
                positions=zeros.copy(),
                velocities=zeros.copy(),
                commands=commands_np,
            )
        )
        entries = [
            SysIdParameterEntry(
                param_type=SysIdParameterType.JOINT_STIFFNESS,
                dof_index=0,
            ),
            SysIdParameterEntry(
                param_type=SysIdParameterType.JOINT_DAMPING,
                dof_index=1,
            ),
            SysIdParameterEntry(
                param_type=SysIdParameterType.LINK_MASS,
                dof_index=-1,
                link_index=0,
            ),
        ]
        theta = torch.ones((1, len(entries)), dtype=torch.float32, requires_grad=True)
        result = await bridge.run_rollout_async(
            theta,
            entries,
            torch.as_tensor(commands_np, dtype=torch.float32),
            num_steps,
        )

        self.assertEqual(tuple(result.positions.shape), (1, num_steps, 2))
        self.assertLessEqual(float(result.torques.detach().abs().max()), 0.30001)
        loss = result.positions.square().mean() + result.torques.square().mean()
        loss.backward()
        self.assertIsNotNone(theta.grad)
        self.assertTrue(torch.all(torch.isfinite(theta.grad)))
        self.assertGreater(float(theta.grad.detach().abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
