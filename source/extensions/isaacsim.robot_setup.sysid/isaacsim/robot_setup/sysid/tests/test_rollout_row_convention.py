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

# ruff: noqa: ANN001, ANN202, D102

"""Row-convention regression tests for the rollout bridges.

Contract (see ``env_bridge.SysIdEnvironmentBridge.run_rollout_async``): sim row
``k`` is the state at measured sample ``k`` — row 0 is the trajectory's initial
state and stepping under command row ``k`` produces row ``k + 1``. The row-0
anchor tests fail deterministically if any bridge regresses to returning
post-step states at row ``k`` (a one-sample skew the optimizer absorbs into
damping: negligible on slow arms, catastrophic on resonant systems).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.newton_diff_sysid_bridge import (
    NewtonDifferentiableSysIdBridge,
)
from isaacsim.robot_setup.sysid.newton_sysid_bridge import (
    NewtonSysIdBridge,
    newton_core_modules_available,
    newton_modules_available,
)
from isaacsim.robot_setup.sysid.parameter_types import (
    SysIdParameterEntry,
    SysIdParameterType,
)
from isaacsim.robot_setup.sysid.run_spec import (
    NEWTON_SOLVER_FEATHERSTONE_DIFF,
    NEWTON_SOLVER_MUJOCO,
    NewtonSimulationRunSpec,
)
from isaacsim.robot_setup.sysid.trajectory_csv import TrajectoryDataset

_NEWTON_CORE_AVAILABLE, _NEWTON_CORE_REASON = newton_core_modules_available()
_NEWTON_MUJOCO_AVAILABLE, _NEWTON_MUJOCO_REASON = newton_modules_available()


class RowConventionAnchorTests(omni.kit.test.AsyncTestCase):
    """Row 0 of every bridge's rollout is the trajectory's initial state."""

    def setUp(self) -> None:
        if not _NEWTON_CORE_AVAILABLE:
            self.skipTest(_NEWTON_CORE_REASON)

    def _two_link_trajectory(self, num_steps: int = 20, num_dof: int = 2) -> TrajectoryDataset:
        times = np.arange(num_steps, dtype=np.float64) * 0.01
        positions = np.zeros((num_steps, num_dof), dtype=np.float64)
        velocities = np.zeros((num_steps, num_dof), dtype=np.float64)
        positions[0] = [0.3, -0.2][:num_dof]
        velocities[0] = [0.5, -0.4][:num_dof]
        commands = 0.3 * np.sin(2.0 * np.pi * 1.5 * times)[:, None] * np.ones((1, num_dof))
        return TrajectoryDataset(times=times, positions=positions, velocities=velocities, commands=commands)

    def _two_link_builder(self):
        import newton
        import warp as wp

        builder = newton.ModelBuilder()
        joints = []
        parent = -1
        for index in range(2):
            link = builder.add_link(mass=1.0, com=(0.0, 0.0, -0.1), inertia=np.diag([0.01, 0.012, 0.014]).tolist())
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=(0.0, 1.0, 0.0),
                    parent_xform=wp.transform((0.0, 0.0, 1.0 if index == 0 else -0.2), wp.quat_identity()),
                    child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
                    target_ke=5.0,
                    target_kd=0.4,
                    label=f"/robot/joint{index}",
                )
            )
            parent = link
        builder.add_articulation(joints, label="chain")
        return builder

    def _snapshots(self, count: int) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                stiffness=5.0,
                damping=0.4,
                friction=0.1,
                dynamic_friction=0.1,
                armature=0.05,
                max_force=float("inf"),
            )
            for _ in range(count)
        ]

    def _assert_row0_is_initial_state(self, result, trajectory: TrajectoryDataset, atol: float = 1e-5) -> None:
        expected_q = torch.as_tensor(trajectory.positions[0], dtype=torch.float32)
        expected_qd = torch.as_tensor(trajectory.velocities[0], dtype=torch.float32)
        for env in range(int(result.positions.shape[0])):
            self.assertTrue(
                torch.allclose(result.positions[env, 0].cpu(), expected_q, atol=atol),
                msg=f"env {env} row 0 positions {result.positions[env, 0].tolist()} != initial {expected_q.tolist()}",
            )
            self.assertTrue(
                torch.allclose(result.velocities[env, 0].cpu(), expected_qd, atol=atol),
                msg=f"env {env} row 0 velocities {result.velocities[env, 0].tolist()} != initial {expected_qd.tolist()}",
            )
        # Physics actually advances: row 1 differs from the anchored row 0.
        self.assertGreater(float((result.positions[:, 1] - result.positions[:, 0]).abs().max()), 1e-6)

    async def test_diff_bridge_row0_is_initial_state(self) -> None:
        trajectory = self._two_link_trajectory()
        config = NewtonSimulationRunSpec(device="cpu", solver=NEWTON_SOLVER_FEATHERSTONE_DIFF, featherstone_substeps=2)
        bridge = NewtonDifferentiableSysIdBridge(
            robot_prim_path="/robot",
            newton_config=config,
            joint_baselines=self._snapshots(2),
            joint_paths=["/robot/joint0", "/robot/joint1"],
            num_joints=2,
            num_links=2,
            robot_builder=self._two_link_builder(),
        )
        bridge.set_trajectory(trajectory)
        commands = torch.as_tensor(trajectory.commands, dtype=torch.float32)
        theta = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
        entries = [
            SysIdParameterEntry(SysIdParameterType.JOINT_STIFFNESS, dof_index=0),
            SysIdParameterEntry(SysIdParameterType.JOINT_STIFFNESS, dof_index=1),
        ]
        result = await bridge.run_rollout_async(theta, entries, commands, int(trajectory.commands.shape[0]))
        self._assert_row0_is_initial_state(result, trajectory)

    async def test_mujoco_bridge_row0_is_initial_state(self) -> None:
        if not _NEWTON_MUJOCO_AVAILABLE:
            self.skipTest(_NEWTON_MUJOCO_REASON)
        trajectory = self._two_link_trajectory()
        bridge = NewtonSysIdBridge(
            robot_prim_path="/robot",
            newton_config=NewtonSimulationRunSpec(solver=NEWTON_SOLVER_MUJOCO),
            joint_baselines=self._snapshots(2),
            joint_paths=["/robot/joint0", "/robot/joint1"],
            num_joints=2,
            robot_builder=self._two_link_builder(),
        )
        bridge.set_trajectory(trajectory)
        commands = torch.as_tensor(trajectory.commands, dtype=torch.float32)
        theta = torch.ones((1, 2), dtype=torch.float32)
        entries = [
            SysIdParameterEntry(SysIdParameterType.JOINT_STIFFNESS, dof_index=0),
            SysIdParameterEntry(SysIdParameterType.JOINT_STIFFNESS, dof_index=1),
        ]
        result = await bridge.run_rollout_async(theta, entries, commands, int(trajectory.commands.shape[0]))
        self._assert_row0_is_initial_state(result, trajectory)
