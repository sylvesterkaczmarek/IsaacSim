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

# ruff: noqa: ANN001, ANN002, ANN003, ANN202, D102

"""Tests for the contact-free differentiable Newton bridge.

The gradient-correctness test is the gate for this bridge: it compares autograd
gradients against central finite differences of the same loss on a programmatic
double pendulum (no USD stage required). Newton-dependent tests skip when the
optional Newton/Warp modules are not installed.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.env_bridge import SysIdEnvironmentBridgeError
from isaacsim.robot_setup.sysid.inertia_param import inertia_matrix_to_log_cholesky
from isaacsim.robot_setup.sysid.newton_diff_sysid_bridge import (
    NewtonDifferentiableSysIdBridge,
    _require_finite_array,
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


def _cuda_available() -> bool:
    if not _NEWTON_AVAILABLE:
        return False
    try:
        import warp as wp

        return wp.get_cuda_device_count() > 0
    except Exception:
        return False


_NUM_DOF = 2
_NUM_STEPS = 30
_DT = 0.01
_LINK0_INERTIA = np.diag([0.02, 0.03, 0.04])
_LINK1_INERTIA = np.diag([0.01, 0.015, 0.02])


class NewtonDifferentiableValidationTests(omni.kit.test.AsyncTestCase):
    """Validate fail-closed optimizer boundary helpers without requiring Newton."""

    async def test_nonfinite_backward_arrays_are_rejected(self) -> None:
        """NaN adjoints must never be silently converted to zero gradients."""
        with self.assertRaises(SysIdEnvironmentBridgeError):
            _require_finite_array(np.asarray([1.0, np.nan], dtype=np.float32), "test adjoint")


def _entry(
    ptype: SysIdParameterType,
    dof_index: int = -1,
    link_index: int = -1,
    component_index: int = 0,
):
    return SysIdParameterEntry(
        param_type=ptype,
        dof_index=dof_index,
        link_index=link_index,
        component_index=component_index,
    )


def _build_pendulum():
    import newton
    import warp as wp

    builder = newton.ModelBuilder()
    b0 = builder.add_link(mass=1.2, com=(0.0, 0.0, -0.15), inertia=_LINK0_INERTIA.tolist(), label="link0")
    b1 = builder.add_link(mass=0.7, com=(0.0, 0.0, -0.1), inertia=_LINK1_INERTIA.tolist(), label="link1")
    j0 = builder.add_joint_revolute(
        parent=-1,
        child=b0,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        target_ke=6.0,
        target_kd=0.4,
    )
    j1 = builder.add_joint_revolute(
        parent=b0,
        child=b1,
        axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform((0.0, 0.0, -0.3), wp.quat_identity()),
        child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
        target_ke=4.0,
        target_kd=0.3,
    )
    builder.add_articulation([j0, j1], label="pendulum")
    return builder


def _make_bridge(
    feedforward: str = "none",
    device: str = "cpu",
    cuda_graph_capture: bool = True,
    effort_clamp: str = "max_effort",
    max_force: float = float("inf"),
) -> tuple[NewtonDifferentiableSysIdBridge, torch.Tensor]:
    snapshots = [
        SimpleNamespace(
            stiffness=6.0,
            damping=0.4,
            friction=0.25,
            dynamic_friction=0.25,
            max_force=max_force,
        ),
        SimpleNamespace(
            stiffness=4.0,
            damping=0.3,
            friction=0.15,
            dynamic_friction=0.15,
            max_force=max_force,
        ),
    ]
    config = NewtonSimulationRunSpec(
        device=device,
        solver=NEWTON_SOLVER_FEATHERSTONE_DIFF,
        featherstone_substeps=2,
        feedforward=feedforward,
        cuda_graph_capture=cuda_graph_capture,
        effort_clamp=effort_clamp,
    )
    bridge = NewtonDifferentiableSysIdBridge(
        robot_prim_path="/pendulum",
        newton_config=config,
        joint_baselines=snapshots,
        num_joints=_NUM_DOF,
        num_links=2,
        robot_builder=_build_pendulum(),
    )
    times = np.arange(_NUM_STEPS, dtype=np.float64) * _DT
    zeros = np.zeros((_NUM_STEPS, _NUM_DOF), dtype=np.float64)
    commands = 0.4 * np.sin(2.0 * np.pi * 1.5 * times)[:, None] * np.array([[1.0, -0.7]])
    bridge.set_trajectory(
        TrajectoryDataset(
            times=times,
            positions=zeros.copy(),
            velocities=zeros.copy(),
            commands=commands,
        )
    )
    return bridge, torch.as_tensor(commands, dtype=torch.float32)


def _hold_trajectory(value: float, num_steps: int = _NUM_STEPS) -> TrajectoryDataset:
    """Constant-pose trajectory chunk; distinct `value`s give distinct feedforwards.

    Args:
        value: Value supplied for ``value``.
        num_steps: Value supplied for ``num_steps``.

    Returns:
        Result produced by the operation.
    """
    times = np.arange(num_steps, dtype=np.float64) * _DT
    hold = np.full((num_steps, _NUM_DOF), value, dtype=np.float64) * np.array([[1.0, -0.5]])
    return TrajectoryDataset(
        times=times,
        positions=hold.copy(),
        velocities=np.zeros_like(hold),
        commands=hold.copy(),
    )


def _count_feedforward_computes():
    """Patch the module-level feedforward computation with a counting wrapper.

    Returns:
        Result produced by the operation.
    """
    import isaacsim.robot_setup.sysid.newton_diff_sysid_bridge as diff_module

    calls = {"count": 0}
    original = diff_module._compute_feedforward

    def counting(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    diff_module._compute_feedforward = counting
    return calls, lambda: setattr(diff_module, "_compute_feedforward", original)


_GRADCHECK_ENTRIES = [
    _entry(SysIdParameterType.JOINT_FRICTION, dof_index=0),
    _entry(SysIdParameterType.JOINT_STIFFNESS, dof_index=0),
    _entry(SysIdParameterType.JOINT_DAMPING, dof_index=1),
    _entry(SysIdParameterType.LINK_MASS),
    _entry(SysIdParameterType.LINK_COM_OFFSET_Z),
    # Component 2 (log L11 -> I_yy) is the inertia entry that affects Y-axis planar dynamics.
    _entry(SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY, component_index=2),
]


def _gradcheck_steps(entries: Sequence[SysIdParameterEntry]) -> np.ndarray:
    """Choose finite-difference steps from each parameter's numerical scale.

    Args:
        entries: Parameter entries being checked.

    Returns:
        One finite-difference step per entry, in matching order.
    """
    return np.asarray(
        [1e-4 if entry.param_type == SysIdParameterType.LINK_INERTIA_LOG_CHOLESKY else 1e-3 for entry in entries],
        dtype=np.float64,
    )


class NewtonDiffBridgeConstructorTests(omni.kit.test.AsyncTestCase):
    """Configuration contracts that are validated before optional modules load."""

    async def test_rejects_mujoco_solver_configuration(self) -> None:
        with self.assertRaises(SysIdEnvironmentBridgeError):
            NewtonDifferentiableSysIdBridge(
                robot_prim_path="/robot",
                newton_config=NewtonSimulationRunSpec(solver="mujoco"),
            )


class NewtonDiffBridgeValidationTests(omni.kit.test.AsyncTestCase):
    """Parameter validation runs without Newton installed."""

    def _bridge_for_validation(self) -> NewtonDifferentiableSysIdBridge:
        if not _NEWTON_AVAILABLE:
            self.skipTest(_NEWTON_REASON)
        bridge, _commands = _make_bridge()
        return bridge

    async def test_rejects_armature_with_fallback_hint(self) -> None:
        bridge = self._bridge_for_validation()
        with self.assertRaises(SysIdEnvironmentBridgeError) as raised:
            bridge.validate_parameter_entries([_entry(SysIdParameterType.JOINT_ARMATURE, dof_index=0)])
        message = str(raised.exception).lower()
        self.assertIn("armature", message)
        self.assertIn("mujoco", message)

    async def test_rejects_non_differentiable_types(self) -> None:
        bridge = self._bridge_for_validation()
        for ptype in (
            SysIdParameterType.ACTUATOR_COMMAND_DELAY_SECONDS,
            SysIdParameterType.JOINT_INTEGRAL_GAIN,
            SysIdParameterType.JOINT_LIMIT_LOWER_SCALE,
        ):
            with self.subTest(ptype=ptype.value):
                with self.assertRaises(SysIdEnvironmentBridgeError):
                    bridge.validate_parameter_entries([_entry(ptype, dof_index=0)])

    async def test_accepts_supported_types(self) -> None:
        bridge = self._bridge_for_validation()
        bridge.validate_parameter_entries(_GRADCHECK_ENTRIES)


class NewtonDiffBridgeRolloutTests(omni.kit.test.AsyncTestCase):
    """Gradient correctness and rollout behavior on a programmatic pendulum."""

    def setUp(self) -> None:
        if not _NEWTON_AVAILABLE:
            self.skipTest(_NEWTON_REASON)

    def _theta0(self) -> np.ndarray:
        lc0 = inertia_matrix_to_log_cholesky(_LINK0_INERTIA)
        return np.array([1.2, 0.9, 1.1, 1.05, 0.01, float(lc0[2]) + 0.05], dtype=np.float64)

    async def _loss(self, bridge, commands, theta_np, weights, requires_grad=False):
        theta = torch.as_tensor(theta_np, dtype=torch.float32).reshape(1, -1)
        if requires_grad:
            theta.requires_grad_(True)
        result = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        w_pos, w_vel = weights
        return theta, (result.positions * w_pos).sum() + (result.velocities * w_vel).sum()

    async def test_cancellation_after_rollout_releases_backward_lease(self) -> None:
        bridge, commands = _make_bridge()
        update_count = 0

        async def cancel_after_rollout() -> None:
            nonlocal update_count
            update_count += 1
            if update_count == 2:
                raise asyncio.CancelledError()

        bridge._next_update_async = cancel_after_rollout
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1).requires_grad_(True)

        with self.assertRaises(asyncio.CancelledError):
            await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)

        self.assertFalse(bridge._backward_pending)

    async def test_gradients_match_finite_differences(self) -> None:
        bridge, commands = _make_bridge()
        generator = torch.Generator().manual_seed(3)
        weights = (
            torch.rand(1, _NUM_STEPS, _NUM_DOF, generator=generator) - 0.5,
            0.1 * (torch.rand(1, _NUM_STEPS, _NUM_DOF, generator=generator) - 0.5),
        )
        theta0 = self._theta0()

        theta, loss = await self._loss(bridge, commands, theta0, weights, requires_grad=True)
        loss.backward()
        grad_autodiff = theta.grad.reshape(-1).double().numpy()

        steps = _gradcheck_steps(_GRADCHECK_ENTRIES)
        for index in range(len(theta0)):
            plus = theta0.copy()
            plus[index] += steps[index]
            minus = theta0.copy()
            minus[index] -= steps[index]
            _, loss_plus = await self._loss(bridge, commands, plus, weights)
            _, loss_minus = await self._loss(bridge, commands, minus, weights)
            grad_fd = (float(loss_plus) - float(loss_minus)) / (2.0 * steps[index])
            with self.subTest(param=_GRADCHECK_ENTRIES[index].param_type.value):
                denominator = max(abs(grad_fd), 1e-4)
                self.assertLess(
                    abs(grad_autodiff[index] - grad_fd) / denominator,
                    5e-2,
                    msg=f"autodiff={grad_autodiff[index]:.6g} fd={grad_fd:.6g}",
                )

    async def test_batch_rows_are_independent_and_cache_reuse_is_deterministic(
        self,
    ) -> None:
        bridge, commands = _make_bridge()
        theta0 = self._theta0()
        theta_batch = torch.as_tensor(np.stack([theta0, theta0 * 1.1, theta0]), dtype=torch.float32)

        first = await bridge.run_rollout_async(theta_batch, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertTrue(torch.allclose(first.positions[0], first.positions[2], atol=1e-6))
        self.assertFalse(torch.allclose(first.positions[0], first.positions[1], atol=1e-4))

        second = await bridge.run_rollout_async(theta_batch, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertTrue(torch.allclose(first.positions, second.positions, atol=1e-6))
        self.assertTrue(torch.allclose(first.velocities, second.velocities, atol=1e-6))

    async def test_rejects_second_differentiable_forward_before_backward(self) -> None:
        bridge, commands = _make_bridge()
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1).requires_grad_(True)
        first = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)

        with self.assertRaises(SysIdEnvironmentBridgeError):
            await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)

        first.positions.sum().backward()
        second = await bridge.run_rollout_async(theta.detach(), _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertTrue(torch.all(torch.isfinite(second.positions)))

    async def test_chunk_reset_uses_trajectory_initial_state(self) -> None:
        bridge, commands = _make_bridge()
        theta0 = self._theta0()
        theta = torch.as_tensor(theta0, dtype=torch.float32).reshape(1, -1)
        baseline = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)

        times = np.arange(_NUM_STEPS, dtype=np.float64) * _DT
        offset = np.full((_NUM_STEPS, _NUM_DOF), 0.2, dtype=np.float64)
        bridge.set_trajectory(
            TrajectoryDataset(
                times=times,
                positions=offset.copy(),
                velocities=np.zeros_like(offset),
                commands=commands.numpy().astype(np.float64),
            )
        )
        shifted = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertFalse(torch.allclose(baseline.positions, shifted.positions, atol=1e-3))
        self.assertGreater(
            float(shifted.positions[0, 0].mean()),
            float(baseline.positions[0, 0].mean()),
        )

    async def test_torque_channel_matches_pd_friction_convention(self) -> None:
        bridge, commands = _make_bridge()
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)
        result = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertEqual(tuple(result.torques.shape), (1, _NUM_STEPS, _NUM_DOF))
        self.assertTrue(torch.all(torch.isfinite(result.torques)))

    async def test_effort_clamp_limits_applied_and_reported_torque(self) -> None:
        bridge, commands = _make_bridge(max_force=0.2)
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1).requires_grad_(True)
        result = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands * 20.0, _NUM_STEPS)

        self.assertLessEqual(float(result.torques.detach().abs().max()), 0.20001)
        result.torques.square().sum().backward()
        self.assertTrue(torch.all(torch.isfinite(theta.grad)))

    async def test_unresolved_joint_path_is_fatal(self) -> None:
        """A bad path must not silently fall back to Newton import order."""
        bridge, commands = _make_bridge()
        bridge._joint_paths = ["/pendulum/missing_joint", "/pendulum/joint_2"]
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)

        with self.assertRaises(SysIdEnvironmentBridgeError) as raised:
            await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)

        self.assertIn("cannot uniquely map", str(raised.exception).lower())

    async def test_trajectory_may_cover_subset_of_actuated_joints(self) -> None:
        """A 3-joint model with 2-DOF telemetry maps by joint path; the extra joint holds."""
        import newton
        import warp as wp

        builder = newton.ModelBuilder()
        links = [
            builder.add_link(
                mass=1.0,
                com=(0.0, 0.0, -0.1),
                inertia=np.diag([0.01, 0.012, 0.014]).tolist(),
            )
            for _ in range(3)
        ]
        joints = []
        parent = -1
        for index, link in enumerate(links):
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
        builder.add_articulation(joints, label="chain")

        config = NewtonSimulationRunSpec(
            device="cpu",
            solver=NEWTON_SOLVER_FEATHERSTONE_DIFF,
            featherstone_substeps=2,
        )
        bridge = NewtonDifferentiableSysIdBridge(
            robot_prim_path="/robot",
            newton_config=config,
            joint_baselines=[SimpleNamespace(stiffness=5.0, damping=0.4, friction=0.1, dynamic_friction=0.1)] * 2,
            joint_paths=[
                "/robot/joint0",
                "/robot/joint2",
            ],  # telemetry covers joints 0 and 2 only
            num_joints=2,
            num_links=3,
            robot_builder=builder,
        )
        times = np.arange(_NUM_STEPS, dtype=np.float64) * _DT
        zeros = np.zeros((_NUM_STEPS, 2), dtype=np.float64)
        commands = 0.3 * np.sin(2.0 * np.pi * 1.5 * times)[:, None] * np.array([[1.0, -1.0]])
        bridge.set_trajectory(
            TrajectoryDataset(
                times=times,
                positions=zeros.copy(),
                velocities=zeros.copy(),
                commands=commands,
            )
        )
        entries = [
            _entry(SysIdParameterType.JOINT_STIFFNESS, dof_index=0),
            _entry(SysIdParameterType.JOINT_STIFFNESS, dof_index=1),
        ]
        theta = torch.tensor([[1.0, 1.0]], requires_grad=True)
        result = await bridge.run_rollout_async(
            theta, entries, torch.as_tensor(commands, dtype=torch.float32), _NUM_STEPS
        )
        self.assertEqual(tuple(result.positions.shape), (1, _NUM_STEPS, 2))
        self.assertTrue(torch.all(torch.isfinite(result.positions)))
        # Both mapped joints respond to their commands; gradients still flow.
        self.assertGreater(float(result.positions[0, :, 0].detach().abs().max()), 0.02)
        self.assertGreater(float(result.positions[0, :, 1].detach().abs().max()), 0.02)
        result.positions.sum().backward()
        self.assertTrue(torch.all(torch.isfinite(theta.grad)))

    async def test_gravity_compensation_removes_static_sag_and_keeps_gradients(
        self,
    ) -> None:
        """The nominal-model feedforward holds a commanded pose against gravity.

        Without compensation the pendulum sags below a constant target by
        roughly gravity_torque / ke; with it the steady-state error should
        shrink by an order of magnitude, gradients must stay finite, and the
        torque channel must carry the feedforward (gravity-scale bias).
        """
        num_steps = 80
        times = np.arange(num_steps, dtype=np.float64) * _DT
        hold = np.full((num_steps, _NUM_DOF), 0.6, dtype=np.float64) * np.array([[1.0, -0.5]])
        theta = self._theta0()

        errors: dict[bool, float] = {}
        torque_means: dict[bool, np.ndarray] = {}
        for compensate in (False, True):
            bridge, _ = _make_bridge(feedforward="gravity" if compensate else "none")
            bridge.set_trajectory(
                TrajectoryDataset(
                    times=times,
                    positions=hold.copy(),  # feedforward evaluated at the held pose
                    velocities=np.zeros_like(hold),
                    commands=hold.copy(),
                )
            )
            theta_t = torch.as_tensor(theta, dtype=torch.float32).reshape(1, -1).requires_grad_(True)
            result = await bridge.run_rollout_async(
                theta_t,
                _GRADCHECK_ENTRIES,
                torch.as_tensor(hold, dtype=torch.float32),
                num_steps,
            )
            self.assertTrue(torch.all(torch.isfinite(result.positions)))
            tail = result.positions[0, num_steps // 2 :, :]
            errors[compensate] = float(
                (tail - torch.as_tensor(hold[num_steps // 2 :], dtype=torch.float32)).detach().abs().max()
            )
            torque_means[compensate] = result.torques[0].mean(dim=0).detach().numpy()
            result.positions.sum().backward()
            self.assertTrue(torch.all(torch.isfinite(theta_t.grad)))

        self.assertLess(errors[True], 0.15 * errors[False])
        # The compensated torque channel reports applied torque including the
        # feedforward; near-perfect holding means it is gravity-dominated.
        self.assertGreater(float(np.abs(torque_means[True] - torque_means[False]).max()), 0.1)

    async def test_gradients_match_finite_differences_with_gravity_compensation(
        self,
    ) -> None:
        """The feedforward is a taped constant; gradients must stay FD-exact."""
        bridge, commands = _make_bridge(feedforward="gravity")
        generator = torch.Generator().manual_seed(5)
        weights = (
            torch.rand(1, _NUM_STEPS, _NUM_DOF, generator=generator) - 0.5,
            0.1 * (torch.rand(1, _NUM_STEPS, _NUM_DOF, generator=generator) - 0.5),
        )
        theta0 = self._theta0()
        theta, loss = await self._loss(bridge, commands, theta0, weights, requires_grad=True)
        loss.backward()
        grad_autodiff = theta.grad.reshape(-1).double().numpy()
        steps = _gradcheck_steps(_GRADCHECK_ENTRIES)
        for index in range(len(theta0)):
            plus, minus = theta0.copy(), theta0.copy()
            plus[index] += steps[index]
            minus[index] -= steps[index]
            _, loss_plus = await self._loss(bridge, commands, plus, weights)
            _, loss_minus = await self._loss(bridge, commands, minus, weights)
            grad_fd = (float(loss_plus) - float(loss_minus)) / (2.0 * steps[index])
            with self.subTest(param=_GRADCHECK_ENTRIES[index].param_type.value):
                self.assertLess(
                    abs(grad_autodiff[index] - grad_fd) / max(abs(grad_fd), 1e-4),
                    5e-2,
                    msg=f"autodiff={grad_autodiff[index]:.6g} fd={grad_fd:.6g}",
                )

    async def test_feedforward_matches_analytic_single_pendulum(self) -> None:
        """The numpy Newton-Euler matches the closed-form single-pendulum torque.

        For a link with mass m, CoM at distance l below a revolute Y-axis joint,
        and CoM inertia I_yy: tau = (I_yy + m l^2) qdd + m g l sin(q).
        """
        import newton
        import warp as wp
        from isaacsim.robot_setup.sysid.newton_diff_sysid_bridge import (
            _compute_feedforward,
        )

        mass, length, iyy = 1.7, 0.22, 0.045
        builder = newton.ModelBuilder()
        link = builder.add_link(
            mass=mass,
            com=(0.0, 0.0, -length),
            inertia=np.diag([0.03, iyy, 0.05]).tolist(),
        )
        joint = builder.add_joint_revolute(
            parent=-1,
            child=link,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
            child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            target_ke=5.0,
            target_kd=0.5,
        )
        builder.add_articulation([joint], label="pendulum1")
        config = NewtonSimulationRunSpec(
            device="cpu",
            solver=NEWTON_SOLVER_FEATHERSTONE_DIFF,
            featherstone_substeps=2,
            feedforward="inverse_dynamics",
        )
        bridge = NewtonDifferentiableSysIdBridge(
            robot_prim_path="/p1",
            newton_config=config,
            joint_baselines=[SimpleNamespace(stiffness=5.0, damping=0.5, friction=0.0, dynamic_friction=0.0)],
            num_joints=1,
            num_links=1,
            robot_builder=builder,
        )
        num_steps = 40
        times = np.arange(num_steps, dtype=np.float64) * _DT
        q = 0.7 * np.sin(2.0 * np.pi * 1.2 * times)[:, None]
        qd = 0.7 * 2.0 * np.pi * 1.2 * np.cos(2.0 * np.pi * 1.2 * times)[:, None]
        qdd = -0.7 * (2.0 * np.pi * 1.2) ** 2 * np.sin(2.0 * np.pi * 1.2 * times)[:, None]
        bridge.set_trajectory(
            TrajectoryDataset(times=times, positions=q.copy(), velocities=qd.copy(), commands=q.copy())
        )
        theta = torch.tensor([[1.0]])
        entries = [_entry(SysIdParameterType.JOINT_STIFFNESS, dof_index=0)]
        await bridge.run_rollout_async(theta, entries, torch.as_tensor(q, dtype=torch.float32), 2)
        ctx = bridge._ctx_cache[1]

        feedforward = _compute_feedforward(ctx, newton, q, qd, qdd)[:, 0]
        gravity = 9.80665
        analytic = (iyy + mass * length * length) * qdd[:, 0] + mass * gravity * length * np.sin(q[:, 0])
        max_err = float(np.abs(feedforward - analytic).max())
        self.assertLess(max_err, 1e-3 * max(1.0, float(np.abs(analytic).max())))

    async def test_inverse_dynamics_feedforward_tracks_dynamic_trajectory(self) -> None:
        """Full-ID feedforward supplies the motion torque; PD correction shrinks.

        Uses a single-link pendulum so the joint-space mass matrix is
        configuration-independent and the remaining tracking error isolates the
        feedforward quality.
        """
        import newton
        import warp as wp

        mass, length, iyy = 1.7, 0.22, 0.045
        num_steps = 60
        times = np.arange(num_steps, dtype=np.float64) * _DT
        omega_cmd = 2.0 * np.pi * 1.2
        commands = 0.5 * np.sin(omega_cmd * times)[:, None]
        velocities = 0.5 * omega_cmd * np.cos(omega_cmd * times)[:, None]

        errors: dict[str, float] = {}
        for mode in ("none", "inverse_dynamics"):
            builder = newton.ModelBuilder()
            link = builder.add_link(
                mass=mass,
                com=(0.0, 0.0, -length),
                inertia=np.diag([0.03, iyy, 0.05]).tolist(),
            )
            joint = builder.add_joint_revolute(
                parent=-1,
                child=link,
                axis=(0.0, 1.0, 0.0),
                parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()),
                child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
                target_ke=5.0,
                target_kd=0.5,
            )
            builder.add_articulation([joint], label="pendulum1")
            config = NewtonSimulationRunSpec(
                device="cpu",
                solver=NEWTON_SOLVER_FEATHERSTONE_DIFF,
                featherstone_substeps=4,
                feedforward=mode,
            )
            bridge = NewtonDifferentiableSysIdBridge(
                robot_prim_path="/p1",
                newton_config=config,
                joint_baselines=[SimpleNamespace(stiffness=5.0, damping=0.5, friction=0.0, dynamic_friction=0.0)],
                num_joints=1,
                num_links=1,
                robot_builder=builder,
            )
            bridge.set_trajectory(
                TrajectoryDataset(
                    times=times,
                    positions=commands.copy(),
                    velocities=velocities.copy(),
                    commands=commands.copy(),
                )
            )
            theta = torch.tensor([[1.0]], requires_grad=True)
            entries = [_entry(SysIdParameterType.JOINT_STIFFNESS, dof_index=0)]
            result = await bridge.run_rollout_async(
                theta,
                entries,
                torch.as_tensor(commands, dtype=torch.float32),
                num_steps,
            )
            self.assertTrue(torch.all(torch.isfinite(result.positions)))
            tail = slice(num_steps // 3, None)
            errors[mode] = float(
                (result.positions[0, tail] - torch.as_tensor(commands, dtype=torch.float32)[tail]).abs().mean().detach()
            )
            result.positions.sum().backward()
            self.assertTrue(torch.all(torch.isfinite(theta.grad)))
        self.assertLess(errors["inverse_dynamics"], 0.25 * errors["none"])

    async def test_feedforward_rejects_floating_base(self) -> None:
        import newton
        import warp as wp

        builder = newton.ModelBuilder()
        base = builder.add_link(mass=2.0, com=(0.0, 0.0, 0.0), inertia=np.diag([0.1, 0.1, 0.1]).tolist())
        base_joint = builder.add_joint_free(child=base)
        arm = builder.add_link(
            mass=0.7,
            com=(0.0, 0.0, -0.1),
            inertia=np.diag([0.01, 0.015, 0.02]).tolist(),
        )
        rev = builder.add_joint_revolute(
            parent=base,
            child=arm,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform((0.0, 0.0, 0.2), wp.quat_identity()),
            child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
            target_ke=4.0,
            target_kd=0.3,
        )
        builder.add_articulation([base_joint, rev], label="floating")
        config = NewtonSimulationRunSpec(
            device="cpu",
            solver=NEWTON_SOLVER_FEATHERSTONE_DIFF,
            featherstone_substeps=2,
            feedforward="gravity",
        )
        bridge = NewtonDifferentiableSysIdBridge(
            robot_prim_path="/floating",
            newton_config=config,
            joint_baselines=[SimpleNamespace(stiffness=4.0, damping=0.3, friction=0.0, dynamic_friction=0.0)],
            num_joints=1,
            num_links=2,
            robot_builder=builder,
        )
        num_steps = 10
        times = np.arange(num_steps, dtype=np.float64) * _DT
        zeros = np.zeros((num_steps, 1), dtype=np.float64)
        bridge.set_trajectory(
            TrajectoryDataset(
                times=times,
                positions=zeros.copy(),
                velocities=zeros.copy(),
                commands=zeros.copy(),
            )
        )
        entries = [_entry(SysIdParameterType.JOINT_STIFFNESS, dof_index=0)]
        with self.assertRaises(SysIdEnvironmentBridgeError) as raised:
            await bridge.run_rollout_async(
                torch.tensor([[1.0]]),
                entries,
                torch.as_tensor(zeros, dtype=torch.float32),
                num_steps,
            )
        self.assertIn("fixed-base", str(raised.exception))

    async def test_feedforward_computation_failure_is_fatal(self) -> None:
        import isaacsim.robot_setup.sysid.newton_diff_sysid_bridge as diff_module

        bridge, commands = _make_bridge(feedforward="gravity")
        bridge.set_trajectory(_hold_trajectory(0.2))
        original = diff_module._compute_feedforward

        def fail_feedforward(*_args, **_kwargs):
            raise RuntimeError("feedforward failed")

        diff_module._compute_feedforward = fail_feedforward
        try:
            theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)
            with self.assertRaises(SysIdEnvironmentBridgeError) as raised:
                await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        finally:
            diff_module._compute_feedforward = original
        self.assertIn("feedforward failed", str(raised.exception))

    async def test_feedforward_modes_are_applied(self) -> None:
        bridge_gravity, _ = _make_bridge(feedforward="gravity")
        self.assertEqual(bridge_gravity._feedforward_mode, "gravity")
        bridge_mode, _ = _make_bridge(feedforward="inverse_dynamics")
        self.assertEqual(bridge_mode._feedforward_mode, "inverse_dynamics")
        bridge_default, _ = _make_bridge()
        self.assertEqual(bridge_default._feedforward_mode, "none")

    async def test_capture_flag_off_keeps_uncaptured_path(self) -> None:
        """`cuda_graph_capture=False` must never populate the graph cache."""
        if not _cuda_available():
            self.skipTest("no CUDA device")
        bridge, commands = _make_bridge(device="cuda:0", cuda_graph_capture=False)
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)
        for _ in range(3):
            await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertEqual(len(bridge._ctx_cache[1].capture_cache), 0)

    async def test_cuda_graph_replay_is_bit_exact_and_activates_on_second_rollout(
        self,
    ) -> None:
        """Replayed rollouts must reproduce the uncaptured rollout exactly.

        The first rollout of a shape runs uncaptured (warm-up), the second is
        captured into CUDA graphs and replayed. Both run identical launch
        sequences on identical buffers, so positions, velocities, AND autograd
        gradients must match bit-for-bit — any deviation means the graph froze
        state it should not have.
        """
        if not _cuda_available():
            self.skipTest("no CUDA device")
        bridge, commands = _make_bridge(device="cuda:0")
        theta_np = self._theta0()
        results = []
        for _ in range(2):
            theta = torch.as_tensor(theta_np, dtype=torch.float32).reshape(1, -1).requires_grad_(True)
            rollout = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
            (rollout.positions.sum() + rollout.velocities.sum()).backward()
            results.append(
                (
                    rollout.positions.detach().cpu(),
                    rollout.velocities.detach().cpu(),
                    theta.grad.detach().cpu().clone(),
                )
            )
        ctx = bridge._ctx_cache[1]
        self.assertEqual(len(ctx.capture_cache), 1, "capture did not activate on the second rollout")
        self.assertFalse(ctx.capture_failed)
        for plain, replay in zip(results[0], results[1]):
            self.assertTrue(torch.equal(plain, replay))

    async def test_gradients_match_finite_differences_with_cuda_graph_capture(
        self,
    ) -> None:
        """FD-vs-AD parity through replayed forward AND backward graphs (<1%)."""
        if not _cuda_available():
            self.skipTest("no CUDA device")
        bridge, commands = _make_bridge(device="cuda:0")
        generator = torch.Generator().manual_seed(3)
        weights = (
            torch.rand(1, _NUM_STEPS, _NUM_DOF, generator=generator) - 0.5,
            0.1 * (torch.rand(1, _NUM_STEPS, _NUM_DOF, generator=generator) - 0.5),
        )
        theta0 = self._theta0()

        async def loss_of(theta_np, requires_grad=False):
            theta = torch.as_tensor(theta_np, dtype=torch.float32).reshape(1, -1)
            if requires_grad:
                theta.requires_grad_(True)
            result = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
            return theta, (result.positions.cpu() * weights[0]).sum() + (result.velocities.cpu() * weights[1]).sum()

        # Warm-up + capture happen inside the first two calls; the FD probes
        # below all run through graph replay.
        theta, loss = await loss_of(theta0, requires_grad=True)
        loss.backward()
        theta, loss = await loss_of(theta0, requires_grad=True)
        loss.backward()
        self.assertEqual(len(bridge._ctx_cache[1].capture_cache), 1)
        grad_autodiff = theta.grad.reshape(-1).double().numpy()
        steps = _gradcheck_steps(_GRADCHECK_ENTRIES)
        for index in range(len(theta0)):
            plus, minus = theta0.copy(), theta0.copy()
            plus[index] += steps[index]
            minus[index] -= steps[index]
            _, loss_plus = await loss_of(plus)
            _, loss_minus = await loss_of(minus)
            grad_fd = (float(loss_plus) - float(loss_minus)) / (2.0 * steps[index])
            with self.subTest(param=_GRADCHECK_ENTRIES[index].param_type.value):
                self.assertLess(
                    abs(grad_autodiff[index] - grad_fd) / max(abs(grad_fd), 1e-4),
                    1e-2,
                    msg=f"autodiff={grad_autodiff[index]:.6g} fd={grad_fd:.6g}",
                )

    async def test_cuda_graph_capture_with_inverse_dynamics_feedforward(self) -> None:
        """Capture composes with the feedforward path (ff array is part of the key)."""
        if not _cuda_available():
            self.skipTest("no CUDA device")
        bridge, commands = _make_bridge(device="cuda:0", feedforward="inverse_dynamics")
        times = np.arange(_NUM_STEPS, dtype=np.float64) * _DT
        q = 0.3 * np.sin(2.0 * np.pi * 1.2 * times)[:, None] * np.array([[1.0, -0.6]])
        qd = np.gradient(q, times, axis=0)
        bridge.set_trajectory(TrajectoryDataset(times=times, positions=q.copy(), velocities=qd, commands=q.copy()))
        commands_t = torch.as_tensor(q, dtype=torch.float32)
        theta_np = self._theta0()
        results = []
        for _ in range(2):
            theta = torch.as_tensor(theta_np, dtype=torch.float32).reshape(1, -1).requires_grad_(True)
            rollout = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands_t, _NUM_STEPS)
            rollout.positions.sum().backward()
            results.append((rollout.positions.detach().cpu(), theta.grad.detach().cpu().clone()))
        ctx = bridge._ctx_cache[1]
        self.assertEqual(len(ctx.capture_cache), 1)
        self.assertTrue(ctx.gravity_ff_active)
        for plain, replay in zip(results[0], results[1]):
            self.assertTrue(torch.equal(plain, replay))
            self.assertTrue(torch.all(torch.isfinite(replay)))

    async def test_same_trajectory_reset_reuses_feedforward_and_buffer_identity(
        self,
    ) -> None:
        """Re-setting the same trajectory object must not recompute the feedforward.

        The optimizer calls `set_trajectory` on every training-chunk switch,
        every iteration. The feedforward must stay cached and `gravity_ff_wp`
        must keep its identity — the CUDA graph capture key includes
        `id(gravity_ff_wp)`, so a replaced array forces a full recapture.
        """
        bridge, _ = _make_bridge(feedforward="gravity")
        trajectory = _hold_trajectory(0.4)
        commands = torch.as_tensor(trajectory.commands, dtype=torch.float32)
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)
        calls, restore = _count_feedforward_computes()
        try:
            bridge.set_trajectory(trajectory)
            first = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
            ctx = bridge._ctx_cache[1]
            self.assertTrue(ctx.gravity_ff_active)
            self.assertEqual(calls["count"], 1)
            buffer_id = id(ctx.gravity_ff_wp)

            bridge.set_trajectory(trajectory)
            second = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        finally:
            restore()
        self.assertEqual(calls["count"], 1, "feedforward was recomputed for an unchanged trajectory")
        self.assertEqual(
            id(ctx.gravity_ff_wp),
            buffer_id,
            "gravity_ff_wp identity (capture key) changed",
        )
        self.assertTrue(torch.allclose(first.positions, second.positions, atol=1e-6))
        self.assertTrue(torch.allclose(first.torques, second.torques, atol=1e-5))

    async def test_chunk_alternation_reuses_cached_feedforward_per_chunk(self) -> None:
        """Alternating training chunks must hit the per-trajectory feedforward cache.

        One computation per chunk (not per rollout), a stable `gravity_ff_wp`
        identity throughout (stable capture key), and bufferwise-correct
        contents: each chunk's repeat rollout reproduces its first one even
        though the other chunk's feedforward was loaded in between.
        """
        bridge, _ = _make_bridge(feedforward="gravity")
        chunk_a = _hold_trajectory(0.5)
        chunk_b = _hold_trajectory(-0.3)
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)
        calls, restore = _count_feedforward_computes()
        results: dict[str, list] = {"a": [], "b": []}
        buffer_ids = set()
        try:
            for label, chunk in (
                ("a", chunk_a),
                ("b", chunk_b),
                ("a", chunk_a),
                ("b", chunk_b),
            ):
                bridge.set_trajectory(chunk)
                commands = torch.as_tensor(chunk.commands, dtype=torch.float32)
                result = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
                results[label].append(result)
                buffer_ids.add(id(bridge._ctx_cache[1].gravity_ff_wp))
        finally:
            restore()
        self.assertEqual(calls["count"], 2, "expected one feedforward computation per chunk")
        self.assertEqual(
            len(buffer_ids),
            1,
            "gravity_ff_wp identity (capture key) changed during alternation",
        )
        for label in ("a", "b"):
            self.assertTrue(torch.allclose(results[label][0].positions, results[label][1].positions, atol=1e-6))
            self.assertTrue(torch.allclose(results[label][0].torques, results[label][1].torques, atol=1e-5))
        # The two chunks genuinely load different feedforwards/commands.
        self.assertFalse(torch.allclose(results["a"][0].positions, results["b"][0].positions, atol=1e-3))

    async def test_feedforward_recomputes_when_trajectory_mutated_in_place(
        self,
    ) -> None:
        """The identity-keyed cache must verify content: in-place edits invalidate it."""
        bridge, _ = _make_bridge(feedforward="gravity")
        chunk = _hold_trajectory(0.4)
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)
        calls, restore = _count_feedforward_computes()
        try:
            bridge.set_trajectory(chunk)
            commands = torch.as_tensor(chunk.commands, dtype=torch.float32)
            await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
            self.assertEqual(calls["count"], 1)

            chunk.positions += 0.2  # same object identity, different content
            bridge.set_trajectory(chunk)
            await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        finally:
            restore()
        self.assertEqual(
            calls["count"],
            2,
            "mutated trajectory content must recompute the feedforward",
        )

    async def test_cuda_graph_capture_is_reused_across_chunk_alternation(self) -> None:
        """Chunk alternation must replay one captured graph, not recapture per rollout.

        Same rollout shape + stable `gravity_ff_wp` identity mean one capture
        key: the graph is captured once (on the second rollout) and every later
        rollout replays it with in-place-updated feedforward/command contents.
        Each chunk's replayed rollout must reproduce its uncaptured one.
        """
        if not _cuda_available():
            self.skipTest("no CUDA device")
        import isaacsim.robot_setup.sysid.newton_diff_sysid_bridge as diff_module

        bridge, _ = _make_bridge(device="cuda:0", feedforward="gravity")
        chunk_a = _hold_trajectory(0.5)
        chunk_b = _hold_trajectory(-0.3)
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)

        captures = {"count": 0}
        original = diff_module._capture_rollout_graphs

        def counting(*args, **kwargs):
            captures["count"] += 1
            return original(*args, **kwargs)

        diff_module._capture_rollout_graphs = counting
        results: dict[str, list] = {"a": [], "b": []}
        try:
            for label, chunk in (
                ("a", chunk_a),
                ("b", chunk_b),
                ("a", chunk_a),
                ("b", chunk_b),
                ("a", chunk_a),
            ):
                bridge.set_trajectory(chunk)
                commands = torch.as_tensor(chunk.commands, dtype=torch.float32)
                result = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
                results[label].append(result.positions.detach().cpu())
        finally:
            diff_module._capture_rollout_graphs = original
        ctx = bridge._ctx_cache[1]
        self.assertFalse(ctx.capture_failed)
        self.assertEqual(captures["count"], 1, "chunk alternation triggered a graph recapture")
        self.assertEqual(
            len(ctx.capture_cache),
            1,
            "expected exactly one capture record for one shape",
        )
        self.assertTrue(ctx.gravity_ff_active)
        for label in ("a", "b"):
            for later in results[label][1:]:
                self.assertTrue(
                    torch.allclose(results[label][0], later, atol=1e-6),
                    msg=f"replayed chunk {label} rollout deviates from its uncaptured rollout",
                )
        self.assertFalse(torch.allclose(results["a"][0], results["b"][0], atol=1e-3))

    async def test_release_rollout_memory_frees_buffers_and_disables_capture(
        self,
    ) -> None:
        """Post-solve release must drop rollout buffers and latch capture off bridge-wide.

        The run session calls this between the solve and the forward-only
        validation/confidence stages; a context built AFTER the release (the
        confidence batch's multi-world model) must not attempt capture either,
        or it would re-pin the multi-GiB graph memory the release just freed.
        """
        bridge, commands = _make_bridge()
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)
        baseline = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        ctx = bridge._ctx_cache[1]
        self.assertGreater(len(ctx.states), 0)

        bridge.release_rollout_memory()
        self.assertFalse(bridge._capture_enabled)
        self.assertEqual(len(ctx.states), 0)
        self.assertEqual(len(ctx.controls), 0)
        self.assertEqual(len(ctx.solvers), 0)
        self.assertIsNone(ctx.sim_q)
        self.assertIsNone(ctx.sim_qd)
        self.assertIsNone(ctx.sim_tau)
        self.assertIsNone(ctx.final_joint_f)
        self.assertEqual(len(ctx.capture_cache), 0)
        self.assertFalse(bridge._capture_supported(ctx))

        # Buffers rebuild lazily; the released bridge must reproduce the rollout,
        # and a context built after the release must also refuse capture.
        again = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertTrue(torch.allclose(baseline.positions, again.positions, atol=1e-6))
        theta_batch = theta.repeat(2, 1)
        await bridge.run_rollout_async(theta_batch, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertFalse(bridge._capture_supported(bridge._ctx_cache[2]))

    async def test_buffer_allocation_failure_reclaims_memory_and_retries(self) -> None:
        """An OOM growing the per-substep state lists must fall back, not abort.

        The confidence stage's full-trajectory nominal rollout is a NEW, longer
        shape: its state-list growth is where a 16 GB card dies. One failure
        must trigger the reclaim-and-retry path (capture caches dropped,
        growth resumed), producing the same rollout an unpressured bridge does.
        """
        bridge, commands = _make_bridge()
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)
        await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, 10)
        ctx = bridge._ctx_cache[1]

        real_state = ctx.model.state
        failures = {"count": 0}

        def failing_state(*args, **kwargs):
            if failures["count"] == 0:
                failures["count"] += 1
                raise RuntimeError("Failed to allocate 4096 bytes on device 'cuda:0'")
            return real_state(*args, **kwargs)

        ctx.model.state = failing_state
        try:
            recovered = await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        finally:
            ctx.model.state = real_state
        self.assertEqual(failures["count"], 1)
        self.assertTrue(ctx.capture_failed, "memory pressure must latch capture off")
        self.assertTrue(torch.all(torch.isfinite(recovered.positions)))

        reference_bridge, _ = _make_bridge()
        reference = await reference_bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertTrue(torch.allclose(recovered.positions, reference.positions, atol=1e-6))

    async def test_buffer_allocation_failure_after_retry_raises_bridge_error(
        self,
    ) -> None:
        """A persistent allocation failure must surface as the bridge's error type.

        The run session's confidence stage catches exceptions and degrades to an
        empty payload with a logged reason instead of killing the whole run.
        """
        bridge, commands = _make_bridge()
        theta = torch.as_tensor(self._theta0(), dtype=torch.float32).reshape(1, -1)
        await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, 10)
        ctx = bridge._ctx_cache[1]

        def always_failing_state(*args, **kwargs):
            raise RuntimeError("Failed to allocate 4096 bytes on device 'cuda:0'")

        ctx.model.state = always_failing_state
        with self.assertRaises(SysIdEnvironmentBridgeError) as raised:
            await bridge.run_rollout_async(theta, _GRADCHECK_ENTRIES, commands, _NUM_STEPS)
        self.assertIn("after releasing", str(raised.exception))

    async def test_stiff_held_joint_gains_are_capped_for_stable_gradients(self) -> None:
        """A held joint with real-robot drive gains (ke ~ 1e6) must not destabilize the tape.

        Imported gains on held joints routinely violate the explicit-integrator
        stability bound at sub_dt. The forward rollout can look bounded (joint
        limits saturate the unstable mode) while the adjoint follows the
        unsaturated linearization and grows exponentially per taped substep.
        The bridge caps held-joint gains into the stability region, so both the
        rollout and its gradient stay finite; mapped joints are untouched.
        """
        import newton
        import warp as wp

        builder = newton.ModelBuilder()
        links = [
            builder.add_link(
                mass=1.0,
                com=(0.0, 0.0, -0.1),
                inertia=np.diag([0.01, 0.012, 0.014]).tolist(),
            )
            for _ in range(3)
        ]
        joints = []
        parent = -1
        for index, link in enumerate(links):
            stiff_held = index == 1  # middle joint is held (not in the trajectory)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=(0.0, 1.0, 0.0),
                    parent_xform=wp.transform((0.0, 0.0, 1.0 if index == 0 else -0.2), wp.quat_identity()),
                    child_xform=wp.transform((0.0, 0.0, 0.0), wp.quat_identity()),
                    target_ke=1.0e6 if stiff_held else 5.0,
                    target_kd=100.0 if stiff_held else 0.4,
                    label=f"/robot/joint{index}",
                )
            )
            parent = link
        builder.add_articulation(joints, label="chain")

        config = NewtonSimulationRunSpec(
            device="cpu",
            solver=NEWTON_SOLVER_FEATHERSTONE_DIFF,
            featherstone_substeps=2,
        )
        bridge = NewtonDifferentiableSysIdBridge(
            robot_prim_path="/robot",
            newton_config=config,
            joint_baselines=[
                SimpleNamespace(
                    stiffness=5.0,
                    damping=0.4,
                    friction=0.1,
                    dynamic_friction=0.1,
                    armature=0.1,
                )
            ]
            * 2,
            joint_paths=["/robot/joint0", "/robot/joint2"],
            num_joints=2,
            num_links=3,
            robot_builder=builder,
        )
        num_steps = 120
        times = np.arange(num_steps, dtype=np.float64) * _DT
        zeros = np.zeros((num_steps, 2), dtype=np.float64)
        commands = 0.3 * np.sin(2.0 * np.pi * 1.5 * times)[:, None] * np.array([[1.0, -1.0]])
        bridge.set_trajectory(
            TrajectoryDataset(
                times=times,
                positions=zeros.copy(),
                velocities=zeros.copy(),
                commands=commands,
            )
        )
        entries = [
            _entry(SysIdParameterType.JOINT_STIFFNESS, dof_index=0),
            _entry(SysIdParameterType.JOINT_STIFFNESS, dof_index=1),
        ]
        theta = torch.tensor([[1.0, 1.0]], requires_grad=True)
        result = await bridge.run_rollout_async(
            theta, entries, torch.as_tensor(commands, dtype=torch.float32), num_steps
        )
        self.assertTrue(torch.all(torch.isfinite(result.positions)))
        result.positions.sum().backward()
        self.assertTrue(torch.all(torch.isfinite(theta.grad)))
        # A stable adjoint keeps gradients at physical magnitudes; the unstable
        # held-joint mode used to inflate them by many orders of magnitude.
        self.assertLess(float(theta.grad.abs().max()), 1.0e6)
