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

# ruff: noqa: ANN001, ANN202, D101, D102

"""Unit tests for the gradient-descent (Adam) optimizer backend."""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np
import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.env_bridge import SysIdEnvironmentBridgeError
from isaacsim.robot_setup.sysid.optimizer_base import OptimizerConfig
from isaacsim.robot_setup.sysid.optimizer_config import (
    OptimizerBackend,
    OptimizerBackendConfig,
    recommend_backend,
)
from isaacsim.robot_setup.sysid.optimizer_factory import create_optimizer
from isaacsim.robot_setup.sysid.optimizers.gradient_descent import (
    GradientDescentOptimizer,
)
from isaacsim.robot_setup.sysid.parameter_types import (
    SysIdParameterEntry,
    SysIdParameterType,
)
from isaacsim.robot_setup.sysid.residual_config import ResidualWeightConfig
from isaacsim.robot_setup.sysid.rollout_result import SysIdRolloutResult
from isaacsim.robot_setup.sysid.runtime import set_yield_hook
from isaacsim.robot_setup.sysid.trajectory_csv import TrajectoryDataset


@contextmanager
def _patched_kit_app():
    async def yield_once() -> None:
        return None

    set_yield_hook(yield_once)
    try:
        yield
    finally:
        set_yield_hook(None)


def _trajectory(sample_count: int = 2) -> TrajectoryDataset:
    values = np.zeros((sample_count, 1), dtype=np.float64)
    return TrajectoryDataset(
        times=np.arange(sample_count, dtype=np.float64),
        positions=values.copy(),
        velocities=values.copy(),
        commands=values.copy(),
    )


def _config(
    *,
    theta_initial: float = 0.8,
    theta_min: float = 0.0,
    theta_max: float = 1.0,
    max_iterations: int = 25,
    backend_config: OptimizerBackendConfig | None = None,
    training_chunks: list[TrajectoryDataset] | None = None,
) -> OptimizerConfig:
    return OptimizerConfig(
        trajectory=_trajectory(),
        param_entries=[SysIdParameterEntry(SysIdParameterType.JOINT_FRICTION, 0)],
        theta_initial=torch.tensor([theta_initial], dtype=torch.float32),
        theta_min=torch.tensor([theta_min], dtype=torch.float32),
        theta_max=torch.tensor([theta_max], dtype=torch.float32),
        epsilon=1e-4,
        damping_initial=1e-2,
        max_iterations=max_iterations,
        max_rollout_steps=2,
        backend_config=backend_config or OptimizerBackendConfig(gd_learning_rate=0.1),
        training_chunks=training_chunks,
    )


class _DiffThetaBridge:
    """Fake bridge whose rollouts keep the autograd graph (positions = theta^2)."""

    device = torch.device("cpu")
    supports_arbitrary_candidate_batch = True

    def __init__(self) -> None:
        self.batches: list[torch.Tensor] = []
        self.cleanup_remove_clones: bool | None = None

    def set_trajectory(self, trajectory: TrajectoryDataset) -> None:
        self.trajectory = trajectory

    async def run_rollout_async(
        self,
        theta_per_env: torch.Tensor,
        param_entries: list[SysIdParameterEntry],
        commands: torch.Tensor,
        num_steps: int,
    ) -> SysIdRolloutResult:
        self.batches.append(theta_per_env.detach().cpu().clone())
        num_envs = int(theta_per_env.shape[0])
        positions = (theta_per_env[:, :1] ** 2).reshape(num_envs, 1, 1).expand(num_envs, num_steps, 1)
        velocities = torch.zeros(num_envs, num_steps, 1, dtype=theta_per_env.dtype)
        return SysIdRolloutResult(positions=positions, velocities=velocities)

    def cleanup(self, remove_clones: bool = True) -> None:
        self.cleanup_remove_clones = remove_clones


class _DetachedThetaBridge(_DiffThetaBridge):
    """Fake bridge that severs the graph, like the non-differentiable backends do."""

    async def run_rollout_async(self, theta_per_env, param_entries, commands, num_steps):
        result = await super().run_rollout_async(theta_per_env, param_entries, commands, num_steps)
        return SysIdRolloutResult(positions=result.positions.detach(), velocities=result.velocities.detach())


class _SerialDiffThetaBridge(_DiffThetaBridge):
    """Fake bridge that rejects a new forward until the prior output backpropagates."""

    requires_serial_backward = True

    def __init__(self) -> None:
        super().__init__()
        self.backward_pending = False
        self.forward_count = 0

    async def run_rollout_async(self, theta_per_env, param_entries, commands, num_steps):
        if self.backward_pending:
            raise RuntimeError("The previous rollout has not backpropagated.")
        result = await super().run_rollout_async(theta_per_env, param_entries, commands, num_steps)
        self.forward_count += 1
        if theta_per_env.requires_grad:
            self.backward_pending = True

            def release(gradient):
                self.backward_pending = False
                return gradient

            result.positions.register_hook(release)
        return result


class _StabilityBoundaryBridge(_DiffThetaBridge):
    """Fake bridge whose cost minimum lies past a non-finite stability threshold.

    Positions are theta - 1 (so the cost pulls theta toward 1), but any theta above
    ``threshold`` returns NaN — mimicking the explicit integrator crossing its
    damping stability limit while Adam walks the parameter upward.
    """

    threshold = 0.6

    async def run_rollout_async(self, theta_per_env, param_entries, commands, num_steps):
        self.batches.append(theta_per_env.detach().cpu().clone())
        num_envs = int(theta_per_env.shape[0])
        value = theta_per_env[:, :1] - 1.0
        value = torch.where(
            theta_per_env[:, :1] > self.threshold,
            torch.full_like(value, float("nan")),
            value,
        )
        positions = value.reshape(num_envs, 1, 1).expand(num_envs, num_steps, 1)
        velocities = torch.zeros(num_envs, num_steps, 1, dtype=theta_per_env.dtype)
        return SysIdRolloutResult(positions=positions, velocities=velocities)


class _AlwaysNonFiniteBridge(_DiffThetaBridge):
    """Fake bridge that keeps the autograd graph but never yields a finite cost."""

    async def run_rollout_async(self, theta_per_env, param_entries, commands, num_steps):
        result = await super().run_rollout_async(theta_per_env, param_entries, commands, num_steps)
        return SysIdRolloutResult(positions=result.positions * float("nan"), velocities=result.velocities)


class _NonFiniteGradientBridge(_DiffThetaBridge):
    """Fake bridge with a finite value and non-finite derivative at theta zero."""

    async def run_rollout_async(self, theta_per_env, param_entries, commands, num_steps):
        self.batches.append(theta_per_env.detach().cpu().clone())
        num_envs = int(theta_per_env.shape[0])
        value = torch.sqrt(theta_per_env[:, :1])
        positions = value.reshape(num_envs, 1, 1).expand(num_envs, num_steps, 1)
        velocities = torch.zeros(num_envs, num_steps, 1, dtype=theta_per_env.dtype)
        return SysIdRolloutResult(positions=positions, velocities=velocities)


class _RepairingParameterSpace:
    """Stand-in coupled constraint that resets rows after they cross a limit."""

    def __init__(self) -> None:
        self.calls = 0

    def reparameterize_inertia_theta(self, theta, param_entries):
        del param_entries
        self.calls += 1
        return torch.where(theta > 0.55, torch.full_like(theta, 0.5), theta)


class GradientDescentOptimizerTests(omni.kit.test.AsyncTestCase):
    async def test_converges_on_quadratic_cost(self) -> None:
        # Measured positions are zero, simulated positions are theta^2, so the
        # cost is minimized at theta = 0 within [0, 1].
        optimizer = GradientDescentOptimizer()
        optimizer.set_bridge(_DiffThetaBridge())
        config = _config(theta_initial=0.8, max_iterations=40)

        with _patched_kit_app():
            status = await optimizer.run_async(config)

        self.assertTrue(np.isfinite(status.cost))
        self.assertLess(abs(status.theta[0]), 0.15)
        self.assertGreaterEqual(status.theta[0], 0.0)

    async def test_cost_decreases_from_initial(self) -> None:
        optimizer = GradientDescentOptimizer()
        bridge = _DiffThetaBridge()
        optimizer.set_bridge(bridge)
        config = _config(theta_initial=0.9, max_iterations=10)

        with _patched_kit_app():
            status = await optimizer.run_async(config)

        # Cost at the initial theta of 0.9 is 0.5 * (0.9^2)^2 per sample scaling; the
        # exact value is irrelevant — it must strictly improve.
        initial_cost = 0.5 * (0.9**2) ** 2
        self.assertLess(status.cost, initial_cost)

    async def test_bounds_projection_holds(self) -> None:
        # Gradient pushes theta below the lower bound; projection must clamp it.
        optimizer = GradientDescentOptimizer()
        optimizer.set_bridge(_DiffThetaBridge())
        config = _config(theta_initial=0.55, theta_min=0.5, theta_max=1.0, max_iterations=30)

        with _patched_kit_app():
            status = await optimizer.run_async(config)

        self.assertGreaterEqual(status.theta[0], 0.5 - 1e-6)
        self.assertLessEqual(status.theta[0], 1.0 + 1e-6)
        self.assertAlmostEqual(status.theta[0], 0.5, places=2)

    async def test_backs_off_when_step_crosses_non_finite_boundary(self) -> None:
        # The cost minimum (theta = 1) lies beyond the stability threshold at 0.6;
        # rollouts above it return NaN. A previously-finite row must have its step
        # rejected with a learning-rate back-off — converging to the boundary —
        # instead of the whole solve raising.
        optimizer = GradientDescentOptimizer()
        bridge = _StabilityBoundaryBridge()
        optimizer.set_bridge(bridge)
        config = _config(theta_initial=0.2, max_iterations=40)

        with _patched_kit_app():
            status = await optimizer.run_async(config)

        self.assertTrue(np.isfinite(status.cost))
        self.assertLessEqual(status.theta[0], bridge.threshold + 1e-5)
        self.assertGreater(status.theta[0], 0.45)
        # The optimizer must actually have probed past the boundary at least once.
        max_theta_seen = max(float(batch[:, 0].max().item()) for batch in bridge.batches)
        self.assertGreater(max_theta_seen, bridge.threshold)

    async def test_raises_when_no_restart_ever_finite(self) -> None:
        optimizer = GradientDescentOptimizer()
        optimizer.set_bridge(_AlwaysNonFiniteBridge())
        config = _config(max_iterations=5)

        with _patched_kit_app():
            with self.assertRaises(SysIdEnvironmentBridgeError):
                await optimizer.run_async(config)

    async def test_rejects_nonfinite_gradient_and_backs_off_learning_rate(self) -> None:
        optimizer = GradientDescentOptimizer()
        bridge = _NonFiniteGradientBridge()
        optimizer.set_bridge(bridge)
        config = _config(theta_initial=0.0, max_iterations=2)

        with _patched_kit_app():
            status = await optimizer.run_async(config)

        self.assertTrue(np.isfinite(status.cost))
        self.assertAlmostEqual(status.theta[0], 0.0)
        self.assertLess(status.damping, 0.1)
        self.assertTrue(all(torch.equal(batch, torch.zeros_like(batch)) for batch in bridge.batches))

    async def test_repairs_coupled_parameters_after_each_adam_step(self) -> None:
        optimizer = GradientDescentOptimizer()
        bridge = _StabilityBoundaryBridge()
        parameter_space = _RepairingParameterSpace()
        optimizer.set_bridge(bridge)
        optimizer.set_parameter_space(parameter_space)
        config = _config(theta_initial=0.2, max_iterations=12)

        with _patched_kit_app():
            status = await optimizer.run_async(config)

        self.assertTrue(np.isfinite(status.cost))
        self.assertGreater(parameter_space.calls, 2)
        self.assertTrue(all(bool(torch.all(batch <= 0.55)) for batch in bridge.batches))

    async def test_raises_for_non_differentiable_bridge(self) -> None:
        optimizer = GradientDescentOptimizer()
        optimizer.set_bridge(_DetachedThetaBridge())
        config = _config(max_iterations=2)

        with _patched_kit_app():
            with self.assertRaises(SysIdEnvironmentBridgeError):
                await optimizer.run_async(config)

    async def test_multi_restart_batches_rows(self) -> None:
        optimizer = GradientDescentOptimizer()
        bridge = _DiffThetaBridge()
        optimizer.set_bridge(bridge)
        config = _config(
            max_iterations=3,
            backend_config=OptimizerBackendConfig(gd_learning_rate=0.1, gd_num_restarts=4),
        )

        with _patched_kit_app():
            status = await optimizer.run_async(config)

        self.assertTrue(np.isfinite(status.cost))
        self.assertEqual(int(bridge.batches[0].shape[0]), 4)

    async def test_restart_seed_reproduces_initial_population(self) -> None:
        bridges = [_DiffThetaBridge(), _DiffThetaBridge()]
        configs = [
            _config(
                max_iterations=1,
                backend_config=OptimizerBackendConfig(
                    gd_learning_rate=0.1,
                    gd_num_restarts=4,
                    gd_seed=73,
                ),
            )
            for _ in range(2)
        ]

        with _patched_kit_app():
            for bridge, config in zip(bridges, configs):
                optimizer = GradientDescentOptimizer()
                optimizer.set_bridge(bridge)
                await optimizer.run_async(config)

        torch.testing.assert_close(bridges[0].batches[0], bridges[1].batches[0])

    async def test_theta_fallback_runs_as_restart_row(self) -> None:
        # A presolve seed (row 0) that starts far from the optimum must not hide
        # the pre-presolve initials: the fallback runs as restart row 1 and the
        # per-iteration argmin keeps the better start.
        optimizer = GradientDescentOptimizer()
        bridge = _DiffThetaBridge()
        optimizer.set_bridge(bridge)
        config = _config(theta_initial=0.9, max_iterations=1)
        config.theta_fallback = torch.tensor([0.1], dtype=torch.float32)

        with _patched_kit_app():
            status = await optimizer.run_async(config)

        first_batch = bridge.batches[0]
        self.assertEqual(int(first_batch.shape[0]), 2)
        self.assertAlmostEqual(float(first_batch[0, 0].item()), 0.9, places=5)
        self.assertAlmostEqual(float(first_batch[1, 0].item()), 0.1, places=5)
        # Cost at the fallback (0.1^4) beats the seed (0.9^4), so it must win.
        self.assertLess(abs(status.theta[0] - 0.1), 0.11)

    async def test_theta_fallback_grows_single_restart_batch(self) -> None:
        optimizer = GradientDescentOptimizer()
        bridge = _DiffThetaBridge()
        optimizer.set_bridge(bridge)
        config = _config(
            max_iterations=1,
            backend_config=OptimizerBackendConfig(gd_learning_rate=0.1, gd_num_restarts=1),
        )
        config.theta_fallback = torch.tensor([0.2], dtype=torch.float32)

        with _patched_kit_app():
            await optimizer.run_async(config)

        self.assertEqual(int(bridge.batches[0].shape[0]), 2)

    async def test_segmented_training_keeps_gradient(self) -> None:
        trajectory = _trajectory()
        optimizer = GradientDescentOptimizer()
        bridge = _SerialDiffThetaBridge()
        optimizer.set_bridge(bridge)
        config = _config(max_iterations=2, training_chunks=[trajectory, trajectory])

        with _patched_kit_app():
            status = await optimizer.run_async(config)

        self.assertTrue(np.isfinite(status.cost))
        self.assertFalse(bridge.backward_pending)
        self.assertGreaterEqual(bridge.forward_count, 4)

    async def test_factory_and_auto_selection(self) -> None:
        self.assertIsInstance(
            create_optimizer(OptimizerBackend.GRADIENT_DESCENT),
            GradientDescentOptimizer,
        )

        entries = [SysIdParameterEntry(SysIdParameterType.JOINT_FRICTION, 0)]
        residuals = ResidualWeightConfig()
        self.assertEqual(
            recommend_backend(1, residuals, entries, differentiable_bridge=True),
            OptimizerBackend.GRADIENT_DESCENT,
        )
        self.assertNotEqual(
            recommend_backend(1, residuals, entries, differentiable_bridge=False),
            OptimizerBackend.GRADIENT_DESCENT,
        )
        delay_entries = [SysIdParameterEntry(SysIdParameterType.ACTUATOR_COMMAND_DELAY_SECONDS, -1)]
        self.assertEqual(
            recommend_backend(1, residuals, delay_entries, differentiable_bridge=True),
            OptimizerBackend.CMA_ES,
        )

    async def test_cleanup_bridge_forwards_clone_retention(self) -> None:
        """Forward clone-retention requests through synchronous bridge cleanup."""
        optimizer = GradientDescentOptimizer()
        bridge = _DiffThetaBridge()
        optimizer.set_bridge(bridge)

        optimizer.cleanup_bridge(remove_clones=False)

        self.assertFalse(bridge.cleanup_remove_clones)
