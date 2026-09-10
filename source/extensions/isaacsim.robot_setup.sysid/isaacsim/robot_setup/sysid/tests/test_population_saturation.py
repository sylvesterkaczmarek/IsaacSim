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

# ruff: noqa: ANN001, ANN202, D100, D101, D102

from __future__ import annotations

from types import SimpleNamespace

import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.newton_sysid_bridge import NewtonSysIdBridge
from isaacsim.robot_setup.sysid.population_saturation import (
    DeviceMemorySnapshot,
    PopulationSaturationConfig,
    _result_is_finite,
    build_diverse_candidate_population,
    run_population_saturation_benchmark,
)


class _FakeBridge:
    def __init__(self) -> None:
        self.released: list[int] = []

    async def run_rollout_async(self, theta, _entries, commands, num_steps):
        population = int(theta.shape[0])
        width = int(commands.shape[1])
        values = theta[:, :width].reshape(population, 1, width).expand(population, num_steps, width)
        return SimpleNamespace(
            positions=values.clone(),
            velocities=(2.0 * values).clone(),
            torques=(3.0 * values).clone(),
        )

    def release_population_context(self, population: int) -> None:
        self.released.append(int(population))


class _MemoryProbe:
    def __init__(self, snapshots: list[DeviceMemorySnapshot]) -> None:
        self._snapshots = list(snapshots)
        self._last = snapshots[-1]

    def __call__(self) -> DeviceMemorySnapshot:
        if self._snapshots:
            self._last = self._snapshots.pop(0)
        return self._last


class PopulationSaturationTests(omni.kit.test.AsyncTestCase):
    async def test_candidate_zero_is_invariant_across_population_and_generation(self) -> None:
        initial = torch.tensor([0.25, 0.75])
        lower = torch.zeros(2)
        upper = torch.ones(2)

        for population in (1, 4, 8):
            for generation in (0, 3, 17):
                candidates = build_diverse_candidate_population(
                    initial,
                    lower,
                    upper,
                    population,
                    generation=generation,
                )
                torch.testing.assert_close(candidates[0], initial)

    async def test_finite_result_allows_optional_torques(self) -> None:
        values = torch.zeros((1, 2, 1))
        result = SimpleNamespace(positions=values, velocities=values, torques=None)

        self.assertTrue(_result_is_finite(result))

    async def test_peak_memory_stops_sweep_even_after_context_release(self) -> None:
        bridge = _FakeBridge()
        memory_probe = _MemoryProbe(
            [
                DeviceMemorySnapshot(free_bytes=90, total_bytes=100),
                DeviceMemorySnapshot(free_bytes=80, total_bytes=100),
                DeviceMemorySnapshot(free_bytes=5, total_bytes=100),
                DeviceMemorySnapshot(free_bytes=80, total_bytes=100),
                DeviceMemorySnapshot(free_bytes=90, total_bytes=100),
            ]
        )

        report = await run_population_saturation_benchmark(
            bridge=bridge,
            param_entries=[object()],
            commands=torch.zeros((2, 1)),
            num_steps=2,
            duration_s_per_candidate=0.1,
            theta_initial=torch.tensor([1.0]),
            theta_min=torch.tensor([0.5]),
            theta_max=torch.tensor([1.5]),
            config=PopulationSaturationConfig(
                populations=(1, 2),
                steady_replays=1,
                capture_expected=False,
                max_device_memory_fraction=0.9,
                minimum_free_device_bytes=0,
            ),
            synchronize=lambda: None,
            memory_probe=memory_probe,
        )

        self.assertEqual([sample.population for sample in report.samples], [1])
        self.assertEqual(bridge.released, [1])
        self.assertEqual(report.samples[0].status, "memory_limit")
        self.assertEqual(report.samples[0].device_used_after_replays_bytes, 95)
        self.assertIn("during population 1", report.stop_reason)

    async def test_candidate_population_is_bounded_diverse_and_repeatable(self) -> None:
        initial = torch.tensor([1.0, 2.0, 3.0])
        lower = torch.tensor([0.5, 2.0, 1.0])
        upper = torch.tensor([1.5, 2.0, 5.0])
        first = build_diverse_candidate_population(initial, lower, upper, 8, generation=4)
        repeated = build_diverse_candidate_population(initial, lower, upper, 8, generation=4)
        changed = build_diverse_candidate_population(initial, lower, upper, 8, generation=5)

        self.assertTrue(torch.equal(first, repeated))
        self.assertTrue(torch.equal(first[0], initial))
        self.assertTrue(torch.all(first >= lower))
        self.assertTrue(torch.all(first <= upper))
        self.assertTrue(torch.all(first[:, 1] == 2.0))
        self.assertFalse(torch.equal(first[1:], changed[1:]))
        self.assertGreater(torch.unique(first[:, 0]).numel(), 2)

    async def test_sweep_reports_each_population_and_releases_contexts(self) -> None:
        bridge = _FakeBridge()
        initial = torch.tensor([1.0, 1.0])
        report = await run_population_saturation_benchmark(
            bridge=bridge,
            param_entries=[object(), object()],
            commands=torch.zeros((4, 2)),
            num_steps=4,
            duration_s_per_candidate=0.3,
            theta_initial=initial,
            theta_min=torch.tensor([0.5, 0.5]),
            theta_max=torch.tensor([1.5, 1.5]),
            config=PopulationSaturationConfig(
                populations=(1, 3, 5),
                steady_replays=3,
                capture_expected=False,
            ),
            synchronize=lambda: None,
            memory_probe=DeviceMemorySnapshot,
        )

        self.assertEqual([sample.population for sample in report.samples], [1, 3, 5])
        self.assertTrue(all(sample.status == "ok" for sample in report.samples))
        self.assertTrue(all(sample.finite for sample in report.samples))
        self.assertTrue(all(len(sample.replay_samples_s) == 3 for sample in report.samples))
        self.assertEqual(bridge.released, [1, 3, 5])
        self.assertAlmostEqual(report.samples[0].candidate0_position_rmse, 0.0)
        self.assertAlmostEqual(report.samples[-1].candidate0_torque_rmse, 0.0)
        self.assertGreater(report.samples[-1].aggregate_rt, 0.0)

    async def test_invalid_sweep_configuration_fails_before_rollout(self) -> None:
        with self.assertRaises(ValueError):
            PopulationSaturationConfig(populations=(0, -2)).normalized_populations()
        with self.assertRaises(ValueError):
            PopulationSaturationConfig(steady_replays=0).normalized_populations()
        with self.assertRaises(ValueError):
            PopulationSaturationConfig(saturation_efficiency_percent=101.0).normalized_populations()

    async def test_newton_population_context_release_clears_graph_references(self) -> None:
        synchronized = []
        context = SimpleNamespace(
            capture_cache={"graph": object()},
            capture_warm_keys={"key"},
            model=SimpleNamespace(device="cuda:0"),
            state0=object(),
            runtime_actuators=[object()],
            runtime_actuator_states=[object()],
            runtime_next_actuator_states=[object()],
        )
        bridge = NewtonSysIdBridge.__new__(NewtonSysIdBridge)
        bridge._modules = SimpleNamespace(
            warp=SimpleNamespace(synchronize_device=lambda device: synchronized.append(device)),
        )
        bridge._mujoco_cache = {13: context}

        bridge.release_population_context(13)

        self.assertNotIn(13, bridge._mujoco_cache)
        self.assertEqual(synchronized, ["cuda:0"])
        self.assertEqual(context.capture_cache, {})
        self.assertEqual(context.capture_warm_keys, set())
        self.assertEqual(context.runtime_actuators, [])
        self.assertIsNone(context.state0)
        self.assertIsNone(context.model)
