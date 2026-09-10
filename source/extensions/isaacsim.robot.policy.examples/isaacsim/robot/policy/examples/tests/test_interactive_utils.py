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

"""Tests for ``isaacsim.robot.policy.examples.interactive.utils``."""

from unittest import mock

import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.policy.examples.interactive.quadruped.quadruped_example import QuadrupedExample
from isaacsim.robot.policy.examples.interactive.utils import (
    restore_physics_simulation_state,
    snapshot_physics_simulation_state,
)


class TestInteractiveUtils(omni.kit.test.AsyncTestCase):
    """Unit tests for the snapshot/restore helpers used by the interactive policy examples.

    These helpers exist to prevent the PhysX direct-GPU-API flag and fabric extension from
    leaking across scene clears: ``cuda`` device implies fabric + readback suppression, which
    must be undone when the example tears down so subsequent USD edits don't hit
    ``PxArticulationJointReducedCoordinate::setDriveTarget`` errors.
    """

    async def setUp(self) -> None:
        """Create a fresh stage and capture the current physics state for restoration."""
        await stage_utils.create_new_stage_async()
        self._initial_device, self._initial_fabric = snapshot_physics_simulation_state()
        # Start from a known baseline: CPU + fabric off.
        SimulationManager.set_physics_sim_device("cpu")
        SimulationManager.enable_fabric(False)

    async def tearDown(self) -> None:
        """Restore the pre-test physics state so this test doesn't leak global flags."""
        restore_physics_simulation_state(self._initial_device, self._initial_fabric)
        await omni.kit.app.get_app().next_update_async()

    async def test_restore_with_none_is_noop(self) -> None:
        """``None`` snapshot fields leave the corresponding live state untouched."""
        SimulationManager.set_physics_sim_device("cpu")
        SimulationManager.enable_fabric(True)
        device_before = SimulationManager.get_physics_sim_device()
        fabric_before = SimulationManager.is_fabric_enabled()

        restore_physics_simulation_state(None, None)

        self.assertEqual(SimulationManager.get_physics_sim_device(), device_before)
        self.assertEqual(SimulationManager.is_fabric_enabled(), fabric_before)

    async def test_snapshot_then_restore_roundtrip(self) -> None:
        """``snapshot`` followed by ``restore`` reverses any intermediate mutation."""
        SimulationManager.set_physics_sim_device("cpu")
        SimulationManager.enable_fabric(False)
        prev_device, prev_fabric = snapshot_physics_simulation_state()

        # Simulate the example flipping the device to cuda (which enables fabric).
        SimulationManager.set_physics_sim_device("cuda")
        self.assertTrue(SimulationManager.is_fabric_enabled())

        restore_physics_simulation_state(prev_device, prev_fabric)

        self.assertEqual(SimulationManager.get_physics_sim_device(), prev_device)
        self.assertEqual(SimulationManager.is_fabric_enabled(), prev_fabric)


class TestInteractiveExamplePhysicsStateRoundtrip(omni.kit.test.AsyncTestCase):
    """Verify the shared locomotion example base restores global physics state on cleanup.

    Each example's ``setup_scene`` flips the physics sim device to ``cuda`` (which enables
    fabric and the PhysX direct-GPU API). If ``physics_cleanup`` / ``setup_post_clear`` does
    not undo that, the next session hits ``setDriveTarget`` errors when the user modifies USD.
    These tests exercise the snapshot → mutate → cleanup path on a fresh example instance
    without loading any robot assets.
    """

    async def setUp(self) -> None:
        """Establish a known baseline (CPU + fabric off) before each test."""
        await stage_utils.create_new_stage_async()
        self._initial_device, self._initial_fabric = snapshot_physics_simulation_state()
        SimulationManager.set_physics_sim_device("cpu")
        SimulationManager.enable_fabric(False)

    async def tearDown(self) -> None:
        """Put the global physics state back to whatever the previous test left it as."""
        restore_physics_simulation_state(self._initial_device, self._initial_fabric)
        await omni.kit.app.get_app().next_update_async()

    def _run_roundtrip(self, example: object) -> None:
        """Simulate ``setup_scene`` + ``physics_cleanup`` and assert state is restored.

        Args:
            example: Interactive example instance to exercise.
        """
        before_device, before_fabric = snapshot_physics_simulation_state()

        # Mirror the snapshot performed at the top of ``setup_scene``.
        example._prev_physics_sim_device, example._prev_fabric_enabled = snapshot_physics_simulation_state()

        # Mirror the state mutations ``setup_scene`` performs (skip asset loading).
        SimulationManager.set_backend(example._world_settings["backend"])
        SimulationManager.set_physics_sim_device(example._world_settings["device"])

        self.assertIn("cuda", SimulationManager.get_physics_sim_device())
        self.assertTrue(SimulationManager.is_fabric_enabled())

        example.physics_cleanup()

        after_device, after_fabric = snapshot_physics_simulation_state()
        self.assertEqual(after_device, before_device)
        self.assertEqual(after_fabric, before_fabric)

    async def test_locomotion_example_restores_physics_state(self) -> None:
        """The inherited locomotion cleanup restores the prior device and fabric flag."""
        self._run_roundtrip(QuadrupedExample())

    async def test_failed_runner_restart_stops_until_reset(self) -> None:
        """A failed first-step restart is not retried until the example resets."""

        class _Articulation:
            def is_physics_tensor_entity_valid(self) -> bool:
                return True

        class _Runner:
            articulation = _Articulation()

            def __init__(self) -> None:
                self.restart_calls = 0

            def restart_from_default_state(self, command: object) -> None:
                self.restart_calls += 1
                raise RuntimeError("restart failed")

        example = QuadrupedExample()
        runner = _Runner()
        example._runner = runner

        with mock.patch("carb.log_error") as log_error:
            example.on_physics_step(0.005, None)
            example.on_physics_step(0.005, None)

            self.assertFalse(example._physics_ready)
            self.assertTrue(example._policy_failed)
            self.assertEqual(runner.restart_calls, 1)
            log_error.assert_called_once()

            await example.setup_pre_reset()
            example.on_physics_step(0.005, None)

            self.assertTrue(example._policy_failed)
            self.assertEqual(runner.restart_calls, 2)
            self.assertEqual(log_error.call_count, 2)

    async def test_failed_control_tick_stops_until_reset(self) -> None:
        """A control tick that raises is not retried until the example resets.

        The runner does not advance its tick when a control tick raises, so without the latch a
        diverged policy would re-run inference on every physics step rather than every control
        tick.
        """

        class _Articulation:
            def is_physics_tensor_entity_valid(self) -> bool:
                return True

        class _Runner:
            articulation = _Articulation()

            def __init__(self) -> None:
                self.step_calls = 0

            def restart_from_default_state(self, command: object) -> None:
                self.step(0.005, command)

            def step(self, dt: float, command: object) -> None:
                self.step_calls += 1
                raise ValueError("policy produced non-finite values")

        example = QuadrupedExample()
        runner = _Runner()
        example._runner = runner

        with mock.patch("carb.log_error") as log_error:
            example.on_physics_step(0.005, None)  # startup control tick raises
            example.on_physics_step(0.005, None)  # latched, no second attempt

            self.assertTrue(example._policy_failed)
            self.assertEqual(runner.step_calls, 1)
            log_error.assert_called_once()

            # The reset clears _physics_ready, so the next tick retries startup once.
            await example.setup_post_reset()
            example.on_physics_step(0.005, None)

            self.assertEqual(runner.step_calls, 2)
            self.assertEqual(log_error.call_count, 2)

    async def test_timeline_stop_requires_clean_policy_restart(self) -> None:
        """Stopping clears both lifecycle latches; pausing does not call this hook."""
        example = QuadrupedExample()
        example._physics_ready = True
        example._policy_failed = True

        example._on_timeline_stop(None)

        self.assertFalse(example._physics_ready)
        self.assertFalse(example._policy_failed)
