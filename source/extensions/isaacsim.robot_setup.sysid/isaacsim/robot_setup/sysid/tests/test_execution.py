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

# ruff: noqa: D101, D102

"""Tests for the public direct execution API."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import isaacsim.robot_setup.sysid.execution as execution
import isaacsim.robot_setup.sysid.run_session as run_session
import isaacsim.robot_setup.sysid.runtime as runtime
import omni.kit.test
from isaacsim.robot_setup.sysid.optimizer_base import OptimizationIterationStatus
from isaacsim.robot_setup.sysid.run_spec import SIMULATION_ENGINE_NEWTON, SysIdRunSpec
from isaacsim.robot_setup.sysid.usd_write_layer import PhysicsWriteTargets


class TestExecution(omni.kit.test.AsyncTestCase):
    async def test_prepared_run_uses_public_runner_and_callbacks(self) -> None:
        spec = SysIdRunSpec()
        prepared = SimpleNamespace(simulation_engine=spec.simulation.engine)
        expected = object()
        session = SimpleNamespace(run_async=AsyncMock(return_value=expected))
        on_iteration = Mock()
        on_progress = Mock()

        with patch.object(execution, "create_sysid_session", return_value=session) as create:
            result = await execution.run_sysid(
                spec,
                on_iteration=on_iteration,
                on_progress=on_progress,
                prepared_run=prepared,
                provenance_sidecar_path="result.json",
            )

        self.assertIs(result, expected)
        create.assert_called_once_with(
            spec,
            prepared,
            provenance_sidecar_path="result.json",
        )
        session.run_async.assert_awaited_once_with(on_iteration=on_iteration, on_progress=on_progress)

    async def test_newton_input_path_wins_over_current_kit_stage(self) -> None:
        spec = SysIdRunSpec()
        spec.simulation.engine = SIMULATION_ENGINE_NEWTON
        spec.stage.input_path = "robot.usd"
        opened_stage = object()

        with (
            patch.object(runtime, "kit_runtime_available", return_value=True),
            patch.object(runtime, "open_stage_async", AsyncMock(return_value=opened_stage)) as opener,
            patch.object(runtime, "get_current_stage", return_value=object()) as current,
        ):
            stage = await execution._resolve_stage(spec, SIMULATION_ENGINE_NEWTON)

        self.assertIs(stage, opened_stage)
        opener.assert_awaited_once_with("robot.usd")
        current.assert_not_called()

    async def test_prepared_run_engine_must_match_spec(self) -> None:
        spec = SysIdRunSpec()
        prepared = SimpleNamespace(simulation_engine=SIMULATION_ENGINE_NEWTON)

        with self.assertRaisesRegex(ValueError, "simulation engine does not match"):
            await execution.run_sysid(spec, prepared_run=prepared)

    async def test_yield_control_uses_asyncio_without_bootstrapped_kit(self) -> None:
        sleeper = AsyncMock()
        with (
            patch.object(runtime, "kit_runtime_available", return_value=False),
            patch.object(runtime.asyncio, "sleep", sleeper),
        ):
            await runtime.yield_control()

        sleeper.assert_awaited_once_with(0)

    def test_kit_runtime_is_unavailable_when_iapp_is_not_bootstrapped(self) -> None:
        """An installed Kit module is not proof that a live application exists."""
        with patch("omni.kit.app.get_app", side_effect=RuntimeError("no IApp")):
            self.assertFalse(runtime.kit_runtime_available())

    async def test_requested_usd_writeback_fails_closed_when_nothing_is_written(self) -> None:
        final = OptimizationIterationStatus(
            iteration=1,
            cost=1.0,
            damping=0.1,
            accepted=True,
            theta=[1.0],
        )
        optimizer = SimpleNamespace(
            reset_cancel=Mock(),
            run_async=AsyncMock(return_value=final),
            get_bridge=Mock(return_value=None),
            cleanup_bridge_async=AsyncMock(),
        )
        prepared = SimpleNamespace(
            config=SimpleNamespace(),
            optimizer=optimizer,
            stage=object(),
            parameter_space=object(),
            link_paths=["/World/robot/link"],
        )
        session = run_session.SysIdRunSession(
            prepared,
            apply_params_to_usd=True,
            write_usd_provenance=False,
            compute_parameter_confidence=False,
        )

        with (
            patch.object(run_session, "write_optimized_parameters_to_usd", return_value=False),
            self.assertRaisesRegex(RuntimeError, "no selected parameter was written"),
        ):
            await session.run_async()

        self.assertEqual(session.state, run_session.RunState.ERROR)
        optimizer.cleanup_bridge_async.assert_awaited_once_with(remove_clones=True)

    async def test_ui_writeback_passes_composition_targets_to_writer(self) -> None:
        final = OptimizationIterationStatus(
            iteration=1,
            cost=1.0,
            damping=0.1,
            accepted=True,
            theta=[1.0],
        )
        optimizer = SimpleNamespace(
            reset_cancel=Mock(),
            run_async=AsyncMock(return_value=final),
            get_bridge=Mock(return_value=None),
            cleanup_bridge_async=AsyncMock(),
        )
        prepared = SimpleNamespace(
            config=SimpleNamespace(trajectory=SimpleNamespace(times=[]), param_entries=[]),
            optimizer=optimizer,
            stage=object(),
            parameter_space=object(),
            link_paths=["/Robot/link"],
            validation_chunks=[],
            train_chunks=[],
            physics_backend="physx",
            physics_solver="mujoco",
        )
        neutral_target = object()
        solver_target = object()
        targets = PhysicsWriteTargets(neutral=neutral_target, solver=solver_target)
        session = run_session.SysIdRunSession(
            prepared,
            apply_params_to_usd=True,
            write_usd_provenance=False,
            robot_path="/Robot",
            compute_parameter_confidence=False,
        )

        with (
            patch.object(run_session, "resolve_multiphysics_write_targets", return_value=targets) as resolver,
            patch.object(run_session, "write_optimized_parameters_to_usd", return_value=True) as writer,
        ):
            result = await session.run_async()

        self.assertTrue(result.parameters_written)
        resolver.assert_called_once_with(prepared.stage, "/Robot", solver_layer_name="physx")
        self.assertIs(writer.call_args.kwargs["neutral_write_layer"], neutral_target)
        self.assertIs(writer.call_args.kwargs["solver_write_layer"], solver_target)
