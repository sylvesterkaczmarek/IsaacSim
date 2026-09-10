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

"""Tests for direct SimulationManager stepping in the Isaac Sim bridge."""

from __future__ import annotations

import asyncio
from unittest.mock import Mock, call, patch

import omni.kit.test
from isaacsim.robot_setup.sysid import runtime
from isaacsim.robot_setup.sysid.isaac_sim_sysid_bridge import IsaacSimSysIdBridge
from isaacsim.robot_setup.sysid.preflight import preflight_sysid_run_spec
from isaacsim.robot_setup.sysid.run_spec import SysIdRunSpec
from isaacsim.robot_setup.sysid.schema_validation import validate_sysid_run_spec_payload


class TestIsaacSimOfflineStepping(omni.kit.test.AsyncTestCase):
    """Validate offline stepping configuration and bridge behavior."""

    async def test_run_spec_defaults_to_offline_stepping_and_round_trips(self) -> None:
        """Verify offline stepping defaults and serialization."""
        default_spec = SysIdRunSpec()
        self.assertTrue(default_spec.simulation.offline_stepping)

        interactive_spec = SysIdRunSpec.from_dict({"simulation": {"offline_stepping": False}})
        self.assertFalse(interactive_spec.simulation.offline_stepping)
        self.assertFalse(interactive_spec.to_dict()["simulation"]["offline_stepping"])
        issues = validate_sysid_run_spec_payload(interactive_spec.to_dict())
        self.assertFalse([issue for issue in issues if issue.path == "$.simulation.offline_stepping"])

    async def test_preflight_requires_play_only_for_interactive_stepping(self) -> None:
        """Verify only interactive stepping requires a playing timeline."""
        spec = SysIdRunSpec()
        offline = preflight_sysid_run_spec(
            spec,
            timeline_playing=False,
            allow_empty_parameters=True,
        )
        self.assertFalse([issue for issue in offline.errors if issue.code == "timeline_not_playing"])

        spec.simulation.offline_stepping = False
        interactive = preflight_sysid_run_spec(
            spec,
            timeline_playing=False,
            allow_empty_parameters=True,
        )
        self.assertTrue([issue for issue in interactive.errors if issue.code == "timeline_not_playing"])

    async def test_offline_loop_steps_warmup_and_sample_intervals_without_kit_updates(self) -> None:
        """Verify offline rollouts step warmup and sample intervals directly."""
        bridge = object.__new__(IsaacSimSysIdBridge)
        bridge.fabric_clones = False
        bridge._rollout_warmup_remaining = 3
        bridge._rollout_active = True
        bridge._rollout_cancel_requested = False
        bridge._rollout_csv_index = 0
        bridge._rollout_num_steps = 3
        bridge._rollout_done = asyncio.Event()
        bridge._rollout_error = None
        bridge._articulation = object()

        events: list[str] = []
        bridge._reapply_trajectory_initial_state = lambda: events.append("reset")
        bridge._steps_per_csv_sample = lambda: 2

        def record() -> None:
            events.append(f"record_{bridge._rollout_csv_index}")
            bridge._rollout_csv_index += 1
            if bridge._rollout_csv_index >= bridge._rollout_num_steps:
                bridge._rollout_active = False
                bridge._rollout_done.set()

        bridge._record_rollout_sample = record

        timeline = Mock()
        timeline.is_playing.side_effect = [True, False]
        timeline.pause.side_effect = lambda: events.append("pause")
        with (
            patch(
                "isaacsim.robot_setup.sysid.isaac_sim_sysid_bridge.omni.timeline.get_timeline_interface",
                return_value=timeline,
            ),
            patch("isaacsim.robot_setup.sysid.isaac_sim_sysid_bridge.runtime.step_simulation") as step,
        ):
            step.side_effect = lambda **_kwargs: events.append("step")
            await bridge._run_rollout_offline_async(num_steps=3, num_envs=1)

        timeline.pause.assert_called_once_with()
        self.assertEqual(
            step.call_args_list,
            [
                call(steps=3, update_fabric=False),
                call(steps=2, update_fabric=False),
                call(steps=2, update_fabric=False),
            ],
        )
        self.assertEqual(
            events,
            ["pause", "step", "reset", "record_0", "step", "record_1", "step", "record_2"],
        )
        self.assertEqual(bridge._rollout_warmup_remaining, 0)
        self.assertIsNone(bridge._rollout_error)

    async def test_zero_step_runtime_request_is_a_noop(self) -> None:
        """Do not advance physics when an offline warmup requests zero steps."""
        with patch("isaacsim.core.simulation_manager.SimulationManager.step") as step:
            runtime.step_simulation(steps=0)

        step.assert_not_called()
