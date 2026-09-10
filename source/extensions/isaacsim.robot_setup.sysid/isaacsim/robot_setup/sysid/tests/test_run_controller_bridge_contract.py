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

# ruff: noqa: D102

"""Regression coverage for the Isaac Sim bridge construction contract."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import isaacsim.robot_setup.sysid.isaac_sim_sysid_bridge as bridge_module
import omni.kit.test
from isaacsim.core.experimental.utils import stage as stage_utils
from isaacsim.robot_setup.sysid.isaac_sim_sysid_bridge import IsaacSimSysIdBridge
from isaacsim.robot_setup.sysid.run_controller import SysIdRunController
from isaacsim.robot_setup.sysid.run_spec import SysIdRunSpec
from pxr import UsdPhysics


class TestRunControllerBridgeContract(omni.kit.test.AsyncTestCase):
    """Keep controller construction aligned with the timeline-driven bridge."""

    async def test_bridge_reads_the_live_stage_through_runtime_facade(self) -> None:
        stage = await stage_utils.create_new_stage_async()
        root = stage.DefinePrim("/World/Robot", "Xform")
        UsdPhysics.ArticulationRootAPI.Apply(root)
        link = stage.DefinePrim("/World/Robot/link", "Cube")
        UsdPhysics.RigidBodyAPI.Apply(link)

        bridge = object.__new__(IsaacSimSysIdBridge)
        bridge._articulation_root = "/World/Robot"
        bridge._num_links = 1

        self.assertEqual(bridge._collect_link_prim_paths(), ["/World/Robot/link"])

    async def test_isaac_bridge_uses_current_constructor_arguments(self) -> None:
        spec = SysIdRunSpec()
        spec.simulation.robot_prim_path = "/World/Robot"
        spec.simulation.parallel_clones = False
        spec.simulation.offline_stepping = False
        spec.simulation.physics_backend = "newton"
        spec.simulation.actuator_runtime = "mixed"
        spec.simulation.explicit_actuator_policy = "promote_selected"
        expected = object()
        captured: dict[str, object] = {}

        def current_bridge_contract(
            robot_prim_path: str,
            source_env_path: str = "/World/envs/env_0",
            env_paths_root: str = "/World/envs/env",
            clone_spacing: float = 2.0,
            joint_dof_indices: list[int] | None = None,
            use_parallel_clones: bool = False,
            co_locate_clones: bool = True,
            fabric_clones: bool = False,
            offline_stepping: bool = True,
            end_effector_link_index: int = -1,
            selected_dof_paths: list[str] | None = None,
            physics_backend: str = "physx",
            actuator_runtime: str = "implicit_drive",
            explicit_actuator_policy: str = "authored_only",
        ) -> object:
            del source_env_path, env_paths_root, clone_spacing, joint_dof_indices
            del use_parallel_clones, co_locate_clones, fabric_clones, end_effector_link_index
            self.assertEqual(robot_prim_path, "/World/Robot")
            captured.update(
                offline_stepping=offline_stepping,
                selected_dof_paths=selected_dof_paths,
                physics_backend=physics_backend,
                actuator_runtime=actuator_runtime,
                explicit_actuator_policy=explicit_actuator_policy,
            )
            return expected

        controller = SysIdRunController(spec)
        with patch.object(bridge_module, "IsaacSimSysIdBridge", current_bridge_contract):
            bridge = controller._create_bridge(
                stage=object(),
                parameter_space=object(),
                link_paths=[],
                joint_paths=["/World/Robot/Joint"],
                residual_cfg=SimpleNamespace(end_effector_link_index=-1),
            )

        self.assertIs(bridge, expected)
        self.assertEqual(
            captured,
            {
                "offline_stepping": False,
                "selected_dof_paths": ["/World/Robot/Joint"],
                "physics_backend": "newton",
                "actuator_runtime": "mixed",
                "explicit_actuator_policy": "promote_selected",
            },
        )

    async def test_custom_bridge_factory_receives_runtime_selection_context(self) -> None:
        spec = SysIdRunSpec()
        spec.simulation.robot_prim_path = "/World/Robot"
        spec.simulation.parallel_clones = False
        captured: dict[str, object] = {}
        expected = object()

        def factory(**kwargs: object) -> object:
            captured.update(kwargs)
            return expected

        controller = SysIdRunController(spec, bridge_factory=factory)
        parameter_space = object()
        stage = object()
        bridge = controller._create_bridge(
            stage=stage,
            parameter_space=parameter_space,
            link_paths=["/World/Robot/link"],
            joint_paths=["/World/Robot/joint"],
            residual_cfg=SimpleNamespace(end_effector_link_index=-1),
        )

        self.assertIs(bridge, expected)
        self.assertIs(captured["stage"], stage)
        self.assertIs(captured["parameter_space"], parameter_space)
        self.assertEqual(captured["simulation_engine"], spec.simulation.engine)
        self.assertEqual(captured["explicit_actuator_policy"], spec.simulation.explicit_actuator_policy)
        self.assertEqual(captured["newton_config"], spec.simulation.newton)
        self.assertEqual(captured["offline_stepping"], spec.simulation.offline_stepping)
        self.assertEqual(captured["selected_dof_paths"], ["/World/Robot/joint"])
        self.assertIn("physics_backend", captured)
        self.assertIn("actuator_runtime", captured)
