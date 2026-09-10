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

"""Focused coverage for Isaac Sim and Fabric runtime adapter contracts."""

from __future__ import annotations

import contextlib

import numpy as np
import omni.kit.test
import torch
from isaacsim.robot_setup.sysid.fabric_batch import (
    read_batched_joint_state,
    read_batched_link_pose,
)
from isaacsim.robot_setup.sysid.isaac_sim_sysid_bridge import IsaacSimSysIdBridge
from isaacsim.robot_setup.sysid.parameter_types import (
    SysIdParameterEntry,
    SysIdParameterType,
)

# ruff: noqa: ANN001, ANN003, ANN202, D102


class _RecordingArticulation:
    def __init__(self) -> None:
        self.com_call: tuple[tuple[object, ...], dict[str, object]] | None = None

    def set_dof_friction_properties(self, **_kwargs) -> None:
        pass

    def set_dof_gains(self, **_kwargs) -> None:
        pass

    def set_link_coms(self, *args: object, **kwargs: object) -> None:
        self.com_call = (args, kwargs)


class _FabricEffortFallback:
    def get_dof_positions(self, **_kwargs) -> np.ndarray:
        return np.asarray([[1.0]], dtype=np.float32)

    def get_dof_velocities(self, **_kwargs) -> np.ndarray:
        return np.asarray([[2.0]], dtype=np.float32)

    def get_dof_projected_joint_forces(self, **_kwargs) -> np.ndarray:
        raise NotImplementedError

    def get_dof_efforts(self, **_kwargs) -> np.ndarray:
        return np.asarray([[3.0]], dtype=np.float32)


class RuntimeBridgeAdapterTests(omni.kit.test.AsyncTestCase):
    """Lock runtime API conventions that otherwise fail silently."""

    async def test_com_apply_updates_positions_without_overwriting_orientation(self) -> None:
        """COM optimization must preserve each link's authored inertia-frame orientation."""
        articulation = _RecordingArticulation()
        bridge = object.__new__(IsaacSimSysIdBridge)
        bridge._articulation = articulation
        bridge.use_parallel_clones = False
        bridge._cloned_for_count = 0
        bridge._logical_env_count = 3
        bridge._joint_dof_indices = [0]
        bridge._num_links = 1
        bridge._baseline_dynamic_friction = np.asarray([[0.1]], dtype=np.float32)
        bridge._baseline_static_friction = np.asarray([[0.2]], dtype=np.float32)
        bridge._baseline_stiffness = np.asarray([[10.0]], dtype=np.float32)
        bridge._baseline_damping = np.asarray([[1.0]], dtype=np.float32)
        bridge._baseline_armature = None
        bridge._baseline_link_masses = None
        bridge._baseline_link_com = np.asarray([[0.1, 0.2, 0.3]], dtype=np.float32)
        bridge._baseline_link_com_orientation = np.asarray(
            [[0.9238795, 0.0, 0.0, 0.3826834]],
            dtype=np.float32,
        )
        bridge._baseline_link_inertia = None
        bridge._baseline_link_inertia_lc = {}
        bridge._baseline_dof_lower = None
        bridge._baseline_dof_upper = None
        bridge._explicit_pd_compat = None
        bridge._articulation_backend_context = contextlib.nullcontext
        bridge._flush_physics_changes = lambda: None

        bridge.apply_parameter_vector(
            torch.tensor([[0.05]], dtype=torch.float32),
            [
                SysIdParameterEntry(
                    param_type=SysIdParameterType.LINK_COM_OFFSET_X,
                    dof_index=-1,
                    link_index=0,
                )
            ],
        )

        self.assertIsNotNone(articulation.com_call)
        args, kwargs = articulation.com_call
        self.assertEqual(len(args), 2)
        np.testing.assert_allclose(args[0], [[[0.15, 0.2, 0.3]]], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            args[1],
            [[[0.9238795, 0.0, 0.0, 0.3826834]]],
            rtol=0.0,
            atol=1e-7,
        )
        self.assertEqual(kwargs, {"indices": [0]})
        self.assertEqual(bridge._physics_env_count(), 1)

    async def test_fabric_effort_uses_supported_fallback(self) -> None:
        """An unsupported projected-force API may fall back to ordinary efforts."""
        _position, _velocity, effort = read_batched_joint_state(
            _FabricEffortFallback(),
            num_envs=1,
            dof_indices=[0],
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(effort)
        self.assertEqual(float(effort[0, 0]), 3.0)

    async def test_fabric_pose_propagates_unexpected_runtime_failure(self) -> None:
        """Real Fabric failures must not masquerade as an unavailable residual channel."""

        class BrokenArticulation:
            def get_world_poses(self, **_kwargs):
                raise RuntimeError("device read failed")

        with self.assertRaisesRegex(RuntimeError, "device read failed"):
            read_batched_link_pose(
                BrokenArticulation(),
                num_envs=1,
                link_index=0,
                device=torch.device("cpu"),
            )
