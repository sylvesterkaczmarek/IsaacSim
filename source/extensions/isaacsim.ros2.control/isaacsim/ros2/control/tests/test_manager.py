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

"""Unit tests for Ros2ControlManager.setup() input validation and backend
return-code mapping. The C++ backend, USD stage, and URDF synthesis are mocked;
this exercises only the Python glue and its error contract."""

from __future__ import annotations

import contextlib
from unittest import mock

import omni.kit.test
from isaacsim.ros2.control import ros2_control_manager as rm
from isaacsim.ros2.control.ros2_control_manager import Ros2ControlManager


def _stage(source_urdf=None):
    stage = mock.MagicMock()
    stage.GetRootLayer.return_value.customLayerData = (
        {"isaac:sourceUrdf": source_urdf} if source_urdf is not None else {}
    )
    return stage


class _Patches:
    """Patch backend / stage / urdf synthesis for a single setup() call."""

    def __init__(self, *, ready=True, stage=None, rc=0):
        self.ready, self.stage, self.rc = ready, stage, rc

    def __enter__(self):
        self.backend = mock.MagicMock()
        self.backend.is_ready.return_value = self.ready
        self.backend.setup_cm.return_value = self.rc
        self.build = mock.MagicMock(return_value="<robot/>")
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(mock.patch.object(rm, "_backend", self.backend))
        self._stack.enter_context(mock.patch.object(rm, "build_full_urdf", self.build))
        get_context = self._stack.enter_context(mock.patch("omni.usd.get_context"))
        get_context.return_value.get_stage.return_value = self.stage
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


class TestSetupContract(omni.kit.test.AsyncTestCase):
    async def test_raises_when_backend_not_ready(self):
        with _Patches(ready=False, stage=_stage()):
            with self.assertRaises(RuntimeError) as ctx:
                Ros2ControlManager.setup("/World/R", "/cfg.yaml")
            self.assertIn("not ready", str(ctx.exception))

    async def test_raises_when_no_stage(self):
        with _Patches(stage=None):
            with self.assertRaises(RuntimeError) as ctx:
                Ros2ControlManager.setup("/World/R", "/cfg.yaml")
            self.assertIn("no active USD stage", str(ctx.exception))

    async def test_success_returns_zero_and_forwards_args(self):
        with _Patches(stage=_stage(), rc=0) as p:
            rc = Ros2ControlManager.setup("/World/R", "/cfg.yaml", namespace="robot1")
        self.assertEqual(rc, 0)
        # URDF is synthesized for the requested prim, not some other path.
        p.build.assert_called_once()
        self.assertEqual(p.build.call_args.args[1], "/World/R")
        p.backend.setup_cm.assert_called_once()
        kwargs = p.backend.setup_cm.call_args.kwargs
        self.assertEqual(kwargs["articulation_path"], "/World/R")
        self.assertEqual(kwargs["controller_yaml_path"], "/cfg.yaml")
        self.assertEqual(kwargs["ns_name"], "robot1")
        self.assertEqual(kwargs["urdf_xml"], "<robot/>")
        self.assertIs(kwargs["publish_robot_description"], True)  # default
        self.assertIs(kwargs["use_sim_time"], True)  # default

    async def test_forwards_policy_flags(self):
        with _Patches(stage=_stage()) as p:
            Ros2ControlManager.setup("/World/R", "/cfg.yaml", publish_robot_description=False, use_sim_time=False)
        kwargs = p.backend.setup_cm.call_args.kwargs
        self.assertIs(kwargs["publish_robot_description"], False)
        self.assertIs(kwargs["use_sim_time"], False)

    async def test_use_sim_time_is_keyword_only(self):
        with self.assertRaises(TypeError):
            Ros2ControlManager.setup("/World/R", "/cfg.yaml", None, "", True, False)

    async def test_already_registered_maps_to_runtime_error(self):
        with _Patches(stage=_stage(), rc=-2):
            with self.assertRaises(RuntimeError) as ctx:
                Ros2ControlManager.setup("/World/R", "/cfg.yaml")
            self.assertIn("already registered", str(ctx.exception))

    async def test_init_failure_maps_to_runtime_error(self):
        with _Patches(stage=_stage(), rc=-3):
            with self.assertRaises(RuntimeError) as ctx:
                Ros2ControlManager.setup("/World/R", "/cfg.yaml")
            self.assertIn("init failed", str(ctx.exception))

    async def test_generic_failure_code_surfaced(self):
        # -1 (backend "not initialized") and any unmapped code hit the generic branch.
        for rc in (-1, 42):
            with _Patches(stage=_stage(), rc=rc):
                with self.assertRaises(RuntimeError) as ctx:
                    Ros2ControlManager.setup("/World/R", "/cfg.yaml")
                self.assertIn(str(rc), str(ctx.exception))

    async def test_overlay_falls_back_to_custom_layer_data(self):
        with _Patches(stage=_stage(source_urdf="/from/usd.urdf")) as p:
            Ros2ControlManager.setup("/World/R", "/cfg.yaml", urdf_path=None)
        # build_full_urdf should receive the URDF path read from customLayerData.
        self.assertEqual(p.build.call_args.kwargs["sensor_overlay_urdf_path"], "/from/usd.urdf")

    async def test_explicit_overlay_takes_precedence(self):
        with _Patches(stage=_stage(source_urdf="/from/usd.urdf")) as p:
            Ros2ControlManager.setup("/World/R", "/cfg.yaml", urdf_path="/explicit.urdf")
        self.assertEqual(p.build.call_args.kwargs["sensor_overlay_urdf_path"], "/explicit.urdf")
