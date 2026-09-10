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

"""Unit tests for the OgnROS2ControlManager compute lifecycle: configure-once,
re-configure on timeline Stop, and input validation. Ros2ControlManager.setup
is mocked; the OG database is a lightweight fake.

The node lives under python/nodes/ (scanned by the OG runtime), so it is loaded
by file path rather than imported.
"""

from __future__ import annotations

import importlib.util
import os
import types
from unittest import mock

import omni.graph.core as og
import omni.kit.test


def _load_node_module():
    here = os.path.dirname(__file__)
    path = os.path.normpath(os.path.join(here, "../nodes/OgnROS2ControlManager.py"))
    if not os.path.isfile(path):
        raise FileNotFoundError("OgnROS2ControlManager.py not found relative to tests/")
    spec = importlib.util.spec_from_file_location("_ogn_ros2_control_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_node = _load_node_module()


def _db(
    *,
    target=("/World/R",),
    cfg="/cfg.yaml",
    configured=False,
    needs_reconfigure=False,
    urdf_path="",
    namespace="",
    publish_robot_description=True,
    use_sim_time=True,
):
    state = types.SimpleNamespace(
        _needs_reconfigure=needs_reconfigure,
        _failed=False,
        ensure_timeline_subscription=mock.MagicMock(),
    )
    return types.SimpleNamespace(
        per_instance_state=state,
        state=types.SimpleNamespace(configured=configured),
        inputs=types.SimpleNamespace(
            targetPrim=target,
            execIn=og.ExecutionAttributeState.ENABLED,
            controllerConfig=cfg,
            urdfPath=urdf_path,
            namespace=namespace,
            publishRobotDescription=publish_robot_description,
            useSimTime=use_sim_time,
        ),
        outputs=types.SimpleNamespace(execOut=None),
        log_error=mock.MagicMock(),
    )


class TestOgnComputeLifecycle(omni.kit.test.AsyncTestCase):
    def _patch_setup(self):
        return mock.patch.object(_node.Ros2ControlManager, "setup")

    async def test_missing_target_prim_errors(self):
        db = _db(target=())
        with self._patch_setup() as setup:
            self.assertFalse(_node.OgnROS2ControlManager.compute(db))
        db.log_error.assert_called_once()
        setup.assert_not_called()

    async def test_multiple_target_prims_errors(self):
        db = _db(target=("/World/A", "/World/B"))
        with self._patch_setup() as setup:
            self.assertFalse(_node.OgnROS2ControlManager.compute(db))
        db.log_error.assert_called_once()
        setup.assert_not_called()

    async def test_missing_controller_config_errors(self):
        db = _db(cfg="")
        with self._patch_setup() as setup:
            self.assertFalse(_node.OgnROS2ControlManager.compute(db))
        db.log_error.assert_called_once()
        setup.assert_not_called()

    async def test_first_compute_configures(self):
        db = _db(namespace="robot1")
        with self._patch_setup() as setup:
            self.assertTrue(_node.OgnROS2ControlManager.compute(db))
        setup.assert_called_once()
        kwargs = setup.call_args.kwargs
        self.assertEqual(kwargs["prim_path"], "/World/R")
        self.assertEqual(kwargs["controller_config"], "/cfg.yaml")
        self.assertEqual(kwargs["namespace"], "robot1")
        self.assertEqual(kwargs["urdf_path"], None)  # empty string -> None
        db.per_instance_state.ensure_timeline_subscription.assert_called_once()
        self.assertTrue(db.state.configured)
        self.assertEqual(db.outputs.execOut, og.ExecutionAttributeState.ENABLED)

    async def test_noop_when_already_configured(self):
        db = _db(configured=True)
        with self._patch_setup() as setup:
            self.assertTrue(_node.OgnROS2ControlManager.compute(db))
        setup.assert_not_called()
        self.assertIsNone(db.outputs.execOut)  # execOut fires only at configure time

    async def test_reconfigures_after_stop(self):
        db = _db(configured=True, needs_reconfigure=True)
        with self._patch_setup() as setup:
            self.assertTrue(_node.OgnROS2ControlManager.compute(db))
        setup.assert_called_once()
        self.assertFalse(db.per_instance_state._needs_reconfigure)  # flag cleared
        self.assertTrue(db.state.configured)

    async def test_setup_exception_returns_false(self):
        db = _db()
        with self._patch_setup() as setup, mock.patch.object(_node.carb, "log_error") as log_error:
            setup.side_effect = RuntimeError("boom")
            self.assertFalse(_node.OgnROS2ControlManager.compute(db))
        log_error.assert_called_once()
        self.assertFalse(db.state.configured)
        self.assertIsNone(db.outputs.execOut)  # no exec on failure

    async def test_setup_failure_is_latched_not_retried_every_tick(self):
        # A hard setup failure must be latched: compute() should attempt setup()
        # once, not re-run it (re-exporting the URDF) on every tick until the next
        # Stop->Play. Re-attempting silently each tick spams errors and wastes work.
        db = _db()
        with self._patch_setup() as setup, mock.patch.object(_node.carb, "log_error"):
            setup.side_effect = RuntimeError("boom")
            for _ in range(3):
                _node.OgnROS2ControlManager.compute(db)
        self.assertEqual(setup.call_count, 1)

    async def test_forwards_policy_flags(self):
        db = _db(publish_robot_description=False, use_sim_time=False)
        with self._patch_setup() as setup:
            _node.OgnROS2ControlManager.compute(db)
        kwargs = setup.call_args.kwargs
        self.assertIs(kwargs["publish_robot_description"], False)
        self.assertIs(kwargs["use_sim_time"], False)

    async def test_nonempty_urdf_path_forwarded(self):
        db = _db(urdf_path="/overlay.urdf")
        with self._patch_setup() as setup:
            _node.OgnROS2ControlManager.compute(db)
        self.assertEqual(setup.call_args.kwargs["urdf_path"], "/overlay.urdf")


class TestInternalStateSubscription(omni.kit.test.AsyncTestCase):
    """The internal state subscribes to timeline Stop and flips _needs_reconfigure."""

    def _timeline_patches(self, stream):
        # returns (get_timeline_interface patch, TimelineEventType patch)
        timeline = mock.MagicMock()
        timeline.get_timeline_event_stream.return_value = stream
        return (
            mock.patch("omni.timeline.get_timeline_interface", return_value=timeline),
            mock.patch("omni.timeline.TimelineEventType", mock.MagicMock(STOP=2)),
        )

    async def test_stop_event_sets_needs_reconfigure(self):
        captured = {}
        stream = mock.MagicMock()
        stream.create_subscription_to_pop_by_type.side_effect = lambda t, cb: captured.update(type=t, cb=cb)
        state = _node.OgnROS2ControlManagerInternalState()
        p1, p2 = self._timeline_patches(stream)
        with p1, p2:
            state.ensure_timeline_subscription()
            self.assertEqual(captured["type"], 2)  # subscribed to the Stop event type
            self.assertFalse(state._needs_reconfigure)
            captured["cb"](object())  # deliver a Stop event
        self.assertTrue(state._needs_reconfigure)

    async def test_subscription_is_idempotent(self):
        stream = mock.MagicMock()
        state = _node.OgnROS2ControlManagerInternalState()
        p1, p2 = self._timeline_patches(stream)
        with p1 as gti, p2:
            state.ensure_timeline_subscription()
            state.ensure_timeline_subscription()
        self.assertEqual(gti.call_count, 1)  # second call short-circuits
