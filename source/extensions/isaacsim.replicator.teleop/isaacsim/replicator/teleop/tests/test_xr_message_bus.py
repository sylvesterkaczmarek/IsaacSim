# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Regression tests for the XR Core message-bus boundary the teleop runtime exposes.

Two boundaries are covered:

* :func:`_extract_xr_teleop_command` - both shapes the CloudXR runtime is
  observed to emit (JSON-encoded string and inline dict) must yield the same
  command. The dict path was inadvertently dropped during a refactor and the
  command was silently swallowed; this test pins the contract.
* :func:`activate_pre_session_anchor` / :func:`restore_pre_session_anchor` -
  the snapshot/restore refcount must only advance when the underlying XR
  override actually committed, otherwise an XR module that comes online after
  activation can leave a permanently-overwritten anchor mode behind.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.test
from isaacsim.replicator.teleop import xr_anchor_manager as xam
from isaacsim.replicator.teleop.teleop_manager import TeleopManager, _extract_xr_teleop_command
from pxr import Gf, UsdGeom


class ExtractXrTeleopCommandTests(omni.kit.test.AsyncTestCase):
    """Define ExtractXrTeleopCommandTests behavior."""

    async def test_json_string_payload_extracts_command(self) -> None:
        """Run the json string payload extracts command test."""
        payload = {"message": '{"command":"start teleop"}'}
        self.assertEqual(_extract_xr_teleop_command(payload), "start teleop")

    async def test_dict_payload_extracts_command(self) -> None:
        """The inline-dict payload shape must yield the same command as the JSON-encoded string."""
        payload = {"message": {"command": "start teleop"}}
        self.assertEqual(_extract_xr_teleop_command(payload), "start teleop")

    async def test_none_payload_returns_empty_string(self) -> None:
        """Run the none payload returns empty string test."""
        self.assertEqual(_extract_xr_teleop_command(None), "")

    async def test_missing_message_returns_empty_string(self) -> None:
        """Run the missing message returns empty string test."""
        self.assertEqual(_extract_xr_teleop_command({"other": "thing"}), "")

    async def test_invalid_json_string_returns_empty_string(self) -> None:
        """Run the invalid json string returns empty string test."""
        self.assertEqual(_extract_xr_teleop_command({"message": "not valid json"}), "")

    async def test_dict_without_command_key_returns_empty_string(self) -> None:
        """Run the dict without command key returns empty string test."""
        self.assertEqual(_extract_xr_teleop_command({"message": {"other": "value"}}), "")

    async def test_non_string_non_dict_message_returns_empty_string(self) -> None:
        """Run the non string non dict message returns empty string test."""
        self.assertEqual(_extract_xr_teleop_command({"message": 42}), "")


class KitXrRuntimeStateTests(omni.kit.test.AsyncTestCase):
    """Verify late-loaded Kit XR discovery and runtime classification."""

    async def setUp(self) -> None:
        """Preserve the module-level lazy-import cache."""
        self._xr_globals = (xam.XRCore, xam.XRCoreEventType, xam.XRSettings)
        xam.XRCore = None
        xam.XRCoreEventType = None
        xam.XRSettings = None

    async def tearDown(self) -> None:
        """Restore the module-level lazy-import cache."""
        xam.XRCore, xam.XRCoreEventType, xam.XRSettings = self._xr_globals

    async def test_loader_retries_after_xr_becomes_available(self) -> None:
        """An early 2D import miss must not prevent XR discovery at Connect."""
        fake_core = object()
        fake_event_type = object()
        fake_settings = object()
        fake_module = SimpleNamespace(
            XRCore=fake_core,
            XRCoreEventType=fake_event_type,
            XRSettings=fake_settings,
        )

        with patch.object(
            xam.importlib,
            "import_module",
            side_effect=[ModuleNotFoundError("2D application"), fake_module],
        ):
            self.assertEqual(xam._load_xr_api(), (None, None, None))
            self.assertEqual(xam._load_xr_api(), (fake_core, fake_event_type, fake_settings))

    async def test_runtime_state_is_unavailable_without_kit_xr(self) -> None:
        """The normal 2D application is a supported state, not an XR failure."""
        with patch.object(xam, "_load_xr_api", return_value=(None, None, None)):
            self.assertEqual(xam.get_kit_xr_runtime_state(), xam.KitXrRuntimeState.UNAVAILABLE)

    async def test_runtime_state_is_active_when_stereo_display_is_enabled(self) -> None:
        """An active stereo display requires Teleop to publish the profile anchor."""
        core = SimpleNamespace(is_xr_display_enabled=lambda: True, is_xr_enabled=lambda: True)
        with (
            patch.object(xam, "_load_xr_api", return_value=(object(), object(), object())),
            patch.object(xam, "_get_xr_core", return_value=core),
        ):
            self.assertEqual(xam.get_kit_xr_runtime_state(), xam.KitXrRuntimeState.ACTIVE)

    async def test_runtime_state_is_inactive_when_xr_is_loaded_but_disabled(self) -> None:
        """A loaded but disabled XR extension does not make its profile a Connect prerequisite."""
        core = SimpleNamespace(is_xr_display_enabled=lambda: False, is_xr_enabled=lambda: False)
        with (
            patch.object(xam, "_load_xr_api", return_value=(object(), object(), object())),
            patch.object(xam, "_get_xr_core", return_value=core),
        ):
            self.assertEqual(xam.get_kit_xr_runtime_state(), xam.KitXrRuntimeState.INACTIVE)

    async def test_runtime_state_is_inactive_when_xr_core_singleton_is_unreachable(self) -> None:
        """A loaded but unqueryable XR Core degrades to INACTIVE instead of raising."""
        with (
            patch.object(xam, "_load_xr_api", return_value=(object(), object(), object())),
            patch.object(xam, "_get_xr_core", return_value=None),
        ):
            self.assertEqual(xam.get_kit_xr_runtime_state(), xam.KitXrRuntimeState.INACTIVE)


class XrAnchorSetupTests(omni.kit.test.AsyncTestCase):
    """Verify stage authoring without depending on an installed XR runtime."""

    async def setUp(self) -> None:
        """Create a fresh stage for anchor setup."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        stage_utils.define_prim("/World", "Xform")
        self._manager = xam.XrAnchorManager()

    async def tearDown(self) -> None:
        """Release anchor state and close the test stage."""
        self._manager.cleanup()
        stage_utils.close_stage()
        await app_utils.update_app_async()
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

    async def test_setup_creates_anchor_xform_without_xr(self) -> None:
        """Setup creates only a session-layer anchor when XR settings are unavailable."""
        with (
            patch.object(xam, "activate_pre_session_anchor", return_value=False),
            patch.object(self._manager, "_sync", return_value=True),
            patch.object(self._manager, "_update_sync_subscription"),
        ):
            self.assertTrue(self._manager.setup())

        stage = stage_utils.get_current_stage()
        prim = stage.GetPrimAtPath(xam.XrAnchorManager.DEFAULT_ANCHOR_PATH)
        self.assertTrue(prim.IsValid())
        self.assertEqual(prim.GetTypeName(), "Xform")
        self.assertIsNone(stage.GetRootLayer().GetPrimAtPath(xam.XrAnchorManager.DEFAULT_ANCHOR_PATH))

        self._manager.cleanup()
        self.assertFalse(stage.GetPrimAtPath(xam.XrAnchorManager.DEFAULT_ANCHOR_PATH).IsValid())

    async def test_setup_rejects_an_existing_user_anchor(self) -> None:
        """Never reset or reuse a user-owned prim at the generated runtime path."""
        stage = stage_utils.get_current_stage()
        prim = stage.DefinePrim(xam.XrAnchorManager.DEFAULT_ANCHOR_PATH, "Xform")
        UsdGeom.Xformable(prim).AddTransformOp()
        properties_before = set(prim.GetPropertyNames())

        self.assertFalse(self._manager.setup())

        self.assertEqual(set(prim.GetPropertyNames()), properties_before)

    async def test_custom_tracking_prim_is_read_without_rewriting_xform_ops(self) -> None:
        """Selecting a custom anchor must not normalize the user's xform stack."""
        stage = stage_utils.get_current_stage()
        prim = stage.DefinePrim("/World/CustomAnchor", "Xform")
        UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(1.0))
        properties_before = set(prim.GetPropertyNames())

        manager = TeleopManager.__new__(TeleopManager)
        manager._markers_manager = None
        manager._xr_anchor = None
        manager._locomotion_controller = None
        manager._tracking_space_world_pose_cache = type(
            "_PoseCache",
            (),
            {"set_prim_path": lambda _self, _path: None},
        )()
        manager._cached_tracking_space_frame = -1
        manager._tracking_space_retry_failed = False

        ok, _message = manager.set_tracking_space_prim_path("/World/CustomAnchor")

        self.assertTrue(ok)
        self.assertEqual(set(prim.GetPropertyNames()), properties_before)


class XrAnchorTransformTests(omni.kit.test.AsyncTestCase):
    """Verify the canonical transform shared by XR rendering and teleop poses."""

    @staticmethod
    def _matrix(
        position: tuple[float, float, float],
        yaw_degrees: float,
    ) -> Gf.Matrix4d:
        matrix = Gf.Matrix4d(1.0)
        matrix.SetRotateOnly(Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), yaw_degrees).GetQuat())
        matrix.SetTranslateOnly(Gf.Vec3d(*position))
        return matrix

    @staticmethod
    def _yaw(quaternion: Gf.Quatd) -> float:
        image = quaternion.GetImaginary()
        return math.atan2(
            2.0 * (quaternion.GetReal() * image[2] + image[0] * image[1]),
            1.0 - 2.0 * (image[1] * image[1] + image[2] * image[2]),
        )

    async def test_fixed_mode_preserves_initial_absolute_yaw_and_local_offset(self) -> None:
        """A rotated custom anchor remains a usable persistent alignment frame."""
        manager = xam.XrAnchorManager(
            anchor_pos=(1.0, 0.0, 0.0),
            tracking_space_prim_path="/World/CustomAnchor",
            rotation_mode=xam.AnchorRotationMode.FIXED,
            fixed_height=False,
        )
        first_matrix = self._matrix((10.0, 20.0, 0.0), 90.0)
        second_matrix = self._matrix((11.0, 21.0, 0.0), 180.0)

        with patch.object(manager, "_read_tracking_space_prim", return_value=(Gf.Vec3d(10.0, 20.0, 0.0), first_matrix)):
            first_position, first_orientation = manager._compute_anchor_pose()
        with patch.object(
            manager,
            "_read_tracking_space_prim",
            return_value=(Gf.Vec3d(11.0, 21.0, 0.0), second_matrix),
        ):
            second_position, second_orientation = manager._compute_anchor_pose()

        np.testing.assert_allclose(tuple(first_position), (10.0, 21.0, 0.0), atol=1e-6)
        np.testing.assert_allclose(tuple(second_position), (11.0, 22.0, 0.0), atol=1e-6)
        self.assertAlmostEqual(self._yaw(first_orientation), math.pi / 2.0, places=6)
        self.assertAlmostEqual(self._yaw(second_orientation), math.pi / 2.0, places=6)

    async def test_follow_mode_uses_absolute_custom_anchor_yaw(self) -> None:
        """Follow mode must not calibrate away the prim's authored initial rotation."""
        manager = xam.XrAnchorManager(
            tracking_space_prim_path="/World/CustomAnchor",
            rotation_mode=xam.AnchorRotationMode.FOLLOW_PRIM,
            fixed_height=False,
        )
        matrix = self._matrix((0.0, 0.0, 0.0), -90.0)
        with patch.object(manager, "_read_tracking_space_prim", return_value=(Gf.Vec3d(), matrix)):
            _position, orientation = manager._compute_anchor_pose()

        self.assertAlmostEqual(self._yaw(orientation), -math.pi / 2.0, places=6)

    async def test_smoothing_time_is_not_overridden_by_a_per_frame_alpha_floor(self) -> None:
        """A three-second smoothing setting remains slow at normal frame rates."""
        manager = xam.XrAnchorManager(
            rotation_mode=xam.AnchorRotationMode.FOLLOW_PRIM_SMOOTHED,
            smoothing_time=3.0,
            fixed_height=False,
        )
        identity = Gf.Quatd(1.0, Gf.Vec3d())
        manager._compute_rotation(self._matrix((0.0, 0.0, 0.0), 0.0), identity)
        smoothed = manager._compute_rotation(self._matrix((0.0, 0.0, 0.0), 90.0), identity)

        self.assertLess(abs(self._yaw(smoothed)), 0.02)

    async def test_missing_custom_anchor_holds_last_valid_pose(self) -> None:
        """A temporarily unavailable prim must not teleport the XR user to world origin."""
        manager = xam.XrAnchorManager(
            tracking_space_prim_path="/World/CustomAnchor",
            rotation_mode=xam.AnchorRotationMode.FOLLOW_PRIM,
            fixed_height=False,
        )
        matrix = self._matrix((3.0, 4.0, 5.0), 45.0)
        with patch.object(manager, "_read_tracking_space_prim", return_value=(Gf.Vec3d(3.0, 4.0, 5.0), matrix)):
            self.assertTrue(manager.sync())
        expected_position, expected_orientation = manager.get_world_pose()

        with patch.object(manager, "_read_tracking_space_prim", return_value=(None, None)):
            self.assertTrue(manager.sync())
        held_position, held_orientation = manager.get_world_pose()

        np.testing.assert_allclose(tuple(held_position), tuple(expected_position), atol=1e-6)
        self.assertAlmostEqual(self._yaw(held_orientation), self._yaw(expected_orientation), places=6)


class _FakeXrSettings:
    """Minimal stand-in for the Kit ``XRSettings`` singleton used by the snapshot helpers.

    Args:
        initial: Value for initial.
    """

    def __init__(self, initial: dict[str, str | float | None] | None = None) -> None:
        self.values: dict[str, str | float | None] = dict(initial or {})
        self.writes: list[tuple[str, str | float | None]] = []

    def get_setting(self, token: str) -> str | float | None:
        return self.values.get(token)

    def set_setting(self, token: str, value: str | float | None) -> None:
        self.values[token] = value
        self.writes.append((token, value))


class PreSessionAnchorRefcountTests(omni.kit.test.AsyncTestCase):
    """Pin the refcount lifecycle described in :func:`activate_pre_session_anchor`."""

    async def setUp(self) -> None:
        # Force a clean global state for every test; a leaked refcount from a
        # prior test would mask the very bugs we want to catch here.
        """Set up the test fixture."""
        xam._settings_snapshot = None
        xam._settings_snapshot_refs = 0

    async def tearDown(self) -> None:
        # Don't leave a stray refcount or snapshot behind for the rest of the suite.
        """Tear down the test fixture."""
        xam._settings_snapshot = None
        xam._settings_snapshot_refs = 0

    async def test_activate_returns_false_when_xr_unavailable(self) -> None:
        """No XR core - no activation, no refcount bump, restore is a safe no-op."""
        with patch.object(xam, "_xr_settings", lambda: None):
            self.assertFalse(xam.activate_pre_session_anchor())
        self.assertEqual(xam._settings_snapshot_refs, 0)
        self.assertIsNone(xam._settings_snapshot)
        xam.restore_pre_session_anchor()
        self.assertEqual(xam._settings_snapshot_refs, 0)

    async def test_late_xr_after_failed_activation_does_not_strip_anchor(self) -> None:
        """Verify late XR startup does not strip the anchor.

        If activation no-op'd because XR was offline, a late-arriving XR module must not
        get its baseline overwritten by a stray restore.
        """
        with patch.object(xam, "_xr_settings", lambda: None):
            xam.activate_pre_session_anchor()
            xam.activate_pre_session_anchor()
        late_xs = _FakeXrSettings({xam._XR_TOKEN_ANCHOR_MODE: "active camera"})
        with patch.object(xam, "_xr_settings", lambda: late_xs):
            xam.restore_pre_session_anchor()
            xam.restore_pre_session_anchor()
        self.assertEqual(
            late_xs.values[xam._XR_TOKEN_ANCHOR_MODE],
            "active camera",
            "Late-arriving XR settings must not be overwritten by a restore that has no snapshot.",
        )

    async def test_nested_activation_only_restores_on_final_release(self) -> None:
        """Verify nested activation restores only on final release.

        Two activations plus one restore must keep the override in place; the baseline
        returns only when the matching second restore lands.
        """
        xs = _FakeXrSettings({xam._XR_TOKEN_ANCHOR_MODE: "active camera"})
        with patch.object(xam, "_xr_settings", lambda: xs):
            self.assertTrue(xam.activate_pre_session_anchor())
            self.assertTrue(xam.activate_pre_session_anchor())
            self.assertEqual(xs.values[xam._XR_TOKEN_ANCHOR_MODE], "scene origin")

            xam.restore_pre_session_anchor()
            self.assertEqual(
                xs.values[xam._XR_TOKEN_ANCHOR_MODE],
                "scene origin",
                "Inner restore must keep the override active while the outer holder is alive.",
            )
            xam.restore_pre_session_anchor()
            self.assertEqual(
                xs.values[xam._XR_TOKEN_ANCHOR_MODE],
                "active camera",
                "Final restore must return the user's original anchor mode.",
            )

    async def test_restore_is_idempotent_when_refcount_is_zero(self) -> None:
        """Run the restore is idempotent when refcount is zero test."""
        xs = _FakeXrSettings({xam._XR_TOKEN_ANCHOR_MODE: "active camera"})
        with patch.object(xam, "_xr_settings", lambda: xs):
            xam.restore_pre_session_anchor()
            xam.restore_pre_session_anchor()
        self.assertEqual(xam._settings_snapshot_refs, 0)
        self.assertEqual(xs.values[xam._XR_TOKEN_ANCHOR_MODE], "active camera")

    async def test_restore_clears_settings_that_were_originally_absent(self) -> None:
        """A temporary render setting must not become persistent when its baseline was unset."""
        xs = _FakeXrSettings({xam._XR_TOKEN_ANCHOR_MODE: "active camera"})
        with patch.object(xam, "_xr_settings", lambda: xs):
            self.assertTrue(xam.activate_pre_session_anchor())
            xs.set_setting(xam._XR_TOKEN_NEAR_PLANE, 0.15)
            xam.restore_pre_session_anchor()

        self.assertIsNone(xs.values[xam._XR_TOKEN_NEAR_PLANE])

    async def test_disconnect_restores_near_plane_while_window_lease_remains(self) -> None:
        """Per-connect rendering settings must not leak until the Teleop window closes."""
        xs = _FakeXrSettings(
            {
                xam._XR_TOKEN_ANCHOR_MODE: "active camera",
                xam._XR_TOKEN_CUSTOM_ANCHOR: "/World/UserAnchor",
                xam._XR_TOKEN_NEAR_PLANE: 0.05,
            }
        )
        manager = xam.XrAnchorManager()
        with patch.object(xam, "_xr_settings", lambda: xs):
            self.assertTrue(xam.activate_pre_session_anchor())
            self.assertTrue(xam.activate_pre_session_anchor())
            manager._settings_active = True
            xs.set_setting(xam._XR_TOKEN_NEAR_PLANE, 0.15)
            xs.set_setting(xam._XR_TOKEN_ANCHOR_MODE, "custom anchor")
            xs.set_setting(xam._XR_TOKEN_CUSTOM_ANCHOR, xam.XrAnchorManager.DEFAULT_ANCHOR_PATH)

            manager.cleanup()

            self.assertEqual(xs.values[xam._XR_TOKEN_NEAR_PLANE], 0.05)
            self.assertEqual(xs.values[xam._XR_TOKEN_ANCHOR_MODE], "scene origin")
            self.assertEqual(xs.values[xam._XR_TOKEN_CUSTOM_ANCHOR], "")
            xam.restore_pre_session_anchor()
            self.assertEqual(xs.values[xam._XR_TOKEN_ANCHOR_MODE], "active camera")
            self.assertEqual(xs.values[xam._XR_TOKEN_CUSTOM_ANCHOR], "/World/UserAnchor")
