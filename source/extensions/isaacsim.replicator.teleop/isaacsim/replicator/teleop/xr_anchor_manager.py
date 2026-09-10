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

"""Canonical XR and teleop anchor management.

The XR anchor determines **where in the scene the VR headset user sees
from** and supplies the same resolved transform to teleop head/controller
poses and frame markers. Kit's XR Core rendering subsystem places the
headset camera at the generated anchor prim's world transform.

The anchor does **not** have its own prim-path UI field. Instead it
derives its pose from the Session panel's **Tracking Space** prim
(e.g. ``/World/TeleopTrackingSpace``). The relationship is:

    Tracking Space prim  →  canonical anchor pose  →  XR + teleop + markers

Two modes of operation:

Static anchoring (no Tracking Space prim set)
    The anchor sits at a fixed world position defined by the UI "Anchor
    Offset" (X/Y/Z).  The headset always renders from that location.
    Useful for a stationary workspace.

Dynamic anchoring (Tracking Space prim set)
    The anchor follows the Tracking Space prim every frame. Its world position
    is the prim's world position plus ``anchor_offset`` expressed in the
    resolved anchor's local axes. This is needed when the reference object
    moves (e.g. a mobile robot base) and the VR camera should track it.

Rotation modes control how the anchor's yaw tracks the Tracking Space prim:

- **Fixed** - capture and hold the prim's initial absolute yaw.
- **Follow Prim** - follow the prim's current absolute yaw.
- **Follow (Smoothed)** - follow the current yaw with slerp smoothing.

Roll and pitch are stripped in every mode to prevent VR nausea. The resolved
pose is shared with teleop controller/head composition and marker display, so
an authored yaw correction remains registered with Kit XR.

**Fixed Height**, when enabled, locks the anchor's Z to the value it
had on the first sync frame - prevents the VR camera from bobbing when
the Tracking Space prim has vertical motion.
"""

from __future__ import annotations

import importlib
import math
import time
from enum import Enum
from typing import Any

import carb
import carb.events
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.usd
from isaacsim.core.experimental.prims import XformPrim
from pxr import Gf, Sdf, Usd, UsdGeom

from .coordinate_utils import OXR_TO_ISS_ROTATION

XRCore = None
XRCoreEventType = None
XRSettings = None


def _load_xr_api() -> tuple[object | None, object | None, object | None]:
    """Resolve Kit XR lazily so Teleop may load before ``omni.kit.xr.core``.

    The normal 2D application intentionally does not load Kit XR, while the XR
    experience may initialize it after this module is imported. A failed import
    is therefore not cached: a later Connect can discover XR after it becomes
    available.

    Returns:
        ``(XRCore, XRCoreEventType, XRSettings)`` or ``None`` entries when Kit
        XR is unavailable.
    """
    global XRCore, XRCoreEventType, XRSettings
    if XRCore is not None and XRCoreEventType is not None and XRSettings is not None:
        return XRCore, XRCoreEventType, XRSettings
    try:
        module = importlib.import_module("omni.kit.xr.core")
        XRCore = getattr(module, "XRCore")
        XRCoreEventType = getattr(module, "XRCoreEventType")
        XRSettings = getattr(module, "XRSettings")
    except (ImportError, ModuleNotFoundError, AttributeError, RuntimeError):
        # Expected in the documented 2D controller-tracking mode.
        return None, None, None
    return XRCore, XRCoreEventType, XRSettings


class KitXrRuntimeState(Enum):
    """Whether Kit XR stereo rendering is active in this process."""

    UNAVAILABLE = "unavailable"
    INACTIVE = "inactive"
    ACTIVE = "active"


def _get_xr_core() -> object | None:
    """Return the lazily resolved XR Core singleton, if available."""
    xr_core_type, _event_type, _settings_type = _load_xr_api()
    if xr_core_type is None:
        return None
    try:
        return xr_core_type.get_singleton()
    except (AttributeError, RuntimeError):
        return None


def get_kit_xr_runtime_state() -> KitXrRuntimeState:
    """Report whether Kit XR stereo rendering is active in this process.

    Only ``ACTIVE`` requires Teleop to publish the Kit XR profile anchor;
    ``UNAVAILABLE`` and ``INACTIVE`` are both valid for the documented 2D
    controller-tracking mode.

    Returns:
        Current Kit XR runtime state.
    """
    xr_core_type, _event_type, _settings_type = _load_xr_api()
    if xr_core_type is None:
        return KitXrRuntimeState.UNAVAILABLE
    xr_core = _get_xr_core()
    if xr_core is None:
        return KitXrRuntimeState.INACTIVE

    for method_name in ("is_xr_display_enabled", "is_xr_enabled"):
        method = getattr(xr_core, method_name, None)
        if method is None:
            continue
        try:
            if bool(method()):
                return KitXrRuntimeState.ACTIVE
        except (AttributeError, RuntimeError, TypeError):
            continue
    return KitXrRuntimeState.INACTIVE


class AnchorRotationMode(Enum):
    """How the XR anchor rotation tracks the tracking-space prim."""

    FIXED = "fixed"
    FOLLOW_PRIM = "follow_prim"
    FOLLOW_PRIM_SMOOTHED = "follow_prim_smoothed"


# Kit XR profile tokens. ``XRSettings`` translates these to the active VR
# profile's raw carb path - the only path the renderer reads.
_XR_TOKEN_ANCHOR_MODE: str = "profile/persistent/anchorMode"
_XR_TOKEN_CUSTOM_ANCHOR: str = "profile/stage/customAnchor"
_XR_TOKEN_NEAR_PLANE: str = "profile/persistent/render/nearPlane"

# Tokens captured on activation and restored on the final release. ``nearPlane``
# is included even though only :meth:`XrAnchorManager.setup` writes it - one
# shared snapshot must cover every token any path may overwrite.
_OVERRIDDEN_TOKENS: tuple[str, ...] = (
    _XR_TOKEN_ANCHOR_MODE,
    _XR_TOKEN_CUSTOM_ANCHOR,
    _XR_TOKEN_NEAR_PLANE,
)

# Refcounted so the window-level pre-session anchor and the per-Connect
# XrAnchorManager nest without stomping on each other's restore baseline.
_settings_snapshot: dict[str, Any] | None = None
_settings_snapshot_refs: int = 0


def _xr_settings() -> object | None:
    """Return the XRSettings singleton, or None if XR core is unavailable.

    Kit XR is resolved on every call until it becomes available. This supports
    both the 2D application, where XR is intentionally absent, and an XR
    experience that initializes after Teleop.

    Returns:
        The settings singleton when Kit XR is available, otherwise None.
    """
    _xr_core_type, _event_type, xr_settings_type = _load_xr_api()
    if xr_settings_type is None:
        return None
    try:
        return xr_settings_type.get_singleton()
    except (AttributeError, RuntimeError, TypeError):
        return None


def _snapshot_settings() -> bool:
    """Capture the overridden tokens once; reused by subsequent activations.

    Returns ``True`` when a snapshot is in place so the caller can decide
    whether to bump the activation refcount.

    Returns:
        The requested value.
    """
    global _settings_snapshot
    if _settings_snapshot is not None:
        return True
    xs = _xr_settings()
    if xs is None:
        return False
    try:
        _settings_snapshot = {token: xs.get_setting(token) for token in _OVERRIDDEN_TOKENS}
    except (AttributeError, RuntimeError, TypeError) as exc:
        carb.log_warn(f"[Teleop][Anchor] Failed to snapshot XR profile settings: {exc!r}")
        _settings_snapshot = None
        return False
    return True


def _restore_settings() -> None:
    """Replay the snapshot taken on first activation."""
    global _settings_snapshot
    if _settings_snapshot is None:
        return
    xs = _xr_settings()
    _settings_snapshot_local = _settings_snapshot
    _settings_snapshot = None
    if xs is None:
        return
    for token, original in _settings_snapshot_local.items():
        try:
            xs.set_setting(token, original)
        except (AttributeError, RuntimeError, TypeError) as exc:
            carb.log_warn(f"[Teleop][Anchor] Failed to restore XR setting {token!r}: {exc!r}")


def activate_pre_session_anchor() -> bool:
    """Override ``anchorMode = "scene origin"`` for the teleop UI lifetime.

    Replaces the VR experience's default ``"active camera"`` mode (which
    raycasts from the perspective camera and pins the headset to a static
    pose when no floor is present) so real motion moves the headset
    before Connect. Refcounted - safe to nest with the per-Connect anchor.

    Returns:
        The requested value.
    """
    global _settings_snapshot_refs
    if not _snapshot_settings():
        return False
    xs = _xr_settings()
    if xs is None:
        return False
    try:
        xs.set_setting(_XR_TOKEN_ANCHOR_MODE, "scene origin")
        if xs.get_setting(_XR_TOKEN_ANCHOR_MODE) != "scene origin":
            raise RuntimeError("XR profile rejected the scene-origin anchor mode")
    except (AttributeError, RuntimeError, TypeError) as exc:
        carb.log_warn(f"[Teleop][Anchor] Failed to acquire the XR profile anchor: {exc!r}")
        if _settings_snapshot_refs == 0:
            _restore_settings()
        return False
    _settings_snapshot_refs += 1
    return True


def restore_pre_session_anchor() -> None:
    """Release one activation; restore the original settings on the final release."""
    global _settings_snapshot_refs
    if _settings_snapshot_refs <= 0:
        return
    _settings_snapshot_refs -= 1
    if _settings_snapshot_refs == 0:
        _restore_settings()


class XrAnchorManager:
    """Manage the canonical anchor shared by Kit XR and teleop pose composition.

    The anchor prim is an Xform in the USD stage whose world transform tells
    Kit's XR Core where to place the headset camera.  This class:

    1. Creates the anchor prim at ``/World/XRAnchor``.
    2. Configures the active VR profile via :class:`XRSettings` to use it
       as ``custom anchor``.
    3. Subscribes to ``pre_sync_update`` to re-position the anchor every
       frame when a dynamic Tracking Space prim is configured.
    4. Exposes ``get_world_matrix()`` - a 4x4 that converts raw OpenXR
       poses (Y-up) into Isaac Sim world space (Z-up) at the current anchor.

    The tracking-space prim is set externally by :class:`TeleopManager` - it
    forwards the Session panel's Tracking Space prim path here via
    ``set_tracking_space_prim_path()``.

    Args:
        anchor_pos: Value for anchor pos.
        anchor_rot_xyzw: Value for anchor rot xyzw.
        tracking_space_prim_path: Value for tracking space prim path.
        rotation_mode: Value for rotation mode.
        smoothing_time: Value for smoothing time.
        fixed_height: Value for fixed height.
        near_plane: Value for near plane.
    """

    DEFAULT_ANCHOR_PATH = "/World/XRAnchor"

    def __init__(
        self,
        anchor_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
        anchor_rot_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        tracking_space_prim_path: str = "",
        rotation_mode: AnchorRotationMode = AnchorRotationMode.FIXED,
        smoothing_time: float = 1.0,
        fixed_height: bool = True,
        near_plane: float = 0.15,
    ) -> None:
        self._anchor_pos = np.array(anchor_pos, dtype=np.float64)
        self._anchor_rot_xyzw = np.array(anchor_rot_xyzw, dtype=np.float64)
        self._tracking_space_prim_path = tracking_space_prim_path
        self._rotation_mode = rotation_mode
        self._smoothing_time = max(0.01, smoothing_time)
        self._fixed_height = fixed_height
        self._near_plane = near_plane

        self._xr_core = _get_xr_core()
        self._pre_sync_sub: carb.events.ISubscription | None = None
        self._anchor_prim_path: str = ""
        self._anchor_layer: Sdf.Layer | None = None
        self._anchor_layer_id: str | None = None
        self._fabric_stage = None
        self._settings_active: bool = False
        self._reference_missing: bool = False

        # Dynamic-sync state
        self._initial_ref_quat: Gf.Quatd | None = None
        self._initial_height: float | None = None
        self._smoothed_quat: Gf.Quatd | None = None
        self._last_quat: Gf.Quatd | None = None
        self._rotation_enabled: bool = True
        self._last_sync_time: float = 0.0

        # Cached world transform (pos, quat_xyzw) written by _sync()
        self._cached_pos: np.ndarray | None = None
        self._cached_quat_xyzw: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def setup(self) -> bool:
        """Create the anchor prim, configures carb settings, and starts sync.

        Returns:
            The requested value.
        """
        if self._anchor_layer is not None:
            return True
        # XR may initialize after this manager is constructed. Retry at Connect
        # so the XR experience gets a pre-sync subscription when available.
        self._xr_core = _get_xr_core()
        if not stage_utils.is_stage_set() and omni.usd.get_context().get_stage() is None:
            print("[Teleop][Anchor] No USD stage.")
            return False
        stage = stage_utils.get_current_stage()

        self._anchor_prim_path = self.DEFAULT_ANCHOR_PATH

        existing = stage.GetPrimAtPath(self._anchor_prim_path)
        if existing and existing.IsValid():
            carb.log_warn(
                f"[Teleop][Anchor] Runtime anchor path '{self._anchor_prim_path}' is already occupied. "
                "Rename or remove that prim before connecting."
            )
            return False

        # Generated XR state belongs to a private session sublayer. It is
        # removed exactly on cleanup and never dirties the user's root layer.
        self._anchor_layer = Sdf.Layer.CreateAnonymous("anon_teleop_xr_anchor")
        stage.GetSessionLayer().subLayerPaths.append(self._anchor_layer.identifier)
        try:
            with Usd.EditContext(stage, self._anchor_layer):
                stage.DefinePrim(self._anchor_prim_path, "Xform")
                x, y, z, w = self._anchor_rot_xyzw
                xf = XformPrim(self._anchor_prim_path, reset_xform_op_properties=True)
                xf.set_world_poses(
                    positions=np.array([[*self._anchor_pos]], dtype=np.float32),
                    orientations=np.array([[w, x, y, z]], dtype=np.float32),
                )
        except Exception as exc:
            carb.log_error(f"[Teleop][Anchor] Failed to create runtime anchor: {exc}")
            self._drop_anchor_layer()
            return False
        self._anchor_layer_id = self._anchor_layer.identifier

        # Refcounted snapshot of the user's XR settings; the bool tracks whether
        # this call actually committed so cleanup() pairs with it correctly.
        try:
            self._settings_active = activate_pre_session_anchor()

            # Only write when the snapshot was taken - XR coming online late
            # between activation and this block would otherwise leave no
            # restorable baseline.
            if self._settings_active:
                xs = _xr_settings()
                if xs is not None:
                    xs.set_setting(_XR_TOKEN_NEAR_PLANE, float(self._near_plane))
                    xs.set_setting(_XR_TOKEN_ANCHOR_MODE, "custom anchor")
                    xs.set_setting(_XR_TOKEN_CUSTOM_ANCHOR, self._anchor_prim_path)
                    if (
                        xs.get_setting(_XR_TOKEN_ANCHOR_MODE) != "custom anchor"
                        or xs.get_setting(_XR_TOKEN_CUSTOM_ANCHOR) != self._anchor_prim_path
                    ):
                        raise RuntimeError("XR profile rejected the generated custom anchor")
        except Exception as exc:
            carb.log_error(f"[Teleop][Anchor] Failed to configure XR profile settings: {exc}")
            self.cleanup()
            return False

        print(
            f"[Teleop][Anchor] Created at '{self._anchor_prim_path}', "
            f"pos={tuple(self._anchor_pos)}, mode={self._rotation_mode.value}"
        )

        # Do an initial sync so cached values are populated.
        if not self._sync():
            self.cleanup()
            return False
        if self._tracking_space_prim_path and self._reference_missing:
            carb.log_warn(
                f"[Teleop][Anchor] Tracking-space prim '{self._tracking_space_prim_path}' "
                "could not provide a world transform."
            )
            self.cleanup()
            return False

        self._update_sync_subscription()

        return True

    def cleanup(self) -> None:
        """Release event subscriptions and the per-Connect settings activation.

        Removes the anonymous runtime-anchor layer. When the window-level
        activation is still outstanding, re-applies the ``scene origin``
        baseline (and clears the custom anchor token) so the headset returns
        to its pre-Connect mode; the final release replays the snapshot.
        """
        self._pre_sync_sub = None
        self._reset_sync_state()
        self._cached_pos = None
        self._cached_quat_xyzw = None
        self._fabric_stage = None
        if self._settings_active:
            self._settings_active = False
            xs = _xr_settings()
            still_held = _settings_snapshot_refs > 1
            if still_held and xs is not None:
                xs.set_setting(_XR_TOKEN_ANCHOR_MODE, "scene origin")
                xs.set_setting(_XR_TOKEN_CUSTOM_ANCHOR, "")
                if _settings_snapshot is not None:
                    original_near_plane = _settings_snapshot.get(_XR_TOKEN_NEAR_PLANE)
                    if original_near_plane is not None:
                        xs.set_setting(_XR_TOKEN_NEAR_PLANE, original_near_plane)
            restore_pre_session_anchor()
        self._drop_anchor_layer()
        print("[Teleop][Anchor] Cleaned up.")

    def reset(self) -> None:
        """Reset dynamic sync state (called on timeline reset)."""
        self._reset_sync_state()
        self._sync()

    @property
    def anchor_prim_path(self) -> str:
        """Return the USD path of the XR anchor prim.

        Returns:
            The requested value.
        """
        return self._anchor_prim_path

    @property
    def tracking_space_prim_path(self) -> str:
        """Return the currently configured tracking-space prim path.

        Returns:
            The requested value.
        """
        return self._tracking_space_prim_path

    @property
    def is_xr_profile_configured(self) -> bool:
        """Whether this manager successfully acquired and configured XR profile settings."""
        return self._settings_active

    # ------------------------------------------------------------------
    # Configuration setters (can be called live from UI)
    # ------------------------------------------------------------------

    def set_anchor_pos(self, pos: tuple[float, float, float]) -> None:
        """Update the anchor position offset and re-sync.

        Args:
            pos: Value for pos.
        """
        self._anchor_pos = np.array(pos, dtype=np.float64)
        self._initial_height = None
        self._sync()

    def set_anchor_rot(self, rot_xyzw: tuple[float, float, float, float]) -> None:
        """Update the anchor orientation offset and re-sync.

        Args:
            rot_xyzw: Value for rot xyzw.
        """
        self._anchor_rot_xyzw = np.array(rot_xyzw, dtype=np.float64)
        self._sync()

    def set_tracking_space_prim_path(self, path: str) -> None:
        """Changes the dynamic tracking-space prim and reapplies live sync.

        Args:
            path: Value for path.
        """
        self._tracking_space_prim_path = path
        self._reset_sync_state()
        self._sync()
        self._update_sync_subscription()

    def set_rotation_mode(self, mode: AnchorRotationMode) -> None:
        """Change the rotation-tracking mode and re-sync.

        Args:
            mode: Value for mode.
        """
        self._rotation_mode = mode
        self._reset_sync_state()
        self._sync()

    def set_smoothing_time(self, seconds: float) -> None:
        """Set the slerp smoothing time constant in seconds.

        Args:
            seconds: Value for seconds.
        """
        self._smoothing_time = max(0.01, seconds)

    def set_fixed_height(self, fixed: bool) -> None:
        """Toggle fixed-height mode and re-sync.

        Args:
            fixed: Value for fixed.
        """
        self._fixed_height = fixed
        self._initial_height = None
        self._sync()

    def toggle_rotation(self) -> None:
        """Toggle rotation following (e.g. from a VR controller button)."""
        self._rotation_enabled = not self._rotation_enabled
        print(f"[Teleop][Anchor] Rotation {'enabled' if self._rotation_enabled else 'frozen'}.")

    # ------------------------------------------------------------------
    # World matrix for pose transformation
    # ------------------------------------------------------------------

    def get_world_matrix(self) -> np.ndarray:
        """Return the 4x4 transform from OpenXR local space to Isaac Sim world.

        Composes: ``world_T_anchor @ oxr_to_usd`` so a raw OpenXR pose
        can be transformed with a single matrix multiply.

        Returns:
            The requested value.
        """
        if self._cached_pos is not None and self._cached_quat_xyzw is not None:
            return self._build_matrix(self._cached_pos, self._cached_quat_xyzw)
        return self._build_matrix(self._anchor_pos, self._anchor_rot_xyzw)

    def sync(self) -> bool:
        """Resolve and publish the current canonical anchor pose.

        Teleop pose composition calls this before consuming a frame so robot
        targets and Kit XR use the same transform.

        Returns:
            Whether a pose was resolved successfully.
        """
        return self._sync()

    def get_world_pose(self) -> tuple[Gf.Vec3d, Gf.Quatd]:
        """Return the last resolved anchor pose without coordinate conversion.

        Returns:
            Position and orientation in Isaac Sim world coordinates.
        """
        if self._cached_pos is not None and self._cached_quat_xyzw is not None:
            x, y, z, w = [float(value) for value in self._cached_quat_xyzw]
            return Gf.Vec3d(*self._cached_pos), Gf.Quatd(w, Gf.Vec3d(x, y, z))
        x, y, z, w = [float(value) for value in self._anchor_rot_xyzw]
        return Gf.Vec3d(*self._anchor_pos), Gf.Quatd(w, Gf.Vec3d(x, y, z))

    # ------------------------------------------------------------------
    # Internal: sync anchor prim to Tracking Space + write XR Core
    # ------------------------------------------------------------------

    def _sync(self) -> bool:
        """Read tracking-space pose, apply offset and rotation mode, update XR anchor, and cache result."""
        try:
            anchor_pos, anchor_quat = self._compute_anchor_pose()
        except Exception as exc:
            print(f"[Teleop][Anchor] _sync failed: {exc}")
            return False

        # Cache for get_world_matrix() / get_world_transform()
        img = anchor_quat.GetImaginary()
        self._cached_pos = np.array([anchor_pos[0], anchor_pos[1], anchor_pos[2]], dtype=np.float64)
        self._cached_quat_xyzw = np.array([img[0], img[1], img[2], anchor_quat.GetReal()], dtype=np.float64)

        # Write to XR Core so the rendering camera follows
        if self._xr_core is not None and self._anchor_layer_id is not None:
            try:
                mat = Gf.Matrix4d()
                mat.SetTranslateOnly(anchor_pos)
                mat.SetRotateOnly(anchor_quat)
                self._xr_core.set_world_transform_matrix(self._anchor_prim_path, mat, self._anchor_layer_id)
            except Exception as exc:
                carb.log_warn(f"[Teleop][Anchor] Failed to publish XR anchor transform: {exc}")
                return False
        return True

    def _compute_anchor_pose(self) -> tuple[Gf.Vec3d, Gf.Quatd]:
        """Compute the final anchor world pose from config + tracking-space prim.

        Returns:
            The requested value.
        """
        x, y, z, w = self._anchor_rot_xyzw
        cfg_quat = Gf.Quatd(float(w), Gf.Vec3d(float(x), float(y), float(z)))

        if not self._tracking_space_prim_path:
            return Gf.Vec3d(*self._anchor_pos), cfg_quat

        # Read tracking-space prim world pose from Fabric for physics accuracy
        ref_pos, ref_matrix = self._read_tracking_space_prim()
        if ref_pos is None:
            if not self._reference_missing:
                carb.log_warn(
                    f"[Teleop][Anchor] Tracking-space prim '{self._tracking_space_prim_path}' is unavailable; "
                    "holding the last valid anchor pose."
                )
            self._reference_missing = True
            if self._cached_pos is not None and self._cached_quat_xyzw is not None:
                x_c, y_c, z_c, w_c = [float(value) for value in self._cached_quat_xyzw]
                return Gf.Vec3d(*self._cached_pos), Gf.Quatd(w_c, Gf.Vec3d(x_c, y_c, z_c))
            return Gf.Vec3d(*self._anchor_pos), cfg_quat
        if self._reference_missing:
            carb.log_info(f"[Teleop][Anchor] Tracking-space prim '{self._tracking_space_prim_path}' recovered.")
        self._reference_missing = False

        # Resolve rotation before translation because Offset is expressed in
        # the final anchor's local axes rather than in world axes.
        anchor_quat = self._compute_rotation(ref_matrix, cfg_quat)
        offset = Gf.Rotation(anchor_quat).TransformDir(Gf.Vec3d(*self._anchor_pos))
        anchor_pos = ref_pos + offset

        # Fixed Height locks the final shared anchor so Kit XR, markers, and
        # robot targets remain registered vertically.
        if self._fixed_height:
            if self._initial_height is None:
                self._initial_height = float(anchor_pos[2])
            anchor_pos = Gf.Vec3d(anchor_pos[0], anchor_pos[1], self._initial_height)

        return anchor_pos, anchor_quat

    def _read_tracking_space_prim(self) -> tuple[Gf.Vec3d | None, Gf.Matrix4d | None]:
        """Read the tracking-space prim's world transform, preferring Fabric.

        Fabric reads give physics-accurate transforms for prims driven by
        the physics engine, avoiding the USD/Fabric desync that can occur
        when the prim hierarchy is updated by physics on the Fabric side
        but not yet flushed back to USD.

        Returns:
            The requested value.
        """
        # Attempt Fabric read via usdrt
        try:
            from usdrt import Rt
        except ImportError:
            Rt = None

        if Rt is not None:
            try:
                rt_stage = self._get_fabric_stage()
                if rt_stage is not None:
                    rt_prim = rt_stage.GetPrimAtPath(self._tracking_space_prim_path)
                    if rt_prim is not None:
                        rt_xf = Rt.Xformable(rt_prim)
                        if rt_xf is not None:
                            attr = rt_xf.GetFabricHierarchyWorldMatrixAttr()
                            if attr is not None:
                                rt_mat = attr.Get()
                                if rt_mat is not None:
                                    pos = rt_mat.ExtractTranslation()
                                    pxr_mat = Gf.Matrix4d(*[float(rt_mat[r][c]) for r in range(4) for c in range(4)])
                                    return Gf.Vec3d(*pos), pxr_mat
            except Exception as exc:
                print(f"[Teleop][Anchor] Fabric read failed, falling back to USD: {exc}")

        # Fallback: read the composed USD matrix directly. This supports
        # arbitrary user-authored xform stacks without normalizing them.
        try:
            stage = stage_utils.get_current_stage()
            prim = stage.GetPrimAtPath(self._tracking_space_prim_path)
            if prim and prim.IsValid() and prim.IsA(UsdGeom.Xformable):
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                return matrix.ExtractTranslation(), matrix
        except Exception:
            pass

        return None, None

    def _compute_rotation(self, ref_matrix: Gf.Matrix4d | None, cfg_quat: Gf.Quatd) -> Gf.Quatd:
        """Apply the rotation mode to produce the final anchor quaternion.

        Args:
            ref_matrix: Value for ref matrix.
            cfg_quat: Value for cfg quat.

        Returns:
            The requested value.
        """
        if ref_matrix is None:
            final = cfg_quat
        else:
            ref_quat = ref_matrix.ExtractRotationQuat()

            # Extract the absolute world-Z yaw. Roll and pitch are excluded
            # from the shared XR/teleop frame for operator comfort.
            w_d = ref_quat.GetReal()
            img = ref_quat.GetImaginary()
            yaw = math.atan2(
                2.0 * (w_d * img[2] + img[0] * img[1]),
                1.0 - 2.0 * (img[1] * img[1] + img[2] * img[2]),
            )
            cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
            yaw_quat = Gf.Quatd(cy, Gf.Vec3d(0.0, 0.0, sy))

            # Fixed preserves the custom anchor's authored initial yaw. Follow
            # modes use its current absolute yaw.
            if self._initial_ref_quat is None:
                self._initial_ref_quat = yaw_quat
            selected_yaw = self._initial_ref_quat if self._rotation_mode == AnchorRotationMode.FIXED else yaw_quat
            anchor_quat = selected_yaw * cfg_quat

            if self._rotation_mode == AnchorRotationMode.FOLLOW_PRIM_SMOOTHED:
                if self._smoothed_quat is None:
                    self._smoothed_quat = anchor_quat
                else:
                    now = time.monotonic()
                    dt = now - self._last_sync_time if self._last_sync_time > 0 else 1.0 / 60.0
                    dt = min(dt, 0.1)
                    self._last_sync_time = now
                    alpha = 1.0 - math.exp(-dt / self._smoothing_time)
                    alpha = min(1.0, max(0.0, alpha))
                    self._smoothed_quat = Gf.Slerp(alpha, self._smoothed_quat, anchor_quat)
                anchor_quat = self._smoothed_quat

            final = anchor_quat

        # Apply rotation freeze toggle
        if self._rotation_enabled:
            self._last_quat = final
        else:
            if self._last_quat is None:
                self._last_quat = final
            final = self._last_quat
            self._smoothed_quat = self._last_quat

        return final

    def _build_matrix(self, pos: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
        """Assembles ``world_T_anchor @ oxr_to_usd`` as a single 4x4 matrix.

        Args:
            pos: Value for pos.
            quat_xyzw: Value for quat xyzw.

        Returns:
            The requested value.
        """
        x, y, z, w = [float(v) for v in quat_xyzw]
        # Rotation matrix from quaternion (xyzw)
        r_anchor = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        r_combined = r_anchor @ OXR_TO_ISS_ROTATION
        mat = np.eye(4, dtype=np.float32)
        mat[:3, :3] = r_combined
        mat[:3, 3] = [float(pos[0]), float(pos[1]), float(pos[2])]
        return mat

    def _reset_sync_state(self) -> None:
        self._initial_ref_quat = None
        self._initial_height = None
        self._smoothed_quat = None
        self._last_quat = None
        self._rotation_enabled = True
        self._last_sync_time = 0.0
        self._reference_missing = False

    def _get_fabric_stage(self) -> Any:
        """Return a cached usdrt stage attached to the current USD stage.

        Returns:
            The requested value.
        """
        if self._fabric_stage is not None:
            return self._fabric_stage

        from isaacsim.core.experimental.utils.stage import get_current_stage

        self._fabric_stage = get_current_stage(backend="fabric")
        return self._fabric_stage

    def _drop_anchor_layer(self) -> None:
        """Remove the anonymous runtime-anchor layer from the current stage."""
        layer = self._anchor_layer
        self._anchor_layer = None
        self._anchor_layer_id = None
        if layer is None:
            return
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        session = stage.GetSessionLayer()
        if layer.identifier in session.subLayerPaths:
            session.subLayerPaths.remove(layer.identifier)

    def _update_sync_subscription(self) -> None:
        """Subscribe to per-frame sync only when following a live tracking-space prim."""
        _xr_core_type, xr_event_type, _settings_type = _load_xr_api()
        should_sync = bool(self._tracking_space_prim_path) and self._xr_core is not None and xr_event_type is not None
        if not should_sync:
            self._pre_sync_sub = None
            return

        if self._pre_sync_sub is None:
            self._pre_sync_sub = self._xr_core.get_message_bus().create_subscription_to_pop_by_type(
                xr_event_type.pre_sync_update,
                lambda _: self._sync(),
                name="teleop_xr_anchor_sync",
            )
