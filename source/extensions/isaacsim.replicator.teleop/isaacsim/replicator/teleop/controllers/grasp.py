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

"""Grasp controller for VR teleop.

Maps VR controller analog input (trigger) to drive joint targets using
configurable per-joint mappings.  Supports YAML-based grasp configurations
for multi-finger hands with sequenced joint activation.

When the gripper is part of a larger articulation (e.g. assembled onto a
robot arm via Robot Assembler), the controller automatically discovers the
articulation root and uses the tensor-backed Articulation API to set drive
targets.  This avoids conflicts with other controllers (IK, etc.) that use
the same Articulation backend for the same robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.usd
from isaacsim.core.experimental.prims import Articulation
from pxr import PhysxSchema, Sdf, Usd, UsdPhysics

from .base import find_owning_articulation_root

# ── Data types ───────────────────────────────────────────────────────

BUILTIN_GRASP_CONFIG_SCHEME = "builtin://"


@dataclass
class JointMapping:
    """Maps a portion of trigger input [0,1] to a joint target range.

    Attributes:
        name: Joint prim name to match in the gripper/hand hierarchy.
        input_range: Sub-range of [0,1] trigger input that activates this joint.
            Joints with higher start values begin moving later in the squeeze.
        target_range: Joint position [open, closed] mapped from input_range.
            Revolute-joint values are angles in degrees. Prismatic-joint values
            are distances in stage linear units.
        drive_stiffness: Optional USD drive stiffness override. Leave unset to
            preserve the value authored in the gripper asset.
        drive_damping: Optional USD drive damping override. Leave unset to
            preserve the value authored in the gripper asset.
        drive_max_force: Optional USD drive maximum-force override. Leave unset
            to preserve the value authored in the gripper asset.
    """

    name: str
    input_range: tuple[float, float] = (0.0, 1.0)
    target_range: tuple[float, float] = (0.0, 1.0)
    drive_stiffness: float | None = None
    drive_damping: float | None = None
    drive_max_force: float | None = None

    def compute_target(self, input_value: float) -> float:
        """Compute drive target for a given trigger input value.

        Below input_range[0] returns target_range[0].
        Above input_range[1] returns target_range[1].
        Within range, linearly interpolates.

        Args:
            input_value: Value for input value.

        Returns:
            Configured joint target. Revolute-joint values are in degrees;
            prismatic-joint values are in stage linear units.
        """
        input_value = max(0.0, min(1.0, input_value))
        lo, hi = self.input_range
        if hi <= lo:
            return self.target_range[1]
        if input_value <= lo:
            return self.target_range[0]
        if input_value >= hi:
            return self.target_range[1]
        t = (input_value - lo) / (hi - lo)
        return self.target_range[0] + t * (self.target_range[1] - self.target_range[0])


@dataclass
class GraspConfig:
    """Grasp configuration - loaded from YAML or constructed programmatically.

    Attributes:
        name: Human-readable name for display in UI.
        description: Optional description of the grasp behaviour.
        joints: Per-joint mappings from trigger input to drive targets.
    """

    name: str = ""
    description: str = ""
    joints: list[JointMapping] = field(default_factory=list)


@dataclass
class GraspValidationResult:
    """Result of grasp prim validation."""

    is_valid: bool = False
    total_joints: int = 0
    drive_joints: int = 0
    mimic_joints: int = 0
    controllable_joints: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    drive_joint_paths: list[str] = field(default_factory=list)


# ── Per-side runtime state ───────────────────────────────────────────


@dataclass
class _GraspState:
    prim_path: str | None = None
    config: GraspConfig | None = None
    # joint USD path -> (mapping, cached DriveAPI) -- USD fallback path
    active_joints: dict[str, tuple[JointMapping, UsdPhysics.DriveAPI]] = field(default_factory=dict)
    input_value: float = 0.0
    drive_mode: str = "trigger"
    retargeter_kind: str | None = None
    joint_aliases: dict[str, str] = field(default_factory=dict)
    retargeted_targets: dict[str, float] = field(default_factory=dict)
    # Articulation-backed path (used when gripper is part of a larger articulation)
    articulation: Articulation | None = None
    # joint USD path -> (mapping, DOF index in articulation)
    art_joint_map: dict[str, tuple[JointMapping, int]] = field(default_factory=dict)


# ── YAML loading ─────────────────────────────────────────────────────


def _get_builtin_grasp_configs_dir() -> Path | None:
    try:
        ext_path = app_utils.get_extension_path("isaacsim.replicator.teleop")
        if not ext_path:
            return None
        configs_dir = Path(ext_path) / "data" / "grasp_configs"
        if not configs_dir.is_dir():
            return None
        return configs_dir
    except Exception:
        return None


def get_builtin_grasp_config_uri(name: str) -> str:
    """Return the symbolic URI for a built-in grasp config name.

    Args:
        name: Value for name.

    Returns:
        The requested value.
    """
    return f"{BUILTIN_GRASP_CONFIG_SCHEME}{name.strip()}"


def normalize_grasp_config_path(path: str) -> str:
    """Normalize grasp config references to a portable built-in URI when possible.

    Args:
        path: Value for path.

    Returns:
        The requested value.
    """
    raw_path = path.strip()
    if not raw_path:
        return ""

    if raw_path.startswith(BUILTIN_GRASP_CONFIG_SCHEME):
        builtin_name = raw_path[len(BUILTIN_GRASP_CONFIG_SCHEME) :].strip()
        return get_builtin_grasp_config_uri(builtin_name) if builtin_name else ""

    candidate = Path(raw_path)
    if candidate.suffix.lower() in (".yaml", ".yml"):
        config_name = candidate.stem
        parent_name = candidate.parent.name.lower()
        if parent_name == "grasp_configs":
            builtin_dir = _get_builtin_grasp_configs_dir()
            if builtin_dir is not None:
                for suffix in (".yaml", ".yml"):
                    if (builtin_dir / f"{config_name}{suffix}").is_file():
                        return get_builtin_grasp_config_uri(config_name)

    return raw_path


def resolve_grasp_config_path(path: str) -> str:
    """Resolve a grasp config reference to a filesystem path when possible.

    Args:
        path: Value for path.

    Returns:
        The requested value.
    """
    normalized = normalize_grasp_config_path(path)
    if not normalized:
        return ""

    if not normalized.startswith(BUILTIN_GRASP_CONFIG_SCHEME):
        return normalized

    builtin_name = normalized[len(BUILTIN_GRASP_CONFIG_SCHEME) :].strip()
    if not builtin_name:
        return ""

    builtin_dir = _get_builtin_grasp_configs_dir()
    if builtin_dir is None:
        return ""

    for suffix in (".yaml", ".yml"):
        candidate = builtin_dir / f"{builtin_name}{suffix}"
        if candidate.is_file():
            return str(candidate)
    return ""


def load_grasp_config(path: str) -> tuple[GraspConfig | None, list[str]]:
    """Load a grasp configuration from a YAML file.

    Args:
        path: Filesystem path to the YAML file.

    Returns:
        The requested value.
    """
    import yaml

    errors: list[str] = []
    normalized_path = normalize_grasp_config_path(path)
    resolved_path = resolve_grasp_config_path(normalized_path)
    if normalized_path.startswith(BUILTIN_GRASP_CONFIG_SCHEME):
        if not resolved_path:
            return None, [f"Built-in grasp config not found: '{normalized_path}'"]
        p = Path(resolved_path)
    else:
        p = Path(resolved_path or normalized_path)

    if not p.exists():
        return None, [f"File not found: '{normalized_path or path}'"]
    if p.suffix.lower() not in (".yaml", ".yml"):
        return None, [f"Expected .yaml or .yml file: '{normalized_path or path}'"]

    try:
        with open(p) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return None, [f"Failed to parse YAML: {e}"]

    if not isinstance(data, dict):
        return None, ["YAML root must be a mapping"]

    joints_data = data.get("joints", [])
    if not isinstance(joints_data, list) or not joints_data:
        return None, ["'joints' must be a non-empty list"]

    config = GraspConfig(name=data.get("name", p.stem), description=data.get("description", ""))

    for i, jd in enumerate(joints_data):
        if not isinstance(jd, dict):
            errors.append(f"Entry {i}: must be a mapping")
            continue
        name = jd.get("name", "")
        if not name:
            errors.append(f"Entry {i}: missing 'name'")
            continue
        ir = jd.get("input_range", [0.0, 1.0])
        tr = jd.get("target_range", [0.0, 1.0])
        if not (isinstance(ir, (list, tuple)) and len(ir) == 2):
            errors.append(f"Joint '{name}': input_range must be [start, end]")
            continue
        if not (isinstance(tr, (list, tuple)) and len(tr) == 2):
            errors.append(f"Joint '{name}': target_range must be [start, end]")
            continue
        drive = jd.get("drive", {})
        if not isinstance(drive, dict):
            errors.append(f"Joint '{name}': drive must be a mapping")
            continue
        drive_values: dict[str, float | None] = {}
        invalid_drive = False
        for key in ("stiffness", "damping", "max_force"):
            value = drive.get(key)
            if value is None:
                drive_values[key] = None
                continue
            try:
                drive_values[key] = float(value)
            except (TypeError, ValueError):
                errors.append(f"Joint '{name}': drive.{key} must be a number")
                invalid_drive = True
        if invalid_drive:
            continue
        config.joints.append(
            JointMapping(
                name=name,
                input_range=(float(ir[0]), float(ir[1])),
                target_range=(float(tr[0]), float(tr[1])),
                drive_stiffness=drive_values["stiffness"],
                drive_damping=drive_values["damping"],
                drive_max_force=drive_values["max_force"],
            )
        )

    if errors:
        return None, errors

    return config, []


def get_builtin_grasp_configs() -> list[tuple[str, str]]:
    """Return (display_name, portable_config_path) pairs for built-in grasp configs.

    Scans the extension's ``data/grasp_configs/`` directory.

    Returns:
        The requested value.
    """
    configs_dir = _get_builtin_grasp_configs_dir()
    if configs_dir is None:
        return []

    result: list[tuple[str, str]] = []
    for suffix in ("*.yaml", "*.yml"):
        for p in sorted(configs_dir.glob(suffix)):
            display = p.stem
            result.append((display, get_builtin_grasp_config_uri(display)))
    return result


# ── Controller ───────────────────────────────────────────────────────


class GraspController:
    """Controls grasping via VR trigger input.

    Maps VR controller analog input (trigger) to drive joint targets
    using YAML-based per-joint mappings with custom input sub-ranges
    for sequenced multi-finger grasps.  Each side (left/right) has its
    own prim path and config, supporting different end effectors.
    """

    def __init__(self) -> None:
        self._sides: dict[str, _GraspState] = {
            "left": _GraspState(),
            "right": _GraspState(),
        }
        self._enabled = False
        self._tracking_enabled: dict[str, bool] = {"left": False, "right": False}

    def _side(self, side: str) -> _GraspState:
        return self._sides[side.lower()]

    # ── Validation ───────────────────────────────────────────────────

    def validate_prim(self, prim_path: str) -> GraspValidationResult:
        """Validate whether a prim has controllable drive joints.

        A valid prim must have at least one joint with DriveAPI that is not
        a mimic joint.

        Args:
            prim_path: USD path to the gripper/hand root prim.

        Returns:
            The requested value.
        """
        result = GraspValidationResult()

        if not stage_utils.is_stage_set() and omni.usd.get_context().get_stage() is None:
            result.errors.append("Stage not available")
            return result
        stage = stage_utils.get_current_stage()

        if not prim_path or not Sdf.Path.IsValidPathString(prim_path):
            result.errors.append(f"Invalid path: '{prim_path}'")
            return result

        prim = stage.GetPrimAtPath(prim_path)
        if not prim or not prim.IsValid():
            result.errors.append(f"Prim not found: '{prim_path}'")
            return result

        # Traverse instance proxies so joints inside instanced gripper hierarchies are included.
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
            if p.IsA(UsdPhysics.Joint):
                result.total_joints += 1
                path = str(p.GetPath())
                is_mimic = prim_utils.has_api(p, PhysxSchema.PhysxMimicJointAPI)
                has_drive = prim_utils.has_api(p, UsdPhysics.DriveAPI, instance_name="angular") or prim_utils.has_api(
                    p, UsdPhysics.DriveAPI, instance_name="linear"
                )
                if is_mimic:
                    result.mimic_joints += 1
                if has_drive:
                    result.drive_joints += 1
                if has_drive and not is_mimic:
                    result.controllable_joints += 1
                    result.drive_joint_paths.append(path)

        if result.controllable_joints == 0:
            result.errors.append("No controllable joints (need DriveAPI without MimicJointAPI)")
        else:
            result.is_valid = True

        if result.total_joints == 0:
            result.warnings.append("No joints found in hierarchy")

        return result

    # ── Configure ─────────────────────────────────────────────────────

    def configure(
        self,
        prim_path: str,
        side: str,
        config: GraspConfig,
        *,
        drive_mode: str = "trigger",
        retargeter_kind: str | None = None,
        joint_aliases: dict[str, str] | None = None,
    ) -> bool:
        """Configure grasp control for a side.

        Matches YAML joint names in the config to USD drive joints under
        the given prim path.

        Args:
            prim_path: USD path to the gripper/hand root prim.
            side: "left" or "right".
            config: Grasp configuration (loaded from YAML).
            drive_mode: ``trigger`` or ``retargeted``.
            retargeter_kind: Isaac Teleop retargeter id when ``drive_mode`` is ``retargeted``.
            joint_aliases: Optional TriHand semantic to USD joint mapping.

        Returns:
            True when at least one joint was matched and configured.
        """
        if not prim_path:
            print(f"[Teleop][Grasp] Cannot configure {side}: empty path")
            return False
        drive_mode = drive_mode.strip().lower()
        retargeter_kind = retargeter_kind.strip().lower() if retargeter_kind else None
        if drive_mode not in {"trigger", "retargeted"}:
            print(f"[Teleop][Grasp] Unsupported drive mode: {drive_mode!r}")
            return False
        if drive_mode == "retargeted" and (retargeter_kind != "trihand" or not joint_aliases):
            print("[Teleop][Grasp] Retargeted drive requires retargeter_kind='trihand' and joint aliases")
            return False

        result = self.validate_prim(prim_path)
        if not result.is_valid:
            print(f"[Teleop][Grasp] Validation failed for '{prim_path}': {result.errors}")
            return False

        if drive_mode == "retargeted":
            from ..retargeting_grasp import validate_trihand_joint_aliases

            controllable_joint_names = {Sdf.Path(path).name for path in result.drive_joint_paths}
            alias_errors = validate_trihand_joint_aliases(
                config,
                joint_aliases,
                available_joint_names=controllable_joint_names,
            )
            if alias_errors:
                print(f"[Teleop][Grasp] Retargeting validation failed for '{prim_path}': {alias_errors}")
                return False

        state = self._side(side)
        state.prim_path = prim_path
        state.config = config
        state.articulation = None
        state.art_joint_map = {}
        state.drive_mode = drive_mode
        state.retargeter_kind = retargeter_kind
        state.joint_aliases = dict(joint_aliases or {})
        state.retargeted_targets = {}

        state.active_joints = self._match_config_joints(config, result.drive_joint_paths)

        # Try to bind to the owning Articulation so drive targets go through
        # the tensor backend (required when an IK controller or similar is
        # managing the same articulation). Any failure here is only reported
        # below if the DriveAPI fallback is also unusable.
        art_bind_error: str | None = None
        art_root = self._find_runtime_articulation_root(prim_path)
        if art_root:
            art_bind_error = self._bind_articulation(state, art_root)

        if not state.active_joints and not state.art_joint_map:
            if art_bind_error:
                print(f"[Teleop][Grasp] {art_bind_error}")
            print(f"[Teleop][Grasp] No joints matched for {side}")
            return False

        self._tracking_enabled[side.lower()] = False
        self._enabled = True
        matched = len(state.art_joint_map or state.active_joints)
        mode = "articulation" if state.articulation else "DriveAPI"
        drive_label = state.drive_mode
        if state.drive_mode == "retargeted" and state.retargeter_kind:
            drive_label = f"{state.drive_mode}/{state.retargeter_kind}"
        print(f"[Teleop][Grasp] Configured {side}: '{prim_path}' ({matched} joint(s), {mode}, {drive_label})")
        return True

    @staticmethod
    def _find_runtime_articulation_root(prim_path: str) -> str | None:
        """Resolve the PhysX articulation root for standalone or assembled grippers.

        A gripper can retain ArticulationRootAPI on its own root after a fixed
        joint attaches it to an external floating or robot articulation. PhysX
        then exposes only the external root through the tensor backend. Prefer
        that connected root and use the gripper's own root only when standalone.
        """
        if stage_utils.is_stage_set() or omni.usd.get_context().get_stage() is not None:
            stage = stage_utils.get_current_stage()
            gripper_root = Sdf.Path(prim_path)
            joints = prim_utils.get_all_matching_child_prims(
                stage.GetPrimAtPath(gripper_root),
                predicate=lambda prim, _: prim.IsA(UsdPhysics.Joint),
                include_self=True,
            )
            for joint_prim in joints:
                joint = UsdPhysics.Joint(joint_prim)
                body_paths = list(joint.GetBody0Rel().GetTargets()) + list(joint.GetBody1Rel().GetTargets())
                for body_path in body_paths:
                    if body_path.HasPrefix(gripper_root):
                        continue
                    connected_root = find_owning_articulation_root(str(body_path))
                    if connected_root:
                        return connected_root
        return find_owning_articulation_root(prim_path)

    def _bind_articulation(self, state: _GraspState, art_root_path: str) -> str | None:
        """Bind matched joints to an Articulation's DOF indices.

        Uses ``dof_names`` (from the physics tensor) for matching rather
        than ``dof_paths`` (from USD), because assembled robots can have
        DOF type mismatches that make USD-derived paths unreliable.

        Returns an error message describing why binding did not happen, or
        ``None`` on success or when the articulation simply has no matching
        DOFs. Callers decide whether to surface the message based on whether
        the DriveAPI fallback is viable.

        Args:
            state: Value for state.
            art_root_path: Value for art root path.

        Returns:
            The requested value.
        """
        try:
            robot = Articulation(art_root_path)
            dof_names_attr = robot.dof_names
        except Exception as exc:
            return f"Could not create Articulation at '{art_root_path}': {exc}"

        if not dof_names_attr:
            return f"Articulation at '{art_root_path}' exposes no DOFs yet"

        dof_names = list(dof_names_attr)
        dof_name_to_idx: dict[str, int] = {name: idx for idx, name in enumerate(dof_names)}

        art_map: dict[str, tuple[JointMapping, int]] = {}
        for jp, (mapping, _drive_api) in state.active_joints.items():
            joint_name = Sdf.Path(jp).name
            dof_idx = dof_name_to_idx.get(joint_name)
            if dof_idx is not None:
                art_map[jp] = (mapping, dof_idx)
                print(f"[Teleop][Grasp] Mapped '{joint_name}' -> DOF index {dof_idx}")
            else:
                print(f"[Teleop][Grasp] Joint '{joint_name}' not found in articulation DOFs ({dof_names})")

        if art_map:
            state.articulation = robot
            state.art_joint_map = art_map
        return None

    def _match_config_joints(
        self,
        config: GraspConfig,
        drive_joint_paths: list[str],
    ) -> dict[str, tuple[JointMapping, UsdPhysics.DriveAPI]]:
        """Matches config joint names to USD joints and caches DriveAPIs.

        Args:
            config: Value for config.
            drive_joint_paths: Value for drive joint paths.

        Returns:
            The requested value.
        """
        if not stage_utils.is_stage_set() and omni.usd.get_context().get_stage() is None:
            return {}
        stage = stage_utils.get_current_stage()

        name_to_path: dict[str, str] = {}
        for jp in drive_joint_paths:
            name_to_path[Sdf.Path(jp).name] = jp

        active: dict[str, tuple[JointMapping, UsdPhysics.DriveAPI]] = {}
        for mapping in config.joints:
            joint_path = name_to_path.get(mapping.name)
            if joint_path is None:
                print(f"[Teleop][Grasp] Warning: config joint '{mapping.name}' not found in prim hierarchy")
                continue

            prim = stage.GetPrimAtPath(joint_path)
            if not prim or not prim.IsValid():
                continue

            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive_type = "angular"
            elif prim.IsA(UsdPhysics.PrismaticJoint):
                drive_type = "linear"
            else:
                continue

            drive_api = UsdPhysics.DriveAPI.Get(prim, drive_type)
            if drive_api:
                self._apply_drive_overrides(mapping, drive_api)
                active[joint_path] = (mapping, drive_api)

        return active

    @staticmethod
    def _apply_drive_overrides(mapping: JointMapping, drive_api: UsdPhysics.DriveAPI) -> None:
        """Apply optional grasp-config drive properties to the matched USD drive."""
        overrides = (
            (mapping.drive_stiffness, drive_api.GetStiffnessAttr()),
            (mapping.drive_damping, drive_api.GetDampingAttr()),
            (mapping.drive_max_force, drive_api.GetMaxForceAttr()),
        )
        for value, attribute in overrides:
            if value is not None and attribute:
                attribute.Set(value)

    # ── Runtime ──────────────────────────────────────────────────────

    def set_input(self, side: str, input_value: float) -> None:
        """Set trigger input for a side and applies to joints.

        Args:
            side: "left" or "right".
            input_value: Trigger value (0=open, 1=closed).
        """
        if not self.is_side_tracking_enabled(side):
            return
        state = self._side(side)
        if state.drive_mode != "trigger":
            return
        state.input_value = max(0.0, min(1.0, input_value))
        self._apply_input(state)

    def set_joint_targets(self, side: str, joint_targets: dict[str, float]) -> None:
        """Set explicit per-joint drive targets for retargeted grasping.

        Args:
            side: ``left`` or ``right``.
            joint_targets: USD joint name to target angle in degrees.
        """
        if not self.is_side_tracking_enabled(side):
            return
        state = self._side(side)
        if state.drive_mode != "retargeted":
            return
        state.retargeted_targets = dict(joint_targets)
        self._apply_retargeted(state)

    def _apply_input(self, state: _GraspState) -> None:
        """Apply current input to all matched joints.

        Uses the Articulation tensor API when available (required for
        assembled robots where another controller owns the articulation).
        Falls back to direct DriveAPI writes for standalone grippers.

        Args:
            state: Value for state.
        """
        if state.drive_mode == "retargeted":
            self._apply_retargeted(state)
            return
        if state.articulation and state.art_joint_map:
            self._apply_via_articulation(state)
        else:
            self._apply_via_drive_api(state)

    def _apply_retargeted(self, state: _GraspState) -> None:
        """Apply explicit per-joint targets produced by a grasp retargeter."""
        if state.articulation and state.art_joint_map:
            robot = state.articulation
            try:
                indices: list[int] = []
                articulation_targets: list[float] = []
                for jp, (mapping, dof_idx) in state.art_joint_map.items():
                    configured_target = state.retargeted_targets.get(mapping.name)
                    if configured_target is None:
                        continue
                    indices.append(dof_idx)
                    articulation_targets.append(
                        self._to_articulation_position(joint_path=jp, configured_target=float(configured_target))
                    )
                if indices:
                    robot.set_dof_position_targets(
                        np.array([articulation_targets], dtype=np.float32),
                        dof_indices=indices,
                    )
            except (AssertionError, RuntimeError):
                self._apply_retargeted_via_drive_api(state)
            return
        self._apply_retargeted_via_drive_api(state)

    @staticmethod
    def _to_articulation_position(*, joint_path: str, configured_target: float) -> float:
        """Convert a USD drive target to tensor articulation units.

        USD angular DriveAPI positions are authored in degrees, while the
        articulation tensor backend consumes radians. Prismatic targets use
        the same linear units in both backends.
        """
        if stage_utils.is_stage_set() or omni.usd.get_context().get_stage() is not None:
            prim = stage_utils.get_current_stage().GetPrimAtPath(joint_path)
            if prim and prim.IsValid() and prim.IsA(UsdPhysics.RevoluteJoint):
                return float(np.deg2rad(configured_target))
        return float(configured_target)

    def _apply_retargeted_via_drive_api(self, state: _GraspState) -> None:
        """Write retargeted joint targets through cached DriveAPI attributes."""
        for _jp, (mapping, drive_api) in state.active_joints.items():
            target = state.retargeted_targets.get(mapping.name)
            if target is None:
                continue
            target_attr = drive_api.GetTargetPositionAttr()
            if target_attr:
                target_attr.Set(float(target))

    def _apply_via_articulation(self, state: _GraspState) -> None:
        """Set drive targets through the Articulation tensor API.

        Args:
            state: Value for state.
        """
        robot = state.articulation
        if robot is None:
            return
        try:
            indices: list[int] = []
            articulation_targets: list[float] = []
            for jp, (mapping, dof_idx) in state.art_joint_map.items():
                indices.append(dof_idx)
                configured_target = mapping.compute_target(state.input_value)
                articulation_targets.append(
                    self._to_articulation_position(joint_path=jp, configured_target=configured_target)
                )
            robot.set_dof_position_targets(
                np.array([articulation_targets], dtype=np.float32),
                dof_indices=indices,
            )
        except (AssertionError, RuntimeError):
            # Tensor not ready (simulation not playing) -- fall back to USD
            self._apply_via_drive_api(state)

    def _apply_via_drive_api(self, state: _GraspState) -> None:
        """Set drive targets directly on USD DriveAPI attributes.

        Args:
            state: Value for state.
        """
        for _jp, (mapping, drive_api) in state.active_joints.items():
            target = mapping.compute_target(state.input_value)
            target_attr = drive_api.GetTargetPositionAttr()
            if target_attr:
                target_attr.Set(target)

    def remove(self, side: str) -> None:
        """Clear grasp configuration for one side.

        Args:
            side: Value for side.
        """
        side = side.lower()
        state = self._sides.get(side)
        if state is None:
            return
        state.prim_path = None
        state.config = None
        state.active_joints.clear()
        state.art_joint_map.clear()
        state.articulation = None
        state.input_value = 0.0
        state.drive_mode = "trigger"
        state.retargeter_kind = None
        state.joint_aliases = {}
        state.retargeted_targets = {}
        self._tracking_enabled[side] = False
        if not any(s.active_joints or s.art_joint_map for s in self._sides.values()):
            self._enabled = False

    def remove_all(self) -> None:
        """Clear all grasp configurations for both sides."""
        for side in list(self._sides):
            self.remove(side)
        print("[Teleop][Grasp] Controllers removed.")

    def set_side_tracking_enabled(self, side: str, enabled: bool) -> None:
        """Enable/disable trigger tracking for one side.

        Args:
            side: Value for side.
            enabled: Value for enabled.
        """
        side = side.lower()
        if side not in self._tracking_enabled:
            return
        self._tracking_enabled[side] = bool(enabled)

    def is_side_tracking_enabled(self, side: str) -> bool:
        """Return True if trigger tracking is enabled for one side.

        Args:
            side: Value for side.

        Returns:
            The requested value.
        """
        side = side.lower()
        state = self._sides.get(side)
        if state is None:
            return False
        has_joints = bool(state.active_joints or state.art_joint_map)
        return bool(self._tracking_enabled.get(side, False) and has_joints and state.config is not None)

    @property
    def has_any_side_tracking_enabled(self) -> bool:
        """True when at least one side accepts trigger tracking.

        Returns:
            The requested value.
        """
        return self.is_side_tracking_enabled("left") or self.is_side_tracking_enabled("right")

    # ── Properties ───────────────────────────────────────────────────

    @property
    def is_enabled(self) -> bool:
        """True if any side has active joints.

        Returns:
            The requested value.
        """
        return self._enabled and any(s.active_joints or s.art_joint_map for s in self._sides.values())

    def get_side_config(self, side: str) -> GraspConfig | None:
        """Return the loaded grasp config for one side."""
        return self._side(side).config

    def get_side_drive_settings(self, side: str) -> tuple[str, str | None, dict[str, str]]:
        """Return drive mode, retargeter kind, and joint aliases for one side."""
        state = self._side(side)
        return state.drive_mode, state.retargeter_kind, dict(state.joint_aliases)

    @property
    def left_prim_path(self) -> str | None:
        """Return the left side prim path.

        Returns:
            The requested value.
        """
        return self._side("left").prim_path

    @property
    def right_prim_path(self) -> str | None:
        """Return the right side prim path.

        Returns:
            The requested value.
        """
        return self._side("right").prim_path
