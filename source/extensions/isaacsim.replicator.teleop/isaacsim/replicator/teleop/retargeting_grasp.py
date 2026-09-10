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

"""Controller-to-joint grasp retargeting helpers.

Retargeting here is algorithm + profile driven. USD joint names, ranges, and
semantic aliases live in teleop profiles and grasp YAML — not in this module.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from .controllers.grasp import GraspConfig, JointMapping

# TriHand semantic joint order shared by Isaac Teleop's controller retargeter.
TRIHAND_SEMANTIC_JOINTS: tuple[str, ...] = (
    "thumb_rotation",
    "thumb_proximal",
    "thumb_distal",
    "index_proximal",
    "index_distal",
    "middle_proximal",
    "middle_distal",
)


class GraspDriveMode(str, Enum):
    """How grasp joint targets are produced for a configured side."""

    TRIGGER = "trigger"
    RETARGETED = "retargeted"


class GraspRetargeterKind(str, Enum):
    """Supported controller retargeting algorithms for grasp drive."""

    TRIHAND = "trihand"


def _read_controller_analogs(controller_snapshot: object | None) -> tuple[float, float]:
    """Extract trigger and squeeze analog values from a controller snapshot."""
    if controller_snapshot is None:
        return 0.0, 0.0
    inputs = getattr(controller_snapshot, "inputs", None)
    if inputs is None:
        return 0.0, 0.0
    trigger = float(getattr(inputs, "trigger_value", 0.0) or 0.0)
    squeeze = float(getattr(inputs, "squeeze_value", 0.0) or 0.0)
    return max(0.0, min(1.0, trigger)), max(0.0, min(1.0, squeeze))


def compute_trihand_joint_values(trigger: float, squeeze: float, *, hand_side: str) -> list[float]:
    """Map controller trigger/squeeze to seven TriHand semantic joint values.

    Implements the documented TriHand controller mapping locally. Keeping the
    small mapping here avoids depending on Isaac Teleop's private
    ``_map_to_hand_joints`` method.

    Args:
        trigger: Index trigger value in ``[0, 1]``.
        squeeze: Grip/squeeze value in ``[0, 1]``.
        hand_side: ``left`` or ``right``.

    Returns:
        Seven joint values in :data:`TRIHAND_SEMANTIC_JOINTS` order.
    """
    normalized_side = hand_side.strip().lower()
    if normalized_side not in {"left", "right"}:
        raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side!r}")
    trigger = max(0.0, min(1.0, float(trigger)))
    squeeze = max(0.0, min(1.0, float(squeeze)))
    is_left = normalized_side == "left"
    thumb_button = max(trigger, squeeze)
    thumb_angle = -thumb_button
    thumb_rotation = 0.5 * trigger - 0.5 * squeeze
    if not is_left:
        thumb_rotation = -thumb_rotation

    values = [
        thumb_rotation,
        thumb_angle * 0.4,
        thumb_angle * 0.7,
        trigger,
        trigger,
        squeeze,
        squeeze,
    ]
    if is_left:
        values = [-v for v in values]
    return [float(v) for v in values]


def _mapping_for_joint_name(config: GraspConfig, joint_name: str) -> JointMapping | None:
    for mapping in config.joints:
        if mapping.name == joint_name:
            return mapping
    return None


def map_trihand_to_joint_targets(
    trihand_values: list[float],
    grasp_config: GraspConfig,
    joint_aliases: dict[str, str],
) -> dict[str, float]:
    """Convert TriHand semantic joint values into USD joint targets (degrees).

    Each TriHand output is treated as a normalized activation in ``[0, 1]`` via
    ``abs(value)`` and mapped through the grasp config's per-joint target range.

    Args:
        trihand_values: Seven values in :data:`TRIHAND_SEMANTIC_JOINTS` order.
        grasp_config: Grasp YAML config defining joint names and target ranges.
        joint_aliases: TriHand semantic name to USD joint name mapping from profile.

    Returns:
        USD joint name to drive target in degrees.
    """
    semantic_to_value = {
        semantic: trihand_values[idx] if idx < len(trihand_values) else 0.0
        for idx, semantic in enumerate(TRIHAND_SEMANTIC_JOINTS)
    }
    targets: dict[str, float] = {}
    for semantic, usd_name in joint_aliases.items():
        mapping = _mapping_for_joint_name(grasp_config, usd_name)
        if mapping is None:
            continue
        activation = max(0.0, min(1.0, abs(float(semantic_to_value.get(semantic, 0.0)))))
        targets[usd_name] = mapping.compute_target(activation)
    return targets


def validate_trihand_joint_aliases(
    grasp_config: GraspConfig | None,
    joint_aliases: dict[str, str] | None,
    *,
    available_joint_names: set[str] | None = None,
) -> list[str]:
    """Validate TriHand aliases against semantic, config, and optional USD joints.

    Args:
        grasp_config: Grasp config that defines target ranges.
        joint_aliases: TriHand semantic name to USD joint name mapping.
        available_joint_names: Optional controllable USD joint names below the selected grasp prim.

    Returns:
        Actionable validation errors. An empty list means the aliases are valid.
    """
    aliases = parse_joint_aliases(joint_aliases)
    if not aliases:
        return ["TriHand retargeting requires at least one joint alias."]

    errors: list[str] = []
    unknown_semantics = sorted(set(aliases) - set(TRIHAND_SEMANTIC_JOINTS))
    if unknown_semantics:
        errors.append(f"Unknown TriHand semantic alias(es): {', '.join(unknown_semantics)}.")

    alias_targets = set(aliases.values())
    if grasp_config is not None:
        config_joint_names = {mapping.name for mapping in grasp_config.joints}
        missing_config = sorted(alias_targets - config_joint_names)
        if missing_config:
            errors.append(f"Alias target joint(s) missing from grasp config: {', '.join(missing_config)}.")

    if available_joint_names is not None:
        missing_usd = sorted(alias_targets - available_joint_names)
        if missing_usd:
            errors.append(f"Alias target joint(s) not controllable below grasp prim: {', '.join(missing_usd)}.")
    return errors


def compute_retargeted_joint_targets(
    *,
    retargeter_kind: GraspRetargeterKind,
    controller_snapshot: object | None,
    grasp_config: GraspConfig,
    hand_side: str,
    joint_aliases: dict[str, str] | None = None,
) -> dict[str, float]:
    """Compute per-joint grasp targets for one hand side.

    Args:
        retargeter_kind: Selected controller retargeting algorithm.
        controller_snapshot: Live or debug controller snapshot.
        grasp_config: Configured grasp YAML for joint ranges.
        hand_side: ``left`` or ``right``.
        joint_aliases: TriHand semantic to USD joint mapping from the teleop profile.

    Returns:
        USD joint name to target angle in degrees.
    """
    if retargeter_kind != GraspRetargeterKind.TRIHAND:
        return {}
    if not joint_aliases:
        return {}

    trigger, squeeze = _read_controller_analogs(controller_snapshot)
    trihand_values = compute_trihand_joint_values(trigger, squeeze, hand_side=hand_side)
    return map_trihand_to_joint_targets(trihand_values, grasp_config, joint_aliases)


def parse_grasp_retargeter_kind(value: str | GraspRetargeterKind | None) -> GraspRetargeterKind | None:
    """Parse a profile or UI string into a grasp retargeter kind."""
    if value is None or value == "":
        return None
    if isinstance(value, GraspRetargeterKind):
        return value
    normalized = str(value).strip().lower()
    if not normalized or normalized == "none":
        return None
    return GraspRetargeterKind(normalized)


def parse_grasp_drive_mode(value: str | GraspDriveMode | None) -> GraspDriveMode:
    """Parse a profile or UI string into a grasp drive mode."""
    if value is None or value == "":
        return GraspDriveMode.TRIGGER
    if isinstance(value, GraspDriveMode):
        return value
    return GraspDriveMode(str(value).strip().lower())


def parse_joint_aliases(value: Any) -> dict[str, str]:
    """Parse a joint alias mapping from profile YAML."""
    if not isinstance(value, dict):
        return {}
    aliases: dict[str, str] = {}
    for key, mapped in value.items():
        key_str = str(key).strip()
        mapped_str = str(mapped).strip()
        if key_str and mapped_str:
            aliases[key_str] = mapped_str
    return aliases
