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

"""Typed accessors and parsing helpers for exported Isaac Lab policy environment configs.

The env config configures the simulation (timing, spawn, joint drives, actuator models); the
policy interface itself is derived from the exported IO descriptor, not from here.

:class:`PolicyEnvConfig` is the parse-once typed entry point. Raw-mapping extraction stays an
implementation detail except for :func:`get_newton_actuator_specs`, which is also consumed by
the Newton actuator adapter tests.
"""

from __future__ import annotations

import copy
import io
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import yaml

from .spec import read_artifact_bytes

__all__ = [
    "JointProperties",
    "NewtonActuatorSpec",
    "PhysicsTiming",
    "PolicyEnvConfig",
    "StartupMaterialEvent",
    "SUPPORTED_NEWTON_ACTUATOR_CLASSES",
    "SpawnProps",
    "get_newton_actuator_specs",
]

#: Exported Isaac Lab actuator classes the Newton actuator adapter can deploy. Membership also
#: decides whether an actuator group's ``effort_limit`` clamps the actuator model (supported
#: classes) or configures the physics drive (legacy classless groups).
SUPPORTED_NEWTON_ACTUATOR_CLASSES = frozenset(
    {
        "ActuatorNetLSTM",
        "ActuatorNetMLP",
        "DCMotor",
        "DelayedDCMotor",
        "DelayedPDActuator",
        "IdealPDActuator",
        "RemotizedPDActuator",
    }
)


# ------------------------------------------------------------------------------------ loading


class _EnvConfigYamlLoader(yaml.SafeLoader):
    """Safe YAML loader constructing python tuples and mapping unknown tags to plain containers."""

    def construct_unknown(self, node: object) -> object:
        if isinstance(node, yaml.MappingNode):
            return self.construct_mapping(node, deep=True)
        if isinstance(node, yaml.SequenceNode):
            return self.construct_sequence(node, deep=True)
        if isinstance(node, yaml.ScalarNode):
            return self.construct_scalar(node)
        return None

    def tuple_constructor(self, node: object) -> tuple:
        return tuple(self.construct_sequence(node))


_EnvConfigYamlLoader.add_constructor("tag:yaml.org,2002:python/tuple", _EnvConfigYamlLoader.tuple_constructor)
_EnvConfigYamlLoader.add_constructor(None, _EnvConfigYamlLoader.construct_unknown)


def _load_env_config_data(env_config_path: str) -> dict[str, Any]:
    """Read and parse an exported env config from a local path or ``omni.client`` URL.

    Args:
        env_config_path: Path to the artifact's ``env.yaml``.

    Returns:
        The resulting dict.
    """
    content = read_artifact_bytes(env_config_path)
    data = yaml.load(io.BytesIO(content), Loader=_EnvConfigYamlLoader)
    if not isinstance(data, dict):
        raise ValueError(f"Policy environment config {env_config_path!r} did not contain a YAML mapping.")
    return data


# ---------------------------------------------------------------------- joint pattern matching
# Shared by the joint-property, actuator, and typed-accessor sections below.


def _matches_joint_pattern(joint_name: str, pattern: str) -> bool:
    """Match one Isaac Lab regular expression, including a literal USD-uniquified alias.

    Args:
        joint_name: DOF name to match.
        pattern: Joint-name pattern from the env config.

    Returns:
        True on success.
    """
    if re.fullmatch(pattern, joint_name):
        return True

    # USD import may uniquify a literal trained name (for example, torso -> torso_1). Keep that
    # compatibility narrow: arbitrary prefixes and glob syntax are not Isaac Lab semantics.
    return re.fullmatch(re.escape(pattern) + r"_\d+", joint_name) is not None


def _first_matching_pattern(joint_name: str, patterns: Iterable[str]) -> str | None:
    """Return the first pattern matching ``joint_name``, or None.

    First match in declaration order wins; exported configs rely on this to place explicit
    per-joint entries ahead of broad patterns.

    Args:
        joint_name: DOF name to match.
        patterns: Joint-name patterns to match against.

    Returns:
        The resulting str | None.
    """
    for pattern in patterns:
        if _matches_joint_pattern(joint_name, pattern):
            return pattern
    return None


def _actuator_class_name(actuator_config: dict[str, Any]) -> str:
    """Return the trailing class name from an exported Isaac Lab actuator config.

    Args:
        actuator_config: One actuator group from the env config.

    Returns:
        The resolved string.
    """
    return str(actuator_config.get("class_type") or "").rsplit(":", 1)[-1]


# ------------------------------------------------------------------------------------- timing


@dataclass(frozen=True)
class PhysicsTiming:
    """Simulation timing declared by an exported env config.

    Args:
        decimation: Physics steps per control step; the deployed policy infers once per
            ``decimation`` physics ticks, matching the training cadence.
        physics_dt: Physics step size in seconds.
        render_interval: Physics steps per rendered frame.
    """

    decimation: int
    physics_dt: float
    render_interval: int


# -------------------------------------------------------------------------------------- spawn


@dataclass(frozen=True)
class SpawnProps:
    """Spawn-time authoring properties from the robot spawn block of an env config.

    All properties must be authored on the USD stage before the articulation view is
    constructed; absent blocks are exposed as empty mappings and preserve authored USD values.

    Args:
        usd_path: Robot USD to spawn, or None when the config does not declare one (the
            deployment spec then must supply the asset).
        articulation_props: ``scene.robot.spawn.articulation_props`` block (solver iteration
            counts, self-collision flags, ...).
        rigid_body_props: ``scene.robot.spawn.rigid_props`` block (gravity, damping, ...).
        joint_drive_props: ``scene.robot.spawn.joint_drive_props`` block (drive authoring
            directives such as ``ensure_drives_exist``).
    """

    usd_path: str | None
    articulation_props: Mapping[str, Any]
    rigid_body_props: Mapping[str, Any]
    joint_drive_props: Mapping[str, Any]


@dataclass(frozen=True)
class StartupMaterialEvent:
    """Deterministic standalone form of an exported startup material event.

    Isaac Lab samples material buckets across vectorized environments. A standalone deployment
    has one scene instance, so each exported range is represented by its midpoint.

    Args:
        name: Exported event name.
        entity_name: Scene entity selected by the event.
        body_patterns: Regular expressions selecting links on that entity.
        static_friction: Midpoint static-friction value.
        dynamic_friction: Midpoint dynamic-friction value.
        restitution: Midpoint restitution value.
    """

    name: str
    entity_name: str
    body_patterns: tuple[str, ...]
    static_friction: float
    dynamic_friction: float
    restitution: float


# --------------------------------------------------------------------------- joint properties


@dataclass(frozen=True)
class JointProperties:
    """Per-joint simulation properties resolved in simulator DOF order.

    A None effort or velocity limit preserves the imported USD value; all other properties are
    complete per-joint vectors ordered by ``joint_names``.

    Args:
        joint_names: DOF names the vectors below are ordered by.
        effort_limits: Per-joint solver effort limits authored into the engine — the
            ``effort_limit_sim`` family (plain ``effort_limit`` only for legacy implicit
            groups); None preserves the imported USD limit.
        velocity_limits: Per-joint solver velocity limits — ``velocity_limit_sim`` ONLY
            (plain ``velocity_limit`` clamps the actuator model, never the engine); None
            preserves the imported USD limit.
        stiffness: Per-joint drive stiffness gains.
        damping: Per-joint drive damping gains.
        armature: Per-joint rotor armature (reflected inertia); None preserves the imported value.
        default_positions: Per-joint reset positions; also the action offset for policies
            trained with default-offset joint position actions.
        default_velocities: Per-joint reset velocities.
    """

    joint_names: tuple[str, ...]
    effort_limits: tuple[float | None, ...]
    velocity_limits: tuple[float | None, ...]
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    armature: tuple[float | None, ...]
    default_positions: tuple[float, ...]
    default_velocities: tuple[float, ...]


def _finite_limit(value: object) -> object:
    """Convert a scalar infinite simulation limit to the largest finite limit.

    Args:
        value: Raw exported value.

    Returns:
        The resulting object.
    """
    return float(sys.maxsize) if value == float("inf") else value


def _expand_property(target: dict, value: object, patterns: Sequence[str], missing: float | None) -> None:
    """Expand a scalar or pattern-keyed property value into ``target`` for the given patterns.

    Args:
        target: Mapping to fill in place.
        value: Raw exported value.
        patterns: Joint-name patterns to match against.
        missing: Value used when a pattern names no entry, or None.
    """
    if isinstance(value, dict):
        target.update(value)
        return
    for pattern in patterns:
        target[pattern] = missing if value is None else float(value)


def _resolve_robot_joint_properties(data: dict[str, Any], joint_names: list[str], source: str) -> JointProperties:
    """Extract per-joint properties in ``joint_names`` order with pattern matching.

    Resolution is strict: every joint must match exactly one actuator group, every property must
    cover it, and the init state must supply its default position and velocity -- a miss raises
    rather than silently zeroing a gain or pose. Explicitly null gains still resolve to zero, and
    absent simulation limits resolve to None so the imported USD values stand.

    Returns:
        Joint properties in the requested order.

    Args:
        data: Parsed env-config mapping.
        joint_names: Joint names to resolve, in the order required.
        source: Config path or label used in error messages.
    """
    robot = (data.get("scene") or {}).get("robot") or {}
    actuator_data = robot.get("actuators") or {}
    stiffness: dict[str, Any] = {}
    damping: dict[str, Any] = {}
    armature: dict[str, Any] = {}
    effort_limits: dict[str, Any] = {}
    velocity_limits: dict[str, Any] = {}
    joint_names_expr_list: list[str] = []

    # Build pattern-keyed property maps: a mapping-valued property contributes its explicit
    # per-joint keys, a scalar is written under every pattern of its actuator group.
    group_patterns: dict[str, list[str]] = {}
    for group_name, actuator_config in actuator_data.items():
        patterns = actuator_config.get("joint_names_expr") or actuator_config.get("joint_names") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        patterns = [str(pattern) for pattern in patterns]
        group_patterns[group_name] = patterns
        joint_names_expr_list.extend(patterns)

        # For supported explicit actuator classes ``effort_limit`` clamps the actuator model,
        # never the solver; only legacy classless groups fall back to it for physics.
        effort_limit = actuator_config.get("effort_limit_sim")
        if effort_limit is None and _actuator_class_name(actuator_config) not in SUPPORTED_NEWTON_ACTUATOR_CLASSES:
            effort_limit = actuator_config.get("effort_limit")
        _expand_property(effort_limits, _finite_limit(effort_limit), patterns, missing=None)
        # ``velocity_limit`` constrains the actuator model; only ``velocity_limit_sim`` configures physics.
        _expand_property(
            velocity_limits, _finite_limit(actuator_config.get("velocity_limit_sim")), patterns, missing=None
        )
        _expand_property(stiffness, actuator_config.get("stiffness"), patterns, missing=0.0)
        _expand_property(damping, actuator_config.get("damping"), patterns, missing=0.0)
        _expand_property(armature, actuator_config.get("armature"), patterns, missing=None)

    init_state = robot.get("init_state") or {}
    for key in ("joint_pos", "joint_vel"):
        if init_state.get(key) is None:
            raise ValueError(f"{source}: required config key 'scene.robot.init_state.{key}' is missing.")
    default_pos: dict[str, Any] = {}
    default_vel: dict[str, Any] = {}
    _expand_property(default_pos, init_state.get("joint_pos"), joint_names_expr_list, missing=0.0)
    for pattern in joint_names_expr_list:
        default_pos.setdefault(pattern, 0.0)
    _expand_property(default_vel, init_state.get("joint_vel"), joint_names_expr_list, missing=0.0)

    stiffness_inorder = []
    damping_inorder = []
    armature_inorder = []
    effort_limits_inorder = []
    velocity_limits_inorder = []
    default_pos_inorder = []
    default_vel_inorder = []

    def resolve(property_map: dict, label: str, joint: str, owner: str) -> object:
        pattern = _first_matching_pattern(joint, property_map)
        if pattern is None:
            raise ValueError(
                f"{source}: joint {joint!r} resolves through pattern {owner!r} but has no "
                f"{label} entry in 'scene.robot.actuators'."
            )
        return property_map[pattern]

    # Resolve every property through its own patterns. Isaac Lab commonly declares one broad
    # actuator-group pattern and narrower mapping-valued gain patterns within that group.
    for joint in joint_names:
        matched = [
            group_name
            for group_name, patterns in group_patterns.items()
            if any(_matches_joint_pattern(joint, pattern) for pattern in patterns)
        ]
        if not matched:
            raise ValueError(f"{source}: joint {joint!r} does not match any pattern in 'scene.robot.actuators'.")
        if len(matched) > 1:
            raise ValueError(
                f"{source}: joint {joint!r} is matched by multiple actuator groups {matched}; "
                "overlapping actuator groups are ambiguous."
            )
        owner = _first_matching_pattern(joint, group_patterns[matched[0]])

        stiffness_inorder.append(resolve(stiffness, "stiffness", joint, owner))
        damping_inorder.append(resolve(damping, "damping", joint, owner))
        armature_inorder.append(resolve(armature, "armature", joint, owner))
        effort_limits_inorder.append(resolve(effort_limits, "effort limit", joint, owner))
        velocity_limits_inorder.append(resolve(velocity_limits, "velocity limit", joint, owner))

        pos_pattern = _first_matching_pattern(joint, default_pos)
        if pos_pattern is None:
            raise ValueError(
                f"{source}: joint {joint!r} has no default joint position in 'scene.robot.init_state.joint_pos'."
            )
        default_pos_inorder.append(default_pos[pos_pattern])

        vel_pattern = _first_matching_pattern(joint, default_vel)
        if vel_pattern is None:
            raise ValueError(
                f"{source}: joint {joint!r} has no default joint velocity in 'scene.robot.init_state.joint_vel'."
            )
        default_vel_inorder.append(default_vel[vel_pattern])

    return JointProperties(
        joint_names=tuple(joint_names),
        effort_limits=tuple(effort_limits_inorder),
        velocity_limits=tuple(velocity_limits_inorder),
        stiffness=tuple(stiffness_inorder),
        damping=tuple(damping_inorder),
        armature=tuple(armature_inorder),
        default_positions=tuple(default_pos_inorder),
        default_velocities=tuple(default_vel_inorder),
    )


# ---------------------------------------------------------------------------------- actuators


@dataclass(frozen=True)
class NewtonActuatorSpec:
    """Newton actuator parameters resolved from an exported Isaac Lab config.

    Args:
        class_name: Exported Isaac Lab actuator class (one of
            :data:`SUPPORTED_NEWTON_ACTUATOR_CLASSES`); selects the adapter's actuator model.
        joints: DOF names this group drives, in simulator DOF order.
        network_file: Learned actuator-network weights path or URL; None for analytic models.
        stiffness: Per-joint P gains applied inside the actuator model, not on the physics drive.
        damping: Per-joint D gains applied inside the actuator model, not on the physics drive.
        effort_limit: Per-joint torque clamp of the actuator model.
        velocity_limit: Per-joint velocity bound of the actuator model (the DC-motor
            torque-speed curve corner, not the solver limit).
        saturation_effort: Per-joint DC-motor peak torque; falls back to the effort limit when
            the config leaves it unset.
        max_delay: Fixed physics-step actuation delay used for delayed actuator classes. Isaac Lab
            may randomize between minimum and maximum delays, while deployment uses the exported maximum.
        joint_parameter_lookup: (angle, transmission ratio, output torque) interpolation rows
            (RemotizedPDActuator).
    """

    class_name: str
    joints: tuple[str, ...]
    network_file: str | None
    stiffness: tuple[float, ...]
    damping: tuple[float, ...]
    effort_limit: tuple[float, ...]
    velocity_limit: tuple[float, ...]
    saturation_effort: tuple[float, ...]
    max_delay: int = 0
    joint_parameter_lookup: tuple[tuple[float, float, float], ...] = ()


def _resolve_per_joint(value: object, default: float, group_joints: list[str]) -> list[float]:
    """Expand a scalar or pattern-keyed mapping to one float per joint in ``group_joints``.

    Args:
        value: Raw exported value.
        default: Value used when the config omits the field.
        group_joints: Joints belonging to the actuator group.

    Returns:
        The resulting list.
    """
    # Per-joint precedence: the first matching key of a mapping in declaration order (exported
    # configs place explicit per-joint entries ahead of broad patterns) > a scalar applied to
    # every joint > the fallback default (for absent or None-valued properties).
    resolved = []
    for joint in group_joints:
        if isinstance(value, dict):
            pattern = _first_matching_pattern(joint, value)
            resolved.append(float(value[pattern]) if pattern is not None else float(default))
        elif isinstance(value, (float, int)):
            resolved.append(float(value))
        else:
            resolved.append(float(default))
    return resolved


def get_newton_actuator_specs(data: dict[str, Any], joint_names: list[str]) -> list[NewtonActuatorSpec]:
    """Resolve exported non-implicit actuator groups into Newton actuator specs.

    Implicit actuators remain on the physics-engine drive and are intentionally omitted; scalar
    and pattern-keyed fields are expanded per joint. Explicit groups outside the requested policy
    joint subset are omitted. Unsupported actuator classes raise NotImplementedError.

    Args:
        data: Parsed env-config mapping.
        joint_names: Joint names to resolve, in the order required.

    Returns:
        The resulting list.
    """
    actuator_data = ((data.get("scene") or {}).get("robot") or {}).get("actuators") or {}
    big = float(sys.maxsize)
    specs: list[NewtonActuatorSpec] = []
    for actuator_config in actuator_data.values():
        class_name = _actuator_class_name(actuator_config)
        if not class_name or class_name == "ImplicitActuator":
            continue
        if class_name not in SUPPORTED_NEWTON_ACTUATOR_CLASSES:
            raise NotImplementedError(f"Actuator class {class_name!r} is not supported by the Newton actuator adapter.")

        patterns = actuator_config.get("joint_names_expr") or actuator_config.get("joint_names") or []
        if isinstance(patterns, str):
            patterns = [patterns]
        group_joints = [
            joint for joint in joint_names if any(_matches_joint_pattern(joint, pattern) for pattern in patterns)
        ]
        if not group_joints:
            continue

        joint_parameter_lookup = actuator_config.get("joint_parameter_lookup") or []
        if class_name == "RemotizedPDActuator" and not joint_parameter_lookup:
            raise ValueError("RemotizedPDActuator requires joint_parameter_lookup.")

        # Actuator-model limits prefer the explicit model field and fall back to the sim field;
        # this is the reverse of the physics extraction in _resolve_robot_joint_properties.
        effort_limit = actuator_config.get("effort_limit")
        if effort_limit is None:
            effort_limit = actuator_config.get("effort_limit_sim")
        velocity_limit = actuator_config.get("velocity_limit")
        if velocity_limit is None:
            velocity_limit = actuator_config.get("velocity_limit_sim")
        saturation_effort = actuator_config.get("saturation_effort")
        if saturation_effort is None:
            saturation_effort = effort_limit

        specs.append(
            NewtonActuatorSpec(
                class_name=class_name,
                joints=tuple(group_joints),
                network_file=actuator_config.get("network_file"),
                stiffness=tuple(_resolve_per_joint(actuator_config.get("stiffness"), 0.0, group_joints)),
                damping=tuple(_resolve_per_joint(actuator_config.get("damping"), 0.0, group_joints)),
                effort_limit=tuple(_resolve_per_joint(effort_limit, big, group_joints)),
                velocity_limit=tuple(_resolve_per_joint(velocity_limit, big, group_joints)),
                saturation_effort=tuple(_resolve_per_joint(saturation_effort, big, group_joints)),
                max_delay=int(actuator_config.get("max_delay") or 0),
                joint_parameter_lookup=tuple(tuple(float(value) for value in row) for row in joint_parameter_lookup),
            )
        )
    return specs


# ------------------------------------------------------------------------------ initial state


def _get_initial_root_state(
    data: dict[str, Any], entity_name: str = "robot"
) -> tuple[list[float] | None, list[float] | None]:
    """Return the initial root position and wxyz orientation from an env config.

    Current Isaac Lab configs store ``init_state.rot`` as xyzw while Isaac Sim spawn APIs take
    wxyz. Legacy configs with a ``sim.physx`` block predate that convention change and already
    store wxyz, so they are preserved.

    Returns:
        Initial root position and wxyz orientation; either value is None when not configured.

    Args:
        data: Parsed env-config mapping.
        entity_name: Scene entity whose initial state is resolved.
    """
    init_state = ((data.get("scene") or {}).get(entity_name) or {}).get("init_state") or {}
    position = list(init_state["pos"]) if init_state.get("pos") is not None else None
    orientation = None
    if init_state.get("rot") is not None:
        rotation = list(init_state["rot"])
        sim_config = data.get("sim") or {}
        if "physx" in sim_config and "physics" not in sim_config:
            orientation = rotation
        else:
            orientation = [rotation[3], rotation[0], rotation[1], rotation[2]]

    return position, orientation


# ------------------------------------------------------------------- parse-once typed accessor


class PolicyEnvConfig:
    """Parse-once typed accessor over an exported Isaac Lab policy environment config.

    Exposes typed views of the timing, spawn, joint-property, and actuator-model structure of
    one exported ``env.yaml``. Joint-property resolution raises on incomplete, overlapping, or
    ambiguous actuator coverage instead of silently zero-defaulting a joint; numeric extraction
    delegates to this module's parsing helpers so extracted values match the legacy loader
    exactly.

    Args:
        data: Parsed environment configuration mapping.
        source_path: Config path used in error messages, or None for in-memory mappings.

    Raises:
        ValueError: If the timing or spawn blocks are missing or malformed.
    """

    def __init__(self, data: Mapping[str, Any], source_path: str | None = None) -> None:
        # Deep-copied so later mutation of the caller's mapping cannot change the views.
        self._data: dict[str, Any] = copy.deepcopy(dict(data))
        self._source = source_path if source_path else "policy env config"

        self._timing = self._build_timing()
        self._spawn = self._build_spawn()

    @classmethod
    def from_file(cls, path: str) -> PolicyEnvConfig:
        """Load and parse a policy environment config from a local path or ``omni.client`` URL.

        Args:
            path: Artifact path, local or an Omniverse URL.

        Returns:
            The resulting :class:`PolicyEnvConfig`.
        """
        return cls(_load_env_config_data(path), source_path=path)

    # -- typed views --------------------------------------------------------------------------

    @property
    def timing(self) -> PhysicsTiming:
        """Simulation timing declared by the config."""
        return self._timing

    @property
    def spawn(self) -> SpawnProps:
        """Spawn-time authoring properties declared by the config."""
        return self._spawn

    @property
    def initial_root_pose(self) -> tuple[list[float] | None, list[float] | None]:
        """Initial root position and wxyz orientation; either value is None when unconfigured.

        Current exported xyzw quaternions are converted to wxyz; legacy wxyz values are preserved.
        """
        return _get_initial_root_state(self._data)

    @property
    def newton_shape_defaults(self) -> Mapping[str, Any]:
        """Exported Newton shape defaults to author before articulation import."""
        physics = (self._data.get("sim") or {}).get("physics") or {}
        shape_defaults = physics.get("default_shape_cfg") or {}
        return copy.deepcopy(shape_defaults) if isinstance(shape_defaults, Mapping) else {}

    @property
    def episode_length_s(self) -> float:
        """Episode duration declared by the exported task config."""
        return float(self._data["episode_length_s"])

    def scene_entity_usd_path(self, entity_name: str) -> str | None:
        """Return a named scene entity's exported USD path, if present.

        Args:
            entity_name: Key below the exported ``scene`` block.

        Returns:
            The entity's ``spawn.usd_path``, or None when absent.
        """
        spawn = ((self._data.get("scene") or {}).get(entity_name) or {}).get("spawn") or {}
        usd_path = spawn.get("usd_path")
        return usd_path if isinstance(usd_path, str) and usd_path else None

    def scene_entity_root_pose(self, entity_name: str) -> tuple[list[float] | None, list[float] | None]:
        """Return a scene entity's initial root position and wxyz orientation.

        Args:
            entity_name: Key below the exported ``scene`` block.

        Returns:
            Initial root position and wxyz orientation; either value is None when unconfigured.
        """
        return _get_initial_root_state(self._data, entity_name)

    def scene_entity_joint_default_state(
        self, entity_name: str, joint_names: Sequence[str]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Resolve a named scene entity's default joint state in simulator DOF order.

        Args:
            entity_name: Key below the exported ``scene`` block.
            joint_names: Entity DOF names in simulator order.

        Returns:
            Default joint positions and velocities.
        """
        init_state = ((self._data.get("scene") or {}).get(entity_name) or {}).get("init_state") or {}
        return (
            tuple(_resolve_per_joint(init_state.get("joint_pos"), 0.0, list(joint_names))),
            tuple(_resolve_per_joint(init_state.get("joint_vel"), 0.0, list(joint_names))),
        )

    @property
    def startup_material_events(self) -> tuple[StartupMaterialEvent, ...]:
        """Return supported startup material events as deterministic midpoint values."""
        result = []
        for event_name, event in (self._data.get("events") or {}).items():
            if (
                not isinstance(event, dict)
                or event.get("mode") != "startup"
                or str(event.get("func") or "").rsplit(":", 1)[-1] != "randomize_rigid_body_material"
            ):
                continue

            params = event.get("params") or {}
            asset = params.get("asset_cfg") or {}
            entity_name = asset.get("name")
            if not isinstance(entity_name, str) or not entity_name:
                continue

            body_names = asset.get("body_names")
            body_patterns = (body_names,) if isinstance(body_names, str) else tuple(body_names or ())

            def midpoint(key: str, default: tuple[float, float]) -> float:
                bounds = params.get(key) or default
                return 0.5 * (float(bounds[0]) + float(bounds[1]))

            static_friction = midpoint("static_friction_range", (1.0, 1.0))
            dynamic_friction = midpoint("dynamic_friction_range", (1.0, 1.0))
            if params.get("make_consistent"):
                dynamic_friction = min(static_friction, dynamic_friction)
            result.append(
                StartupMaterialEvent(
                    name=str(event_name),
                    entity_name=entity_name,
                    body_patterns=body_patterns,
                    static_friction=static_friction,
                    dynamic_friction=dynamic_friction,
                    restitution=midpoint("restitution_range", (0.0, 0.0)),
                )
            )
        return tuple(result)

    def joint_properties(self, joint_names: Sequence[str]) -> JointProperties:
        """Resolve per-joint simulation properties for the named joints.

        Every joint must resolve through exactly one actuator group with a complete property
        set (ValueError otherwise); absent simulation effort and velocity limits resolve to
        None and preserve the imported USD values.

        Args:
            joint_names: Joint names to resolve, in the order required.

        Returns:
            The resulting :class:`JointProperties`.
        """
        return _resolve_robot_joint_properties(self._data, list(joint_names), self._source)

    def actuator_model_specs(self, joint_names: Sequence[str]) -> list[NewtonActuatorSpec]:
        """Resolve exported non-implicit actuator groups into actuator model specs.

        Implicit and legacy classless actuator groups configure the physics drives instead and
        are omitted. Explicit groups outside the policy joint subset are also omitted. Unsupported
        actuator classes raise NotImplementedError; joint ownership itself is enforced by
        :meth:`joint_properties`, which every deployment resolves first.

        Args:
            joint_names: Joint names to resolve, in the order required.

        Returns:
            The resulting list.
        """
        return get_newton_actuator_specs(self._data, list(joint_names))

    # -- config block access ------------------------------------------------------------------

    def _build_timing(self) -> PhysicsTiming:
        """Extract the simulation timing block.

        Returns:
            The resulting :class:`PhysicsTiming`.
        """
        sim = self._data["sim"]
        return PhysicsTiming(
            decimation=self._data["decimation"],
            physics_dt=float(sim["dt"]),
            render_interval=sim["render_interval"],
        )

    def _build_spawn(self) -> SpawnProps:
        """Extract the robot spawn block.

        Returns:
            The resulting :class:`SpawnProps`.
        """
        robot = (self._data.get("scene") or {}).get("robot") or {}
        spawn = robot.get("spawn") or {}

        return SpawnProps(
            usd_path=spawn.get("usd_path"),
            articulation_props=spawn.get("articulation_props") or {},
            rigid_body_props=spawn.get("rigid_props") or {},
            joint_drive_props=spawn.get("joint_drive_props") or {},
        )
