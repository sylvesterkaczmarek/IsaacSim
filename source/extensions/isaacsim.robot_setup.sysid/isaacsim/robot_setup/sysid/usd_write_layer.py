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

"""Small helpers for multi-physics SysID output stages."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_NEUTRAL_PHYSICS_LAYER_NAMES = {
    "physics.usd",
    "physics.usda",
    "_physics.usd",
    "_physics.usda",
}
_SOLVER_LAYER_NAMES = {
    "mujoco": {"mujoco.usd", "mujoco.usda"},
    "physx": {"physx.usd", "physx.usda"},
}


@dataclass(frozen=True)
class PhysicsWriteTargets:
    """Composition-aware edit targets for multi-physics parameter ownership."""

    neutral: object
    solver: object | None


def _layer_name(layer: object) -> str:
    """Return a normalized basename for a USD layer.

    Args:
        layer: USD layer whose identifier is inspected.

    Returns:
        Lowercase layer basename.
    """
    return str(getattr(layer, "identifier", "")).replace("\\", "/").rsplit("/", 1)[-1].lower()


def _iter_composition_nodes(root_node: object) -> Iterator[object]:
    """Yield a composition node and its descendants.

    Args:
        root_node: Root Pcp composition node.

    Yields:
        Composition nodes in depth-first order.
    """
    pending = [root_node]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(reversed(list(node.children)))


def _has_multiphysics_variants(robot: object) -> bool:
    """Return whether a robot advertises the standard multi-physics variants.

    Args:
        robot: Composed robot prim to inspect.

    Returns:
        Whether the robot has the standard multi-physics variant layout.
    """
    variant_set = robot.GetVariantSet("Physics")
    names = {str(name).lower() for name in variant_set.GetVariantNames()}
    solver_names = names & {"mujoco", "physx"}
    return bool(solver_names) and ("physics" in names or len(solver_names) > 1)


def resolve_multiphysics_write_targets(
    stage: object,
    robot_prim_path: str,
    *,
    solver_layer_name: str,
) -> PhysicsWriteTargets | None:
    """Resolve editable payload-layer targets for a composed multi-physics robot.

    Flat stages return ``None`` so callers retain their current edit target. A
    recognized multi-physics asset fails closed when its active composition
    does not expose the requested solver layer.

    Args:
        stage: Open USD stage containing the robot.
        robot_prim_path: Scene path of the composed robot prim.
        solver_layer_name: Required solver layer (``physx`` or ``mujoco``), or
            an empty value when all parameters belong to the neutral layer.

    Returns:
        Composition-aware neutral and optional solver edit targets, or ``None``
        for a flat/non-multi-physics stage.

    Raises:
        RuntimeError: If a multi-physics asset's writable layers are not active.
        ValueError: If ``solver_layer_name`` is unsupported.
    """
    from pxr import Usd

    solver_name = str(solver_layer_name).strip().lower()
    if solver_name not in ("", *_SOLVER_LAYER_NAMES):
        raise ValueError(f"Unsupported solver layer: {solver_layer_name!r}")
    robot = stage.GetPrimAtPath(robot_prim_path)
    if robot is None or not robot.IsValid():
        return None

    required_solver_names = _SOLVER_LAYER_NAMES.get(solver_name, set())
    for node in _iter_composition_nodes(robot.GetPrimIndex().rootNode):
        layers = list(node.layerStack.layers)
        neutral_layer = next((layer for layer in layers if _layer_name(layer) in _NEUTRAL_PHYSICS_LAYER_NAMES), None)
        solver_layer = next((layer for layer in layers if _layer_name(layer) in required_solver_names), None)
        if neutral_layer is None or (required_solver_names and solver_layer is None):
            continue
        neutral_target = Usd.EditTarget(neutral_layer, node)
        solver_target = Usd.EditTarget(solver_layer, node) if solver_layer is not None else None
        if neutral_target.MapToSpecPath(robot.GetPath()).isEmpty:
            continue
        if solver_target is not None and solver_target.MapToSpecPath(robot.GetPath()).isEmpty:
            continue
        return PhysicsWriteTargets(neutral=neutral_target, solver=solver_target)

    if _has_multiphysics_variants(robot):
        expected = f"{solver_name}.usd[a] and physics.usd[a]" if solver_name else "physics.usd[a]"
        raise RuntimeError(
            f"The robot is multi-physics, but its active composition does not expose {expected}. "
            "Select the matching Physics variant before applying SysID parameters."
        )
    return None


def create_physics_override_stage(
    source_stage_path: str | os.PathLike[str],
    output_stage_path: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    solver_layer_name: str = "",
) -> tuple[Path, Path | None]:
    """Create output-owned neutral and active-solver layers above the source stage.

    Args:
        source_stage_path: Source USD stage to compose into the output.
        output_stage_path: Destination path for the composed output stage.
        overwrite: Whether existing output layers may be replaced.
        solver_layer_name: Optional active solver layer name (``physx`` or ``mujoco``).

    Returns:
        Paths to the neutral physics layer and optional solver layer.
    """
    from pxr import Sdf

    source_path = Path(source_stage_path).expanduser().resolve()
    output_path = Path(output_stage_path).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("Input and output USD paths must be different.")
    if not source_path.is_file():
        raise FileNotFoundError(f"Input USD does not exist: {source_path}")

    solver_name = str(solver_layer_name).strip().lower()
    if solver_name not in ("", "physx", "mujoco"):
        raise ValueError(f"Unsupported output solver layer: {solver_layer_name!r}")
    physics_dir = output_path.parent / f"{output_path.stem}_layers" / "Physics"
    physics_path = physics_dir / "physics.usda"
    solver_path = physics_dir / f"{solver_name}.usda" if solver_name else None
    output_layers = [output_path, physics_path]
    if solver_path is not None:
        output_layers.append(solver_path)
    existing = [path for path in output_layers if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Output USD layer already exists: {existing[0]}")
    physics_dir.mkdir(parents=True, exist_ok=True)

    def _new_or_cleared_layer(path: Path) -> Sdf.Layer:
        if path.exists():
            layer = Sdf.Layer.FindOrOpen(str(path))
            if layer is None:
                raise RuntimeError(f"Could not open output USD layer: {path}")
            layer.Clear()
            return layer
        layer = Sdf.Layer.CreateNew(str(path))
        if layer is None:
            raise RuntimeError(f"Could not create output USD layer: {path}")
        return layer

    physics_layer = _new_or_cleared_layer(physics_path)
    solver_layer = _new_or_cleared_layer(solver_path) if solver_path is not None else None
    root_layer = _new_or_cleared_layer(output_path)
    active_layer_path = solver_path or physics_path
    active_asset_path = os.path.relpath(active_layer_path, output_path.parent).replace("\\", "/")
    source_asset_path = os.path.relpath(source_path, output_path.parent).replace("\\", "/")
    root_layer.subLayerPaths = [active_asset_path, source_asset_path]
    if solver_layer is not None:
        solver_layer.subLayerPaths = [os.path.relpath(physics_path, solver_path.parent).replace("\\", "/")]

    source_layer = Sdf.Layer.FindOrOpen(str(source_path))
    if source_layer is not None and source_layer.defaultPrim:
        root_layer.defaultPrim = source_layer.defaultPrim
    if source_layer is not None:
        # Carry stage-level metadata (up-axis, units) from the source root layer.
        # These are read from the ROOT layer only (not composed from sublayers), so a
        # freshly created output root would otherwise fall back to USD's Y-up / 1.0
        # defaults and render the whole stage rotated on its side.
        src_root = source_layer.pseudoRoot
        dst_root = root_layer.pseudoRoot
        for _key in ("upAxis", "metersPerUnit", "kilogramsPerUnit"):
            if src_root.HasInfo(_key):
                dst_root.SetInfo(_key, src_root.GetInfo(_key))
    if not physics_layer.Save():
        raise RuntimeError(f"Could not save output physics layer: {physics_path}")
    if solver_layer is not None and not solver_layer.Save():
        raise RuntimeError(f"Could not save output solver layer: {solver_path}")
    if not root_layer.Save():
        raise RuntimeError(f"Could not save composed output stage: {output_path}")
    return physics_path, solver_path


def is_multiphysics_stage(stage: object, robot_prim_path: str) -> bool:
    """Return whether the robot composes neutral and solver-specific physics layers.

    Args:
        stage: Open USD stage to inspect.
        robot_prim_path: Path of the robot prim whose composition is inspected.

    Returns:
        Whether neutral and solver-specific physics layers are composed.
    """
    from pxr import Usd

    robot = stage.GetPrimAtPath(robot_prim_path)
    if robot is None or not robot.IsValid():
        return False
    has_neutral = False
    has_solver = False
    for prim in Usd.PrimRange(robot):
        for spec in prim.GetPrimStack():
            name = _layer_name(spec.layer)
            has_neutral = has_neutral or name in _NEUTRAL_PHYSICS_LAYER_NAMES
            has_solver = has_solver or any(name in names for names in _SOLVER_LAYER_NAMES.values())
        if has_neutral and has_solver:
            return True
    return False
