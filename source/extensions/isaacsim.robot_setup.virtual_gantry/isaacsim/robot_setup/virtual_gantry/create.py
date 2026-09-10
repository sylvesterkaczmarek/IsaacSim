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

"""Authoring helper for IsaacVirtualGantry prims."""

from __future__ import annotations

import logging
from typing import Any, Optional

import isaacsim.robot_setup.virtual_gantry_schema as gantry_schema
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

_LOGGER = logging.getLogger(__name__)
_ATTACH_KEYS = ("torso", "pelvis", "base", "trunk")


def create_virtual_gantry(
    stage: Usd.Stage,
    parent_path: str = "/World",
    wire_hint: str = "",
    anchor_z: float = 2.0,
) -> Usd.Prim:
    """Author an ``IsaacVirtualGantry`` prim and wire it to a robot.

    The prim is ``Xform``-derived; its world transform is the rope anchor. When a
    robot articulation is found it is auto-wired (``attachBody`` -> a torso/pelvis
    link, ``articulation`` -> the articulation root) and the anchor is placed at
    world height ``anchor_z`` above it, so the gantry works immediately with no
    manual relationship editing. Edit the relationships in the property panel to
    retarget.

    Args:
        stage: The USD stage to create the prim in.
        parent_path: Path the gantry prim is created under (e.g. ``/World``).
        wire_hint: A prim path (e.g. the selected robot) used to prefer one
            articulation when several exist; empty selects the first found.
        anchor_z: Anchor world height (Z), in meters.

    Returns:
        The created Virtual Gantry prim.
    """
    parent = (parent_path or "/World").rstrip("/")
    gantry_path = omni.usd.get_stage_next_free_path(stage, parent + "/VirtualGantry", False)
    prim = gantry_schema.CreateVirtualGantry(stage, gantry_path)

    art_path, link_path = _find_articulation_and_link(stage, wire_hint)
    if art_path is not None and link_path is not None:
        prim.GetRelationship("isaac:gantry:attachBody").SetTargets([link_path])
        prim.GetRelationship("isaac:gantry:articulation").SetTargets([art_path])
        lp = (
            UsdGeom.XformCache(Usd.TimeCode.Default())
            .GetLocalToWorldTransform(stage.GetPrimAtPath(link_path))
            .ExtractTranslation()
        )
        _set_anchor_world(stage, prim, (float(lp[0]), float(lp[1]), anchor_z))
        _LOGGER.info(f"[virtual-gantry] created {gantry_path}, wired to {link_path} ({art_path})")
    else:
        _set_anchor_world(stage, prim, (0.0, 0.0, anchor_z))
        _LOGGER.warning(
            f"[virtual-gantry] created {gantry_path} but found no articulation to wire; "
            f"set its Attach Body / Articulation relationships manually",
        )
    return prim


def _rigid_bodies_under(prim: Any) -> list:
    """Return every prim in ``prim``'s subtree carrying ``RigidBodyAPI``.

    Args:
        prim: The gantry prim to read from.

    Returns:
        The resulting list.
    """
    return [p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def _find_articulation_and_link(stage: Usd.Stage, hint: str = "") -> tuple[Optional[Any], Optional[Any]]:
    """Return ``(articulation_root_path, attach_link_path)`` or ``(None, None)``.

        Prefers an articulation whose path is related to ``hint`` (the selection),
        and a rigid-body link named like a torso/pelvis as the attach point.

    Args:
        stage: The USD stage to operate on.
        hint: Prim path used to disambiguate which robot to wire.

    Returns:
        The resulting tuple.
    """
    roots = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not roots:
        return None, None
    root = roots[0]
    if hint:
        hint_path = Sdf.Path(hint) if Sdf.Path.IsValidPathString(hint) else None
        for r in roots:
            rp = r.GetPath()
            # Proper path-prefix comparison: a plain string startswith would
            # match "/World/g1" against "/World/g1_a".
            if hint_path is not None and (rp.HasPrefix(hint_path) or hint_path.HasPrefix(rp)):
                root = r
                break

    # Attach links are often siblings of the articulation root rather than its
    # descendants, so the search may need to widen to the root's parent. Only do
    # that when the parent holds this robot alone: on a multi-robot stage where
    # each robot carries ArticulationRootAPI on its top-level Xform, the parent
    # is the shared scope (e.g. /World) and widening would pick another robot's
    # torso.
    rigid = _rigid_bodies_under(root)
    if not rigid:
        parent = root.GetParent()
        if parent is not None and parent.IsValid():
            roots_under_parent = [p for p in Usd.PrimRange(parent) if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
            if len(roots_under_parent) == 1:
                rigid = _rigid_bodies_under(parent)
    link = None
    for key in _ATTACH_KEYS:
        for p in rigid:
            if key in p.GetName().lower():
                link = p
                break
        if link is not None:
            break
    if link is None:
        link = rigid[0] if rigid else root
    return root.GetPath(), link.GetPath()


def _set_anchor_world(stage: Usd.Stage, prim: Any, world_xyz: tuple[float, float, float]) -> None:
    """Set the prim's translate op so it sits at ``world_xyz`` (parent-aware).

    Args:
        stage: The USD stage to operate on.
        prim: The gantry prim to read from.
        world_xyz: Anchor position in world space.
    """
    parent = prim.GetParent()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    if parent is not None and parent.IsValid():
        local = cache.GetLocalToWorldTransform(parent).GetInverse().Transform(Gf.Vec3d(*world_xyz))
    else:
        local = Gf.Vec3d(*world_xyz)
    xform = UsdGeom.Xformable(prim)
    op = (
        next(
            (o for o in xform.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        or xform.AddTranslateOp()
    )
    op.Set(Gf.Vec3d(local))
