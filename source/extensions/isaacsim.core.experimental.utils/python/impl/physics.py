# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Authoring helpers for applying/removing physics (rigid body) and collision APIs on USD prims.

These are lightweight, single-prim authoring functions (no physics tensor views or simulation
state involved), suitable for use during stage authoring (e.g. inside an ``Sdf.ChangeBlock``).
They are the engine-agnostic, in-repo replacement for ``omni.physx.scripts.utils`` helpers such as
``setCollider``, ``removeCollider``, ``setRigidBody`` and ``removePhysics``.

For batched, runtime operations over many prims, use the methods on
:py:class:`~isaacsim.core.experimental.prims.GeomPrim` and
:py:class:`~isaacsim.core.experimental.prims.RigidPrim` instead.
"""

from __future__ import annotations

import carb
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

from . import prim as prim_utils

# Mapping from the mesh-collision approximation token to the PhysX schema API to apply.
# A value of ``None`` means no additional PhysX approximation API is required for that token.
_MESH_APPROXIMATIONS = {
    UsdPhysics.Tokens.none: PhysxSchema.PhysxTriangleMeshCollisionAPI,
    UsdPhysics.Tokens.convexHull: PhysxSchema.PhysxConvexHullCollisionAPI,
    PhysxSchema.Tokens.convexDecomposition: PhysxSchema.PhysxConvexDecompositionCollisionAPI,
    UsdPhysics.Tokens.meshSimplification: PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI,
    UsdPhysics.Tokens.boundingCube: None,
    UsdPhysics.Tokens.boundingSphere: None,
    PhysxSchema.Tokens.sphereFill: PhysxSchema.PhysxSphereFillCollisionAPI,
    PhysxSchema.Tokens.sdf: PhysxSchema.PhysxSDFMeshCollisionAPI,
}

# Multiple-apply ``PhysxCookedDataAPI`` instance tokens that may be authored alongside a collider.
_COOKED_DATA_TOKENS = [
    PhysxSchema.Tokens.convexHull,
    PhysxSchema.Tokens.convexDecomposition,
    PhysxSchema.Tokens.triangleMesh,
]


def _is_part_of_rigid_body(prim: Usd.Prim) -> bool:
    current = prim
    while current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return True
        current = current.GetParent()
    return False


def apply_collision(prim: str | Usd.Prim, *, approximation: str = UsdPhysics.Tokens.none) -> None:
    """Apply collision APIs to a prim to enable collision detection.

    Backends: :guilabel:`usd`.

    This applies ``UsdPhysics.CollisionAPI`` and ``PhysxSchema.PhysxCollisionAPI``. For mesh
    (or instanceable) prims, it additionally applies ``UsdPhysics.MeshCollisionAPI`` and the
    PhysX approximation API associated with ``approximation``.

    .. note::

        This is the in-repo replacement for ``omni.physx.scripts.utils.setCollider``. The prim is
        skipped if it has an ``omni:no_collision`` attribute. Existing collision APIs are updated,
        making the operation idempotent. For a mesh that is part of a rigid body, a ``none``
        approximation is promoted to ``convexHull`` (a triangle mesh cannot be a dynamic collider).

    Args:
        prim: Prim path or prim instance.
        approximation: Mesh-collision approximation token (e.g. ``"none"``, ``"convexHull"``,
            ``"convexDecomposition"``, ``"meshSimplification"``, ``"boundingCube"``,
            ``"boundingSphere"``, ``"sdf"``). Ignored for non-mesh, non-instanceable prims.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.physics as physics_utils
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> prim = stage_utils.define_prim("/World/Collider", "Cube")
        >>> physics_utils.apply_collision(prim)
    """
    prim = prim_utils.get_prim_at_path(prim)
    # using this attribute instead of purpose=guide, so that the volumes will be easily renderable
    if prim.GetAttribute("omni:no_collision"):
        return
    is_mesh = prim.IsA(UsdGeom.Mesh)
    if is_mesh and approximation == UsdPhysics.Tokens.none and _is_part_of_rigid_body(prim):
        carb.log_warn(
            f"apply_collision: {prim.GetPath()} is a part of a rigid body. "
            "Resetting approximation shape from none (trimesh) to convexHull"
        )
        approximation = UsdPhysics.Tokens.convexHull

    collision_api = (
        UsdPhysics.CollisionAPI(prim) if prim.HasAPI(UsdPhysics.CollisionAPI) else UsdPhysics.CollisionAPI.Apply(prim)
    )
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision_api.CreateCollisionEnabledAttr().Set(True)

    if is_mesh or prim.IsInstanceable():
        api = _MESH_APPROXIMATIONS.get(approximation, 0)  # None is a valid value
        if api == 0:
            carb.log_warn(
                f"apply_collision: invalid approximation type {approximation} provided for "
                f"{prim.GetPath()}. Falling back to convexHull."
            )
            approximation = UsdPhysics.Tokens.convexHull
            api = _MESH_APPROXIMATIONS[approximation]
        for approximation_api in set(_MESH_APPROXIMATIONS.values()) - {None, api}:
            prim.RemoveAPI(approximation_api)
        if api is not None and not prim.HasAPI(api):
            api.Apply(prim)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision_api.CreateApproximationAttr().Set(approximation)


def remove_collision(prim: str | Usd.Prim) -> None:
    """Remove collision APIs from a prim to disable collision detection.

    Backends: :guilabel:`usd`.

    This removes ``UsdPhysics.CollisionAPI`` and ``PhysxSchema.PhysxCollisionAPI``. For mesh or
    instanceable prims, it additionally removes ``UsdPhysics.MeshCollisionAPI``, the PhysX
    mesh-approximation APIs, and any ``PhysxSchema.PhysxCookedDataAPI`` instances. Removing an API
    that is not applied is a no-op.

    .. note::

        This is the in-repo replacement for ``omni.physx.scripts.utils.removeCollider``.

    Args:
        prim: Prim path or prim instance.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.physics as physics_utils
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> prim = stage_utils.define_prim("/World/ColliderToRemove", "Cube")
        >>> physics_utils.apply_collision(prim)
        >>> physics_utils.remove_collision(prim)
    """
    prim = prim_utils.get_prim_at_path(prim)
    removed = prim.RemoveAPI(UsdPhysics.CollisionAPI)
    prim.RemoveAPI(PhysxSchema.PhysxCollisionAPI)
    # mirror apply_collision, which authors the mesh-collision/approximation APIs for meshes
    # *or* instanceable prims, so the same set must be removed for both to avoid stale state
    if prim.IsA(UsdGeom.Mesh) or prim.IsInstanceable():
        prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        prim.RemoveAPI(PhysxSchema.PhysxConvexHullCollisionAPI)
        prim.RemoveAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
        prim.RemoveAPI(PhysxSchema.PhysxTriangleMeshSimplificationCollisionAPI)
        prim.RemoveAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI)
        prim.RemoveAPI(PhysxSchema.PhysxSphereFillCollisionAPI)
        prim.RemoveAPI(PhysxSchema.PhysxSDFMeshCollisionAPI)
    for token in _COOKED_DATA_TOKENS:
        prim.RemoveAPI(PhysxSchema.PhysxCookedDataAPI, token)
    if not removed:
        carb.log_error(f"Failed to remove a UsdPhysics.CollisionAPI from prim {prim.GetPath()}")


def apply_rigid_body(
    prim: str | Usd.Prim,
    *,
    approximation: str = UsdPhysics.Tokens.convexHull,
    kinematic: bool = False,
) -> None:
    """Apply rigid body and collision APIs to a prim (and its descendants for an ``Xformable``).

    Backends: :guilabel:`usd`.

    This applies ``UsdPhysics.RigidBodyAPI`` and ``PhysxSchema.PhysxRigidBodyAPI`` to ``prim``, then
    applies collision APIs (see :py:func:`apply_collision`) either to each collidable descendant
    (when ``prim`` is an ``Xformable``) or to ``prim`` itself.

    .. note::

        This is the in-repo replacement for ``omni.physx.scripts.utils.setRigidBody``. The rigid
        body is skipped (with a warning) if the ``RigidBodyAPI`` is already applied.

    Args:
        prim: Prim path or prim instance.
        approximation: Mesh-collision approximation token used for the collider(s).
        kinematic: Whether the rigid body is kinematic.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.physics as physics_utils
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> prim = stage_utils.define_prim("/World/RigidBody", "Cube")
        >>> physics_utils.apply_rigid_body(prim, approximation="convexHull")
    """
    prim = prim_utils.get_prim_at_path(prim)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        carb.log_warn(f"RigidBodyAPI is already defined on {prim.GetPath()}")
        return

    rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rigid_body_api.CreateRigidBodyEnabledAttr(True)
    rigid_body_api.CreateKinematicEnabledAttr(kinematic)

    if prim.IsA(UsdGeom.Xformable):
        prim_range = iter(Usd.PrimRange(prim))
        for descendant in prim_range:
            if descendant.GetMetadata("hide_in_stage_window"):
                prim_range.PruneChildren()
                continue
            if descendant.IsA(UsdGeom.Gprim) or descendant.IsInstanceable():
                apply_collision(descendant, approximation=approximation)
    else:
        apply_collision(prim, approximation=approximation)


def remove_rigid_body(prim: str | Usd.Prim) -> None:
    """Remove the rigid body APIs from a prim.

    Backends: :guilabel:`usd`.

    This removes ``UsdPhysics.RigidBodyAPI`` and ``PhysxSchema.PhysxRigidBodyAPI``. Collision APIs
    (see :py:func:`remove_collision`) are left untouched. Removing an API that is not applied is a
    no-op.

    .. note::

        This is the in-repo replacement for ``omni.physx.scripts.utils.removePhysics``.

    Args:
        prim: Prim path or prim instance.

    Example:

    .. code-block:: python

        >>> import isaacsim.core.experimental.utils.physics as physics_utils
        >>> import isaacsim.core.experimental.utils.stage as stage_utils
        >>>
        >>> prim = stage_utils.define_prim("/World/RigidBodyToRemove", "Cube")
        >>> physics_utils.apply_rigid_body(prim)
        >>> physics_utils.remove_rigid_body(prim)
    """
    prim = prim_utils.get_prim_at_path(prim)
    removed = prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
    prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
    if not removed:
        carb.log_error(f"Failed to remove a UsdPhysics.RigidBodyAPI from prim {prim.GetPath()}")
