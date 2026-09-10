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

"""Verifies the physics authoring helpers for applying/removing collision and rigid body APIs on USD prims.

Covers collider application on shapes and meshes (including the mesh approximation APIs), the
``setCollider`` guards (``omni:no_collision`` skip, already-applied skip, mesh-in-rigid-body
``none``->``convexHull`` promotion), rigid body application (including the kinematic flag and the
``Xformable`` subtree collider authoring), and the removal helpers.
"""

import isaacsim.core.experimental.utils.physics as physics_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.kit.test
from pxr import PhysxSchema, Sdf, UsdPhysics


class TestPhysics(omni.kit.test.AsyncTestCase):
    """Test physics authoring helpers."""

    async def setUp(self) -> None:
        """Method called to prepare the test fixture."""
        super().setUp()
        await stage_utils.create_new_stage_async()

    async def tearDown(self) -> None:
        """Method called immediately after the test method has been called."""
        super().tearDown()

    # --------------------------------------------------------------------

    async def test_apply_collision_shape(self) -> None:
        """Applying collision to a (non-mesh) shape applies only the collision APIs."""
        prim = stage_utils.define_prim("/World/Cube", "Cube")
        physics_utils.apply_collision(prim)
        self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
        self.assertTrue(prim.HasAPI(PhysxSchema.PhysxCollisionAPI))
        self.assertTrue(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        # a non-mesh, non-instanceable prim should not get a mesh-collision approximation
        self.assertFalse(prim.HasAPI(UsdPhysics.MeshCollisionAPI))

    async def test_apply_collision_mesh_approximation(self) -> None:
        """Applying collision to a mesh authors the mesh-collision + approximation APIs."""
        prim = stage_utils.define_prim("/World/Mesh", "Mesh")
        physics_utils.apply_collision(prim, approximation="convexHull")
        self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
        self.assertTrue(prim.HasAPI(PhysxSchema.PhysxCollisionAPI))
        self.assertTrue(prim.HasAPI(UsdPhysics.MeshCollisionAPI))
        self.assertTrue(prim.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI))
        self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), UsdPhysics.Tokens.convexHull)

    async def test_apply_collision_updates_existing_mesh_collider(self) -> None:
        """Applying collision updates and enables an existing mesh collider."""
        prim = stage_utils.define_prim("/World/ExistingMesh", "Mesh")
        collision_api = UsdPhysics.CollisionAPI.Apply(prim)
        collision_api.CreateCollisionEnabledAttr().Set(False)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr().Set(UsdPhysics.Tokens.none)
        PhysxSchema.PhysxTriangleMeshCollisionAPI.Apply(prim)

        physics_utils.apply_collision(prim, approximation=UsdPhysics.Tokens.convexHull)

        self.assertTrue(collision_api.GetCollisionEnabledAttr().Get())
        self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), UsdPhysics.Tokens.convexHull)
        self.assertTrue(prim.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI))
        self.assertFalse(prim.HasAPI(PhysxSchema.PhysxTriangleMeshCollisionAPI))

    async def test_apply_collision_no_collision_attr_guard(self) -> None:
        """A prim with an ``omni:no_collision`` attribute is skipped."""
        prim = stage_utils.define_prim("/World/Cube", "Cube")
        prim.CreateAttribute("omni:no_collision", Sdf.ValueTypeNames.Bool).Set(True)
        physics_utils.apply_collision(prim)
        self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))

    async def test_apply_collision_mesh_in_rigid_body_promotes_to_convex_hull(self) -> None:
        """A trimesh (``none``) collider that is part of a rigid body is promoted to ``convexHull``."""
        body = stage_utils.define_prim("/World/Body", "Xform")
        UsdPhysics.RigidBodyAPI.Apply(body)
        mesh = stage_utils.define_prim("/World/Body/Mesh", "Mesh")
        physics_utils.apply_collision(mesh, approximation="none")
        self.assertEqual(UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get(), UsdPhysics.Tokens.convexHull)
        self.assertTrue(mesh.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI))

    async def test_remove_collision(self) -> None:
        """Removing collision strips the collision and mesh-approximation APIs."""
        prim = stage_utils.define_prim("/World/Mesh", "Mesh")
        physics_utils.apply_collision(prim, approximation="convexHull")
        physics_utils.remove_collision(prim)
        self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))
        self.assertFalse(prim.HasAPI(PhysxSchema.PhysxCollisionAPI))
        self.assertFalse(prim.HasAPI(UsdPhysics.MeshCollisionAPI))
        self.assertFalse(prim.HasAPI(PhysxSchema.PhysxConvexHullCollisionAPI))

    async def test_remove_collision_sphere_fill_and_sdf(self) -> None:
        """Removing collision strips the sphere-fill and SDF mesh-approximation APIs."""
        for approximation, approximation_api in [
            (PhysxSchema.Tokens.sphereFill, PhysxSchema.PhysxSphereFillCollisionAPI),
            (PhysxSchema.Tokens.sdf, PhysxSchema.PhysxSDFMeshCollisionAPI),
        ]:
            prim = stage_utils.define_prim(f"/World/Mesh_{approximation}", "Mesh")
            physics_utils.apply_collision(prim, approximation=approximation)
            self.assertTrue(prim.HasAPI(approximation_api))
            physics_utils.remove_collision(prim)
            self.assertFalse(prim.HasAPI(approximation_api))

    async def test_apply_rigid_body(self) -> None:
        """Applying a rigid body authors the rigid body APIs plus collision on the subtree."""
        prim = stage_utils.define_prim("/World/Mesh", "Mesh")
        physics_utils.apply_rigid_body(prim, approximation="convexHull")
        self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertTrue(prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI))
        self.assertTrue(UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get())
        self.assertFalse(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
        # a Mesh is an Xformable, so the subtree branch authors the collider on the mesh itself
        self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), UsdPhysics.Tokens.convexHull)

    async def test_apply_rigid_body_kinematic(self) -> None:
        """The ``kinematic`` flag is authored on the rigid body."""
        prim = stage_utils.define_prim("/World/Mesh", "Mesh")
        physics_utils.apply_rigid_body(prim, kinematic=True)
        self.assertTrue(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())

    async def test_apply_rigid_body_xform_subtree(self) -> None:
        """For an ``Xform`` parent, colliders are authored on the collidable descendants."""
        root = stage_utils.define_prim("/World/Root", "Xform")
        child = stage_utils.define_prim("/World/Root/Cube", "Cube")
        physics_utils.apply_rigid_body(root)
        # rigid body on the root, collider on the child Gprim
        self.assertTrue(root.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertTrue(child.HasAPI(UsdPhysics.CollisionAPI))

    async def test_remove_rigid_body(self) -> None:
        """Removing the rigid body strips the rigid body APIs but leaves collision untouched."""
        prim = stage_utils.define_prim("/World/Mesh", "Mesh")
        physics_utils.apply_rigid_body(prim, approximation="convexHull")
        physics_utils.remove_rigid_body(prim)
        self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        self.assertFalse(prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI))
        # collision APIs are not removed by remove_rigid_body
        self.assertTrue(prim.HasAPI(UsdPhysics.CollisionAPI))

    async def test_helpers_accept_prim_path_string(self) -> None:
        """The helpers accept a prim path string as well as a prim instance."""
        stage_utils.define_prim("/World/Cube", "Cube")
        physics_utils.apply_rigid_body("/World/Cube")
        prim = stage_utils.get_current_stage(backend="usd").GetPrimAtPath("/World/Cube")
        self.assertTrue(prim.HasAPI(UsdPhysics.RigidBodyAPI))
        physics_utils.remove_rigid_body("/World/Cube")
        self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
