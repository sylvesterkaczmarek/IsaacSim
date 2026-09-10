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

# Import extension python module we are testing with absolute import path, as if we are an external user (i.e. a different extension)

"""Verify synchronization from USD obstacles into a planning-world interface.

The tests cover empty and invalid bindings, collision API filtering, initial
property extraction for all supported primitive shapes, mesh and
triangulated-mesh insertion, oriented bounding boxes, transform synchronization,
and ancestor-scale validation.
"""

import numpy as np
import omni.kit.test
from isaacsim.core.experimental.objects import (
    Capsule,
    Cone,
    Cube,
    Cylinder,
    Mesh,
    Plane,
    Sphere,
)
from isaacsim.core.experimental.prims import GeomPrim
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.core.experimental.utils.stage import create_new_stage_async
from isaacsim.robot_motion.experimental.motion_generation import (
    ObstacleConfiguration,
    ObstacleRepresentation,
    ObstacleStrategy,
    TrackableApi,
    WorldBinding,
)

# a world interface which exactly mirrors the inputs it is given. This will be used to test the WorldBinding class.
from .mirror_world_interface import MirrorOrientedBoundingBox, MirrorTriangulatedMesh, MirrorWorldInterface


# Having a test class derived from omni.kit.test.AsyncTestCase declared on the root of the module will make it auto-discoverable by omni.kit.test
class TestWorldBinding(omni.kit.test.AsyncTestCase):
    """Test class for validating WorldBinding functionality.

    This class provides comprehensive test coverage for the WorldBinding class, which initializes USD stage
    objects in planning world interfaces and synchronizes their transforms.

    The test suite validates:

    - Basic WorldBinding initialization and error handling
    - Initial obstacle-property extraction and transform synchronization
    - Collision API validation and requirements
    - Mesh handling for both regular and triangulated representations
    - Oriented bounding box collision representations
    - Scene validation for ancestor scaling constraints

    Tests use a MirrorWorldInterface that mirrors input data, enabling validation of correct data
    transfer from USD stage to planning world objects.
    """

    # Before running each test
    async def setUp(self) -> None:
        """Set up test environment before each test."""
        await create_new_stage_async()

        await app_utils.update_app_async()

    # After running each test
    async def tearDown(self) -> None:
        """Clean up test environment after each test."""
        if app_utils.is_playing():
            app_utils.stop()

        # Clean up physics callbacks if any
        if hasattr(self, "sample") and self.sample:
            self.sample.physics_cleanup()

        await app_utils.update_app_async()

    async def test_empty_world_binding(self) -> None:
        """Test creating WorldBinding with empty world interface and handling non-existent prims."""
        # can create a WorldBinding, and initialize it with an empty world interface and no tracked prims:
        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=ObstacleStrategy(),
            tracked_prims=[],
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )
        world_binding.initialize()
        await app_utils.update_app_async()

        # Transform synchronization is a no-op for an empty binding.
        world_binding.synchronize_transforms()

        # create a new world binding, with a tracked prim which does not exist in the stage:
        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=ObstacleStrategy(),
            tracked_prims=["/World/Sphere"],
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )

        # initialize should raise a RuntimeError, since the prim does not exist in the stage.
        self.assertRaises(RuntimeError, world_binding.initialize)

        # world binding should not be initialized, so transform synchronization should raise a RuntimeError.
        self.assertFalse(world_binding._initialized)
        self.assertRaises(RuntimeError, world_binding.synchronize_transforms)

    async def test_world_binding_rejects_non_collision_api(self) -> None:
        """Test that WorldBinding rejects non-collision APIs and validates API selection."""
        # Collision Enables have to use a collision API:
        self.assertRaises(
            ValueError,
            WorldBinding,
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=ObstacleStrategy(),
            tracked_prims=[],
            tracked_collision_api=TrackableApi.PHYSICS_RIGID_BODY,
        )

        # Empty tracked_prims still validates the API selection.
        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=ObstacleStrategy(),
            tracked_prims=[],
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )
        world_binding.initialize()
        self.assertTrue(world_binding._initialized)

        # Motion generation collision API should also be accepted
        world_binding_motion: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=ObstacleStrategy(),
            tracked_prims=[],
            tracked_collision_api=TrackableApi.MOTION_GENERATION_COLLISION,
        )
        world_binding_motion.initialize()
        self.assertTrue(world_binding_motion._initialized)

    async def test_initialization_and_transform_updates(self) -> None:
        """Test initial obstacle population and runtime transform synchronization."""
        # create a couple of prims to use:
        stage_sphere = Sphere(
            paths="/World/Sphere",
            radii=0.05,
            positions=[1.0, 2.0, 3.0],
            orientations=[1.0, 0.0, 0.0, 0.0],
            scales=[1.0, 1.0, 1.0],
        )
        stage_cube = Cube(
            paths="/World/Cube",
            sizes=0.15,
            positions=[-1.0, -2.0, -3.0],
            orientations=[0.0, 1.0, 0.0, 0.0],
            scales=[2.0, 2.0, 2.0],
        )

        sphere_geom = GeomPrim(
            "/World/Sphere",
        )

        cube_geom = GeomPrim("/World/Cube")

        # should be able to add these prims to our world binding:
        obstacle_strategy = ObstacleStrategy()
        obstacle_strategy.set_default_safety_tolerance(0.06)
        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            tracked_prims=["/World/Sphere", "/World/Cube"],
            obstacle_strategy=obstacle_strategy,
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )

        await app_utils.update_app_async()
        # Trying to initialize will raise a RuntimeError, since the prims do not have the CollisionAPI applied.
        self.assertRaises(RuntimeError, world_binding.initialize)

        # apply the collision APIs to the prims:
        sphere_geom.apply_collision_apis()
        sphere_geom.set_enabled_collisions(True)
        cube_geom.apply_collision_apis()
        cube_geom.set_enabled_collisions(True)

        await app_utils.update_app_async()

        # initialize should now succeed:
        world_binding.initialize()

        # If I get the world now, there should be two objects
        # which should have the most up to date transforms of the
        # sphere and cube.
        # There is a sphere object which mirrors the sphere of the
        # stage:
        world_interface = world_binding.get_world_interface()
        planning_world_sphere = world_interface.collision_objects["/World/Sphere"]
        planning_world_cube = world_interface.collision_objects["/World/Cube"]
        self.assertIsNotNone(planning_world_sphere)
        self.assertAlmostEqual(planning_world_sphere.radius, 0.05)
        self.assertAlmostEqual(planning_world_sphere.safety_tolerance, 0.06)
        self.assertTrue(np.isclose(planning_world_sphere.pose[0], [1.0, 2.0, 3.0]).all())
        self.assertTrue(np.isclose(planning_world_sphere.pose[1], [1.0, 0.0, 0.0, 0.0]).all())

        # The cube object should have the most up to date transform of the cube:
        self.assertIsNotNone(planning_world_cube)
        self.assertAlmostEqual(planning_world_cube.size, 0.15)
        self.assertAlmostEqual(planning_world_cube.safety_tolerance, 0.06)
        self.assertTrue(np.isclose(planning_world_cube.pose[0], [-1.0, -2.0, -3.0]).all())
        self.assertTrue(np.isclose(planning_world_cube.pose[1], [0.0, 1.0, 0.0, 0.0]).all())

        # Moving the sphere and cube should update the binding after transform synchronization.
        stage_sphere.set_world_poses(
            positions=[4.0, 5.0, 6.0],
            orientations=[0.0, 0.0, 1.0, 0.0],
        )

        stage_cube.set_world_poses(
            positions=[-4.0, -5.0, -6.0],
            orientations=[0.0, 0.0, 0.0, 1.0],
        )
        # update the sim with the new poses:
        await app_utils.update_app_async()

        # before synchronizing, the planning world parameters will not match the stage parameters:
        self.assertFalse(np.isclose(planning_world_sphere.pose[0], [4.0, 5.0, 6.0]).all())
        self.assertFalse(np.isclose(planning_world_cube.pose[0], [-4.0, -5.0, -6.0]).all())
        self.assertFalse(np.isclose(planning_world_sphere.pose[1], [0.0, 0.0, 1.0, 0.0]).all())
        self.assertFalse(np.isclose(planning_world_cube.pose[1], [0.0, 0.0, 0.0, 1.0]).all())

        world_binding.synchronize_transforms()

        # The sphere object should have the most up to date transform of the sphere:
        self.assertIsNotNone(planning_world_sphere)
        self.assertTrue(np.isclose(planning_world_sphere.pose[0], [4.0, 5.0, 6.0]).all())
        self.assertTrue(np.isclose(planning_world_sphere.pose[1], [0.0, 0.0, 1.0, 0.0]).all())

        # The cube object should have the most up to date transform of the cube:
        self.assertIsNotNone(planning_world_cube)
        self.assertTrue(np.isclose(planning_world_cube.pose[0], [-4.0, -5.0, -6.0]).all())
        self.assertTrue(np.isclose(planning_world_cube.pose[1], [0.0, 0.0, 0.0, 1.0]).all())

        # Runtime property changes are intentionally not synchronized.
        sphere_geom.set_enabled_collisions(enabled=False)
        stage_sphere.set_local_scales([2.0, 3.0, 4.0])
        stage_cube.set_local_scales([3.0, 4.0, 5.0])
        await app_utils.update_app_async()
        world_binding.synchronize_transforms()
        self.assertTrue(planning_world_sphere.enabled)
        self.assertTrue(planning_world_cube.enabled)
        self.assertTrue(np.isclose(planning_world_sphere.scale, [1.0, 1.0, 1.0]).all())
        self.assertTrue(np.isclose(planning_world_cube.scale, [2.0, 2.0, 2.0]).all())

    async def test_initializes_all_primitive_shape_types(self) -> None:
        """Test initial property extraction for primitive shape types."""
        primitive_paths = ["/World/Cone", "/World/Plane", "/World/Capsule", "/World/Cylinder"]

        Cone(paths=primitive_paths[0], radii=0.1, heights=0.2, axes="X")
        Plane(paths=primitive_paths[1], lengths=1.0, widths=2.0, axes="Y")
        Capsule(paths=primitive_paths[2], radii=0.3, heights=0.4, axes="Z")
        Cylinder(paths=primitive_paths[3], radii=0.5, heights=0.6, axes="X")
        GeomPrim(paths=primitive_paths, apply_collision_apis=True)
        await app_utils.update_app_async()

        # Initialize one binding containing all primitive shape types:
        obstacle_strategy = ObstacleStrategy()
        obstacle_strategy.set_default_safety_tolerance(0.07)
        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=obstacle_strategy,
            tracked_prims=primitive_paths,
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )
        world_binding.initialize()
        collision_objects = world_binding.get_world_interface().collision_objects

        # Verify that each shape is inserted with its authored geometry:
        planning_world_cone = collision_objects["/World/Cone"]
        self.assertAlmostEqual(planning_world_cone.radius, 0.1)
        self.assertAlmostEqual(planning_world_cone.length, 0.2)
        self.assertEqual(planning_world_cone.axis, "X")

        planning_world_plane = collision_objects["/World/Plane"]
        self.assertAlmostEqual(planning_world_plane.length, 1.0)
        self.assertAlmostEqual(planning_world_plane.width, 2.0)
        self.assertEqual(planning_world_plane.axis, "Y")

        planning_world_capsule = collision_objects["/World/Capsule"]
        self.assertAlmostEqual(planning_world_capsule.radius, 0.3)
        self.assertAlmostEqual(planning_world_capsule.length, 0.4)
        self.assertEqual(planning_world_capsule.axis, "Z")

        planning_world_cylinder = collision_objects["/World/Cylinder"]
        self.assertAlmostEqual(planning_world_cylinder.radius, 0.5)
        self.assertAlmostEqual(planning_world_cylinder.length, 0.6)
        self.assertEqual(planning_world_cylinder.axis, "X")

        # Verify common collision properties for every inserted shape:
        for prim_path in primitive_paths:
            with self.subTest(prim_path=prim_path):
                collision_object = collision_objects[prim_path]
                self.assertAlmostEqual(collision_object.safety_tolerance, 0.07)
                self.assertTrue(collision_object.enabled)

    async def test_add_mesh(self) -> None:
        """Test that mesh objects are correctly added and tracked in WorldBinding."""
        stage_mesh = Mesh(
            paths="/World/Mesh",
            positions=[1.0, 2.0, 3.0],
            orientations=[1.0, 0.0, 0.0, 0.0],
            scales=[1.0, 1.0, 1.0],
        )

        mesh_points = [
            [
                (-0.5, -0.5, -0.5),
                (0.5, -0.5, -0.5),
                (0.5, 0.5, -0.5),
                (-0.5, 0.5, -0.5),
                (-0.5, -0.5, 0.5),
                (0.5, -0.5, 0.5),
                (0.5, 0.5, 0.5),
                (-0.5, 0.5, 0.5),
            ]
        ]
        mesh_face_counts = [[4, 4, 4, 4, 4, 4]]
        mesh_face_indices = [
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                0,
                4,
                7,
                3,
                1,
                5,
                6,
                2,
                0,
                1,
                5,
                4,
                3,
                2,
                6,
                7,
            ]
        ]
        mesh_normals = [
            [
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        ]
        stage_mesh.set_points(mesh_points)
        stage_mesh.set_face_specs(vertex_indices=mesh_face_indices, vertex_counts=mesh_face_counts)
        stage_mesh.set_normals(mesh_normals)

        mesh_geom = GeomPrim("/World/Mesh", apply_collision_apis=True)
        mesh_geom.set_enabled_collisions(True)

        await app_utils.update_app_async()

        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=ObstacleStrategy(),
            tracked_prims=["/World/Mesh"],
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )
        world_binding.initialize()
        planning_world_mesh = world_binding.get_world_interface().collision_objects["/World/Mesh"]

        self.assertIsNotNone(planning_world_mesh)
        self.assertTrue(np.allclose(planning_world_mesh.points.numpy(), np.array(mesh_points[0], dtype=np.float32)))
        self.assertTrue(np.allclose(planning_world_mesh.normals.numpy(), np.array(mesh_normals[0], dtype=np.float32)))
        self.assertTrue(
            np.allclose(planning_world_mesh.face_vertex_indices.numpy(), np.array(mesh_face_indices[0], dtype=np.int32))
        )
        self.assertTrue(
            np.allclose(planning_world_mesh.face_vertex_counts.numpy(), np.array(mesh_face_counts[0], dtype=np.int32))
        )
        self.assertTrue(np.isclose(planning_world_mesh.pose[0], [1.0, 2.0, 3.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.pose[1], [1.0, 0.0, 0.0, 0.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.scale, [1.0, 1.0, 1.0]).all())
        self.assertTrue(planning_world_mesh.enabled)

        stage_mesh.set_world_poses(
            positions=[-1.0, -2.0, -3.0],
            orientations=[0.0, 1.0, 0.0, 0.0],
        )
        stage_mesh.set_local_scales([2.0, 3.0, 4.0])
        mesh_geom.set_enabled_collisions(False)
        await app_utils.update_app_async()

        self.assertTrue(np.isclose(planning_world_mesh.pose[0], [1.0, 2.0, 3.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.pose[1], [1.0, 0.0, 0.0, 0.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.scale, [1.0, 1.0, 1.0]).all())
        self.assertTrue(planning_world_mesh.enabled)

        # Only transforms are synchronized; enabled state and scale retain their initialized values.
        world_binding.synchronize_transforms()
        self.assertTrue(np.isclose(planning_world_mesh.pose[0], [-1.0, -2.0, -3.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.pose[1], [0.0, 1.0, 0.0, 0.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.scale, [1.0, 1.0, 1.0]).all())
        self.assertTrue(planning_world_mesh.enabled)

    async def test_add_triangulated_mesh(self) -> None:
        """Test that triangulated mesh representation is correctly created and tracked."""
        stage_mesh = Mesh(
            paths="/World/Mesh",
            positions=[0.0, 0.0, 0.0],
            orientations=[1.0, 0.0, 0.0, 0.0],
            scales=[1.0, 1.0, 1.0],
        )

        mesh_points = [
            [
                (-0.5, -0.5, -0.5),
                (0.5, -0.5, -0.5),
                (0.5, 0.5, -0.5),
                (-0.5, 0.5, -0.5),
                (-0.5, -0.5, 0.5),
                (0.5, -0.5, 0.5),
                (0.5, 0.5, 0.5),
                (-0.5, 0.5, 0.5),
            ]
        ]
        mesh_face_counts = [[4, 4, 4, 4, 4, 4]]
        mesh_face_indices = [
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                0,
                4,
                7,
                3,
                1,
                5,
                6,
                2,
                0,
                1,
                5,
                4,
                3,
                2,
                6,
                7,
            ]
        ]
        stage_mesh.set_points(mesh_points)
        stage_mesh.set_face_specs(vertex_indices=mesh_face_indices, vertex_counts=mesh_face_counts)

        mesh_geom = GeomPrim("/World/Mesh", apply_collision_apis=True)
        mesh_geom.set_enabled_collisions(True)

        await app_utils.update_app_async()

        obstacle_strategy = ObstacleStrategy()
        obstacle_strategy.set_default_configuration(
            Mesh,
            ObstacleConfiguration(
                representation=ObstacleRepresentation.TRIANGULATED_MESH,
                safety_tolerance=0.0,
            ),
        )

        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=obstacle_strategy,
            tracked_prims=["/World/Mesh"],
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )
        world_binding.initialize()
        planning_world_mesh = world_binding.get_world_interface().collision_objects["/World/Mesh"]

        self.assertIsNotNone(planning_world_mesh)
        # The triangulator output depends on PhysX/mesh cooking, so only verify points.
        self.assertTrue(np.allclose(planning_world_mesh.points.numpy(), np.array(mesh_points[0], dtype=np.float32)))

        # Verify that the object is of type MirrorTriangulatedMesh:
        self.assertIsInstance(planning_world_mesh, MirrorTriangulatedMesh)
        self.assertIsNotNone(planning_world_mesh.face_vertex_indices)
        self.assertIsNotNone(planning_world_mesh.scale)
        self.assertIsNotNone(planning_world_mesh.safety_tolerance)
        self.assertIsNotNone(planning_world_mesh.pose)
        self.assertIsNotNone(planning_world_mesh.enabled)

        mesh_geom.set_enabled_collisions(False)
        stage_mesh.set_world_poses(
            positions=[-1.0, -2.0, -3.0],
            orientations=[0.0, 1.0, 0.0, 0.0],
        )
        stage_mesh.set_local_scales([2.0, 3.0, 4.0])
        await app_utils.update_app_async()

        self.assertTrue(planning_world_mesh.enabled)
        self.assertTrue(np.isclose(planning_world_mesh.pose[0], [0.0, 0.0, 0.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.pose[1], [1.0, 0.0, 0.0, 0.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.scale, [1.0, 1.0, 1.0]).all())

        # Only transforms are synchronized; enabled state and scale retain their initialized values.
        world_binding.synchronize_transforms()
        self.assertTrue(planning_world_mesh.enabled)
        self.assertTrue(np.isclose(planning_world_mesh.pose[0], [-1.0, -2.0, -3.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.pose[1], [0.0, 1.0, 0.0, 0.0]).all())
        self.assertTrue(np.isclose(planning_world_mesh.scale, [1.0, 1.0, 1.0]).all())

    async def test_add_oriented_bounding_box(self) -> None:
        """Test that oriented bounding box representation is correctly created and tracked."""
        sphere_path = "/World/Sphere"
        stage_sphere = Sphere(
            paths=sphere_path,
            radii=0.5,
            positions=[0.0, 1.0, 2.0],
            orientations=[1.0, 0.0, 0.0, 0.0],
            scales=[2.0, 1.0, 1.0],
        )

        cube_geom = GeomPrim(sphere_path, apply_collision_apis=True)
        cube_geom.set_enabled_collisions(True)

        await app_utils.update_app_async()

        obstacle_strategy = ObstacleStrategy()
        obstacle_strategy.set_default_configuration(
            Sphere,
            ObstacleConfiguration(
                representation=ObstacleRepresentation.OBB,
                safety_tolerance=0.01,
            ),
        )

        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=obstacle_strategy,
            tracked_prims=[sphere_path],
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )
        world_binding.initialize()
        planning_world_obb = world_binding.get_world_interface().collision_objects[sphere_path]

        self.assertIsNotNone(planning_world_obb)
        self.assertIsInstance(planning_world_obb, MirrorOrientedBoundingBox)
        print("OBB ROTATION OUTPUT")
        print("ROTATION (quaternion w, x, y, z)")
        print(planning_world_obb.rotation)
        print("HALF SIDE LENGTHS")
        print(planning_world_obb.half_side_length)
        # Identity quaternion is [1.0, 0.0, 0.0, 0.0]
        self.assertTrue(np.allclose(planning_world_obb.rotation, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))
        self.assertAlmostEqual(planning_world_obb.safety_tolerance, 0.01)
        self.assertTrue(np.allclose(planning_world_obb.center, np.array([0.0, 0.0, 0.0], dtype=np.float32)))
        self.assertTrue(np.allclose(planning_world_obb.half_side_length, np.array([0.5, 0.5, 0.5], dtype=np.float32)))

        cube_geom.set_enabled_collisions(False)
        stage_sphere.set_world_poses(
            positions=[3.0, 4.0, 5.0],
            orientations=[0.0, 0.0, 1.0, 0.0],
        )
        stage_sphere.set_local_scales([2.0, 2.0, 2.0])
        await app_utils.update_app_async()

        self.assertTrue(planning_world_obb.enabled)
        self.assertTrue(np.isclose(planning_world_obb.pose[0], [0.0, 1.0, 2.0]).all())
        self.assertTrue(np.isclose(planning_world_obb.pose[1], [1.0, 0.0, 0.0, 0.0]).all())
        self.assertTrue(np.isclose(planning_world_obb.scale, [2.0, 1.0, 1.0]).all())

        # Only transforms are synchronized; enabled state and scale retain their initialized values.
        world_binding.synchronize_transforms()
        self.assertTrue(planning_world_obb.enabled)
        self.assertTrue(np.isclose(planning_world_obb.pose[0], [3.0, 4.0, 5.0]).all())
        self.assertTrue(np.isclose(planning_world_obb.pose[1], [0.0, 0.0, 1.0, 0.0]).all())
        self.assertTrue(np.isclose(planning_world_obb.scale, [2.0, 1.0, 1.0]).all())
        # Identity quaternion is [1.0, 0.0, 0.0, 0.0]
        self.assertTrue(np.allclose(planning_world_obb.rotation, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)))
        self.assertAlmostEqual(planning_world_obb.safety_tolerance, 0.01)
        self.assertTrue(np.allclose(planning_world_obb.center, np.array([0.0, 0.0, 0.0], dtype=np.float32)))
        print(planning_world_obb.half_side_length)
        self.assertTrue(np.allclose(planning_world_obb.half_side_length, np.array([0.5, 0.5, 0.5], dtype=np.float32)))

    async def test_world_binding_validates_ancestor_scaling(self) -> None:
        """Integration test to verify that WorldBinding performs scene validation and rejects prims with invalid ancestor scaling."""
        # Create a parent with non-uniform scaling
        parent = Sphere(
            paths="/World/ScaledParent",
            positions=[0.0, 0.0, 0.0],
            orientations=[1.0, 0.0, 0.0, 0.0],
            scales=[2.0, 1.0, 2.0],  # Non-uniform scaling
        )

        # Create a child sphere
        child = Sphere(
            paths="/World/ScaledParent/ChildSphere",
            radii=0.5,
            positions=[0.0, 0.0, 0.0],
            orientations=[1.0, 0.0, 0.0, 0.0],
            scales=[1.0, 1.0, 1.0],
        )

        # Apply collision API
        child_geom = GeomPrim("/World/ScaledParent/ChildSphere", apply_collision_apis=True)
        child_geom.set_enabled_collisions(True)

        await app_utils.update_app_async()

        # WorldBinding should reject this due to ancestor scaling validation
        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=ObstacleStrategy(),
            tracked_prims=["/World/ScaledParent/ChildSphere"],
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )

        # Should raise AssertionError due to scene validation
        with self.assertRaises(AssertionError) as context:
            world_binding.initialize()

        # Verify the error message mentions ancestor scaling
        self.assertIn("ancestor", str(context.exception).lower())

        # Verify WorldBinding was not initialized
        self.assertFalse(world_binding._initialized)

    async def test_synchronize_transforms(self) -> None:
        """Test that synchronize_transforms updates only poses without checking properties."""
        # Create a sphere
        stage_sphere = Sphere(
            paths="/World/Sphere",
            radii=0.05,
            positions=[1.0, 2.0, 3.0],
            orientations=[1.0, 0.0, 0.0, 0.0],
            scales=[1.0, 1.0, 1.0],
        )

        sphere_geom = GeomPrim("/World/Sphere", apply_collision_apis=True)
        await app_utils.update_app_async()

        # Create world binding
        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            obstacle_strategy=ObstacleStrategy(),
            tracked_prims=["/World/Sphere"],
            tracked_collision_api=TrackableApi.PHYSICS_COLLISION,
        )
        world_binding.initialize()
        planning_world_sphere = world_binding.get_world_interface().collision_objects["/World/Sphere"]

        # Verify initial position
        self.assertTrue(np.isclose(planning_world_sphere.pose[0], [1.0, 2.0, 3.0]).all())

        # Move the sphere and set the radii and collision enable.
        stage_sphere.set_world_poses(
            positions=[5.0, 6.0, 7.0],
            orientations=[0.0, 1.0, 0.0, 0.0],
        )
        stage_sphere.set_radii(1.0)
        sphere_geom.set_enabled_collisions(False)
        await app_utils.update_app_async()

        # Before synchronizing transforms, pose should not be updated
        self.assertTrue(np.isclose(planning_world_sphere.pose[0], [1.0, 2.0, 3.0]).all())

        # Synchronize only transforms
        world_binding.synchronize_transforms()

        # After synchronizing transforms, pose should be updated
        self.assertTrue(np.isclose(planning_world_sphere.pose[0], [5.0, 6.0, 7.0]).all())
        self.assertTrue(np.isclose(planning_world_sphere.pose[1], [0.0, 1.0, 0.0, 0.0]).all())

        # The radii and collision enables are not updated:
        self.assertAlmostEqual(planning_world_sphere.radius, 0.05)
        self.assertTrue(planning_world_sphere.enabled)

    async def test_motion_generation_collision_api(self) -> None:
        """Test WorldBinding with the motion generation collision API."""
        import omni.usd
        from isaacsim.robot_motion.schema import apply_motion_planning_api

        # Create prims with motion planning API
        stage_sphere = Sphere(
            paths="/World/MotionSphere",
            radii=0.05,
            positions=[1.0, 2.0, 3.0],
            orientations=[1.0, 0.0, 0.0, 0.0],
            scales=[1.0, 1.0, 1.0],
        )
        stage_cube = Cube(
            paths="/World/MotionCube",
            sizes=0.15,
            positions=[-1.0, -2.0, -3.0],
            orientations=[0.0, 1.0, 0.0, 0.0],
            scales=[2.0, 2.0, 2.0],
        )

        # Apply motion planning API to the prims
        stage = omni.usd.get_context().get_stage()
        sphere_prim = stage.GetPrimAtPath("/World/MotionSphere")
        cube_prim = stage.GetPrimAtPath("/World/MotionCube")
        apply_motion_planning_api(sphere_prim, enabled=True)
        apply_motion_planning_api(cube_prim, enabled=True)

        await app_utils.update_app_async()

        # Create world binding with motion generation collision API
        obstacle_strategy = ObstacleStrategy()
        obstacle_strategy.set_default_safety_tolerance(0.06)
        world_binding: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            tracked_prims=["/World/MotionSphere", "/World/MotionCube"],
            obstacle_strategy=obstacle_strategy,
            tracked_collision_api=TrackableApi.MOTION_GENERATION_COLLISION,
        )

        world_binding.initialize()

        world_interface = world_binding.get_world_interface()
        planning_world_sphere = world_interface.collision_objects["/World/MotionSphere"]
        planning_world_cube = world_interface.collision_objects["/World/MotionCube"]
        self.assertIsNotNone(planning_world_sphere)
        self.assertAlmostEqual(planning_world_sphere.radius, 0.05)
        self.assertAlmostEqual(planning_world_sphere.safety_tolerance, 0.06)
        self.assertTrue(np.isclose(planning_world_sphere.pose[0], [1.0, 2.0, 3.0]).all())
        self.assertTrue(np.isclose(planning_world_sphere.pose[1], [1.0, 0.0, 0.0, 0.0]).all())
        self.assertTrue(planning_world_sphere.enabled)

        self.assertIsNotNone(planning_world_cube)
        self.assertAlmostEqual(planning_world_cube.size, 0.15)
        self.assertAlmostEqual(planning_world_cube.safety_tolerance, 0.06)
        self.assertTrue(np.isclose(planning_world_cube.pose[0], [-1.0, -2.0, -3.0]).all())
        self.assertTrue(np.isclose(planning_world_cube.pose[1], [0.0, 1.0, 0.0, 0.0]).all())
        self.assertTrue(planning_world_cube.enabled)

        # Test that initialization fails if prim doesn't have the API
        world_binding_no_api: WorldBinding[MirrorWorldInterface] = WorldBinding(
            world_interface=MirrorWorldInterface(),
            tracked_prims=["/World/MotionSphere"],
            obstacle_strategy=ObstacleStrategy(),
            tracked_collision_api=TrackableApi.MOTION_GENERATION_COLLISION,
        )

        # Remove the API from the sphere
        sphere_prim.RemoveAppliedSchema("IsaacMotionPlanningAPI")
        await app_utils.update_app_async()

        # Initialize should raise RuntimeError
        self.assertRaises(RuntimeError, world_binding_no_api.initialize)
