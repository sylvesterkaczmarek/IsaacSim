# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Provide utility functions for object-based synthetic data generation."""

from typing import Any

import numpy as np
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
from isaacsim.core.experimental.utils.prim import get_prim_at_path
from isaacsim.core.experimental.utils.transform import look_at_quaternion
from omni.kit.viewport.utility import get_active_viewport
from pxr import Usd, UsdGeom


def add_colliders(root_path: str) -> None:
    """Enable collisions on the asset (without rigid body dynamics the asset will be static).

    Args:
        root_path: Asset root path whose geometry descendants should receive collision APIs.
    """
    root_prim = get_prim_at_path(root_path)
    mesh_paths = [str(p.GetPrimPath()) for p in Usd.PrimRange(root_prim) if p.IsA(UsdGeom.Gprim)]
    if not mesh_paths:
        return
    geom = GeomPrim(mesh_paths, apply_collision_apis=True)
    geom.set_offsets(contact_offsets=[0.001], rest_offsets=[0.0])
    geom.set_collision_approximations("convexHull")


def create_collision_box_walls(
    path: str,
    width: float,
    depth: float,
    height: float,
    thickness: float = 0.5,
    visible: bool = False,
) -> None:
    """Create a collision box area wrapping the given working area with origin at (0, 0, 0).

    Args:
        stage: Stage on which to define the six wall prims.
        path: Parent prim path for the collision walls.
        width: Interior extent along the x-axis in stage units.
        depth: Interior extent along the y-axis in stage units.
        height: Interior extent along the z-axis in stage units.
        thickness: Outward wall thickness in stage units.
        visible: Whether the collision geometry should remain visible.
    """
    # Define the walls (name, location, size) with thickness towards outside of the working area
    walls = [
        ("floor", (0, 0, (height + thickness) / -2.0), (width, depth, thickness)),
        ("ceiling", (0, 0, (height + thickness) / 2.0), (width, depth, thickness)),
        ("left_wall", ((width + thickness) / -2.0, 0, 0), (thickness, depth, height)),
        ("right_wall", ((width + thickness) / 2.0, 0, 0), (thickness, depth, height)),
        ("front_wall", (0, (depth + thickness) / 2.0, 0), (width, thickness, height)),
        ("back_wall", (0, (depth + thickness) / -2.0, 0), (width, thickness, height)),
    ]
    for name, location, size in walls:
        wall_path = f"{path}/{name}"
        Cube(wall_path, sizes=[1.0], translations=[location], scales=[size])
        add_colliders(wall_path)
        if not visible:
            XformPrim(wall_path).set_visibilities([False])


def get_random_transform_values(
    rng: Any,
    loc_min: tuple[float, float, float] = (0, 0, 0),
    loc_max: tuple[float, float, float] = (1, 1, 1),
    rot_min: tuple[float, float, float] = (0, 0, 0),
    rot_max: tuple[float, float, float] = (360, 360, 360),
    scale_min_max: tuple[float, float] = (0.1, 1.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    """Create random transformation values for location, rotation, and scale.

    Args:
        rng: NumPy random generator used for sampling.
        loc_min: Inclusive lower bounds for each translation component.
        loc_max: Inclusive upper bounds for each translation component.
        rot_min: Inclusive lower bounds for each Euler rotation component in degrees.
        rot_max: Inclusive upper bounds for each Euler rotation component in degrees.
        scale_min_max: Lower and upper bounds for the uniform scale factor.

    Returns:
        Random translation, Euler rotation, and uniform three-axis scale tuples.
    """
    location = tuple(rng.generator.uniform(loc_min, loc_max).tolist())
    rotation = tuple(rng.generator.uniform(rot_min, rot_max).tolist())
    scale_value = float(rng.generator.uniform(scale_min_max[0], scale_min_max[1]))
    scale = (scale_value, scale_value, scale_value)
    return location, rotation, scale


def get_random_pose_on_sphere(
    rng: Any,
    origin: tuple[float, float, float],
    radius: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Generate a random pose on a sphere looking at the origin (wxyz quaternion).

    Args:
        rng: NumPy random generator used for sampling.
        origin: Center of the sampling sphere and look-at target.
        radius: Distance from the sampled position to the target in stage units.

    Returns:
        Random position on the sphere and orientation quaternion that points the camera at its center.
    """
    origin = np.asarray(origin, dtype=np.float32)

    # Generate random angles for spherical coordinates
    theta = float(rng.generator.uniform(0, 2 * np.pi))
    phi = float(np.arcsin(rng.generator.uniform(-1, 1)))

    # Spherical to Cartesian conversion
    offset = np.array(
        [
            radius * np.cos(theta) * np.cos(phi),
            radius * np.sin(phi),
            radius * np.sin(theta) * np.cos(phi),
        ],
        dtype=np.float32,
    )
    location = origin + offset
    orientation = look_at_quaternion(eye=location, target=origin).numpy()
    return tuple(location.tolist()), tuple(orientation.tolist())


def set_render_products_updates(render_products: list, enabled: bool, include_viewport: bool = False) -> None:
    """Enable or disable the render products and viewport rendering.

    Args:
        render_products: Render products whose Hydra texture updates should be toggled.
        enabled: Whether subsequent render-product updates should run.
        include_viewport: Whether to apply the same update state to the active viewport.
    """
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(enabled)
    if include_viewport:
        get_active_viewport().updates_enabled = enabled


def apply_velocities_towards_target(
    prim_paths: list[str],
    rng: Any,
    target: tuple[float, float, float] = (0, 0, 0),
    strength_range: tuple[float, float] = (0.1, 1.0),
) -> None:
    """Apply velocities to prims directing them towards a target point.

    Args:
        prim_paths: Physics prim paths whose linear velocities should be updated.
        rng: NumPy random generator used for sampling.
        target: World-space point toward which each prim should move.
        strength_range: Lower and upper random factors applied to the displacement vector.
    """
    for path in prim_paths:
        loc = XformPrim(path).get_local_poses()[0].numpy()[0]
        strength = float(rng.generator.uniform(strength_range[0], strength_range[1]))
        velocity = ((target[0] - loc[0]) * strength, (target[1] - loc[1]) * strength, (target[2] - loc[2]) * strength)
        RigidPrim(path).set_velocities(linear_velocities=[velocity])


def apply_random_velocities(
    prim_paths: list[str],
    rng: Any,
    linear_range: tuple[float, float] = (-2.5, 2.5),
    angular_range: tuple[float, float] = (-45, 45),
) -> None:
    """Apply random linear and angular velocities to prims.

    Args:
        prim_paths: Physics prim paths whose velocities should be randomized.
        rng: NumPy random generator used for sampling.
        linear_range: Lower and upper bounds for each linear velocity component in stage units per second.
        angular_range: Lower and upper bounds for each angular velocity component in degrees per second.
    """
    for path in prim_paths:
        lin_vel = tuple(rng.generator.uniform(linear_range[0], linear_range[1], size=3).tolist())
        # RigidPrim.set_velocities expects angular velocities in radians.
        ang_vel = np.deg2rad(rng.generator.uniform(angular_range[0], angular_range[1], size=3)).tolist()
        RigidPrim(path).set_velocities(linear_velocities=[lin_vel], angular_velocities=[ang_vel])
