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

"""Verify custom OG Replicator randomizers and matching tutorial script-editor workflows."""

import numpy as np
import omni.kit
import omni.replicator.core as rep
import omni.replicator.core.functional as rep_functional
import omni.usd
from isaacsim.core.experimental.prims import XformPrim
from isaacsim.core.experimental.utils import app as app_utils
from isaacsim.core.experimental.utils import stage as stage_utils
from omni.replicator.core.scripts.utils import ReplicatorItem, ReplicatorWrapper, create_node, set_target_prims

PRIM_COUNT = 50
PRIM_SCALE = 0.1
RAD_IN = 0.5
RAD_ON = 1.5
RAD_BET1 = 2.5
RAD_BET2 = 3.5
NUM_ITERATIONS = 3
NUM_GRAPH_STEPS = 10
POSITION_TOLERANCE = 0.05


def _world_distances(replicator_item: ReplicatorItem) -> np.ndarray:
    stage = omni.usd.get_context().get_stage()
    distances = []
    for path in replicator_item.node.get_attribute("outputs:prims").get():
        prim = stage.GetPrimAtPath(str(path))
        distances.append(np.linalg.norm(rep_functional.utils.get_world_position(prim)))
    return np.array(distances)


def _world_positions(replicator_item: ReplicatorItem) -> np.ndarray:
    """Return world-space positions for all prims produced by a Replicator item.

    Args:
        replicator_item: Replicator item that produces the prim paths.

    Returns:
        World-space point coordinates with shape ``(N, 3)``.
    """
    stage = omni.usd.get_context().get_stage()
    positions: list[np.ndarray] = []
    for path in replicator_item.node.get_attribute("outputs:prims").get():
        prim = stage.GetPrimAtPath(str(path))
        positions.append(rep_functional.utils.get_world_position(prim))
    return np.array(positions)


def sample_points_on_sphere(radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample random 3D points on the surface of a sphere.

    Args:
        radius: Positive sphere radius assumed by the sampling helper.
        count: Nonnegative number of points to sample.
        rng: Random number generator used for sampling.

    Returns:
        Cartesian point coordinates with shape ``(count, 3)``.
    """
    phi = rng.uniform(0, 2 * np.pi, count)
    costheta = rng.uniform(-1, 1, count)
    theta = np.arccos(costheta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.stack([x, y, z], axis=1)


def sample_points_in_sphere(radius: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample random 3D points uniformly within a sphere volume.

    Args:
        radius: Positive sphere radius assumed by the sampling helper.
        count: Nonnegative number of points to sample.
        rng: Random number generator used for sampling.

    Returns:
        Cartesian point coordinates with shape ``(count, 3)``.
    """
    phi = rng.uniform(0, 2 * np.pi, count)
    costheta = rng.uniform(-1, 1, count)
    theta = np.arccos(costheta)
    r = radius * (rng.random(count) ** (1 / 3))
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


def sample_points_between_spheres(radius1: float, radius2: float, count: int, rng: np.random.Generator) -> np.ndarray:
    """Sample random 3D points uniformly between two concentric spheres.

    Args:
        radius1: Nonnegative first shell boundary assumed by the sampling helper.
        radius2: Positive second shell boundary assumed by the sampling helper. Valid boundaries are ordered
            before sampling.
        count: Nonnegative number of points to sample.
        rng: Random number generator used for sampling.

    Returns:
        Cartesian point coordinates with shape ``(count, 3)``.
    """
    if radius1 > radius2:
        radius1, radius2 = radius2, radius1
    phi = rng.uniform(0, 2 * np.pi, count)
    costheta = rng.uniform(-1, 1, count)
    theta = np.arccos(costheta)
    r = rng.uniform(radius1**3, radius2**3, count) ** (1 / 3)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


@ReplicatorWrapper
def on_sphere(
    radius: float = 1.0,
    input_prims: ReplicatorItem | list[str] | None = None,
) -> ReplicatorItem:
    """Create a custom OGN node that samples target prims on a sphere.

    Args:
        radius: Positive sphere radius; nonpositive values make the node fail during graph evaluation.
        input_prims: Target prims, or ``None`` or an empty list to use the current Replicator context.

    Returns:
        Replicator item for the custom OmniGraph node.
    """
    node = create_node("isaacsim.replicator.examples.OgnSampleOnSphere", radius=radius)
    if input_prims:
        set_target_prims(node, "inputs:prims", input_prims)
    return node


@ReplicatorWrapper
def in_sphere(
    radius: float = 1.0,
    input_prims: ReplicatorItem | list[str] | None = None,
) -> ReplicatorItem:
    """Create a custom OGN node that samples target prims inside a sphere.

    Args:
        radius: Positive sphere radius; nonpositive values make the node fail during graph evaluation.
        input_prims: Target prims, or ``None`` or an empty list to use the current Replicator context.

    Returns:
        Replicator item for the custom OmniGraph node.
    """
    node = create_node("isaacsim.replicator.examples.OgnSampleInSphere", radius=radius)
    if input_prims:
        set_target_prims(node, "inputs:prims", input_prims)
    return node


@ReplicatorWrapper
def between_spheres(
    radius1: float = 0.5,
    radius2: float = 1.0,
    input_prims: ReplicatorItem | list[str] | None = None,
) -> ReplicatorItem:
    """Create a custom OGN node that samples target prims within a spherical shell.

    Args:
        radius1: First shell boundary, which must be nonnegative.
        radius2: Second shell boundary, which must be positive. The node validates the named inputs before
            ordering them, so zero is accepted only as ``radius1``; two positive boundaries may be supplied
            in either order.
        input_prims: Target prims, or ``None`` or an empty list to use the current Replicator context.

    Returns:
        Replicator item for the custom OmniGraph node.
    """
    node = create_node("isaacsim.replicator.examples.OgnSampleBetweenSpheres", radius1=radius1, radius2=radius2)
    if input_prims:
        set_target_prims(node, "inputs:prims", input_prims)
    return node


class TestOgnCustomReplicatorRandomizer(omni.kit.test.AsyncTestCase):
    """Exercise tutorial custom OG randomizers via functional API and Replicator graph wrappers."""

    async def setUp(self) -> None:
        """Create a clean stage before each test."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()
        await app_utils.update_app_async()

    async def tearDown(self) -> None:
        """Close the stage and wait for pending loads."""
        stage_utils.close_stage()
        await app_utils.update_app_async()
        while omni.usd.get_context().get_stage_loading_status()[2] > 0:
            await app_utils.update_app_async()

    async def test_custom_randomizer_script_editor(self) -> None:
        """Match ``custom_og_randomizer_script_editor.py``: batched sampling and ``modify.pose``."""
        rep.functional.create.xform(name="World")
        rep.functional.create.dome_light(parent="/World", intensity=500)

        rng = rep.rng.ReplicatorRNG(seed=42).generator

        on_sphere_prims = rep.functional.create_batch.sphere(
            count=PRIM_COUNT, parent="/World", name="sphere", scale=PRIM_SCALE
        )
        in_sphere_prims = rep.functional.create_batch.cube(
            count=PRIM_COUNT, parent="/World", name="cube", scale=PRIM_SCALE
        )
        between_spheres_prims = rep.functional.create_batch.cylinder(
            count=PRIM_COUNT, parent="/World", name="cylinder", scale=PRIM_SCALE
        )

        for _ in range(NUM_ITERATIONS):
            await app_utils.update_app_async()
            rep.functional.modify.pose(
                in_sphere_prims,
                position_value=sample_points_in_sphere(RAD_IN, PRIM_COUNT, rng),
                rotation_value=rng.uniform(0, 360, size=(PRIM_COUNT, 3)),
            )
            rep.functional.modify.pose(
                on_sphere_prims,
                position_value=sample_points_on_sphere(RAD_ON, PRIM_COUNT, rng),
                rotation_value=rng.uniform(0, 360, size=(PRIM_COUNT, 3)),
            )
            rep.functional.modify.pose(
                between_spheres_prims,
                position_value=sample_points_between_spheres(RAD_BET1, RAD_BET2, PRIM_COUNT, rng),
                rotation_value=rng.uniform(0, 360, size=(PRIM_COUNT, 3)),
            )

        await app_utils.update_app_async()

        in_positions, _ = XformPrim("/World/cube_*").get_world_poses()
        in_distances = np.linalg.norm(in_positions.numpy(), axis=1)
        self.assertTrue(np.all(in_distances <= RAD_IN + POSITION_TOLERANCE))

        on_positions, _ = XformPrim("/World/sphere_*").get_world_poses()
        on_distances = np.linalg.norm(on_positions.numpy(), axis=1)
        self.assertTrue(np.all(np.abs(on_distances - RAD_ON) <= POSITION_TOLERANCE))

        between_positions, _ = XformPrim("/World/cylinder_*").get_world_poses()
        between_distances = np.linalg.norm(between_positions.numpy(), axis=1)
        self.assertTrue(np.all(between_distances >= RAD_BET1 - POSITION_TOLERANCE))
        self.assertTrue(np.all(between_distances <= RAD_BET2 + POSITION_TOLERANCE))

    async def test_custom_replicator_randomizer_graph(self) -> None:
        """Match ``replicator_wrapper_script_editor.py``: OG nodes via ``@ReplicatorWrapper``."""
        sphere = rep.create.sphere(count=PRIM_COUNT, scale=PRIM_SCALE)
        cube = rep.create.cube(count=PRIM_COUNT, scale=PRIM_SCALE)
        cylinder = rep.create.cylinder(count=PRIM_COUNT, scale=PRIM_SCALE)

        with rep.trigger.on_frame():
            with sphere:
                rep.randomizer.rotation()
                on_sphere(RAD_ON)
            with cube:
                rep.randomizer.rotation()
                in_sphere(RAD_IN)
            with cylinder:
                rep.randomizer.rotation()
                between_spheres(RAD_BET1, RAD_BET2)

        rep.orchestrator.set_capture_on_play(False)
        await rep.orchestrator.preview_async()
        for _ in range(NUM_GRAPH_STEPS):
            await rep.orchestrator.step_async()

        on_distances = _world_distances(sphere)
        self.assertTrue(np.all(np.abs(on_distances - RAD_ON) <= POSITION_TOLERANCE))

        in_distances = _world_distances(cube)
        self.assertTrue(np.all(in_distances <= RAD_IN + POSITION_TOLERANCE))

        between_distances = _world_distances(cylinder)
        self.assertTrue(np.all(between_distances >= RAD_BET1 - POSITION_TOLERANCE))
        self.assertTrue(np.all(between_distances <= RAD_BET2 + POSITION_TOLERANCE))

    async def test_custom_replicator_randomizer_global_seed_repeatability(self) -> None:
        """Rebuilding the graph with the same Replicator global seed reproduces all samples."""
        sampled_positions: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for _ in range(2):
            await stage_utils.create_new_stage_async()
            rep.utils.get_graph()
            await app_utils.update_app_async()
            rep.set_global_seed(42)

            sphere = rep.create.sphere(count=PRIM_COUNT, scale=PRIM_SCALE)
            cube = rep.create.cube(count=PRIM_COUNT, scale=PRIM_SCALE)
            cylinder = rep.create.cylinder(count=PRIM_COUNT, scale=PRIM_SCALE)

            with rep.trigger.on_frame():
                with sphere:
                    on_sphere(RAD_ON)
                with cube:
                    in_sphere(RAD_IN)
                with cylinder:
                    between_spheres(RAD_BET1, RAD_BET2)

            rep.orchestrator.set_capture_on_play(False)
            await rep.orchestrator.preview_async()
            await rep.orchestrator.step_async()
            sampled_positions.append(
                (
                    _world_positions(sphere),
                    _world_positions(cube),
                    _world_positions(cylinder),
                )
            )

        for first_run, second_run in zip(sampled_positions[0], sampled_positions[1], strict=True):
            np.testing.assert_allclose(first_run, second_run)
