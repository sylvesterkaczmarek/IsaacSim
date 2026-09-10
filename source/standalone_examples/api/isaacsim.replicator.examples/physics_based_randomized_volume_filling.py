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

"""Fill randomized pallet volumes with physics-enabled box assets."""

from isaacsim import SimulationApp

simulation_app = SimulationApp(launch_config={"headless": False})

import argparse
import os
from itertools import chain
from typing import Any

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.bounds as bounds_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.transform as transform_utils
import isaacsim.core.experimental.utils.xform as xform_utils
import numpy as np
import omni.kit.app
import omni.physx
import omni.replicator.core as rep
from isaacsim.core.experimental.materials import RigidBodyMaterial
from isaacsim.core.experimental.objects import Cube
from isaacsim.core.experimental.prims import GeomPrim, XformPrim
from isaacsim.storage.native import get_assets_root_path
from pxr import PhysicsSchemaTools, UsdUtils

NUM_PALLETS = 6
ENV_URL = "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
WRITE_DATA = True
SEED = 42


def _parse_env_url(env_url: str) -> str | None:
    return None if env_url.lower() in {"none", "null"} else env_url


parser = argparse.ArgumentParser()
parser.add_argument("--num-pallets", type=int, default=NUM_PALLETS, help="Number of pallets to spawn.")
parser.add_argument(
    "--env-url",
    type=_parse_env_url,
    default=ENV_URL,
    help="Environment USD path relative to the assets root, or none for an empty stage.",
)
parser.add_argument(
    "--write-data",
    action=argparse.BooleanOptionalAction,
    default=WRITE_DATA,
    help="Write captured output to disk.",
)
args, _ = parser.parse_known_args()


def resolve_scene_root_path(stage: Any) -> str | None:
    """Return a valid root path for spawning scene content.

    Args:
        stage: USD stage to inspect.

    Returns:
        The scene root path, or None if the stage has no usable root.
    """
    if stage is None:
        return None
    for root_path in ("/World", "/Root"):
        if stage.GetPrimAtPath(root_path).IsValid():
            return root_path
    default_prim = stage.GetDefaultPrim()
    if default_prim.IsValid():
        return str(default_prim.GetPath())
    return None


def add_rigid_body_dynamics(prim: Any, disable_gravity: bool = False, angular_damping: float | None = None) -> None:
    """Apply rigid-body dynamics to a prim.

    Args:
        prim: Prim to configure.
        disable_gravity: Whether to disable gravity on the body.
        angular_damping: Angular damping to apply, or None to use the default.
    """
    # This flow adds bodies at runtime, so author the APIs without creating a persistent tensor view.
    rigid_body_attributes = {
        "rigidBodyEnabled": True,
        "disableGravity": disable_gravity,
        "maxDepenetrationVelocity": 3.0,
        "maxLinearVelocity": float("inf"),
    }
    if angular_damping is not None:
        rigid_body_attributes["angularDamping"] = angular_damping
    rep.functional.physics.apply_rigid_body(prim, with_collider=False, **rigid_body_attributes)


def create_asset(
    asset_url: str,
    path: str,
    *,
    translations: Any = None,
    orientations: Any = None,
    scales: Any = None,
) -> Any:
    """Create a referenced asset with the requested transform.

    Args:
        asset_url: URL of the USD asset to reference.
        path: Desired prim path for the asset.
        translations: Local translation values, or None to preserve the asset values.
        orientations: Local orientation values, or None to preserve the asset values.
        scales: Local scale values, or None to preserve the asset values.

    Returns:
        The created asset prim.
    """
    prim_path = stage_utils.generate_next_free_path(path, prepend_default_prim=False)
    prim = stage_utils.add_reference_to_stage(usd_path=asset_url, path=prim_path)
    XformPrim(
        prim_path,
        translations=translations,
        orientations=orientations,
        scales=scales,
        reset_xform_op_properties=True,
    )
    return prim


def create_asset_with_colliders(
    asset_url: str,
    path: str,
    *,
    translations: Any = None,
    orientations: Any = None,
    scales: Any = None,
) -> Any:
    """Create a referenced asset and apply convex-hull colliders.

    Args:
        asset_url: URL of the USD asset to reference.
        path: Desired prim path for the asset.
        translations: Local translation values, or None to preserve the asset values.
        orientations: Local orientation values, or None to preserve the asset values.
        scales: Local scale values, or None to preserve the asset values.

    Returns:
        The created asset prim.
    """
    prim = create_asset(asset_url, path, translations=translations, orientations=orientations, scales=scales)
    pallet_geom = GeomPrim(f"{prim.GetPath()}/.*", apply_collision_apis=True)
    pallet_geom.set_collision_approximations("convexHull")
    return prim


def place_collision_walls(
    pallet_prim: Any,
    walls_root: str,
    bbox_cache: Any,
    height: float = 2,
    thickness: float = 0.3,
    material: RigidBodyMaterial | None = None,
    visible: bool = False,
) -> None:
    """Place collision walls around a pallet.

    Args:
        pallet_prim: Pallet prim whose bounds define the wall layout.
        walls_root: Parent path for the collision walls.
        bbox_cache: Bounding-box cache used to measure the pallet.
        height: Wall height above the pallet.
        thickness: Wall thickness.
        material: Physics material to apply, or None to leave the walls unbound.
        visible: Whether to render the collision walls.
    """
    bbox_cache.Clear()
    # Untransformed (pallet-local) bounds so the walls can follow the pallet's position and orientation.
    aabb = bounds_utils.compute_aabb(pallet_prim, bbox_cache=bbox_cache, space="untransformed")
    width, depth, local_height = aabb[3:] - aabb[:3]
    mid = (aabb[:3] + aabb[3:]) * 0.5 + np.array([0.0, 0.0, local_height / 2])

    walls = [
        ("floor", (mid[0], mid[1], mid[2] - thickness / 2), (width, depth, thickness)),
        ("ceiling", (mid[0], mid[1], mid[2] + height + thickness / 2), (width, depth, thickness)),
        (
            "left_wall",
            (mid[0] - (width + thickness) / 2, mid[1], mid[2] + height / 2),
            (thickness, depth, height),
        ),
        (
            "right_wall",
            (mid[0] + (width + thickness) / 2, mid[1], mid[2] + height / 2),
            (thickness, depth, height),
        ),
        (
            "front_wall",
            (mid[0], mid[1] + (depth + thickness) / 2, mid[2] + height / 2),
            (width, thickness, height),
        ),
        (
            "back_wall",
            (mid[0], mid[1] - (depth + thickness) / 2, mid[2] + height / 2),
            (width, thickness, height),
        ),
    ]

    # Map pallet-local wall poses into the world frame so the shared wall set matches each pallet's pose.
    pallet_position, pallet_orientation = xform_utils.get_world_pose(pallet_prim)

    stage = stage_utils.get_current_stage(backend="usd")
    for name, location, size in walls:
        wall_path = f"{walls_root}/{name}"
        wall_scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        world_pos = tuple(
            transform_utils.transform_local_to_world(location, pallet_position, pallet_orientation).numpy()
        )
        if stage.GetPrimAtPath(wall_path).IsValid():
            wall = XformPrim(
                wall_path,
                positions=world_pos,
                orientations=pallet_orientation.numpy(),
                scales=wall_scale,
                reset_xform_op_properties=True,
            )
            wall_geom = GeomPrim(wall_path)
        else:
            wall = Cube(
                wall_path,
                sizes=2.0,
                positions=world_pos,
                orientations=pallet_orientation.numpy(),
                scales=wall_scale,
                reset_xform_op_properties=True,
            )
            wall_geom = GeomPrim(wall_path, apply_collision_apis=True)
            wall_geom.set_collision_approximations("convexHull")
        if material is not None:
            wall_geom.apply_physics_materials(material, weaker_than_descendants=[True])
        wall.set_visibilities([visible])


def apply_forces(boxes: list[Any], pallet: Any, strength: float = 550, strength_center_multiplier: float = 2) -> None:
    """Apply settling forces to boxes on a pallet.

    Args:
        boxes: Box prims to push from each horizontal direction.
        pallet: Pallet prim that defines the final centering force.
        strength: Magnitude of each directional force.
        strength_center_multiplier: Multiplier for the final centering force.
    """
    stage = stage_utils.get_current_stage(backend="usd")
    app_utils.play()
    pallet_center, pallet_orientation = xform_utils.get_world_pose(pallet)
    force_forward = transform_utils.rotate_vectors_by_quaternion([1.0, 0.0, 0.0], pallet_orientation).numpy() * strength
    force_right = transform_utils.rotate_vectors_by_quaternion([0.0, 1.0, 0.0], pallet_orientation).numpy() * strength

    physx_simulation_interface = omni.physx.get_physx_simulation_interface()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    directional_forces = [force_forward, force_right, -force_forward, -force_right]
    for box_prim in boxes:
        body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
        for force in chain(directional_forces, directional_forces):
            box_position, _ = xform_utils.get_world_pose(box_prim)
            box_position = carb.Float3(*box_position.numpy())
            physx_simulation_interface.apply_force_at_pos(stage_id, body_path, carb.Float3(*force), box_position)
            app_utils.update_app(steps=10)

    for box_prim in boxes:
        body_path = PhysicsSchemaTools.sdfPathToInt(box_prim.GetPath())
        box_location, _ = xform_utils.get_world_pose(box_prim)
        box_location = box_location.numpy()
        force_to_center = (pallet_center.numpy() - box_location) * strength * strength_center_multiplier
        physx_simulation_interface.apply_force_at_pos(
            stage_id,
            body_path,
            carb.Float3(*force_to_center),
            carb.Float3(*box_location),
        )
    app_utils.update_app(steps=20)
    app_utils.pause()


def stack_boxes_on_pallet(
    pallet_prim: Any,
    walls_root: str,
    boxes_urls_and_weights: list[tuple[str, float]],
    num_boxes: int,
    drop_height: float = 1.5,
    drop_margin: float = 0.2,
    gen: Any = None,
) -> list[Any]:
    """Stack randomized boxes on one pallet.

    Args:
        pallet_prim: Pallet prim on which to stack boxes.
        walls_root: Parent path for temporary collision walls.
        boxes_urls_and_weights: Box asset URLs paired with sampling weights.
        num_boxes: Number of boxes to create.
        drop_height: Vertical offset of the temporary spawn volume above the pallet.
        drop_margin: Horizontal inset from each pallet edge used for spawning.
        gen: NumPy-compatible random generator used to sample box assets and poses. Must not be None.

    Returns:
        The stacked box prims.
    """
    pallet_path = pallet_prim.GetPath()
    print(f"[BoxStacking] Running scenario for pallet {pallet_path} with {num_boxes} boxes..")
    bbox_cache = bounds_utils.create_bbox_cache()

    physics_material = RigidBodyMaterial(
        f"{pallet_path}/Looks/PhysicsMaterial",
        static_frictions=[0.01],
        dynamic_frictions=[0.01],
        restitutions=[0.0],
    )
    GeomPrim(f"{pallet_path}/.*").apply_physics_materials(physics_material, weaker_than_descendants=[True])

    place_collision_walls(
        pallet_prim, walls_root, bbox_cache, height=drop_height + drop_margin, material=physics_material
    )

    box_urls, box_weights = zip(*boxes_urls_and_weights)
    box_weights_arr = np.asarray(box_weights, dtype=float)
    box_weights_arr /= box_weights_arr.sum()
    rand_boxes_urls = gen.choice(box_urls, size=num_boxes, p=box_weights_arr)
    boxes = [create_asset(box_url, f"{pallet_path}_Boxes/Box_{i}") for i, box_url in enumerate(rand_boxes_urls)]
    boxes.sort(
        key=lambda box: bounds_utils.compute_bound_volume(box, bbox_cache=bbox_cache, space="local"),
        reverse=True,
    )

    spawn_midpoint, spawn_size = bounds_utils.compute_bound_range(
        pallet_prim, bbox_cache=bbox_cache, space="untransformed"
    )
    pallet_width, pallet_depth, pallet_height = spawn_size
    spawn_center = spawn_midpoint + np.array([0.0, 0.0, pallet_height / 2 + drop_height])
    spawn_width, spawn_depth = pallet_width / 2 - drop_margin, pallet_depth / 2 - drop_margin

    pallet_position, pallet_orientation = xform_utils.get_world_pose(pallet_prim)

    for box_prim in boxes:
        local_loc = spawn_center + np.array(
            [gen.uniform(-spawn_width, spawn_width), gen.uniform(-spawn_depth, spawn_depth), 0.0]
        )
        angles = [
            gen.choice([180, 90, 0, -90, -180]) + gen.uniform(-3, 3),
            gen.choice([180, 90, 0, -90, -180]) + gen.uniform(-3, 3),
            gen.choice([180, 90, 0, -90, -180]) + gen.uniform(-3, 3),
        ]
        local_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for axis_index, angle in enumerate(angles):
            axis_angles = [0.0, 0.0, 0.0]
            axis_angles[axis_index] = angle
            axis_quat = transform_utils.euler_angles_to_quaternion(axis_angles, degrees=True).numpy().astype(np.float32)
            local_orientation = transform_utils.quaternion_multiplication(local_orientation, axis_quat).numpy()
        world_loc = transform_utils.transform_local_to_world(local_loc, pallet_position, pallet_orientation).numpy()
        world_orientation = transform_utils.quaternion_multiplication(pallet_orientation, local_orientation).numpy()
        box_path = str(box_prim.GetPath())

        XformPrim(
            box_path,
            positions=tuple(world_loc),
            orientations=world_orientation,
            reset_xform_op_properties=True,
        )
        box_geom = GeomPrim(f"{box_path}/.*", apply_collision_apis=True)
        box_geom.set_collision_approximations("convexHull")
        add_rigid_body_dynamics(box_prim, angular_damping=0.9)

        box_geom.apply_physics_materials(physics_material, weaker_than_descendants=[True])
        app_utils.update_app()

        app_utils.play()
        app_utils.update_app(steps=20)
        app_utils.pause()

    apply_forces(boxes, pallet_prim)

    # Flush the velocity changes while simulation is still enabled. Writing velocities after disabling a body
    # causes PhysX to reject the operation, and a persistent RigidPrim view would be invalidated by the next batch.
    zero_velocities = np.zeros((len(boxes), 3), dtype=np.float32)
    rep.functional.modify.attribute(boxes, "physics:velocity", zero_velocities)
    rep.functional.modify.attribute(boxes, "physics:angularVelocity", zero_velocities)
    omni.physx.get_physx_simulation_interface().flush_changes()

    rep.functional.modify.attribute(boxes, "physics:rigidBodyEnabled", False)
    omni.physx.get_physx_simulation_interface().flush_changes()

    physics_material.set_friction_coefficients(static_frictions=[0.9], dynamic_frictions=[0.9])

    return boxes


def run_box_stacking_scenarios(
    num_pallets: int, env_url: str | None = None, write_data: bool = True, rng: Any = None
) -> None:
    """Run randomized box-stacking scenarios.

    Args:
        num_pallets: Number of pallets to populate.
        env_url: Environment URL, or None to create an empty stage.
        write_data: Whether to capture and write an RGB image.
        rng: Replicator random-number generator, or None to create one from the example seed.
    """
    if rng is None:
        rep.set_global_seed(SEED)
        rng = rep.rng.ReplicatorRNG(seed=SEED)
    gen = rng.generator

    assets_root_path = get_assets_root_path()

    pallets_urls_and_weights = [
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_01.usd", 0.25),
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_PaletteA_02.usd", 0.75),
    ]
    boxes_urls_and_weights = [
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01.usd", 0.02),
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxB_01.usd", 0.06),
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxC_01.usd", 0.12),
        (assets_root_path + "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_01.usd", 0.80),
    ]

    scene_root = "/World"
    if env_url is not None:
        env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
        success, _stage = stage_utils.open_stage(env_path)
        if not success or _stage is None:
            carb.log_error(f"[BoxStacking] Failed to open stage: {env_path}")
            return
        app_utils.update_app()
        resolved_root = resolve_scene_root_path(_stage)
        if resolved_root is None:
            carb.log_error(
                "[BoxStacking] Could not find a valid scene root in the opened environment "
                "(expected /World or /Root, or a default prim on the stage)."
            )
            return
        scene_root = resolved_root
        print(f"[BoxStacking] Using scene root: {scene_root}")
    else:
        stage_utils.create_new_stage(template="empty")
        rep.functional.create.scope(name="Lights", parent="/World")
        rep.functional.create.distant_light(
            intensity=400, parent="/World/Lights", name="DistantLight", rotation=(0, 60, 0)
        )
        rep.functional.create.dome_light(intensity=500, parent="/World/Lights", name="DomeLight")
        ground_plane = rep.functional.create.plane(parent=scene_root, name="GroundPlane", scale=(10, 10, 1))
        rep.functional.physics.apply_collider(ground_plane)
        app_utils.update_app()

    pallets = []
    pallets_urls, pallets_weights = zip(*pallets_urls_and_weights)
    pallet_weights_arr = np.asarray(pallets_weights, dtype=float)
    pallet_weights_arr /= pallet_weights_arr.sum()
    rand_pallet_urls = gen.choice(pallets_urls, size=num_pallets, p=pallet_weights_arr)
    custom_pallet_locations = [
        (-9.3, 5.3, 1.3),
        (-9.3, 7.3, 1.3),
        (-9.3, -0.6, 1.3),
    ]
    gen.shuffle(custom_pallet_locations)
    for i, pallet_url in enumerate(rand_pallet_urls):
        if env_url is not None:
            if i % 2 == 0 and custom_pallet_locations:
                rand_loc = custom_pallet_locations.pop()
            else:
                rand_loc = (
                    -6.5 + gen.uniform(-0.2, 0.2),
                    i * 1.75 + gen.uniform(0, 0.2),
                    gen.uniform(0, 0.2),
                )
        else:
            rand_loc = (
                i * 1.5 + gen.uniform(0, 0.2),
                gen.uniform(-0.2, 0.2),
                0.0,
            )
        rand_rot = (0, 0, gen.choice([180, 90, 0, -90, -180]) + gen.uniform(-15, 15))
        pallet_prim = create_asset_with_colliders(
            pallet_url,
            f"{scene_root}/Pallet_{i}",
            translations=rand_loc,
            orientations=transform_utils.euler_angles_to_quaternion(rand_rot, degrees=True).numpy(),
        )
        pallets.append(pallet_prim)

    walls_root = f"{scene_root}/_CollisionWalls"
    stage_utils.define_prim(walls_root, "Xform")

    total_boxes = []
    for pallet in pallets:
        drop_height = 1.0 if env_url is not None else 1.5
        rand_num_boxes = int(gen.integers(8, 16) if env_url is not None else gen.integers(12, 21))
        stacked_boxes = stack_boxes_on_pallet(
            pallet,
            walls_root,
            boxes_urls_and_weights,
            num_boxes=rand_num_boxes,
            drop_height=drop_height,
            gen=gen,
        )
        total_boxes.extend(stacked_boxes)

    if stage_utils.get_current_stage(backend="usd").GetPrimAtPath(walls_root).IsValid():
        stage_utils.delete_prim(walls_root)

    rep.functional.modify.attribute(total_boxes, "physics:rigidBodyEnabled", True)
    omni.physx.get_physx_simulation_interface().flush_changes()
    app_utils.play()
    app_utils.update_app(steps=200)
    app_utils.pause()

    if write_data:
        out_dir = os.path.join(os.getcwd(), "_out_box_stacking")
        print(f"Writing data to {out_dir}..")
        backend = rep.backends.get("DiskBackend")
        backend.initialize(output_dir=out_dir)
        writer = rep.WriterRegistry.get("BasicWriter")
        writer.initialize(backend=backend, rgb=True)
        cam = rep.functional.create.camera(position=(5, -5, 2), look_at=(0, 0, 0), name="PalletCamera")
        rp = rep.create.render_product(cam, resolution=(512, 512))
        writer.attach(rp)

        rep.orchestrator.step(rt_subframes=8)

        rep.orchestrator.wait_until_complete()
        writer.detach()
        rp.destroy()


def run_example(num_pallets: int, env_url: str | None, write_data: bool, rng: Any = None) -> None:
    """Run the pallet volume-filling example.

    Args:
        num_pallets: Number of pallets to populate.
        env_url: Environment URL, or None to create an empty stage.
        write_data: Whether to capture and write an RGB image.
        rng: Replicator random-number generator, or None to create one from the example seed.
    """
    run_box_stacking_scenarios(num_pallets=num_pallets, env_url=env_url, write_data=write_data, rng=rng)


run_example(args.num_pallets, args.env_url, args.write_data)

# <start-physics-based-randomized-volume-filling-test>
test_parser = argparse.ArgumentParser()
test_parser.add_argument(
    "--test",
    action="store_true",
    help="Validate captured output files against expected counts and exit.",
)
test_args, _ = test_parser.parse_known_args()

if test_args.test:
    import sys

    from isaacsim.core.utils.extensions import enable_extension

    enable_extension("isaacsim.test.utils")
    enable_extension("isaacsim.replicator.examples")
    from isaacsim.test.utils.file_validation import get_folder_file_summary, validate_folder_contents
    from isaacsim.test.utils.image_comparison import compare_images_in_directories

    if not args.write_data:
        print("[SDG][Test][SKIP] Output validation skipped because --no-write-data was set")
    else:
        rgb_mean_diff_tolerance = 5
        replicator_examples_ext_path = (
            omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module("isaacsim.replicator.examples")
        )
        golden_name = "_out_box_stacking_env_none" if args.env_url is None else "_out_box_stacking_env_warehouse"
        golden_dir = os.path.join(
            replicator_examples_ext_path,
            "isaacsim",
            "replicator",
            "examples",
            "tests",
            "data",
            "golden",
            golden_name,
        )
        out_dir = os.path.join(os.getcwd(), "_out_box_stacking")
        ok = validate_folder_contents(
            path=out_dir,
            recursive=True,
            expected_counts={"png": 1},
            fail_on_empty_files=True,
        )
        if not ok:
            summary = get_folder_file_summary(out_dir, recursive=True)
            print(f"[SDG][Test][FAIL] Output validation failed for {out_dir}: " f"expected png=1, found {summary}")
            sys.exit(1)

        rgb_result = compare_images_in_directories(
            golden_dir=golden_dir,
            test_dir=out_dir,
            path_pattern=r"^rgb_.*\.png$",
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=rgb_mean_diff_tolerance,
            print_all_stats=False,
        )
        if not rgb_result["all_passed"]:
            print(
                f"[SDG][Test][FAIL] RGB image comparison failed (tol={rgb_mean_diff_tolerance}). "
                f"Golden dir: {golden_dir}, output dir: {out_dir}"
            )
            sys.exit(1)
        print(f"[SDG][Test][PASS] Output validation succeeded for {out_dir}")
# <end-physics-based-randomized-volume-filling-test>

simulation_app.close()
