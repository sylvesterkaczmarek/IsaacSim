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

"""Generate synthetic data using object-based scene randomization."""

import argparse
import json
import os

import yaml
from isaacsim import SimulationApp

# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
# <start-config-snippet>
config = {
    "launch_config": {
        "renderer": "RealTimePathTracing",
        "headless": False,
    },
    "env_url": "",
    "working_area_size": (2.5, 2.5, 1.5),
    "rt_subframes": 4,
    "num_frames": 3,
    "num_cameras": 2,
    "camera_collider_radius": 0.25,
    "disable_render_products_between_captures": False,
    "simulation_duration_between_captures": 0.05,
    "resolution": (640, 480),
    "camera_properties_kwargs": {
        "focal_length": 24.0,
        "focus_distance": 400,
        "f_stop": 0.0,
        "clipping_range": (0.01, 10000),
    },
    "camera_look_at_target_offset": 0.1,
    "camera_distance_to_target_min_max": (0.35, 0.9),
    "writer_type": "PoseWriter",
    "writer_kwargs": {
        "output_dir": "_out_obj_based_sdg_pose_writer",
        "format": None,
        "use_subfolders": True,
        "write_debug_images": True,
        "skip_empty_frames": False,
    },
    "labeled_assets_and_properties": [
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
            "label": "pudding_box",
            "count": 4,
            "floating": True,
            "scale_min_max": (0.85, 1.25),
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned/011_banana.usd",
            "label": "banana",
            "count": 3,
            "floating": False,
            "scale_min_max": (0.85, 1.25),
        },
        {
            "url": "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
            "label": "mustard_bottle",
            "count": 4,
            "floating": True,
            "scale_min_max": (0.85, 1.25),
        },
    ],
    "shape_distractors_types": ["capsule", "cone", "cylinder", "sphere", "cube"],
    "shape_distractors_scale_min_max": (0.015, 0.15),
    "shape_distractors_num": 80,
    "mesh_distractors_urls": [
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd",
        "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
    ],
    "mesh_distractors_scale_min_max": (0.35, 1.0),
    "mesh_distractors_num": 4,
}
# <end-config-snippet>

# Check if there are any config files (yaml or json) are passed as arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
parser.add_argument(
    "--num-frames",
    type=int,
    default=None,
    help="Override the number of frames to capture.",
)
args, _ = parser.parse_known_args()
args_config = {}
if args.config and os.path.isfile(args.config):
    with open(args.config) as f:
        if args.config.endswith(".json"):
            args_config = json.load(f)
        elif args.config.endswith(".yaml"):
            args_config = yaml.safe_load(f)
        else:
            print(f"[SDG][WARN] File {args.config} is not json or yaml, will use default config")
else:
    print(f"[SDG][WARN] File {args.config} does not exist, will use default config")

# Update the default config dict with the external one
config.update(args_config)
if args.num_frames is not None:
    config["num_frames"] = args.num_frames

print(f"[SDG] Using config:\n{config}")

launch_config = config.get("launch_config", {})
simulation_app = SimulationApp(launch_config=launch_config)

import time
from itertools import chain

import carb
import carb.settings
import isaacsim.core.experimental.utils.app as app_utils

# Custom util functions for the example
import object_based_sdg_utils
import omni.kit.app
import omni.physics.core
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from isaacsim.core.experimental.objects import Sphere
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
from isaacsim.core.experimental.utils.semantics import add_labels, remove_all_labels, upgrade_prim_semantics_to_labels
from isaacsim.core.simulation_manager import PhysicsScene, SimulationManager
from isaacsim.storage.native import get_assets_root_path
from omni.physics.core import get_physics_scene_query_interface
from pxr import PhysicsSchemaTools

# Isaac nucleus assets root path
assets_root_path = get_assets_root_path()
stage = None

# Deterministic RNG for spawn and runtime randomizations
rep.set_global_seed(42)
rng = rep.rng.ReplicatorRNG(seed=42)

# ENVIRONMENT
# Create an empty or load a custom stage (clearing any previous semantics)
env_url = config.get("env_url", "")
if env_url:
    env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
    omni.usd.get_context().open_stage(env_path)
    stage = omni.usd.get_context().get_stage()
    # Remove any previous semantics in the loaded stage
    for prim in stage.Traverse():
        # Make sure old semantics api are upgraded to the new labels api
        upgrade_prim_semantics_to_labels(prim, include_descendants=True)
        remove_all_labels(prim, include_descendants=True)
else:
    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    rep.functional.create.xform(name="World")
    rep.functional.create.distant_light(intensity=400.0, rotation=(0, 60, 0), name="DistantLight")

# Get the working area size and bounds (width=x, depth=y, height=z)
working_area_size = config.get("working_area_size", (2.5, 2.5, 1.5))
working_area_min = (working_area_size[0] / -2, working_area_size[1] / -2, working_area_size[2] / -2)
working_area_max = (working_area_size[0] / 2, working_area_size[1] / 2, working_area_size[2] / 2)

# Create a collision box area around the assets to prevent them from drifting away
object_based_sdg_utils.create_collision_box_walls(
    "/World/CollisionWalls", working_area_size[0], working_area_size[1], working_area_size[2]
)

rep.functional.physics.create_physics_scene("/PhysicsScene", timeStepsPerSecond=60)
physics_scene = PhysicsScene("/PhysicsScene")

# <start-labeled-assets-snippet>
# TRAINING ASSETS
# Add the objects to be trained in the environment with their labels and properties
labeled_assets_and_properties = config.get("labeled_assets_and_properties", [])
floating_labeled_prims = []
falling_labeled_prims = []
labeled_prims = []
rep.functional.create.scope(name="Labeled", parent="/World")
for obj in labeled_assets_and_properties:
    obj_url = obj.get("url", "")
    label = obj.get("label", "unknown")
    count = obj.get("count", 1)
    floating = obj.get("floating", False)
    scale_min_max = obj.get("scale_min_max", (1, 1))
    for i in range(count):
        # Create a prim and add the asset reference
        rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
            rng, loc_min=working_area_min, loc_max=working_area_max, scale_min_max=scale_min_max
        )
        asset_path = obj_url if obj_url.startswith("omniverse://") else assets_root_path + obj_url
        prim = rep.functional.create.reference(
            usd_path=asset_path,
            parent="/World/Labeled",
            name=label,
            position=rand_loc,
            rotation=rand_rot,
            scale=rand_scale,
        )
        # Apply colliders and rigid body dynamics
        object_based_sdg_utils.add_colliders(str(prim.GetPrimPath()))
        rep.functional.physics.apply_rigid_body(prim, disableGravity=floating)
        # Label the asset (any previous 'class' label will be overwritten)
        add_labels(prim, labels=[label], taxonomy="class")
        if floating:
            floating_labeled_prims.append(prim)
        else:
            falling_labeled_prims.append(prim)
labeled_prims = floating_labeled_prims + falling_labeled_prims
# <end-labeled-assets-snippet>


# <start-shape-distractors-snippet>
# DISTRACTORS
# Add shape distractors to the environment as floating or falling objects
rep.functional.create.scope(name="Distractors", parent="/World")
shape_distractors_types = config.get("shape_distractors_types", ["capsule", "cone", "cylinder", "sphere", "cube"])
shape_distractors_scale_min_max = config.get("shape_distractors_scale_min_max", (0.02, 0.2))
shape_distractors_num = config.get("shape_distractors_num", 80)
shape_distractors = []
floating_shape_distractors = []
falling_shape_distractors = []
shape_creators = {
    "capsule": rep.functional.create.capsule,
    "cone": rep.functional.create.cone,
    "cube": rep.functional.create.cube,
    "cylinder": rep.functional.create.cylinder,
    "sphere": rep.functional.create.sphere,
}
for i in range(shape_distractors_num):
    rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
        rng, loc_min=working_area_min, loc_max=working_area_max, scale_min_max=shape_distractors_scale_min_max
    )
    rand_shape = shape_distractors_types[int(rng.generator.integers(0, len(shape_distractors_types)))]
    prim = shape_creators[rand_shape](
        parent="/World/Distractors",
        name=rand_shape,
        position=rand_loc,
        rotation=rand_rot,
        scale=rand_scale,
    )
    disable_gravity = bool(rng.generator.integers(0, 2))
    object_based_sdg_utils.add_colliders(str(prim.GetPrimPath()))
    rep.functional.physics.apply_rigid_body(prim, disableGravity=disable_gravity)
    if disable_gravity:
        floating_shape_distractors.append(prim)
    else:
        falling_shape_distractors.append(prim)
    shape_distractors.append(prim)
# <end-shape-distractors-snippet>

# Add mesh distractors to the environment as floating of falling objects
mesh_distactors_urls = config.get("mesh_distractors_urls", [])
mesh_distactors_scale_min_max = config.get("mesh_distractors_scale_min_max", (0.1, 2.0))
mesh_distactors_num = config.get("mesh_distractors_num", 4)
mesh_distractors = []
floating_mesh_distractors = []
falling_mesh_distractors = []
for i in range(mesh_distactors_num):
    rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
        rng, loc_min=working_area_min, loc_max=working_area_max, scale_min_max=mesh_distactors_scale_min_max
    )
    mesh_url = mesh_distactors_urls[int(rng.generator.integers(0, len(mesh_distactors_urls)))]
    prim_name = os.path.basename(mesh_url).split(".")[0]
    asset_path = mesh_url if mesh_url.startswith("omniverse://") else assets_root_path + mesh_url
    prim = rep.functional.create.reference(
        usd_path=asset_path,
        parent="/World/Distractors",
        name=prim_name,
        position=rand_loc,
        rotation=rand_rot,
        scale=rand_scale,
    )
    disable_gravity = bool(rng.generator.integers(0, 2))
    object_based_sdg_utils.add_colliders(str(prim.GetPrimPath()))
    rep.functional.physics.apply_rigid_body(prim, disableGravity=disable_gravity)
    if disable_gravity:
        floating_mesh_distractors.append(prim)
    else:
        falling_mesh_distractors.append(prim)
    mesh_distractors.append(prim)
    # Remove any previous semantics on the mesh distractor
    upgrade_prim_semantics_to_labels(prim, include_descendants=True)
    remove_all_labels(prim, include_descendants=True)

# REPLICATOR

# Disable capturing every frame (capture will be triggered manually using the step function)
rep.orchestrator.set_capture_on_play(False)

# Set DLSS to Quality mode (2) for best SDG results , options: 0 (Performance), 1 (Balanced), 2 (Quality), 3 (Auto)
carb.settings.get_settings().set("rtx/post/dlss/execMode", 2)

# Create the camera prims and their properties
cameras = []
num_cameras = config.get("num_cameras", 1)
camera_properties_kwargs = config.get("camera_properties_kwargs", {})
rep.functional.create.scope(name="Cameras", parent="/World")
for i in range(num_cameras):
    cam_prim = rep.functional.create.camera(parent="/World/Cameras", name=f"cam_{i}", **camera_properties_kwargs)
    cameras.append(cam_prim)

# Add collision spheres (disabled by default) to cameras to avoid objects overlaping with the camera view
camera_colliders = []
camera_collider_geoms = []
camera_collider_radius = config.get("camera_collider_radius", 0)
if camera_collider_radius > 0:
    for cam in cameras:
        cam_path = cam.GetPath()
        cam_collider_path = f"{cam_path}/CollisionSphere"
        cam_collider = Sphere(cam_collider_path, radii=camera_collider_radius)
        cam_collider_geom = GeomPrim(cam_collider_path, apply_collision_apis=True)
        cam_collider_geom.set_enabled_collisions([False])
        cam_collider.set_visibilities([False])
        camera_colliders.append(cam_collider.prims[0])
        camera_collider_geoms.append(cam_collider_geom)

# Wait an app update to ensure the prim changes are applied
simulation_app.update()

# Create render products using the cameras
render_products = []
resolution = config.get("resolution", (640, 480))
for cam in cameras:
    rp = rep.create.render_product(cam.GetPath(), resolution, name=cam.GetName())
    render_products.append(rp)

# Enable rendering only at capture time
disable_render_products_between_captures = config.get("disable_render_products_between_captures", True)
if disable_render_products_between_captures:
    object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

# Create the writer and attach the render products
writer_type = config.get("writer_type", None)
writer_kwargs = config.get("writer_kwargs", {})
# If not an absolute path, set it relative to the current working directory
if out_dir := writer_kwargs.get("output_dir"):
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
        writer_kwargs["output_dir"] = out_dir
    print(f"[SDG] Writing data to: {out_dir}")
if writer_type is not None and len(render_products) > 0:
    writer = rep.writers.get(writer_type)
    writer.initialize(**writer_kwargs)
    writer.attach(render_products)
else:
    print("[SDG][WARN] No writer attached, the capture loop will not write any data to disk")


# <start-overlap-randomizer-snippet>
# RANDOMIZERS
def on_overlap_hit(hit: object) -> bool:
    """Apply a random upwards velocity to objects overlapping the bounce area.

    Args:
        hit: PhysX overlap result identifying the rigid body in the bounce area.

    Returns:
        ``True`` so the overlap query continues visiting additional bodies.
    """
    prim_path = str(PhysicsSchemaTools.intToSdfPath(hit.rigid_body))
    prim = stage.GetPrimAtPath(prim_path)
    # Skip the camera collision spheres
    if prim not in camera_colliders:
        rand_vel = (
            float(rng.generator.uniform(-2, 2)),
            float(rng.generator.uniform(-2, 2)),
            float(rng.generator.uniform(4, 8)),
        )
        RigidPrim(prim_path).set_velocities(linear_velocities=[rand_vel])
    return True  # return True to continue the query


# Area to check for overlapping objects (above the bottom collision box)
overlap_area_thickness = 0.1
overlap_area_origin = (0, 0, (-working_area_size[2] / 2) + (overlap_area_thickness / 2))
overlap_area_extent = (
    working_area_size[0] / 2 * 0.99,
    working_area_size[1] / 2 * 0.99,
    overlap_area_thickness / 2 * 0.99,
)


def on_physics_step(dt: float, context: object) -> None:
    """Check for overlapping objects on every physics update step.

    Args:
        dt: Duration of the physics step. The callback does not use it.
        context: Physics callback context. The callback does not use it.
    """
    get_physics_scene_query_interface().overlap_box(
        carb.Float3(overlap_area_extent),
        carb.Float3(overlap_area_origin),
        carb.Float4(0, 0, 0, 1),
        on_overlap_hit,
    )


# Subscribe to the physics step events to check for objects overlapping the 'bounce' area
physics_sub = omni.physics.core.get_physics_simulation_interface().subscribe_physics_on_step_events(
    pre_step=False, order=0, on_update=on_physics_step
)
# <end-overlap-randomizer-snippet>


def unsubscribe_physics() -> None:
    """Release the physics step callback subscription."""
    global physics_sub
    if physics_sub is not None:
        physics_sub.unsubscribe()
        physics_sub = None


camera_distance_to_target_min_max = config.get("camera_distance_to_target_min_max", (0.1, 0.5))
camera_look_at_target_offset = config.get("camera_look_at_target_offset", 0.2)


def randomize_camera_poses() -> None:
    """Randomize camera poses to look at a random target asset with random distance and offset."""
    for cam in cameras:
        target_asset = labeled_prims[int(rng.generator.integers(0, len(labeled_prims)))]
        # Add a look_at offset so the target is not always in the center of the camera view
        loc_offset = rng.generator.uniform(-camera_look_at_target_offset, camera_look_at_target_offset, size=3)
        target_loc = XformPrim(str(target_asset.GetPrimPath())).get_local_poses()[0].numpy()[0] + loc_offset
        distance = float(
            rng.generator.uniform(camera_distance_to_target_min_max[0], camera_distance_to_target_min_max[1])
        )
        cam_loc, quat = object_based_sdg_utils.get_random_pose_on_sphere(
            rng, origin=tuple(target_loc.tolist()), radius=distance
        )
        XformPrim(str(cam.GetPrimPath()), reset_xform_op_properties=True).set_local_poses(
            translations=[cam_loc], orientations=[quat]
        )


def simulate_camera_collision(num_simulation_steps: int = 1) -> None:
    """Enable camera colliders temporarily and simulate to push out overlapping objects.

    Args:
        num_simulation_steps: Number of physics steps during which camera collisions remain enabled.
    """
    for cam_collider_geom in camera_collider_geoms:
        cam_collider_geom.set_enabled_collisions([True])
    if not app_utils.is_playing():
        app_utils.play()
    SimulationManager.step(steps=num_simulation_steps)
    for cam_collider_geom in camera_collider_geoms:
        cam_collider_geom.set_enabled_collisions([False])


# <start-colors-randomizer-snippet>
# Create a randomizer for the shape distractors colors, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_shape_distractor_colors"):
    shape_distractors_paths = [prim.GetPath() for prim in chain(floating_shape_distractors, falling_shape_distractors)]
    shape_distractors_group = rep.create.group(shape_distractors_paths)
    with shape_distractors_group:
        rep.randomizer.color(colors=rep.distribution.uniform((0, 0, 0), (1, 1, 1)))
# <end-colors-randomizer-snippet>


# <start-lights-randomizer-snippet>
# Create a randomizer for lights in the working area, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_lights"):
    lights = rep.create.light(
        light_type="Sphere",
        color=rep.distribution.uniform((0, 0, 0), (1, 1, 1)),
        temperature=rep.distribution.normal(6500, 500),
        intensity=rep.distribution.normal(35000, 5000),
        position=rep.distribution.uniform(working_area_min, working_area_max),
        scale=rep.distribution.uniform(0.1, 1),
        count=3,
    )
# <end-lights-randomizer-snippet>


# Create a randomizer for the dome background, manually triggered at custom events
with rep.trigger.on_custom_event(event_name="randomize_dome_background"):
    dome_textures = [
        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/hotel_room_4k.hdr",
        assets_root_path + "/NVIDIA/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
    ]
    dome_light = rep.create.light(light_type="Dome")
    with dome_light:
        rep.modify.attribute("inputs:texture:file", rep.distribution.choice(dome_textures))
        rep.randomizer.rotation()


def capture_with_motion_blur_and_pathtracing(
    physics_scene: PhysicsScene,
    duration: float = 1.0 / 90.0,
    num_samples: int = 16,
    spp: int = 64,
) -> None:
    """Capture motion blur by combining pathtraced subframe samples simulated for the given duration.

    On normal completion, restore the prior render mode, motion-blur enablement, and physics rate.

    Args:
        physics_scene: Physics scene whose step rate may be raised for the capture.
        duration: Positive simulated exposure interval in seconds.
        num_samples: Positive number of motion-blur subframes to accumulate.
        spp: Positive number of path-tracing samples per pixel for each captured frame.
    """
    original_physics_dt = physics_scene.get_dt()
    target_physics_fps = 1 / duration * num_samples
    target_physics_dt = 1.0 / target_physics_fps
    if target_physics_dt < original_physics_dt:
        print(f"[SDG] Changing physics FPS from {1.0 / original_physics_dt:.0f} to {target_physics_fps:.0f}")
        physics_scene.set_dt(target_physics_dt)

    is_motion_blur_enabled = carb.settings.get_settings().get("/omni/replicator/captureMotionBlur")
    if not is_motion_blur_enabled:
        carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
    carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)

    # Set the render mode to PathTracing
    prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
    carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
    carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
    carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

    # Make sure the timeline is playing
    if not app_utils.is_playing():
        app_utils.play()

    # Capture the frame by advancing the simulation for the given duration and combining the sub samples
    rep.orchestrator.step(delta_time=duration, pause_timeline=False)

    # Restore the original physics FPS
    if target_physics_dt < original_physics_dt:
        print(f"[SDG] Restoring physics FPS from {1.0 / target_physics_dt:.0f} " f"to {1.0 / original_physics_dt:.0f}")
        physics_scene.set_dt(original_physics_dt)

    carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", is_motion_blur_enabled)
    carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)


def run_simulation_loop(duration: float) -> None:
    """Update the app until a given simulation duration has passed.

    Args:
        duration: Amount of simulation time to advance in seconds.
    """
    timeline = omni.timeline.get_timeline_interface()
    elapsed_time = 0.0
    previous_time = timeline.get_current_time()
    if not app_utils.is_playing():
        app_utils.play()
    app_updates_counter = 0
    while elapsed_time <= duration:
        simulation_app.update()
        elapsed_time += timeline.get_current_time() - previous_time
        previous_time = timeline.get_current_time()
        app_updates_counter += 1
        print(
            f"\t Simulation loop at {timeline.get_current_time():.2f}, current elapsed time: {elapsed_time:.2f}, counter: {app_updates_counter}"
        )
    print(
        f"[SDG] Simulation loop finished in {elapsed_time:.2f} seconds at {timeline.get_current_time():.2f} with {app_updates_counter} app updates."
    )


# SDG
# Number of frames to capture
num_frames = config.get("num_frames", 3)

# Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
# see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
rt_subframes = config.get("rt_subframes", -1)

# Amount of simulation time to wait between captures
sim_duration_between_captures = config.get("simulation_duration_between_captures", 0.025)

# Initial trigger for randomizers before the SDG loop with several app updates (ensures materials/textures are loaded)
rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
rep.utils.send_og_event(event_name="randomize_dome_background")
for _ in range(5):
    simulation_app.update()

# Set the timeline parameters (start, end, no looping) and start the timeline
timeline = omni.timeline.get_timeline_interface()
timeline.set_start_time(0)
timeline.set_end_time(1000000)
timeline.set_looping(False)
# If no custom physx scene is created, a default one will be created by the physics engine once the timeline starts
physics_scene.set_dt(1.0 / 60.0)
SimulationManager.initialize_physics()
app_utils.play(commit=True)
simulation_app.update()

# Store the wall start time for stats
wall_time_start = time.perf_counter()

# <start-sdg-loop-snippet>
# Run the simulation and capture data triggering randomizations and actions at custom frame intervals
for i in range(num_frames):
    if i % 2 == 0:
        print(f"\t Randomizing camera poses")
        randomize_camera_poses()
        # Temporarily enable camera colliders and simulate for a few frames to push out any overlapping objects
        if camera_colliders:
            simulate_camera_collision(num_simulation_steps=2)
        print(f"\t Randomizing lights")
        rep.utils.send_og_event(event_name="randomize_lights")
        print(f"\t Randomizing shape distractors colors")
        rep.utils.send_og_event(event_name="randomize_shape_distractor_colors")
        object_based_sdg_utils.apply_random_velocities(
            [str(p.GetPrimPath()) for p in chain(floating_shape_distractors, floating_mesh_distractors)], rng
        )
    if i % 4 == 0:
        print(f"\t Applying velocity towards the origin")
        object_based_sdg_utils.apply_velocities_towards_target(
            [str(p.GetPrimPath()) for p in chain(labeled_prims, shape_distractors, mesh_distractors)], rng
        )
        print(f"\t Randomizing dome background")
        rep.utils.send_og_event(event_name="randomize_dome_background")

    # Enable render products only at capture time
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

    # Capture the current frame
    print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
    if (i + 1) % 3 == 0:
        capture_with_motion_blur_and_pathtracing(physics_scene, duration=1.0 / 90.0, num_samples=16, spp=32)
    else:
        rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

    # Disable render products between captures
    if disable_render_products_between_captures:
        object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

    # Run the simulation for a given duration between frame captures
    if sim_duration_between_captures > 0:
        run_simulation_loop(duration=sim_duration_between_captures)
    else:
        simulation_app.update()

# Wait for the data to be written (default writer backends are asynchronous)
rep.orchestrator.wait_until_complete()
# <end-sdg-loop-snippet>

# Get the stats
wall_duration = time.perf_counter() - wall_time_start
sim_duration = timeline.get_current_time()
avg_frame_fps = num_frames / wall_duration
num_captures = num_frames * num_cameras
avg_capture_fps = num_captures / wall_duration
print(
    f"[SDG] Captured {num_frames} frames, {num_captures} entries (frames * cameras) in {wall_duration:.2f} seconds.\n"
    f"\t Simulation duration: {sim_duration:.2f}\n"
    f"\t Simulation duration between captures: {sim_duration_between_captures:.2f}\n"
    f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
    f"\t Average capture entries (frames * cameras) FPS: {avg_capture_fps:.2f}\n"
)

unsubscribe_physics()
simulation_app.update()
app_utils.stop()

# <start-object-based-sdg-test>
test_parser = argparse.ArgumentParser()
test_parser.add_argument(
    "--test",
    action="store_true",
    help="Validate captured output files against expected counts and exit.",
)
test_args, _ = test_parser.parse_known_args()

if test_args.test:
    import sys

    app_utils.enable_extension("isaacsim.test.utils")
    app_utils.enable_extension("isaacsim.replicator.examples")
    from isaacsim.test.utils.file_validation import get_folder_file_summary, validate_folder_contents
    from isaacsim.test.utils.image_comparison import compare_images_in_directories

    rgb_mean_diff_tolerance = 5
    writer_type = config.get("writer_type", "PoseWriter")
    writer_kwargs = config.get("writer_kwargs", {})
    out_dir = writer_kwargs.get("output_dir")
    if out_dir and not os.path.isabs(out_dir):
        out_dir = os.path.join(os.getcwd(), out_dir)
    num_frames = config.get("num_frames", 3)
    num_cameras = config.get("num_cameras", 1)
    num_captures = num_frames * num_cameras
    if writer_type == "PoseWriter":
        # PoseWriter with use_subfolders numbers 000000..num_frames-1 in each camera folder.
        # write_debug_images adds 1 overlay png per capture (000000.png + 000000_overlay.png).
        expected_counts = {"json": num_captures, "png": num_captures * 2}
        if not writer_kwargs.get("write_debug_images", False):
            expected_counts = {"json": num_captures, "png": num_captures}
    elif writer_type == "BasicWriter":
        # BasicWriter with rgb + semantic_segmentation (colorize default True) writes
        # 1 rgb png + 1 colorized seg png + 1 labels json per capture.
        expected_counts = {"png": num_captures * 2, "json": num_captures}
    else:
        print(f"[SDG][Test][FAIL] Unsupported writer_type for validation: {writer_type}")
        sys.exit(1)

    ok = validate_folder_contents(
        path=out_dir,
        recursive=True,
        expected_counts=expected_counts,
        fail_on_empty_files=True,
    )
    if not ok:
        summary = get_folder_file_summary(out_dir, recursive=True)
        print(
            f"[SDG][Test][FAIL] Output validation failed for {out_dir}\n"
            f"\t Expected: {expected_counts}\n"
            f"\t Found: {summary['extension_counts']}"
        )
        sys.exit(1)

    replicator_examples_ext_path = (
        omni.kit.app.get_app().get_extension_manager().get_extension_path_by_module("isaacsim.replicator.examples")
    )
    golden_dir = os.path.join(
        replicator_examples_ext_path,
        "isaacsim",
        "replicator",
        "examples",
        "tests",
        "data",
        "golden",
        "_out_obj_based_sdg_pose_writer",
    )
    compare_rgb = os.path.basename(os.path.normpath(out_dir)) == "_out_obj_based_sdg_pose_writer"
    rgb_results = []
    if compare_rgb:
        for golden_root, _, golden_files in os.walk(golden_dir):
            if any(file_name.endswith(".png") for file_name in golden_files):
                relative_root = os.path.relpath(golden_root, golden_dir)
                test_root = out_dir if relative_root == "." else os.path.join(out_dir, relative_root)
                rgb_results.append(
                    compare_images_in_directories(
                        golden_dir=golden_root,
                        test_dir=test_root,
                        path_pattern=r"^\d+\.png$",
                        allclose_rtol=None,
                        allclose_atol=None,
                        mean_tolerance=rgb_mean_diff_tolerance,
                        print_all_stats=False,
                    )
                )
    if compare_rgb and (not rgb_results or not all(result["all_passed"] for result in rgb_results)):
        print(
            f"[SDG][Test][FAIL] RGB image comparison failed (tol={rgb_mean_diff_tolerance}). "
            f"Golden dir: {golden_dir}, output dir: {out_dir}"
        )
        sys.exit(1)
    print(f"[SDG][Test][PASS] Output validation succeeded for {out_dir} ({num_captures} captures)")
# <end-object-based-sdg-test>

simulation_app.close()
