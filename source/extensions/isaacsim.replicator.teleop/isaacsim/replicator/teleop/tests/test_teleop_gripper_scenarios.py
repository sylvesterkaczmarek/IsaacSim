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

"""Golden-image tests for the Script Editor floating-gripper scenario."""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.bounds as bounds_utils
import isaacsim.core.experimental.utils.prim as prim_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.core.experimental.utils.transform as transform_utils
import numpy as np
import omni.kit.test
import omni.replicator.core as rep
import omni.usd
from isaacsim.core.experimental.objects import Camera, Cube, DistantLight, DomeLight, GroundPlane
from isaacsim.core.experimental.prims import GeomPrim, RigidPrim, XformPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.test.utils.file_validation import validate_folder_contents
from isaacsim.test.utils.image_comparison import compare_images_in_directories
from omni.replicator.core.functional import write_image

if TYPE_CHECKING:
    from isaacsim.replicator.teleop import Pose

TELEOP_ROOT = "/World/TeleopScenario"
TABLE_PATH = "/World/Objects/TableCube"
BASE_TOP_Z = 1.0
RED_CUBE_SCALE = (0.14, 0.14, 0.06)
BLUE_CUBE_SCALE = (0.05, 0.05, 0.15)
RNG_SEED = 42
TABLE_EDGE_MARGIN = 0.08
START_DISTANCE_FROM_TABLE = 0.25
VISUAL_CUE_REFERENCE_Z = 0.0
GRASP_HEIGHT_OFFSET = 0.04
START_HEIGHT = 0.25
LIFT_HEIGHT = 0.30
PLACE_CLEARANCE = 0.02
PHASE_SAMPLES = 40
POSITION_TOLERANCE = 0.015
MAX_STEPS_PER_TARGET = 20
TARGET_ROTATION_DEGREES = (90.0, 0.0, 0.0)
CAMERA_RESOLUTION = (640, 480)
TCP_CAMERA_DISTANCE = 1.25
CAPTURE_PHASES = ("before_grasp", "after_grasp", "after_lift", "after_place")

SIDE_LAYOUT = {
    "left": {
        "cube": "/World/Objects/RedCube",
        "color": "red",
        "scale": RED_CUBE_SCALE,
        "mass": 0.5,
    },
    "right": {
        "cube": "/World/Objects/BlueCube",
        "drop_cube": "/World/Objects/RedCube",
        "color": "blue",
        "scale": BLUE_CUBE_SCALE,
        "mass": 0.05,
    },
}


async def _capture_phase_async(capture: dict[str, dict], output_dir: str, phase: str) -> None:
    """Capture one phase and write RGB plus raw and preview depth outputs."""
    render_products = tuple(capture["render_products"].values())
    for render_product in render_products:
        render_product.hydra_texture.set_updates_enabled(True)
    try:
        await rep.orchestrator.step_async(rt_subframes=4, delta_time=0.0)
        for name, annotators in capture["annotators"].items():
            view_dir = Path(output_dir) / name
            view_dir.mkdir(parents=True, exist_ok=True)
            rgb = np.asarray(annotators["rgb"].get_data())
            depth = np.asarray(annotators["depth"].get_data(), dtype=np.float32)
            write_image(path=str(view_dir / f"rgb_{phase}.png"), data=rgb)
            np.save(view_dir / f"depth_{phase}.npy", depth)
            valid = np.isfinite(depth) & (depth > 0.0)
            preview = np.zeros(depth.shape, dtype=np.uint8)
            if np.any(valid):
                values = depth[valid]
                near, far = np.percentile(values, (1.0, 99.0))
                if far <= near:
                    preview[valid] = 255
                else:
                    normalized = np.clip((depth[valid] - near) / (far - near), 0.0, 1.0)
                    preview[valid] = np.asarray((1.0 - normalized) * 255.0, dtype=np.uint8)
            write_image(path=str(view_dir / f"depth_{phase}.png"), data=preview)
    finally:
        for render_product in render_products:
            render_product.hydra_texture.set_updates_enabled(False)
        # Stateless Replicator stepping can pause the timeline; resume controller updates.
        app_utils.play()
        await app_utils.update_app_async()


def _poses_for_tcp_positions(
    grippers: dict[str, str],
    tcp_positions: dict[str, tuple[float, float, float]],
    target_rotation_degrees: tuple[float, float, float],
) -> dict[str, Pose]:
    """Convert externally supplied TCP positions to floating-root targets."""
    from isaacsim.replicator.teleop import get_floating_gripper_spec, make_pose

    targets = {}
    for side, gripper_name in grippers.items():
        spec = get_floating_gripper_spec(gripper_name)
        target_orientation = tuple(
            float(value)
            for value in transform_utils.euler_angles_to_quaternion(target_rotation_degrees, degrees=True)
            .numpy()
            .reshape(4)
        )
        rotation_offset = transform_utils.euler_angles_to_quaternion(
            spec["floating_controller"]["rotation_offset_degrees"],
            degrees=True,
            extrinsic=False,
        ).numpy()
        gripper_orientation = transform_utils.quaternion_multiplication(
            np.asarray(target_orientation), rotation_offset
        ).numpy()
        gripper_rotation = transform_utils.quaternion_to_rotation_matrix(gripper_orientation).numpy().reshape(3, 3)
        root_position = np.asarray(tcp_positions[side], dtype=np.float64) - gripper_rotation @ np.asarray(
            spec["tcp"]["translation"]
        )
        targets[side] = make_pose(root_position, target_orientation)
    return targets


def _positions_above(
    tcp_positions: dict[str, tuple[float, float, float]], height: float
) -> dict[str, tuple[float, float, float]]:
    """Return TCP positions translated upward in world Z."""
    return {side: (position[0], position[1], position[2] + height) for side, position in tcp_positions.items()}


def _create_randomized_cubes_on_table(
    table_path: str,
    side_layout: dict[str, dict[str, object]],
    rng_seed: int,
    table_edge_margin: float,
) -> np.ndarray:
    """Create rigid cubes at seeded positions on opposite table halves and return the table's world AABB."""
    table_aabb = bounds_utils.compute_aabb(
        table_path,
        bbox_cache=bounds_utils.create_bbox_cache(),
        space="world",
    )
    table_center_x = (table_aabb[0] + table_aabb[3]) / 2.0
    rng = np.random.default_rng(rng_seed)
    for side in ("left", "right"):
        layout = side_layout[side]
        path = layout["cube"]
        scale = np.asarray(layout["scale"], dtype=np.float64)
        half_extent = scale / 2.0
        x_range = (
            (
                table_aabb[0] + half_extent[0] + table_edge_margin,
                table_center_x - table_edge_margin,
            )
            if side == "left"
            else (
                table_center_x + table_edge_margin,
                table_aabb[3] - half_extent[0] - table_edge_margin,
            )
        )
        y_range = (
            table_aabb[1] + half_extent[1] + table_edge_margin,
            table_aabb[4] - half_extent[1] - table_edge_margin,
        )
        if x_range[0] > x_range[1] or y_range[0] > y_range[1]:
            raise ValueError(f"{table_path} is too small to place the {side} cube with the requested margin")
        position = (
            rng.uniform(*x_range),
            rng.uniform(*y_range),
            table_aabb[5] + half_extent[2],
        )
        carb.log_info(f"Seeded {layout['color']} cube position: {tuple(float(value) for value in position)}")
        Cube(path, sizes=1.0, positions=position, scales=layout["scale"], colors=layout["color"])
        GeomPrim(path, apply_collision_apis=True)
        RigidPrim(path, masses=[layout["mass"]])
    return table_aabb


async def _goto_tcp_positions_async(
    context: dict[str, object],
    grippers: dict[str, str],
    name: str,
    tcp_positions: dict[str, tuple[float, float, float]],
) -> None:
    """Move to explicit TCP positions supplied by the caller."""
    from isaacsim.replicator.teleop import move_floating_grippers_to_targets_async

    carb.log_info(f"Teleop scenario phase: {name}")
    result = await move_floating_grippers_to_targets_async(
        context,
        _poses_for_tcp_positions(grippers, tcp_positions, TARGET_ROTATION_DEGREES),
        sample_count=PHASE_SAMPLES,
        position_tolerance=POSITION_TOLERANCE,
        orientation_tolerance=math.pi,
        max_steps_per_target=MAX_STEPS_PER_TARGET,
    )
    if not result["reached"]:
        raise RuntimeError(
            f"Phase '{name}' timed out after {result['completed_samples']}/{PHASE_SAMPLES} samples "
            f"with position error {result['position_error']:.4f} m"
        )


async def run_gripper_scenario_async(name: str, gripper_name: str) -> list[str]:
    """Run the Script Editor scenario for one supported right-side gripper."""
    from isaacsim.replicator.teleop import (
        get_floating_gripper_spec,
        load_floating_grippers_async,
        set_floating_gripper_grasp_async,
        setup_floating_gripper_controllers_async,
        stop_floating_gripper_controllers,
    )

    grippers = {"right": gripper_name}
    await stage_utils.create_new_stage_async(template="empty")
    for path in ("/World", "/World/Lights", "/World/Objects"):
        stage_utils.define_prim(path, "Xform")

    DomeLight("/World/Lights/DomeLight").set_intensities([400.0])
    distant = DistantLight("/World/Lights/DistantLight", orientations=[(0.887, -0.41, 0.18, 0.09)])
    distant.set_intensities([1500.0])
    SimulationManager.setup_simulation()

    GroundPlane(
        "/World/CollisionFloor",
        sizes=10.0,
        positions=(0.0, 0.0, 0.0),
        colors=(0.18, 0.18, 0.2),
        templates=None,
    )
    Cube(
        TABLE_PATH,
        sizes=1.0,
        positions=(0.0, 0.0, BASE_TOP_Z / 2.0),
        scales=(1.0, 1.0, BASE_TOP_Z),
        colors=(0.35, 0.35, 0.38),
    )
    GeomPrim(TABLE_PATH, apply_collision_apis=True)
    table_aabb = _create_randomized_cubes_on_table(TABLE_PATH, SIDE_LAYOUT, RNG_SEED, TABLE_EDGE_MARGIN)

    start_tcp_position = (
        float((table_aabb[0] + table_aabb[3]) / 2.0),
        float(table_aabb[1] - START_DISTANCE_FROM_TABLE),
        float(table_aabb[5] + START_HEIGHT),
    )
    start_tcp_positions = {"right": start_tcp_position}
    start_targets = _poses_for_tcp_positions(grippers, start_tcp_positions, TARGET_ROTATION_DEGREES)
    gripper_paths = await load_floating_grippers_async(
        grippers,
        start_targets,
        teleop_root=TELEOP_ROOT,
        disable_gravity=True,
    )
    assert prim_utils.is_prim_valid(gripper_paths["right"]["tcp"])
    context = await setup_floating_gripper_controllers_async(
        grippers,
        gripper_paths,
        start_targets,
        visual_cues=True,
        visual_cue_reference_z=VISUAL_CUE_REFERENCE_Z,
    )
    assert context["visual_cues"].get_override_prim_path("right") == gripper_paths["right"]["tcp"]

    stage_utils.define_prim("/World/Cameras", "Xform")
    scene_camera_path = "/World/Cameras/SceneCameraTop"
    cube_aabb = bounds_utils.compute_combined_aabb([layout["cube"] for layout in SIDE_LAYOUT.values()])
    scene_target = tuple(float(value) for value in (cube_aabb[:3] + cube_aabb[3:]) / 2.0)
    scene_eye = (scene_target[0], scene_target[1], scene_target[2] + 2.0)
    scene_orientation = transform_utils.look_at_quaternion(scene_eye, scene_target).numpy()
    Camera(scene_camera_path).set_world_poses(positions=[scene_eye], orientations=[scene_orientation])
    cameras = {"scene_top": scene_camera_path}

    target_orientation = transform_utils.euler_angles_to_quaternion(TARGET_ROTATION_DEGREES, degrees=True).numpy()
    target_rotation = transform_utils.quaternion_to_rotation_matrix(target_orientation).numpy().reshape(3, 3)
    finger_forward = target_rotation[:, 2]
    world_up = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    side_direction = np.cross(finger_forward, world_up)
    side_direction /= np.linalg.norm(side_direction)
    paths = gripper_paths["right"]
    spec = get_floating_gripper_spec(gripper_name)
    root_orientation = XformPrim(paths["rigid_root"]).get_world_poses()[1].numpy().reshape(4)
    tcp_orientation = XformPrim(paths["tcp"]).get_world_poses()[1].numpy().reshape(4)
    root_to_tcp = transform_utils.quaternion_multiplication(
        transform_utils.quaternion_conjugate(root_orientation), tcp_orientation
    ).numpy()
    rotation_offset = transform_utils.euler_angles_to_quaternion(
        spec["floating_controller"]["rotation_offset_degrees"],
        degrees=True,
        extrinsic=False,
    ).numpy()
    desired_root_orientation = transform_utils.quaternion_multiplication(target_orientation, rotation_offset).numpy()
    desired_tcp_orientation = transform_utils.quaternion_multiplication(desired_root_orientation, root_to_tcp).numpy()
    camera_root = f"{paths['tcp']}/Cameras"
    stage_utils.define_prim(camera_root, "Xform")
    XformPrim(camera_root, reset_xform_op_properties=True).set_local_poses(
        translations=[[0.0, 0.0, 0.0]],
        orientations=[transform_utils.quaternion_conjugate(desired_tcp_orientation).numpy()],
    )
    camera_prefix = f"Right_{spec['base_link_name']}"
    for view_name, (view_direction, camera_up) in {
        "TCPCameraTop": (world_up, finger_forward),
        "TCPCameraSide": (side_direction, world_up),
    }.items():
        camera_path = f"{camera_root}/{view_name}"
        eye = view_direction * TCP_CAMERA_DISTANCE
        orientation = transform_utils.look_at_quaternion(eye, (0.0, 0.0, 0.0), up=camera_up).numpy()
        Camera(camera_path).set_local_poses(translations=[eye], orientations=[orientation])
        cameras[f"{camera_prefix}_{view_name}"] = camera_path

    output_dir = os.path.join(os.getcwd(), f"_out_teleop_{name}")
    shutil.rmtree(output_dir, ignore_errors=True)
    capture = {"render_products": {}, "annotators": {}}
    for camera_name, camera_path in cameras.items():
        render_product = rep.create.render_product(camera_path, CAMERA_RESOLUTION, name=f"Teleop_{camera_name}")
        render_product.hydra_texture.set_updates_enabled(False)
        rgb = rep.annotators.get("rgb")
        depth = rep.annotators.get("distance_to_image_plane")
        rgb.attach(render_product)
        depth.attach(render_product)
        capture["render_products"][camera_name] = render_product
        capture["annotators"][camera_name] = {"rgb": rgb, "depth": depth}
    rep.orchestrator.set_capture_on_play(False)

    pick_tcp_positions = {}
    drop_tcp_positions = {}
    bbox_cache = bounds_utils.create_bbox_cache()
    layout = SIDE_LAYOUT["right"]
    pick_path = layout["cube"]
    drop_path = layout["drop_cube"]
    pick_center = RigidPrim(pick_path).get_world_poses()[0].numpy().reshape(-1, 3)[0]
    drop_center = RigidPrim(drop_path).get_world_poses()[0].numpy().reshape(-1, 3)[0]
    pick_aabb = bounds_utils.compute_aabb(pick_path, bbox_cache=bbox_cache, space="world")
    drop_aabb = bounds_utils.compute_aabb(drop_path, bbox_cache=bbox_cache, space="world")
    pick_height = pick_aabb[5] - pick_aabb[2]
    pick_tcp_positions["right"] = (
        float(pick_center[0]),
        float(pick_center[1]),
        float(pick_center[2] + GRASP_HEIGHT_OFFSET),
    )
    drop_tcp_positions["right"] = (
        float(drop_center[0]),
        float(drop_center[1]),
        float(drop_aabb[5] + pick_height / 2.0 + GRASP_HEIGHT_OFFSET + PLACE_CLEARANCE),
    )

    try:
        await set_floating_gripper_grasp_async(context, False)
        await app_utils.update_app_async(steps=5)
        await _goto_tcp_positions_async(
            context, grippers, "goto_blue_box", _positions_above(pick_tcp_positions, LIFT_HEIGHT)
        )
        await _goto_tcp_positions_async(context, grippers, "descend_to_blue_box", pick_tcp_positions)
        await _capture_phase_async(capture, output_dir, "before_grasp")

        await set_floating_gripper_grasp_async(context, True)
        await _capture_phase_async(capture, output_dir, "after_grasp")

        await _goto_tcp_positions_async(
            context, grippers, "pick_up_blue_box", _positions_above(pick_tcp_positions, LIFT_HEIGHT)
        )
        await _capture_phase_async(capture, output_dir, "after_lift")

        await _goto_tcp_positions_async(
            context, grippers, "goto_red_box", _positions_above(drop_tcp_positions, LIFT_HEIGHT)
        )
        await _goto_tcp_positions_async(context, grippers, "descend_to_red_box", drop_tcp_positions)
        await set_floating_gripper_grasp_async(context, False)
        await app_utils.update_app_async(steps=10)
        await _capture_phase_async(capture, output_dir, "after_place")
        await _goto_tcp_positions_async(context, grippers, "return_to_start", start_tcp_positions)
    finally:
        for annotators in capture["annotators"].values():
            for annotator in annotators.values():
                annotator.detach()
        for render_product in capture["render_products"].values():
            render_product.destroy()
        stop_floating_gripper_controllers(context)
        app_utils.stop()
        await app_utils.update_app_async(steps=2)
    return list(cameras)


class TestTeleopGripperScenarios(omni.kit.test.AsyncTestCase):
    """Script Editor-equivalent scenarios with RGB goldens and depth output checks."""

    MEAN_DIFF_TOLERANCE = 15

    async def setUp(self) -> None:
        """Create an empty stage before each scenario."""
        await app_utils.update_app_async()
        await stage_utils.create_new_stage_async()

    async def tearDown(self) -> None:
        """Close the stage and wait for pending loads."""
        if stage_utils.is_stage_set() or omni.usd.get_context().get_stage() is not None:
            stage_utils.close_stage()
            await app_utils.update_app_async()
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

    def _validate_scenario(self, name: str, camera_names: list[str]) -> None:
        output_dir = os.path.join(os.getcwd(), f"_out_teleop_{name}")
        golden_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", f"_out_teleop_{name}")
        capture_count = len(camera_names) * len(CAPTURE_PHASES)
        self.assertTrue(
            validate_folder_contents(
                path=output_dir,
                expected_counts={"png": capture_count * 2, "npy": capture_count},
                recursive=True,
            ),
            f"Folder contents validation failed. Output dir: {output_dir}",
        )
        for camera_name in camera_names:
            result = compare_images_in_directories(
                golden_dir=os.path.join(golden_root, camera_name),
                test_dir=os.path.join(output_dir, camera_name),
                path_pattern=r"rgb_.*\.png$",
                allclose_rtol=None,
                allclose_atol=None,
                mean_tolerance=self.MEAN_DIFF_TOLERANCE,
                print_all_stats=False,
            )
            self.assertTrue(
                result["all_passed"],
                f"RGB comparison failed ({name}/{camera_name}). Output: {output_dir}",
            )

    async def test_script_editor_scenario_for_supported_grippers(self) -> None:
        """Run the tutorial's right-side scenario with each supported gripper."""
        for gripper_name in ("xarm", "dex3"):
            name = f"script_editor_{gripper_name}"
            with self.subTest(name=name):
                camera_names = await run_gripper_scenario_async(name, gripper_name)
                self._validate_scenario(name, camera_names)
