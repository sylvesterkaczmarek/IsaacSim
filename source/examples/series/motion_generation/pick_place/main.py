# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pick and place a cube with a Franka articulation in the OVGL viewport."""

from __future__ import annotations

import argparse
import math
import os
import time

_ROBOT_PATH = "/World/Franka"
_CUBE_PATH = "/World/PickCube"
_CAMERA_PATH = "/ViewportCamera"
_RENDER_PRODUCT_PATH = "/Render/ViewportProduct"
_DEFAULT_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.1"
_ASSET_ROOT = (os.environ.get("ISAACSIM_ASSET_ROOT") or _DEFAULT_ASSET_ROOT).rstrip("/\\")
_FRANKA_USD = f"{_ASSET_ROOT}/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
_ARM_JOINTS = tuple(f"panda_joint{index}" for index in range(1, 8))
_FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")
_DEFAULT_JOINT_POSITIONS = {
    "panda_joint1": 0.012,
    "panda_joint2": -0.568,
    "panda_joint3": 0.0,
    "panda_joint4": -2.811,
    "panda_joint5": 0.0,
    "panda_joint6": 3.037,
    "panda_joint7": 0.741,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}
_PICK_POSITION = (0.45, 0.0, 0.025)
_PLACE_POSITION = (0.45, 0.35, 0.025)
_HAND_TO_GRASP_Z = 0.1034
_PHYSICS_DT = 1.0 / 60.0
_ARM_WAYPOINTS = {
    "above_pick": (0.007357, -0.240003, -0.007217, -2.659611, 0.001301, 2.927464, 0.741000),
    "grasp_pick": (0.008230, 0.386177, -0.007403, -2.491865, 0.001196, 2.758889, 0.741000),
    "above_place": (0.329874, 0.114022, 0.313730, -2.290599, -0.168417, 3.021942, 0.741000),
    "grasp_place": (0.421680, 0.612215, 0.220036, -2.145009, -0.142168, 2.843007, 0.741000),
}


def _author_stage() -> str:
    """Author the Franka task and return its unresolved root layer as USDA.

    Returns:
        Serialized USDA text for the unresolved root layer.
    """
    from isaacsim.foundation.objects import Camera, Cube, DistantLight, Plane, Prim
    from isaacsim.foundation.objects import Stage as FoundationStage
    from isaacsim.foundation.prims import ColliderBody, RigidBody
    from pxr import Gf, Sdf, Usd, UsdUtils

    stage = FoundationStage("openusd").create_stage()
    stage_id = stage.get_stage_id()
    usd_stage = UsdUtils.StageCache.Get().Find(Usd.StageCache.Id.FromLongInt(stage_id))
    if not usd_stage:
        stage.close_stage()
        raise RuntimeError("Foundation did not create a cached USD stage.")
    root_layer = usd_stage.GetRootLayer()

    try:
        stage.set_up_axis("Z")
        stage.set_units(meters_per_unit=1.0, kilograms_per_unit=1.0)
        stage.define_prim("/World")

        ground = Cube(
            "/World/Ground",
            sizes=1.0,
            translations=[0.25, 0.15, -0.05],
            scales=[2.2, 1.8, 0.1],
            colors=[[0.22, 0.24, 0.28]],
        )
        ground.set_visibilities([[False]])
        Cube(
            _CUBE_PATH,
            sizes=0.05,
            translations=list(_PICK_POSITION),
            colors=[[0.85, 0.12, 0.08]],
        )
        Plane(
            "/World/GroundVisual",
            widths=2.2,
            lengths=1.8,
            axes="Z",
            colors=[[0.22, 0.24, 0.28]],
            translations=[0.25, 0.15, 0.001],
        )
        stage.define_prim("/World/PhysicsScene", "PhysicsScene")
        physics_scene = Prim("/World/PhysicsScene")
        physics_scene.set_attribute_values("physics:gravityDirection", [0.0, 0.0, -1.0])
        physics_scene.set_attribute_values("physics:gravityMagnitude", 9.81)
        ColliderBody("/World/Ground")
        ColliderBody(_CUBE_PATH)
        RigidBody(_CUBE_PATH)

        light = DistantLight("/World/KeyLight")
        light.set_intensities([[3500.0]])
        light.set_colors([[1.0, 0.93, 0.82]])

        viewport_camera = Camera(
            _CAMERA_PATH,
            positions=[[0.0, 0.0, 3.0]],
            orientations=[[1.0, 0.0, 0.0, 0.0]],
        )
        viewport_camera.set_focal_lengths(1.8147562)
        viewport_camera.set_clipping_ranges(1.0, 10_000_000.0)
        stage.define_prim("/Render/Vars/LdrColor", "RenderVar")
        usd_render_var = usd_stage.GetPrimAtPath("/Render/Vars/LdrColor")
        usd_render_var.CreateAttribute(
            "dataType", Sdf.ValueTypeNames.Token, custom=False, variability=Sdf.VariabilityUniform
        ).Set("color4f")
        usd_render_var.CreateAttribute(
            "sourceName", Sdf.ValueTypeNames.String, custom=False, variability=Sdf.VariabilityUniform
        ).Set("LdrColor")
        stage.define_prim(_RENDER_PRODUCT_PATH, "RenderProduct")

        # Foundation does not expose attribute creation or relationship authoring for these Render prims yet.
        usd_render_product = usd_stage.GetPrimAtPath(_RENDER_PRODUCT_PATH)
        usd_render_product.CreateAttribute(
            "resolution", Sdf.ValueTypeNames.Int2, custom=False, variability=Sdf.VariabilityUniform
        ).Set(Gf.Vec2i(1280, 720))
        usd_render_product.GetRelationship("camera").SetTargets([_CAMERA_PATH])
        usd_render_product.GetRelationship("orderedVars").SetTargets(["/Render/Vars/LdrColor"])
        authored_stage_text = root_layer.ExportToString()
    finally:
        stage.close_stage()

    # Add the remote reference to a detached layer so only OVStage's bundled OmniUsdResolver attempts to compose it.
    detached_layer = Sdf.Layer.CreateAnonymous("pick_place.usda")
    if not detached_layer.ImportFromString(authored_stage_text):
        raise RuntimeError("OpenUSD could not reconstruct the authored stage layer.")
    robot_spec = Sdf.CreatePrimInLayer(detached_layer, _ROBOT_PATH)
    robot_spec.specifier = Sdf.SpecifierDef
    robot_spec.typeName = "Xform"
    robot_spec.referenceList.prependedItems = [Sdf.Reference(_FRANKA_USD)]
    return detached_layer.ExportToString()


def _make_joint_vector(dof_names: list[str]) -> object:
    """Build the named Franka home configuration in articulation order.

    Args:
        dof_names: Ordered degree-of-freedom names.

    Returns:
        The resulting value.
    """
    import numpy as np

    missing = [name for name in (*_ARM_JOINTS, *_FINGER_JOINTS) if name not in dof_names]
    if missing:
        raise RuntimeError(f"Franka articulation is missing joints: {', '.join(missing)}")
    return np.asarray([_DEFAULT_JOINT_POSITIONS.get(name, 0.0) for name in dof_names], dtype=np.float32)


def _make_trajectory(motion_generation: object, waypoints: object, dof_names: list[str]) -> object:
    """Time-parameterize the solved Franka pick/place joint waypoints.

    Args:
        motion_generation: Motion-generation interface.
        waypoints: Joint-space waypoints.
        dof_names: Ordered degree-of-freedom names.

    Returns:
        The resulting value.
    """
    import numpy as np

    maximum_velocities = np.asarray([0.55 if name in _ARM_JOINTS else 0.08 for name in dof_names], dtype=np.float32)
    maximum_accelerations = np.asarray([1.2 if name in _ARM_JOINTS else 0.20 for name in dof_names], dtype=np.float32)
    return motion_generation.Path(waypoints).to_minimal_time_joint_trajectory(
        max_velocities=maximum_velocities,
        max_accelerations=maximum_accelerations,
        robot_joint_space=dof_names,
        active_joints=dof_names,
    )


def main() -> None:
    """Run Franka motion generation, physics, and visualization against one OVStage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless", action="store_true", help="run and validate without opening a user-visible viewport window"
    )
    arguments = parser.parse_args()

    os.environ.setdefault("OVGL_SS", "1")
    from isaacsim.physics_engines.ovphysx import activate, set_suppress_readback, shutdown

    if not activate():
        raise RuntimeError("OvPhysX physics engine is unavailable.")

    ovstage_stage = None
    physics_manager = None
    viewport = None

    try:
        stage_text = _author_stage()

        import isaacsim.robot_motion.experimental.motion_generation as motion_generation
        import numpy as np
        import warp as wp
        from isaacsim.foundation.objects import Stage as FoundationStage
        from isaacsim.foundation.objects import Xform
        from isaacsim.ovgl_viewport.debug import Camera, CameraPose, Viewport, ViewportConfig
        from isaacsim.physics.entities import ArticulationEntity, RigidBodyEntity
        from isaacsim.physics.manager import PhysicsManager

        ovstage_stage = FoundationStage("ovstage").import_stage_from_string(stage_text)
        physics_manager = PhysicsManager.get_instance()
        if not physics_manager.switch_physics_engine("ovphysx"):
            raise RuntimeError("OvPhysX physics engine is unavailable.")
        physics_manager.setup(_PHYSICS_DT)
        set_suppress_readback(False)
        if not physics_manager.initialize(ovstage_stage.get_stage_ptr(), 0):
            raise RuntimeError("Physics initialization failed.")

        cube = RigidBodyEntity("ovphysx", _CUBE_PATH)
        articulation = ArticulationEntity("ovphysx", _ROBOT_PATH)
        if cube.num_prims != 1:
            raise RuntimeError("The pick cube rigid body was not found.")
        if articulation.num_prims != 1:
            raise RuntimeError(f"The Franka articulation was not found at {_ROBOT_PATH}.")

        dof_names = list(articulation.dof_names)
        link_names = list(articulation.link_names)
        home = _make_joint_vector(dof_names)
        finger_indices = [dof_names.index(name) for name in _FINGER_JOINTS]
        if "panda_hand" not in link_names:
            raise RuntimeError("Franka articulation does not expose the panda_hand link.")
        hand_index = link_names.index("panda_hand")
        robot_link_xforms = Xform(
            [f"{_ROBOT_PATH}/{name}" for name in link_names],
            resolve_paths=False,
            reset_xform_op_properties=False,
        )
        cube_xform = Xform(_CUBE_PATH, resolve_paths=False, reset_xform_op_properties=False)

        def sync_stage_from_physics() -> tuple[np.ndarray, np.ndarray]:
            """Publish measured physics poses to the real OVStage prims rendered by OVGL.

            Returns:
                The resulting value.
            """
            links = articulation.get_data("link-transforms").numpy().reshape(1, len(link_names), 7)[0]
            # Physics tensor poses use `xyzw`; Foundation `Xform` expects `wxyz`.
            robot_link_xforms.set_world_poses(
                positions=links[:, :3].astype(np.float64),
                orientations=np.column_stack((links[:, 6], links[:, 3:6])).astype(np.float64),
            )
            cube_transforms = cube.get_data("transforms").numpy()
            cube_xform.set_world_poses(
                positions=cube_transforms[:, :3].astype(np.float64),
                orientations=cube_transforms[:, [6, 3, 4, 5]].astype(np.float64),
            )
            return links, cube_transforms[0, :3]

        def arm_waypoint(name: str) -> np.ndarray:
            positions = home.copy()
            values = dict(zip(_ARM_JOINTS, _ARM_WAYPOINTS[name]))
            for joint_name, value in values.items():
                positions[dof_names.index(joint_name)] = value
            return positions

        above_pick = arm_waypoint("above_pick")
        grasp_pick = arm_waypoint("grasp_pick")
        above_place = arm_waypoint("above_place")
        grasp_place = arm_waypoint("grasp_place")

        open_fingers = home[finger_indices].copy()
        closed_fingers = np.zeros(len(finger_indices), dtype=np.float32)

        def with_fingers(positions: np.ndarray, finger_positions: np.ndarray) -> np.ndarray:
            result = positions.copy()
            result[finger_indices] = finger_positions
            return result

        waypoints = np.asarray(
            [
                with_fingers(home, open_fingers),
                with_fingers(above_pick, open_fingers),
                with_fingers(grasp_pick, open_fingers),
                with_fingers(grasp_pick, closed_fingers),
                with_fingers(above_pick, closed_fingers),
                with_fingers(above_place, closed_fingers),
                with_fingers(grasp_place, closed_fingers),
                with_fingers(grasp_place, open_fingers),
                with_fingers(above_place, open_fingers),
            ],
            dtype=np.float32,
        )
        trajectory = _make_trajectory(motion_generation, waypoints, dof_names)

        articulation.set_dof_positions(home.reshape(1, -1))
        articulation.set_dof_position_targets(home.reshape(1, -1))
        cube.set_world_poses(positions=np.asarray([_PICK_POSITION], dtype=np.float32))
        cube.set_velocities(
            linear_velocities=np.zeros((1, 3), dtype=np.float32),
            angular_velocities=np.zeros((1, 3), dtype=np.float32),
        )
        physics_manager.step()
        sync_stage_from_physics()

        camera = Camera()
        camera.target = (0.30, 0.15, 0.45)
        camera.yaw_radians = 0.80
        camera.pitch_radians = 0.30
        camera.distance = 2.6
        config = ViewportConfig()
        config.title = "Franka Motion Generation Pick/Place"
        config.visible = not arguments.headless
        config.camera = camera
        config.render_product_path = _RENDER_PRODUCT_PATH

        camera_xform = Xform(_CAMERA_PATH, resolve_paths=False, reset_xform_op_properties=False)

        def write_camera_pose(pose: CameraPose) -> None:
            camera_xform.set_world_poses(
                positions=np.asarray([pose.position], dtype=np.float64),
                orientations=np.asarray([pose.orientation], dtype=np.float64),
            )

        follower = motion_generation.TrajectoryFollower()

        def robot_state(values: np.ndarray) -> object:
            return motion_generation.RobotState(
                joints=motion_generation.JointState.from_name(
                    robot_joint_space=dof_names,
                    positions=(dof_names, wp.array(values, dtype=wp.float32, device="cpu")),
                )
            )

        estimated = robot_state(home)
        follower.set_trajectory(trajectory)
        if not follower.reset(estimated, None, 0.0):
            raise RuntimeError("TrajectoryFollower rejected the Franka pick/place trajectory.")

        contact_grasp_observed = False
        contact_hold_steps = 0
        release_observed = False
        maximum_cube_height = _PICK_POSITION[2]
        trajectory_step = 0
        trajectory_step_count = math.ceil(trajectory.duration / _PHYSICS_DT) + 1
        settle_step = 0
        simulation_complete = False
        print(f"Following a {trajectory.duration:.2f} second Franka motion-generation trajectory.")

        def advance_simulation() -> None:
            nonlocal contact_grasp_observed
            nonlocal contact_hold_steps
            nonlocal estimated
            nonlocal maximum_cube_height
            nonlocal release_observed
            nonlocal settle_step
            nonlocal simulation_complete
            nonlocal trajectory_step

            if simulation_complete:
                return
            if trajectory_step < trajectory_step_count:
                simulation_time = min(trajectory_step * _PHYSICS_DT, trajectory.duration)
                desired = follower.forward(estimated, None, simulation_time)
                if desired is None or desired.joints is None or desired.joints.positions is None:
                    raise RuntimeError("TrajectoryFollower returned no Franka joint command before completion.")
                desired_by_name = dict(zip(desired.joints.position_names, desired.joints.positions.numpy()))
                commanded = np.asarray([desired_by_name[name] for name in dof_names], dtype=np.float32)

                articulation.set_dof_position_targets(commanded.reshape(1, -1))
                physics_manager.step()
                measured_links, cube_position = sync_stage_from_physics()
                measured_joint_positions = articulation.get_dof_positions().numpy()[0].copy()
                measured_joint_velocities = articulation.get_dof_velocities().numpy()[0].copy()
                estimated = robot_state(measured_joint_positions)
                maximum_cube_height = max(maximum_cube_height, float(cube_position[2]))
                measured_fingers = measured_joint_positions[finger_indices]
                measured_finger_velocities = measured_joint_velocities[finger_indices]
                fingers_commanded_closed = float(np.max(commanded[finger_indices])) <= 0.005
                fingers_stopped_short = (
                    float(np.min(measured_fingers)) >= 0.005
                    and float(np.max(np.abs(measured_finger_velocities))) <= 0.1
                    and float(np.max(measured_fingers)) <= 0.035
                )
                grasp_frame = np.asarray(
                    [cube_position[0], cube_position[1], cube_position[2] + _HAND_TO_GRASP_Z], dtype=np.float32
                )
                hand_position = measured_links[hand_index, :3].astype(np.float32)
                holding_cube = (
                    fingers_commanded_closed
                    and fingers_stopped_short
                    and np.linalg.norm(hand_position - grasp_frame) <= 0.06
                    and cube_position[2] >= _PICK_POSITION[2] + 0.05
                )
                contact_hold_steps = contact_hold_steps + 1 if holding_cube else 0
                if not contact_grasp_observed and contact_hold_steps >= math.ceil(0.2 / _PHYSICS_DT):
                    contact_grasp_observed = True
                    print("Franka sustained a contact-driven cube lift.")
                fingers_open = (
                    float(np.min(commanded[finger_indices])) >= 0.035 and float(np.min(measured_fingers)) >= 0.03
                )
                if contact_grasp_observed and not release_observed and fingers_open:
                    release_observed = True
                    print("Franka opened its fingers above the place target.")
                trajectory_step += 1
                return

            if settle_step < 90:
                settle_step += 1
                physics_manager.step()
                sync_stage_from_physics()
                return

            final_positions, _ = cube.get_world_poses()
            final_velocities, _ = cube.get_velocities()
            final_position = final_positions.numpy()[0]
            final_velocity = final_velocities.numpy()[0]
            place_error = float(np.linalg.norm(final_position - np.asarray(_PLACE_POSITION)))
            if not contact_grasp_observed:
                raise RuntimeError("The Franka never achieved a measured contact-driven grasp.")
            if not release_observed:
                raise RuntimeError("The Franka never completed a measured finger release.")
            if maximum_cube_height < _PICK_POSITION[2] + 0.15:
                raise RuntimeError("The cube did not satisfy the 0.15 m lift gate.")
            if place_error > 0.05:
                raise RuntimeError(f"The cube missed the place target by {place_error:.3f} m.")
            if np.linalg.norm(final_velocity) > 0.10:
                raise RuntimeError("The cube did not settle after placement.")
            simulation_complete = True
            print(
                f"Franka pick/place completed: maximum height {maximum_cube_height:.3f} m, "
                f"placement error {place_error:.3f} m."
            )

        viewport = Viewport(ovstage_stage.get_stage_ptr(), write_camera_pose, config)
        if not viewport.poll_events():
            raise RuntimeError("OVGL viewport closed before its initial render.")
        initial_frame = viewport.render()
        initial_width = initial_frame.width
        initial_height = initial_frame.height
        initial_pixels = np.frombuffer(initial_frame.rgba, dtype=np.uint8).copy().reshape(-1, 4)

        if arguments.headless:
            while not simulation_complete:
                advance_simulation()
            frame = viewport.render()
            if frame.width != initial_width or frame.height != initial_height:
                raise RuntimeError("OVGL changed the output layout during the Franka pick/place operation.")
            final_pixels = np.frombuffer(frame.rgba, dtype=np.uint8).reshape(-1, 4)
            changed_pixels = int(
                np.count_nonzero(
                    np.any(np.abs(final_pixels.astype(np.int16) - initial_pixels.astype(np.int16)) > 4, axis=1)
                )
            )
            minimum_changed_pixels = initial_width * initial_height // 200
            if changed_pixels < minimum_changed_pixels:
                raise RuntimeError(
                    f"OVGL changed only {changed_pixels} pixels during the Franka pick/place operation; "
                    f"expected at least {minimum_changed_pixels}."
                )
            print("Headless Franka rendering smoke test passed.")
            print(f"Changed pixels: {changed_pixels}.")
            return

        last_frame_time = time.monotonic()
        accumulated_simulation_time = 0.0
        print("Controls: left-drag look, WASD move, Q/E down/up, wheel dolly, R reset, Escape quit.")
        while viewport.poll_events():
            current_time = time.monotonic()
            accumulated_simulation_time += min(current_time - last_frame_time, 0.25)
            last_frame_time = current_time
            steps_this_frame = 0
            while accumulated_simulation_time >= _PHYSICS_DT and steps_this_frame < 15 and not simulation_complete:
                advance_simulation()
                accumulated_simulation_time -= _PHYSICS_DT
                steps_this_frame += 1
            frame = viewport.render()
        print(f"Viewport rendered {frame.frame_number} frame(s).")
    finally:
        viewport = None
        try:
            if physics_manager is not None and physics_manager.is_initialized():
                physics_manager.invalidate()
        finally:
            try:
                if ovstage_stage is not None:
                    ovstage_stage.close_stage()
            finally:
                shutdown()


if __name__ == "__main__":
    main()
