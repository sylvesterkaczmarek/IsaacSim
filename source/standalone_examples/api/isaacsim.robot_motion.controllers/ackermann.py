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

"""Drive an Ackermann robot around using AckermannController.

Usage:
    ./python.sh standalone_examples/api/isaacsim.robot_motion.controllers/ackermann.py
    ./python.sh standalone_examples/api/isaacsim.robot_motion.controllers/ackermann.py --forklift
    ./python.sh standalone_examples/api/isaacsim.robot_motion.controllers/ackermann.py --direct
    ./python.sh standalone_examples/api/isaacsim.robot_motion.controllers/ackermann.py --forklift --direct

Sliders in the UI window let you adjust speed and turning angle at runtime.
"""

import argparse
import math

# Parse args before SimulationApp so --test is available for the loop guard.
_parser = argparse.ArgumentParser(description="Ackermann example")
_parser.add_argument(
    "--forklift",
    action="store_true",
    help="Use the ForkliftC (rear-wheel steering) instead of the Leatherback.",
)
_parser.add_argument(
    "--direct",
    action="store_true",
    help="Use direct-command mode: pass linear_speed and turning_angle as kwargs "
    "instead of encoding them as a velocity-vector site setpoint.",
)
_parser.add_argument(
    "--test",
    default=False,
    action="store_true",
    help="Run in test mode (a few frames, no window interaction).",
)
args, _ = _parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import isaacsim.robot_motion.controllers as ctrl
import isaacsim.robot_motion.experimental.motion_generation as mg
import warp as wp
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.storage.native import get_assets_root_path

if not args.test:
    from isaacsim.gui.components.element_wrappers import ScrollingWindow
    from isaacsim.gui.components.ui_utils import combo_floatfield_slider_builder

# ---------------------------------------------------------------------------
# Robot configurations
# Geometry derived from base.usda (both USDs authored in centimetres,
# root xformOp:scale = 0.01, so all offsets below are already in metres).
# ---------------------------------------------------------------------------

# <start-leatherback-config-snippet>
_LEATHERBACK = dict(
    # Asset
    asset_subpath="/Isaac/Robots_Multiphysics/NVIDIA/Leatherback/leatherback.usda",
    prim_path="/World/Leatherback",
    # Geometry — wheel CoM offsets from base.usda
    # Wheelbase: front axle X=0.150 m, rear axle X=-0.170 m → 0.320 m
    # Track:     wheel CoM Y = ±0.121 m (same front and rear) → 0.242 m
    # Radius:    wheel CoM Z = 0.052 m  (confirm from physx collision cylinder)
    wheel_base=0.320,
    track_width=0.242,
    wheel_radius=0.052,
    # Motion defaults
    v_linear=0.5,
    turning_angle=0.3,
    # Camera framing — the Leatherback turns with a radius of wheel_base/tan(theta) ≈ 1.0 m,
    # small enough that the default viewport leaves it a speck.  The forklift's
    # radius is ~5 m, which the default camera already frames, so it sets none.
    camera_eye=[2.4, -0.8, 1.6],
    camera_target=[0.0, 0.6, 0.1],
    # Joint names — front-wheel steering
    left_steerable_wheel_joint="Wheel__Knuckle__Front_Left",
    right_steerable_wheel_joint="Wheel__Knuckle__Front_Right",
    left_steering_joint="Knuckle__Upright__Front_Left",
    right_steering_joint="Knuckle__Upright__Front_Right",
    left_non_steerable_wheel_joint="Wheel__Upright__Rear_Left",
    right_non_steerable_wheel_joint="Wheel__Upright__Rear_Right",
    steerable_wheels_at_rear=False,
)
# <end-leatherback-config-snippet>

# <start-forklift-config-snippet>
_FORKLIFT = dict(
    # Asset
    asset_subpath="/Isaac/Robots_Multiphysics/IsaacSim/ForkliftC/forklift_c/forklift_c.usda",
    prim_path="/World/Forklift",
    # Geometry — collision cylinders from base.usda
    # Wheelbase:              front X=0.269 m, rear X=-1.383 m → 1.652 m
    # Rear (steerable) track: cylinder centre Y = ±0.570 m    → 1.140 m
    # Front (NS) track:       cylinder centre Y = ±0.522 m    → 1.044 m
    # Rear wheel radius:      0.5 × 51 cm × 0.01              = 0.255 m
    # Front wheel radius:     0.5 × 65 cm × 0.01              = 0.325 m
    wheel_base=1.652,
    track_width=1.140,
    non_steerable_track_width=1.044,
    wheel_radius=0.255,
    non_steerable_wheel_radius=0.325,
    # Motion defaults
    v_linear=1.5,
    turning_angle=0.3,
    # Joint names — rear-wheel steering (forklift)
    left_steerable_wheel_joint="left_back_wheel_joint",
    right_steerable_wheel_joint="right_back_wheel_joint",
    left_steering_joint="left_rotator_joint",
    right_steering_joint="right_rotator_joint",
    left_non_steerable_wheel_joint="left_front_wheel_joint",
    right_non_steerable_wheel_joint="right_front_wheel_joint",
    steerable_wheels_at_rear=True,
)
# <end-forklift-config-snippet>


# ---------------------------------------------------------------------------
# Stage loader
# ---------------------------------------------------------------------------


async def _load_stage(usd_path: str, prim_path: str) -> Articulation:
    """Create a fresh stage, load the robot USD, return an Articulation handle."""
    await stage_utils.create_new_stage_async(template="default stage")
    stage_utils.add_reference_to_stage(usd_path=usd_path, path=prim_path)
    await app_utils.update_app_async()
    robot = Articulation(prim_path)
    await app_utils.update_app_async()
    SimulationManager.setup_simulation(dt=1.0 / 60.0)
    return robot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = _FORKLIFT if args.forklift else _LEATHERBACK
    print(f"Robot: {'ForkliftC' if args.forklift else 'Leatherback'}")
    print(f"Mode:  {'direct' if args.direct else 'site setpoint'}")

    # --- 1. Scene setup ---------------------------------------------------------

    assets_root_path = get_assets_root_path()
    robot = simulation_app.run_coroutine(_load_stage(assets_root_path + cfg["asset_subpath"], cfg["prim_path"]))

    print(f"Joint space:   {robot.dof_names}")

    # --- 2. Controller ----------------------------------------------------------

    # Speed and angle limits are shared between the controller (enforced in the
    # Ackermann kernel) and the sliders (clamped in the UI).  Defining them here
    # makes the connection explicit: dragging a slider past v_max or angle_max
    # has no effect because the controller clamps first.
    v_max = cfg["v_linear"] * 3.0
    angle_max = math.pi / 2.0 - 0.05  # just under the Ackermann singularity

    # <start-controller-snippet>
    controller = ctrl.AckermannController(
        robot_joint_space=robot.dof_names,
        left_steerable_wheel_joint=cfg["left_steerable_wheel_joint"],
        right_steerable_wheel_joint=cfg["right_steerable_wheel_joint"],
        left_steering_joint=cfg["left_steering_joint"],
        right_steering_joint=cfg["right_steering_joint"],
        left_non_steerable_wheel_joint=cfg.get("left_non_steerable_wheel_joint"),
        right_non_steerable_wheel_joint=cfg.get("right_non_steerable_wheel_joint"),
        non_steerable_wheel_radius=cfg.get("non_steerable_wheel_radius"),
        non_steerable_track_width=cfg.get("non_steerable_track_width"),
        steerable_wheel_radius=cfg["wheel_radius"],
        wheel_base=cfg["wheel_base"],
        track_width=cfg["track_width"],
        steerable_wheels_at_rear=cfg["steerable_wheels_at_rear"],
        max_linear_speed=v_max,
        max_turning_angle=angle_max,
        direct_command=args.direct,
    )
    # <end-controller-snippet>

    # --- 3. Control sliders (interactive mode only) -----------------------------

    # combo_floatfield_slider_builder links a FloatField and a FloatSlider to a
    # single shared model, styled consistently with the rest of Isaac Sim's UI.
    # It returns (model, slider_widget); we only need the model for polling.
    # Each builder is the sole child of its own ScrollingWindow — multiple
    # builders in one window require a ui.VStack parent which would pull in
    # a direct omni.ui import.
    # Explicit positions stack them down the left of the viewport rather than
    # letting both open on top of each other at the default location.
    if not args.test:
        speed_window = ScrollingWindow(
            title="Speed (m/s)",
            width=420,
            height=100,
            position_x=40,
            position_y=120,
        )
        with speed_window.frame:
            speed_model, _ = combo_floatfield_slider_builder(
                label="Speed (m/s)",
                default_val=cfg["v_linear"],
                min=-v_max,
                max=v_max,
                step=0.01,
            )

        angle_window = ScrollingWindow(
            title="Turning angle (rad)",
            width=420,
            height=100,
            position_x=40,
            position_y=240,
        )
        with angle_window.frame:
            angle_model, _ = combo_floatfield_slider_builder(
                label="Turning angle (rad)",
                default_val=cfg["turning_angle"],
                min=-angle_max,
                max=angle_max,
                step=0.01,
            )

    # --- 4. Start simulation and place robot ------------------------------------

    app_utils.play()
    simulation_app.update()

    robot.set_world_poses(positions=wp.array([[0.0, 0.0, 0.0]], dtype=wp.float32))
    simulation_app.update()

    if cfg.get("camera_eye") is not None:
        ViewportManager.set_camera_view(
            ViewportManager.get_camera(),
            eye=cfg["camera_eye"],
            target=cfg["camera_target"],
        )

    controller.reset(mg.RobotState(), mg.RobotState(), 0.0)

    # --- 5. Main loop -----------------------------------------------------------

    frame_count = 0
    while simulation_app.is_running():
        simulation_app.update()

        if args.test:
            v = cfg["v_linear"]
            theta = cfg["turning_angle"]
        else:
            v = speed_model.get_value_as_float()
            theta = angle_model.get_value_as_float()

        if args.direct:
            # Direct mode — pass scalar speed and angle straight to the controller.
            # No velocity-vector encoding/decoding; also avoids the theta snap
            # at v ≈ 0 that the site-setpoint path can exhibit.
            # <start-direct-snippet>
            desired_state = controller.forward(
                mg.RobotState(),
                None,
                0.0,
                linear_speed=v,
                turning_angle=theta,
            )
            # <end-direct-snippet>
        else:
            # Site-setpoint mode — encode (v, θ) as a velocity vector.
            # The controller projects onto the forward/lateral axes internally
            # to recover linear_speed and turning_angle.
            # <start-setpoint-snippet>
            setpoint = mg.RobotState(
                sites=mg.SpatialState.from_name(
                    spatial_space=["control_point"],
                    linear_velocities=(
                        ["control_point"],
                        wp.array([[v * math.cos(theta), v * math.sin(theta), 0.0]], dtype=wp.float32),
                    ),
                )
            )
            # <end-setpoint-snippet>
            desired_state = controller.forward(mg.RobotState(), setpoint, 0.0)

        # <start-apply-snippet>
        if desired_state is not None and desired_state.joints is not None:
            if desired_state.joints.velocities is not None:
                robot.set_dof_velocity_targets(
                    desired_state.joints.velocities,
                    dof_indices=robot.get_dof_indices(desired_state.joints.velocity_names),
                )
            if desired_state.joints.positions is not None:
                robot.set_dof_position_targets(
                    desired_state.joints.positions,
                    dof_indices=robot.get_dof_indices(desired_state.joints.position_names),
                )
        # <end-apply-snippet>

        frame_count += 1
        if args.test and frame_count >= 10:
            break

    simulation_app.close()


if __name__ == "__main__":
    main()
