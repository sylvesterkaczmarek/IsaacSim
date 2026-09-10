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

"""Workspace-free smoke test for isaacsim.ros2.control.

Runs entirely inside Isaac Sim through python.sh: builds a one-joint arm, starts
the in-process Controller Manager, activates a joint_state_broadcaster and a
joint_trajectory_controller through the Controller Manager services with rclpy,
commands the joint, and checks that it reaches the target. Prints PASS or FAIL
and exits 0 or 1. No ROS 2 workspace, MoveIt, or external ros2_control_node.

    ./python.sh standalone_examples/api/isaacsim.ros2.control/ros2_control_smoke_test.py [--headless]
"""

import argparse
import os
import sys
import tempfile

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
args, _ = parser.parse_known_args()
simulation_app = SimulationApp({"headless": args.headless})

import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import omni.graph.core as og
import omni.timeline
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, UsdGeom, UsdPhysics

ARM = "/World/Arm"
JOINT = "joint1"
TARGET = 0.5
TOLERANCE = 0.1

YAML = """
controller_manager:
  ros__parameters:
    update_rate: 60
    joint_state_broadcaster:
      type: joint_state_broadcaster/JointStateBroadcaster
    position_controller:
      type: joint_trajectory_controller/JointTrajectoryController

position_controller:
  ros__parameters:
    joints:
      - joint1
    command_interfaces:
      - position
    state_interfaces:
      - position
      - velocity
"""


def build_arm(stage):
    """Fixed-base one-joint arm with a position drive on joint1."""
    UsdGeom.Xform.Define(stage, "/World")
    arm = UsdGeom.Xform.Define(stage, ARM).GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(arm)

    base = UsdGeom.Xform.Define(stage, f"{ARM}/base").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(base)
    UsdPhysics.MassAPI.Apply(base).CreateMassAttr(1.0)
    UsdPhysics.FixedJoint.Define(stage, f"{ARM}/base_fixed").CreateBody1Rel().SetTargets([base.GetPath()])

    link1 = UsdGeom.Xform.Define(stage, f"{ARM}/link1").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(link1)
    mass = UsdPhysics.MassAPI.Apply(link1)
    mass.CreateMassAttr(1.0)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(0.1, 0.1, 0.1))

    joint = UsdPhysics.RevoluteJoint.Define(stage, f"{ARM}/{JOINT}")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([link1.GetPath()])
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-180.0)
    joint.CreateUpperLimitAttr(180.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(2000.0)
    drive.CreateDampingAttr(200.0)
    drive.CreateTargetPositionAttr(0.0)


def create_clock_graph():
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2ClockGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
        },
    )


def main() -> int:
    app_utils.enable_extension("isaacsim.ros2.core")
    app_utils.enable_extension("isaacsim.ros2.bridge")
    app_utils.enable_extension("isaacsim.ros2.control")
    simulation_app.update()

    import rclpy
    from controller_manager_msgs.srv import ConfigureController, LoadController, SwitchController
    from isaacsim.ros2.control import Ros2ControlManager
    from isaacsim.ros2.control.bindings import _isaacsim_ros2_control as backend
    from sensor_msgs.msg import JointState
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

    stage = stage_utils.create_new_stage()
    stage_utils.set_stage_units(meters_per_unit=1.0)
    build_arm(stage)
    create_clock_graph()

    fd, yaml_path = tempfile.mkstemp(suffix=".yaml")
    os.close(fd)
    with open(yaml_path, "w", encoding="utf-8") as handle:
        handle.write(YAML)

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    timeline = omni.timeline.get_timeline_interface()
    owns_rclpy = not rclpy.ok()
    if owns_rclpy:
        rclpy.init()
    node = rclpy.create_node("ros2_control_smoke_test")

    def spin():
        rclpy.spin_once(node, timeout_sec=0)

    def wait_until(cond, max_iters, per_iter=spin):
        for _ in range(max_iters):
            if cond():
                return True
            per_iter()
            simulation_app.update()
        return cond()

    def finish(code, message):
        print(message)
        try:
            os.unlink(yaml_path)
        except OSError:
            pass
        timeline.stop()
        for _ in range(5):
            simulation_app.update()
        node.destroy_node()
        if owns_rclpy:
            rclpy.shutdown()
        simulation_app.close()
        return code

    def call(client, request, max_iters=900):
        if not wait_until(client.service_is_ready, max_iters):
            return None
        future = client.call_async(request)
        if not wait_until(future.done, max_iters):
            return None
        return future.result()

    # Play to initialize physics, wait for the backend, pause to set up the
    # Controller Manager, then resume.
    timeline.play()
    for _ in range(5):
        simulation_app.update()
    if not wait_until(backend.is_ready, 600):
        return finish(1, "DEMO SMOKE TEST: FAIL (backend never became ready)")
    timeline.pause()
    for _ in range(2):
        simulation_app.update()
    rc = Ros2ControlManager.setup(ARM, yaml_path)
    if rc != 0:
        return finish(1, f"DEMO SMOKE TEST: FAIL (setup returned {rc})")
    timeline.play()

    # Activate the broadcaster and the trajectory controller through the CM.
    cm = "/controller_manager"
    load = node.create_client(LoadController, f"{cm}/load_controller")
    configure = node.create_client(ConfigureController, f"{cm}/configure_controller")
    switch = node.create_client(SwitchController, f"{cm}/switch_controller")
    controllers = ["joint_state_broadcaster", "position_controller"]
    for name in controllers:
        res = call(load, LoadController.Request(name=name))
        if res is None or not res.ok:
            return finish(1, f"DEMO SMOKE TEST: FAIL (load_controller {name})")
        res = call(configure, ConfigureController.Request(name=name))
        if res is None or not res.ok:
            return finish(1, f"DEMO SMOKE TEST: FAIL (configure_controller {name})")
    req = SwitchController.Request(activate_controllers=controllers, strictness=SwitchController.Request.STRICT)
    res = call(switch, req)
    if res is None or not res.ok:
        return finish(1, "DEMO SMOKE TEST: FAIL (switch_controller activate)")

    # Command the joint and confirm it reaches the target.
    latest = {}

    def on_state(msg):
        if JOINT in msg.name:
            latest["pos"] = msg.position[list(msg.name).index(JOINT)]

    node.create_subscription(JointState, "/joint_states", on_state, 10)
    publisher = node.create_publisher(JointTrajectory, "/position_controller/joint_trajectory", 10)
    command = JointTrajectory()
    command.joint_names = [JOINT]
    point = JointTrajectoryPoint()
    point.positions = [TARGET]
    point.time_from_start.sec = 0
    point.time_from_start.nanosec = 200_000_000
    command.points = [point]

    sent = False

    def push():
        nonlocal sent
        if not sent and publisher.get_subscription_count() > 0:
            publisher.publish(command)
            sent = True
        spin()

    reached = wait_until(
        lambda: "pos" in latest and abs(latest["pos"] - TARGET) < TOLERANCE,
        3000,
        per_iter=push,
    )
    if not reached:
        return finish(1, f"DEMO SMOKE TEST: FAIL (joint1 did not reach {TARGET}, last={latest.get('pos')})")

    # Let the joint settle, then report where it ended up.
    for _ in range(300):
        push()
        simulation_app.update()

    return finish(0, f"DEMO SMOKE TEST: PASS (joint1 reached {latest['pos']:.3f}, target {TARGET})")


if __name__ == "__main__":
    sys.exit(main())
