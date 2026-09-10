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

"""Per-robot namespaced ROS 2 bridge action graph factory for fleet demos.

Creates one OmniGraph action graph per robot at `/ROS2_Bridge_<robot_name>`.
Each graph publishes namespaced odometry, TF, laser scan, and joint states,
and subscribes to a namespaced `cmd_vel` topic. A shared clock graph publishes
`/clock` for Nav2 `use_sim_time`.

All graphs use `OnPlaybackTick` as the execution trigger, so **no topics are
published until Play is pressed** — the graphs remain dormant in the stopped
state.

Reference scenario in the repo:
    source/standalone_examples/api/isaacsim.ros2.bridge/carter_multiple_robot_navigation.py

Use the current `isaacsim.ros2.bridge.*` node names; the legacy
`omni.isaac.ros2_bridge.*` namespace will not load on Kit 110.

Usage (standalone):
    ./python.sh scripts/multi_robot_namespacing.py \\
        --robots carter1:/World/Carter1 carter2:/World/Carter2 \\
        --usd_path /Isaac/Samples/ROS2/Scenario/multiple_robot_carter_hospital_navigation.usd
"""

import argparse
import sys
from typing import Optional


def _og():
    import omni.graph.core as og

    return og


def create_clock_graph(graph_path: str = "/ROS2_Clock"):
    """Create a shared clock publisher graph (active only during Play).

    Nav2 requires `/clock` when `use_sim_time: true`. This graph publishes
    simulation time on each playback tick.
    """
    keys = _og().Controller.Keys
    _og().Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.SET_VALUES: [
                ("PublishClock.inputs:topicName", "/clock"),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
            ],
        },
    )


def create_ros2_bridge_for_robot(
    robot_name: str,
    robot_prim_path: str,
    lidar_prim_path: Optional[str] = None,
    publish_joint_states: bool = True,
):
    """Create a namespaced ROS 2 bridge action graph for a single robot.

    The graph uses OnPlaybackTick so it only activates after Play is pressed.

    Args:
        robot_name: Unique name used as ROS 2 namespace and graph suffix.
        robot_prim_path: USD prim path to the robot root (e.g. "/World/Carter1").
        lidar_prim_path: Optional prim path to a lidar sensor for laser scan
            publishing. If None, laser scan node is omitted.
        publish_joint_states: Whether to include a joint state publisher.
    """
    graph_path = f"/ROS2_Bridge_{robot_name}"
    ns = f"/{robot_name}"
    keys = _og().Controller.Keys

    nodes = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("PublishOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
        ("PublishTF", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
        ("SubscribeTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
    ]

    values = [
        ("PublishOdom.inputs:topicName", f"{ns}/odom"),
        ("PublishOdom.inputs:chassisPrim", robot_prim_path),
        ("PublishOdom.inputs:frameId", f"{robot_name}/odom"),
        ("PublishOdom.inputs:childFrameId", f"{robot_name}/base_link"),
        ("PublishTF.inputs:topicName", "/tf"),
        ("PublishTF.inputs:targetPrims", [robot_prim_path]),
        ("SubscribeTwist.inputs:topicName", f"{ns}/cmd_vel"),
    ]

    connections = [
        ("OnPlaybackTick.outputs:tick", "PublishOdom.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishTF.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "SubscribeTwist.inputs:execIn"),
    ]

    if publish_joint_states:
        nodes.append(("PublishJointStates", "isaacsim.ros2.bridge.ROS2PublishJointState"))
        values.extend(
            [
                ("PublishJointStates.inputs:topicName", f"{ns}/joint_states"),
                ("PublishJointStates.inputs:targetPrim", robot_prim_path),
            ]
        )
        connections.append(("OnPlaybackTick.outputs:tick", "PublishJointStates.inputs:execIn"))

    if lidar_prim_path:
        nodes.append(("PublishLaserScan", "isaacsim.ros2.bridge.ROS2PublishLaserScan"))
        values.extend(
            [
                ("PublishLaserScan.inputs:topicName", f"{ns}/scan"),
                ("PublishLaserScan.inputs:frameId", f"{robot_name}/lidar_link"),
            ]
        )
        connections.append(("OnPlaybackTick.outputs:tick", "PublishLaserScan.inputs:execIn"))

    _og().Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: nodes,
            keys.SET_VALUES: values,
            keys.CONNECT: connections,
        },
    )


def setup_fleet(robots: list[tuple[str, str]], lidar_suffix: str = "/Lidar"):
    """Set up bridge graphs for an entire fleet.

    Creates one shared clock graph and one per-robot bridge graph. All graphs
    are play-gated via OnPlaybackTick — nothing publishes until the timeline
    starts.

    Args:
        robots: List of (robot_name, robot_prim_path) tuples.
        lidar_suffix: Prim path suffix appended to robot_prim_path for lidar.
            Set to empty string to skip laser scan publishing.
    """
    create_clock_graph()
    for name, prim_path in robots:
        lidar_path = f"{prim_path}{lidar_suffix}" if lidar_suffix else None
        create_ros2_bridge_for_robot(
            robot_name=name,
            robot_prim_path=prim_path,
            lidar_prim_path=lidar_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fleet demo: per-robot ROS 2 bridge graphs (play-gated).")
    parser.add_argument(
        "--robots",
        nargs="+",
        required=True,
        metavar="NAME:PRIM_PATH",
        help="Robot entries as name:prim_path (e.g. carter1:/World/Carter1)",
    )
    parser.add_argument(
        "--usd_path",
        type=str,
        default="/Isaac/Samples/ROS2/Scenario/multiple_robot_carter_hospital_navigation.usd",
        help="Nucleus USD path to load",
    )
    parser.add_argument(
        "--lidar_suffix",
        type=str,
        default="/Lidar",
        help="Prim suffix for lidar sensor (empty to skip laser scan)",
    )
    parser.add_argument("--headless", action="store_true", help="Run without display")
    parser.add_argument("--test", action="store_true", help="Exit after a few frames")
    args = parser.parse_args()

    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"renderer": "RealTimePathTracing", "headless": args.headless})

    import isaacsim.core.experimental.utils.app as app_utils
    import omni
    from isaacsim.core.experimental.utils.stage import is_stage_loading
    from isaacsim.core.simulation_manager import SimulationManager
    from isaacsim.storage.native import get_assets_root_path

    app_utils.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        print("ERROR: Could not find Isaac Sim assets folder")
        simulation_app.close()
        sys.exit(1)

    omni.usd.get_context().open_stage(assets_root_path + args.usd_path, None)
    simulation_app.update()
    simulation_app.update()

    print("Loading stage...")
    while is_stage_loading():
        simulation_app.update()
    print("Loading complete.")

    # Parse robot list
    robots = []
    for entry in args.robots:
        if ":" not in entry:
            print(f"ERROR: Invalid robot entry '{entry}', expected NAME:PRIM_PATH")
            simulation_app.close()
            sys.exit(1)
        name, prim_path = entry.split(":", 1)
        robots.append((name, prim_path))

    # Build all bridge graphs (dormant until Play)
    setup_fleet(robots, lidar_suffix=args.lidar_suffix)
    print(f"Created bridge graphs for {len(robots)} robots (play-gated).")

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    simulation_app.update()

    # Press Play — this activates all OnPlaybackTick-driven graphs
    app_utils.play()
    print("Play pressed — bridge graphs now active, publishing topics.")
    simulation_app.update()

    frame = 0
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        if args.test and frame >= 20:
            break

    app_utils.stop()
    simulation_app.close()
