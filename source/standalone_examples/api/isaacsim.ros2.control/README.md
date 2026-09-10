# UR10 + MoveIt 2 ros2_control demo

Two-process demo of the in-process ControllerManager.

## Prerequisites (workspace side)

```bash
sudo apt install ros-jazzy-rviz-visual-tools
```

Required by the RViz panels in `isaac_ros2_control_demo`'s motion-planning view.

## Terminal 1 (Isaac Sim root)

```bash
./python.sh standalone_examples/api/isaacsim.ros2.control/ur10_ros2_control_demo.py
```

This boots Isaac Sim, opens the UR10 USD, publishes `/clock` through the ROS 2 bridge, configures
`ROS2ControlManager` against `ur10_controllers.yaml`, and presses Play. The in-process ControllerManager publishes
`/robot_description` (latched, transient_local QoS). Once loaded and activated,
`scaled_joint_trajectory_controller` serves the FollowJointTrajectory action.

## Terminal 2 (workspace)

```bash
source ~/IsaacSim-ros_workspaces/jazzy_ws/install/setup.bash
ros2 launch isaac_ros2_control_demo ur10_in_process.launch.py
```

This starts `move_group`, `robot_state_publisher`, RViz, and both controller spawners.

Wait until Isaac Sim prints `[ros2.control] backend ready` and the CM's services are visible (`ros2 control list_controllers` should respond). Then plan and execute a motion in RViz; the trajectory goes to the in-process CM and the UR10 in Isaac Sim moves.

## Direct trajectory smoke test

The fastest way to confirm the full ros2_control pipeline is wired up is to publish a trajectory directly. The UR10 should move to the target pose in Isaac Sim within ~3 seconds:

```bash
ros2 topic pub --once /scaled_joint_trajectory_controller/joint_trajectory \
    trajectory_msgs/msg/JointTrajectory \
    '{joint_names: ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
      points: [{positions: [0.5, -1.2, 1.2, -1.5, -1.5, 0.0],
                time_from_start: {sec: 3}}]}'
```

This bypasses MoveIt planning; if the arm moves, the simulation-side stack works.

See `jazzy_ws/src/moveit/isaac_ros2_control_demo/README.md` for workspace details.
