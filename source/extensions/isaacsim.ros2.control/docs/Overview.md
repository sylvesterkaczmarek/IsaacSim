# Overview

The `isaacsim.ros2.control` extension hosts a standard ros2_control `controller_manager` in-process inside Isaac Sim and exposes a USD articulation as a `hardware_interface::SystemInterface` (`IsaacSimSystem`). This lets ROS 2 developers drive Isaac Sim robots with standard ros2_control controllers (for example through MoveIt) without running an external `ros2_control_node`.

## Key Components

- **`ROS2ControlManager` OmniGraph node** (provided by `isaacsim.ros2.nodes`): configures the `controller_manager` from USD on the first tick of playback.
- **`Ros2ControlManager` Python API**: programmatic `setup` / `teardown` of the in-process `controller_manager`.
- **`IsaacSimSystem` hardware_interface plugin (C++)**: drives `read` / `update` / `write` from the physics post-step. The URDF is synthesized live from the articulation's USD `DriveAPI`.

It supports position, velocity, and effort command interfaces; force-torque and IMU sensor broadcasters; mimic joints; and multi-robot namespaces. It runs on PhysX and Newton, on ROS 2 Humble and Jazzy. Other ROS 2 distributions are rejected because ros2_control uses distribution-specific C++ libraries and cannot safely use the Jazzy fallback provided by `isaacsim.ros2.core` for its C API.

## Python Usage

```python
from isaacsim.ros2.control import Ros2ControlManager

# Bring up a ControllerManager for a USD articulation (after pressing Play).
Ros2ControlManager.setup("/World/Robot", "/path/to/controllers.yaml", namespace="robot")

# Tear it down when finished.
Ros2ControlManager.teardown("/World/Robot")
```

`setup` raises `RuntimeError` if a `controller_manager` is already registered for the prim or if initialization fails.
With the default `use_sim_time=True`, provide `/clock` with a ROS 2 bridge clock node or another ROS clock source.

For an end-to-end walkthrough, see the {ref}`ROS 2 Control tutorial <isaac_sim_app_tutorial_ros2_control>`.
