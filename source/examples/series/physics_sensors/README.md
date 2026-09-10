<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Physics Sensors

A physics sensor does not measure anything the simulation does not already know. It reads state the
engine computes anyway -- a body's pose and velocity, a joint's position and the force its
constraint carries -- and turns it into what a real device would report: in the frame the device is
mounted in, in the units its datasheet uses, filtered the way its firmware would filter it, and
stamped with the time it was taken.

Each step derives one sensor's reading from that raw state, in a module that knows nothing about a
stage or a physics engine, so the derivation is readable on its own and testable without a
simulation. A `main.py` beside it supplies the state from a running simulation and checks the
readings against what the scene was authored to do.

Steps are independent. Start with whichever sensor you need.

| Step | Level | What it derives |
| --- | --- | --- |
| `imu_sensor` | Intermediate | World pose, sensor-frame linear and angular velocity, and specific force for a sensor frame mounted on a rigid body |
| `joint_state_sensor` | Intermediate | Position, velocity, and measured effort for every degree of freedom of an articulation, converted from stage units to SI per DOF type |

From the examples collection root:

```bash
python examples.py run physics_sensors.imu_sensor
python examples.py test physics_sensors.imu_sensor
python examples.py run physics_sensors.joint_state_sensor
python examples.py test physics_sensors.joint_state_sensor
```

After activating the required Python environment, you can also run a step's `main.py` directly from
its step directory.
