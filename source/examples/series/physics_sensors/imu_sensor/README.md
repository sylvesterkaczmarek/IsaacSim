<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# IMU Sensor

`main.py` authors a spinning cube that falls onto a ground plane, mounts a sensor frame on it as a plain `Xform`
child, and reports the five quantities an IMU exposes for that frame: world position, world orientation,
sensor-frame linear and angular velocity, and sensor-frame linear acceleration.

## What it shows

Everything the sensor does lives in this directory. `imu.py` holds the sensor mathematics and
`main.py` drives it from a simulation, so the whole derivation is readable in one place without
following it into a library.

`imu.py` deliberately knows nothing about a stage, a physics engine, or a simulation. Callers push
one `BodyState` per step with `sample()` and pull a filtered `ImuReading` with `read()`, so the body
state could equally come from a recorded trajectory or a closed-form one. Vectors and quaternions
are plain Warp types -- `wp.vec3d` and `wp.quatd` -- rather than custom ones. Warp is the only
dependency, and the one the library build already declares.

Quaternions are therefore ordered **`xyzw`**, matching Warp and the physics engine tensor buffers
the body state comes from. That choice is what lets the sensor use Warp's own rotation builtins --
`wp.quat_rotate`, `wp.quat_rotate_inv`, `wp.mul`, `wp.normalize`, `wp.cross`, `wp.dot` -- instead of
hand-written quaternion arithmetic. Only the sign-aligned mean in `average_quaternions()` is spelled
out, because Warp has no builtin for it.

`main.py` supplies that state from the lower-level rigid body APIs and makes the checks. That
wiring is where the interesting details live:

- **Reading body state.** `RigidBodyEntity.get_world_poses()` and `get_velocities()` return the pose
  and both velocities as separate arrays. The orientation is `xyzw`, which is `imu.py`'s ordering, so
  the per-step state is passed straight through with no conversion. The engine's own acceleration,
  used only by the cross-check below, has no accessor and is read by buffer name with `get_data()`.
- **The mount offset** is read back from USD with `Xform.get_local_poses()`, which returns `wxyz`,
  so it is converted once at setup with `read_quaternion_from_wxyz()` and handed to `ImuSensor`.
  This is the example's only quaternion conversion: USD and the physics engine disagree on ordering,
  so the seam has to sit somewhere, and putting it here costs one conversion per run rather than one
  per step.
- **Timestamps.** `simulate()` takes the time a step starts from, but the buffers hold the state
  after it, so each sample is stamped with the time the step advanced to.
- **Gravity.** The reading is the *specific force* an accelerometer measures, `a - g`, so a body in
  free fall reads near zero and a resting body reads one gravity along world up. This is not
  selectable: it is the convention `sensor_msgs/Imu` follows and the only mode most simulators
  offer. It is a required argument, because its magnitude depends on the stage's linear unit and its
  direction on the stage's up axis.
- **Acceleration is differenced in the world frame**, and only the result is rotated into the sensor
  frame. Differencing a velocity already expressed in the rotating frame would cancel the transport
  term back out.

The mount is a quarter-turn **yaw**, so the sensor frame is distinct from the body frame while sensor
`+Z` stays along world up. That is the Isaac Sim stage convention and what an IMU datasheet assumes,
and it makes the readings easy to check by eye: falling reads `[0, 0, -v]`, the spin about world `+Z`
reads `[0, 0, w]`, and at rest the accelerometer reads one gravity on `+Z`.

## The checks, and where they live

`main.py` checks only what needs a running simulation: that the airborne readings look like free
fall, that the settled ones read one gravity along world up, and that the sensor ends up one mount
offset above the body center.

As a cross-check it also differences the body's world-frame linear velocity over a single step and
compares it against two things. The engine's own `accelerations` buffer holds the very same
single-step velocity difference, so over the airborne window they agree closely and that comparison
only confirms the buffer slices line up rather than that either value is right. The check
with teeth is the second one: over the airborne steps the only force is gravity, so the difference
has to reproduce the gravity the scene was authored with, independent of what either buffer reports.

Everything that does **not** need a simulation is a unit test in `test_imu.py` instead, because the
scene is geometrically degenerate in ways that hide real mistakes and a synthetic state exercises
them far better. The dropped body spins about the same axis its mount is offset along, so `w x r` is
identically zero for every reading printed; nothing in the scene circles, so the transport term
never shows; and the body and mount rotate about the same axis, so composing them in the wrong order
gives the same answer. `test_imu.py` covers each of those against states chosen so the quantity
under test actually changes the answer: the lever arm and the mount offset rotation, composition
order against non-commuting rotations, the centripetal term against a circling sensor, and the
reported velocities against a mount that does not share the body's axis.

## Note on quaternion ordering

`RigidBodyEntity.get_world_poses()` returns the engine's ordering, `xyzw`, which is what this example and `imu.py`
use, so the body orientation is passed through unreordered. Its header currently documents `wxyz`; the value measured
from a running simulation is `xyzw`, and `RigidBodyEntity::getWorldPoses()` reaches it by slicing the engine's
`(N, 7)` pose buffer without reordering. Should that be reconciled in favour of `wxyz`, this example fails rather than
drifting: the body yaws as it falls, so reading its quaternion the other way tilts the sensor frame and the free-fall
checks stop holding.

## Files

`imu.py` holds the sensor mathematics and depends on nothing but Warp. `test_imu.py` holds its
unit tests. `main.py` authors the scene, runs the simulation, and makes the checks. `viewer.py` holds the optional live view and is
used only when `--ui` is passed. It takes the simulation as callables rather than importing
`main.py`, so it knows nothing about the physics engine or the scene and the dependency runs one
way only.

## Live view (optional)

`--ui` opens a window showing the body and the readings at the same time: a wireframe cube falling
and spinning, with the sensor frame drawn as three colored axes at its mount point (X red, Y green, Z blue, so the blue
axis points up), beside the
live reading values and rolling traces of acceleration and angular-velocity magnitude. Physics is
stepped from the viewer's frame callback, so the simulation and the display advance together in one
process.

The viewer drops from higher than the default run does, so the free-fall stretch lasts long enough
to watch the readings change: acceleration falls to zero on release, spikes on contact, and settles
at one gravity. The batch path keeps its own drop height, which its airborne step window and
contact step are tuned around.

The drop takes about three seconds, after which the body is teleported back to its start pose,
respun, and dropped again, so there is always something moving. Each restart calls
`ImuSensor.reset()`, because a teleport makes the finite difference across it meaningless. Unlike
the default run, the viewer makes no resting-state assertions: it exits whenever the window is
closed, which may be mid-fall.

It is off by default and needs two packages this example deliberately does **not** declare, because
`example.toml` requirements accept only Isaac Sim distributions and the system capabilities listed
in the collection README:

```bash
python -m pip install ovui==0.1.1 glfw==2.10.0     # into the build's Python
python examples.py run physics_sensors.imu_sensor -- --ui
```

`ovui` is the standalone distribution of Omniverse's `omni.ui`, imported as `omni.ui`.
`source/tools/zmq_bridge/zmq_server.py` also runs it outside Kit, and additionally hides the `carb`
and `omni.kit.app` module specs during import, which this example does not need because `carb` is
not importable from the standalone build. A display is required — the viewer
reports an error rather than opening a blank window if none is available. The automated tests never
pass `--ui`, so they neither need nor exercise these packages.

## Run it

From the examples collection root:

```bash
python examples.py run physics_sensors.imu_sensor
python examples.py test physics_sensors.imu_sensor
```

`examples.py test` runs two tests. `default` simulates the drop and checks the readings it prints.
`self_test` runs the unit tests in `test_imu.py`, which cover the sensor mathematics directly:
composition order, the lever arm, the transport term, orientation averaging, gravity, and the
timestamp handling. They are plain functions with plain `assert` statements, so they also run under
pytest:

```bash
python -m pytest test_imu.py            # from this directory
python main.py --self-test              # same tests, no pytest needed
```

After activating the required Python environment, `main.py` can also be run directly from this directory.
