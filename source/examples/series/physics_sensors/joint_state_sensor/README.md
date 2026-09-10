<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Joint State Sensor

`main.py` simulates a two-degree-of-freedom robot -- a carriage driven down a fixed mast while a
bar spins freely on top of it -- and reports the four quantities a joint state sensor exposes for every
DOF: name, position, velocity, and measured effort. It adds the joint accelerations, which the
shipped sensor does not report and which a finite difference gives for free.

## What it shows

Everything the sensor does lives in this directory. `joint_state.py` holds the sensor logic and
`main.py` drives it from a simulation, so the whole derivation is readable in one place without
following it into a library.

`joint_state.py` deliberately knows nothing about a stage, a physics engine, or a simulation.
Callers push one `ArticulationState` per step with `sample()` and pull a filtered
`JointStateReading` with `read()`, so the raw state could equally come from a recorded trajectory
or a closed-form one. Everything a joint state sensor handles is a per-DOF scalar, so unlike the
IMU example there are no vectors or quaternions here and the module depends on nothing outside the
standard library.

That leaves one distinction doing all the work: a DOF is either a **rotation** or a **translation**.
A rotation reads radians and its effort is a torque; a translation reads stage length units and its
effort is a force. Which of the two a DOF is decides every conversion in the module, and it is the
only thing `joint_state.py` needs to know about a robot beyond the DOF's name.

`main.py` supplies the raw state from `isaacsim.physics.entities.ArticulationEntity` and makes the
checks. That wiring is where the interesting details live:

- **Reading DOF state.** `ArticulationEntity.get_dof_positions()`, `get_dof_velocities()`, and
  `get_dof_projected_joint_forces()` each return one value per DOF in articulation order, which is
  the order `joint_state.py` indexes, so the per-step state is passed straight through.
- **Not `get_dof_efforts()`.** The entity has a method by that name and it is a different quantity:
  the actuation force a drive commands, which is zero throughout here because neither joint is
  driven. What the shipped sensor calls `efforts` is the *measured* joint force, which is
  `get_dof_projected_joint_forces()`.
- **DOF metadata** comes from USD instead, through `isaacsim.foundation.prims.Articulation`, which
  reports `dof_names` and `dof_types` before the simulation starts. USD spells the two DOF kinds
  `"rotation"` and `"translation"`, which is exactly what `joint_state.py` uses. The shipped C++
  sensor reports the same distinction as `0` and `1`, so a caller reading from it converts with
  `joint_state.dof_type_from_code()` instead.
- **Driving the simulation.** Everything goes through `PhysicsManager`: `setup()` fixes the step
  size once, `step()` advances, `get_simulated_time()` reports how far, and `invalidate()` tears
  down. The module-level `initialize`, `simulate` and `close` functions do the same job and are on
  their way out of the public API, so nothing here uses them.
- **Timestamps.** The buffers hold the state *after* a step, so each sample is stamped with the
  time the step advanced to, which is what `get_simulated_time()` already reports. Nothing has to
  count steps and multiply.
- **Selection by name.** A consumer with a fixed message layout, such as a ROS 2
  `sensor_msgs/JointState` publisher, needs its own DOF order rather than the articulation's.
  `JointStateSensor(..., names=[...])` picks and orders the DOFs once, at construction; a name that
  matches no DOF, or more than one, is an error rather than a short message. The raw arrays are
  still indexed by articulation order, so a selection never changes what a sample must contain.

## The centimetre stage

`SCENE` is authored in centimetres, and that is the point of the example rather than an
incidental choice. Every joint quantity a physics engine reports is in stage units, so on this
stage the sliding DOF reads centimetres and centimetres per second while the spinning DOF reads
radians and radians per second whatever the stage's linear unit is. That is why the shipped
`JointStateSensorReading` carries `stage_meters_per_unit` at all: it reports the raw values and
leaves the conversion to the consumer, and a consumer that applies it to every DOF alike is as
wrong as one that applies it to none.

`joint_state.py` therefore converts per DOF type:

| Quantity | Rotation DOF | Translation DOF |
| --- | --- | --- |
| position | rad, unscaled | `meters_per_unit` |
| velocity | rad/s, unscaled | `meters_per_unit` |
| effort | `kilograms_per_unit * meters_per_unit^2` (a torque) | `kilograms_per_unit * meters_per_unit` (a force) |

An effort carries one more power of length than the position it acts along, and the mass unit that
neither a position nor a velocity sees. `kilograms_per_unit` is a second stage unit the shipped
reading does not carry, so it is a separate argument here, read from the stage alongside
`meters_per_unit`.

## Efforts are measured, not commanded

What the shipped sensor exposes as `efforts` -- `get_dof_projected_joint_forces()` on the entity,
`dof_efforts` inside the C++ plugin -- is the force the joint constraint carries projected onto the
DOF axis. It is not the actuation force a drive commands, and the two DOFs make the difference
visible side by side. The lift is carrying the carriage and the bar, and reports the 29.4 N that
takes -- while it is descending, held up by its drive, and again once it is parked on its limit,
because what it measures is the load either way. The spin DOF's axis is vertical, so nothing
resists it, and it reports zero however fast it turns. Neither number is a commanded one: the
lift's drive is what produces its 29.4 N, but `get_dof_efforts()`, the *actuation* force, reads
zero for both.

The Newton backend does not report measured efforts at all. The shipped sensor substitutes zeros,
which a consumer cannot tell from a real zero; `ArticulationState.efforts` is `None` in that case
and the reading carries `efforts_available = False` alongside the same zeros.

## The checks, and where they live

`main.py` checks only what needs a running simulation. Both DOFs move at a rate fixed by
construction, so both have a closed form to check against: a velocity drive lowers the carriage at
a constant 50 cm/s until it reaches the joint limit, and the bar keeps whatever rate it is given
because nothing exerts a torque about a vertical axis.

The check with teeth is the velocity one, and it is made twice. The descending lift velocity is
read straight off the engine buffer, before `joint_state.py` sees it, and has to be the drive's
rate **in stage units** -- 50, not 0.5. It then has to be 0.5 in SI once the linear unit is
applied. Getting the conversion backwards, or applying it twice, fails one or the other. The
spinning DOF beside it has to stay at exactly its own rate, because radians are not stage units and
must not pick up the scale at all.

A constant rate is what makes this exact. A filtered velocity is a mean over the window, so it lags
the newest sample by half a window -- but only while the value is changing, and here it is not.
Positions come from the newest sample rather than the window, so the spinning DOF's angle can be
checked against its rate times the reading's timestamp exactly.

The reported acceleration is only checked against zero. That is a weak value but not a free one: it
is the one place a simulation exercises the finite difference at all, and a difference taken over
the wrong interval, or against the wrong window, would not land on zero. Everything else about the
difference is covered by the unit tests, against synthetic states where the answer is not zero.

Everything that does **not** need a simulation is a unit test in `test_joint_state.py` instead,
because this scene is geometrically clean in ways that hide real mistakes. Both DOF axes are
vertical, the stage's mass unit is kilograms so the mass scale is 1 and could be dropped without
anything noticing, no sample ever arrives out of order, and the sensor is only ever asked for every
DOF in articulation order. `test_joint_state.py` covers each of those against synthetic states
chosen so the quantity under test actually changes the answer: a centimetre stage whose mass unit
is grams, DOF counts that do not match, windows whose timestamps repeat or rewind, and selections
that reorder, drop, repeat, or fail to name a DOF.

## Note on the scene being USDA text

The IMU example authors its scene with the object API. This one cannot: a joint names the two
bodies it connects through the `physics:body0` and `physics:body1` **relationships**, and
`isaacsim.foundation.objects.Prim` writes attributes only, so there is no way to point a joint at a
link through it. Nor is that a gap only in the writing direction --
`isaacsim.foundation.prims.Articulation` has a standing `TODO` to scope its search by walking those
same relationships "once the USD backends expose relationship targets". `pxr` is not among the
Python distributions the examples install, so the alternative is USD text.

`Stage.import_stage_from_string()` takes that text directly, so the robot stays in `main.py` as the
`SCENE` constant rather than becoming an asset file beside it. Reading it as a whole is a fair
trade for the lines it costs: three links, a fixed joint anchoring the mast to the world, and the
two joints that become the DOFs.

`main.py` builds two stages from that one string. The first is opened only to be asked for the DOF
names, types, and stage units, which are USD facts available before any simulation exists, and is
then closed. The second is the `ovstage` the simulation actually runs on.

## Files

`joint_state.py` holds the sensor logic and depends on nothing but the standard library.
`test_joint_state.py` holds its unit tests. `main.py` holds the robot, runs the simulation, and
makes the checks. `viewer.py` holds the optional live view and is used only when
`--ui` is passed. It takes the simulation as callables rather than importing `main.py`, so it knows
nothing about the physics engine or the scene and the dependency runs one way only.

## Live view (optional)

`--ui` opens a window showing the robot and its joint state at the same time. The robot is drawn
**from the reading alone**: the two joint values place the carriage on the mast and both ends of the
spinning bar, which is what a joint state sensor is for. Beside it are the live per-DOF values and
rolling traces of each DOF's speed.

The run takes about two seconds, after which the DOFs are written back to their start state and it
begins again, so there is always something moving. Each restart calls `JointStateSensor.reset()`,
because writing the DOFs directly makes the finite difference across that jump meaningless. Unlike
the default run, the viewer makes no resting-state assertions: it exits whenever the window is
closed, which may be mid-descent.

It is off by default and needs two packages this example deliberately does **not** declare, because
`example.toml` requirements accept only Isaac Sim distributions and the system capabilities listed
in the collection README:

```bash
python -m pip install ovui==0.1.1 glfw==2.10.0     # into the build's Python
python examples.py run physics_sensors.joint_state_sensor -- --ui
```

`ovui` is the standalone distribution of Omniverse's `omni.ui`, imported as `omni.ui`.
`source/tools/zmq_bridge/zmq_server.py` also runs it outside Kit, and additionally hides the `carb`
and `omni.kit.app` module specs during import, which this example does not need because `carb` is
not importable from the standalone build. A display is required -- the viewer reports an error
rather than opening a blank window if none is available. The automated tests never pass `--ui`, so
they neither need nor exercise these packages.

## Run it

From the examples collection root:

```bash
python examples.py run physics_sensors.joint_state_sensor
python examples.py test physics_sensors.joint_state_sensor
```

`examples.py test` runs two tests. `default` simulates the robot and checks the readings it prints.
`self_test` runs the unit tests in `test_joint_state.py`, which cover the sensor logic directly:
unit conversion per DOF type, selection by name, filtering, the finite difference, effort
availability, and the timestamp handling. They are plain functions with plain `assert` statements,
so they also run under pytest:

```bash
python -m pytest test_joint_state.py    # from this directory
python main.py --self-test              # same tests, no pytest needed
```

After activating the required Python environment, `main.py` can also be run directly from this
directory.
