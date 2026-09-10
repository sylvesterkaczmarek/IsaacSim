# Overview

```{deprecated} 6.0.0
This extension has been deprecated and is kept only as reference material for
existing Cortex users. There is no drop-in replacement package.

For migration guidance, see
{ref}`isaacsim_cortex_to_open_source_workflows`. For replacement
behavior-programming examples, see
{ref}`isaac_sim_app_tutorial_state_machines`.
```

## Migration

There is no drop-in package replacement for `isaacsim.cortex.framework`. Move
workflow ownership into the application, then replace Cortex decision logic with
application-owned state machines, behavior trees, or task planners.

### Import replacements

| Deprecated import | Replacement |
| --- | --- |
| `from isaacsim.cortex.framework.cortex_world import CortexWorld` | Use application lifecycle code with `isaacsim.core.experimental.utils.stage` utilities and `isaacsim.core.simulation_manager.SimulationManager` callbacks. |
| `from isaacsim.cortex.framework.df import DfNetwork, DfDecider, DfAction, DfState, DfStateMachineDecider` | Reimplement the behavior with an application-owned state machine, a `py_trees` behavior tree, or a `transitions` state machine. |
| `from isaacsim.cortex.framework.dfb import DfRobotApiContext, DfDiagnosticsMonitor` | Move robot context and diagnostics into application-owned classes that update once per simulation tick. |
| `from isaacsim.cortex.framework.motion_commander import MotionCommander, MotionCommand, ApproachParams` | Use supported motion-generation controllers and examples from `isaacsim.robot_motion.experimental.motion_generation`, cuMotion, or PINK. |
| `from isaacsim.cortex.framework.obstacle_monitor_context import ObstacleMonitor, ObstacleMonitorContext` | Track obstacles in application state and update the active motion-generation world before command generation. |
| `from isaacsim.cortex.framework.robot import CortexRobot, CortexUr10, add_franka_to_stage, add_ur10_to_stage` | Set up robot assets and command paths in the application with supported manipulator and robot-control APIs. |

### Command-line and extension changes

| Deprecated workflow | Replacement |
| --- | --- |
| `APP_SCRIPT.sh --enable isaacsim.cortex.framework` or `APP_SCRIPT.bat --enable isaacsim.cortex.framework` | Run an application or standalone script that owns its behavior logic and does not enable `isaacsim.cortex.framework`. |
| Add `"isaacsim.cortex.framework" = {}` to an extension dependency list | Remove the dependency. Add only the supported packages used by the migrated app, such as `py_trees`, `transitions`, robot-control, or motion-generation packages. |
| Run `cortex_main.py --usd_env ...` for the Cortex loop runner | Load the stage from the application entry point, register simulation callbacks, and tick the migrated state machine, behavior tree, or planner from that app loop. |
| Use `cortex_main.py --enable_ros` to start Cortex ROS helpers | Start the ROS 2 bridge and project ROS nodes explicitly from the migrated application workflow. |

### API mapping

| Cortex concept | Migration strategy |
| --- | --- |
| Cortex loop runner and `CortexWorld.add_decider_network(...)` | Register an application simulation callback that updates world state, ticks the behavior policy, and sends robot commands. |
| Belief model paths such as `/cortex/belief` and `/cortex/sim` | Keep only the scene state needed by the migrated application. Store logical state in application data structures or sensors, not Cortex-specific USD conventions. |
| `DfNetwork`, `DfDecider`, `DfAction`, and `DfState` | Replace with a state machine, behavior tree, or task planner. Preserve the original decision priorities as tests. |
| `DfRobotApiContext` monitors and diagnostics | Update logical robot state once per tick before evaluating the behavior policy. |
| `MotionCommander` and Cortex commanders | Issue commands through supported manipulator, robot-control, and motion-generation APIs. |
| Cortex ROS helpers in `cortex_ros.py` and `cortex_sim.py` | Use ROS 2 bridge workflows and application-owned publishers, subscribers, or action clients. |
| Cortex example behavior modules | Treat them as reference material only and port the task logic into application-owned modules. |

### Replacement examples

- {ref}`isaac_sim_app_tutorial_state_machines`
- `standalone_examples/tutorials/state_machine/franka_pick_place_ifelse.py`
- `standalone_examples/tutorials/state_machine/franka_pick_place_fsm.py`
- `standalone_examples/tutorials/state_machine/franka_pick_place_py_trees.py`
- `standalone_examples/tutorials/state_machine/ur10_palletizing_transitions.py`

Use those examples as starting points for behavior logic that should no longer
depend on `isaacsim.cortex.framework`, `isaacsim.cortex.behaviors`, or
`isaacsim.cortex.examples`.

`isaacsim.cortex.framework` provided a decision framework for orchestrating
Isaac Sim robot workflows and executing behavior logic on simulated or physical
robots. It included:

- A Cortex loop runner with a belief model of the world and robot.
- ROS bridge helpers for synchronizing Cortex belief state with physical robot
  perception and actuation.
- A simulated-controller workflow for hardware-in-the-loop development.
- Example environments and behavior scripts, including reactive block stacking.

## Deprecated usage

The Cortex command-line runner, ROS bridge workflow, physical-robot
quickstarts, world setup conventions, and file breakdown below are deprecated.
They remain here only to help existing projects understand and remove their
Cortex dependencies.

## World setup conventions

Cortex USD worlds follow a particular path naming convention. Good examples
are:

```text
Isaac/Samples/Cortex/Franka/BlocksWorld/cortex_franka_blocks_belief.usd
Isaac/Samples/Cortex/Franka/BlocksWorld/cortex_franka_blocks_belief_sim.usd
Isaac/Samples/Cortex/UR10/Basic/cortex_ur10_basic_belief.usd
Isaac/Samples/Cortex/UR10/Basic/cortex_ur10_basic_belief_sim.usd
```

It is assumed these environments are setup in units of centimeters.

The belief environment is added to the path `/cortex/belief` and the simulation
environment, if it exists, is added to `/cortex/sim`. Each environment contains
`robot` and `objects` subprims. The robot has a string metadata attribute
`cortex:robot_type` telling the system the robot type. Currently supported
values are `franka` and `ur10`.

Objects added to the scene can have an optional `cortex:is_obstacle` attribute.
When set to `True`, Cortex loads the object as an obstacle. If the attribute is
not present, the object is assumed to not be an obstacle.

All xform prims representing robots and objects follow the Isaac Sim Core API
transform specification USD conventions. Specifically, they have transform
attributes specified by `xformOp:translate`, `xformOp:orient`, and
`xformOp:scale`, with `xformOpOrder` given as
`[xformOp:translate, xformOp:orient, xformOp:scale]`.

For instance, the block stacking environment containing both belief and
simulation worlds is laid out as:

```text
/cortex
  /belief
    /robot  # Franka USD with cortex:robot_type of 'franka'
    /objects
      /red_block    # cortex:is_obstacle = True
      /yellow_block # cortex:is_obstacle = True
      /green_block  # cortex:is_obstacle = True
      /blue_block   # cortex:is_obstacle = True
  /sim
    /robot  # Franka USD with cortex:robot_type of 'franka'
    /objects
      /red_block
      /yellow_block
      /green_block
      /blue_block
```

It is often useful to set up a common environment that is shared by both
`/cortex/belief` and `/cortex/sim`. That makes it easy to set up both
belief-only and belief-simulation variants of the USD environment.

Other attributes and prims:

- The belief robot has `cortex:adaptive_cycle_dt` (Double) and
  `cortex:is_suppressed` (Bool). These are used internally by Cortex and are
  automatically added on startup. It is okay if the USD environment already has
  them.
- `/cortex/belief/motion_controller_target` is a cube prim used for manually
  controlling the robot. If it already exists in the environment, Cortex uses
  it. Otherwise, Cortex creates one when initializing the motion commander.

## Breakdown of files

Main Cortex loop runner:

- `cortex_main.py`: Primary entry point and main Cortex loop runner. This runs
  the standalone Cortex Python app. It points to its own experience file, which
  includes the `isaacsim.cortex.framework` extension. A Cortex-compatible USD
  environment is passed in with a flag. It starts the `cortex_ros` and
  `cortex_sim` extensions. `cortex_ros` is always running so a physical robot
  can be connected at any time. If the USD environment has a simulation
  environment, that robot is used in place of a physical robot.

Extensions loaded on startup:

- `cortex_ros.py`: Handles ROS connections to get perceptual information into
  Cortex and send control information out of Cortex.
- `cortex_sim.py`: Creates the ROS communication interface to mimic a physical
  robot using a simulated environment.

Decision framework:

- `df.py`: Core framework tools, including decider networks and state machines.
- `dfb.py`: Decision-framework behaviors shared across behavior scripts.
- `df_behavior_watcher.py`: Monitors `df_behavior_module.py` for changes and
  reloads when a change is detected.
- `df_behavior_module.py`: Behavior module monitored by the main Cortex loop
  runner. On startup, nothing runs until a behavior is explicitly activated.

Cortex tools:

- `cortex_utils.py`: Utilities for setting up Cortex.
- `cortex_object.py`: Object representation wrapping Core API objects and
  Cortex attributes, including measured poses.
- `motion_commander.py`: Wrapper around Isaac Sim motion policies that provides
  a command API with pose targets and approach directions.
- `smoothed_command.py`: Tool for smoothing commands automatically.
- `synchronized_time.py`: ROS utility for clock synchronization between the
  embedded robot controller and the machine running Cortex.

Utilities:

- `cli.py`: Helpers for command-line interfaces.
- `gf_conversions.py`: Helpers for reading and writing USD data through the Gf
  interface.
- `math_util.py`: Math tools and utilities.
- `ros_tf_util.py`: ROS-based utilities.
- `tools.py`: Utilities for steady loops and profiling.

Tests:

- `tests/test_df.py`: Unit tests for the decision framework.
- `tests/test_motion_commander.py`: Standalone Python app that starts a motion
  commander in a basic Franka environment.

## Troubleshooting

When restarting the controller for a physical robot, bring down the entire
controller manager on the real-time machine and restart everything.
