# Overview

```{deprecated} 6.0.0
This extension has been deprecated and will be replaced by open source equivalents and simple examples.
See {ref}`isaac_sim_app_tutorial_state_machines` for replacement behavior-programming tutorials.
```

## Migration

There is no drop-in import replacement for `isaacsim.cortex.behaviors`. Move the
behavior module into your application and rebuild it with an application-owned
state machine, behavior tree, or task planner.

Replace Cortex behavior imports with project-owned modules. For example:

```python
# Old: behavior logic imported from the deprecated Cortex package.
from isaacsim.cortex.behaviors.franka import block_stacking_behavior

network = block_stacking_behavior.make_decider_network(robot)

# New: behavior logic owned by the application.
from my_app.behaviors.franka_stack import build_behavior

behavior = build_behavior(robot)
```

Update extension and launch workflows that enable Cortex behavior packages:

- Remove `isaacsim.cortex.behaviors` from extension dependencies and enabled
  extensions.
- Stop loading tasks through `make_decider_network(...)` from this package.
- Run the replacement behavior from your application lifecycle code, and keep
  robot command logic in supported robot and motion-generation APIs.
- Use {ref}`isaac_sim_app_tutorial_state_machines` as the replacement tutorial
  path for `py_trees`, `transitions`, and library-free state-machine examples.

Common API mapping:

| Cortex behavior entry point | Replacement strategy |
| --- | --- |
| `make_decider_network(...)` | Application-owned behavior factory that returns a state machine, `py_trees` tree, or task object. |
| `DfNetwork` | Behavior tree root, `transitions.Machine`, or an application scheduler. |
| `DfDecider` / `DfAction` | `py_trees` `Behaviour` nodes or explicit state/action methods. |
| `DfState` / `DfStateMachineDecider` | Library-free state class or `transitions` state with `on_enter_*` hooks. |
| `DfRobotApiContext` / behavior context classes | Application-owned task context updated once per simulation tick. |
| `ObstacleMonitorContext` / obstacle monitors | Application-owned monitor callbacks or state lifecycle hooks that toggle obstacles. |

`**isaacsim.cortex.behaviors**` provides sample robot behaviors built on the Isaac Cortex decider framework. It includes manipulation examples for block stacking, bin stacking, pecking behaviors, and simple decider or state-machine demonstrations. These behaviors are useful as reference implementations for how Cortex contexts, monitors, deciders, actions, and states work together to drive robot tasks.

The extension focuses on behavior structure rather than UI. Most examples are organized around a `make_decider_network(...)` entry point that builds a configured Cortex behavior network for a robot.

## Concepts

### Cortex behavior networks

The behaviors are built from Cortex decision framework primitives such as `DfNetwork`, `DfDecider`, `DfState`, `DfAction`, and `DfStateMachineDecider`.

A typical behavior separates responsibilities into:

- A context that stores task state and monitors the world.
- Deciders that choose what behavior branch should run next.
- States or actions that send robot commands.
- Monitors that continuously update derived state, such as active objects, gripper state, obstacle requirements, or diagnostics.

This structure makes the examples useful for understanding reactive robot behavior. The decider can change its decision when the context detects that the scene has changed.

### Robot contexts

Several examples define context classes derived from `DfRobotApiContext` or `ObstacleMonitorContext`. These contexts keep track of task-specific data, such as active blocks, target positions, tower state, bin state, or whether the gripper is holding an object.

Context monitor methods are used to update the behavior over time. For example, the block stacking context monitors the tower, gripper state, perception drift, collision avoidance suppression, and diagnostics.

## Functionality

### Block stacking

The block stacking behavior models a pick-and-place task where a robot builds a tower from blocks in a desired order. It includes logic for selecting the next block, computing candidate grasp transforms, choosing a grasp that avoids nearby blocks, picking the block, and placing it either on the tower or back on the table.

Important behavior pieces include:

- `BuildTowerContext`, which tracks all blocks, the active block, the tower stack, gripper state, placement targets, and diagnostics.
- Grasp helpers that generate object-local and world-frame grasp transforms.
- Deciders for choosing between tower buildup and teardown behavior.
- Pick and place RLDS nodes that run when the end-effector is close enough to the grasp or placement target.
- Table placement sampling that avoids blocks and the tower.

The tower state is monitored from block positions near the tower location and ordered by height. The behavior can detect whether the stack is complete and whether the current order matches the desired order.

### Bin stacking

The bin stacking behavior is a larger pick, flip, and place example for a UR10-style bin stacking workflow. It tracks bins arriving in a conveyor region, computes grasp transforms, flips bins when required, and places bins into a predefined stack layout.

Important behavior pieces include:

- `BinStackingContext`, which tracks bins, the active bin, stacked bins, stack coordinates, and elapsed time.
- `BinState`, which stores per-bin state such as grasp transform, attachment state, and whether the bin needs flipping.
- Pick, flip, and place state machines.
- Suction gripper states for closing, opening, and retrying closure.
- Obstacle monitors for flip-station and navigation obstacles.
- Diagnostics reporting through `BinStackingDiagnosticsMonitor`.

The bin behavior also includes movement states that check both position and rotation thresholds before completing. Navigation obstacle monitoring is toggled during movement where obstacle state depends on the end-effector and target side.

### Peck behaviors

The extension includes several pecking examples that show different ways to express similar behavior patterns.

The peck examples include:

- A simple peck state machine that samples a target and moves the end-effector toward it.
- A peck decider network that chooses targets, pecks, lifts, and marks the task as done.
- A reactive peck game behavior that detects moved blocks and pecks at the active block.
- Obstacle-aware target sampling that avoids registered obstacles.

These examples are useful for comparing a sequential state-machine style with a decider-based style. The decider variants can react to context changes, such as a target becoming blocked or a block moving.

### Simple examples

The extension also includes smaller examples that demonstrate the framework with minimal task logic:

- A simple state machine that moves the end-effector to a target position and exits on arrival.
- A simple decider network that monitors the end-effector y-position and dispatches different print actions depending on whether the robot is left, right, or in the middle.

These examples are intended to show the basic structure of contexts, monitors, actions, states, and decider dispatch without the complexity of object manipulation.

## Key Components

### Contexts

Contexts hold the task state used by deciders and states. Examples include `BuildTowerContext`, `PeckContext`, `BinStackingContext`, and the simple lateral-position `Context`.

They are responsible for monitoring the scene and updating variables that guide behavior decisions. For example, a context may determine which block is active, whether the gripper is clear, or whether a bin has reached its grasp pose.

### Deciders

Deciders choose which child behavior should run next. The top-level dispatch deciders route execution based on context state, such as whether the robot is holding an object, whether the tower is complete, or whether a bin needs flipping.

Examples include:

- `BlockPickAndPlaceDispatch`
- `ChooseNextBlock`
- `ReachToPlacementRd`
- `Dispatch` in the peck and bin stacking examples

### States and actions

States and actions perform the actual robot-level steps. They send motion commands, wait for convergence, operate the gripper, or update context flags.

Common patterns include:

- Move until the end-effector reaches a target pose.
- Close or open the gripper.
- Lift the end-effector after a pick or peck.
- Mark an object as complete after placement.

### Obstacle monitors

The bin stacking behavior uses `ObstacleMonitor` and `ObstacleMonitorContext` from `**isaacsim.cortex.framework.obstacle_monitor_context**`. These monitors determine when specific obstacles should be active based on current task state.

For example, the flip station obstacle is required when approaching a bin near the flip station, while navigation obstacles are toggled based on whether the target and end-effector are crossing between regions.

### Diagnostics

Several behaviors maintain diagnostic state for visibility into what the behavior is doing. The block stacking and peck game contexts update diagnostic messages periodically, while bin stacking can report a `BinStackingDiagnostic` object through a callback.

Diagnostics include information such as the active object, grasp state, attachment state, and task progress.

## Relationships

`**isaacsim.cortex.behaviors**` is built on `**isaacsim.cortex.framework**`.

The behavior modules use Cortex framework modules directly, including:

- `**isaacsim.cortex.framework.df**` for deciders, actions, states, state-machine deciders, RLDS nodes, and networks.
- `**isaacsim.cortex.framework.dfb**` for robot API contexts and diagnostics monitors.
- `**isaacsim.cortex.framework.obstacle_monitor_context**` for obstacle-aware bin stacking behavior.

The behavior modules assume a robot API object is provided when constructing a decider network. That robot object is used by the contexts and states to read end-effector state, send motion commands, and operate grippers.
