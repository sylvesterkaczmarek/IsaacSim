# Overview

```{deprecated} 6.0.0
This extension has been deprecated and will be replaced by open source equivalents and simple examples.
See {ref}`isaac_sim_app_tutorial_state_machines` for replacement behavior-programming tutorials.
```

## Migration

There is no drop-in import replacement for `isaacsim.cortex.examples`. Move the
example workflow into your application or start from the replacement standalone
examples in {ref}`isaac_sim_app_tutorial_state_machines`.

Replace the reusable sample UI base classes with the supported examples base
package:

```python
# Old: deprecated Cortex examples UI scaffolding.
from isaacsim.cortex.examples.base_sample import BaseSample, BaseSampleUITemplate

# New: supported examples base package.
from isaacsim.examples.base import BaseSample, BaseSampleUITemplate
```

Replace Cortex sample imports with application-owned examples. For example:

```python
# Old: full Cortex sample classes.
from isaacsim.cortex.examples.franka_cortex.franka_cortex import FrankaCortex
from isaacsim.cortex.examples.ur10_palletizing.ur10_palletizing import BinStacking

# New: project-owned examples based on the replacement tutorials.
from my_app.examples.franka_stack import FrankaStackExample
from my_app.examples.ur10_palletizing import Ur10PalletizingExample
```

Update extension and command-line workflows:

- Remove `isaacsim.cortex.examples`, `isaacsim.cortex.behaviors`, and
  `isaacsim.cortex.framework` from extension dependencies and enabled
  extensions.
- Replace example-browser entries under **Robotics Examples > Cortex** with your
  own extension entry, or run the replacement standalone scripts directly.
- Replace launch commands that enable Cortex examples, such as
  `APP_SCRIPT --enable isaacsim.cortex.examples`, with one of the replacement
  scripts:

```bash
./python.sh standalone_examples/tutorials/state_machine/franka_pick_place_fsm.py
./python.sh standalone_examples/tutorials/state_machine/franka_pick_place_py_trees.py
./python.sh standalone_examples/tutorials/state_machine/ur10_palletizing_transitions.py
```

Common API and feature mapping:

| Cortex examples API or feature | Replacement |
| --- | --- |
| `BaseSample` / `BaseSampleUITemplate` from `isaacsim.cortex.examples.base_sample` | `BaseSample` / `BaseSampleUITemplate` from `isaacsim.examples.base`. |
| `CortexBase` / `CortexWorld` sample lifecycle | Application lifecycle code using explicit stage setup, simulation callbacks, and reset/cleanup handlers. |
| `FrankaCortex` sample class | Franka state-machine or behavior-tree example from {ref}`isaac_sim_app_tutorial_state_machines`, moved into an application module. |
| `BinStacking` / `BinStackingTask` sample classes | UR10 `transitions` palletizing example from {ref}`isaac_sim_app_tutorial_state_machines`, moved into an application module. |
| Behavior dropdown loading Cortex behavior modules | Application-owned behavior selection that instantiates a state machine, `py_trees` tree, or task object. |
| Cortex decision-stack diagnostics UI | Application UI or logs driven by the replacement behavior's current state, behavior-tree status, or task context. |

`**isaacsim.cortex.examples**` provides example environments and behavior scripts for the Isaac Cortex decision framework. It focuses on manipulation examples that show how Cortex behaviors can drive robot tasks, including Franka behavior demos and a UR10 bin stacking scenario. The extension presents these examples through interactive UI entries in the examples browser, with controls for loading scenes, starting tasks, resetting worlds, and viewing behavior diagnostics.

## Functionality

The extension includes two main Cortex example workflows:

- **Franka Cortex examples**: Demonstrates behavior-driven control for a Franka robot with multiple selectable behaviors, including block stacking, state machines, decider networks, and peck game style interactions.
- **UR10 Palletizing example**: Demonstrates autonomous bin stacking with a UR10 robot, suction gripper, moving conveyor, randomized bin spawning, and Cortex decision logic for picking, flipping, and stacking bins.

Both workflows expose common simulation controls such as loading the world, resetting the environment, and starting the selected behavior or task. They also provide diagnostic feedback from Cortex so users can observe what the decision system is doing while the simulation runs.

## Concepts

### CortexWorld

The examples use `CortexWorld` instead of a standard `World` for Cortex-based task execution. This allows the examples to register Cortex decision networks, behavior monitors, and physics callbacks used to step robot behaviors during simulation.

### Decision diagnostics

The UI displays diagnostic information produced by the running Cortex behavior. For the Franka examples, this includes diagnostic messages and the current decision stack. For the UR10 palletizing example, diagnostics include the decision stack, selected bin, bin base, grasp status, attachment status, and whether a bin needs to be flipped.

### Sample lifecycle

The examples follow a shared sample structure built around loading a world, setting up a scene, performing post-load initialization, resetting simulation state, and clearing task state. This keeps the Franka and UR10 examples consistent while allowing each example to define its own robot, scene assets, tasks, and behavior logic.

## UI Components

### Shared Sample UI

`BaseSampleUITemplate` provides the common UI layout used by the examples. It includes:

- Header information for the example title, documentation link, and overview text.
- World controls with Load World and Reset buttons.
- A container for example-specific UI frames.
- Stage and timeline event handling used to update button states and reset UI state.

This shared layout lets each example add task-specific controls and diagnostics without redefining the basic world control interface.

### Franka Cortex UI

`FrankaCortexUI` provides controls for selecting and running Franka Cortex behaviors. The UI includes:

- A behavior selection dropdown.
- Load and reset controls.
- A Start button for running the selected behavior.
- A diagnostics panel showing the current decision stack and diagnostic message.

The selected behavior maps to a behavior script that is loaded into the Franka Cortex sample. The diagnostic panel updates while the behavior runs, making it easier to see how the robot behavior is progressing.

### UR10 Palletizing UI

`BinStackingUI` provides controls and diagnostics for the UR10 bin stacking task. The UI includes:

- A Start Palletizing button.
- A decision stack display.
- Diagnostic fields for selected bin, bin base, grasp reached, attachment state, and flip requirement.

The UI is designed around observing the bin stacking workflow as bins arrive on the conveyor and the robot decides how to pick and stack them.

## Key Components

### FrankaCortex

`FrankaCortex` defines the Franka robot example scene and behavior execution flow. It creates a scene with:

- A Franka robot at `/World/Franka`.
- Four colored cube obstacles: Red, Blue, Yellow, and Green.
- A default ground plane.

The sample can load different Cortex behavior modules, create their decider networks, and connect diagnostics back to the UI through a monitor function.

### BinStacking

`BinStacking` defines the UR10 palletizing simulation. It sets up a UR10 robot workspace with suction gripper behavior, collision obstacles, dynamic bin spawning, and a Cortex decision network for autonomous manipulation.

The robot behavior covers the full bin handling loop: detecting bins, planning collision-free motion, handling flipped bins, grasping with the suction gripper, moving to the stacking area, and placing bins.

### BinStackingTask

`BinStackingTask` manages the dynamic bin lifecycle in the UR10 example. It spawns bins on the conveyor with randomized positions and orientations, tracks active and stashed bins, and resets task state when the world is reset.

The helper `random_bin_spawn_transform()` generates varied bin spawn poses, including a chance for bins to appear upside down.

### Ur10Assets

`Ur10Assets` centralizes the USD asset paths required by the UR10 example. These include the UR10 table setup with suction gripper, small KLT bins, warehouse background, and Rubik's cube props. The paths are resolved relative to the Isaac Sim assets root directory.

## Relationships

`**isaacsim.cortex.examples**` is built around the Cortex framework:

- It uses `**isaacsim.cortex.framework**` for `CortexWorld`, decision networks, and diagnostics monitoring.
- It uses `**isaacsim.cortex.behaviors**` for the behavior scripts demonstrated by the Franka and UR10 examples.
- It uses `**isaacsim.core.api**` for simulation worlds, scenes, tasks, and world operations.
- It uses `**isaacsim.examples.browser**` to present the Franka Cortex and UR10 Palletizing examples under the Cortex category.
- It uses `**isaacsim.gui.components**` and `**omni.ui**` based UI patterns for the shared sample controls and diagnostic panels.
