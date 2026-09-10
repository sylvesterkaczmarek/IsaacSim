# Overview

```{deprecated} 6.0.0
This extension is deprecated. Use `isaacsim.robot_motion.experimental.motion_generation` and `isaacsim.robot_motion.cumotion` instead.
```

`**isaacsim.robot_motion.motion_generation.examples**` provides UI-based tutorials for robot motion generation examples using the Lula planning library. The extension is organized as separate example modules for RMPflow, RRT, kinematics, and trajectory generation, each exposing a {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>` class that builds the tutorial interface and connects it to the simulation timeline, stage events, and physics steps.

The examples are focused on interactive robot motion workflows. Users can load a prepared robot scenario, reset the scene, and run motion generation demonstrations through dedicated controls.

## Concepts

### Example Modules

The extension contains four sibling tutorial modules:

- `**isaacsim.robot_motion.motion_generation.examples.rmp_flow**`
- `**isaacsim.robot_motion.motion_generation.examples.rrt**`
- `**isaacsim.robot_motion.motion_generation.examples.kinematics**`
- `**isaacsim.robot_motion.motion_generation.examples.trajectory_generator**`

Each module follows the same structure: an extension-level window delegates tutorial-specific UI and behavior to a {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>` class. The {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>` is the main public API for each module.

### Scenario-Based Tutorials

Each {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>` works with a specific scenario class:

- `FrankaRmpFlowExample` for the RMPflow tutorial
- `FrankaRrtExample` for the RRT path planning tutorial
- `FrankaKinematicsExample` for the kinematics tutorial
- `UR10TrajectoryGenerationExample` for the trajectory generation tutorial

The scenario classes provide the robot-specific demonstration logic, while {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>` provides the controls and event handling needed to run the tutorial.

## Functionality

### World Controls

Each example UI includes a World Controls frame for scene setup. The controls allow users to load the tutorial scenario and reset the scene state.

The UI also handles common scene preparation tasks, including camera positioning and lighting setup for visualization.

### Scenario Controls

Each example includes controls for running the motion generation demonstration.

For the Franka examples, the Run Scenario frame provides a state button that starts and stops the demonstration:

- RMPflow demonstrates robot motion generation using RMPflow.
- RRT demonstrates path planning using Rapidly-exploring Random Tree planning.
- Kinematics demonstrates robot kinematics behavior.

The trajectory generation example provides separate controls for multiple trajectory types:

- Configuration space trajectories
- Task space trajectories
- Advanced trajectory demonstrations

### Simulation Event Handling

The UI builders respond to simulation and scene events so the controls stay in sync with the current state.

Each {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>` provides callbacks for:

- `on_timeline_event(event)` to respond to timeline stop events
- `on_physics_step(step)` to update while the timeline is playing
- `on_stage_event(event)` to respond when a stage is opened
- `cleanup()` to release wrapped UI element callbacks
- `build_ui()` to construct the tutorial window content
- `on_menu_callback()` to update UI state when opened from the menu

## Key Components

### {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>`

{class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>` is the main public class exposed by each example module. It builds the tutorial UI, manages the world and scenario controls, and connects user actions to the corresponding scenario.

All four modules expose a {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>`, but each one targets a different robot motion generation workflow:

```python
from isaacsim.robot_motion.motion_generation.examples.rmp_flow import UIBuilder as RmpFlowUIBuilder
from isaacsim.robot_motion.motion_generation.examples.rrt import UIBuilder as RrtUIBuilder
from isaacsim.robot_motion.motion_generation.examples.kinematics import UIBuilder as KinematicsUIBuilder
from isaacsim.robot_motion.motion_generation.examples.trajectory_generator import UIBuilder as TrajectoryUIBuilder
```

### Wrapped UI Elements

The UI builders track wrapped UI elements through `wrapped_ui_elements`. Buttons imported from `**isaacsim.gui.components.element_wrappers**` implement cleanup behavior, so the UI builder calls cleanup on those elements when needed.

### Frames

Each {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>` maintains UI frames through `frames`. The documented frames include World Controls and Run Scenario sections, which group scene setup actions separately from motion execution actions.

## Relationships

### Lula Motion Generation

The extension package description states that these are examples for using the Lula planning library under motion generation. The tutorials demonstrate Lula-related motion generation workflows through RMPflow, RRT, kinematics, and trajectory generation examples.

### Timeline, Stage, and Physics

The example UIs are designed around simulation state:

- Timeline events control play and stop related UI behavior.
- Physics step callbacks run only while the timeline is playing.
- Stage events reset or update the UI when a stage is opened.

This relationship is visible in each {class}`UIBuilder <isaacsim.robot_motion.motion_generation.examples.rmp_flow.UIBuilder>` through `on_timeline_event`, `on_physics_step`, and `on_stage_event`.

### Example Module Relationship

The four modules share the same public API shape and UI pattern. The main difference is the scenario they connect to and the motion generation concept they demonstrate.

Use the module that matches the workflow you want to explore:

- Use `rmp_flow` for Franka RMPflow motion generation.
- Use `rrt` for Franka RRT path planning.
- Use `kinematics` for Franka kinematics.
- Use `trajectory_generator` for UR10 trajectory generation.
