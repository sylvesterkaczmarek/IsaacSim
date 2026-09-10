# Overview

```{deprecated} 6.0.0
This extension is deprecated. Refer to `isaacsim.robot_motion.examples` for maintained examples.
```

`**isaacsim.robot.manipulators.ui**` provides UI menus for creating OmniGraph-based robot controller graphs. It focuses on common manipulator workflows: joint position control, joint velocity control, and gripper control. Users access these tools from the `Tools/Robotics/OmniGraph Controllers` menu and use the windows to generate the required controller nodes and connections.

## Functionality

The extension creates controller setup windows that build OmniGraph networks for robot manipulators.

- **Articulation position control**: Creates a graph for commanding joint positions on an articulation.
- **Articulation velocity control**: Creates a graph for commanding joint velocities on an articulation.
- **Gripper control**: Creates a graph for controlling gripper joints with open, close, and stop behavior.

For articulation controllers, the windows can discover joints from the selected robot prim and create the supporting joint command and joint name arrays. The generated graphs include an articulation controller node configured for the selected control mode.

The controller creation process stops the simulation timeline before modifying or creating the graph. After the graph is generated, users can edit command values in the generated nodes, such as through the Property Manager, and run simulation to apply the control behavior.

## UI Components

### OmniGraph Controllers Menu

The extension adds entries under:

`Tools/Robotics/OmniGraph Controllers`

The menu provides access to three controller windows:

- `Articulation Position Controller`
- `Articulation Velocity Controller`
- `Gripper Controller`

### Articulation Position Window

The `ArticulationPositionWindow` creates an OmniGraph for position-based joint control.

Users can specify the robot prim and choose whether to create a new graph or add controller nodes to an existing graph. The generated graph includes joint command arrays, joint name arrays, and an articulation controller configured for position control.

The window supports revolute and prismatic joints. Revolute joint values account for the unit difference between USD degree-based properties and PhysX radian-based control.

### Articulation Velocity Window

The `ArticulationVelocityWindow` creates an OmniGraph for velocity-based joint control.

Users select the robot prim, graph path, and whether to use an existing graph or create a new one. The generated graph includes joint command arrays, joint name arrays, and an articulation controller configured for velocity control.

Default velocity values are read from joint drive APIs when available. Revolute joint velocity control uses radians, while USD properties may display values in degrees.

### Gripper Window

The `GripperWindow` creates an OmniGraph configuration for gripper control.

The window includes fields for the parent robot prim, gripper root prim, graph path, joint names, open and close positions, and movement speed. Users can define which articulated joints belong to the gripper when the full articulation contains additional non-gripper joints.

The gripper window can also add optional keyboard controls:

- `O`: open
- `C`: close
- `N`: stop

## Workflow

A typical workflow is:

1. Open one of the controller windows from `Tools/Robotics/OmniGraph Controllers`.
2. Select the robot or gripper prim to control.
3. Choose whether to create a new OmniGraph or add nodes to an existing graph.
4. Configure graph path and controller-specific parameters.
5. Create the graph.
6. Run simulation and adjust generated node values to control the robot.

For articulation position and velocity controllers, the generated graph is centered around joint command arrays and an articulation controller. For grippers, the generated graph adds gripper-specific command values, speed settings, and optional keyboard input nodes.

## Relationships

This extension uses `**omni.kit.menu.utils**` through `MenuHelperExtensionFull` and `MenuHelperWindow` to create the Robotics menu entries and controller windows.

The generated graphs are designed for robot articulation and gripper workflows provided by the manipulator stack. They operate on robot prims in the stage and create OmniGraph controller networks for position, velocity, or gripper control.
