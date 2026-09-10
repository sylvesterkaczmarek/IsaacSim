# Overview

```{deprecated} 6.0.0
This extension is deprecated. Use `isaacsim.robot_motion.examples` instead.
```

`**isaacsim.robot.manipulators.examples**` provides example implementations for robot manipulator workflows using Franka Panda and UR10 robots. It includes robot wrappers, task definitions, motion controllers, gripper setups, and kinematics helpers that demonstrate common manipulator behaviors such as follow-target, pick-and-place, stacking, and bin filling.

For new work, use `**isaacsim.robot_motion.examples**`. This extension is retained for existing Franka and UR10 manipulator examples that depend on the non-experimental module path.

## Concepts

### Robot examples

The extension centers on two robot families:

- `Franka`: A Franka Panda robot arm with a `ParallelGripper`.
- `UR10`: A Universal Robots UR10 arm with optional `SurfaceGripper` support.

Each robot example wraps a robot articulation, exposes its end effector, and configures the relevant gripper behavior during initialization and reset.

### Task examples

The task classes specialize common task templates from `**isaacsim.core.api.tasks**` for specific robot examples. They create the robot, configure task-specific objects, and provide observations needed by controllers.

Supported task patterns include:

- Follow a target prim.
- Pick an object and place it at a target position.
- Stack cubes in a target location.
- Fill a bin with cubes using a UR10 and surface gripper behavior.

### Controller examples

The controller classes adapt manipulator controllers for the robot and gripper used in each example. They produce `ArticulationAction` outputs that can be applied to the robot articulation during simulation.

## Functionality

### Franka examples

The Franka examples demonstrate a Franka Panda arm configured with a parallel gripper.

Key behaviors include:

- Creating a Franka robot at a USD prim path.
- Accessing the robot end effector as a `SingleRigidPrim`.
- Accessing the configured `ParallelGripper`.
- Running follow-target, pick-and-place, and stacking tasks.
- Using Franka-specific kinematics and RMPflow motion control.

The Franka task classes are useful as reference implementations for workflows where the gripper opens and closes around objects.

### UR10 examples

The UR10 examples demonstrate a UR10 robot arm with optional surface gripper support.

Key behaviors include:

- Creating a UR10 robot at a USD prim path.
- Optionally attaching a surface gripper.
- Accessing the robot end effector and gripper.
- Running follow-target, pick-and-place, stacking, and bin-filling tasks.
- Using UR10-specific inverse kinematics, articulation kinematics, and RMPflow motion control.

The UR10 pick-and-place and stacking controllers include `forward()` methods that compute one controller step from task inputs such as picking position, placing position, current joint positions, and optional end effector orientation or offset.

### Bin filling

`BinFilling` is a UR10 task that loads a stage, adds the robot and packing bin to the scene, and drops cubes from a pipe on scheduled simulation steps. It exposes observations for the packing bin and UR10 robot, including positions, orientations, joint positions, and end effector state.

It also provides task utilities to:

- Query how many cubes remain to be added.
- Add a requested number of cubes.
- Reset cube drop counters.
- Hide and deactivate spawned cubes during cleanup.
- Return task parameters such as `bin_name` and `robot_name`.

## Key Components

### Robot wrappers

`Franka` and `UR10` are robot-specific wrappers built on `**isaacsim.core.api.robots.robot.Robot**`.

They define the robot setup used by the examples, including:

- USD prim path and optional USD asset path.
- Initial position and orientation.
- End effector prim selection.
- Gripper configuration.
- Initialization and post-reset behavior.

`Franka` uses `ParallelGripper`, while `UR10` can use `SurfaceGripper` when gripper attachment is requested.

### Kinematics solvers

The extension includes robot-specific kinematics helpers:

- Franka `KinematicsSolver`
- UR10 `KinematicsSolver`
- UR10 `InverseKinematicsSolver`

These classes connect initialized robot articulations to motion-generation kinematics solvers. The UR10 variants also account for whether a gripper is attached.

### Motion controllers

The RMPflow controller examples specialize `MotionPolicyController` for each robot:

- Franka `RMPFlowController`
- UR10 `RMPFlowController`

They configure RMPflow-based motion control and reset the controller state, including restoring the robot base pose to its default position and orientation.

### Manipulation controllers

The extension includes robot-specific pick-and-place and stacking controllers:

- Franka `PickPlaceController`
- Franka `StackingController`
- UR10 `PickPlaceController`
- UR10 `StackingController`

These examples show how manipulator task controllers are paired with the appropriate gripper type and robot articulation.

## Workflows

### Follow target

The follow-target examples create a robot and a target object, then drive the robot toward the target pose. Franka and UR10 each provide a `FollowTarget` task that specializes the base follow-target task and creates the matching robot in `set_robot()`.

### Pick and place

The pick-and-place examples define a cube, a target placement position, and a robot controller. The controller advances through pick-and-place phases and returns articulation actions for the robot.

For UR10, the controller accepts optional end effector orientation and offset values during `forward()`, which allows the task to adjust how the surface gripper approaches the object.

### Stacking

The stacking examples build on the base stacking task and configure the robot-specific controller and gripper. The controller uses an ordered list of cube names to decide the picking sequence and produces actions for stacking cubes at the target position.

### Bin filling

The UR10 bin-filling example focuses on surface gripper behavior and cube dropping. It adds cubes to a queue, drops them during `pre_step()`, and reports observations for both the packing bin and the robot so a behavior layer can react during simulation.

## Relationships

This extension uses `**isaacsim.core.api**` for robot, scene, and task primitives such as `Robot`, `BaseTask`, `FollowTarget`, `PickPlace`, `Stacking`, and `Scene`.

It uses `**isaacsim.robot.manipulators**` for manipulator-specific controllers and grippers, including `PickPlaceController`, `StackingController`, `ParallelGripper`, and `SurfaceGripper`.

It uses `**isaacsim.robot_motion.motion_generation**` for motion-generation components such as `MotionPolicyController`, `ArticulationKinematicsSolver`, and inverse kinematics solver support.
