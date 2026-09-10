# Overview

```{deprecated} 6.0.0
This extension is deprecated. Refer to `isaacsim.robot_motion.examples` for maintained examples.
```

`**isaacsim.robot.manipulators**` provides Python APIs for working with robotic manipulators. Its primary public API is {class}`SingleManipulator <isaacsim.robot.manipulators.SingleManipulator>`, a high-level wrapper for a single-arm articulation with one tracked end effector and an optional gripper.

Use this module when you want one object that represents the manipulator articulation, exposes the end effector as a rigid prim, and coordinates reset and initialization behavior for the manipulator and gripper.

## Concepts

### Single manipulator

A {class}`SingleManipulator <isaacsim.robot.manipulators.SingleManipulator>` represents one articulated robot arm with a single end effector. It inherits from `SingleArticulation`, so it can be used as an articulation while adding manipulator-specific access to the end effector and optional gripper.

The end effector is identified by `end_effector_prim_path`. This path is used to track the rigid body corresponding to the tool or hand at the end of the robot arm.

### End effector

The `end_effector` property returns a `SingleRigidPrim`. This object can be used to query end effector state, such as its world pose, angular velocity, and other rigid body properties.

This keeps end effector access separate from the full articulation, which is useful for manipulation tasks that need to reason about the tool frame directly.

### Optional gripper

A {class}`SingleManipulator <isaacsim.robot.manipulators.SingleManipulator>` can be created with a `Gripper`. When provided, the `gripper` property exposes that object so user code can open or close the gripper and query its pose or velocity.

The gripper is optional, so the same manipulator wrapper can be used for arms with or without an attached grasping component.

## Key Components

### {class}`SingleManipulator <isaacsim.robot.manipulators.SingleManipulator>`

{class}`SingleManipulator <isaacsim.robot.manipulators.SingleManipulator>` is the main public class exported by `**isaacsim.robot.manipulators**`.

It wraps:

- The manipulator articulation.
- A single end effector rigid body.
- An optional gripper.

The constructor accepts placement and visibility options for the manipulator prim, including `position`, `translation`, `orientation`, `scale`, and `visible`.

```python
from isaacsim.robot.manipulators import SingleManipulator

manipulator = SingleManipulator(
    prim_path="/World/Robot",
    end_effector_prim_path="/World/Robot/end_effector",
    name="robot_arm",
)
```

If a gripper object is available, pass it during construction:

```python
from isaacsim.robot.manipulators import SingleManipulator

manipulator = SingleManipulator(
    prim_path="/World/Robot",
    end_effector_prim_path="/World/Robot/end_effector",
    name="robot_arm",
    gripper=my_gripper,
)
```

## Functionality

### Initialization

Call `initialize()` before interacting with the manipulator after a hard reset, such as stopping and playing the timeline. If no physics simulation view is passed, the method creates one and creates the articulation view needed by the physics tensor API.

```python
manipulator.initialize()
```

You can also pass an existing simulation view:

```python
manipulator.initialize(physics_sim_view=simulation_view)
```

### Accessing the end effector

Use `end_effector` to work directly with the rigid body at the end of the manipulator.

```python
end_effector = manipulator.end_effector

position, orientation = end_effector.get_world_pose()
```

This is useful for tasks such as tracking tool pose, computing grasp targets, or checking the current state of the manipulator tip.

### Accessing the gripper

Use `gripper` to access the optional gripper object associated with the manipulator.

```python
gripper = manipulator.gripper

if gripper is not None:
    gripper.open()
```

The exact gripper operations depend on the gripper object supplied to the manipulator.

### Reset behavior

`post_reset()` resets the manipulator, end effector, and gripper to their default states.

```python
manipulator.post_reset()
```

This is intended for restoring the manipulator-related objects after simulation reset behavior.

## Relationships

{class}`SingleManipulator <isaacsim.robot.manipulators.SingleManipulator>` builds on `SingleArticulation` from `**isaacsim.core.prims**`, which provides the articulation-level behavior for the robot. Its `end_effector` property exposes a `SingleRigidPrim`, also from `**isaacsim.core.prims**`, for rigid body access to the end effector.

The optional `gripper` argument and property use the `Gripper` interface from `**isaacsim.robot.manipulators.grippers.gripper**`, allowing gripper behavior to be associated with the manipulator without requiring every manipulator to have one.
