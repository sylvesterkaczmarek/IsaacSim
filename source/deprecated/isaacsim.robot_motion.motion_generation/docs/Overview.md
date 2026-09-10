# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.robot_motion.experimental.motion_generation` and `isaacsim.robot_motion.cumotion`.
```

`**isaacsim.robot_motion.motion_generation**` provides interfaces and Lula-based implementations for generating robot motion. It covers common motion-generation workflows such as reactive motion policies, forward and inverse kinematics, path planning, and time-parameterized trajectories. The module also includes wrapper classes that convert solver outputs into `ArticulationAction` objects that can be applied to simulated robot articulations.

The extension is useful when you want to control a robot from task-space or configuration-space targets, account for obstacles, or convert planned paths and trajectories into simulation actions.

<div align="center">

```{mermaid}
graph TD
    %% Inheritance relationships
    WorldInterface --> PathPlanner
    WorldInterface --> MotionPolicy
    WorldInterface --> KinematicsSolver
    KinematicsSolver --> LulaKinematicsSolver
    MotionPolicy --> RmpFlow

    %% Composition relationships (uses)
    ArticulationMotionPolicy -.->|uses| MotionPolicy
    ArticulationKinematicsSolver -.->|uses| KinematicsSolver
    ArticulationTrajectory -.->|uses| Trajectory
    MotionPolicyController -.->|uses| ArticulationMotionPolicy
    MotionPolicyController -.->|uses| MotionPolicy
```

</div>

## Concepts

### World-aware motion

{class}`WorldInterface <isaacsim.robot_motion.motion_generation.WorldInterface>` defines how motion-generation algorithms receive obstacle information from the scene. It supports adding, removing, enabling, disabling, and updating obstacle objects such as cuboids, spheres, capsules, cylinders, cones, and ground planes.

Algorithms that inherit from {class}`WorldInterface <isaacsim.robot_motion.motion_generation.WorldInterface>`, such as {class}`MotionPolicy <isaacsim.robot_motion.motion_generation.MotionPolicy>`, {class}`PathPlanner <isaacsim.robot_motion.motion_generation.PathPlanner>`, and {class}`KinematicsSolver <isaacsim.robot_motion.motion_generation.KinematicsSolver>`, can use the same obstacle-management pattern when they support collision-aware behavior.

### Active and watched joints

Several APIs distinguish between active joints and watched joints.

- Active joints are directly controlled by a solver, policy, planner, or trajectory.
- Watched joints affect the robot state but are not directly commanded by that component.

The order returned by `get_active_joints()` and `get_watched_joints()` is important. Input arrays passed to APIs such as `compute_joint_targets()` or `compute_path()` must follow the same order.

### Targets and base pose

Motion policies and path planners can accept either configuration-space targets or end-effector targets.

- `set_cspace_target()` sets desired active joint positions.
- `set_end_effector_target()` sets a task-space target position and optional orientation.
- `set_robot_base_pose()` updates the robot base pose relative to the USD stage origin.

The target translation values use the same units as the stage unless a specific API states otherwise.

### Frame-based motion and continuous trajectories

The module supports two motion styles:

- {class}`MotionPolicy <isaacsim.robot_motion.motion_generation.MotionPolicy>` computes joint targets for the next simulation frame.
- {class}`Trajectory <isaacsim.robot_motion.motion_generation.Trajectory>` represents continuous-time joint targets between a start time and an end time.

Wrapper classes such as {class}`ArticulationMotionPolicy <isaacsim.robot_motion.motion_generation.ArticulationMotionPolicy>` and {class}`ArticulationTrajectory <isaacsim.robot_motion.motion_generation.ArticulationTrajectory>` convert these outputs into `ArticulationAction` objects for use with a robot articulation.

## Functionality

### Reactive motion policies

{class}`MotionPolicy <isaacsim.robot_motion.motion_generation.MotionPolicy>` is the base interface for collision-aware, frame-by-frame robot motion. Implementations compute joint position and velocity targets from the current robot state, target state, obstacle state, and physics timestep.

{class}`RmpFlow <isaacsim.robot_motion.motion_generation.RmpFlow>` is a Lula-based {class}`MotionPolicy <isaacsim.robot_motion.motion_generation.MotionPolicy>` implementation. It provides reactive task-space motion toward an end-effector target while supporting obstacle avoidance through the {class}`WorldInterface <isaacsim.robot_motion.motion_generation.WorldInterface>` APIs.

### Kinematics

{class}`KinematicsSolver <isaacsim.robot_motion.motion_generation.KinematicsSolver>` defines a compact interface for forward and inverse kinematics. It supports querying joint names, querying available frame names, computing a frame pose from joint positions, and solving joint positions for a desired frame pose.

{class}`LulaKinematicsSolver <isaacsim.robot_motion.motion_generation.LulaKinematicsSolver>` implements this interface using a robot description YAML file and a URDF file. It also exposes solver parameters for IK behavior, including CCD and BFGS-related parameters, default tolerances, c-space seeds, and joint limits.

### Path planning

{class}`PathPlanner <isaacsim.robot_motion.motion_generation.PathPlanner>` defines an interface for algorithms that compute configuration-space waypoints. These waypoints are intended to be linearly interpolated into a path from the current robot state to a configuration-space or task-space target.

A path planner can use both active joint positions and watched joint positions when computing a path.

### {class}`Trajectory <isaacsim.robot_motion.motion_generation.Trajectory>` generation

{class}`Trajectory <isaacsim.robot_motion.motion_generation.Trajectory>` defines continuous-time joint targets. A trajectory returns joint position and velocity targets at a requested time.

{class}`LulaCSpaceTrajectoryGenerator <isaacsim.robot_motion.motion_generation.LulaCSpaceTrajectoryGenerator>` creates time-parameterized trajectories from configuration-space waypoints. It supports time-optimal trajectory generation using velocity, acceleration, and jerk limits, as well as timestamped trajectory generation with cubic spline or linear interpolation.

{class}`LulaTaskSpaceTrajectoryGenerator <isaacsim.robot_motion.motion_generation.LulaTaskSpaceTrajectoryGenerator>` generates trajectories from task-space paths. It converts task-space waypoints or path specifications into configuration-space trajectories, then applies trajectory generation over the active joints.

## Key Components

### {class}`WorldInterface <isaacsim.robot_motion.motion_generation.WorldInterface>`

{class}`WorldInterface <isaacsim.robot_motion.motion_generation.WorldInterface>` standardizes obstacle management for world-aware algorithms. It supports common object types from `**isaacsim.core.api.objects**`, including cuboids, spheres, capsules, cylinders, cones, and ground planes.

Use it when an algorithm needs to maintain an internal world representation for collision avoidance or world-relative computation.

### {class}`MotionPolicy <isaacsim.robot_motion.motion_generation.MotionPolicy>` and {class}`RmpFlow <isaacsim.robot_motion.motion_generation.RmpFlow>`

{class}`MotionPolicy <isaacsim.robot_motion.motion_generation.MotionPolicy>` is the base interface for frame-by-frame motion generation. Its main method is `compute_joint_targets()`, which returns joint position and velocity targets for the next physics frame.

{class}`RmpFlow <isaacsim.robot_motion.motion_generation.RmpFlow>` is a Lula-based implementation of {class}`MotionPolicy <isaacsim.robot_motion.motion_generation.MotionPolicy>`. It supports task-space and c-space targets, obstacle updates, robot base pose updates, and debugging helpers for visualizing internal collision spheres and the believed end-effector pose.

### {class}`ArticulationMotionPolicy <isaacsim.robot_motion.motion_generation.ArticulationMotionPolicy>`

{class}`ArticulationMotionPolicy <isaacsim.robot_motion.motion_generation.ArticulationMotionPolicy>` connects a {class}`MotionPolicy <isaacsim.robot_motion.motion_generation.MotionPolicy>` to a simulated robot articulation. It reads joint state from a `SingleArticulation`, calls the underlying policy, and returns or applies an `ArticulationAction`.

Use `get_next_articulation_action()` when you want to compute the next action yourself. Use `move()` when you want the wrapper to compute and apply the action for the current frame.

### {class}`KinematicsSolver <isaacsim.robot_motion.motion_generation.KinematicsSolver>` and {class}`LulaKinematicsSolver <isaacsim.robot_motion.motion_generation.LulaKinematicsSolver>`

{class}`KinematicsSolver <isaacsim.robot_motion.motion_generation.KinematicsSolver>` defines FK and IK operations for a robot model. {class}`LulaKinematicsSolver <isaacsim.robot_motion.motion_generation.LulaKinematicsSolver>` provides a Lula-based implementation initialized from robot description and URDF files.

{class}`LulaKinematicsSolver <isaacsim.robot_motion.motion_generation.LulaKinematicsSolver>` can compute forward kinematics for any known frame and inverse kinematics for a desired target pose. It also provides access to default position, velocity, acceleration, and jerk limits from the robot description.

### {class}`ArticulationKinematicsSolver <isaacsim.robot_motion.motion_generation.ArticulationKinematicsSolver>`

{class}`ArticulationKinematicsSolver <isaacsim.robot_motion.motion_generation.ArticulationKinematicsSolver>` connects a {class}`KinematicsSolver <isaacsim.robot_motion.motion_generation.KinematicsSolver>` to a simulated robot articulation. It uses the robot current joint state as input and returns results in forms that are directly useful for simulation.

For IK, it returns an `ArticulationAction` and a success flag.

### {class}`PathPlanner <isaacsim.robot_motion.motion_generation.PathPlanner>`

{class}`PathPlanner <isaacsim.robot_motion.motion_generation.PathPlanner>` is the interface for configuration-space waypoint planners. It supports setting a robot base pose, setting either c-space or end-effector targets, and computing a waypoint path from active and watched joint positions.

### {class}`Trajectory <isaacsim.robot_motion.motion_generation.Trajectory>` and {class}`ArticulationTrajectory <isaacsim.robot_motion.motion_generation.ArticulationTrajectory>`

{class}`Trajectory <isaacsim.robot_motion.motion_generation.Trajectory>` represents continuous-time joint targets. {class}`ArticulationTrajectory <isaacsim.robot_motion.motion_generation.ArticulationTrajectory>` samples a {class}`Trajectory <isaacsim.robot_motion.motion_generation.Trajectory>` and converts each sample into an `ArticulationAction`.

Use `get_action_at_time()` to sample one point, or `get_action_sequence()` to discretize the full trajectory.

### {class}`LulaCSpaceTrajectoryGenerator <isaacsim.robot_motion.motion_generation.LulaCSpaceTrajectoryGenerator>` and {class}`LulaTaskSpaceTrajectoryGenerator <isaacsim.robot_motion.motion_generation.LulaTaskSpaceTrajectoryGenerator>`

{class}`LulaCSpaceTrajectoryGenerator <isaacsim.robot_motion.motion_generation.LulaCSpaceTrajectoryGenerator>` creates trajectories from c-space waypoints and exposes c-space limit setters for position, velocity, acceleration, and jerk limits.

{class}`LulaTaskSpaceTrajectoryGenerator <isaacsim.robot_motion.motion_generation.LulaTaskSpaceTrajectoryGenerator>` creates trajectories from task-space points or Lula task-space path specifications. It also exposes a path conversion configuration object through `get_path_conversion_config()`.

### {class}`MotionPolicyController <isaacsim.robot_motion.motion_generation.MotionPolicyController>`

{class}`MotionPolicyController <isaacsim.robot_motion.motion_generation.MotionPolicyController>` wraps an {class}`ArticulationMotionPolicy <isaacsim.robot_motion.motion_generation.ArticulationMotionPolicy>` in a controller-style interface. Its `forward()` method accepts an end-effector target and returns an `ArticulationAction` for the next simulation frame.

It also forwards obstacle add and remove calls to the underlying motion policy.

## Usage Examples

### Computing frame-by-frame actions with {class}`RmpFlow <isaacsim.robot_motion.motion_generation.RmpFlow>`

```python
import numpy as np
from isaacsim.robot_motion.motion_generation import RmpFlow, ArticulationMotionPolicy

robot_articulation = ...  # Initialized SingleArticulation

rmpflow = RmpFlow(
    robot_description_path="path/to/robot_description.yaml",
    urdf_path="path/to/robot.urdf",
    rmpflow_config_path="path/to/rmpflow_config.yaml",
    end_effector_frame_name="tool_frame",
    maximum_substep_size=1.0 / 300.0,
)

articulation_policy = ArticulationMotionPolicy(
    robot_articulation=robot_articulation,
    motion_policy=rmpflow,
    default_physics_dt=1.0 / 60.0,
)

rmpflow.set_end_effector_target(
    target_position=np.array([0.4, 0.0, 0.3]),
    target_orientation=None,
)

action = articulation_policy.get_next_articulation_action()
robot_articulation.apply_action(action)
```

### Computing inverse kinematics for an articulation

```python
import numpy as np
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver

robot_articulation = ...  # Initialized SingleArticulation

kinematics = LulaKinematicsSolver(
    robot_description_path="path/to/robot_description.yaml",
    urdf_path="path/to/robot.urdf",
)

articulation_kinematics = ArticulationKinematicsSolver(
    robot_articulation=robot_articulation,
    kinematics_solver=kinematics,
    end_effector_frame_name="tool_frame",
)

target_position = np.array([0.4, 0.1, 0.3])
action, success = articulation_kinematics.compute_inverse_kinematics(target_position)

if success:
    robot_articulation.apply_action(action)
```

### Converting a c-space trajectory into articulation actions

```python
import numpy as np
from isaacsim.robot_motion.motion_generation import LulaCSpaceTrajectoryGenerator, ArticulationTrajectory

robot_articulation = ...  # Initialized SingleArticulation

trajectory_generator = LulaCSpaceTrajectoryGenerator(
    robot_description_path="path/to/robot_description.yaml",
    urdf_path="path/to/robot.urdf",
)

waypoints = np.array([
    [0.0, -0.5, 0.2, -1.0, 0.0, 1.0],
    [0.2, -0.3, 0.4, -0.8, 0.1, 0.8],
    [0.4, -0.1, 0.6, -0.6, 0.2, 0.6],
])

trajectory = trajectory_generator.compute_c_space_trajectory(waypoints)

if trajectory is not None:
    articulation_trajectory = ArticulationTrajectory(
        robot_articulation=robot_articulation,
        trajectory=trajectory,
        physics_dt=1.0 / 60.0,
    )

    actions = articulation_trajectory.get_action_sequence()
```

## Relationships

{class}`ArticulationMotionPolicy <isaacsim.robot_motion.motion_generation.ArticulationMotionPolicy>`, {class}`ArticulationKinematicsSolver <isaacsim.robot_motion.motion_generation.ArticulationKinematicsSolver>`, and {class}`ArticulationTrajectory <isaacsim.robot_motion.motion_generation.ArticulationTrajectory>` are bridge classes between motion-generation algorithms and simulated robot articulations. They operate on `SingleArticulation` objects and produce `ArticulationAction` outputs.

{class}`WorldInterface <isaacsim.robot_motion.motion_generation.WorldInterface>` works with object wrappers from `**isaacsim.core.api.objects**` for obstacle management. Lula-based classes use robot description YAML files and URDF files to define the robot model, active joints, limits, frames, and solver behavior.

## Considerations

- `LulaKinematicsSolver.supports_collision_avoidance()` returns `False`; Lula IK does not avoid USD obstacles through the {class}`WorldInterface <isaacsim.robot_motion.motion_generation.WorldInterface>` obstacle APIs.
- `RmpFlow.get_watched_joints()` returns an empty list. {class}`RmpFlow <isaacsim.robot_motion.motion_generation.RmpFlow>` does not currently support watched joints that are not actively controlled.
- If a robot arm is mounted on an externally controlled body, update the base pose with `set_robot_base_pose()`.
- Debug visualization methods such as `visualize_collision_spheres()` can significantly reduce simulation framerate.
- `ArticulationTrajectory.get_action_at_time()` raises `ValueError` if the requested time is outside the trajectory bounds.
