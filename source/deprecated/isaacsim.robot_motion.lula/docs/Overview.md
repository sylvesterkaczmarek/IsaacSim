# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.robot_motion.experimental.motion_generation` and `isaacsim.robot_motion.cumotion`.
```

`**isaacsim.robot_motion.lula**` provides a Python interface to the Lula robotics motion library. It is used for robot motion generation tasks such as kinematics, inverse kinematics, path specification, trajectory generation, collision checking, motion planning, and RMPflow motion policy evaluation.

The module is centered around robot configuration space, task space poses, and world obstacles. It gives developers a single interface for loading robot descriptions, querying kinematics, planning paths, and generating smooth robot motion.

## Concepts

### Configuration space

Configuration space, or c-space, represents the robot state in joint coordinates. Many APIs in this module accept or return `numpy.ndarray` values whose length matches the number of c-space coordinates for the robot.

Common c-space operations include:

- Reading joint names and limits from `RobotDescription` or `Kinematics`
- Creating c-space paths with `CSpacePathSpec`
- Generating time-parameterized trajectories with `CSpaceTrajectoryGenerator`
- Evaluating robot state along a path or trajectory

### Task space

Task space represents poses in 3D space, usually for an end effector or control frame. The module provides `Pose3` and `Rotation3` for representing rigid transforms and rotations.

Task-space paths can be built from linear motion, translations, rotations, and arc segments. These task-space paths can then be converted into c-space paths using robot kinematics and inverse kinematics settings.

### World model

The module includes a simple world representation for geometric obstacles. `World` stores obstacles, while `WorldView` provides a queryable view for collision checks and distance evaluations.

World data is used by motion planning, RMPflow, and robot-world inspection APIs.

## Functionality

### Robot loading and kinematics

`load_robot()` and `load_robot_from_memory()` create a `RobotDescription` from a robot description YAML and URDF. From there, users can access `Kinematics` to query frame names, c-space coordinate names, limits, poses, positions, orientations, and Jacobians.

The full Lula API is provided by the bundled `lula` module and is accessed with `import lula`. The extension itself only re-exports the logging helpers `LogLevel`, `set_log_level()`, and `set_default_logger_prefix()` under `isaacsim.robot_motion.lula`.

```python
import lula
import numpy as np

robot = lula.load_robot("robot_description.yaml", "robot.urdf")
kinematics = robot.kinematics()

print(robot.num_c_space_coords())
print(kinematics.frame_names())

q = robot.default_c_space_configuration()
ee_position = kinematics.position(q, "end_effector")
```

### Inverse kinematics

The module exposes cyclic coordinate descent inverse kinematics through `compute_ik_ccd()`. Solver behavior is controlled with `CyclicCoordDescentIkConfig`, including tolerances, iteration limits, seed configurations, and relative position or orientation weights.

The result is returned as `CyclicCoordDescentIkResults`, which reports whether a solution was found, the resulting c-space position, and the remaining position and orientation error.

### Path and trajectory generation

Lula separates path definition from trajectory generation.

- `CSpacePathSpec` defines a sequence of c-space waypoints.
- `TaskSpacePathSpec` defines continuous task-space motion.
- `CompositePathSpec` combines c-space and task-space segments.
- `LinearCSpacePath` represents a linear c-space path.
- `Trajectory` represents a time-parameterized c-space path.

`CSpaceTrajectoryGenerator` converts waypoints into smooth trajectories and supports position, velocity, acceleration, and jerk limits.

```python
import lula
import numpy as np

q0 = np.array([0.0, 0.0, 0.0])
q1 = np.array([0.2, 0.4, 0.1])

path_spec = lula.create_c_space_path_spec(q0)
path_spec.add_c_space_waypoint(q1)

path = lula.create_linear_c_space_path(path_spec)

generator = lula.create_c_space_trajectory_generator(path.num_c_space_coords())
trajectory = generator.generate_trajectory([q0, q1])

position, velocity, acceleration, jerk = trajectory.eval_all(trajectory.domain().lower)
```

### Motion planning

`MotionPlanner` provides collision-free path planning for robotic manipulators. It can plan to c-space targets, translation targets, or pose targets, using a `RobotDescription` and `WorldView`.

Planner results are returned through `MotionPlanner.Results`, which includes whether a path was found, the raw path, and an interpolated path.

### RMPflow

`RmpFlow` evaluates an RMPflow motion policy for smooth reactive motion generation. It can use c-space attractors, end-effector position attractors, and end-effector orientation attractors.

`RmpFlowConfig` stores the robot description, RMPflow parameters, end-effector frame, and world view used for obstacle avoidance. Parameters can be queried and updated by name.

```python
import lula
import numpy as np

config = lula.create_rmpflow_config(
    "rmpflow_config.yaml",
    robot,
    "end_effector",
    world_view,
)

rmpflow = lula.create_rmpflow(config)

q = robot.default_c_space_configuration()
qd = np.zeros_like(q)
qdd = np.zeros_like(q)

rmpflow.set_end_effector_position_attractor(np.array([0.4, 0.0, 0.3]))
rmpflow.eval_accel(q, qd, qdd)
```

### Collision and spatial queries

The module provides multiple collision-related tools:

- `World` and `WorldView` for obstacle management and distance checks
- `RobotWorldInspector` for robot-obstacle and self-collision queries
- `CollisionSphereGenerator` for approximating mesh volumes with spheres

`RobotWorldInspector` is the main API for querying collision sphere positions, collision sphere radii, obstacle collision state, and self-collision state.

## Key Components

### `RobotDescription`

`RobotDescription` stores the geometric and kinematic properties of a robot. It provides the default c-space configuration, c-space coordinate names, and access to the robot `Kinematics`.

### `Kinematics`

`Kinematics` evaluates robot frame state from a c-space position. It supports position, orientation, pose, full Jacobian, position Jacobian, and orientation Jacobian queries.

### `Pose3` and `Rotation3`

`Pose3` represents a 3D rigid transform, and `Rotation3` represents a 3D rotation. These types are used throughout task-space path creation, inverse kinematics, and pose-target planning.

### `World` and `WorldView`

`World` manages obstacles. `WorldView` provides the query interface used by planners, RMPflow, and collision inspection.

### {class}`LogLevel <isaacsim.robot_motion.lula.LogLevel>`

{class}`LogLevel <isaacsim.robot_motion.lula.LogLevel>` controls Lula logging verbosity. Levels are ordered from least to most verbose:

- `LogLevel.FATAL`
- `LogLevel.ERROR`
- `LogLevel.WARNING`
- `LogLevel.INFO`
- `LogLevel.VERBOSE`

Use `set_log_level()` to suppress messages above the selected verbosity.

```python
import isaacsim.robot_motion.lula as lula

lula.set_log_level(lula.LogLevel.WARNING)
lula.set_default_logger_prefix("[lula] ")
```

## Usage Examples

### Create a world with an obstacle

```python
import lula
import numpy as np

world = lula.create_world()

obstacle = lula.create_obstacle(lula.Obstacle.Type.SPHERE)
obstacle.set_attribute(lula.Obstacle.Attribute.RADIUS, lula.Obstacle.AttributeValue(0.1))

pose = lula.Pose3.from_translation(np.array([0.5, 0.0, 0.25]))
handle = world.add_obstacle(obstacle, pose)

world_view = world.add_world_view()
print(world_view.num_enabled_obstacles())
```

### Inspect robot collision state

```python
import lula

robot = lula.load_robot("robot_description.yaml", "robot.urdf")
world = lula.create_world()
world_view = world.add_world_view()

inspector = lula.create_robot_world_inspector(robot, world_view)

q = robot.default_c_space_configuration()

print(inspector.num_collision_spheres())
print(inspector.in_self_collision(q))
print(inspector.in_collision_with_obstacle(q))
```

## Relationships

`**isaacsim.robot_motion.lula**` exposes the underlying `lula` Python bindings and uses `numpy` arrays for vector, matrix, c-space, and pose-related data. Robot descriptions are loaded from YAML and URDF inputs, while several path specifications can also be loaded from or exported to YAML strings.
