# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.robot.experimental.wheeled_robots` and `isaacsim.robot.wheeled_robots.nodes`.
```

The `isaacsim.robot.wheeled_robots` extension provides Python utilities for simulating and controlling wheeled robots in Isaac Sim. It supplies robot wrappers, motion controllers, and path-planning helpers for differential-drive, Ackermann-steered, and holonomic platforms, together with a small Carbonite plugin that exposes the wheeled-robots C++ interface.

Use this extension when you need to drive a wheeled robot articulation from Python, convert a desired body motion into per-wheel commands, or follow a planned path.

## Key Components

### Robot wrappers

The `isaacsim.robot.wheeled_robots.robots` package provides robot-level helpers:

- {class}`WheeledRobot <isaacsim.robot.wheeled_robots.robots.WheeledRobot>`: Wraps and manages the articulation for a wheeled robot base, exposing its wheel joints and an API for applying wheel actions and querying degree-of-freedom parameters.
- {class}`HolonomicRobotUsdSetup <isaacsim.robot.wheeled_robots.robots.HolonomicRobotUsdSetup>`: Reads the USD setup of a holonomic (for example, mecanum) robot to derive the parameters needed by the holonomic controller.

### Controllers

The `isaacsim.robot.wheeled_robots.controllers` package provides controllers that convert a desired body motion into per-wheel commands:

- {class}`DifferentialController <isaacsim.robot.wheeled_robots.controllers.DifferentialController>`: Differential-drive control from linear and angular velocity commands.
- {class}`AckermannController <isaacsim.robot.wheeled_robots.controllers.AckermannController>`: Ackermann steering control for car-like platforms.
- {class}`HolonomicController <isaacsim.robot.wheeled_robots.controllers.HolonomicController>`: Holonomic control for omnidirectional (for example, mecanum) drives.
- {class}`WheelBasePoseController <isaacsim.robot.wheeled_robots.controllers.WheelBasePoseController>`: Drives the robot base toward a target pose using an underlying wheel controller.

### Path planning

The extension also includes path-planning and path-following helpers, including a quintic polynomial trajectory planner ({func}`quintic_polynomials_planner <isaacsim.robot.wheeled_robots.controllers.quintic_polynomials_planner>`) and a Stanley path-tracking controller ({func}`stanley_control <isaacsim.robot.wheeled_robots.controllers.stanley_control>`).

## C++ Interface

The extension ships a Carbonite plugin that registers the `IWheeledRobots` interface. The interface is exposed to Python through the `isaacsim.robot.wheeled_robots.bindings` module and is acquired automatically when the extension loads; most users interact only with the Python controllers and robot wrappers above.

Node-based (OmniGraph) wheeled-robot control is provided by the separate `isaacsim.robot.wheeled_robots.nodes` extension.

## Relationships

The robot wrappers build on `isaacsim.core.api` robot primitives, and the controllers rely on `omni.physx` for the underlying articulation simulation.