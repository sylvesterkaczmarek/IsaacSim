# Robot Motion Examples

This extension contains robot-independent examples built on
`isaacsim.robot_motion.experimental.motion_generation`.

The `manipulation` package has three boundaries:

- `robots.py` declares named joint defaults, USD variants, measured links, controller frames,
  and grasp transforms for Franka and UR10.
- `ManipulationScenario` creates a plain `Articulation`, reads measured joints and tool sites
  into `RobotState`, applies desired `RobotState`, and synchronizes the collision world.
- Examples construct RMPFlow, `PickPlaceController`, and gripper controllers directly.

The Omniverse Behavior Tree example shares one cuMotion RMPflow controller per Franka across
its action nodes. Both robot roots and the manipulated cubes are excluded from each controller's
collision world, so inter-robot coordination is provided by the shared center lock rather than
cuMotion collision avoidance. If motion to a moved cube does not converge, recovery retains the
reservation, keeps trying, and logs a warning every five seconds. Moving the cube to a reachable
pose lets the same robot resume the pickup.

`PickPlaceController` captures immutable pick/place goals at reset. Motion phases advance only
after the measured tool pose remains within tolerance; gripper phases also require observable
completion. Active-phase timeouts report failure. Stacking sequences these same pick/place goals.

`ManipulationScenario` accepts an explicit `RobotConfig`, and `PickPlaceController` accepts an
injected arm controller. Callers that assemble these components directly can use an
`RmpFlowController` constructed from a compatible `CumotionRobot`. `PickPlaceTask` uses the
packaged robot configurations.

Interactive samples are registered in **Window > Examples > Robotics Examples**.
