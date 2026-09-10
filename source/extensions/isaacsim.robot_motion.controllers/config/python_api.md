# Public API for module isaacsim.robot_motion.controllers:

## Classes

- class AckermannController(BaseController)
  - def __init__(self)
  - def reset(self, estimated_state: RobotState, setpoint_state: RobotState | None, t: float, **kwargs: object) -> bool
  - def forward(self, estimated_state: RobotState, setpoint_state: RobotState | None, t: float, **kwargs: object) -> RobotState | None

- class DifferentialDriveController(BaseController)
  - def __init__(self)
  - def reset(self, estimated_state: RobotState, setpoint_state: RobotState | None, t: float, **kwargs: object) -> bool
  - def forward(self, estimated_state: RobotState, setpoint_state: RobotState | None, t: float, **kwargs: object) -> RobotState | None

- class HolonomicController(BaseController)
  - def __init__(self)
  - def reset(self, estimated_state: RobotState, setpoint_state: RobotState | None, t: float, **kwargs: object) -> bool
  - def forward(self, estimated_state: RobotState, setpoint_state: RobotState | None, t: float, **kwargs: object) -> RobotState | None
