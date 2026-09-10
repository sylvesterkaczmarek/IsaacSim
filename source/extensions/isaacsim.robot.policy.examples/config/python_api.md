# Public API for module isaacsim.robot.policy.examples:

## Classes

- class IsaacLabPolicyController(mg.BaseController)
  - def __init__(self, model: PolicyModel, interface: BoundPolicy)
  - def reset(self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object) -> bool
  - def forward(self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object) -> mg.RobotState
  - def close(self)

- class PolicyArtifact
  - model_path: str
  - env_config_path: str
  - backend: str
  - descriptor_path: str | None
  - model_sha256: str | None
  - def load_descriptor(self) -> Mapping
  - class def from_files(cls, model_path: str, env_config_path: str, descriptor_path: str | None = None) -> PolicyArtifact
  - class def from_bundle(cls, bundle_dir: str) -> PolicyArtifact
  - class def from_training_run(cls, run_dir: str) -> PolicyArtifact

- class PolicyEnvConfig
  - def __init__(self, data: Mapping[str, Any], source_path: str | None = None)
  - class def from_file(cls, path: str) -> PolicyEnvConfig
  - [property] def timing(self) -> PhysicsTiming
  - [property] def spawn(self) -> SpawnProps
  - [property] def initial_root_pose(self) -> tuple[list[float] | None, list[float] | None]
  - def joint_properties(self, joint_names: Sequence[str]) -> JointProperties
  - def actuator_model_specs(self, joint_names: Sequence[str]) -> list[NewtonActuatorSpec]

- class RobotPolicyRunner
  - def __init__(self, spec: PolicySpec)
  - [property] def articulation(self) -> Articulation | None
  - [property] def engine(self) -> str | None
  - [property] def physics_dt(self) -> float
  - [property] def decimation(self) -> int
  - def spawn(self) -> Articulation
  - def initialize(self)
  - def restart_from_default_state(self, command: Sequence[float] | None = None)
  - def step(self, dt: float, command: Sequence[float] | None = None)
  - def close(self)

- class PolicySpec
  - engines: Mapping[str, PolicyArtifact]
  - usd_path: str | Mapping[str, str] | None
  - name: str | None
  - default_spawn_position: tuple[float, float, float] | None
  - default_spawn_orientation: tuple[float, float, float, float] | None
  - zero_targets_on_initialize: bool
  - binding: Callable[[PolicyEnvConfig, tuple[str, ...]], PolicyBinding] | None

## Functions

- def get_anymal_spec() -> PolicySpec
- def get_cartpole_spec() -> PolicySpec
- def get_franka_spec() -> PolicySpec
- def get_go2_spec() -> PolicySpec
- def get_h1_spec() -> PolicySpec
- def get_spot_spec() -> PolicySpec
- def make_franka_task_state_provider(cabinet: Articulation, env_config: PolicyEnvConfig) -> Callable[[Articulation], FrankaTaskStateProvider]

# Public API for module isaacsim.robot.policy.examples.interactive.franka:

## Classes

- class FrankaExample(PolicySampleBase)
  - def __init__(self)
  - def setup_scene(self)
  - async def setup_post_load(self)
  - async def setup_pre_reset(self)
  - async def setup_post_reset(self)
  - def on_physics_step(self, dt: float, context: object)
  - def physics_cleanup(self)

# Public API for module isaacsim.robot.policy.examples.interactive.go2:

## Classes

- class Go2Example(LocomotionPolicySample)

# Public API for module isaacsim.robot.policy.examples.interactive.humanoid:

## Classes

- class HumanoidExample(LocomotionPolicySample)

# Public API for module isaacsim.robot.policy.examples.interactive.quadruped:

## Classes

- class QuadrupedExample(LocomotionPolicySample)
