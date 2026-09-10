# Overview

The isaacsim.robot_setup.gain_tuner extension provides the core library for tuning PD (Proportional-Derivative) gains on robot articulations and running validation tests. Use it programmatically to inspect joint drives, compute effective inertia, execute sinusoidal/step/snap/stress tests, and collect USD layer edits for saving tuned gains.

```{image} ../../../../source/extensions/isaacsim.robot_setup.gain_tuner/data/preview.png
---
align: center
---
```

The interactive Gain Tuner window lives in the companion extension `isaacsim.robot_setup.gain_tuner.ui`.

## Key Components

### {class}`GainTuner <isaacsim.robot_setup.gain_tuner.GainTuner>`

The {class}`GainTuner <isaacsim.robot_setup.gain_tuner.GainTuner>` class manages articulation setup, joint discovery, accumulated inertia computation, registered validation tests, and test execution during simulation.

```python
from isaacsim.robot_setup.gain_tuner import GainTuner, GainsTestMode, SinusoidalTest

tuner = GainTuner()
tuner.setup("/World/MyRobot")
tuner.register_test(GainsTestMode.SINUSOIDAL, SinusoidalTest())
tuner.initialize_gains_test({"duration": 2.0, "frequency": 1.0})
```

### Drive helpers and USD save utilities

- {mod}`isaacsim.robot_setup.gain_tuner.joint_drive_attrs` — query stiffness, damping, and drive mode from USD joint prims.
- {mod}`isaacsim.robot_setup.gain_tuner.usd_layer_utils` — resolve physics layers and collect drive attribute edits for saving.

### Multi-backend gain sources

The {mod}`isaacsim.robot_setup.gain_tuner.gain_sources` module resolves and edits joint
gains across the three backends a robot may author them in: PhysX ``PhysicsDrive`` gains,
Newton ``NewtonActuator`` PD/PID controllers, and MuJoCo actuator (``mjc:*``) parameters.
It is engine-aware — for example, when the Newton MuJoCo solver is active
({func}`newton_mujoco_solver_active <isaacsim.robot_setup.gain_tuner.newton_mujoco_solver_active>`)
the MuJoCo gains take precedence.

The {class}`GainSource <isaacsim.robot_setup.gain_tuner.GainSource>` enum names each
backend. {func}`resolve_joint_gains <isaacsim.robot_setup.gain_tuner.resolve_joint_gains>`
reads a joint's effective stiffness/damping and their defining USD attributes into a
{class}`ResolvedGains <isaacsim.robot_setup.gain_tuner.ResolvedGains>`, while
{func}`active_gain_source <isaacsim.robot_setup.gain_tuner.active_gain_source>` and
{func}`available_viewed_sources <isaacsim.robot_setup.gain_tuner.available_viewed_sources>`
report which backends drive a joint and which may be viewed. Backend maps are built with
{func}`build_actuator_gain_map <isaacsim.robot_setup.gain_tuner.build_actuator_gain_map>`
(Newton) and {func}`build_mjc_gain_map <isaacsim.robot_setup.gain_tuner.build_mjc_gain_map>`
(MuJoCo), and {func}`drive_gains_to_mjc <isaacsim.robot_setup.gain_tuner.drive_gains_to_mjc>` /
{func}`mjc_params_to_drive_gains <isaacsim.robot_setup.gain_tuner.mjc_params_to_drive_gains>`
convert between PD gains and MuJoCo parameters.

Edits are written back through a layer-aware save pipeline:
{func}`list_gain_save_target_layers <isaacsim.robot_setup.gain_tuner.list_gain_save_target_layers>`
enumerates candidate physics layers,
{func}`build_gain_save_plan <isaacsim.robot_setup.gain_tuner.build_gain_save_plan>` collects
the attribute edits, and
{func}`apply_gain_save_plan <isaacsim.robot_setup.gain_tuner.apply_gain_save_plan>` authors
them into the chosen layer.

### Per-backend advanced joint parameters

Armature, joint friction, and the joint velocity limit are authored on separate joint
API schemas per backend, and each backend reads its own. The
{mod}`isaacsim.robot_setup.gain_tuner.joint_schema_attrs` module resolves and edits them
without mixing the two backends' tuning.

{func}`resolve_joint_param <isaacsim.robot_setup.gain_tuner.resolve_joint_param>` (and
{func}`resolve_joint_params <isaacsim.robot_setup.gain_tuner.resolve_joint_params>` for a
whole joint) returns a
{class}`JointParamResolution <isaacsim.robot_setup.gain_tuner.JointParamResolution>` naming
every schema's authored value, the schema the active backend resolves, the effective value,
and which authored values are shadowed or never read. Reads follow the schema-resolver order
Newton itself uses, which depends on the running solver — exposed by
{func}`resolver_chain <isaacsim.robot_setup.gain_tuner.resolver_chain>`,
{func}`param_resolver_chain <isaacsim.robot_setup.gain_tuner.param_resolver_chain>`, and
{func}`candidate_param_chains <isaacsim.robot_setup.gain_tuner.candidate_param_chains>`,
which report the chain as undetermined rather than guessing when the solver
({func}`newton_solver_type <isaacsim.robot_setup.gain_tuner.newton_solver_type>`) is unknown.

{func}`author_joint_param <isaacsim.robot_setup.gain_tuner.author_joint_param>` writes only
the schema named by
{func}`backend_write_schema <isaacsim.robot_setup.gain_tuner.backend_write_schema>`, which
returns None for a backend whose joint schema is unknown — check
{func}`backend_supported <isaacsim.robot_setup.gain_tuner.backend_supported>` and disable the
edit rather than guessing. A difference between the two backends is reported as fact by
{func}`diverging_joint_params <isaacsim.robot_setup.gain_tuner.diverging_joint_params>`, and
{func}`copy_joint_param_to_backend <isaacsim.robot_setup.gain_tuner.copy_joint_param_to_backend>`
copies one backend's value onto the other as an explicit action.

For an unauthored parameter,
{meth}`JointParamSpec.engine_default <isaacsim.robot_setup.gain_tuner.JointParamSpec.engine_default>`
gives the value the engine simulates anyway (Newton's armature default is
{data}`NEWTON_DEFAULT_ARMATURE <isaacsim.robot_setup.gain_tuner.NEWTON_DEFAULT_ARMATURE>`,
not zero), and
{meth}`GainTuner.get_dof_engine_armature <isaacsim.robot_setup.gain_tuner.GainTuner.get_dof_engine_armature>` /
{meth}`GainTuner.get_dof_effective_max_velocity <isaacsim.robot_setup.gain_tuner.GainTuner.get_dof_effective_max_velocity>`
report what the running engine really uses, compared with
{func}`max_velocity_agrees <isaacsim.robot_setup.gain_tuner.max_velocity_agrees>`. Because
only PhysX re-reads these values mid-run
({func}`backend_reads_usd_while_playing <isaacsim.robot_setup.gain_tuner.backend_reads_usd_while_playing>`),
an edit made under Newton while the timeline plays does not take effect until the next play.

### Discretization (dt) sweep

The {mod}`isaacsim.robot_setup.gain_tuner.discretization_sweep` module implements the
dt-sweep validation test, which probes how a joint's step response degrades as the physics
timestep grows. {func}`dt_sweep_levels <isaacsim.robot_setup.gain_tuner.dt_sweep_levels>`
builds the geometric ladder of timesteps between a min/max bound (validated by
``validate_dt_bounds``), and
{class}`DiscretizationSweepTest <isaacsim.robot_setup.gain_tuner.DiscretizationSweepTest>`
runs a single-timestep accuracy probe at each level.
{func}`aggregate_dt_sweep <isaacsim.robot_setup.gain_tuner.aggregate_dt_sweep>` compares each
level against the finest-dt reference to compute settle-time degradation, steady-state error,
and the accuracy "cliff" (the coarsest dt still within tolerance), and
{func}`select_target_level_index <isaacsim.robot_setup.gain_tuner.select_target_level_index>`
picks the level closest to a requested target dt so results can be reported for the timestep
the user intends to ship.
