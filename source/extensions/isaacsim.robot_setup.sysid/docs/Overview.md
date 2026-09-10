# Overview

## System Identification

> **Beta:** This extension is in beta. Its run contracts, exported run specifications, and APIs are not guaranteed to stay future compatible.

The `isaacsim.robot_setup.sysid` extension identifies robot simulation parameters from recorded motion. It provides the data contracts, telemetry ingestion, parameter models, optimizers, and simulator bridges used to reduce the difference between measured and simulated trajectories.

The extension is the headless backend for the interactive `isaacsim.robot_setup.sysid.ui` extension. Use the UI for guided identification in Isaac Sim, or use the Python API and command-line tools for repeatable and automated runs.

Headless integrations import `run_sysid` from `isaacsim.robot_setup.sysid.execution` and call
`await run_sysid(spec, ...)`. The package root contains no dynamic exports; portable APIs are imported from
their owning modules without loading Kit. Isaac Sim stage, PhysX, and Fabric execution still requires a
bootstrapped Kit runtime. Newton execution can run in ordinary Python with the `newton` wheel dependencies
installed.

For a complete Kit-hosted lifecycle,
{func}`run_headless_sysid_job <isaacsim.robot_setup.sysid.headless_job.run_headless_sysid_job>` loads the stage and
telemetry, runs preflight and preparation, starts the timeline, and executes the solve without creating or
closing `SimulationApp`.
{class}`SysIdJobService <isaacsim.robot_setup.sysid.job_service.SysIdJobService>` serializes these jobs for automation clients that
reuse one headless Kit process through `isaacsim.code_editor.python_server`.

## Local LeRobot-format telemetry

PyArrow is not bundled while its third-party approval is pending. To use this optional loader, provide
PyArrow 24.0.0 in the active environment. CSV, MCAP, and ROS bag ingestion do not require PyArrow.

The local dataset must contain `meta/info.json` and Parquet files under `data/`. Joint positions must use a
recognized signal such as `observation.state` or `observation.qpos`; SysID does not guess joint state from
unrelated numeric columns. The dataset's own license and usage terms remain the user's responsibility.

- {func}`load_trajectory <isaacsim.robot_setup.sysid.ingest.load_trajectory>` loads CSV, ROS 2 bag, MCAP, or LeRobot telemetry into the common trajectory model.
- {func}`build_sysid_check_report <isaacsim.robot_setup.sysid.check_report.build_sysid_check_report>` checks telemetry quality and parameter identifiability before optimization.
- {func}`build_parameter_confidence_report <isaacsim.robot_setup.sysid.analysis.build_parameter_confidence_report>` estimates how strongly the final data constrains each identified parameter.

## Trajectory ingestion and quality checks

Use `TrajectoryLoadConfig` with `load_trajectory` to select CSV, MCAP, ROS 2 bag, or local LeRobot
ingestion. CSV sources can use an explicit column map or the compact built-in layout. MCAP and ROS 2
bag sources require a JSON or YAML topic map. Topic-based loaders align optional channels to position
timestamps using bounded nearest-neighbor matching and record alignment diagnostics in trajectory
provenance.

When a topic map explicitly names a command channel, that channel must decode and align successfully.
The loader reports an error instead of substituting measured positions. Position-only sources may omit
command mapping and intentionally use positions as the command fallback.

Nearby phase-event and collection-manifest files can supply train and validation chunks when their
timestamps overlap the loaded trajectory. Unrelated or stale event logs are ignored rather than rebased
onto a recording. Preflight and telemetry-quality reports validate channel shapes, finiteness, strictly
increasing timestamps, chunk coverage, excitation, and command alignment.

## Analytical presolve and checks

For fixed-base robots, the optional analytical presolve builds a compact USD-derived
recursive Newton-Euler model and uses torque-domain least squares to seed supported
parameters. Joint context must be ordered from base to tip, USD joint bodies must resolve
unambiguously, the active regressor must be full rank and well conditioned, and unconstrained
solutions must remain inside the configured parameter bounds. If any gate fails, the original
seed is retained and the check report explains why.

Torque telemetry must declare `torque_semantics` explicitly as `link_side` or `external`.
Only link-side transmitted joint torque is used for inertial and friction seeding. External or
unspecified torque channels are excluded rather than guessed.

The pre-solve check report combines telemetry quality with per-parameter identifiability and
uses `unknown` when a stage or compatible analytical equation is unavailable. Unknown verdicts
do not produce an `ok` result. The run report records configuration, optimizer status, validation
metrics and artifacts, delay quantization, tuning assessment, and parameter sensitivity. The
reported `sens. span` is the parameter offset associated with a 10% modeled cost rise; it is not
a statistical confidence interval.

Both reports have versioned JSON schemas under `resources/schemas/`.

## Recipes and validation artifacts

Bundled recipes configure parameter families, solver settings, chunking, and compatible residual
channels for common robot classes. Changing parameter-space flags clears an existing selection so
stale parameter rows cannot survive a recipe change. Optional validation artifacts include
position, velocity, effort, and angle-delta plots, RMSE summaries, and a trajectory animation or
storyboard. Generated artifact paths are contained beneath the configured artifact directory.

Importing the package root does not require Kit or Carbonite. Kit-dependent services are imported
only when their adapters are called, preserving use from standalone Python and wheel builds.

## Getting started with a new robot

### Optimization and simulation

- {class}`ParameterSpace <isaacsim.robot_setup.sysid.parameter_space.ParameterSpace>` maps bounded optimizer variables to physical robot parameters.
- {func}`create_optimizer <isaacsim.robot_setup.sysid.optimizer_factory.create_optimizer>` provides Levenberg-Marquardt, CMA-ES, Bayesian, and Adam gradient-descent backends.
- {class}`IsaacSimSysIdBridge <isaacsim.robot_setup.sysid.isaac_sim_sysid_bridge.IsaacSimSysIdBridge>` evaluates candidates in an Isaac Sim-hosted rollout.
- {class}`NewtonSysIdBridge <isaacsim.robot_setup.sysid.newton_sysid_bridge.NewtonSysIdBridge>` and {class}`NewtonDifferentiableSysIdBridge <isaacsim.robot_setup.sysid.newton_diff_sysid_bridge.NewtonDifferentiableSysIdBridge>` provide standalone Newton rollout paths.

Importing contracts and portable helpers does not bootstrap Kit, USD, or a simulator. Isaac Sim rollouts still require a bootstrapped Kit runtime.

## Interactive workflow

Enable `isaacsim.robot_setup.sysid.ui`, then open **Tools > Robotics > Asset Editors > System Identification**. The UI guides you through six stages:

1. **Robot** selects the articulation, physics engine, and actuator runtime.
2. **Data** loads telemetry and defines training and validation chunks.
3. **Parameters** selects the values and bounds to optimize.
4. **Check** evaluates data quality and identifiability before spending rollout time.
5. **Solve** configures and runs the optimizer.
6. **Results** compares parameter values, residuals, confidence ranges, and held-out validation metrics.

Start with a built-in recipe when possible. Recipes pre-fill the workflow for common robot classes while leaving every setting editable.

## Headless quick start

Synthesize a complete run specification from a robot, telemetry source, and recipe. The tools ship
with the extension under its `tools/` directory and run through the Isaac Sim Python launcher
(`python.sh` on Linux, `python.bat` on Windows):

```bash
./python.sh tools/headless_sysid_solve.py --robot /World/Robot --telemetry data.csv --recipe drive_calibration_inair --stage robot.usd
```

The command synthesizes a complete run spec (saved next to the results for later editing), prints the
pre-solve check, and runs the solve. The UI follows a gated pipeline — Robot → Data →
Parameters → **Check** → Solve → Results: the Check stage validates telemetry quality and
per-parameter identifiability before any rollouts are spent, and the Run button requires a
fresh check. After a solve, the Results panel reports each identified value with a sensitivity
range and a verdict (well/weakly constrained) so you can tell which parameters to trust.
Quick-start parallel cloning is disabled unless `--source-env` identifies a cloneable environment
prim such as `/World/envs/env_0`.

Filesystem paths stored in a RunSpec are portable: relative stage, telemetry, mapping, manifest, and output
paths resolve from the directory containing the RunSpec JSON file, independent of the shell's working directory.
Absolute paths and asset URLs are preserved by both the command-line tools and persistent job service.

## Persistent headless jobs

Launch one headless Kit process with `isaacsim.code_editor.python_server` and this backend enabled. Remote
clients submit a JSON-safe
{class}`SysIdServiceJobRequest <isaacsim.robot_setup.sysid.job_service.SysIdServiceJobRequest>` through
{func}`submit_sysid_job <isaacsim.robot_setup.sysid.job_service.submit_sysid_job>`, then use
{func}`get_sysid_job_status <isaacsim.robot_setup.sysid.job_service.get_sysid_job_status>` or
{func}`cancel_sysid_job <isaacsim.robot_setup.sysid.job_service.cancel_sysid_job>`. The singleton service runs one trial at
a time because the USD context, timeline, and active physics engine are process-wide. By default it stops
simulation and creates a clean empty stage after each terminal job.

The Python server is a trusted-code interface, not an authentication or sandbox boundary. Its process-wide
job service rejects all filesystem-backed requests until the host calls
`get_sysid_job_service(allowed_roots=["C:/trusted/sysid"])` before the first submission. Every request,
run-spec, telemetry, stage, mapping, provenance, animation, and result path that addresses the local filesystem
must resolve beneath one of those roots; paths escaping through `..`, symlinks, or `file://` URLs are rejected.
Keep the server bound to trusted clients and grant the Kit process only the filesystem permissions it needs.

The standalone `tools/headless_sysid_solve.py` command uses the same public headless job API while continuing
to own `SimulationApp` only at its outer process boundary.

## Fast offline PhysX stepping

Isaac Sim rollouts default to `simulation.offline_stepping = true`. In this mode the bridge advances PhysX
directly with `SimulationManager.step()` and does not render or pump a complete Kit frame for every physics
step. If the timeline is stopped, the bridge starts it long enough to initialize physics automatically. The
bridge then pauses timeline-driven updates before calling `SimulationManager.step()` so physics has exactly
one step owner. Set `offline_stepping` to `false` to use the timeline-driven path for interactive playback
and debugging.

## Physics backend and actuator compatibility

The rollout host, physics backend, and actuator implementation are independent run-spec axes.
`simulation.engine` selects `isaac_sim` or standalone `newton`; `simulation.physics_backend`
selects `auto`, `physx`, or `newton`; and `simulation.actuator_runtime` selects `auto`,
`implicit_drive`, `newton_explicit`, or `mixed`. Isaac Sim + Newton requires the optional
`isaacsim.physics.newton` extension and switches through `SimulationManager`, while standalone
Newton uses the Newton 1.5 dependency set declared by the `newton` package extra.

For standalone Newton, `mixed` is the recommended actuator mode. Existing USD Newton actuator prims own only their
target joints; the remaining joints continue to use implicit USD drives. Stateful parameters
such as PID integral gain and `actuator_command_delay_seconds` can promote only the selected joints into temporary
session-layer actuator prims by setting `explicit_actuator_policy` to `promote_selected`.
Promotion is non-destructive during a run. Accepted results are copied to the root layer only
when actuator writeback is requested; cancellation or cleanup discards the session edits.

Telemetry timing correction is a separate operation. `telemetry.command_alignment_seconds`
shifts the imported command series once to remove common recording/transport offset. The actuator
parameter is per-DOF plant behavior, is applied once inside the explicit actuator during rollout,
and is persisted as integer `newton:delaySteps`.

Supported mutable explicit actuator schemas include PD, PID, physics-step delay, maximum-effort
and DC-motor clamping. Standalone Newton can execute already-authored neural controllers, but they
are not mutable SysID parameter targets; the Isaac Sim compatibility adapter rejects them.
Position clamping is also rejected because this extension does not yet expose a compatible
optimization contract. A joint has exactly one control owner: explicit joints have
their native drive gains disabled for the rollout, while all other joints retain their authored
drive behavior.

Example run-spec fragment:

```json
{
  "simulation": {
    "engine": "isaac_sim",
    "physics_backend": "newton",
    "actuator_runtime": "mixed",
    "explicit_actuator_policy": "promote_selected"
  },
  "parameters": {
    "space": {
      "include_basic": true,
      "include_command_delay": true
    }
  }
}
```

## Newton MuJoCo CUDA graph capture

Standalone Newton MuJoCo rollouts enable `simulation.newton.cuda_graph_capture` by default on CUDA devices.
The first rollout for a fixed trajectory and population shape runs uncaptured to compile kernels and allocate
solver storage. The bridge then captures that shape and reuses the graph for later optimizer evaluations.
Capture failures are reported and fall back to the uncaptured path; CPU execution is unchanged.

Each captured and uncaptured rollout starts by resetting MuJoCo's rollout-local acceleration warm-start,
actuator-activation, applied-force, and control buffers before restoring the requested initial joint state.
This prevents solver history from a preceding candidate batch from changing the next rollout. Captured and
uncaptured CUDA results should be compared with numerical tolerances rather than bitwise equality because GPU
kernel execution can still introduce small floating-point differences.

Graph replay is intended for repeated, fixed-shape population evaluations. In a reference Franka run with
16 candidates and 800 steps, replay reduced median evaluation time from 13.59 seconds uncaptured to 0.515
seconds captured, a 26.4x speedup after warmup. Treat this result as workload- and hardware-specific rather
than a guaranteed ratio. Set `simulation.newton.cuda_graph_capture` to `false` when diagnosing an individual
rollout or when capture setup cannot be amortized.

## Differentiable Newton bridge (contact-free)

Setting `simulation.newton.solver = "featherstone_diff"` in the run spec selects a
differentiable Newton bridge that rolls trajectories with `SolverFeatherstone` recorded on a
Warp tape. Rollout costs then backpropagate to the parameters, and the `gradient_descent`
(Adam) optimizer backend — chosen automatically in `auto` mode — obtains the exact gradient of
all selected parameters from a single backward pass per iteration. This makes high-dimensional
identification (per-link mass via `parameters.space.include_per_link_mass`, per-link CoM
offsets, and log-Cholesky inertia) tractable where finite differences are not.

The bridge is contact-free by construction (no contact solve is executed): record trajectories
with the robot fixed-based or in the air, following the PACE identification protocol
(arXiv:2509.06342). Supported parameters are joint friction/stiffness/damping, link mass, CoM
offsets, and log-Cholesky inertia; `joint_armature`, `actuator_command_delay_seconds`, integral gains,
and joint-limit scales are rejected with a pointer to the MuJoCo Newton
bridge or the derivative-free backends. Each taped substep owns its Featherstone
factorization workspace, so the mass matrix is rebuilt at the current configuration without
overwriting values required by the reverse pass. Mapped-joint PD, feedforward, friction, and
maximum-effort limiting are evaluated together, and the reported torque is the torque applied
to the simulated plant.
