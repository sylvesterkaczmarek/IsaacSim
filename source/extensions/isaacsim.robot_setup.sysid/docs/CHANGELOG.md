# Changelog

## [2.2.3] - 2026-08-31
### Fixed

- Fell back to the nearest ancestor scope that contains joints when the articulation authors them in a sibling ``<robot>/Physics`` scope, so scope-based physics exports are no longer detected as having zero joints.
- Let an async ``timeline.pause()`` settle for one Kit update before treating the pause as failed, so offline direct-stepping rollouts no longer abort spuriously in the interactive loop.
- Carried up-axis and unit metadata from the source root layer onto the physics override root layer so written assets no longer fall back to USD's Y-up / 1.0 defaults.

## [2.2.2] - 2026-08-25
### Fixed

- Preserved multi-physics USD composition when writing identified neutral and solver-specific joint parameters.

## [2.2.1] - 2026-08-11
### Fixed

- Resolved relative stage, telemetry, mapping, manifest, and output paths from the RunSpec directory in headless tools and the persistent job service.

## [2.2.0] - 2026-08-11
### Changed
- Updated Newton optional dependencies to 1.5.0.

## [2.1.0] - 2026-08-05
### Changed
- Updated Newton optional dependencies to newton 1.5.0rc2, mujoco 3.11.0, and mujoco-warp 3.11.0
- Use `Control.joint_target_q` / `joint_target_qd` (Newton 1.5 rename)
- Use `ModelFlags` for Newton 1.5 solver refresh notifications

## [2.0.0] - 2026-07-30
### Removed

- Removed the Franka-specific tools and tests, the asset-transformer rules module, and the user guide.
- Removed the duplicate joint-ordering helpers from `run_controller`; import them from `articulation_utils`.

### Changed

- Rejected run specifications and run reports that fail schema validation with error severity.
- Required explicit `allowed_roots` for `SysIdServiceJobRequest.load_spec()` filesystem access.
- Reported headless job `ok` as false when a requested stage save fails, and surfaced `check_report_ok`.
- Deferred `pxr` imports so portable modules import without USD installed.
- Packaged the standalone tools with the extension and made their Python launcher search cross-platform.

### Fixed

- Kept USD traversal order when telemetry joint names match multiple joint prims.
- Stopped `read_link_usd_snapshot` from applying `UsdPhysics.MassAPI` to the stage while reading.
- Wrote `--result-json` through the shared run-report contract instead of a schema-invalid payload.
- Warned instead of silently assuming 60 Hz when no physics timestep is available.
- Carried `torque_semantics` from LeRobot `meta/info.json` so torque residuals are usable.

## [1.3.0] - 2026-07-30
### Added

- Added centralized runtime adapter services for `SimulationManager` setup, stepping, callbacks, and Fabric configuration.

### Changed

- Routed Isaac Sim physics initialization, timing, callbacks, and clone-view access through the backend runtime adapter.
- Wired offline stepping, physics backend, and actuator ownership settings into Isaac Sim rollouts.
- Paused timeline updates before direct offline physics stepping.

### Fixed

- Rejected unverified Isaac Sim Newton availability during preflight.
- Treated zero-step offline warmup requests as no-ops.
- Reset MuJoCo rollout-local warm-start, actuator-activation, applied-force, and control buffers at each captured and uncaptured rollout boundary.

## [1.2.1] - 2026-07-30
### Changed

- Packaged the System Identification user guide with the extension documentation.

## [1.2.0] - 2026-07-29
### Added

- Added headless job execution, run sessions, controller orchestration, and job-service contracts.
- Added asset-transformer rules and end-to-end parameter-space and presolve integration coverage.
- Added standalone analysis, solve, sim-to-sim, command-shift, and USD-application tools.
- Added backend architecture and full headless-workflow documentation.

## [1.1.4] - 2026-07-29
### Added

- Added differentiable Newton and Isaac Sim/Fabric rollout adapters.
- Added explicit-PD actuator compatibility and USD parameter read/write support.
- Added rollout-row, recovery, and Featherstone runtime coverage.

### Changed

- Removed contact-offset runtime tuning and updated Newton runtime compatibility bounds.

### Fixed

- Preserved link inertia-frame orientations when applying COM offsets.
- Rejected non-finite differentiable rollouts, adjoints, and parameter gradients.
- Distinguished unsupported Fabric reads from unexpected runtime failures.

## [1.1.3] - 2026-07-29
### Added

- Added the Newton/MuJoCo batched rollout bridge and feedforward support.
- Added command-delay handling and candidate-population saturation behavior.
- Added focused Newton bridge, delay, and rollout-batching coverage.

### Fixed

- Made configured Newton feedforward and neural-actuator parsing fail closed.
- Rejected non-finite controller values and invalidated non-finite physics worlds.
- Released cached Newton population device buffers explicitly.

## [1.1.2] - 2026-07-29
### Added

- Added analytical presolve, parameter-confidence analysis, and tuning assessment.
- Added experiment recipes, example scenes, kinematics helpers, and reusable presets.
- Added machine-readable run and check reports with focused analytical coverage.

## [1.1.1] - 2026-07-29
### Added

- Added bounded parameter spaces and shared rollout-optimizer infrastructure.
- Added Levenberg-Marquardt, gradient-descent, CMA-ES, and Bayesian optimizer backends.
- Added population-saturation diagnostics and deterministic optimizer-core coverage.

### Fixed

- Matched analytical Levenberg-Marquardt Jacobians to weighted torque residuals and used finite differences for nonlinear or unsupported parameter selections.
- Rejected optimizer runs without any finite rollout and enforced complete rollout signal shapes during training and validation.
- Added reproducible Bayesian and gradient-descent seeds and made population saturation stop on peak device-memory usage.

## [1.1.0] - 2026-07-29
### Added

- Added CSV, MCAP, ROS 2 bag, and local LeRobot-format trajectory ingestion.
- Added telemetry alignment, training-log companion discovery, chunk inference, and quality diagnostics.
- Added lazy optional-reader boundaries and packaged telemetry prebundle staging.

### Fixed

- Removed a spurious clone-path error when both required paths are empty.
- Stopped Newton schema preflight from hiding unexpected bridge failures.

## [1.0.1] - 2026-07-23
### Added

- Added portable run specifications, parameter contracts, residual contracts, trajectory primitives, and schemas.
- Added standalone wheel metadata, packaged configuration resources, and Kit-free import-boundary coverage.
- Added parameter application, provenance, runtime-yield, and schema-validation foundations for later backends.

## [1.0.0] - 2026-07-13
### Added

- Added the headless System Identification backend extension skeleton.
- Added the sibling UI extension skeleton as a separate dependent extension.
- Kept the backend package independent of Kit and Carbonite with no lifecycle entry point.
