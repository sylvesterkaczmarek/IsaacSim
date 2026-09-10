# Overview

## System Identification UI

> **Beta:** This extension is in beta. Its workflow, run contracts, exported run specifications, and APIs are not guaranteed to stay future compatible.

This extension provides the interactive System Identification window, menus, plots, file pickers, and viewport integration.

It depends on `isaacsim.robot_setup.sysid`, which owns the headless runtime, run contracts, optimizers, and simulator bridges. Enable only the backend extension for headless workflows; enable this extension for the **Tools > Robotics > Asset Editors > System Identification** window.

The window follows a six-stage workflow:

1. **Robot** selects the articulation, simulation engine, actuator runtime, and controller settings.
2. **Data** loads telemetry, previews its channels, and defines training and validation chunks.
3. **Parameters** selects the physical parameters and bounds to identify.
4. **Check** validates telemetry quality and parameter identifiability.
5. **Solve** selects residual weights, optimizer settings, and output/writeback behavior.
6. **Results** shows live cost, parameter deltas, residual summaries, sensitivity, and validation metrics.

Run remains disabled until Check has completed for the current robot, data,
parameter selection, residual configuration, and check-relevant solver settings.
Changing any of those inputs requires a fresh Check. Isaac Sim rollouts also
track the timeline state, so Play, Pause, and Stop immediately refresh run
readiness.

Start from a recipe when possible. Recipes populate the UI widgets but remain
fully editable. Review **Outputs & stage changes** before solving: optimized
parameters are authored to the open USD stage only when writeback is enabled
and completes successfully.

For USD-authored explicit Newton actuators, select the corresponding actuator
runtime under **Advanced simulation settings** in the **Robot** stage. Enable
**Command delay** in **Parameters** when delay should be identified; the
resulting value is quantized to the nearest simulation physics step.
