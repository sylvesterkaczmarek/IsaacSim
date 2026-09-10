# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Global variables for the gain tuner extension configuration."""

# ---------------------------------------------------------------------------
# Branding — shown in the panel header and used as the window / menu title in
# extension.py.
# ---------------------------------------------------------------------------
EXTENSION_TITLE = "Gain Tuner"

EXTENSION_DESCRIPTION = "Auto tuner for articulation PD gains"

# GT_NoSchema (Figma) — empty / missing Robot Schema state
GT_NO_SCHEMA_HEADING = "NO ROBOT SCHEMA"
GT_NO_SCHEMA_MESSAGE = (
    "The selected robot does not have a Schema. You will need to add a schema to utilize the Gain Tuner"
)
GT_NO_SCHEMA_MESSAGE_EMPTY_STAGE = (
    "No robot with Robot Schema was found on the stage. Add a Robot Schema to your robot to use the Gain Tuner."
)
GT_NO_SCHEMA_BUTTON = "ADD ROBOT SCHEMA"

# ---------------------------------------------------------------------------
# Header / tab labels
# ---------------------------------------------------------------------------
GT_ROBOT_LABEL = "Robot:"
GT_BACKEND_LABEL = "Backend:"
GT_GAIN_SETTINGS_TAB = "Gain Settings"
# User-visible label for the validation tab. Internal identifiers still use
# "charts" (e.g. _charts_button/_charts_page) to keep the code diff small.
GT_CHARTS_TAB = "Test Gains"

# ---------------------------------------------------------------------------
# Collapsible section titles
# ---------------------------------------------------------------------------
GT_HAND_TUNED_GAINS_TITLE = "Gains Settings"
GT_ADVANCED_PARAMS_TITLE = "Advanced Actuator Parameters"
GT_TEST_CONTROLS_TITLE = "Test Gains Settings"

# ---------------------------------------------------------------------------
# Save target row
# ---------------------------------------------------------------------------
GT_SAVE_TARGET_LABEL = "Save Target:"
GT_SAVE_BUTTON_LABEL = "Save"
GT_WARN_SAVE_TARGET_UNRESOLVED = "Save target could not be resolved. Check USD layer stack."

# Mirror-writeback toggle + per-backend target pickers (shown only when the robot
# has MuJoCo-native gains or Newton actuators alongside the DriveAPI gains).
GT_MIRROR_TOGGLE_LABEL = "Mirror tuned DriveAPI gains to MuJoCo"
GT_MIRROR_TOGGLE_TOOLTIP = (
    "Also write the tuned PhysX DriveAPI Kp/Kd into this robot's MuJoCo (mjc:*) "
    "gain parameters on save. Off by default so saving never touches MuJoCo "
    "actuator prims unless you opt in. MuJoCo stays a mirror here - it does not "
    "become the active source. (When the MuJoCo solver is running, mjc:* is the "
    "active source and is edited/saved directly instead.)"
)
GT_MJC_TARGET_LABEL = "MuJoCo target:"
GT_NEWTON_TARGET_LABEL = "Newton target:"

# ---------------------------------------------------------------------------
# Controller Gains section
# ---------------------------------------------------------------------------
GT_CONTROLLER_GAINS_TITLE = "Controller Gains"

# Tuning-mode toggle for PhysX PhysicsDrive gains: edit stiffness/damping
# directly, or edit natural frequency + damping ratio (converted to stiffness /
# damping using the joint's effective inertia).
GT_TUNING_MODE_STIFFNESS_LABEL = "Stiffness / Damping"
GT_TUNING_MODE_NATURAL_FREQUENCY_LABEL = "Natural Frequency / Damping Ratio"
GT_TUNING_MODE_TOOLTIP = (
    "Switch between editing the drive stiffness/damping directly and editing the "
    "target natural frequency (Hz) + damping ratio, which are converted to "
    "stiffness/damping using the joint's effective inertia."
)
# Shown in natural-frequency mode when the effective inertia has not been
# computed yet (the timeline must play once to populate physics tensors).
GT_NATURAL_FREQUENCY_NEEDS_INERTIA = (
    "Play the timeline once to compute joint inertia for accurate natural-frequency values."
)

# ---------------------------------------------------------------------------
# Gain source / backend warnings
# ---------------------------------------------------------------------------
GT_WARN_INACTIVE_GAINS = "These gain values are not used by the active controller."

# ---------------------------------------------------------------------------
# Advanced Actuator Parameters: per-backend joint parameter values
# ---------------------------------------------------------------------------
# Armature, joint friction, and the joint velocity limit are authored per
# backend, and an edit writes only the active backend's schema.  Two different
# values are a legitimate authoring choice -- the PhysX and Newton formulations
# need different tuning -- so the per-parameter text built by
# ``gain_display.joint_param_info_text`` is informational: it states each
# backend's value and which one is in effect.  These are the fixed strings
# around it.
GT_JOINT_PARAM_PER_BACKEND_NOTE = (
    "Armature, Joint Friction, and Max Joint Velocity are authored per backend. "
    "An edit here writes only the active backend's schema and leaves the other "
    "backend's value unchanged."
)
GT_JOINT_PARAM_WRITE_FAILED = (
    "Could not write that value to the active backend's joint schema. The field has "
    "been put back to what the stage holds. Check that the running engine is PhysX "
    "or Newton and that the omni.usd.schema.newton extension is enabled."
)
GT_JOINT_PARAM_REPLAY_TO_APPLY = (
    "{backend} reads these three values from the stage once, when the simulation "
    "starts, so an edit made now will not change what is being simulated. Stop and "
    "play again to apply it."
)
GT_JOINT_PARAM_BACKEND_UNSUPPORTED = (
    "The running physics engine is not one this panel knows the joint schema for, "
    "so these values cannot be read or edited. An edit would have to guess which "
    "schema the engine reads, and a wrong guess authors a value nothing uses. "
    "Switch to PhysX or Newton to tune them."
)

# Copy-button labels name the backend written rather than "the other backend":
# which backend that is depends on the engine running right now, and a label that
# does not say leaves the direction of the copy to be guessed.  Formatted at build
# time from the live inactive backend, so an engine switch relabels the button.
# ASCII only -- omni.ui renders non-ASCII as "?".
GT_COPY_JOINT_PARAMS_BUTTON = "Copy To {backend}"
GT_COPY_ALL_JOINT_PARAMS_BUTTON = "Copy All To {backend}"
GT_COPY_JOINT_PARAMS_TOOLTIP = (
    "Copy the values shown above to {backend}'s joint schema for this joint, "
    "without changing them here. This changes what {backend} simulates. Use it to "
    "seed {backend} when it has nothing authored, or to start its tuning from "
    "these values; the two backends are otherwise tuned independently. Values "
    "{backend} already holds are left untouched."
)
GT_COPY_ALL_JOINT_PARAMS_TOOLTIP = (
    "Do the same for every tunable joint, as a single undoable edit. This changes "
    "what {backend} simulates on every joint it writes. Joints whose values "
    "{backend} already holds are left untouched."
)

# ---------------------------------------------------------------------------
# Test Gains Settings section
# ---------------------------------------------------------------------------
GT_TEST_DURATION_LABEL = "Test Duration (s)"
GT_TEST_MODE_LABEL = "Test Mode"
GT_RUN_TEST_BUTTON = "Run Test"
GT_CANCEL_TEST_BUTTON = "Cancel Test"

# ---------------------------------------------------------------------------
# Charts page
# ---------------------------------------------------------------------------
GT_POSITION_CHART_TITLE = "Position"
GT_EFFORT_CHART_TITLE = "Effort (Nm)"
GT_TEST_INFO_TITLE = "Test Information"

# Test Information field labels
GT_TEST_INFO_TEST_TYPE = "Test:"
GT_TEST_INFO_PEAK_ERROR = "Peak Error:"
GT_TEST_INFO_RMS_ERROR = "RMS Error:"
GT_TEST_INFO_SETTLING_TIME = "Settling Time (2%):"
GT_TEST_INFO_OVERSHOOT = "Overshoot:"
GT_TEST_INFO_PASS_FAIL = "Pass / Fail:"
GT_TEST_INFO_INTEGRAL_WINDUP = "Integral Windup:"
GT_TEST_INFO_RESULT = "Result:"

# Stress-test per-joint result labels (mirror develop's Stress Test Results table).
GT_TEST_INFO_MAX_VELOCITY = "Max Velocity:"
GT_TEST_INFO_TRIGGER_TIME = "Trigger Time:"
GT_TEST_INFO_TRIGGER_VELOCITY = "Trigger Velocity:"

# Snap-to-limits per-joint result labels (mirror develop's Snap to Limits Results table).
GT_TEST_INFO_LOWER_MEAN_ERROR = "Lower Mean Error:"
GT_TEST_INFO_LOWER_MAX_ERROR = "Lower Max Error:"
GT_TEST_INFO_LOWER_SETTLE = "Lower Settle:"
GT_TEST_INFO_UPPER_MEAN_ERROR = "Upper Mean Error:"
GT_TEST_INFO_UPPER_MAX_ERROR = "Upper Max Error:"
GT_TEST_INFO_UPPER_SETTLE = "Upper Settle:"

# dt physics sweep per-joint result labels (accuracy-vs-timestep characterization).
GT_TEST_INFO_TARGET_DT = "Target dt:"
GT_TEST_INFO_SETTLE_TIME = "Settle Time:"
GT_TEST_INFO_SETTLE_AT_TARGET = "Settle @ Target:"
GT_TEST_INFO_SETTLE_DEGRAD = "Settle Degradation:"
GT_TEST_INFO_SS_ERROR = "Steady-State Error:"
GT_TEST_INFO_ERROR_DEGRAD = "Error Degradation:"
GT_TEST_INFO_ACCURACY_CLIFF = "Accuracy Cliff:"

# Shared all-joints results-table columns (dt Sweep / Snap / Stress).
GT_RESULTS_COL_JOINT = "Joint"
GT_RESULTS_COL_RESULT = "Result"

# dt physics sweep all-joints results table (Test Gains tab).
GT_DT_SWEEP_TABLE_TITLE = "dt Sweep Results"
GT_DT_SWEEP_COL_SETTLE = "Settle @ Target"
GT_DT_SWEEP_COL_SS_ERROR = "SS Error"
GT_DT_SWEEP_COL_CLIFF = "Accuracy Cliff (dt / Hz)"

# Snap-to-limits all-joints results table + verdict (mirrors develop's table).
GT_SNAP_TABLE_TITLE = "Snap to Limits Results"
GT_SNAP_COL_LOWER_MEAN = "Lower Mean"
GT_SNAP_COL_LOWER_MAX = "Lower Max"
GT_SNAP_COL_LOWER_SETTLE = "Lower Settle"
GT_SNAP_COL_UPPER_MEAN = "Upper Mean"
GT_SNAP_COL_UPPER_MAX = "Upper Max"
GT_SNAP_COL_UPPER_SETTLE = "Upper Settle"
GT_SNAP_VERDICT_ALL_PASSED = "ALL JOINTS PASSED"
GT_SNAP_VERDICT_SOME_BLOCKED = "ALL GAINS OK - SOME JOINTS BLOCKED"
GT_SNAP_VERDICT_SOME_FAILED = "SOME JOINTS FAILED"

# Stress-test all-joints results table + verdict (mirrors develop's table).
GT_STRESS_TABLE_TITLE = "Stress Test Results"
GT_STRESS_COL_MAX_VELOCITY = "Max Vel"
GT_STRESS_COL_TRIGGER_TIME = "Trigger Time"
GT_STRESS_COL_TRIGGER_VELOCITY = "Trigger Vel"
GT_STRESS_VERDICT_ALL_STABLE = "ALL JOINTS STABLE"
GT_STRESS_VERDICT_UNSTABLE = "INSTABILITY DETECTED"

# dt physics sweep scene-level verdict banner text (aggregated over all joints).
GT_DT_SWEEP_VERDICT_ALL_ACCURATE = "ALL ACCURATE AT TARGET DT"
GT_DT_SWEEP_VERDICT_DEGRADED = "ACCURACY DEGRADED AT TARGET DT"
GT_DT_SWEEP_VERDICT_DID_NOT_SETTLE = "JOINTS DID NOT SETTLE AT TARGET DT"
GT_DT_SWEEP_VERDICT_INDETERMINATE = "REFERENCE DT DID NOT SETTLE (INDETERMINATE)"

# dt physics sweep header context + column tooltips (finest-dt reference).
GT_DT_SWEEP_HEADER_CONTEXT = "Target dt: {dt:.5f} s ({hz:.0f} Hz)  |  reference = finest dt"
GT_DT_SWEEP_CLIFF_TOOLTIP = (
    "Coarsest physics timestep (largest dt / lowest Hz) whose accuracy still tracks the finest-dt reference run."
)
GT_DT_SWEEP_DEGRAD_TOOLTIP = (
    "Settling time at the target dt, with its percent increase relative to the finest-dt reference run in parentheses."
)
GT_DT_SWEEP_CHARTS_CAPTION = "Position and Effort charts below show the target dt level ({dt:.5f} s)."

# dt physics sweep vs-dt chart titles (per selected joint).
GT_DT_SWEEP_SETTLE_CHART_TITLE = "Settle Time vs dt"
GT_DT_SWEEP_SS_ERROR_CHART_TITLE = "Steady-State Error vs dt"

# dt physics sweep options-panel group headers + per-field tooltips (units in labels).
GT_DT_SWEEP_GROUP_RANGE = "Timestep range"
GT_DT_SWEEP_GROUP_PROBE = "Probe"
GT_DT_SWEEP_GROUP_THRESHOLDS = "Accuracy thresholds"
GT_DT_SWEEP_TIP_DT_MAX = "Coarsest physics timestep to sweep, in seconds (swept first)."
GT_DT_SWEEP_TIP_DT_MIN = "Finest physics timestep to sweep, in seconds (swept last; used as the accuracy reference)."
GT_DT_SWEEP_TIP_STEPS = "Number of logarithmically-spaced timesteps sampled between dt Max and dt Min."
GT_DT_SWEEP_TIP_TARGET_DT = "Physics timestep you intend to simulate at; each joint is classified at this dt."
GT_DT_SWEEP_TIP_TIMEOUT = "Maximum seconds to wait for a joint to settle at each timestep before marking it unsettled."
GT_DT_SWEEP_TIP_HOLD = "Seconds to hold after settling to measure the steady-state error."
GT_DT_SWEEP_TIP_TOLERANCE = "Position error (radians or meters) within which a joint counts as settled."
GT_DT_SWEEP_TIP_SETTLE_DEGRAD = (
    "Max allowed increase in settling time versus the finest-dt reference before a joint is degraded."
)
GT_DT_SWEEP_TIP_ERROR_DEGRAD = (
    "Max allowed increase in steady-state error versus the finest-dt reference before a joint is degraded."
)
