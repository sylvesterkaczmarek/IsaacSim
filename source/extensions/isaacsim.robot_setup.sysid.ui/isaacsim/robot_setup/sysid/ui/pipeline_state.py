# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Gated-pipeline state for the SysID UI (pure Python, no omni.ui dependency).

Tracks whether a pre-solve Check has been run and whether its inputs are still
current. A check goes stale when the telemetry source, chunk windows, parameter
selection, robot path, engine, check-relevant simulation settings, residual
configuration, or solver settings change. Check warnings never block the run.
A missing check, a stale check, or a check whose report is explicitly not ``ok``
(telemetry errors, non-identifiable parameters, or unknown verdicts) does block.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CheckSnapshot:
    """A recorded check report plus the input signature it was computed for."""

    report: Any
    signature: tuple


class SysIdPipelineState:
    """Run-gating state machine for the Robot -> Data -> Check -> Solve pipeline."""

    def __init__(self) -> None:
        self._check: CheckSnapshot | None = None

    @staticmethod
    def compute_signature(
        *,
        source_signature: Any,
        chunk_specs: list,
        selected_entries: list,
        robot_path: str,
        engine: str,
        simulation_settings: tuple = (),
        residual_settings: tuple = (),
        solver_settings: tuple = (),
    ) -> tuple:
        """Build a hashable signature of every input that invalidates a check.

        ``simulation_settings`` carries check-relevant simulation values beyond the
        engine string (the Newton solver and feedforward mode — ``engine`` alone is
        "newton" for both MuJoCo and Featherstone-diff). Performance-only settings
        (device, CUDA graph capture) must stay out: they cannot change what the
        check would report.

        Args:
            source_signature: Source signature value.
            chunk_specs: Chunk specs value.
            selected_entries: Selected entries value.
            robot_path: Robot path value.
            engine: Engine value.
            simulation_settings: Simulation settings value.
            residual_settings: Residual configuration used by the check.
            solver_settings: Solver configuration used by the check.

        Returns:
            The resulting value.
        """
        chunks = tuple(
            (
                str(getattr(spec, "name", "")),
                str(getattr(spec, "role", "")),
                float(getattr(spec, "start", 0.0)),
                float(getattr(spec, "end", 0.0)),
                float(getattr(spec, "weight", 1.0)),
                str(getattr(spec, "excitation", "")),
            )
            for spec in (chunk_specs or [])
        )
        entries = tuple(
            (
                str(getattr(entry, "param_type", "")),
                int(getattr(entry, "dof_index", -1)),
                int(getattr(entry, "link_index", -1)),
                int(getattr(entry, "component_index", 0)),
            )
            for entry in (selected_entries or [])
        )
        return (
            source_signature,
            chunks,
            entries,
            str(robot_path),
            str(engine),
            tuple(str(item) for item in (simulation_settings or ())),
            tuple(str(item) for item in (residual_settings or ())),
            tuple(str(item) for item in (solver_settings or ())),
        )

    def record_check(self, report: Any, signature: tuple) -> None:
        """Record check.

        Args:
            report: Validation report to display.
            signature: Signature value.
        """
        self._check = CheckSnapshot(report=report, signature=signature)

    def clear_check(self) -> None:
        """Clear check."""
        self._check = None

    def has_check(self) -> bool:
        """Return whether check.

        Returns:
            The resulting value.
        """
        return self._check is not None

    def is_check_stale(self, current_signature: tuple) -> bool:
        """Return whether check stale.

        Args:
            current_signature: Current signature value.

        Returns:
            The resulting value.
        """
        return self._check is not None and self._check.signature != current_signature

    def current_report(self) -> Any | None:
        """Handle current report.

        Returns:
            The resulting value.
        """
        return self._check.report if self._check is not None else None

    def run_gate(
        self,
        *,
        trajectory_loaded: bool,
        timeline_ok: bool,
        chunk_error: str,
        current_signature: tuple,
    ) -> tuple[bool, str]:
        """Return ``(run_enabled, hint)`` for the current pipeline inputs.

        Args:
            trajectory_loaded: Trajectory loaded value.
            timeline_ok: Timeline ok value.
            chunk_error: Chunk error value.
            current_signature: Current signature value.

        Returns:
            The resulting value.
        """
        if not trajectory_loaded:
            return False, "Load telemetry data to enable Run."
        if not timeline_ok:
            return False, "Press Play on the timeline before running optimization."
        if chunk_error:
            return False, f"Fix chunks before running: {chunk_error}"
        if self._check is None:
            return False, "Run Check (step 4) to validate data and identifiability before solving."
        if self.is_check_stale(current_signature):
            return False, "Inputs changed since the last check - re-run Check before solving."
        if not self._check_report_ok():
            return False, "Check reported blocking issues - resolve them and re-run Check before solving."
        return True, ""

    def _check_report_ok(self) -> bool:
        """Return whether the recorded check report clears the run gate.

        A report is blocking when ``ok`` is explicitly false: `build_sysid_check_report`
        sets that for telemetry errors, non-identifiable parameters, and unknown
        verdicts. Reports that carry no ``ok`` field (test doubles, warning-only
        payloads) stay non-blocking.

        Returns:
            Whether the run may proceed given the recorded report.
        """
        if self._check is None:
            return False
        ok = getattr(self._check.report, "ok", None)
        return ok is not False


__all__ = ["CheckSnapshot", "SysIdPipelineState"]
