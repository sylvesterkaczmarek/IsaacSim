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

"""Actionable suggestions extracted from a pre-solve check report.

Pure Python (no omni.ui dependency, like :mod:`pipeline_state`) so the
detect-and-apply logic is testable headless. The Check panel renders each
suggestion as a row with an Apply button; the UI builder routes the applied
value to the owning panel (feedforward -> Simulation, delay -> Data).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..telemetry_quality import COMMAND_LAG_WARN_SAMPLES

SUGGESTION_KIND_FEEDFORWARD = "feedforward"
SUGGESTION_KIND_COMMAND_ALIGNMENT = "command_alignment"

_FEEDFORWARD_MODE_ORDER = ("none", "gravity", "inverse_dynamics")


@dataclass(frozen=True)
class CheckSuggestion:
    """One actionable finding from a check report.

    ``value`` semantics depend on ``kind``: for ``feedforward`` it is the
    recommended mode string; for ``command_alignment`` it is an additive DELTA in
    seconds — the lag is re-measured on the already-shifted loaded trajectory,
    so applying means adding it to the current ``command_alignment_seconds``.
    ``issue_codes`` names the raw report issue lines this row supersedes, so
    the panel can drop them instead of rendering the same finding twice.
    """

    kind: str
    title: str
    detail: str
    apply_label: str
    value: Any
    issue_codes: tuple[str, ...] = field(default=())


def extract_check_suggestions(report: Any) -> list[CheckSuggestion]:
    """Return the actionable suggestions carried by a ``SysIdCheckReport``.

    Args:
        report: Validation report to display.

    Returns:
        The resulting value.
    """
    suggestions: list[CheckSuggestion] = []
    presolve = getattr(report, "presolve", None) or {}
    feedforward = presolve.get("feedforward") or {}
    recommended = str(feedforward.get("recommended", "") or "")
    configured = str(feedforward.get("configured", "") or "")
    if recommended and recommended != configured:
        residuals = feedforward.get("residual_rms") or {}
        parts = [f"{mode} {float(residuals[mode]):.3g}" for mode in _FEEDFORWARD_MODE_ORDER if mode in residuals]
        detail = (
            f"Measured torque vs candidate feedforward, residual RMS: {' / '.join(parts)} Nm. "
            f"Configured '{configured}'; '{recommended}' best explains the controller. "
            "Applying sets the Controller feedforward on the Robot panel."
        )
        suggestions.append(
            CheckSuggestion(
                kind=SUGGESTION_KIND_FEEDFORWARD,
                title=f"Controller feedforward: '{recommended}' recommended (configured '{configured}')",
                detail=detail,
                apply_label=f"Apply '{recommended}'",
                value=recommended,
                issue_codes=("feedforward_mode_mismatch",),
            )
        )

    quality = getattr(report, "telemetry_quality", None) or {}
    lag = quality.get("command_lag") or {}
    suggested = lag.get("suggested_command_alignment_seconds")
    common_samples = lag.get("common_mode_lag_samples")
    if suggested is not None and common_samples is not None and abs(float(common_samples)) > COMMAND_LAG_WARN_SAMPLES:
        suggested = float(suggested)
        lag_ms = 1000.0 * float(lag.get("common_mode_lag_seconds") or 0.0)
        spread = lag.get("residual_spread_samples")
        spread_text = (
            f"; per-joint residual spread {float(spread):.2f} samples (actuator latency, not corrected)"
            if spread is not None
            else ""
        )
        detail = (
            f"Commands lead the measured response by a common-mode {float(common_samples):.1f} samples "
            f"(~{lag_ms:.0f} ms){spread_text}. Applying adds {suggested:+.4f} s to the Data panel's command "
            "alignment and reloads the telemetry; uncorrected, the solve mis-attributes the delay to drive "
            "gains, damping, and friction."
        )
        suggestions.append(
            CheckSuggestion(
                kind=SUGGESTION_KIND_COMMAND_ALIGNMENT,
                title=f"Command alignment: {suggested:+.4f} s correction suggested",
                detail=detail,
                apply_label=f"Apply {suggested:+.4f} s",
                value=suggested,
                issue_codes=("telemetry_command_lag_common_mode",),
            )
        )
    return suggestions


def suppressed_issue_codes(suggestions: list[CheckSuggestion]) -> set[str]:
    """Issue codes rendered as structured suggestion rows (drop the raw lines).

    Args:
        suggestions: Suggestions value.

    Returns:
        The resulting value.
    """
    codes: set[str] = set()
    for suggestion in suggestions:
        codes.update(suggestion.issue_codes)
    return codes


def format_presolve_summary(presolve: dict | None) -> str:
    """One-line summary of the identifiability presolve regression, or ''.

    Args:
        presolve: Presolve value.

    Returns:
        The resulting value.
    """
    if not presolve:
        return ""
    rank = presolve.get("rank")
    columns = presolve.get("active_column_count")
    if rank is None or columns is None:
        return ""
    condition = presolve.get("condition")
    condition_text = f", condition {float(condition):.2e}" if condition is not None else ""
    target = presolve.get("target_source")
    target_text = f" (target: {target})" if target else ""
    return f"Presolve: rank {int(rank)}/{int(columns)}{condition_text}{target_text}"


__all__ = [
    "SUGGESTION_KIND_COMMAND_ALIGNMENT",
    "SUGGESTION_KIND_FEEDFORWARD",
    "CheckSuggestion",
    "extract_check_suggestions",
    "format_presolve_summary",
    "suppressed_issue_codes",
]
