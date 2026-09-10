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

"""Check panel: pre-solve telemetry quality and identifiability verdicts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

import omni.ui as ui

from .check_suggestions import (
    CheckSuggestion,
    extract_check_suggestions,
    format_presolve_summary,
    suppressed_issue_codes,
)
from .styles import MUTED_LABEL_STYLE as _MUTED_STYLE
from .styles import STALE_LABEL_STYLE as _STALE_STYLE

_VERDICT_COLORS = {
    "identifiable": 0xFF5A9E58,
    "weak": 0xFF4A8DC2,
    "not_identifiable": 0xFF4444C8,
    "unknown": 0xFF8A8A8A,
}
_VERDICT_LABELS = {
    "identifiable": "identifiable",
    "weak": "weak",
    "not_identifiable": "not identifiable",
    "unknown": "could not assess",
}
_MAX_VERDICT_ROWS = 14
_ROW_HEIGHT = 22


class CheckPanel:
    """Builds and manages the Check collapsible section.

    Args:
        on_run_check: Callback fired when the user clicks Run Check.
        on_apply_suggestion: Callback fired with a :class:`CheckSuggestion`
            when the user clicks a suggestion's Apply button.
    """

    def __init__(
        self,
        on_run_check: Callable[[], None],
        on_apply_suggestion: Callable[[CheckSuggestion], None] | None = None,
    ) -> None:
        self._on_run_check = on_run_check
        self._on_apply_suggestion_callback = on_apply_suggestion
        self._report = None
        self._suggestions: list[CheckSuggestion] = []
        # Guards against a suggestion being applied more than once from the same report:
        # ``_suggestion_frame.rebuild()`` only takes effect next frame, so a rapid
        # double-click would otherwise fire the still-live Apply button repeatedly and,
        # for the command-alignment delta, keep accumulating the shift.
        self._suggestion_applied = False
        self._run_button: Optional[ui.Button] = None
        self._status_label: Optional[ui.Label] = None
        self._stale_label: Optional[ui.Label] = None
        self._quality_label: Optional[ui.Label] = None
        self._suggestion_frame: Optional[ui.Frame] = None
        self._verdict_frame: Optional[ui.Frame] = None
        self._enabled = True

    # ------------------------------------------------------------------ build

    def build(self) -> None:
        """Handle build."""
        with ui.VStack(spacing=4):
            with ui.HStack(spacing=6, height=28):
                self._run_button = ui.Button(
                    "Run Check",
                    width=110,
                    height=26,
                    tooltip=(
                        "Validate telemetry quality and per-parameter identifiability "
                        "before spending rollouts on a solve. Warnings do not block the run."
                    ),
                    clicked_fn=self._on_run_check,
                )
                self._status_label = ui.Label("", word_wrap=True, style=_MUTED_STYLE)
            self._stale_label = ui.Label(
                "Inputs changed since the last check - re-run Check.",
                word_wrap=True,
                visible=False,
                style=_STALE_STYLE,
            )
            self._quality_label = ui.Label(
                "Load data and select parameters, then Run Check to validate before solving.",
                word_wrap=True,
                style=_MUTED_STYLE,
            )
            self._suggestion_frame = ui.Frame()
            self._suggestion_frame.set_build_fn(self._build_suggestion_rows)
            self._verdict_frame = ui.Frame()
            self._verdict_frame.set_build_fn(self._build_verdict_rows)

    # ----------------------------------------------------------------- state

    def set_run_enabled(self, enabled: bool, hint: str = "") -> None:
        """Set run enabled.

        Args:
            enabled: Whether the option is enabled.
            hint: Hint value.
        """
        if self._run_button is not None:
            self._run_button.enabled = bool(enabled) and self._enabled
        if hint:
            self.set_status(hint)

    def set_status(self, text: str) -> None:
        """Set status.

        Args:
            text: Text to display.
        """
        if self._status_label is not None:
            self._status_label.text = text

    def set_stale(self, stale: bool) -> None:
        """Set stale.

        Args:
            stale: Stale value.
        """
        if self._stale_label is not None:
            self._stale_label.visible = bool(stale) and self._report is not None

    def set_report(self, report: Any) -> None:
        """Set report.

        Args:
            report: Validation report to display.
        """
        self._report = report
        self._suggestions = extract_check_suggestions(report) if report is not None else []
        # A fresh report re-arms applying; clearing (report=None) keeps the guard so an
        # already-applied suggestion cannot fire again before Check is re-run.
        if report is not None:
            self._suggestion_applied = False
        if self._quality_label is not None:
            self._quality_label.text = self._format_quality(report, self._suggestions)
        if self._suggestion_frame is not None:
            self._suggestion_frame.rebuild()
        if self._verdict_frame is not None:
            self._verdict_frame.rebuild()

    def get_suggestions(self) -> list[CheckSuggestion]:
        """Get suggestions.

        Returns:
            The resulting value.
        """
        return list(self._suggestions)

    def set_enabled(self, enabled: bool) -> None:
        """Set enabled.

        Args:
            enabled: Whether the option is enabled.
        """
        self._enabled = bool(enabled)
        if self._run_button is not None:
            self._run_button.enabled = self._enabled
        if self._suggestion_frame is not None:
            self._suggestion_frame.rebuild()

    def cleanup(self) -> None:
        """Release resources."""
        self._report = None
        self._suggestions = []
        self._suggestion_applied = False
        self._run_button = None
        self._status_label = None
        self._stale_label = None
        self._quality_label = None
        self._suggestion_frame = None
        self._verdict_frame = None

    # --------------------------------------------------------------- private

    @staticmethod
    def _format_quality(report: Any, suggestions: list[CheckSuggestion] | None = None) -> str:
        if report is None:
            return "Load data and select parameters, then Run Check to validate before solving."
        quality = report.telemetry_quality or {}
        lines = [
            "Telemetry: "
            f"{quality.get('sample_count', '?')} samples, "
            f"{quality.get('num_joints', '?')} joints, "
            f"{quality.get('duration_seconds', 0.0):.2f}s at "
            f"{quality.get('nominal_sample_rate_hz', 0.0):.1f} Hz"
        ]
        presolve_summary = format_presolve_summary(getattr(report, "presolve", None))
        if presolve_summary:
            lines.append(presolve_summary)
        # Findings with a structured suggestion row are not repeated as raw text.
        suppressed = suppressed_issue_codes(suggestions or [])
        for issue in report.issues:
            if issue.code in suppressed:
                continue
            lines.append(f"[{issue.severity}] {issue.message}")
        return "\n".join(lines)

    def _on_apply_clicked(self, suggestion: CheckSuggestion) -> None:
        # One apply per report: subsequent clicks (e.g. an accidental double-click) are
        # ignored until Check is re-run and a fresh report re-arms the button.
        if self._suggestion_applied:
            return
        if self._on_apply_suggestion_callback is not None:
            self._suggestion_applied = True
            self._on_apply_suggestion_callback(suggestion)

    def _build_suggestion_rows(self) -> None:
        if not self._suggestions:
            return
        with ui.VStack(spacing=2):
            for suggestion in self._suggestions:
                with ui.HStack(height=_ROW_HEIGHT, spacing=6):
                    ui.Label(
                        suggestion.title,
                        word_wrap=False,
                        elided_text=True,
                        tooltip=suggestion.detail,
                        style=_STALE_STYLE,
                    )
                    ui.Spacer(width=6)
                    button = ui.Button(
                        suggestion.apply_label,
                        width=170,
                        height=_ROW_HEIGHT - 2,
                        tooltip=suggestion.detail,
                        clicked_fn=lambda s=suggestion: self._on_apply_clicked(s),
                    )
                    button.enabled = self._enabled

    def _build_verdict_rows(self) -> None:
        if self._report is None or not self._report.parameter_verdicts:
            ui.Label("No identifiability verdicts yet.", style=_MUTED_STYLE, word_wrap=True)
            return
        verdicts = self._report.parameter_verdicts
        rows = min(len(verdicts), _MAX_VERDICT_ROWS)
        with ui.ScrollingFrame(
            height=ui.Length(_ROW_HEIGHT * rows + 4),
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
        ):
            with ui.VStack(spacing=2):
                for verdict in verdicts:
                    tooltip = "\n".join(verdict.notes) if verdict.notes else ""
                    with ui.HStack(height=_ROW_HEIGHT):
                        ui.Label(verdict.name, tooltip=tooltip)
                        ui.Spacer()
                        ui.Label(
                            _VERDICT_LABELS.get(verdict.verdict, verdict.verdict),
                            width=120,
                            alignment=ui.Alignment.RIGHT,
                            style={"color": _VERDICT_COLORS.get(verdict.verdict, 0xFF8A8A8A)},
                            tooltip=tooltip,
                        )
