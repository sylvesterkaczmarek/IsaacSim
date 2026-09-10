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

"""All-joints Test Information results panels for the Gain Tuner (Snap / Stress / dt-Sweep).

Each results mode renders the same structure -- an optional context line, a
scene-level verdict banner, and a per-joint results table -- differing only in the
columns, the per-row cell values, the verdict aggregation, and any mode-specific
header/footer content. :class:`ResultsPanel` captures that shared spec-driven flow
once (the ``build_results_panel`` the review asked for), and the three concrete
subclasses supply the per-mode spec. The panels delegate value formatting,
verdict/color helpers, banner/table primitives, and chart building back to the
owning ``UIBuilder`` so behavior matches the single-file implementation exactly.
"""

from __future__ import annotations

import omni.ui as ui
from omni.physics.tensors import DofType

from .global_variables import (
    GT_DT_SWEEP_CHARTS_CAPTION,
    GT_DT_SWEEP_CLIFF_TOOLTIP,
    GT_DT_SWEEP_COL_CLIFF,
    GT_DT_SWEEP_COL_SETTLE,
    GT_DT_SWEEP_COL_SS_ERROR,
    GT_DT_SWEEP_DEGRAD_TOOLTIP,
    GT_DT_SWEEP_HEADER_CONTEXT,
    GT_DT_SWEEP_TABLE_TITLE,
    GT_RESULTS_COL_JOINT,
    GT_RESULTS_COL_RESULT,
    GT_SNAP_COL_LOWER_MAX,
    GT_SNAP_COL_LOWER_MEAN,
    GT_SNAP_COL_LOWER_SETTLE,
    GT_SNAP_COL_UPPER_MAX,
    GT_SNAP_COL_UPPER_MEAN,
    GT_SNAP_COL_UPPER_SETTLE,
    GT_SNAP_TABLE_TITLE,
    GT_STRESS_COL_MAX_VELOCITY,
    GT_STRESS_COL_TRIGGER_TIME,
    GT_STRESS_COL_TRIGGER_VELOCITY,
    GT_STRESS_TABLE_TITLE,
)
from .style import FONT_SIZE, LABEL_COLOR, MUTED_LABEL_COLOR

__all__ = [
    "ResultsPanel",
    "SnapResultsPanel",
    "StressResultsPanel",
    "DtSweepResultsPanel",
]


def _is_discretization(metric: dict) -> bool:
    """Return True when a per-DOF metrics dict came from the dt physics sweep."""
    return metric.get("test_type") == "discretization_sweep"


class ResultsPanel:
    """Base for the all-joints Test Information results panels.

    Renders one results panel from a per-mode spec: filter the tuner's per-DOF
    metrics, show an empty state when there is nothing to report, then draw an
    optional pre-banner context line, the scene-level verdict banner, the results
    table (title + header + one row per tested joint), and any post-table content
    (e.g. the dt-sweep charts). Subclasses provide the spec by overriding the
    :meth:`keep`, :meth:`title`, :meth:`columns`, :meth:`make_row`, and
    :meth:`verdict` hooks, and optionally :meth:`pre_banner` / :meth:`post_table`.

    Args:
        ui_builder: The owning ``UIBuilder`` whose tuner, formatting/verdict
            helpers, and banner/table primitives the panel delegates to.
    """

    def __init__(self, ui_builder) -> None:
        self._ub = ui_builder

    def keep(self, metric: dict) -> bool:
        """Return True to include a per-DOF metrics dict in this panel."""
        raise NotImplementedError

    def title(self) -> str:
        """Return the results-table title label."""
        raise NotImplementedError

    def columns(self) -> list[tuple[str, int, str | None]]:
        """Return the ``(label, fraction, tooltip)`` header column specs."""
        raise NotImplementedError

    def make_row(self, joint_name: str, metric: dict, is_rotational: bool) -> list[tuple[str, int, int | None]]:
        """Return one table row's ``(text, fraction, color)`` cell specs."""
        raise NotImplementedError

    def verdict(self, dof_metrics: list[tuple[int, dict]]) -> tuple[str, int]:
        """Return the ``(banner_text, banner_color)`` scene-level verdict."""
        raise NotImplementedError

    def pre_banner(self, dof_metrics: list[tuple[int, dict]]) -> None:
        """Render optional content above the verdict banner (default nothing)."""

    def post_table(self, dof_metrics: list[tuple[int, dict]], joint_name_fn, is_rotational_fn) -> None:
        """Render optional content below the results table (default nothing)."""

    def build_results_panel(self) -> None:
        """Render the panel: context line, verdict banner, and per-joint table.

        This is the single spec-driven results-panel builder shared by every test
        mode; the per-mode differences come entirely from the overridable hooks.
        """
        ub = self._ub
        metrics = ub._gains_tuner.get_test_result_metrics() or {}
        dof_metrics = [(dof, m) for dof, m in sorted(metrics.items()) if isinstance(m, dict) and self.keep(m)]
        if not dof_metrics:
            ub._build_test_info_empty_state()
            return

        try:
            dof_names = ub._gains_tuner.get_articulation().dof_names
        except (AttributeError, RuntimeError):
            dof_names = None

        def joint_name(dof: int) -> str:
            if dof_names is not None and 0 <= dof < len(dof_names):
                return dof_names[dof]
            return f"DOF {dof}"

        def is_rotational(dof: int) -> bool:
            try:
                return ub._gains_tuner.get_articulation().dof_types[dof] == DofType.Rotation
            except (AttributeError, IndexError, RuntimeError):
                return False

        self.pre_banner(dof_metrics)

        banner_text, banner_color = self.verdict(dof_metrics)
        ub._build_verdict_banner(banner_text, banner_color)
        ui.Spacer(height=6)

        with ui.HStack(height=20):
            ui.Spacer(width=8)
            ui.Label(self.title(), style={"color": LABEL_COLOR, "font_size": FONT_SIZE})
        ub._build_results_table_header(self.columns())
        for dof, metric in dof_metrics:
            ub._build_results_table_row(self.make_row(joint_name(dof), metric, is_rotational(dof)))

        self.post_table(dof_metrics, joint_name, is_rotational)


class SnapResultsPanel(ResultsPanel):
    """Snap-to-Limits results: verdict banner + all-joints PASS/BLOCKED/FAIL table."""

    def keep(self, metric: dict) -> bool:
        return not _is_discretization(metric)

    def title(self) -> str:
        return GT_SNAP_TABLE_TITLE

    def columns(self) -> list[tuple[str, int, str | None]]:
        return [
            (GT_RESULTS_COL_JOINT, 3, None),
            (GT_SNAP_COL_LOWER_MEAN, 2, None),
            (GT_SNAP_COL_LOWER_MAX, 2, None),
            (GT_SNAP_COL_LOWER_SETTLE, 2, None),
            (GT_SNAP_COL_UPPER_MEAN, 2, None),
            (GT_SNAP_COL_UPPER_MAX, 2, None),
            (GT_SNAP_COL_UPPER_SETTLE, 2, None),
            (GT_RESULTS_COL_RESULT, 2, None),
        ]

    def make_row(self, joint_name: str, metric: dict, is_rotational: bool) -> list[tuple[str, int, int | None]]:
        ub = self._ub
        joint, lmean, lmax, lsettle, umean, umax, usettle, result, status = ub._snap_table_row(
            joint_name, metric, is_rotational
        )
        return [
            (joint, 3, None),
            (lmean, 2, None),
            (lmax, 2, None),
            (lsettle, 2, None),
            (umean, 2, None),
            (umax, 2, None),
            (usettle, 2, None),
            (result, 2, ub._snap_status_color(status)),
        ]

    def verdict(self, dof_metrics: list[tuple[int, dict]]) -> tuple[str, int]:
        ub = self._ub
        verdict = ub._snap_verdict([m.get("status", "fail") for _dof, m in dof_metrics])
        return ub._snap_verdict_text(verdict), ub._snap_status_color(verdict)


class StressResultsPanel(ResultsPanel):
    """Stress-test results: mode/seed context, verdict banner + all-joints STABLE/UNSTABLE table."""

    def keep(self, metric: dict) -> bool:
        return not _is_discretization(metric)

    def title(self) -> str:
        return GT_STRESS_TABLE_TITLE

    def columns(self) -> list[tuple[str, int, str | None]]:
        return [
            (GT_RESULTS_COL_JOINT, 3, None),
            (GT_STRESS_COL_MAX_VELOCITY, 2, None),
            (GT_STRESS_COL_TRIGGER_TIME, 2, None),
            (GT_STRESS_COL_TRIGGER_VELOCITY, 2, None),
            (GT_RESULTS_COL_RESULT, 2, None),
        ]

    def make_row(self, joint_name: str, metric: dict, is_rotational: bool) -> list[tuple[str, int, int | None]]:
        ub = self._ub
        joint, max_vel, trigger_time, trigger_vel, result, status = ub._stress_table_row(joint_name, metric)
        return [
            (joint, 3, None),
            (max_vel, 2, None),
            (trigger_time, 2, None),
            (trigger_vel, 2, None),
            (result, 2, ub._stress_status_color(status)),
        ]

    def verdict(self, dof_metrics: list[tuple[int, dict]]) -> tuple[str, int]:
        ub = self._ub
        verdict = ub._stress_verdict([m.get("status", "stable") for _dof, m in dof_metrics])
        return ub._stress_verdict_text(verdict), ub._stress_status_color(verdict)

    def pre_banner(self, dof_metrics: list[tuple[int, dict]]) -> None:
        first = next((m for _dof, m in dof_metrics), {})
        mode_str = str(first.get("test_mode", "random_walk")).replace("_", " ").title()
        seed_val = first.get("seed", "?")
        with ui.HStack(height=18):
            ui.Spacer(width=8)
            ui.Label(
                f"Mode: {mode_str} | Seed: {seed_val}",
                style={"color": MUTED_LABEL_COLOR, "font_size": FONT_SIZE},
            )
            ui.Spacer()
        ui.Spacer(height=4)


class DtSweepResultsPanel(ResultsPanel):
    """dt-Sweep results: target-dt context, verdict banner, all-joints table, and vs-dt charts."""

    def keep(self, metric: dict) -> bool:
        return _is_discretization(metric)

    def title(self) -> str:
        return GT_DT_SWEEP_TABLE_TITLE

    def columns(self) -> list[tuple[str, int, str | None]]:
        return [
            (GT_RESULTS_COL_JOINT, 3, None),
            (GT_RESULTS_COL_RESULT, 2, None),
            (GT_DT_SWEEP_COL_SETTLE, 3, GT_DT_SWEEP_DEGRAD_TOOLTIP),
            (GT_DT_SWEEP_COL_SS_ERROR, 2, None),
            (GT_DT_SWEEP_COL_CLIFF, 3, GT_DT_SWEEP_CLIFF_TOOLTIP),
        ]

    def make_row(self, joint_name: str, metric: dict, is_rotational: bool) -> list[tuple[str, int, int | None]]:
        ub = self._ub
        joint, result, settle, ss_error, cliff, status = ub._discretization_table_row(joint_name, metric, is_rotational)
        return [
            (joint, 3, None),
            (result, 2, ub._dt_sweep_status_color(status)),
            (settle, 3, None),
            (ss_error, 2, None),
            (cliff, 3, None),
        ]

    def verdict(self, dof_metrics: list[tuple[int, dict]]) -> tuple[str, int]:
        ub = self._ub
        statuses = [str(m.get("status", "did_not_settle")) for _dof, m in dof_metrics]
        verdict = ub._dt_sweep_verdict(statuses)
        return ub._dt_sweep_verdict_text(verdict), ub._dt_sweep_status_color(verdict)

    @staticmethod
    def _target_dt(dof_metrics: list[tuple[int, dict]]):
        """Return the sweep's target dt (first joint reporting one), or None."""
        return next((m.get("target_dt") for _dof, m in dof_metrics if m.get("target_dt")), None)

    def pre_banner(self, dof_metrics: list[tuple[int, dict]]) -> None:
        ub = self._ub
        target_dt = self._target_dt(dof_metrics)
        if ub._is_finite_number(target_dt) and target_dt > 0:
            with ui.HStack(height=20):
                ui.Spacer(width=8)
                ui.Label(
                    GT_DT_SWEEP_HEADER_CONTEXT.format(dt=float(target_dt), hz=1.0 / float(target_dt)),
                    style={"color": MUTED_LABEL_COLOR, "font_size": FONT_SIZE},
                )
                ui.Spacer()
            ui.Spacer(height=4)

    def post_table(self, dof_metrics: list[tuple[int, dict]], joint_name_fn, is_rotational_fn) -> None:
        ub = self._ub
        target_dt = self._target_dt(dof_metrics)
        if ub._is_finite_number(target_dt) and target_dt > 0:
            ui.Spacer(height=6)
            with ui.HStack(height=18):
                ui.Spacer(width=8)
                ui.Label(
                    GT_DT_SWEEP_CHARTS_CAPTION.format(dt=float(target_dt)),
                    style={"color": MUTED_LABEL_COLOR, "font_size": FONT_SIZE},
                )
                ui.Spacer()
        ub._build_dt_sweep_vs_dt_charts(dof_metrics, joint_name_fn, is_rotational_fn, target_dt)
