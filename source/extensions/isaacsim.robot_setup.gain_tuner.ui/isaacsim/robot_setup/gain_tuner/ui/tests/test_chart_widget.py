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

"""Chart behaviour: axis framing and placement, the hover readout, legend colours, and the value grid.

Framing and readout are one subject: a series is drawn across the stretch of the
width its samples cover, and the readout inverts that same placement.

The divergence criterion itself is a question about the joint, not the chart, and
is covered by ``isaacsim.robot_setup.gain_tuner``'s ``test_divergence``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from unittest import mock

import numpy as np
import omni.kit.app
import omni.kit.test
import omni.ui as ui
from isaacsim.robot_setup.gain_tuner.ui import chart_widget
from isaacsim.robot_setup.gain_tuner.ui.chart_widget import (
    VIEW_HEADROOM,
    JointGraphWidget,
    framed_window,
    window_placement,
)


def _diverged_run(
    duration: float = 60.0, diverge_at: float = 5.0, hz: float = 600.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(t, command, observed)`` for a joint that loses stability mid-run.

    Mirrors the reported failure: a long run, a step command, and an oscillating
    exponential blow-up that stays finite -- so nothing here is caught by a
    non-finite check alone.
    """
    t = np.arange(int(hz * duration)) / hz
    command = np.zeros_like(t)
    command[t >= 0.5] = 360.0
    observed = np.zeros_like(t)
    tau = 0.12
    for i in range(1, len(t)):
        observed[i] = observed[i - 1] + (command[i - 1] - observed[i - 1]) * (1.0 / hz) / tau
    diverging = t >= diverge_at
    k = np.arange(int(diverging.sum()))
    observed[diverging] = 360.0 * np.exp(np.minimum(k * 0.029, 44.0)) * np.cos(k * 0.9)
    return t, command, observed


class TestFramedWindow(omni.kit.test.AsyncTestCase):
    """Y window centred on the commanded travel, with room for overshoot."""

    async def test_frames_the_reference_span_with_headroom(self) -> None:
        """The window is the commanded span widened by the view headroom."""
        self.assertEqual(framed_window([np.array([0.0, 360.0])], headroom=1.5), (-90.0, 450.0))

    async def test_spans_every_reference_series(self) -> None:
        """Several selected joints share one window covering all their commands."""
        lo, hi = framed_window([np.array([-10.0, 10.0]), np.array([0.0, 90.0])], headroom=1.0)
        self.assertAlmostEqual(lo, -10.0)
        self.assertAlmostEqual(hi, 90.0)

    async def test_non_finite_reference_samples_are_ignored(self) -> None:
        """A NaN in the command does not produce a non-finite window."""
        lo, hi = framed_window([np.array([0.0, np.nan, 10.0, np.inf])], headroom=1.0)
        self.assertTrue(np.isfinite(lo) and np.isfinite(hi))
        self.assertAlmostEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 10.0)

    async def test_flat_reference_still_yields_a_usable_span(self) -> None:
        """A joint commanded to hold still gets a window with width."""
        lo, hi = framed_window([np.array([5.0, 5.0])], headroom=1.0)
        self.assertGreater(hi - lo, 0.0)
        self.assertLess(lo, 5.0)
        self.assertGreater(hi, 5.0)

    async def test_no_usable_reference_yields_none(self) -> None:
        """Without a reference the caller keeps the full data extent."""
        self.assertIsNone(framed_window([]))
        self.assertIsNone(framed_window([np.array([])]))
        self.assertIsNone(framed_window([np.array([np.nan])]))
        self.assertIsNone(framed_window([None]))

    async def test_applies_the_default_headroom(self) -> None:
        """Callers that do not pass a headroom get the module default."""
        lo, hi = framed_window([np.array([0.0, 2.0])])
        self.assertAlmostEqual(hi - lo, 2.0 * VIEW_HEADROOM)


class TestWindowPlacement(omni.kit.test.AsyncTestCase):
    """Horizontal sizing that keeps a clipped trace on its own stretch of the axis."""

    async def test_full_span_series_fills_the_window(self) -> None:
        """A series covering the whole window is drawn at full width."""
        self.assertEqual(window_placement(0.0, 4.0, 0.0, 4.0), (0.0, 1.0))

    async def test_clipped_series_gets_proportional_width(self) -> None:
        """A trace ending halfway through the run occupies half the axis."""
        offset, width = window_placement(0.0, 2.0, 0.0, 4.0)
        self.assertAlmostEqual(offset, 0.0)
        self.assertAlmostEqual(width, 0.5)

    async def test_late_starting_series_is_offset(self) -> None:
        """A series starting mid-window is inset by its start time."""
        offset, width = window_placement(1.0, 3.0, 0.0, 4.0)
        self.assertAlmostEqual(offset, 0.25)
        self.assertAlmostEqual(width, 0.5)

    async def test_degenerate_window_falls_back_to_full_width(self) -> None:
        """A zero-width or non-finite window cannot be divided up."""
        self.assertEqual(window_placement(0.0, 1.0, 2.0, 2.0), (0.0, 1.0))
        self.assertEqual(window_placement(0.0, 1.0, 0.0, np.nan), (0.0, 1.0))

    async def test_zero_span_series_still_gets_drawable_width(self) -> None:
        """A width of zero would reach ``ui.Fraction(0.0)`` and draw nothing."""
        offset, width = window_placement(2.0, 2.0, 0.0, 4.0)
        self.assertGreater(width, 0.0)
        self.assertLessEqual(offset + width, 1.0)

    async def test_series_at_the_right_edge_keeps_width(self) -> None:
        """A series sitting on the window's right edge is pulled back, not zeroed."""
        offset, width = window_placement(4.0, 4.0, 0.0, 4.0)
        self.assertGreater(width, 0.0)
        self.assertLess(offset, 1.0)
        self.assertLessEqual(offset + width, 1.0)

    async def test_fractions_never_exceed_the_window(self) -> None:
        """Samples past the window edge cannot push the plot wider than the axis."""
        offset, width = window_placement(-5.0, 100.0, 0.0, 4.0)
        self.assertGreaterEqual(offset, 0.0)
        self.assertLessEqual(offset + width, 1.0)


class TestDivergedChart(omni.kit.test.AsyncTestCase):
    """The reported failure, end to end through the widget.

    A diverging joint used to render as an empty chart: its astronomical samples set
    the shared axis range and flattened the real motion below a pixel.
    """

    async def setUp(self) -> None:
        """Charts are built inside a live window, as they are in the panel."""
        self._window = ui.Window("gt_chart_widget_test", width=800, height=400)

    async def tearDown(self) -> None:
        """Windows outlive the test unless destroyed, and leak into the next one."""
        self._window.destroy()
        self._window = None

    def _build(self, **kwargs: object) -> JointGraphWidget:
        t, command, observed = _diverged_run()
        self._t, self._command, self._observed = t, command, observed
        with self._window.frame:
            with ui.VStack():
                return JointGraphWidget(
                    x_data=[t, t],
                    y_data=[command, observed],
                    data_colors=[0xFF999999, 0xFFFFFFFF],
                    header_count=2,
                    reference_count=1,
                    **kwargs,
                )

    async def test_unbounded_chart_is_unreadable(self) -> None:
        """Without a bound the diverged tail owns the axis -- the reported symptom."""
        widget = self._build()
        commanded_span = float(np.max(self._command) - np.min(self._command))
        axis_span = widget._y_data_max - widget._y_data_min
        # The real motion occupies a vanishing share of the range it is drawn in.
        self.assertLess(commanded_span / axis_span, 1e-3)

    async def test_bounded_chart_frames_the_response(self) -> None:
        """With a bound, the axis opens on the motion the user needs to read."""
        widget = self._build(divergence_limits=[None, 1440.0])
        self.assertTrue(widget._diverged)
        # The y window covers the commanded travel rather than the blow-up.
        self.assertLessEqual(widget._y_min, 0.0)
        self.assertGreaterEqual(widget._y_max, 360.0)
        self.assertLess(widget._y_max, 1e4)
        # The x window ends where the response was clipped, so the retained prefix
        # fills the width instead of being squeezed into its share of the full run.
        self.assertLess(widget._x_max, 0.2 * float(self._t[-1]))
        self.assertGreater(widget._x_max, 0.0)
        # The full extent stays reachable through the sliders and Fit Frame View.
        self.assertAlmostEqual(widget._x_data_max, float(self._t[-1]), places=3)
        self.assertGreater(widget._y_data_max, 1e3)

    def _fitted_to_full_extent(self) -> tuple[JointGraphWidget, float, float]:
        """Build the diverged chart and press Fit Frame View, as the footer note invites.

        The default view is framed on the retained response, so every trace spans
        the full width and the placement is the trivial one.  Widening to the full
        extent is what leaves the clipped response drawn across a small fraction
        of the axis while the command still spans all of it -- the geometry the
        readout has to invert.

        Returns:
            The widget, the width fraction the response occupies, and the axis span.
        """
        widget = self._build(divergence_limits=[None, 1440.0])
        self.assertLess(len(widget._x_data[1]), len(widget._x_data[0]), "the response is clipped")
        widget._on_fit_frame_view()
        axis = widget._x_max - widget._x_min
        end_frac = (float(widget._x_data[1][-1]) - widget._x_min) / axis
        self.assertLess(end_frac, 0.5, "the response covers a minority of the full extent")
        return widget, end_frac, axis

    async def test_the_readout_reads_a_clipped_trace_at_the_time_under_the_pointer(self) -> None:
        """A clipped trace reports the time it is drawn at, not a compressed one.

        The response covers about a twelfth of the full extent.  Spreading its
        samples across the whole width -- which is what resolving the pointer in
        the full width's index space amounts to -- would report a time roughly
        twelve times too early, silently, for every reading taken on it.
        """
        widget, end_frac, axis = self._fitted_to_full_extent()
        frac = end_frac / 2.0
        expected = widget._x_min + frac * axis
        value, x = widget._sample_at_frac(1, frac)
        self.assertAlmostEqual(x, expected, delta=0.01 * axis)
        self.assertTrue(np.isfinite(value))
        # The reading is genuinely anchored to the pointer: the full-width mapping
        # would have answered from near the very start of the response instead, so
        # the two are far enough apart for this to be a real distinction.
        compressed = widget._x_min + frac * (float(widget._x_data[1][-1]) - widget._x_min)
        self.assertLess(compressed, 0.25 * expected, "the two mappings really differ here")

    async def test_the_readout_leaves_a_clipped_trace_blank_past_its_end(self) -> None:
        """Past the clip there is no response drawn, so none is reported.

        The command keeps reporting across the whole axis: the chart does not go
        quiet because one of its traces ended, and the response does not pin its
        last sample across the empty remainder either.
        """
        widget, end_frac, axis = self._fitted_to_full_extent()
        past = (end_frac + 1.0) / 2.0
        self.assertIsNone(widget._sample_at_frac(1, past), "no response is drawn there")
        command = widget._sample_at_frac(0, past)
        self.assertIsNotNone(command, "the command still spans the axis")
        self.assertAlmostEqual(command[1], widget._x_min + past * axis, delta=0.02 * axis)

    async def test_reference_series_is_never_clipped(self) -> None:
        """The command runs the whole test and must stay whole."""
        widget = self._build(divergence_limits=[None, 1440.0])
        self.assertEqual(len(widget._y_data[0]), len(self._command))
        self.assertLess(len(widget._y_data[1]), len(self._observed))

    async def test_clean_run_is_left_alone(self) -> None:
        """A joint that tracked its command is not clipped or reframed."""
        t = np.arange(600) / 600.0
        command = np.full_like(t, 90.0)
        with self._window.frame:
            with ui.VStack():
                widget = JointGraphWidget(
                    x_data=[t, t],
                    y_data=[command, command * 0.99],
                    data_colors=[0xFF999999, 0xFFFFFFFF],
                    header_count=2,
                    reference_count=1,
                    divergence_limits=[None, 720.0],
                )
        self.assertFalse(widget._diverged)
        self.assertEqual(len(widget._y_data[1]), len(command))
        self.assertAlmostEqual(widget._x_max, widget._x_data_max)
        self.assertAlmostEqual(widget._y_max, widget._y_data_max)

    async def test_fit_frame_view_restores_the_full_extent(self) -> None:
        """The framed view is a starting point, not a limit on what can be seen."""
        widget = self._build(divergence_limits=[None, 1440.0])
        widget._on_fit_frame_view()
        self.assertAlmostEqual(widget._x_max, widget._x_data_max)
        self.assertAlmostEqual(widget._y_max, widget._y_data_max)


class TestComputeBounds(omni.kit.test.AsyncTestCase):
    """Axis-range computation, which must never yield a non-finite range."""

    async def test_bounds_span_finite_data(self) -> None:
        """Bounds cover the min and max across all series."""
        lo, hi = JointGraphWidget._compute_bounds(
            [np.array([-2.0, 0.0, 1.0]), np.array([0.5, 4.0])], default=(-1.0, 1.0)
        )
        self.assertAlmostEqual(lo, -2.0)
        self.assertAlmostEqual(hi, 4.0)

    async def test_nan_samples_are_ignored(self) -> None:
        """A NaN sample does not poison the range, which would blank every series."""
        lo, hi = JointGraphWidget._compute_bounds([np.array([1.0, np.nan, 3.0])], default=(-1.0, 1.0))
        self.assertTrue(np.isfinite(lo))
        self.assertTrue(np.isfinite(hi))
        self.assertAlmostEqual(lo, 1.0)
        self.assertAlmostEqual(hi, 3.0)

    async def test_inf_samples_are_ignored(self) -> None:
        """Infinite samples are excluded from the range."""
        lo, hi = JointGraphWidget._compute_bounds([np.array([-np.inf, 2.0, np.inf])], default=(-1.0, 1.0))
        # Only 2.0 is finite, so the span collapses and is widened to [2.0, 3.0].
        self.assertAlmostEqual(lo, 2.0)
        self.assertAlmostEqual(hi, 3.0)

    async def test_all_non_finite_falls_back_to_default(self) -> None:
        """A wholly non-finite series yields the caller's default range."""
        bounds = JointGraphWidget._compute_bounds([np.array([np.nan, np.inf])], default=(-1.0, 1.0))
        self.assertEqual(bounds, (-1.0, 1.0))

    async def test_empty_input_falls_back_to_default(self) -> None:
        """No data yields the caller's default range."""
        self.assertEqual(JointGraphWidget._compute_bounds([], default=(0.0, 1.0)), (0.0, 1.0))
        self.assertEqual(JointGraphWidget._compute_bounds([np.array([])], default=(0.0, 1.0)), (0.0, 1.0))

    async def test_flat_series_is_widened(self) -> None:
        """A constant series is given a non-zero span so the plot has a usable range."""
        lo, hi = JointGraphWidget._compute_bounds([np.array([5.0, 5.0])], default=(-1.0, 1.0))
        self.assertAlmostEqual(lo, 5.0)
        self.assertGreater(hi - lo, 0.0)


class _FakeCaptureLayer:
    """Stand-in for the capture-layer rectangle the readout resolves the pointer against."""

    def __init__(self, screen_position_x: float = 100.0, screen_position_y: float = 40.0) -> None:
        self.screen_position_x = screen_position_x
        self.screen_position_y = screen_position_y
        self.computed_width = 201.0
        self.computed_height = 100.0


class _FakeLabel:
    """Stand-in for one of the readout table's labels."""

    def __init__(self) -> None:
        self.text = ""


def _build_fake_table(widget: JointGraphWidget) -> list[str]:
    """Lay out ``widget``'s readout table over fake labels, and return its column titles.

    Stands in for :meth:`JointGraphWidget._build_readout_table`, which needs a
    live window, and keeps the same bookkeeping so the real writer runs against
    the real layout.
    """
    titles, rows = widget._readout_layout()
    widget._readout_built_shape = widget._readout_shape()
    widget._readout_x_label = _FakeLabel()
    widget._readout_x_shown = ""
    widget._readout_row_names = [name for name, _ in rows]
    widget._readout_cells = [[(index, is_x, _FakeLabel()) for index, is_x in cells] for _, cells in rows]
    widget._readout_shown = [[""] * len(cells) for _, cells in rows]
    return titles


def _attach_readout(widget: JointGraphWidget) -> None:
    """Give ``widget`` a laid-out capture layer and an open readout, with the poll running."""
    widget._readout_frame = _FakeCaptureLayer()
    widget._readout_collapsed = False
    _build_fake_table(widget)
    # Stand-in for the per-frame subscription; the readout releases it on exit.
    widget._pointer_sub = object()


def _readout_rows(widget: JointGraphWidget) -> list[list[str]]:
    """Return the text of every cell in the readout table, row by row."""
    return [[label.text for _, _, label in cells] for cells in widget._readout_cells]


def _move_pointer(widget: JointGraphWidget, offset_x: float, offset_y: float = 10.0) -> list[list[str]]:
    """Put the pointer ``offset_x`` points into the plot and return the readout's cells.

    The pointer is read from the input system rather than handed to a callback,
    because omni.ui reports nothing but enter and exit on a plain hover.
    """
    frame = widget._readout_frame
    position = (frame.screen_position_x + offset_x, frame.screen_position_y + offset_y)
    with mock.patch.object(chart_widget, "_pointer_screen_pos", return_value=position):
        widget._update_readout()
    return _readout_rows(widget)


def _descendants(widget: ui.Widget) -> Iterator[ui.Widget]:
    """Yield every widget under ``widget``, depth first."""
    for child in ui.Inspector.get_children(widget):
        yield child
        yield from _descendants(child)


def _widget(
    x_data: list[np.ndarray],
    y_data: list[np.ndarray],
    *,
    group_num: int,
    group_visible: list[bool],
    legends: list[str],
    series_labels: list[str] | None = None,
    joint_labels: list[str] | None = None,
    x_window: tuple[float, float] = (0.0, 1.0),
    x_unit: str = "s",
    data_colors: list[int] | None = None,
) -> JointGraphWidget:
    """Build a JointGraphWidget with only the state the readout math reads.

    The readout is pure over the series data and the current view window, so the
    test skips ``__init__`` (and with it every ``omni.ui`` widget) and sets that
    state directly.
    """
    widget = JointGraphWidget.__new__(JointGraphWidget)
    widget._x_data = [np.asarray(a, dtype=float) for a in x_data]
    widget._y_data = [np.asarray(a, dtype=float) for a in y_data]
    # The caller hands in series already at the extent they are drawn at, so
    # nothing here was clipped and the footer has no divergence note to make.
    widget._diverged = False
    widget._group_num = group_num
    widget._group_visible = list(group_visible)
    widget._legends = list(legends)
    widget._series_labels = list(series_labels or [])
    widget._joint_labels = list(joint_labels or [])
    widget._x_min, widget._x_max = x_window
    widget._y_min, widget._y_max = -1.0, 1.0
    widget._x_data_min, widget._x_data_max = x_window
    widget._y_data_min, widget._y_data_max = -1.0, 1.0
    widget._x_unit = x_unit
    widget._y_unit = "\u00b0"
    widget._shared_timeline = None
    widget._readout_collapsed = True
    widget._readout_frame = None
    widget._readout_section = None
    widget._readout_x_label = None
    widget._readout_x_shown = ""
    widget._readout_cells = []
    widget._readout_shown = []
    widget._readout_built_shape = None
    widget._pointer_sub = None
    widget._data_colors = list(data_colors or [])
    widget._header_count = len(group_visible)
    return widget


class TestSeriesValuesAtFrac(omni.kit.test.AsyncTestCase):
    """Every visible series is reported, not just whichever one wins the pointer."""

    _TIMES = np.linspace(0.0, 1.0, 11)

    def _command_and_observed(self, **kwargs: object) -> JointGraphWidget:
        """Two joints x (command, observed), the Position chart's layout."""
        command_a = self._TIMES * 10.0
        command_b = self._TIMES * 20.0
        return _widget(
            [self._TIMES] * 4,
            [command_a, command_b, command_a - 1.0, command_b - 2.0],
            group_num=2,
            group_visible=[True, True],
            legends=["Command Joint", "Observed Joint"],
            series_labels=["Command (j0)", "Command (j1)", "Observed (j0)", "Observed (j1)"],
            **kwargs,
        )

    async def test_reports_command_and_observed_for_every_joint(self) -> None:
        """The bug: only the Command series was ever reported."""
        readings = self._command_and_observed()._series_values_at_frac(0.5)
        self.assertAlmostEqual(readings[0][2], 0.5, places=5)
        self.assertEqual(
            [label for label, _, _ in readings],
            ["Command (j0)", "Command (j1)", "Observed (j0)", "Observed (j1)"],
        )
        self.assertAlmostEqual(readings[0][1], 5.0, places=5)
        self.assertAlmostEqual(readings[1][1], 10.0, places=5)
        self.assertAlmostEqual(readings[2][1], 4.0, places=5)
        self.assertAlmostEqual(readings[3][1], 8.0, places=5)

    async def test_interpolates_between_samples(self) -> None:
        """A pointer between two samples reads an interpolated value."""
        readings = self._command_and_observed()._series_values_at_frac(0.55)
        self.assertAlmostEqual(readings[0][2], 0.55, places=5)
        self.assertAlmostEqual(readings[0][1], 5.5, places=5)

    async def test_pointer_resolves_in_index_space_not_time(self) -> None:
        """Unevenly sampled x must read where the trace is drawn, not where time falls.

        ``ui.Plot`` receives only y values and spreads them evenly across the rect
        it is given, so the dt sweep's logarithmic timesteps are drawn at uniform
        spacing.  A time-linear mapping would report the wrong level entirely.
        """
        widget = _widget(
            [np.array([1.0, 2.0, 4.0, 8.0, 16.0])],
            [np.array([10.0, 20.0, 30.0, 40.0, 50.0])],
            group_num=1,
            group_visible=[True],
            legends=["settle"],
            x_window=(1.0, 16.0),
            x_unit="ms",
        )
        readings = widget._series_values_at_frac(0.5)
        # Midpoint of five samples is the third one, at 4 ms.  Interpolating
        # linearly in time would land near 8.5 ms and report ~40 instead.
        self.assertAlmostEqual(readings[0][2], 4.0, places=5)
        self.assertAlmostEqual(readings[0][1], 30.0, places=5)

    async def test_edges_report_the_first_and_last_drawn_sample(self) -> None:
        """The viewport edges map onto the end samples without running off the array."""
        widget = self._command_and_observed()
        left = widget._series_values_at_frac(0.0)
        right = widget._series_values_at_frac(1.0)
        self.assertAlmostEqual(left[0][2], 0.0, places=5)
        self.assertAlmostEqual(left[0][1], 0.0, places=5)
        self.assertAlmostEqual(right[0][2], 1.0, places=5)
        self.assertAlmostEqual(right[0][1], 10.0, places=5)

    async def test_pointer_outside_the_viewport_is_clamped(self) -> None:
        """A fraction beyond either edge clamps rather than indexing out of range."""
        widget = self._command_and_observed()
        self.assertEqual(widget._series_values_at_frac(-0.5), widget._series_values_at_frac(0.0))
        self.assertEqual(widget._series_values_at_frac(1.5), widget._series_values_at_frac(1.0))

    async def test_hidden_legend_group_is_skipped(self) -> None:
        """Unchecking a legend group drops its series from the readout."""
        widget = self._command_and_observed()
        widget._group_visible = [True, False]
        readings = widget._series_values_at_frac(0.5)
        self.assertEqual([label for label, _, _ in readings], ["Command (j0)", "Command (j1)"])

    def _full_and_short(self) -> JointGraphWidget:
        """A full-run trace beside one that stops a fifth of the way across."""
        short = np.linspace(0.0, 0.2, 5)
        return _widget(
            [self._TIMES, short],
            [self._TIMES * 10.0, short * 10.0],
            group_num=1,
            group_visible=[True, True],
            legends=["Full", "Truncated"],
            series_labels=["Full", "Truncated"],
        )

    async def test_short_series_is_read_within_its_own_span(self) -> None:
        """A trace that ends early is read across the stretch of the axis it covers.

        It is drawn only over the first fifth of the width, so the pointer has to
        be inverted over that fifth: a fifth of the way across is that trace's
        midpoint, not its first sample.
        """
        readings = self._full_and_short()._series_values_at_frac(0.1)
        self.assertEqual([label for label, _, _ in readings], ["Full", "Truncated"])
        self.assertAlmostEqual(readings[0][1], 1.0, places=5)
        # Midpoint of the truncated trace is 0.1 s, where it holds 1.0.
        self.assertAlmostEqual(readings[1][2], 0.1, places=5)
        self.assertAlmostEqual(readings[1][1], 1.0, places=5)

    async def test_short_series_is_not_reported_past_its_end(self) -> None:
        """Beyond a truncated trace there is nothing drawn, so it reports nothing.

        Pinning its last sample across the empty remainder of the chart would put
        a number under the pointer that no trace is under.
        """
        readings = self._full_and_short()._series_values_at_frac(0.5)
        self.assertEqual([label for label, _, _ in readings], ["Full"])
        self.assertAlmostEqual(readings[0][1], 5.0, places=5)

    async def test_series_with_no_samples_in_window_is_skipped(self) -> None:
        """A trace zoomed entirely out of view is neither drawn nor reported."""
        short = np.linspace(0.0, 0.2, 5)
        widget = _widget(
            [self._TIMES, short],
            [self._TIMES * 10.0, short * 10.0],
            group_num=1,
            group_visible=[True, True],
            legends=["Full", "Truncated"],
            series_labels=["Full", "Truncated"],
            x_window=(0.5, 1.0),
        )
        readings = widget._series_values_at_frac(0.75)
        self.assertEqual([label for label, _, _ in readings], ["Full"])

    async def test_single_sample_series_is_skipped(self) -> None:
        """A series with fewer than two samples is not drawn and not reported."""
        widget = _widget(
            [np.array([0.5]), self._TIMES],
            [np.array([3.0]), self._TIMES * 10.0],
            group_num=1,
            group_visible=[True, True],
            legends=["A", "B"],
            series_labels=["Single", "Full"],
        )
        readings = widget._series_values_at_frac(0.5)
        self.assertEqual([label for label, _, _ in readings], ["Full"])

    async def test_labels_fall_back_to_legends(self) -> None:
        """Without explicit series labels the legend group names the series."""
        widget = self._command_and_observed()
        widget._series_labels = []
        readings = widget._series_values_at_frac(0.5)
        self.assertEqual(
            [label for label, _, _ in readings],
            ["Command Joint 0", "Command Joint 1", "Observed Joint 0", "Observed Joint 1"],
        )


class TestSeriesPlacementReadback(omni.kit.test.AsyncTestCase):
    """The readout inverts the placement the traces were drawn with.

    ``_build_plots`` gives a series clipped at divergence only the stretch of the
    width its samples cover, so the readout has to invert that same placement.
    Resolving against the full width instead reports a sample from a completely
    different time under the pointer.
    """

    async def test_placement_matches_the_span_the_series_covers(self) -> None:
        """A series over the first half of the window is placed on the first half."""
        half = np.linspace(0.0, 0.5, 6)
        widget = _widget(
            [half],
            [half * 10.0],
            group_num=1,
            group_visible=[True],
            legends=["Clipped"],
            series_labels=["Clipped"],
        )
        _start, _stop, offset, width = widget._series_placement(0)
        self.assertAlmostEqual(offset, 0.0, places=6)
        self.assertAlmostEqual(width, 0.5, places=6)

    async def test_reading_agrees_with_the_time_axis(self) -> None:
        """On evenly sampled data the placement makes the readout x-accurate.

        A clipped trace and a full-length one drawn over the same window report the
        same x under the same pixel, which is only true because each is inverted
        over the stretch it was drawn across.
        """
        full = np.linspace(0.0, 1.0, 21)
        half = np.linspace(0.0, 0.5, 11)
        widget = _widget(
            [full, half],
            [full * 10.0, half * 10.0],
            group_num=1,
            group_visible=[True, True],
            legends=["Full", "Clipped"],
            series_labels=["Full", "Clipped"],
        )
        readings = widget._series_values_at_frac(0.25)
        self.assertEqual([label for label, _, _ in readings], ["Full", "Clipped"])
        self.assertAlmostEqual(readings[0][2], 0.25, places=5)
        self.assertAlmostEqual(readings[1][2], 0.25, places=5)

    async def test_series_offset_from_the_left_edge_is_not_read_before_it_starts(self) -> None:
        """A series starting mid-window is not reported left of where it is drawn."""
        late = np.linspace(0.5, 1.0, 11)
        widget = _widget(
            [late],
            [late * 10.0],
            group_num=1,
            group_visible=[True],
            legends=["Late"],
            series_labels=["Late"],
        )
        _start, _stop, offset, width = widget._series_placement(0)
        self.assertAlmostEqual(offset, 0.5, places=6)
        self.assertAlmostEqual(width, 0.5, places=6)
        self.assertEqual(widget._series_values_at_frac(0.25), [])
        readings = widget._series_values_at_frac(0.75)
        self.assertAlmostEqual(readings[0][2], 0.75, places=5)


class TestHoverReadoutText(omni.kit.test.AsyncTestCase):
    """The multi-line readout text: one heading and a labeled line per series."""

    _TIMES = np.linspace(0.0, 1.0, 11)

    def _widget(self) -> JointGraphWidget:
        return _widget(
            [self._TIMES, self._TIMES],
            [self._TIMES * 10.0, self._TIMES * 10.0 - 1.0],
            group_num=1,
            group_visible=[True, True],
            legends=["Command Joint", "Observed Joint"],
            series_labels=["Command (j0)", "Observed (j0)"],
        )

    async def test_readout_names_each_series_with_time_and_units(self) -> None:
        """The readout leads with the hovered time, then one labeled line per series."""
        lines = self._widget()._readout_text(0.5).splitlines()
        self.assertEqual(lines[0], "x = 0.500 s")
        self.assertIn("Command (j0)", lines[2])
        self.assertIn("5.000 \u00b0", lines[2])
        self.assertIn("Observed (j0)", lines[3])
        self.assertIn("4.000 \u00b0", lines[3])

    async def test_readout_is_empty_when_nothing_is_visible(self) -> None:
        """Hovering with every series hidden shows no tooltip rather than a bare time."""
        widget = self._widget()
        widget._group_visible = [False, False]
        self.assertEqual(widget._readout_text(0.5), "")

    async def test_series_at_different_x_are_labeled_individually(self) -> None:
        """Series that do not share a timeline must not share one x label.

        The dt sweep drops levels per joint, so two traces spanning the same window
        can sit at different timesteps at the same pixel; a single heading would
        attribute one joint's value to another joint's timestep.
        """
        widget = _widget(
            [np.array([1.0, 2.0, 16.0]), np.array([1.0, 8.0, 16.0])],
            [np.array([10.0, 20.0, 30.0]), np.array([11.0, 21.0, 31.0])],
            group_num=1,
            group_visible=[True, True],
            legends=["j0", "j1"],
            series_labels=["j0", "j1"],
            x_window=(1.0, 16.0),
            x_unit="ms",
        )
        text = widget._readout_text(0.5)
        # No shared heading; each line carries the timestep it was read at.
        self.assertFalse(text.splitlines()[0].startswith("x = "))
        self.assertIn("@ x = 2.000 ms", text)
        self.assertIn("@ x = 8.000 ms", text)

    async def test_shared_x_is_stated_once(self) -> None:
        """Traces on a common timeline keep the single heading, not per-line noise."""
        text = self._widget()._readout_text(0.5)
        self.assertEqual(text.splitlines()[0], "x = 0.500 s")
        self.assertNotIn("@ x =", text)


_TIMES = np.linspace(0.0, 1.0, 11)


def _position_chart(joints: int, *, group_visible: list[bool] | None = None, **kwargs: object) -> JointGraphWidget:
    """N joints x (command, observed), the Position chart's layout."""
    names = [f"j{i}" for i in range(joints)]
    return _widget(
        [_TIMES] * (2 * joints),
        [_TIMES * float(i + 1) for i in range(2 * joints)],
        group_num=joints,
        group_visible=group_visible or [True, True],
        legends=["Command Joint", "Observed Joint"],
        series_labels=[f"Command ({n})" for n in names] + [f"Observed ({n})" for n in names],
        joint_labels=names,
        **kwargs,
    )


def _dt_sweep_chart(*, shared_levels: bool) -> JointGraphWidget:
    """Two joints swept over dt levels, each joint its own legend group.

    The x window is the extent of the data, as a fitted chart's is, so the joint
    reaching furthest spans the full width.  With the levels unshared that leaves
    the other joint drawn across only its own stretch of the axis.
    """
    levels = [np.array([1.0, 2.0, 4.0]), np.array([1.0, 2.0, 4.0] if shared_levels else [1.0, 4.0, 16.0])]
    return _widget(
        levels,
        [np.array([10.0, 20.0, 30.0]), np.array([11.0, 21.0, 31.0])],
        group_num=1,
        group_visible=[True, True],
        legends=["j0", "j1"],
        series_labels=["j0", "j1"],
        x_window=(1.0, 4.0 if shared_levels else 16.0),
        x_unit="ms",
    )


class TestReadoutTracksThePointer(omni.kit.test.AsyncTestCase):
    """Pointer position -> readout cells, the step the pure helpers do not cover."""

    def _widget(self, joints: int = 1) -> JointGraphWidget:
        widget = _position_chart(joints)
        _attach_readout(widget)
        return widget

    async def test_values_change_as_the_pointer_moves(self) -> None:
        """The bug: the readout froze at wherever the pointer entered the plot.

        omni.ui builds a tooltip once and caches it, so a readout carried by one
        showed the same figures for the whole hover no matter where the pointer
        went.  Three pointer positions must give three readings.
        """
        widget = self._widget()
        left = _move_pointer(widget, 0.0)
        middle = _move_pointer(widget, 100.0)
        right = _move_pointer(widget, 200.0)
        self.assertEqual(left, [["0.000 \u00b0", "0.000 \u00b0"]])
        self.assertEqual(middle, [["0.500 \u00b0", "1.000 \u00b0"]])
        self.assertEqual(right, [["1.000 \u00b0", "2.000 \u00b0"]])

    async def test_the_hovered_x_is_reported_once_and_follows_the_pointer(self) -> None:
        """Traces on a common timeline head the table with one x, which tracks the pointer."""
        widget = self._widget()
        _move_pointer(widget, 0.0)
        self.assertEqual(widget._readout_x_label.text, "x = 0.000 s")
        _move_pointer(widget, 100.0)
        self.assertEqual(widget._readout_x_label.text, "x = 0.500 s")

    async def test_every_visible_series_is_reported_for_a_wide_selection(self) -> None:
        """The bug the user hit: joints dropped off the readout for want of room.

        Ten joints is twenty series.  Every one of them has a cell, and every cell
        holds a figure -- the readout is laid out for the selection rather than
        fitted into a space the selection has to fit in.
        """
        widget = self._widget(joints=10)
        rows = _move_pointer(widget, 100.0)
        self.assertEqual(len(rows), 10)
        for row in rows:
            self.assertEqual(len(row), 2)
            for cell in row:
                self.assertTrue(cell.endswith(" \u00b0"), cell)
                self.assertNotEqual(cell, widget.READOUT_NO_VALUE)

    async def test_readout_is_anchored_on_the_plots_own_screen_position(self) -> None:
        """A plot moved across the screen reads the same for the same offset into it."""
        widget = self._widget()
        before = _move_pointer(widget, 50.0)
        widget._readout_frame.screen_position_x += 400.0
        widget._readout_frame.screen_position_y += 120.0
        self.assertEqual(_move_pointer(widget, 50.0), before)

    async def test_pointer_leaving_the_plot_clears_the_readout(self) -> None:
        """No reading may be left in the table for a pointer that is elsewhere."""
        widget = self._widget()
        self.assertNotEqual(_move_pointer(widget, 100.0), [["", ""]])
        self.assertEqual(_move_pointer(widget, widget._readout_frame.computed_width + 30.0), [["", ""]])
        self.assertEqual(widget._readout_x_label.text, "")
        self.assertIsNone(widget._pointer_sub)

    async def test_pointer_leaving_above_or_below_the_plot_also_ends_the_hover(self) -> None:
        """The plot is left vertically as easily as horizontally."""
        widget = self._widget()
        _move_pointer(widget, 100.0)
        self.assertEqual(_move_pointer(widget, 100.0, widget._readout_frame.computed_height + 20.0), [["", ""]])
        self.assertIsNone(widget._pointer_sub)

    async def test_hover_exit_clears_the_readout(self) -> None:
        """The leave event ends the hover and stops the poll with it."""
        widget = self._widget()
        _move_pointer(widget, 100.0)
        widget._on_readout_hovered(False)
        self.assertEqual(_readout_rows(widget), [["", ""]])
        self.assertIsNone(widget._pointer_sub)

    async def test_chart_with_no_drawn_series_reports_nothing(self) -> None:
        """An empty chart builds no cells, so it can report no figures."""
        widget = _position_chart(1, group_visible=[False, False])
        _attach_readout(widget)
        self.assertEqual(_move_pointer(widget, 100.0), [])
        self.assertEqual(widget._readout_x_label.text, "")

    async def test_unlaid_out_plot_reports_nothing(self) -> None:
        """Before layout there is no width to resolve the pointer against."""
        widget = self._widget()
        widget._readout_frame.computed_width = 0.0
        self.assertEqual(_move_pointer(widget, 0.0), [["", ""]])

    async def test_unlocatable_pointer_reports_nothing(self) -> None:
        """With no app window to read the pointer from, the readout cannot guess."""
        widget = self._widget()
        _move_pointer(widget, 100.0)
        with mock.patch.object(chart_widget, "_pointer_screen_pos", return_value=None):
            widget._update_readout()
        self.assertEqual(_readout_rows(widget), [["", ""]])

    async def test_unchanged_text_is_not_rewritten(self) -> None:
        """A pointer resting between two samples must not rewrite the cells every frame."""
        widget = self._widget()
        _move_pointer(widget, 100.0)
        widget._readout_cells[0][0][2].text = "sentinel"
        _move_pointer(widget, 100.0)
        self.assertEqual(widget._readout_cells[0][0][2].text, "sentinel")

    async def test_nothing_is_polled_while_the_readout_is_closed(self) -> None:
        """A closed readout has no cell on screen, so a hover must not start a poll."""
        widget = self._widget()
        widget._pointer_sub = None
        widget._readout_collapsed = True
        widget._on_readout_hovered(True)
        self.assertIsNone(widget._pointer_sub)

    async def test_opening_the_readout_starts_reading_the_pointer(self) -> None:
        """No hover event follows the click that opened it, so opening has to start the poll."""
        widget = self._widget()
        widget._pointer_sub = None
        widget._readout_collapsed = True
        frame = widget._readout_frame
        inside = (frame.screen_position_x + 100.0, frame.screen_position_y + 10.0)
        with mock.patch.object(chart_widget, "_pointer_screen_pos", return_value=inside):
            widget._on_readout_collapsed(False)
        self.assertFalse(widget._readout_collapsed)
        self.assertIsNotNone(widget._pointer_sub)
        self.assertEqual(_readout_rows(widget), [["0.500 \u00b0", "1.000 \u00b0"]])
        widget._on_readout_collapsed(True)  # release the per-frame subscription

    async def test_closing_the_readout_stops_reading_the_pointer(self) -> None:
        """Nothing may be polled or left showing once the section is closed."""
        widget = self._widget()
        _move_pointer(widget, 100.0)
        widget._on_readout_collapsed(True)
        self.assertTrue(widget._readout_collapsed)
        self.assertIsNone(widget._pointer_sub)
        self.assertEqual(_readout_rows(widget), [["", ""]])


class TestReadoutLayout(omni.kit.test.AsyncTestCase):
    """The rows and columns the readout is laid out in, for one joint and for many."""

    async def test_one_row_per_joint_and_one_column_per_legend_group(self) -> None:
        """A joint's Command and Observed figures are read side by side, named once."""
        titles, rows = _position_chart(3)._readout_layout()
        self.assertEqual(titles, ["Command Joint", "Observed Joint"])
        self.assertEqual([name for name, _ in rows], ["j0", "j1", "j2"])
        self.assertEqual([[index for index, _ in cells] for _, cells in rows], [[0, 3], [1, 4], [2, 5]])

    async def test_a_single_joint_still_gets_its_own_row(self) -> None:
        """The layout reads the same way at either extreme of the selection."""
        titles, rows = _position_chart(1)._readout_layout()
        self.assertEqual(titles, ["Command Joint", "Observed Joint"])
        self.assertEqual(rows, [("j0", [(0, False), (1, False)])])

    async def test_ten_joints_are_ten_rows_and_no_more_columns(self) -> None:
        """Height follows the selection; width does not, so nothing is squeezed out."""
        titles, rows = _position_chart(10)._readout_layout()
        self.assertEqual(len(titles), 2)
        self.assertEqual([name for name, _ in rows], [f"j{i}" for i in range(10)])
        self.assertTrue(all(len(cells) == 2 for _, cells in rows))

    async def test_hiding_a_legend_group_drops_its_column(self) -> None:
        """An unchecked group is not drawn, so it is not a column either."""
        titles, rows = _position_chart(2, group_visible=[True, False])._readout_layout()
        self.assertEqual(titles, ["Command Joint"])
        self.assertEqual(rows, [("j0", [(0, False)]), ("j1", [(1, False)])])

    async def test_effort_chart_is_one_column_of_joints(self) -> None:
        """A chart with a single legend group needs no second column."""
        widget = _widget(
            [_TIMES] * 3,
            [_TIMES, _TIMES * 2.0, _TIMES * 3.0],
            group_num=3,
            group_visible=[True],
            legends=["Effort"],
            series_labels=["Effort (j0)", "Effort (j1)", "Effort (j2)"],
            joint_labels=["j0", "j1", "j2"],
        )
        titles, rows = widget._readout_layout()
        self.assertEqual(titles, ["Effort"])
        self.assertEqual([name for name, _ in rows], ["j0", "j1", "j2"])

    async def test_a_joint_with_nothing_to_report_is_not_given_a_row(self) -> None:
        """A row of nothing but dashes is noise; the joint is simply not listed."""
        single = np.zeros(1)
        widget = _widget(
            [_TIMES] * 4,
            [_TIMES, single, _TIMES, single],
            group_num=2,
            group_visible=[True, True],
            legends=["Command Joint", "Observed Joint"],
            joint_labels=["kept", "not drawn"],
        )
        _titles, rows = widget._readout_layout()
        self.assertEqual([name for name, _ in rows], ["kept"])

    async def test_series_on_their_own_timelines_get_an_x_column_each(self) -> None:
        """The dt sweep drops levels per joint, so no single x can head the table.

        Reporting one x for all of them would attribute one joint's value to
        another joint's timestep.
        """
        widget = _dt_sweep_chart(shared_levels=False)
        titles, rows = widget._readout_layout()
        self.assertEqual(titles, ["x", "Value"])
        self.assertEqual(rows, [("j0", [(0, True), (0, False)]), ("j1", [(1, True), (1, False)])])
        _attach_readout(widget)
        # A tenth of the way across: halfway along j0's stretch, which ends at 4 ms
        # of a 16 ms axis, and a tenth of the way along j1's full-width one.  The
        # two joints are at different timesteps there, which is the point.
        self.assertEqual(_move_pointer(widget, 20.0), [["2.000 ms", "20.000 \u00b0"], ["1.600 ms", "13.000 \u00b0"]])
        # The per-row x replaces the heading rather than joining it.
        self.assertEqual(widget._readout_x_label.text, "")

    async def test_a_joint_is_left_blank_where_its_own_trace_has_ended(self) -> None:
        """Past j0's last level there is no j0 trace to read, so it reports no value.

        The other joint keeps reporting: one trace ending is not the whole
        readout ending.  Before the traces were clipped to their own stretch of
        the axis every series spanned the full width, and this position would
        have been answered with j0's values from a part of the chart it is not
        drawn in.
        """
        widget = _dt_sweep_chart(shared_levels=False)
        _attach_readout(widget)
        self.assertEqual(_move_pointer(widget, 100.0), [["-", "-"], ["4.000 ms", "21.000 \u00b0"]])

    async def test_series_sharing_a_timeline_are_headed_by_one_x(self) -> None:
        """A dt sweep that kept every level for every joint needs no x column."""
        widget = _dt_sweep_chart(shared_levels=True)
        titles, rows = widget._readout_layout()
        self.assertEqual(titles, ["Value"])
        self.assertEqual(rows, [("j0", [(0, False)]), ("j1", [(1, False)])])
        _attach_readout(widget)
        _move_pointer(widget, 100.0)
        self.assertEqual(widget._readout_x_label.text, "x = 2.000 ms")

    async def test_nothing_is_laid_out_for_a_chart_with_no_visible_series(self) -> None:
        """An empty chart reports no rows, so its section shows its header and nothing else."""
        self.assertEqual(_position_chart(2, group_visible=[False, False])._readout_layout(), ([], []))

    async def test_rows_fall_back_to_series_labels_without_joint_names(self) -> None:
        """A caller that named its series but not its joints still gets named rows."""
        widget = _position_chart(2)
        widget._joint_labels = []
        _titles, rows = widget._readout_layout()
        self.assertEqual([name for name, _ in rows], ["Command (j0)", "Command (j1)"])


def _unrenderable(text: str) -> list[str]:
    """Return the characters of ``text`` the app's default font has no glyph for.

    Label text is ASCII plus the degree sign, the same rule ``NA_CELL_TEXT`` and
    the results rows follow: anything else -- an em dash, an ellipsis, a
    typographic quote -- is drawn as a stray "?" missing-glyph box.
    """
    return sorted({ch for ch in text if ord(ch) > 127 and ch != "\u00b0"})


class TestReadoutTextIsRenderable(omni.kit.test.AsyncTestCase):
    """Every string the readout shows has to be one the app's font can draw.

    The bug: the section header separated its title from its prompt with an em
    dash, so it read "Value Readout ? hover the chart to read every joint".
    """

    def _shown(self, widget: JointGraphWidget) -> list[str]:
        """Return every string the readout puts on screen for a hovered chart."""
        _attach_readout(widget)
        titles, rows = widget._readout_layout()
        cells = [cell for row in _move_pointer(widget, 100.0) for cell in row]
        return [
            widget.READOUT_TITLE,
            widget.READOUT_NO_VALUE,
            widget._readout_x_label.text,
            *titles,
            *[name for name, _ in rows],
            *cells,
        ]

    async def test_everything_the_readout_shows_is_renderable(self) -> None:
        """Header, column titles, joint names, the x heading and every figure."""
        for widget in (_position_chart(2), _dt_sweep_chart(shared_levels=False)):
            for text in self._shown(widget):
                self.assertEqual(_unrenderable(text), [], f"unrenderable characters in {text!r}")

    async def test_the_check_rejects_the_character_that_caused_the_bug(self) -> None:
        """The guard is worth nothing if it passes everything, so prove it fails."""
        self.assertEqual(_unrenderable("Value Readout \u2014 hover"), ["\u2014"])
        self.assertEqual(_unrenderable("5.000 \u00b0"), [], "the degree sign is the one exception")


class TestReadoutShape(omni.kit.test.AsyncTestCase):
    """What may rebuild the readout table, and what may not.

    The real stability invariant: the table is rebuilt by the deliberate actions
    that change the selection, and never by pointer movement, which would resize
    the section while it is being read.
    """

    async def test_pointer_movement_never_changes_the_shape(self) -> None:
        """Nothing about where the pointer is may change the rows or columns."""
        widget = _position_chart(3)
        shape = widget._readout_shape()
        for frac in (0.0, 0.13, 0.5, 0.87, 1.0):
            widget._series_readings_at_frac(frac)
            self.assertEqual(widget._readout_shape(), shape)

    async def test_a_chart_on_mixed_timelines_keeps_one_shape_across_the_plot(self) -> None:
        """Whether the series share an x is decided from the data, not from the samples.

        Two dt-sweep traces can agree at one pixel and disagree at the next.  Were
        that what chose the layout, the table would gain and lose its x column as
        the pointer crossed the plot.
        """
        widget = _dt_sweep_chart(shared_levels=False)
        at_left = widget._series_readings_at_frac(0.0)
        self.assertEqual(at_left[0][2], at_left[1][2], "the traces start on the same timestep")
        shape = widget._readout_shape()
        for frac in (0.0, 0.5, 1.0):
            widget._series_readings_at_frac(frac)
            self.assertEqual(widget._readout_shape(), shape)
            self.assertEqual(widget._readout_layout()[0], ["x", "Value"])

    async def test_hiding_a_legend_group_changes_the_shape(self) -> None:
        """A legend toggle has to rebuild the table, or a column would go stale."""
        widget = _position_chart(2)
        before = widget._readout_shape()
        widget._group_visible = [True, False]
        self.assertNotEqual(widget._readout_shape(), before)

    async def test_zooming_a_series_out_of_the_window_changes_the_shape(self) -> None:
        """A series that stops being drawn has to stop being a cell."""
        short = np.linspace(0.0, 0.2, 5)
        widget = _widget(
            [_TIMES, short],
            [_TIMES, short],
            group_num=1,
            group_visible=[True, True],
            legends=["Full", "Truncated"],
        )
        before = widget._readout_shape()
        widget._x_min, widget._x_max = 0.5, 1.0
        self.assertNotEqual(widget._readout_shape(), before)


async def _settle(frames: int = 8) -> None:
    """Let omni.ui lay out and build what the last change asked for."""
    for _ in range(frames):
        await omni.kit.app.get_app().next_update_async()


async def _build_chart(title: str, joints: int = 1, **kwargs: object) -> tuple[JointGraphWidget, ui.Window]:
    """Build a whole chart card, readout section and all, in a live window."""
    widget = _position_chart(joints, data_colors=[0xFFB44FE0, 0xFF8A2FC0], **kwargs)
    window = ui.Window(title, width=760, height=700)
    with window.frame:
        widget._build()
    await _settle()
    return widget, window


async def _expand_readout(widget: JointGraphWidget) -> None:
    """Open the readout section the way clicking its header does."""
    widget._readout_section.collapsed = False
    await _settle()


class TestReadoutSection(omni.kit.test.AsyncTestCase):
    """The collapsible section: its default, its persistence, and its one-readout guarantee."""

    async def test_the_readout_starts_closed(self) -> None:
        """A ten-joint table is tall, so the panel must not open with one by default."""
        widget, window = await _build_chart("gt_readout_default_test")
        self.assertTrue(widget._readout_collapsed)
        self.assertTrue(widget._readout_section.collapsed)
        self.assertEqual(_readout_rows(widget), [["", ""]], "a closed section shows no values")
        window.destroy()

    async def test_the_header_says_what_the_readout_is_for(self) -> None:
        """Closed by default, so the header is the only thing telling the user it exists."""
        widget, window = await _build_chart("gt_readout_header_test")
        self.assertEqual(widget._readout_section.title, widget.READOUT_TITLE)
        self.assertIn("hover", widget.READOUT_TITLE.lower())
        window.destroy()

    async def test_opening_the_readout_is_remembered_across_a_plot_rebuild(self) -> None:
        """A slider drag rebuilds the plot every frame; it may not close the readout.

        The state is the widget's, not the rebuilt frame's, which is what keeps a
        section the user opened from springing shut under a drag.
        """
        widget, window = await _build_chart("gt_readout_persist_open_test")
        await _expand_readout(widget)
        self.assertFalse(widget._readout_collapsed)
        widget._plot_frame.rebuild()
        await _settle()
        self.assertFalse(widget._readout_collapsed)
        self.assertFalse(widget._readout_section.collapsed)
        self.assertNotEqual(widget._readout_cells, [], "the table survives the rebuild")
        window.destroy()

    async def test_a_closed_readout_stays_closed_across_a_plot_rebuild(self) -> None:
        """The default must not be re-imposed either -- or reasserted as a surprise."""
        widget, window = await _build_chart("gt_readout_persist_closed_test")
        widget._plot_frame.rebuild()
        await _settle()
        self.assertTrue(widget._readout_collapsed)
        self.assertTrue(widget._readout_section.collapsed)
        window.destroy()

    async def test_the_state_survives_a_rebuild_of_the_whole_card(self) -> None:
        """Rebuilding the card recreates the section, which has to come back as it was."""
        widget, window = await _build_chart("gt_readout_persist_card_test")
        await _expand_readout(widget)
        with window.frame:
            widget._build()
        await _settle()
        self.assertFalse(widget._readout_section.collapsed)
        window.destroy()

    async def test_every_visible_series_is_shown_when_expanded(self) -> None:
        """The user's bug: joints missing from the readout when several are selected."""
        widget, window = await _build_chart("gt_readout_expanded_rows_test", joints=4)
        await _expand_readout(widget)
        self.assertEqual(len(widget._readout_cells), 4)
        self.assertTrue(all(len(cells) == 2 for cells in widget._readout_cells))
        labels = [child.text for child in _descendants(window.frame) if isinstance(child, ui.Label)]
        for joint in range(4):
            self.assertIn(f"j{joint}", labels)
        self.assertIn("Command Joint", labels)
        self.assertIn("Observed Joint", labels)
        window.destroy()

    async def test_an_empty_chart_builds_no_readout_table(self) -> None:
        """With nothing drawn the section shows its header and no figures at all."""
        widget, window = await _build_chart("gt_readout_empty_test", group_visible=[False, False])
        await _expand_readout(widget)
        self.assertEqual(widget._readout_cells, [])
        self.assertIsNone(widget._readout_x_label)
        window.destroy()

    async def test_hiding_a_legend_group_rebuilds_the_table(self) -> None:
        """A dropped column has to leave the readout, not go stale in it."""
        widget, window = await _build_chart("gt_readout_relayout_test", joints=2)
        await _expand_readout(widget)
        self.assertTrue(all(len(cells) == 2 for cells in widget._readout_cells))
        widget._on_legend_toggle(1, False)
        await _settle()
        self.assertTrue(all(len(cells) == 1 for cells in widget._readout_cells))
        window.destroy()

    async def test_the_readout_is_the_only_readout_the_chart_can_show(self) -> None:
        """Two popups over one chart is the regression; there has to be exactly one readout.

        The readout is a section of the chart's own, so no widget in the chart may
        carry a tooltip -- neither a static one nor a lazily built one, which
        omni.ui builds once and caches and which is what froze the readout's
        figures when it carried them.  ImGui's own per-plot tooltip is the other
        half of the guarantee and is asserted in
        :class:`TestReadoutPointerInteraction`.
        """
        widget, window = await _build_chart("gt_readout_single_popup_test")
        await _expand_readout(widget)
        for child in _descendants(window.frame):
            self.assertEqual(child.tooltip, "")
            self.assertFalse(child.has_tooltip_fn())
        self.assertTrue(widget._readout_frame.has_mouse_hovered_fn())
        window.destroy()

    async def test_rebuilding_the_plot_drops_a_running_pointer_poll(self) -> None:
        """A legend toggle or axis change replaces the layer the poll was reading."""
        widget, window = await _build_chart("gt_readout_rebuild_test")
        widget._pointer_sub = object()
        widget._plot_frame.rebuild()
        await _settle()
        self.assertIsNone(widget._pointer_sub)
        window.destroy()

    async def test_opening_the_readout_leaves_the_plot_where_it_was(self) -> None:
        """The readout sits below the plot, so opening it may not move what is hovered."""
        widget, window = await _build_chart("gt_readout_plot_position_test", joints=4)
        before = (widget._readout_frame.screen_position_y, widget._readout_frame.computed_height)
        await _expand_readout(widget)
        self.assertEqual(
            (widget._readout_frame.screen_position_y, widget._readout_frame.computed_height),
            before,
        )
        window.destroy()


class TestReadoutPointerInteraction(omni.kit.test.AsyncTestCase):
    """End to end: a real pointer hovering a real plot area."""

    async def _build(self, title: str, *, expanded: bool = True) -> tuple[JointGraphWidget, ui.Window]:
        widget, window = await _build_chart(title)
        if expanded:
            await _expand_readout(widget)
        return widget, window

    async def _hover_at(self, widget: JointGraphWidget, offset: float) -> tuple[str, list[list[str]]]:
        """Hover ``offset`` points into the plot and return the x heading and the cells."""
        import omni.kit.ui_test as ui_test

        frame = widget._readout_frame
        await ui_test.emulate_mouse_move(
            ui_test.Vec2(
                frame.screen_position_x + offset,
                frame.screen_position_y + frame.computed_height * 0.5,
            )
        )
        await _settle()
        heading = "" if widget._readout_x_label is None else widget._readout_x_label.text
        return heading, _readout_rows(widget)

    async def test_hovering_reads_out_where_the_pointer_is(self) -> None:
        """The regression: the readout's figures froze where the pointer entered the plot.

        A tooltip is built once and cached, so the numbers stopped agreeing with
        the pointer as soon as it moved.  A real hover across the plot has to
        report the samples drawn under the pointer at either edge and in between.
        """
        import omni.kit.ui_test as ui_test

        widget, window = await self._build("gt_readout_hover_probe")
        span = widget._readout_frame.computed_width - 1

        heading, rows = await self._hover_at(widget, 0.0)
        self.assertEqual(heading, "x = 0.000 s")
        self.assertEqual(rows, [["0.000 \u00b0", "0.000 \u00b0"]])

        heading, rows = await self._hover_at(widget, span)
        self.assertEqual(heading, "x = 1.000 s")
        self.assertEqual(rows, [["1.000 \u00b0", "2.000 \u00b0"]])

        # The emulated pointer lands on whole pixels, so allow the middle hover a
        # pixel of slack rather than demanding the exact centre sample.
        heading, _rows = await self._hover_at(widget, round(span * 0.5))
        hovered_x = float(heading.removeprefix("x = ").removesuffix(" s"))
        self.assertAlmostEqual(hovered_x, 0.5, delta=2.0 / span)

        await ui_test.emulate_mouse_move(ui_test.Vec2(5.0, 5.0))
        window.destroy()

    async def test_the_reading_holds_still_while_the_pointer_does(self) -> None:
        """A stationary pointer has to give a steady reading, not one flickering every frame."""
        import omni.kit.ui_test as ui_test

        widget, window = await self._build("gt_readout_steady_probe")
        heading, settled = await self._hover_at(widget, round((widget._readout_frame.computed_width - 1) * 0.75))
        self.assertNotEqual(settled, [["", ""]])
        for _ in range(30):
            await omni.kit.app.get_app().next_update_async()
            self.assertEqual(_readout_rows(widget), settled)
            self.assertEqual(widget._readout_x_label.text, heading)
        await ui_test.emulate_mouse_move(ui_test.Vec2(5.0, 5.0))
        window.destroy()

    async def test_the_readout_is_cleared_when_the_pointer_leaves(self) -> None:
        """No reading may be left in the table, and the per-frame poll has to end."""
        import omni.kit.ui_test as ui_test

        widget, window = await self._build("gt_readout_exit_probe")
        _heading, rows = await self._hover_at(widget, round((widget._readout_frame.computed_width - 1) * 0.5))
        self.assertNotEqual(rows, [["", ""]])
        await ui_test.emulate_mouse_move(ui_test.Vec2(5.0, 5.0))
        await _settle()
        self.assertEqual(_readout_rows(widget), [["", ""]])
        self.assertEqual(widget._readout_x_label.text, "")
        self.assertIsNone(widget._pointer_sub)
        window.destroy()

    async def test_a_closed_readout_does_no_per_frame_work_on_hover(self) -> None:
        """Nothing is on screen to read, so a hover must not subscribe to the update loop.

        A collapsed ``CollapsableFrame`` still runs its build once, so the cells
        exist; what must not happen is the per-frame poll that writes to them.
        """
        import omni.kit.ui_test as ui_test

        widget, window = await self._build("gt_readout_closed_hover_probe", expanded=False)
        frame = widget._readout_frame
        await ui_test.emulate_mouse_move(
            ui_test.Vec2(
                frame.screen_position_x + frame.computed_width * 0.5,
                frame.screen_position_y + frame.computed_height * 0.5,
            )
        )
        await _settle()
        self.assertIsNone(widget._pointer_sub)
        self.assertEqual(_readout_rows(widget), [["", ""]], "no reading is written while closed")
        await ui_test.emulate_mouse_move(ui_test.Vec2(5.0, 5.0))
        window.destroy()

    async def test_no_trace_is_hovered_so_imgui_shows_no_plot_tooltip(self) -> None:
        """The traces must stay outside ImGui's hover test, or ImGui pops up its own tooltip.

        ``ui.Plot`` renders through ImGui's plot primitive, which shows a tooltip
        naming a single unlabeled sample index and value as soon as ImGui counts
        it as hovered.  ImGui gates that on ``ItemHoverable``, which is strictly
        narrower than the ``IsItemHovered`` query behind ``mouse_hovered_fn``:
        omni.ui asks with the overlap and blocked-by-active-item checks relaxed,
        and every other condition is shared.  A trace that never reports hovered
        therefore cannot have emitted a tooltip, which makes this assertion the
        signal that fails if the traces are ever moved back into the same ImGui
        window as the pointer.
        """
        import omni.kit.ui_test as ui_test

        widget, window = await self._build("gt_readout_imgui_probe")

        plots = [w for w in _descendants(window.frame) if isinstance(w, ui.Plot)]
        self.assertGreater(len(plots), 1, "expected a plot per trace, so ImGui has something to disagree about")

        plot_hovers: list[list[bool]] = []
        for plot in plots:
            recorded: list[bool] = []
            plot.set_mouse_hovered_fn(recorded.append)
            plot_hovers.append(recorded)
        layer_hovers: list[bool] = []

        def _record_hover(hovered: bool) -> None:
            # Chained rather than replacing the widget's handler, so the hover runs
            # the readout for this check exactly as it does in use.
            layer_hovers.append(hovered)
            widget._on_readout_hovered(hovered)

        widget._readout_frame.set_mouse_hovered_fn(_record_hover)

        frame = widget._readout_frame
        centre = ui_test.Vec2(
            frame.screen_position_x + frame.computed_width * 0.5,
            frame.screen_position_y + frame.computed_height * 0.5,
        )
        await ui_test.emulate_mouse_move(ui_test.Vec2(5.0, 5.0))
        for _ in range(6):
            await omni.kit.app.get_app().next_update_async()
        await ui_test.emulate_mouse_move(centre)
        for _ in range(12):
            await omni.kit.app.get_app().next_update_async()

        # The traces still cover the pointer, so this is suppression and not a
        # plot that got laid out somewhere else or collapsed to nothing.
        for plot in plots:
            self.assertGreater(plot.computed_width, 0.0)
            self.assertLessEqual(plot.screen_position_x, centre.x)
            self.assertGreaterEqual(plot.screen_position_x + plot.computed_width, centre.x)

        self.assertTrue(any(layer_hovers), "the readout layer has to receive the pointer")
        for plot, recorded in zip(plots, plot_hovers):
            self.assertNotIn(True, recorded, "a trace was hovered, so ImGui showed its own plot tooltip")

        await ui_test.emulate_mouse_move(ui_test.Vec2(5.0, 5.0))
        window.destroy()


class TestLegendColors(omni.kit.test.AsyncTestCase):
    """Legend swatches must name the colours the traces are actually drawn in."""

    _TIMES = np.linspace(0.0, 1.0, 11)

    _PINK = 0xFFB44FE0
    _MAGENTA = 0xFF8A2FC0
    _TEAL = 0xFF9FD64A

    def _position_chart(self, data_colors: list[int]) -> JointGraphWidget:
        """One joint x (command, observed), the Position chart's layout."""
        return _widget(
            [self._TIMES] * 2,
            [self._TIMES, self._TIMES - 1.0],
            group_num=1,
            group_visible=[True, True],
            legends=["Command Joint", "Observed Joint"],
            data_colors=data_colors,
        )

    async def test_legend_uses_the_color_its_group_is_drawn_in(self) -> None:
        """The bug: the legend was a fixed cyan/blue while the traces drew pink/magenta."""
        widget = self._position_chart([self._PINK, self._MAGENTA])
        self.assertEqual(widget._group_colors(0), [self._PINK])
        self.assertEqual(widget._group_colors(1), [self._MAGENTA])

    async def test_legend_matches_the_trace_for_every_series(self) -> None:
        """Legend and trace read the same accessor, so they cannot disagree."""
        widget = self._position_chart([self._PINK, self._MAGENTA])
        for i in range(len(widget._y_data)):
            self.assertIn(widget._series_color(i), widget._group_colors(i // widget._group_num))

    async def test_multi_joint_group_shows_a_swatch_per_joint(self) -> None:
        """A group toggling several joints cannot be named by one dot."""
        widget = _widget(
            [self._TIMES] * 4,
            [self._TIMES] * 4,
            group_num=2,
            group_visible=[True, True],
            legends=["Command Joint", "Observed Joint"],
            data_colors=[self._PINK, self._TEAL, self._MAGENTA, self._TEAL],
        )
        self.assertEqual(widget._group_colors(0), [self._PINK, self._TEAL])
        self.assertEqual(widget._group_colors(1), [self._MAGENTA, self._TEAL])

    async def test_repeated_color_is_shown_once(self) -> None:
        """Two joints drawn in one colour need only one swatch."""
        widget = _widget(
            [self._TIMES] * 2,
            [self._TIMES] * 2,
            group_num=2,
            group_visible=[True],
            legends=["Effort"],
            data_colors=[self._TEAL, self._TEAL],
        )
        self.assertEqual(widget._group_colors(0), [self._TEAL])

    async def test_missing_colors_fall_back_to_the_trace_fallback(self) -> None:
        """Callers may pass no colours; the legend then shows what is drawn."""
        widget = self._position_chart([])
        self.assertEqual(widget._group_colors(0), [widget._series_color(0)])

    async def test_group_past_the_end_of_the_data_still_reports_a_color(self) -> None:
        """An empty chart still builds a legend row per header, so never return []."""
        widget = _widget(
            [],
            [],
            group_num=1,
            group_visible=[True, True],
            legends=["Command Joint", "Observed Joint"],
            data_colors=[],
        )
        self.assertEqual(len(widget._group_colors(1)), 1)


def _range_widget(y_min: float, y_max: float) -> JointGraphWidget:
    """Build a JointGraphWidget carrying only the current y view range.

    The grid and axis-label maths read nothing but that range, so the test skips
    ``__init__`` (and with it every ``omni.ui`` widget) and sets it directly.
    """
    widget = JointGraphWidget.__new__(JointGraphWidget)
    widget._y_min, widget._y_max = y_min, y_max
    widget._y_unit = "\u00b0"
    return widget


def _mantissa(step: float) -> float:
    """Return ``step`` scaled into ``[1, 10)``, so its 1/2/5 shape can be checked."""
    return round(step / 10.0 ** math.floor(math.log10(step)), 6)


class TestNiceStep(omni.kit.test.AsyncTestCase):
    """The grid step has to be a round number, and never finer than asked for."""

    async def test_step_is_one_two_or_five_times_a_power_of_ten(self) -> None:
        """Anything else is not a number a reader can count in."""
        for raw in (0.003, 0.07, 0.4, 1.0, 1.3, 2.0, 3.7, 6.2, 15.0, 480.0, 9.9e5):
            step = chart_widget._nice_step(raw)
            self.assertIn(_mantissa(step), (1.0, 2.0, 5.0), f"step {step!r} for raw {raw!r}")

    async def test_step_is_never_finer_than_requested(self) -> None:
        """Rounding up is what bounds the line count."""
        for raw in (0.003, 0.07, 1.3, 3.7, 6.2, 15.0, 480.0):
            self.assertGreaterEqual(chart_widget._nice_step(raw), raw, f"raw {raw!r}")

    async def test_an_already_round_step_is_not_pushed_up_a_rung(self) -> None:
        """`100.0 / 100.0` can divide out to just over 1.0, which would give 200."""
        for raw in (0.2, 1.0, 5.0, 100.0, 2000.0):
            self.assertAlmostEqual(chart_widget._nice_step(raw), raw, places=9, msg=f"raw {raw!r}")

    async def test_unusable_input_has_no_round_step(self) -> None:
        """Reported as 0.0 so the caller draws no grid rather than dividing by it."""
        for raw in (0.0, -1.0, float("nan"), float("inf"), 5e-324):
            self.assertEqual(chart_widget._nice_step(raw), 0.0, f"raw {raw!r}")


class TestGridIsAnchoredToTheData(omni.kit.test.AsyncTestCase):
    """Grid lines fall on round values inside the view range.

    The bug: `_build_grid` divided the viewport into equal bands geometrically,
    with no relationship to the data, so zero landed on a line only by
    coincidence and no other line was a value that could be read off either.
    """

    _MAX = chart_widget._GRID_MAX_LINES

    _SPREAD = (
        (-1.0, 1.0),
        (-42.0, 37.0),
        (10.0, 20.0),
        (0.5, 4.0),
        (-9.99, -0.01),
        (0.309, 0.331),
        (-0.004, 0.001),
        (-1e-7, 3e-7),
        (-1e6, 5e5),
    )

    async def test_zero_gets_a_line_whenever_it_is_in_range(self) -> None:
        """The point of the fix, and what the Effort chart is read against."""
        for lo, hi in ((-1.0, 1.0), (-42.0, 37.0), (-0.004, 0.001), (-1e6, 5e5), (0.0, 12.0), (-9.0, 0.0)):
            self.assertIn(0.0, chart_widget._nice_ticks(lo, hi), f"no zero line across ({lo!r}, {hi!r})")

    async def test_a_range_that_excludes_zero_has_no_zero_line(self) -> None:
        """A zero line outside the range would be drawn at a clamped edge and lie."""
        for lo, hi in ((10.0, 20.0), (0.5, 4.0), (-9.99, -0.01), (-1e5, -1.0)):
            self.assertNotIn(0.0, chart_widget._nice_ticks(lo, hi), f"({lo!r}, {hi!r})")

    async def test_ticks_are_round_numbers(self) -> None:
        """Whole multiples of a 1/2/5 step, not fractions of the viewport."""
        expected = {
            (-1.0, 1.0): [-1.0, -0.5, 0.0, 0.5, 1.0],
            (-42.0, 37.0): [-40.0, -20.0, 0.0, 20.0],
            (10.0, 20.0): [10.0, 12.0, 14.0, 16.0, 18.0, 20.0],
            (0.0, 1.0): [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            (0.309, 0.331): [0.31, 0.315, 0.32, 0.325, 0.33],
        }
        for (lo, hi), values in expected.items():
            ticks = chart_widget._nice_ticks(lo, hi)
            self.assertEqual(len(ticks), len(values), f"({lo!r}, {hi!r}) gave {ticks}")
            for tick, value in zip(ticks, values):
                self.assertAlmostEqual(tick, value, places=9, msg=f"({lo!r}, {hi!r}) gave {ticks}")

    async def test_the_range_is_never_snapped_to_suit_the_ticks(self) -> None:
        """The range is the user's, set by the slider, the fields, or Fit Frame View."""
        for lo, hi in self._SPREAD:
            ticks = chart_widget._nice_ticks(lo, hi)
            tolerance = (hi - lo) * 1e-9
            self.assertGreaterEqual(ticks[0], lo - tolerance, f"({lo!r}, {hi!r}) gave {ticks}")
            self.assertLessEqual(ticks[-1], hi + tolerance, f"({lo!r}, {hi!r}) gave {ticks}")

    async def test_line_count_stays_bounded_across_a_wide_span_of_ranges(self) -> None:
        """Dragging the range slider must not produce hundreds of rectangles."""
        for exponent in range(-9, 10):
            scale = 10.0**exponent
            for lo, hi in ((-scale, scale), (-7.3 * scale, 2.1 * scale), (0.0, scale), (0.5 * scale, 4.0 * scale)):
                ticks = chart_widget._nice_ticks(lo, hi)
                self.assertGreaterEqual(len(ticks), 2, f"({lo!r}, {hi!r}) drew no usable grid")
                self.assertLessEqual(len(ticks), self._MAX, f"({lo!r}, {hi!r}) drew {len(ticks)} lines")

    async def test_an_absurd_target_is_coarsened_rather_than_honoured(self) -> None:
        """The ceiling holds even if the requested line count is nonsense."""
        ticks = chart_widget._nice_ticks(-1.0, 1.0, target=10_000)
        self.assertLessEqual(len(ticks), self._MAX)
        self.assertIn(0.0, ticks, "coarsening must not lose the zero line")

    async def test_degenerate_ranges_draw_no_lines_and_do_not_raise(self) -> None:
        """No lines is the safe answer; a divide by zero or a hang is not."""
        for lo, hi in (
            (0.0, 0.0),
            (1.0, 1.0),
            (5.0, -5.0),
            (float("nan"), 1.0),
            (0.0, float("nan")),
            (float("-inf"), float("inf")),
            (0.0, float("inf")),
            (-1e308, 1e308),
            (0.0, 5e-324),
        ):
            self.assertEqual(chart_widget._nice_ticks(lo, hi), [], f"({lo!r}, {hi!r})")


class TestZeroGridLineIsDistinct(omni.kit.test.AsyncTestCase):
    """Where the widget places its horizontal lines, and which one is zero."""

    async def test_zero_is_flagged_once_at_its_data_position(self) -> None:
        """Effort-like range: zero sits 37 of 79 units down from the top."""
        lines = _range_widget(-42.0, 37.0)._y_grid_lines()
        zeros = [fraction for fraction, is_zero in lines if is_zero]
        self.assertEqual(len(zeros), 1, f"lines {lines}")
        self.assertAlmostEqual(zeros[0], 37.0 / 79.0, places=9)

    async def test_lines_run_top_to_bottom_inside_the_viewport(self) -> None:
        """Fractions are consumed in order as spacer weights, so order matters."""
        for lo, hi in ((-42.0, 37.0), (10.0, 20.0), (-1.0, 1.0)):
            fractions = [fraction for fraction, _ in _range_widget(lo, hi)._y_grid_lines()]
            self.assertEqual(fractions, sorted(fractions), f"({lo!r}, {hi!r})")
            self.assertGreaterEqual(min(fractions), 0.0, f"({lo!r}, {hi!r})")
            self.assertLessEqual(max(fractions), 1.0, f"({lo!r}, {hi!r})")

    async def test_the_grid_is_no_longer_an_equal_division_of_the_height(self) -> None:
        """The old grid always put a line on the top edge; this one need not."""
        lines = _range_widget(-42.0, 37.0)._y_grid_lines()
        self.assertGreater(lines[0][0], 0.0, "top line should sit at the highest round value, not the top edge")

    async def test_a_range_excluding_zero_flags_no_line(self) -> None:
        """Nothing may be highlighted as the datum when the datum is off screen."""
        for lo, hi in ((10.0, 20.0), (-9.99, -0.01)):
            lines = _range_widget(lo, hi)._y_grid_lines()
            self.assertTrue(lines, f"({lo!r}, {hi!r}) drew no grid")
            self.assertEqual([fraction for fraction, is_zero in lines if is_zero], [], f"({lo!r}, {hi!r})")

    async def test_a_collapsed_range_is_widened_rather_than_divided_by(self) -> None:
        """Same guard `_build_plots` applies, so grid and traces agree."""
        widget = _range_widget(5.0, 5.0)
        self.assertEqual(widget._y_view_range(), (5.0, 6.0))
        lines = widget._y_grid_lines()
        self.assertGreaterEqual(len(lines), 2)
        self.assertLessEqual(len(lines), chart_widget._GRID_MAX_LINES)

    async def test_a_non_finite_range_draws_no_grid(self) -> None:
        """An empty grid is fine; an exception during a rebuild is not."""
        self.assertEqual(_range_widget(float("nan"), 1.0)._y_grid_lines(), [])


class TestYAxisLabelNamesAGridLine(omni.kit.test.AsyncTestCase):
    """The interior y label reports a line that is drawn, not the range midpoint.

    A round-valued line is only useful if the reader can tell what value it is,
    and the old label reported ``(y_min + y_max) / 2``, which no line was at.
    """

    async def test_zero_is_labelled_when_it_is_in_range(self) -> None:
        """The most valuable label on the Effort chart."""
        fraction, value = _range_widget(-42.0, 37.0)._y_label_line()
        self.assertEqual(value, 0.0)
        self.assertAlmostEqual(fraction, 37.0 / 79.0, places=9)

    async def test_the_label_sits_on_a_line_that_is_drawn(self) -> None:
        """A label offset from its line would misreport by its own offset."""
        for lo, hi in ((-42.0, 37.0), (10.0, 20.0), (-1.0, 1.0), (0.309, 0.331)):
            widget = _range_widget(lo, hi)
            label = widget._y_label_line()
            self.assertIsNotNone(label, f"({lo!r}, {hi!r})")
            fraction, _ = label
            drawn = [line for line, _ in widget._y_grid_lines()]
            self.assertTrue(
                any(abs(line - fraction) < 1e-9 for line in drawn),
                f"({lo!r}, {hi!r}) labelled {fraction!r}, lines at {drawn}",
            )

    async def test_the_label_is_not_the_arithmetic_midpoint(self) -> None:
        """Across 10..20 the midpoint 15 is not a round line; 14 and 16 are."""
        _, value = _range_widget(10.0, 20.0)._y_label_line()
        self.assertIn(value, (14.0, 16.0))

    async def test_the_label_falls_strictly_inside_the_range(self) -> None:
        """The endpoints already have their own labels at the top and bottom."""
        for lo, hi in ((-42.0, 37.0), (10.0, 20.0), (-1.0, 1.0)):
            fraction, value = _range_widget(lo, hi)._y_label_line()
            self.assertGreater(fraction, 0.0, f"({lo!r}, {hi!r})")
            self.assertLess(fraction, 1.0, f"({lo!r}, {hi!r})")
            self.assertGreater(value, lo, f"({lo!r}, {hi!r})")
            self.assertLess(value, hi, f"({lo!r}, {hi!r})")

    async def test_a_non_finite_range_is_left_unlabelled(self) -> None:
        """No line to name, so name none rather than inventing one."""
        self.assertIsNone(_range_widget(float("nan"), 1.0)._y_label_line())

    async def test_the_label_text_is_drawable(self) -> None:
        """Axis labels follow the same ASCII-plus-degree rule as the readout."""
        for lo, hi in ((-42.0, 37.0), (10.0, 20.0), (0.309, 0.331)):
            widget = _range_widget(lo, hi)
            _, value = widget._y_label_line()
            text = f"{chart_widget._fmt(value)}{widget._y_unit}"
            self.assertEqual(_unrenderable(text), [], f"unrenderable characters in {text!r}")
