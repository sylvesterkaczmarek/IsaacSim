# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Joint-position graph widget.

Layout:
    - Legend row with checkboxes toggling the Command / Observed series.
    - A vertical range slider (Y axis) with type-in value fields at the top (max)
      and bottom (min).
    - A plot viewport whose axis value labels are drawn *inside* the chart area,
      over a transparent capture layer that reports the pointer to the readout.
    - A horizontal range slider (X axis / time) with type-in value fields on the
      left (min) and right (max).
    - A "Fit Frame View" button that resets both axes to the full data extent.
    - A collapsible readout section, below all of the above so that opening it
      moves neither the plot nor the axis controls.  It reports every visible
      series at the hovered time as a row per joint and a column per legend
      group, and its header states what it is for while it is closed.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import carb.events
import carb.input
import numpy as np
import omni.appwindow
import omni.kit.app
import omni.ui as ui
from isaacsim.robot_setup.gain_tuner.divergence import clip_series_to_valid

from .frame_widget import CustomCollapsableFrame as CollapsableFrame
from .style import (
    CHART_AXIS_LABEL_COLOR,
    CHART_BTN_BG,
    CHART_CARD_BG,
    CHART_GRID_COLOR,
    CHART_READOUT_BG,
    CHART_READOUT_TEXT,
    CHART_SLIDER_HANDLE,
    CHART_SLIDER_TRACK,
    CHART_STROKE,
    CHART_UNIT_COLOR,
    CHART_VALUE_FIELD_BG,
    CHART_VIEWPORT_BG,
    CHART_ZERO_LINE_COLOR,
    LABEL_COLOR,
    MUTED_LABEL_COLOR,
)

_GRID_TARGET_LINES = 5
#: Roughly how many horizontal grid lines to aim for. The step is rounded up
#: from ``span / target``, so the count never exceeds ``target + 1``.

_GRID_MAX_LINES = 12
#: Hard ceiling on horizontal grid lines, so no zoom level can fill the viewport
#: with rectangles.

_TICK_INDEX_EPS = 1e-9
#: Tolerance, in units of the step, for admitting a tick that floating-point
#: division puts a hair outside the range instead of exactly on its edge.

_STEP_LADDER_LIMIT = 16
#: Cap on how many rungs of the 1/2/5 ladder the step may be coarsened by while
#: bringing the line count under `_GRID_MAX_LINES`.


def _fmt(v: float) -> str:
    """Format a number: integer when whole, else one decimal place."""
    try:
        if abs(v - round(v)) < 1e-6:
            return str(int(round(v)))
        return f"{v:.1f}"
    except (ValueError, OverflowError):
        return "0"


def _fmt_readout(v: float) -> str:
    """Format a reported series value, falling back to scientific notation."""
    if not np.isfinite(v):
        return "n/a"
    if v != 0.0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
        return f"{v:.3e}"
    return f"{v:.3f}"


VIEW_HEADROOM = 1.5
"""Multiple of the reference travel the y axis opens on when a trace was clipped.

The retained samples run right up to the divergence bound, which is several times
the joint's travel, so framing the axis on the data would squash the response that
matters into a band a few pixels tall.  Framing it on the reference instead fills
the chart with the tracking behaviour and lets the diverging tail leave the top.
"""

Y_MARGIN = 0.1
"""Fraction of the plotted y span left clear above and below the trace."""


def framed_window(reference: list[np.ndarray], headroom: float = VIEW_HEADROOM) -> tuple[float, float] | None:
    """Return the ``(lo, hi)`` y window to open on, centred on the reference travel.

    Args:
        reference: The commanded series, which are bounded by construction.
        headroom: Multiple of the reference half-span to leave for overshoot.

    Returns:
        The window, or ``None`` when the reference carries no finite samples, in
        which case the caller keeps the full data extent.

    Example:

    .. code-block:: python

        >>> import numpy as np
        >>> from isaacsim.robot_setup.gain_tuner.ui.chart_widget import framed_window

        >>> framed_window([np.array([0.0, 360.0])], headroom=1.5)
        (-90.0, 450.0)
    """
    finite: list[float] = []
    for a in reference:
        if a is None or len(a) == 0:
            continue
        arr = np.asarray(a, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr):
            finite.extend((float(np.min(arr)), float(np.max(arr))))
    if not finite:
        return None
    lo, hi = min(finite), max(finite)
    center = (lo + hi) / 2.0
    half = (hi - lo) / 2.0
    if half <= 1e-9:
        half = max(abs(center), 1.0)
    return center - half * headroom, center + half * headroom


_MIN_PLACEMENT_WIDTH = 1e-3
"""Smallest width fraction a plot is given, so it always has pixels to draw in."""

_POINTER_EDGE_SLACK = 1e-6
"""Slack, as a fraction of a series' drawn width, allowed at either end of it.

Well under a pixel at any viewport size, so it only absorbs the rounding in
turning a pointer position into a fraction and back; a pointer genuinely past the
end of a clipped trace still reads as past it.
"""


def window_placement(x_first: float, x_last: float, x_min: float, x_max: float) -> tuple[float, float]:
    """Return the ``(offset, width)`` fractions a series occupies in the x window.

    ``ui.Plot`` spreads the samples it is handed evenly across its whole width, so
    a series that covers only part of the window has to be given a proportionally
    narrower widget.  Without this, a trace clipped at divergence gets stretched
    to fill the axis and appears to diverge at the end of the run rather than
    where it actually did.

    Args:
        x_first: First x value the series covers.
        x_last: Last x value the series covers.
        x_min: Left edge of the current x window.
        x_max: Right edge of the current x window.

    Returns:
        Tuple of the leading offset and the width, each as a fraction of the
        window.  A degenerate or non-finite window yields ``(0.0, 1.0)``, which
        renders as the full width.  The width is never zero, which would reach
        ``ui.Fraction(0.0)`` and draw nothing at all; a series sitting at the
        right edge is pulled back to a hairline instead.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner.ui.chart_widget import window_placement

        >>> window_placement(0.0, 2.0, 0.0, 4.0)
        (0.0, 0.5)
    """
    span = x_max - x_min
    if not np.isfinite(span) or span <= 1e-12:
        return 0.0, 1.0
    offset = min(max((x_first - x_min) / span, 0.0), 1.0)
    width = min(max((x_last - x_first) / span, 0.0), 1.0 - offset)
    if width < _MIN_PLACEMENT_WIDTH:
        width = _MIN_PLACEMENT_WIDTH
        offset = min(offset, 1.0 - width)
    return offset, width


def _nice_step(raw_step: float) -> float:
    """Round a spacing up to the nearest 1, 2, or 5 times a power of ten.

    Args:
        raw_step: The ideal spacing.

    Returns:
        The smallest 1/2/5-times-power-of-ten step that is at least ``raw_step``,
        or 0.0 for a step that is not finite and positive or whose decade cannot
        be represented.
    """
    if not math.isfinite(raw_step) or raw_step <= 0.0:
        return 0.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        # A denormal span underflows its decade; there is no round step for it.
        return 0.0
    fraction = raw_step / magnitude
    for multiple in (1.0, 2.0, 5.0):
        # A whole `raw_step` can divide out to 1.0000000000000002, which would
        # otherwise be rounded up a whole rung of the ladder.
        if fraction <= multiple * (1.0 + _TICK_INDEX_EPS):
            return multiple * magnitude
    return 10.0 * magnitude


def _nice_ticks(
    lo: float, hi: float, target: int = _GRID_TARGET_LINES, max_lines: int = _GRID_MAX_LINES
) -> list[float]:
    """Choose round values to draw grid lines at across ``[lo, hi]``.

    Ticks are whole multiples of a 1/2/5-times-power-of-ten step, so every line
    lands on a value that can be read off the chart, and zero is always among
    them when it falls inside the range: zero is index 0 of the step, and the
    index range spans it whenever ``lo <= 0 <= hi``.

    The range itself is never widened or snapped to make the ticks come out
    round.  It belongs to the user, who sets it with the range slider, the
    type-in fields, or Fit Frame View, so the ticks are chosen within it.

    Args:
        lo: Bottom of the range.
        hi: Top of the range.
        target: Roughly how many lines to aim for.
        max_lines: Hard ceiling on the number of lines returned.

    Returns:
        The tick values in ascending order, or an empty list for a range that is
        not finite, is empty, or is inverted.
    """
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return []
    span = hi - lo
    if not math.isfinite(span) or span <= 0.0:
        return []
    max_lines = max(int(max_lines), 1)
    step = _nice_step(span / max(int(target), 1))
    if step <= 0.0:
        return []
    # Rounding the step up already caps the count at `target + 1`.  This only
    # coarsens further if a pathological range slips past that cap, and every
    # rung of the ladder is strictly larger, so it terminates.
    for _ in range(_STEP_LADDER_LIMIT):
        first = math.ceil(lo / step - _TICK_INDEX_EPS)
        last = math.floor(hi / step + _TICK_INDEX_EPS)
        if last - first < max_lines:
            break
        coarser = _nice_step(step * 1.5)
        if coarser <= step:
            return []
        step = coarser
    else:
        return []
    return [value for value in (index * step for index in range(first, last + 1)) if math.isfinite(value)]


def _pointer_screen_pos() -> tuple[float, float] | None:
    """Return the pointer position in the space widget screen positions use.

    ``omni.ui`` hands a pointer position only to ``mouse_moved_fn``, which fires
    while a mouse button is held rather than on plain hover, and its hover
    callback reports nothing but enter and exit.  A readout that follows a
    hovering pointer therefore has to ask the input system where it is.
    ``carb.input`` answers in device pixels while widget screen positions are in
    points, hence the DPI divide.

    Returns:
        The pointer (x, y), or None when there is no app window to read it from.
    """
    app_window = omni.appwindow.get_default_app_window()
    if app_window is None:
        return None
    coords = carb.input.acquire_input_interface().get_mouse_coords_pixel(app_window.get_mouse())
    dpi_scale = ui.Workspace.get_dpi_scale() or 1.0
    return coords[0] / dpi_scale, coords[1] / dpi_scale


class _RangeSlider:
    """A dual-handle range slider built from draggable ``ui.Placer`` handles.

    Works either vertically (top = max value, bottom = min value) or horizontally
    (left = min value, right = max value).  Reports the current (lo, hi) values via
    ``on_change`` during drags.  The slider persists across plot rebuilds, so the
    handle being dragged is never destroyed mid-interaction.
    """

    HANDLE = 12

    def __init__(
        self,
        length_px: int,
        vertical: bool,
        data_min: float,
        data_max: float,
        lo: float,
        hi: float,
        on_change: Callable[[float, float], None],
    ):
        self._length = length_px
        self._vertical = vertical
        self._data_min = float(data_min)
        self._data_max = float(data_max)
        # Guard against a zero-width data range.
        if self._data_max - self._data_min < 1e-9:
            self._data_max = self._data_min + 1.0
        self._lo = float(lo)
        self._hi = float(hi)
        self._on_change = on_change
        self._suppress = False
        self._lo_placer: ui.Placer | None = None
        self._hi_placer: ui.Placer | None = None

    # ---- value <-> pixel mapping ------------------------------------------
    @property
    def _travel(self) -> float:
        return max(self._length - self.HANDLE, 1)

    @property
    def _range(self) -> float:
        return self._data_max - self._data_min

    def _val_to_px(self, v: float) -> float:
        frac = (v - self._data_min) / self._range
        frac = max(0.0, min(1.0, frac))
        if self._vertical:
            # top = max value → small offset; bottom = min value → large offset
            return (1.0 - frac) * self._travel
        return frac * self._travel

    def _px_to_val(self, px: float) -> float:
        px = max(0.0, min(px, self._travel))
        frac = px / self._travel
        if self._vertical:
            frac = 1.0 - frac
        return self._data_min + frac * self._range

    # ---- handle rendering --------------------------------------------------
    def _make_handle(self):
        ui.Rectangle(
            width=self.HANDLE,
            height=self.HANDLE,
            style={
                "background_color": CHART_SLIDER_HANDLE,
                "border_radius": self.HANDLE / 2,
            },
        )

    def build(self):
        """Construct the slider UI. Call once; it survives plot rebuilds."""
        if self._vertical:
            with ui.ZStack(width=self.HANDLE, height=self._length):
                with ui.HStack():
                    ui.Spacer()
                    ui.Rectangle(width=2, style={"background_color": CHART_SLIDER_TRACK})
                    ui.Spacer()
                self._hi_placer = ui.Placer(draggable=True, drag_axis=ui.Axis.Y, offset_y=self._val_to_px(self._hi))
                with self._hi_placer:
                    self._make_handle()
                self._lo_placer = ui.Placer(draggable=True, drag_axis=ui.Axis.Y, offset_y=self._val_to_px(self._lo))
                with self._lo_placer:
                    self._make_handle()
                self._hi_placer.set_offset_y_changed_fn(lambda *_: self._on_drag(is_hi=True))
                self._lo_placer.set_offset_y_changed_fn(lambda *_: self._on_drag(is_hi=False))
        else:
            with ui.ZStack(width=self._length, height=self.HANDLE):
                with ui.VStack():
                    ui.Spacer()
                    ui.Rectangle(height=2, style={"background_color": CHART_SLIDER_TRACK})
                    ui.Spacer()
                self._lo_placer = ui.Placer(draggable=True, drag_axis=ui.Axis.X, offset_x=self._val_to_px(self._lo))
                with self._lo_placer:
                    self._make_handle()
                self._hi_placer = ui.Placer(draggable=True, drag_axis=ui.Axis.X, offset_x=self._val_to_px(self._hi))
                with self._hi_placer:
                    self._make_handle()
                self._lo_placer.set_offset_x_changed_fn(lambda *_: self._on_drag(is_hi=False))
                self._hi_placer.set_offset_x_changed_fn(lambda *_: self._on_drag(is_hi=True))

    def _placer_px(self, placer: ui.Placer) -> float:
        off = placer.offset_y if self._vertical else placer.offset_x
        # offset_x / offset_y may return a ui.Length; extract the pixel value.
        return float(getattr(off, "value", off))

    def _set_placer_px(self, placer: ui.Placer, px: float):
        if self._vertical:
            placer.offset_y = px
        else:
            placer.offset_x = px

    def _on_drag(self, is_hi: bool):
        if self._suppress:
            return
        placer = self._hi_placer if is_hi else self._lo_placer
        v = self._px_to_val(self._placer_px(placer))
        # Keep hi >= lo (with a tiny separation so both remain grabbable).
        if is_hi:
            v = max(v, self._lo)
            self._hi = v
        else:
            v = min(v, self._hi)
            self._lo = v
        # Snap the handle to the clamped value (guarded to avoid recursion).
        snapped = self._val_to_px(v)
        if abs(snapped - self._placer_px(placer)) > 0.5:
            self._suppress = True
            self._set_placer_px(placer, snapped)
            self._suppress = False
        if self._on_change:
            self._on_change(self._lo, self._hi)

    def set_values(self, lo: float, hi: float):
        """Programmatically move both handles (e.g. from type-in fields / Fit)."""
        self._lo = float(lo)
        self._hi = float(hi)
        self._suppress = True
        if self._lo_placer is not None:
            self._set_placer_px(self._lo_placer, self._val_to_px(self._lo))
        if self._hi_placer is not None:
            self._set_placer_px(self._hi_placer, self._val_to_px(self._hi))
        self._suppress = False


class JointGraphWidget:
    """Joint-position graph.

    Args:
        x_data: list of 1-D arrays (time per series), sorted ascending.
        y_data: list of 1-D arrays (values per series).
        data_colors: per-series colours (omni.ui ABGR ints).  The legend swatches
            are read from here too, so a trace and its legend entry always agree.
        header_count: number of legend groups (2 = Command / Observed).
        legends: legend labels; defaults to ["Command Joint", "Observed Joint"].
        series_labels: per-series names for the readout, one entry per ``y_data``
            series.  Pass the joint each series belongs to when a legend group
            holds more than one joint, otherwise the readout cannot tell them
            apart.  Falls back to the group's legend label.
        joint_labels: joint names, one entry per joint, naming the readout's rows.
            A legend group holds one series per joint, so this is ``y_data``
            divided by ``header_count``.  Falls back to the series labels, which
            repeat the group's name in every row.
        y_unit / x_unit: unit suffixes shown in the value fields and axis labels.
        reference_count: number of leading ``y_data`` series that are commanded
            trajectories rather than recorded responses.  They are never clipped,
            and they frame the axes when a response is.
        divergence_limits: per-series magnitude bounds aligned with ``y_data``, as
            returned by :func:`~isaacsim.robot_setup.gain_tuner.divergence.divergence_limit`.
            A ``None`` entry clips only non-finite samples in that series.
    """

    VIEWPORT_H = 220
    Y_SLIDER_H = 160
    X_SLIDER_W = 200
    FIELD_W = 72

    READOUT_TITLE = "Value Readout: hover the chart to read every joint"
    #: Header of the readout section.  It carries the prompt rather than a label
    #: inside the section, because the section is closed until the user opens it
    #: and a hint they cannot see until they open it explains nothing.

    READOUT_ROW_H = 18
    READOUT_NAME_W = 150
    #: Row height and joint-name column width of the readout table.

    READOUT_NO_VALUE = "-"
    #: Stands in for a joint that has nothing to report in a column -- a series
    #: with too few samples left inside a zoomed window -- so a row is never
    #: silently short of a cell.

    #: Both strings, and every figure written into the table, are ASCII apart from
    #: `\u00b0`: the app's default font has no glyph for an em dash, an ellipsis or
    #: a typographic quote and draws each as a stray "?", which is what the header
    #: showed while it separated its title from its prompt with an em dash.  Same
    #: rule as `NA_CELL_TEXT` in `gain_table_view`, guarded by a test.

    def __init__(
        self,
        x_data: list[np.ndarray],
        y_data: list[np.ndarray],
        data_colors: list[int],
        header_count: int = 2,
        legends: list[str] | None = None,
        y_unit: str = "\u00b0",
        x_unit: str = "s",
        series_labels: list[str] | None = None,
        joint_labels: list[str] | None = None,
        reference_count: int = 0,
        divergence_limits: list[float | None] | None = None,
    ):
        raw_x = [np.asarray(a, dtype=float) for a in x_data]
        raw_y = [np.asarray(a, dtype=float) for a in y_data]
        # Clip each series where it stops carrying usable data, so a joint that
        # diverged mid-run plots its valid prefix rather than dictating the shared
        # axis range -- as NaN samples or a wound-up magnitude both otherwise do.
        self._x_data, self._y_data = clip_series_to_valid(raw_x, raw_y, limits=divergence_limits)
        # Only a clipped *response* means divergence; a reference cut short carries
        # no verdict about the joint, so it must not claim one.
        self._diverged = any(
            len(self._y_data[i]) < len(raw_y[i]) for i in range(reference_count, len(raw_y)) if i < len(self._y_data)
        )
        self._data_colors = list(data_colors) if data_colors else []
        self._header_count = max(header_count, 1)
        self._legends = legends or ["Command Joint", "Observed Joint", "User Data Driven"]
        self._series_labels = list(series_labels) if series_labels else []
        self._joint_labels = list(joint_labels) if joint_labels else []
        self._y_unit = y_unit
        self._x_unit = x_unit

        self._group_num = max(len(self._y_data) // self._header_count, 1)
        self._group_visible = [True] * self._header_count

        # Compute full data bounds.
        self._x_data_min, self._x_data_max = self._compute_bounds(self._x_data, default=(0.0, 1.0))
        y_lo, y_hi = self._compute_bounds(self._y_data, default=(-1.0, 1.0))
        # Pad the y range slightly so the trace isn't flush against the edges.
        pad = (y_hi - y_lo) * Y_MARGIN if (y_hi - y_lo) > 1e-9 else 1.0
        self._y_data_min, self._y_data_max = y_lo - pad, y_hi + pad

        # Current view window.  Normally the full extent, but a run that diverged
        # is mostly wasted axis: the retained response can be a few percent of the
        # run and reaches several times the joint's travel, so opening on the full
        # extent squeezes it to a sliver in both directions -- the empty-looking
        # chart this all exists to prevent.  Frame it on what is worth reading and
        # leave the extent to the sliders and Fit Frame View.
        self._x_min, self._x_max = self._x_data_min, self._x_data_max
        self._y_min, self._y_max = self._y_data_min, self._y_data_max
        if self._diverged:
            retained = self._retained_x_extent(reference_count)
            if retained is not None:
                self._x_min, self._x_max = retained
            framed = framed_window(self._y_data[:reference_count])
            if framed is not None:
                self._y_min, self._y_max = framed

        # Whether every series is plotted against the same x samples, resolved on
        # first use because it is a scan of the data.
        self._shared_timeline = None

        # Collapsed state of the readout section.  Held here rather than in the
        # section itself because everything that rebuilds the chart -- a slider
        # drag, a legend toggle -- would otherwise reset it to its default.
        self._readout_collapsed = True

        # Widget refs.
        self._plot_frame = None
        self._readout_frame = None
        self._readout_section = None
        self._readout_x_label = None
        self._readout_x_shown = ""
        self._readout_cells = []
        self._readout_shown = []
        self._readout_built_shape = None
        self._pointer_sub = None
        self._y_slider = None
        self._x_slider = None
        self._y_max_model = None
        self._y_min_model = None
        self._x_min_model = None
        self._x_max_model = None

        self._container = ui.Frame(build_fn=self._build)
        self._container.rebuild()

    def _retained_x_extent(self, reference_count: int) -> tuple[float, float] | None:
        """Return the ``(first, last)`` time span the clipped responses still cover.

        References are excluded: they run the whole test, so including them would
        report the full run and defeat the framing.
        """
        last = None
        for i in range(reference_count, len(self._x_data)):
            xa = self._x_data[i]
            if xa is None or len(xa) == 0:
                continue
            end = float(xa[-1])
            last = end if last is None else max(last, end)
        if last is None or last <= self._x_data_min + 1e-9:
            return None
        return self._x_data_min, last

    @staticmethod
    def _compute_bounds(arrays: list[np.ndarray], default) -> tuple:
        lo, hi = None, None
        for a in arrays:
            if a is None or len(a) == 0:
                continue
            # Ignore non-finite samples: `np.min`/`np.max` propagate NaN, which
            # would make the axis range NaN and blank every series in the chart.
            finite = np.asarray(a, dtype=float)
            finite = finite[np.isfinite(finite)]
            if len(finite) == 0:
                continue
            amin, amax = float(np.min(finite)), float(np.max(finite))
            lo = amin if lo is None else min(lo, amin)
            hi = amax if hi is None else max(hi, amax)
        if lo is None or hi is None:
            return default
        if hi - lo < 1e-9:
            hi = lo + 1.0
        return lo, hi

    # ---- top-level build ---------------------------------------------------
    def _build(self):
        with ui.ZStack():
            ui.Rectangle(
                style={
                    "background_color": CHART_CARD_BG,
                    "border_color": CHART_STROKE,
                    "border_width": 1,
                    "border_radius": 4,
                }
            )
            with ui.VStack(spacing=0):
                ui.Spacer(height=10)  # padding above the legend checkboxes
                self._build_legend()
                ui.Spacer(height=12)  # gap between legend controls and the chart
                # Workspace: Y slider column + plot column
                with ui.HStack(spacing=8, height=self.VIEWPORT_H + 24):
                    ui.Spacer(width=12)
                    self._build_y_column()
                    self._build_plot_column()
                    ui.Spacer(width=12)
                # X slider row (aligned under the plot, offset past the Y column)
                self._build_x_row()
                # Footer
                self._build_footer()
                # Readout last, so opening it moves nothing above it
                self._build_readout_section()
                ui.Spacer(height=8)  # padding below the readout section

    # ---- legend ------------------------------------------------------------
    def _series_color(self, index: int) -> int:
        """Return the colour series ``index`` is drawn in.

        The single source of truth for both the traces and the legend swatches,
        so the two cannot drift apart.
        """
        if index < len(self._data_colors):
            return self._data_colors[index]
        return LABEL_COLOR

    def _group_colors(self, group_idx: int) -> list[int]:
        """Return the distinct colours drawn for a legend group, in series order.

        A group spans one series per joint, so a multi-joint selection needs a
        swatch each -- a single dot could only name one of the joints it toggles.
        """
        colors: list[int] = []
        for i in range(group_idx * self._group_num, (group_idx + 1) * self._group_num):
            if i >= len(self._y_data):
                break
            color = self._series_color(i)
            if color not in colors:
                colors.append(color)
        return colors or [LABEL_COLOR]

    def _build_legend(self):
        _ROW_H = 20
        with ui.HStack(height=_ROW_H, spacing=16):
            ui.Spacer(width=12)
            for gi in range(self._header_count):
                colors = self._group_colors(gi)
                # Colouring the label is only meaningful when the group draws in
                # a single colour; otherwise it would pick one joint arbitrarily.
                label_color = colors[0] if len(colors) == 1 else LABEL_COLOR
                with ui.HStack(width=0, height=_ROW_H, spacing=8):
                    # Checkbox — centered with a slight downward bias (the check
                    # glyph renders high in its box, so bias top:bottom = 5:1).
                    with ui.VStack(width=14):
                        ui.Spacer(height=5)
                        model = ui.SimpleBoolModel()
                        model.set_value(self._group_visible[gi])
                        ui.CheckBox(model=model, width=14, height=14)
                        model.add_value_changed_fn(lambda m, idx=gi: self._on_legend_toggle(idx, m.get_value_as_bool()))
                        ui.Spacer(height=1)
                    # Colored dots — explicitly centered in the row
                    with ui.HStack(width=0, height=_ROW_H, spacing=3):
                        for color in colors:
                            with ui.VStack(width=8):
                                ui.Spacer()
                                ui.Rectangle(
                                    width=8,
                                    height=8,
                                    style={"background_color": color, "border_radius": 4},
                                )
                                ui.Spacer()
                    # Label — spans full row height and centers its text
                    ui.Label(
                        self._legends[gi] if gi < len(self._legends) else f"Series {gi}",
                        width=0,
                        height=_ROW_H,
                        alignment=ui.Alignment.LEFT_CENTER,
                        style={"color": label_color, "font_size": 12},
                    )
            ui.Spacer()

    def _on_legend_toggle(self, group_idx: int, visible: bool):
        if 0 <= group_idx < len(self._group_visible):
            self._group_visible[group_idx] = visible
            if self._plot_frame:
                self._plot_frame.rebuild()

    # ---- Y slider column ---------------------------------------------------
    def _build_y_column(self):
        with ui.VStack(width=self.FIELD_W, spacing=6):
            ui.Spacer(height=4)
            self._y_max_model = self._value_field(self._y_max, self._y_unit, self._on_y_max_commit)
            with ui.HStack(height=self.Y_SLIDER_H):
                ui.Spacer()
                self._y_slider = _RangeSlider(
                    length_px=self.Y_SLIDER_H,
                    vertical=True,
                    data_min=self._y_data_min,
                    data_max=self._y_data_max,
                    lo=self._y_min,
                    hi=self._y_max,
                    on_change=self._on_y_slider,
                )
                self._y_slider.build()
                ui.Spacer()
            self._y_min_model = self._value_field(self._y_min, self._y_unit, self._on_y_min_commit)

    # ---- plot column -------------------------------------------------------
    def _build_plot_column(self):
        with ui.VStack():
            ui.Spacer(height=4)
            self._plot_frame = ui.Frame(height=self.VIEWPORT_H, build_fn=self._build_plot_area)
            ui.Spacer()

    def _build_plot_area(self):
        with ui.ZStack(
            style={
                "Rectangle::viewport": {
                    "background_color": CHART_VIEWPORT_BG,
                    "border_color": CHART_STROKE,
                    "border_width": 1,
                    "border_radius": 4,
                }
            },
        ):
            ui.Rectangle(name="viewport")
            self._build_grid()
            self._build_traces()
            self._build_axis_labels()

    def _y_view_range(self) -> tuple[float, float]:
        """Return the y range the traces are drawn against.

        Widened when the range has collapsed, matching what ``ui.Plot`` is given,
        so the grid, the axis labels, and the traces all describe one range.
        """
        y_min, y_max = self._y_min, self._y_max
        if y_max - y_min < 1e-9:
            y_max = y_min + 1.0
        return y_min, y_max

    def _y_grid_lines(self) -> list[tuple[float, bool]]:
        """Return the horizontal grid lines as ``(fraction from top, is zero)``.

        The fraction is where the line's value falls in the current view range,
        which is what decouples the grid from the viewport's geometry: a line is
        drawn at a round value rather than at an equal division of the height.

        Returns:
            The lines ordered top to bottom, empty when the range admits none.
        """
        y_min, y_max = self._y_view_range()
        span = y_max - y_min
        if span <= 0.0:
            return []
        lines = [(min(max((y_max - value) / span, 0.0), 1.0), value == 0.0) for value in _nice_ticks(y_min, y_max)]
        lines.sort(key=lambda line: line[0])
        return lines

    def _build_grid(self, cols: int = 6):
        # Horizontal grid lines, anchored to round data values rather than to
        # equal fractions of the viewport, so zero lands on a line whenever it is
        # in range and every other line is a number that can be read off.  The
        # 1px rectangles take their height out of the space the fractions divide,
        # which at this line count shifts a line by well under a pixel.
        lines = self._y_grid_lines()
        if lines:
            with ui.VStack():
                above = 0.0
                for fraction, is_zero in lines:
                    if fraction > above:
                        ui.Spacer(height=ui.Fraction(fraction - above))
                    ui.Rectangle(
                        height=1,
                        style={"background_color": CHART_ZERO_LINE_COLOR if is_zero else CHART_GRID_COLOR},
                    )
                    above = fraction
                if above < 1.0:
                    ui.Spacer(height=ui.Fraction(1.0 - above))
        # Vertical grid lines stay at equal fractions of the width.  `ui.Plot`
        # spreads its samples evenly across the rect it is given whatever their x
        # values are (see `_build_plots`), so a line drawn at a round time would
        # not sit where that time is plotted -- anchoring these to data would
        # mislead rather than help.  A clipped series is placed on its own stretch
        # of the axis, which makes its x proportional over that stretch, but a
        # series whose samples are unevenly spaced (the dt sweep) is still drawn
        # evenly within it.  Time also starts at the left edge, so there is no
        # interior zero crossing to mark.
        with ui.HStack():
            for i in range(cols + 1):
                ui.Rectangle(width=1, style={"background_color": CHART_GRID_COLOR})
                if i < cols:
                    ui.Spacer(width=ui.Fraction(1))

    def _window_bounds(self, xa: np.ndarray, ya: np.ndarray) -> tuple[int, int]:
        """Return the ``[start, stop)`` sample range inside the current x window.

        Returns an empty range when the series has fewer than two samples in
        view, which is also the condition under which it is not drawn.
        """
        if xa is None or ya is None or len(xa) == 0:
            return 0, 0
        i0 = int(np.searchsorted(xa, self._x_min, side="left"))
        i1 = int(np.searchsorted(xa, self._x_max, side="right"))
        i0 = max(0, min(i0, len(ya)))
        i1 = max(0, min(i1, len(ya), len(xa)))
        return (i0, i1) if i1 - i0 >= 2 else (0, 0)

    def _series_placement(self, index: int) -> tuple[int, int, float, float]:
        """Return where one series is drawn: its window slice and its span of the width.

        The single description of a series' drawn geometry, so that the pointer is
        resolved against exactly what :meth:`_build_plots` put on screen.  A series
        clipped at divergence is given only the stretch of the axis its samples
        cover, and the readout has to invert the same placement or it reports the
        wrong sample.

        Args:
            index: Series index into ``y_data``.

        Returns:
            ``(start, stop, offset, width)`` -- the ``[start, stop)`` sample range
            inside the x window, and the offset and width of the series' stretch of
            the viewport as fractions of the full width.  An empty range comes back
            as ``(0, 0, 0.0, 1.0)`` and is not drawn.
        """
        ya = self._y_data[index] if index < len(self._y_data) else None
        xa = self._x_data[index] if index < len(self._x_data) else None
        i0, i1 = self._window_bounds(xa, ya)
        if i1 - i0 < 2:
            return 0, 0, 0.0, 1.0
        offset, width = window_placement(float(xa[i0]), float(xa[i1 - 1]), self._x_min, self._x_max)
        return i0, i1, offset, width

    def _build_plots(self):
        y_min, y_max = self._y_view_range()
        # Final backstop: `ui.Plot` draws nothing at all when handed a non-finite
        # range.  Only the traces are substituted for; the grid and the axis labels
        # leave a range they cannot describe undrawn rather than naming values that
        # are not the ones in force.
        if not (np.isfinite(y_min) and np.isfinite(y_max)):
            y_min, y_max = -1.0, 1.0
        for i in range(len(self._y_data)):
            group = i // self._group_num
            if group < len(self._group_visible) and not self._group_visible[group]:
                continue
            i0, i1, offset, width = self._series_placement(i)
            if i1 - i0 < 2:
                continue
            values = [float(v) for v in self._y_data[i][i0:i1]]
            color = self._series_color(i)
            # Zero margin and padding so a plot's drawing rect is exactly the rect
            # it was given: that is what lines the trace up with the grid and the
            # axis labels, and what lets `offset` and `width` below be read as
            # fractions of the viewport -- which the readout relies on to invert.
            # No `border_width`: on `ui.Plot` it outlines the widget rather than
            # thickening the trace, so a plot narrowed to its clipped span would
            # draw a stray rule at the clip point.
            style = {"color": color, "background_color": 0x0, "margin": 0, "padding": 0}
            if offset <= 1e-6 and width >= 1.0 - 1e-6:
                ui.Plot(ui.Type.LINE, y_min, y_max, *values, style=style)
                continue
            # Series ends before the window does (clipped at divergence), so keep
            # it on its own stretch of the axis instead of spanning the full width.
            with ui.HStack():
                if offset > 1e-6:
                    ui.Spacer(width=ui.Fraction(offset))
                ui.Plot(ui.Type.LINE, y_min, y_max, *values, style=style, width=ui.Fraction(width))
                trailing = 1.0 - offset - width
                if trailing > 1e-6:
                    ui.Spacer(width=ui.Fraction(trailing))

    def _y_label_line(self) -> tuple[float, float] | None:
        """Return the interior y label as ``(fraction from top, value)``.

        Names one grid line between the range endpoints: zero when zero is in
        range, otherwise whichever line is nearest the middle.  A round-valued
        line only helps if its value can be told, and the old label reported the
        arithmetic midpoint of the range, which no line was drawn at.

        One label rather than one per line: the viewport is 220px tall and the
        traces are drawn in a detached ImGui window that paints over these
        labels, so each extra label is another figure a trace can cross.  Keeping
        the count at three leaves that overdraw no worse than before.

        Returns:
            The line to label, or None when no line falls strictly inside the
            range and the endpoint labels already say everything.
        """
        y_min, y_max = self._y_view_range()
        span = y_max - y_min
        if span <= 0.0:
            return None
        # Endpoints are already labelled at the top and bottom of the column.
        values = [value for value in _nice_ticks(y_min, y_max) if y_min < value < y_max]
        if not values:
            return None
        if any(value == 0.0 for value in values):
            value = 0.0
        else:
            middle = (y_min + y_max) / 2.0
            value = min(values, key=lambda candidate: abs(candidate - middle))
        return (y_max - value) / span, value

    def _build_axis_labels(self):
        x_mid = (self._x_min + self._x_max) / 2.0
        lbl_style = {"color": CHART_AXIS_LABEL_COLOR, "font_size": 10}
        y_label = self._y_label_line()
        # Y axis labels (top=max, bottom=min) along the left edge, with one
        # interior label placed on the grid line whose value it reports.
        with ui.HStack():
            with ui.VStack(width=0):
                ui.Spacer(height=6)
                ui.Label(f"{_fmt(self._y_max)}{self._y_unit}", width=0, style=lbl_style)
                if y_label is None:
                    ui.Spacer()
                else:
                    fraction, value = y_label
                    ui.Spacer(height=ui.Fraction(fraction))
                    ui.Label(f"{_fmt(value)}{self._y_unit}", width=0, style=lbl_style)
                    ui.Spacer(height=ui.Fraction(1.0 - fraction))
                ui.Label(f"{_fmt(self._y_min)}{self._y_unit}", width=0, style=lbl_style)
                ui.Spacer(height=20)
            ui.Spacer()
        # X axis labels (left=min, middle=mid, right=max) along the bottom edge.
        with ui.VStack():
            ui.Spacer()
            with ui.HStack(height=0):
                ui.Spacer(width=5)
                ui.Label(f"{_fmt(self._x_min)}{self._x_unit}", width=0, style=lbl_style)
                ui.Spacer()
                ui.Label(f"{_fmt(x_mid)}{self._x_unit}", width=0, style=lbl_style)
                ui.Spacer()
                ui.Label(f"{_fmt(self._x_max)}{self._x_unit}", width=0, style=lbl_style)
                ui.Spacer(width=8)
            ui.Spacer(height=3)

    # ---- hover readout -----------------------------------------------------
    def _build_traces(self) -> None:
        """Build the traces, detached from hover, under the layer that owns the readout.

        ``ui.Plot`` draws through ImGui's plot primitive, which pops up a tooltip
        of its own reporting one unlabeled sample index and value the moment ImGui
        considers it hovered.  Each trace is a separate plot, so that popup names
        whichever single trace ImGui resolved last and contradicts the readout
        beside it.  ``separate_window`` draws the traces into an ImGui window of
        their own, and ImGui only hover-tests items belonging to the window the
        pointer is actually over, so no trace is ever hovered and none of them
        emits a tooltip.  Nothing at the omni.ui level suppresses it: mouse-event
        routing options such as ``send_mouse_events_to_back`` decide which widget
        omni.ui hands its own callbacks to and leave ImGui's hover test alone.
        """
        with ui.ZStack():
            with ui.Frame(separate_window=True):
                with ui.ZStack():
                    self._build_plots()
            self._build_readout_layer()

    def _build_readout_layer(self) -> None:
        """Build the transparent rectangle that reports the pointer entering the plot.

        ``mouse_hovered_fn`` is the only hover signal omni.ui offers and it
        carries no pointer position, so it starts and stops the per-frame poll in
        :meth:`_update_readout` rather than filling the readout in itself.  The
        rectangle is invisible; it exists to take the pointer from the traces and
        to give the readout a rect to resolve the pointer against.

        Anything that rebuilds the plot -- a legend toggle, an axis change --
        replaces this rectangle, so a poll still running against the previous one
        is dropped here, and the readout is brought back into step with whatever
        the chart now draws.
        """
        self._stop_pointer_tracking()
        self._refresh_readout()
        self._readout_frame = ui.Rectangle(style={"background_color": 0x0}, opaque_for_mouse_events=True)
        self._readout_frame.set_mouse_hovered_fn(self._on_readout_hovered)

    def _build_readout_section(self) -> None:
        """Build the collapsible section the hovered readout is reported in.

        The readout is a section of the chart's own rather than a popup over the
        plot, so it cannot cover the traces it reports and is always read in the
        same place.  It also cannot be a tooltip: omni.ui builds a tooltip once
        and caches it for the life of the widget, which pins its figures to
        wherever the pointer first crossed into the plot.  Labels are rewritten as
        the pointer moves instead, and this is the chart's only readout -- no
        tooltip is set anywhere on the plot, and :meth:`_build_traces` keeps the
        traces out of ImGui's hover test, so nothing else can appear.

        Closed by default, because the table is a row per joint and a ten-joint
        selection is a tall panel to scroll past for a user who is not inspecting
        values.  It is opened deliberately and stays as it was left: it is not
        tied to the pointer, which would open and close it as the mouse crossed
        the plot and shift everything under it each time.

        It is built here rather than inside ``_plot_frame`` -- which a slider drag,
        a legend toggle and Fit Frame View all rebuild -- and reads its state from
        `_readout_collapsed`, so neither a rebuild of the plot nor of the whole
        card springs it back open.

        It sits below the plot and its axis controls, so opening it leaves the
        plot viewport and every control of the chart exactly where they were.
        """
        with ui.HStack(height=0):
            ui.Spacer(width=12)
            self._readout_section = CollapsableFrame(
                self.READOUT_TITLE,
                collapsed=self._readout_collapsed,
                build_fn=self._build_readout_table,
                show_copy_button=False,
            )
            self._readout_section.frame.set_collapsed_changed_fn(self._on_readout_collapsed)
            ui.Spacer(width=12)

    def _build_readout_table(self) -> None:
        """Fill the readout section with a row per joint and a column per legend group.

        A row per joint rather than a line per series, so a joint's Command and
        Observed figures are read side by side and its name is stated once instead
        of once per group -- which is what ran a multi-joint selection out of room.
        Charts whose series do not share a timeline get a row per series and an x
        column of their own, because there is no single x to head the table with;
        the dt sweep is the one that does this, dropping timestep levels per joint.

        The table is laid out for the series the chart reports right now, so it is
        as tall as the selection needs and no entry is dropped for want of room.
        Only a change of selection, of legend visibility or of the axis window
        rebuilds it -- never pointer movement, which writes into the cells built
        here -- so the section cannot resize while it is being read.

        A chart with nothing to report builds no table at all, so an empty chart
        shows its header and nothing else.
        """
        self._readout_x_label = None
        self._readout_x_shown = ""
        self._readout_cells = []
        self._readout_shown = []
        self._readout_built_shape = self._readout_shape()
        titles, rows = self._readout_layout()
        if not rows:
            return
        with ui.ZStack(
            height=0,
            style={
                "Label::readout_x": {"color": CHART_READOUT_TEXT, "font_size": 12},
                "Label::readout_head": {"color": MUTED_LABEL_COLOR, "font_size": 11},
                "Label::readout_joint": {"color": LABEL_COLOR, "font_size": 12},
                "Label::readout_value": {"color": CHART_READOUT_TEXT, "font_size": 12},
            },
        ):
            ui.Rectangle(
                style={
                    "background_color": CHART_READOUT_BG,
                    "border_color": CHART_STROKE,
                    "border_width": 1,
                    "border_radius": 2,
                }
            )
            with ui.VStack(spacing=2, height=0):
                ui.Spacer(height=6)
                with ui.HStack(height=self.READOUT_ROW_H):
                    ui.Spacer(width=8)
                    self._readout_x_label = ui.Label("", name="readout_x", width=self.READOUT_NAME_W)
                    for title in titles:
                        ui.Label(title, name="readout_head", elided_text=True)
                    ui.Spacer(width=8)
                for joint_name, cells in rows:
                    row_labels = []
                    with ui.HStack(height=self.READOUT_ROW_H):
                        ui.Spacer(width=8)
                        ui.Label(joint_name, name="readout_joint", width=self.READOUT_NAME_W, elided_text=True)
                        for index, is_x in cells:
                            row_labels.append((index, is_x, ui.Label("", name="readout_value", elided_text=True)))
                        ui.Spacer(width=8)
                    self._readout_cells.append(row_labels)
                    self._readout_shown.append([""] * len(row_labels))
                ui.Spacer(height=6)

    def _series_label(self, index: int) -> str:
        """Return the readout name for a series index."""
        if index < len(self._series_labels):
            return str(self._series_labels[index])
        group = index // self._group_num
        legend = self._legends[group] if group < len(self._legends) else f"Series {group}"
        if self._group_num > 1:
            return f"{legend} {index % self._group_num}"
        return legend

    def _joint_label(self, joint: int) -> str:
        """Return the readout's row name for a joint index."""
        if joint < len(self._joint_labels):
            return str(self._joint_labels[joint])
        return self._series_label(joint)

    def _legend_label(self, group: int) -> str:
        """Return the readout's column name for a legend group index."""
        if group < len(self._legends):
            return str(self._legends[group])
        return f"Series {group}"

    def _reported_series(self) -> list[int]:
        """Return the indices of the series the readout reports.

        A series is reported when its legend group is shown and it has at least
        two samples inside the current window, matching what :meth:`_build_plots`
        draws.  Independent of where the pointer is, so it is also what decides
        the shape of the readout table.
        """
        reported: list[int] = []
        for i, ya in enumerate(self._y_data):
            group = i // self._group_num
            if group < len(self._group_visible) and not self._group_visible[group]:
                continue
            xa = self._x_data[i] if i < len(self._x_data) else None
            i0, i1 = self._window_bounds(xa, ya)
            if i1 - i0 < 2:
                continue
            reported.append(i)
        return reported

    def _shares_timeline(self) -> bool:
        """Whether every series is plotted against the same x samples.

        Decided from the data, not from the x values sampled under the pointer, so
        that a chart cannot change the shape of its readout as the pointer moves
        across it.  Resolved once and kept: the data does not change for the life
        of the widget.
        """
        if self._shared_timeline is None:
            first = self._x_data[0] if self._x_data else None
            self._shared_timeline = first is not None and all(
                xa.shape == first.shape and np.array_equal(xa, first) for xa in self._x_data[1:]
            )
        return self._shared_timeline

    def _readout_shape(self) -> tuple:
        """Return what the readout table's rows and columns are built from.

        The table is rebuilt when this changes -- a legend toggle, a different
        selection, a window that zoomed a series out of range -- and only then.
        """
        return (tuple(self._reported_series()), self._shares_timeline(), tuple(self._group_visible))

    def _readout_layout(self) -> tuple[list[str], list[tuple[str, list[tuple[int, bool]]]]]:
        """Return the readout table's column titles and rows.

        Each row is its name and one cell per column, each cell naming the series
        it reports and whether it reports that series' x rather than its value.

        Returns:
            The titles and the rows, both empty when there is nothing to report.
        """
        reported = self._reported_series()
        if not reported:
            return [], []
        if not self._shares_timeline():
            # No single x to head the table with, so every series states its own.
            return ["x", "Value"], [(self._series_label(i), [(i, True), (i, False)]) for i in reported]
        if self._group_num == 1 and not self._joint_labels:
            # A group per series and no joint to name a row with -- the dt sweep,
            # whose legend groups are the joints -- so the series are the rows.
            return ["Value"], [(self._series_label(i), [(i, False)]) for i in reported]
        groups = [g for g in range(self._header_count) if g >= len(self._group_visible) or self._group_visible[g]]
        titles = [self._legend_label(g) for g in groups]
        rows: list[tuple[str, list[tuple[int, bool]]]] = []
        for joint in range(self._group_num):
            indices = [g * self._group_num + joint for g in groups]
            if not any(i in reported for i in indices):
                continue
            rows.append((self._joint_label(joint), [(i, False) for i in indices]))
        return titles, rows

    def _sample_at_frac(self, index: int, frac: float) -> tuple[float, float] | None:
        """Return one series' ``(value, x)`` under the pointer, or None when it has none.

        ``ui.Plot`` is given only y values, so it spreads the samples it is handed
        evenly across the rect it was given, whatever their timestamps are.  The
        pointer is therefore resolved in sample-index space, which is what is
        actually drawn.  Mapping it linearly in time instead misreads every chart
        whose samples are not evenly spaced -- the dt sweep plots logarithmic
        timesteps -- and drifts on any zoomed window, whose first sample sits at
        the left edge rather than at ``x_min``.

        That rect is not always the whole viewport.  A series clipped where its
        response diverged is drawn only across the stretch of the axis its samples
        cover, so its index space is inverted over that stretch alone.  Where the
        series has nothing drawn under the pointer it reports nothing, rather than
        pinning its last sample across the empty remainder of the chart.

        Args:
            index: Series index into ``y_data``.
            frac: Pointer position across the plot viewport, 0.0 at the left edge
                and 1.0 at the right.

        Returns:
            The interpolated value and the x it was read at, or None when the
            series is not drawn under the pointer.
        """
        i0, i1, offset, width = self._series_placement(index)
        if i1 - i0 < 2:
            return None
        # Invert exactly the placement `_build_plots` drew this series with, so the
        # pointer lands on the sample that is under it rather than on the one that
        # would have been under it had the series spanned the full width.
        local = (frac - offset) / width
        if local < -_POINTER_EDGE_SLACK or local > 1.0 + _POINTER_EDGE_SLACK:
            return None
        pos = min(max(local, 0.0), 1.0) * (i1 - i0 - 1)
        lo = int(pos)
        hi = min(lo + 1, i1 - i0 - 1)
        weight = pos - lo
        ya = self._y_data[index]
        xa = self._x_data[index] if index < len(self._x_data) else None

        def _at(arr):
            return float(arr[i0 + lo]) * (1.0 - weight) + float(arr[i0 + hi]) * weight

        return _at(ya), _at(xa)

    def _series_readings_at_frac(self, frac: float) -> list[tuple[int, float, float]]:
        """Return ``(series index, value, x)`` for every series under the pointer.

        Series hidden by a legend checkbox, series with fewer than two samples in
        the window, and series whose drawn stretch of the axis does not reach the
        pointer are all skipped, matching what :meth:`_build_plots` draws.

        Each series carries its own x because they need not share one.  The
        Position and Effort charts plot every trace against the same timeline, but
        the dt sweep drops levels per joint, so two series at the same pixel can
        sit at different timesteps.

        Args:
            frac: Pointer position across the plot viewport, 0.0 at the left edge
                and 1.0 at the right.

        Returns:
            One entry per reported series, in series order.
        """
        frac = max(0.0, min(1.0, frac))
        readings: list[tuple[int, float, float]] = []
        for i in self._reported_series():
            sample = self._sample_at_frac(i, frac)
            if sample is None:
                continue
            readings.append((i, sample[0], sample[1]))
        return readings

    def _series_values_at_frac(self, frac: float) -> list[tuple[str, float, float]]:
        """Return ``(label, value, x)`` for every series visible under the pointer."""
        return [(self._series_label(i), value, x) for i, value, x in self._series_readings_at_frac(frac)]

    def _readout_parts(self, frac: float) -> tuple[str, list[tuple[str, str]]]:
        """Return the x heading and one ``(label, value)`` pair per reported series.

        A shared x is stated once, as the heading.  When the series disagree the
        heading is empty and each value carries the x it was read at instead, so a
        dt sweep that skipped different levels for different joints cannot label
        one joint's value with another's timestep.

        Args:
            frac: Pointer position across the plot viewport, 0.0 at the left edge
                and 1.0 at the right.

        Returns:
            The heading and the pairs, both empty when no series is reported.
        """
        readings = self._series_values_at_frac(frac)
        if not readings:
            return "", []
        x_values = [x for _, _, x in readings]
        shared_x = all(math.isclose(x, x_values[0], rel_tol=1e-9, abs_tol=1e-12) for x in x_values)
        if shared_x:
            heading = f"x = {_fmt_readout(x_values[0])} {self._x_unit}"
            return heading, [(label, f"{_fmt_readout(value)} {self._y_unit}") for label, value, _ in readings]
        return "", [
            (label, f"{_fmt_readout(value)} {self._y_unit}  @ x = {_fmt_readout(x)} {self._x_unit}")
            for label, value, x in readings
        ]

    def _readout_text(self, frac: float) -> str:
        """Return the multi-line readout text at ``frac``, or "" when empty.

        The labels are padded to a common width so the values line up in a column.
        """
        heading, entries = self._readout_parts(frac)
        if not entries:
            return ""
        width = max(len(label) for label, _ in entries)
        lines = [heading, ""] if heading else []
        lines += [f"{label.ljust(width)} : {value}" for label, value in entries]
        return "\n".join(lines)

    def _on_readout_collapsed(self, collapsed: bool) -> None:
        """Remember the section's state, and only read the pointer while it is open.

        Kept on the widget so that the next rebuild of the chart restores it
        rather than reverting to closed.  A closed section shows nothing to
        update, so the per-frame poll is dropped for as long as it stays closed.

        Args:
            collapsed: The section's new state.
        """
        self._readout_collapsed = collapsed
        if collapsed:
            self._stop_pointer_tracking()
        else:
            # The pointer may already be inside the plot, and no further hover
            # event is coming if it is; the poll stops itself if it is not.
            self._start_pointer_tracking()

    def _on_readout_hovered(self, hovered: bool) -> None:
        """Read the pointer while it is inside the plot, and clear the readout once it leaves."""
        if hovered:
            self._start_pointer_tracking()
        else:
            self._stop_pointer_tracking()

    def _start_pointer_tracking(self) -> None:
        """Begin polling the pointer every frame, if it is not already being polled.

        Nothing is polled while the readout is closed: there is no cell on screen
        for a reading to be written into.

        The subscription belongs to this chart and lives only for the duration of
        a hover, rather than hanging off the panel's render step: the readout is
        per chart and a panel holds several, and a poll scoped to the hover costs
        nothing at all on the frames the pointer is elsewhere.
        """
        if self._pointer_sub is not None or self._readout_collapsed:
            return
        self._pointer_sub = (
            omni.kit.app.get_app()
            .get_update_event_stream()
            .create_subscription_to_pop(self._on_update_step, name="isaacsim.robot_setup.gain_tuner.ui chart readout")
        )
        self._update_readout()

    def _stop_pointer_tracking(self) -> None:
        """Stop polling the pointer and leave no reading behind for a pointer that has left."""
        self._pointer_sub = None
        self._clear_readout()

    def _on_update_step(self, event: carb.events.IEvent) -> None:
        """Rewrite the readout for wherever the pointer has moved to since the last frame."""
        self._update_readout()

    def _refresh_readout(self) -> None:
        """Rebuild the readout table if it no longer matches the series being drawn.

        Called from every rebuild of the plot, which is the one path a legend
        toggle, a slider drag and Fit Frame View all share, so the table follows
        the chart without each of them having to know about it.
        """
        if self._readout_collapsed or self._readout_section is None:
            return
        if self._readout_shape() != self._readout_built_shape:
            self._readout_section.frame.rebuild()

    def _update_readout(self) -> None:
        """Report where the pointer is now, or clear the readout.

        The pointer leaving the plot is what ends a hover, so it also stops the
        poll -- a hover ending without a leave event, which a pointer moved fast
        enough out of the window can do, is caught here rather than leaving a
        reading in the table for a pointer that is somewhere else entirely.
        """
        frame = self._readout_frame
        if frame is None:
            self._stop_pointer_tracking()
            return
        pointer = _pointer_screen_pos()
        # -1 because the offset is zero-based over `computed_width` pixels.
        width = frame.computed_width - 1
        height = frame.computed_height
        if pointer is None or width <= 0 or height <= 0:
            # Not laid out yet, or the pointer cannot be located: there is nothing
            # to resolve the pointer against, so report nothing and try again next
            # frame rather than reading a position off an unplaced rect.
            self._clear_readout()
            return
        offset_x = pointer[0] - frame.screen_position_x
        offset_y = pointer[1] - frame.screen_position_y
        if not (0.0 <= offset_x <= width and 0.0 <= offset_y <= height):
            self._stop_pointer_tracking()
            return
        if self._readout_shape() != self._readout_built_shape:
            # The chart changed under a table built for something else.  Rebuilding
            # is deferred a frame by omni.ui, so this frame reports nothing rather
            # than writing readings into the wrong rows.
            self._refresh_readout()
            return
        self._write_readout(self._series_readings_at_frac(offset_x / width))

    def _clear_readout(self) -> None:
        """Empty every cell, so no reading outlives the hover that produced it."""
        self._write_readout([])

    def _write_readout(self, readings: list[tuple[int, float, float]]) -> None:
        """Write ``readings`` into the table's cells.

        Only the cells whose text changed are assigned, so a pointer resting
        between two samples costs nothing but the comparisons.  An empty
        ``readings`` empties the table instead.

        Args:
            readings: ``(series index, value, x)`` per reported series.
        """
        by_index = {index: (value, x) for index, value, x in readings}
        if self._readout_x_label is not None:
            x_text = ""
            if readings and self._shares_timeline():
                x_text = f"x = {_fmt_readout(readings[0][2])} {self._x_unit}"
            if x_text != self._readout_x_shown:
                self._readout_x_label.text = x_text
                self._readout_x_shown = x_text
        for cells, shown in zip(self._readout_cells, self._readout_shown):
            for slot, (index, is_x, label) in enumerate(cells):
                reading = by_index.get(index)
                if not readings:
                    text = ""
                elif reading is None:
                    text = self.READOUT_NO_VALUE
                elif is_x:
                    text = f"{_fmt_readout(reading[1])} {self._x_unit}"
                else:
                    text = f"{_fmt_readout(reading[0])} {self._y_unit}"
                if text != shown[slot]:
                    label.text = text
                    shown[slot] = text

    # ---- X slider row ------------------------------------------------------
    def _build_x_row(self):
        with ui.HStack(spacing=19, height=22):
            # Align the row under the plot area (past the Y slider column + gaps).
            ui.Spacer(width=99)
            self._x_min_model = self._value_field(self._x_min, self._x_unit, self._on_x_min_commit)
            with ui.HStack():
                ui.Spacer()
                self._x_slider = _RangeSlider(
                    length_px=self.X_SLIDER_W,
                    vertical=False,
                    data_min=self._x_data_min,
                    data_max=self._x_data_max,
                    lo=self._x_min,
                    hi=self._x_max,
                    on_change=self._on_x_slider,
                )
                self._x_slider.build()
                ui.Spacer()
            self._x_max_model = self._value_field(self._x_max, self._x_unit, self._on_x_max_commit)
            ui.Spacer(width=6)

    # ---- footer ------------------------------------------------------------
    def _build_footer(self):
        with ui.HStack(height=0):
            ui.Spacer(width=12)
            with ui.VStack(width=0):
                ui.Spacer(height=4)
                ui.Button(
                    "Fit Frame View",
                    height=22,
                    width=0,
                    clicked_fn=self._on_fit_frame_view,
                    style={
                        "Button": {
                            "background_color": CHART_BTN_BG,
                            "border_radius": 2,
                            "padding": 4,
                        },
                        "Button.Label": {"color": LABEL_COLOR, "font_size": 12},
                    },
                )
                ui.Spacer(height=12)
            # A clipped trace ends mid-chart on framed axes, which reads as missing
            # data unless both the reason and the way back are stated.
            if self._diverged:
                ui.Spacer(width=12)
                with ui.VStack(width=0):
                    ui.Spacer(height=9)
                    ui.Label(
                        "Trace clipped where the response diverged; axes framed on what remains. "
                        "Fit Frame View shows the full run.",
                        style={"color": CHART_UNIT_COLOR, "font_size": 12},
                    )
            ui.Spacer()

    # ---- value fields ------------------------------------------------------
    def _value_field(self, value: float, unit: str, on_commit: Callable[[float], None]):
        """A #202020 rounded field with an editable number and a muted unit suffix."""
        with ui.ZStack(width=self.FIELD_W, height=22):
            ui.Rectangle(style={"background_color": CHART_VALUE_FIELD_BG, "border_radius": 2})
            with ui.HStack():
                ui.Spacer(width=4)
                field = ui.FloatField(
                    height=20,
                    style={
                        "background_color": 0x0,
                        "color": LABEL_COLOR,
                        "font_size": 12,
                    },
                )
                field.model.set_value(value)
                field.model.add_end_edit_fn(lambda m, cb=on_commit: cb(m.get_value_as_float()))
                ui.Label(
                    unit,
                    width=0,
                    style={"color": CHART_UNIT_COLOR, "font_size": 12},
                )
                ui.Spacer(width=4)
        return field.model

    # ---- range change handlers --------------------------------------------
    def _on_y_slider(self, lo: float, hi: float):
        self._y_min, self._y_max = lo, hi
        if self._y_min_model:
            self._y_min_model.set_value(lo)
        if self._y_max_model:
            self._y_max_model.set_value(hi)
        if self._plot_frame:
            self._plot_frame.rebuild()

    def _on_x_slider(self, lo: float, hi: float):
        self._x_min, self._x_max = lo, hi
        if self._x_min_model:
            self._x_min_model.set_value(lo)
        if self._x_max_model:
            self._x_max_model.set_value(hi)
        if self._plot_frame:
            self._plot_frame.rebuild()

    def _on_y_max_commit(self, v: float):
        self._y_max = max(v, self._y_min + 1e-3)
        self._sync_y()

    def _on_y_min_commit(self, v: float):
        self._y_min = min(v, self._y_max - 1e-3)
        self._sync_y()

    def _on_x_min_commit(self, v: float):
        self._x_min = min(v, self._x_max - 1e-3)
        self._sync_x()

    def _on_x_max_commit(self, v: float):
        self._x_max = max(v, self._x_min + 1e-3)
        self._sync_x()

    def _sync_y(self):
        if self._y_slider:
            self._y_slider.set_values(self._y_min, self._y_max)
        if self._plot_frame:
            self._plot_frame.rebuild()

    def _sync_x(self):
        if self._x_slider:
            self._x_slider.set_values(self._x_min, self._x_max)
        if self._plot_frame:
            self._plot_frame.rebuild()

    def _on_fit_frame_view(self):
        self._x_min, self._x_max = self._x_data_min, self._x_data_max
        self._y_min, self._y_max = self._y_data_min, self._y_data_max
        if self._y_min_model:
            self._y_min_model.set_value(self._y_min)
        if self._y_max_model:
            self._y_max_model.set_value(self._y_max)
        if self._x_min_model:
            self._x_min_model.set_value(self._x_min)
        if self._x_max_model:
            self._x_max_model.set_value(self._x_max)
        if self._y_slider:
            self._y_slider.set_values(self._y_min, self._y_max)
        if self._x_slider:
            self._x_slider.set_values(self._x_min, self._x_max)
        if self._plot_frame:
            self._plot_frame.rebuild()
