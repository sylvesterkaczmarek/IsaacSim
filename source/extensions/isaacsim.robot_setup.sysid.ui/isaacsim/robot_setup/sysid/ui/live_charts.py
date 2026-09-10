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

"""Live cost plot, rollout comparison, and run status for SysId optimization."""

from __future__ import annotations

import ctypes
import math
from typing import Optional

import carb
import carb.input
import omni.appwindow
import omni.kit.app
import omni.ui as ui
from isaacsim.gui.components.element_wrappers import TextBlock

from ..rollout_plot_data import SysIdRolloutPlotData

_COST_COLOR = [100, 200, 100]
_ERROR_COLORS = [
    [180, 180, 255],
    [255, 180, 120],
    [120, 220, 160],
    [220, 140, 200],
    [200, 200, 120],
    [140, 200, 220],
    [220, 160, 140],
    [160, 140, 220],
]

_JOINT_ERROR_PLOT_HEIGHT = 8 * 23
_COST_PLOT_HEIGHT = 150
_COST_PLOT_CONTAINER_HEIGHT = _COST_PLOT_HEIGHT + 44
_AXIS_LABEL_WIDTH = 64
_CURSOR_BOX_COLUMN_WIDTH = 260
_CURSOR_BOX_ROW_HEIGHT = 18
_CURSOR_COLOR = 0xFF62AEEF
_GRID_COLOR = 0xFF303030


class _NativeCursorPoint(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _native_cursor_position() -> Optional[tuple[float, float]]:
    """Return the desktop cursor position when the host exposes a native API.

    Returns:
        The ``(x, y)`` desktop cursor position in physical pixels, or ``None``
        when no native cursor API is available.
    """
    try:
        user32 = ctypes.windll.user32
    except (AttributeError, OSError):
        return None
    point = _NativeCursorPoint()
    if not user32.GetCursorPos(ctypes.byref(point)):
        return None
    return float(point.x), float(point.y)


def _ui_dpi_scale() -> float:
    """Return the UI DPI scale, or ``1.0`` when it cannot be resolved.

    ``GetCursorPos`` reports physical desktop pixels while omni.ui screen
    positions are in DPI-scaled (logical) units, so the native cursor must be
    divided by this to hit-test against widget bounds.

    Returns:
        The UI DPI scale factor, or ``1.0`` when it cannot be resolved.
    """
    try:
        scale = float(ui.Workspace.get_dpi_scale())
    except Exception:  # noqa: BLE001 - any failure falls back to no scaling
        return 1.0
    return scale if scale > 0 else 1.0


def _rgb_to_ui_color(rgb: list[int]) -> int:
    r, g, b = [max(0, min(255, int(v))) for v in rgb[:3]]
    return 0xFF * 16**6 + b * 16**4 + g * 16**2 + r


def _format_axis_value(value: float) -> str:
    if not math.isfinite(float(value)):
        return "0"
    return f"{float(value):.3g}"


class _JointErrorPlot:
    """Multi-series joint plot with a Gain-Tuner-style cursor inspector.

    Args:
        title: Plot title.
        unit: Value unit label.
    """

    def __init__(self, title: str, unit: str = "") -> None:
        self._title = title
        self._unit = unit
        self._times: list[float] = []
        self._series: list[list[float]] = []
        self._legends: list[str] = []
        self._colors: list[list[int]] = []
        self._y_min = -1.0
        self._y_max = 1.0
        self._container_frame: Optional[ui.Frame] = None
        self._base_plot: Optional[ui.Plot] = None
        self._cursor_surface: Optional[ui.Plot] = None
        self._cursor_overlay: Optional[ui.Frame] = None
        self._cursor_line_placer: Optional[ui.Placer] = None
        self._cursor_box_placer: Optional[ui.Placer] = None
        self._cursor_time_label: Optional[ui.Label] = None
        self._cursor_value_labels: list[Optional[ui.Label]] = []
        self._cursor_groups: list[tuple[str, list[int], list[int]]] = []
        self._last_cursor_index: Optional[int] = None
        self._input_interface = None
        self._mouse = None
        self._hover_update_subscription = None

    def build(self) -> ui.Frame:
        """Handle build.

        Returns:
            The resulting value.
        """
        self._container_frame = ui.Frame(build_fn=self._build_widget)
        return self._container_frame

    def set_data(
        self,
        times: list[float],
        series: list[list[float]],
        legends: list[str],
        colors: list[list[int]],
    ) -> None:
        """Set data.

        Args:
            times: Times value.
            series: Series value.
            legends: Legends value.
            colors: Colors value.
        """
        self._times = list(times)
        self._series = [list(row) for row in series]
        self._legends = list(legends)
        self._colors = [list(color) for color in colors]
        flat = [float(value) for row in self._series for value in row]
        self._y_min, self._y_max = _padded_bounds(flat, -1.0, 1.0)
        if self._container_frame is not None:
            self._container_frame.rebuild()

    def clear(self) -> None:
        """Handle clear."""
        self._times = []
        self._series = []
        self._legends = []
        self._colors = []
        self._y_min = -1.0
        self._y_max = 1.0
        # The rebuild below early-returns (no series) without recreating the cursor
        # surface, so release the per-frame hover subscription and drop the cursor
        # refs. Otherwise the poll keeps dereferencing a Plot widget the rebuild
        # already destroyed until the next set_data recreates it.
        self._release_hover_polling()
        self._cursor_surface = None
        self._cursor_overlay = None
        self._cursor_line_placer = None
        self._cursor_box_placer = None
        self._cursor_time_label = None
        self._cursor_value_labels = []
        self._cursor_groups = []
        self._last_cursor_index = None
        if self._container_frame is not None:
            self._container_frame.rebuild()

    def cleanup(self) -> None:
        """Release resources."""
        self._times = []
        self._series = []
        self._legends = []
        self._colors = []
        self._container_frame = None
        self._base_plot = None
        self._cursor_surface = None
        self._cursor_overlay = None
        self._cursor_line_placer = None
        self._cursor_box_placer = None
        self._cursor_time_label = None
        self._cursor_value_labels = []
        self._cursor_groups = []
        self._last_cursor_index = None
        self._release_hover_polling()

    def _build_widget(self) -> None:
        with ui.VStack(spacing=3):
            with ui.HStack(height=18):
                ui.Label(self._title, style={"font_size": 13})
                ui.Spacer()
                ui.Label(
                    "Hover for joint values",
                    width=160,
                    alignment=ui.Alignment.RIGHT,
                    style={"font_size": 12, "color": 0xFF999999},
                )
            if not self._series:
                ui.Label(
                    f"No {self._title.lower()} data is available for the final rollout.",
                    height=_JOINT_ERROR_PLOT_HEIGHT,
                    alignment=ui.Alignment.CENTER,
                    style={"color": 0xFF7F7F7F},
                )
                return

            with ui.HStack(spacing=4):
                with ui.VStack(width=64, height=_JOINT_ERROR_PLOT_HEIGHT):
                    ui.Label(_format_axis_value(self._y_max), alignment=ui.Alignment.RIGHT_TOP)
                    ui.Spacer()
                    ui.Label(_format_axis_value(self._y_min), alignment=ui.Alignment.RIGHT_BOTTOM)

                with ui.ZStack(height=_JOINT_ERROR_PLOT_HEIGHT):
                    ui.Rectangle(
                        height=_JOINT_ERROR_PLOT_HEIGHT,
                        style={
                            "background_color": 0xFF202020,
                            "border_color": 0xFF3A3A3A,
                            "border_width": 1,
                            "border_radius": 3,
                        },
                    )
                    self._build_grid()
                    # The Gain Tuner uses a base plot to establish stable plot geometry
                    # before it adds the series and the top-level pointer surface.
                    self._base_plot = ui.Plot(
                        ui.Type.LINE,
                        0.0,
                        1.0,
                        0.0,
                        height=_JOINT_ERROR_PLOT_HEIGHT,
                        style={"color": 0x00FFFFFF, "background_color": 0x00000000},
                    )
                    for idx, row in enumerate(self._series):
                        color = (
                            self._colors[idx] if idx < len(self._colors) else _ERROR_COLORS[idx % len(_ERROR_COLORS)]
                        )
                        ui.Plot(
                            ui.Type.LINE,
                            self._y_min,
                            self._y_max,
                            *row,
                            height=_JOINT_ERROR_PLOT_HEIGHT,
                            style={
                                "color": _rgb_to_ui_color(color),
                                "background_color": 0x00000000,
                            },
                        )
                    self._cursor_groups = _group_cursor_series(self._legends, self._colors)
                    self._cursor_overlay = ui.Frame(visible=False)
                    with self._cursor_overlay:
                        self._build_cursor_overlay()

                    # Keep a full-size top plot as the geometry surface. Passive
                    # pointer movement comes from carb.input because ui.Plot's
                    # mouse_moved callback is drag-oriented in this Kit build.
                    self._cursor_surface = ui.Plot(
                        width=ui.Fraction(1),
                        height=_JOINT_ERROR_PLOT_HEIGHT,
                        style={"color": 0xFFFFFFFF, "background_color": 0x00000000},
                    )
                    self._cursor_surface.set_computed_content_size_changed_fn(self._on_plot_size_changed)
                    self._ensure_hover_polling()

            if self._times:
                with ui.HStack(spacing=4):
                    ui.Label(f"{_format_axis_value(self._times[0])}s", width=70)
                    ui.Spacer()
                    ui.Label("Time (s)", alignment=ui.Alignment.CENTER)
                    ui.Spacer()
                    ui.Label(f"{_format_axis_value(self._times[-1])}s", width=70, alignment=ui.Alignment.RIGHT)

            with ui.HStack(spacing=8):
                for idx, legend in enumerate(self._legends):
                    color = self._colors[idx] if idx < len(self._colors) else _ERROR_COLORS[idx % len(_ERROR_COLORS)]
                    ui.Rectangle(width=8, height=8, style={"background_color": _rgb_to_ui_color(color)})
                    ui.Label(legend, width=70)

    def _build_grid(self, rows: int = 4, columns: int = 6) -> None:
        with ui.VStack(spacing=0):
            for row in range(rows + 1):
                ui.Rectangle(height=1, style={"background_color": _GRID_COLOR})
                if row < rows:
                    ui.Spacer(height=ui.Fraction(1))
        with ui.HStack(spacing=0):
            for column in range(columns + 1):
                ui.Rectangle(width=1, style={"background_color": _GRID_COLOR})
                if column < columns:
                    ui.Spacer(width=ui.Fraction(1))

    def _build_cursor_overlay(self) -> None:
        group_count = len(self._cursor_groups)
        column_count = 2 if group_count > 7 else 1
        row_count = max(1, math.ceil(group_count / column_count))
        box_width = _CURSOR_BOX_COLUMN_WIDTH * column_count
        box_height = 34 + row_count * _CURSOR_BOX_ROW_HEIGHT

        self._cursor_line_placer = ui.Placer(offset_x=0, offset_y=0)
        with self._cursor_line_placer:
            ui.Rectangle(
                width=1,
                height=_JOINT_ERROR_PLOT_HEIGHT,
                style={"background_color": _CURSOR_COLOR},
            )

        self._cursor_box_placer = ui.Placer(offset_x=10, offset_y=7)
        with self._cursor_box_placer:
            with ui.ZStack(width=box_width, height=box_height):
                ui.Rectangle(
                    style={
                        "background_color": 0xF22A2A2A,
                        "border_color": _CURSOR_COLOR,
                        "border_width": 1,
                        "border_radius": 4,
                    }
                )
                with ui.VStack(spacing=0):
                    ui.Spacer(height=6)
                    with ui.HStack(height=20):
                        ui.Spacer(width=8)
                        self._cursor_time_label = ui.Label(
                            "Time: 0 s",
                            style={"font_size": 12, "color": 0xFFE5E5E5},
                        )
                        ui.Spacer(width=8)
                    self._cursor_value_labels = [None] * group_count
                    for row in range(row_count):
                        with ui.HStack(height=_CURSOR_BOX_ROW_HEIGHT, spacing=0):
                            for column in range(column_count):
                                group_index = column * row_count + row
                                with ui.HStack(width=ui.Fraction(1), spacing=5):
                                    ui.Spacer(width=8)
                                    if group_index < group_count:
                                        _name, _indices, color = self._cursor_groups[group_index]
                                        ui.Rectangle(
                                            width=7,
                                            height=7,
                                            style={
                                                "background_color": _rgb_to_ui_color(color),
                                                "border_radius": 4,
                                            },
                                        )
                                        label = ui.Label(
                                            "",
                                            width=0,
                                            style={"font_size": 12, "color": 0xFFD8D8D8},
                                        )
                                        self._cursor_value_labels[group_index] = label
                                    ui.Spacer(width=8)

    def _mouse_moved_on_plot(self, x: float, _y: float, *args: object) -> None:
        """Compatibility path for Kit builds that emit passive widget movement.

        Args:
            x: Cursor x position.
            _y: Cursor y position (unused).
            *args: Additional widget callback arguments.
        """
        self._update_cursor_from_screen_x(float(x))

    def _update_cursor_from_screen_x(self, x: float) -> None:
        """Move the vertical cursor and show every joint at the nearest sample.

        Args:
            x: Cursor screen x position.
        """
        if self._cursor_surface is None or not self._times or not self._series:
            return
        width = self._cursor_surface.computed_width - 14
        if width <= 0:
            return
        x_origin = self._cursor_surface.screen_position_x + 7
        fraction = max(0.0, min(1.0, (float(x) - x_origin) / width))
        sample_index = min(len(self._times) - 1, round(fraction * (len(self._times) - 1)))
        cursor_x = 7 + fraction * width
        if self._cursor_line_placer is not None:
            self._cursor_line_placer.offset_x = cursor_x

        group_count = len(self._cursor_groups)
        column_count = 2 if group_count > 7 else 1
        box_width = _CURSOR_BOX_COLUMN_WIDTH * column_count
        box_x = cursor_x + 10
        if box_x + box_width > self._cursor_surface.computed_width:
            box_x = max(0.0, cursor_x - box_width - 10)
        if self._cursor_box_placer is not None:
            self._cursor_box_placer.offset_x = box_x

        if sample_index != self._last_cursor_index:
            self._last_cursor_index = sample_index
            self._update_cursor_values(sample_index)
        if self._cursor_overlay is not None:
            self._cursor_overlay.visible = True

    def _ensure_hover_polling(self) -> None:
        if self._hover_update_subscription is not None:
            return
        try:
            app_window = omni.appwindow.get_default_app_window()
            self._mouse = app_window.get_mouse()
            self._input_interface = carb.input.acquire_input_interface()
            self._hover_update_subscription = (
                omni.kit.app.get_app()
                .get_update_event_stream()
                .create_subscription_to_pop(
                    self._on_hover_update,
                    name=f"sysid joint chart hover {self._title}",
                )
            )
        except (AttributeError, RuntimeError) as exc:
            carb.log_warn(f"SysId: failed to start joint-chart hover polling: {exc}")

    def _release_hover_polling(self) -> None:
        self._hover_update_subscription = None
        self._input_interface = None
        self._mouse = None

    def _on_hover_update(self, _event: object) -> None:
        if self._cursor_surface is None or self._input_interface is None or self._mouse is None:
            return
        native_coords = _native_cursor_position()
        if native_coords is not None:
            # Win32 GetCursorPos returns physical desktop pixels while omni.ui screen
            # positions are DPI-scaled logical units, so convert before hit-testing --
            # otherwise a scaled display offsets the inspector onto the chart below the
            # cursor. Scoped to this path only: carb's get_mouse_coords_pixel already
            # shares omni.ui's coordinate space (and on some hosts reports logical units),
            # so dividing it too would double-apply the scale.
            scale = _ui_dpi_scale()
            mouse_x, mouse_y = native_coords[0] / scale, native_coords[1] / scale
        else:
            try:
                coords = self._input_interface.get_mouse_coords_pixel(self._mouse)
            except (AttributeError, RuntimeError):
                return
            if hasattr(coords, "x"):
                mouse_x, mouse_y = float(coords.x), float(coords.y)
            else:
                mouse_x, mouse_y = float(coords[0]), float(coords[1])

        left = float(self._cursor_surface.screen_position_x)
        top = float(self._cursor_surface.screen_position_y)
        right = left + float(self._cursor_surface.computed_width)
        bottom = top + float(self._cursor_surface.computed_height)
        hovered = left <= mouse_x <= right and top <= mouse_y <= bottom
        if hovered:
            self._update_cursor_from_screen_x(mouse_x)
        elif self._cursor_overlay is not None:
            self._cursor_overlay.visible = False

    def _update_cursor_values(self, sample_index: int) -> None:
        if self._cursor_time_label is not None:
            self._cursor_time_label.text = f"Time: {float(self._times[sample_index]):.4g} s"
        for group_index, (name, series_indices, _color) in enumerate(self._cursor_groups):
            label = self._cursor_value_labels[group_index] if group_index < len(self._cursor_value_labels) else None
            if label is None:
                continue
            parts = []
            for series_index in series_indices:
                if series_index >= len(self._series) or sample_index >= len(self._series[series_index]):
                    continue
                value = float(self._series[series_index][sample_index])
                if not math.isfinite(value):
                    continue
                legend = self._legends[series_index] if series_index < len(self._legends) else ""
                suffix = _cursor_series_suffix(legend)
                value_text = f"{value:+.4g}"
                parts.append(f"{suffix} {value_text}".strip())
            unit_suffix = f" {self._unit}" if self._unit else ""
            short_name = name.replace("Joint ", "J")
            label.text = f"{short_name}: {' | '.join(parts) or 'n/a'}{unit_suffix}"

    def _on_plot_size_changed(self, *args: object) -> None:
        if self._last_cursor_index is not None and self._cursor_surface is not None:
            fraction = self._last_cursor_index / max(1, len(self._times) - 1)
            cursor_x = 7 + fraction * max(1, self._cursor_surface.computed_width - 14)
            if self._cursor_line_placer is not None:
                self._cursor_line_placer.offset_x = cursor_x


class _CostPlot:
    """Compact fixed-height line plot for live optimization cost."""

    def __init__(self) -> None:
        self._iterations: list[float] = []
        self._costs: list[float] = []
        self._container_frame: Optional[ui.Frame] = None

    def build(self) -> ui.Frame:
        """Handle build.

        Returns:
            The resulting value.
        """
        self._container_frame = ui.Frame(height=_COST_PLOT_CONTAINER_HEIGHT, build_fn=self._build_widget)
        return self._container_frame

    def set_data(self, iterations: list[float], costs: list[float]) -> None:
        """Set data.

        Args:
            iterations: Iterations value.
            costs: Costs value.
        """
        self._iterations = list(iterations)
        self._costs = list(costs)
        self.refresh()

    def ensure_visible(self) -> None:
        """Handle ensure visible."""
        if self._container_frame is not None:
            self._container_frame.enabled = True
            self._container_frame.visible = True

    def refresh(self) -> None:
        """Handle refresh."""
        if self._container_frame is not None:
            self._container_frame.rebuild()

    def cleanup(self) -> None:
        """Release resources."""
        self._iterations = []
        self._costs = []
        self._container_frame = None

    def _build_widget(self) -> None:
        x_data, y_data = _cost_plot_series(self._iterations, self._costs)
        x_min, x_max, y_min, y_max = _cost_plot_bounds(x_data, y_data)

        with ui.VStack(spacing=3):
            with ui.HStack(spacing=4, height=_COST_PLOT_HEIGHT):
                with ui.VStack(width=_AXIS_LABEL_WIDTH, height=_COST_PLOT_HEIGHT):
                    ui.Label(_format_axis_value(y_max), alignment=ui.Alignment.RIGHT_TOP)
                    ui.Spacer()
                    ui.Label(_format_axis_value(y_min), alignment=ui.Alignment.RIGHT_BOTTOM)

                with ui.ZStack(height=_COST_PLOT_HEIGHT):
                    ui.Rectangle(
                        height=_COST_PLOT_HEIGHT,
                        style={"background_color": 0xFF202020, "border_radius": 2},
                    )
                    if y_data:
                        ui.Plot(
                            ui.Type.LINE,
                            y_min,
                            y_max,
                            *y_data,
                            height=_COST_PLOT_HEIGHT,
                            style={
                                "color": _rgb_to_ui_color(_COST_COLOR),
                                "background_color": 0x00000000,
                            },
                        )
                    else:
                        ui.Label(
                            "Cost history appears when optimization starts.",
                            alignment=ui.Alignment.CENTER,
                            style={"color": 0xFF7F7F7F},
                        )

            with ui.HStack(spacing=4, height=20):
                ui.Spacer(width=_AXIS_LABEL_WIDTH + 4)
                ui.Label(_format_axis_value(x_min), width=70)
                ui.Spacer()
                ui.Label("Iteration", alignment=ui.Alignment.CENTER)
                ui.Spacer()
                ui.Label(_format_axis_value(x_max), width=70, alignment=ui.Alignment.RIGHT)

            with ui.HStack(spacing=6, height=18):
                ui.Spacer(width=_AXIS_LABEL_WIDTH + 4)
                ui.Rectangle(width=8, height=8, style={"background_color": _rgb_to_ui_color(_COST_COLOR)})
                ui.Label("Cost", width=60)


class SysIdLiveChartsWidget:
    """Cost plot while optimizing; per-joint error plots after the final rollout."""

    def __init__(self) -> None:
        self._built = False
        self._cost_iterations: list[float] = []
        self._cost_values: list[float] = []
        self._activity_text = "Idle"
        self._progress_fraction = 0.0
        self._num_joints = 1

        self._activity_label: Optional[TextBlock] = None
        self._progress_bar: Optional[ui.ProgressBar] = None
        self._cost_plot: Optional[_CostPlot] = None
        self._joint_plots_frame: Optional[ui.Frame] = None
        self._pos_error_plot: Optional[_JointErrorPlot] = None
        self._vel_error_plot: Optional[_JointErrorPlot] = None
        self._effort_plot: Optional[_JointErrorPlot] = None
        self.wrapped_ui_elements: list = []

    def build(self) -> None:
        """Handle build."""
        if self._built:
            return
        self._built = True

        with ui.VStack(spacing=6):
            self._activity_label = TextBlock(
                "Activity",
                self._activity_text,
                include_copy_button=False,
                num_lines=2,
            )
            self.wrapped_ui_elements.append(self._activity_label)

            self._progress_bar = ui.ProgressBar(height=10, width=ui.Fraction(1))
            self._progress_bar.model.set_value(self._progress_fraction)

            ui.Spacer(height=2)
            ui.Line(style={"color": 0x338A8777}, width=ui.Fraction(1), alignment=ui.Alignment.CENTER)
            ui.Spacer(height=2)

            ui.Label("Cost per iteration", style={"font_size": 14})
            self._cost_plot = _CostPlot()
            self._cost_plot.build()
            self.wrapped_ui_elements.append(self._cost_plot)

            self._joint_plots_frame = ui.Frame(visible=False)
            with self._joint_plots_frame:
                with ui.VStack(spacing=6):
                    ui.Label(
                        "Final rollout joint plots",
                        style={"font_size": 14},
                    )
                    self._pos_error_plot = _JointErrorPlot("Position error", "rad")
                    self._pos_error_plot.build()
                    self.wrapped_ui_elements.append(self._pos_error_plot)

                    self._vel_error_plot = _JointErrorPlot("Velocity error", "rad/s")
                    self._vel_error_plot.build()
                    self.wrapped_ui_elements.append(self._vel_error_plot)

                    self._effort_plot = _JointErrorPlot("Effort overlay (simulated and recorded)", "N m")
                    self._effort_plot.build()
                    self.wrapped_ui_elements.append(self._effort_plot)

    def set_num_joints(self, num_joints: int) -> None:
        """Set num joints.

        Args:
            num_joints: Num joints value.
        """
        self._num_joints = max(1, int(num_joints))

    def reset_for_run(self, num_joints: int) -> None:
        """Handle reset for run.

        Args:
            num_joints: Num joints value.
        """
        self._num_joints = max(1, int(num_joints))
        self._cost_iterations = []
        self._cost_values = []
        self.ensure_live_progress_visible()
        self.set_activity("Starting optimization...", 0.0)
        self._refresh_cost_plot()
        self._hide_joint_error_plots()

    def set_activity(self, text: str, progress_fraction: float) -> None:
        """Set activity.

        Args:
            text: Text to display.
            progress_fraction: Progress fraction value.
        """
        self._activity_text = text
        self._progress_fraction = max(0.0, min(1.0, float(progress_fraction)))
        if self._activity_label is not None:
            self._activity_label.set_text(text)
        if self._progress_bar is not None:
            self._progress_bar.model.set_value(self._progress_fraction)

    def add_cost_point(self, iteration: int, cost: float) -> None:
        """Handle add cost point.

        Args:
            iteration: Iteration value.
            cost: Cost value.
        """
        iteration_value = float(iteration)
        cost_value = float(cost)
        if not math.isfinite(iteration_value) or not math.isfinite(cost_value):
            return
        if self._cost_iterations and iteration_value == self._cost_iterations[-1]:
            self._cost_values[-1] = cost_value
        else:
            self._cost_iterations.append(iteration_value)
            self._cost_values.append(cost_value)
        try:
            self._refresh_cost_plot()
        except Exception as exc:
            carb.log_warn(f"SysId: failed to refresh live cost plot; optimization will continue. {exc}")

    def show_final_joint_errors(self, rollout: SysIdRolloutPlotData) -> None:
        """Plot per-joint position and velocity errors for the converged model.

        Args:
            rollout: Rollout data to display.
        """
        if self._pos_error_plot is None or self._vel_error_plot is None or self._effort_plot is None:
            return

        if self._joint_plots_frame is not None:
            self._joint_plots_frame.visible = True
            self._joint_plots_frame.rebuild()

        times = rollout.times
        pos_errors, vel_errors, legends = _joint_error_series(rollout)
        pos_times, pos_errors, pos_legends = _prepare_joint_error_plot_series(times, pos_errors, legends)
        vel_times, vel_errors, vel_legends = _prepare_joint_error_plot_series(times, vel_errors, legends)
        effort_series, effort_legends = _joint_effort_series(rollout)
        effort_times, effort_series, effort_legends = _prepare_joint_error_plot_series(
            times, effort_series, effort_legends
        )
        effort_colors = _joint_effort_colors(effort_legends)
        num_pos_joints = len(pos_errors)
        num_vel_joints = len(vel_errors)
        num_effort_series = len(effort_series)
        if num_pos_joints < 1 and num_vel_joints < 1 and num_effort_series < 1:
            return

        pos_colors = [_ERROR_COLORS[i % len(_ERROR_COLORS)] for i in range(num_pos_joints)]
        vel_colors = [_ERROR_COLORS[i % len(_ERROR_COLORS)] for i in range(num_vel_joints)]

        if pos_errors:
            self._pos_error_plot.set_data(pos_times, pos_errors, pos_legends, pos_colors)
        else:
            self._pos_error_plot.clear()

        if vel_errors:
            self._vel_error_plot.set_data(vel_times, vel_errors, vel_legends, vel_colors)
        else:
            self._vel_error_plot.clear()

        if effort_series:
            self._effort_plot.set_data(effort_times, effort_series, effort_legends, effort_colors)
        else:
            self._effort_plot.clear()

        pos_count = len(pos_errors)
        vel_count = len(vel_errors)
        effort_count = len(effort_series)
        sample_count = max(
            (len(pos_times) if pos_errors else 0),
            (len(vel_times) if vel_errors else 0),
            (len(effort_times) if effort_series else 0),
        )
        carb.log_info(
            f"SysId: final joint error plots received {pos_count} position series, "
            f"{vel_count} velocity series, {effort_count} effort series, {sample_count} samples."
        )

        if self._joint_plots_frame is not None:
            self._joint_plots_frame.visible = True
            self._joint_plots_frame.rebuild()

    def _hide_joint_error_plots(self) -> None:
        if self._joint_plots_frame is not None:
            self._joint_plots_frame.visible = False
        if self._pos_error_plot is not None:
            self._pos_error_plot.clear()
        if self._vel_error_plot is not None:
            self._vel_error_plot.clear()
        if self._effort_plot is not None:
            self._effort_plot.clear()

    def _refresh_cost_plot(self) -> None:
        if self._cost_plot is None:
            return
        self._ensure_cost_plot_visible()
        self._cost_plot.set_data(self._cost_iterations, self._cost_values)

    def set_controls_enabled(self, enabled: bool) -> None:
        """Keep result status widgets and live plots visible while surrounding inputs are disabled.

        Args:
            enabled: Whether the option is enabled.
        """
        self.ensure_live_progress_visible()

    def refresh_live_progress_layout(self) -> None:
        """Replay live chart state after Omni UI has recomputed expanded-frame layout."""
        self.ensure_live_progress_visible()
        try:
            self._refresh_cost_plot()
        except Exception as exc:
            carb.log_warn(f"SysId: failed to refresh live progress layout; optimization will continue. {exc}")

    def ensure_live_progress_visible(self) -> None:
        """Force the live run widgets to stay visible when the Results frame is opened for a run."""
        if self._activity_label is not None:
            self._activity_label.enabled = True
            self._activity_label.visible = True
        if self._progress_bar is not None:
            self._progress_bar.enabled = True
            self._progress_bar.visible = True
        self._ensure_cost_plot_visible()
        if self._cost_plot is not None:
            self._cost_plot.refresh()

    def _ensure_cost_plot_visible(self) -> None:
        if self._cost_plot is None:
            return
        self._cost_plot.ensure_visible()

    def cleanup(self) -> None:
        """Release resources."""
        for element in self.wrapped_ui_elements:
            if hasattr(element, "cleanup"):
                element.cleanup()
        self.wrapped_ui_elements.clear()
        self._built = False
        self._activity_label = None
        self._progress_bar = None
        self._cost_plot = None
        self._joint_plots_frame = None
        self._pos_error_plot = None
        self._vel_error_plot = None
        self._effort_plot = None


def _cost_plot_series(iterations: list[float], costs: list[float]) -> tuple[list[float], list[float]]:
    """XYPlot needs at least two distinct x samples to interpolate and draw a line.

    Args:
        iterations: Iterations value.
        costs: Costs value.

    Returns:
        The resulting value.
    """
    if len(iterations) < 1:
        return [], []
    if len(iterations) == 1:
        x0 = float(iterations[0])
        y0 = float(costs[0])
        return [max(0.0, x0 - 0.5), x0 + 0.5], [y0, y0]
    return [float(x) for x in iterations], [float(y) for y in costs]


def _cost_plot_bounds(x_data: list[float], y_data: list[float]) -> tuple[float, float, float, float]:
    """Return padded axis bounds for live cost data.

    Args:
        x_data: X data value.
        y_data: Y data value.

    Returns:
        The resulting value.
    """
    x_min, x_max = _padded_bounds(x_data, 0.0, 1.0)
    y_min, y_max = _padded_bounds(y_data, 0.0, 1.0, clamp_min_to_zero=True)
    return x_min, x_max, y_min, y_max


def _padded_bounds(
    values: list[float],
    default_min: float,
    default_max: float,
    clamp_min_to_zero: bool = False,
) -> tuple[float, float]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return default_min, default_max

    lower = min(finite_values)
    upper = max(finite_values)
    span = upper - lower
    if span <= 0.0:
        padding = max(abs(lower) * 0.05, 1e-6)
    else:
        padding = span * 0.05

    lower -= padding
    upper += padding
    if clamp_min_to_zero:
        lower = max(0.0, lower)
    if upper <= lower:
        upper = lower + max(abs(lower) * 0.05, 1e-6)
    return lower, upper


def _cursor_series_suffix(legend: str) -> str:
    lowered = legend.lower()
    if lowered.endswith(" recorded"):
        return "rec"
    if lowered.endswith(" sim"):
        return "sim"
    return ""


def _cursor_series_name(legend: str) -> str:
    suffix = _cursor_series_suffix(legend)
    if suffix == "rec":
        return legend[: -len(" recorded")]
    if suffix == "sim":
        return legend[: -len(" sim")]
    return legend


def _group_cursor_series(
    legends: list[str],
    colors: list[list[int]],
) -> list[tuple[str, list[int], list[int]]]:
    """Group simulated/recorded effort traces into one cursor row per joint.

    Args:
        legends: Series legend labels.
        colors: Per-series RGB colors.

    Returns:
        The resulting value.
    """
    grouped: dict[str, tuple[list[int], list[int]]] = {}
    for series_index, legend in enumerate(legends):
        name = _cursor_series_name(legend)
        color = colors[series_index] if series_index < len(colors) else _ERROR_COLORS[series_index % len(_ERROR_COLORS)]
        if name not in grouped:
            grouped[name] = ([], list(color))
        grouped[name][0].append(series_index)
    return [(name, indices, color) for name, (indices, color) in grouped.items()]


def _prepare_joint_error_plot_series(
    times: list[float],
    errors: list[list[float]],
    legends: list[str],
) -> tuple[list[float], list[list[float]], list[str]]:
    """Trim and filter one family of error series so plot x/y shapes match.

    Args:
        times: Times value.
        errors: Errors value.
        legends: Legends value.

    Returns:
        The resulting value.
    """
    if not errors:
        return [], [], []
    valid_lengths = [len(series) for series in errors if len(series) >= 2]
    if not valid_lengths:
        return [], [], []
    num_steps = min([len(times)] + valid_lengths)
    if num_steps < 2:
        return [], [], []
    trimmed_times = times[:num_steps]
    trimmed_errors: list[list[float]] = []
    trimmed_legends: list[str] = []
    for idx, series in enumerate(errors):
        if len(series) < num_steps:
            continue
        trimmed = series[:num_steps]
        if len(trimmed) < 2 or not all(math.isfinite(float(value)) for value in trimmed):
            continue
        trimmed_errors.append(trimmed)
        trimmed_legends.append(legends[idx] if idx < len(legends) else f"Joint {idx + 1}")
    if not trimmed_errors:
        return [], [], []
    return trimmed_times, trimmed_errors, trimmed_legends


def _joint_error_series(
    rollout: SysIdRolloutPlotData,
) -> tuple[list[list[float]], list[list[float]], list[str]]:
    """Return (position errors, velocity errors, legend labels) per joint.

    Args:
        rollout: Rollout data to display.

    Returns:
        The resulting value.
    """
    pos_errors: list[list[float]] = []
    vel_errors: list[list[float]] = []
    legends: list[str] = []
    num_joints = min(len(rollout.measured_positions), len(rollout.simulated_positions))
    for joint_index in range(num_joints):
        meas_pos = rollout.measured_positions[joint_index]
        sim_pos = rollout.simulated_positions[joint_index]
        pos_errors.append([s - m for s, m in zip(sim_pos, meas_pos)])
        if joint_index < len(rollout.measured_velocities) and joint_index < len(rollout.simulated_velocities):
            meas_vel = rollout.measured_velocities[joint_index]
            sim_vel = rollout.simulated_velocities[joint_index]
            vel_errors.append([s - m for s, m in zip(sim_vel, meas_vel)])
        else:
            vel_errors.append([])
        label = f"Joint {joint_index + 1}" if len(rollout.measured_positions) > 1 else "Joint"
        legends.append(label)
    return pos_errors, vel_errors, legends


def _joint_effort_series(rollout: SysIdRolloutPlotData) -> tuple[list[list[float]], list[str]]:
    """Return simulated efforts and optional recorded efforts per joint.

    Args:
        rollout: Rollout data to display.

    Returns:
        The resulting value.
    """
    sim_efforts = rollout.simulated_efforts or []
    meas_efforts = rollout.measured_efforts or []
    num_joints = max(len(sim_efforts), len(meas_efforts))
    series: list[list[float]] = []
    legends: list[str] = []
    for joint_index in range(num_joints):
        label = f"Joint {joint_index + 1}" if num_joints > 1 else "Joint"
        if joint_index < len(sim_efforts):
            series.append(sim_efforts[joint_index])
            legends.append(f"{label} sim")
        if joint_index < len(meas_efforts):
            series.append(meas_efforts[joint_index])
            legends.append(f"{label} recorded")
    return series, legends


def _joint_effort_colors(legends: list[str]) -> list[list[int]]:
    colors: list[list[int]] = []
    sim_index = 0
    for legend in legends:
        if "recorded" in legend:
            colors.append([235, 235, 235])
        else:
            colors.append(_ERROR_COLORS[sim_index % len(_ERROR_COLORS)])
            sim_index += 1
    return colors
