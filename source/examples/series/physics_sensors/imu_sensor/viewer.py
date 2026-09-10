# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional live viewer for the IMU example.

Kept out of ``main.py`` so the simulation and its checks read on their own. `run()` takes the
simulation as callables rather than importing it, so this module knows nothing about the physics
engine or the scene.
"""

from __future__ import annotations

import ctypes
import glob
import math
import os
from collections.abc import Callable, Sequence

import warp as wp
from imu import BodyState, ImuReading, ImuSensor

# The viewer needs ovui (the standalone distribution of Omniverse's omni.ui, imported as
# `omni.ui`) and glfw. example.toml cannot declare them: its requirements accept only Isaac Sim
# distributions plus the system capabilities listed in source/examples/README.md. Install into the
# build's Python to use --ui:
#
#   _build/.../python -m pip install ovui==0.1.1 glfw==2.10.0
#
# source/tools/zmq_bridge/zmq_server.py also runs ovui outside Kit. It additionally hides the
# carb and omni.kit.app specs during import, which is unnecessary here because carb is not
# importable from the standalone build, so ovui cannot take its in-Kit path.
INSTALL_HINT = (
    "The --ui viewer needs the 'ovui' and 'glfw' packages, which are optional extras this example "
    "does not declare. Install them into the build's Python with:\n"
    "  python -m pip install ovui==0.1.1 glfw==2.10.0"
)

# The viewer drops from higher than the batch run so the free-fall stretch, and the change in the
# readings across it, last long enough to watch. The batch path keeps DROP_HEIGHT, whose value the
# airborne step window and the contact step are tuned around.
DROP_HEIGHT = 6.0
WINDOW_WIDTH = 1120
WINDOW_HEIGHT = 620
VIEW_WIDTH = 560
VIEW_HEIGHT = 520
PIXELS_PER_METER = 55.0
TRACE_LENGTH = 240
# Resting reads one gravity and free fall reads zero, so a ceiling above 1 g keeps both readable
# while leaving headroom for the contact spike, which saturates rather than rescaling the plot.
ACCELERATION_SCALE = 25.0
ANGULAR_VELOCITY_SCALE = 4.0
GROUND_COLOR = (90, 90, 110)
BODY_COLOR = (120, 190, 255)
SENSOR_AXIS_COLORS = ((255, 90, 90), (120, 235, 120), (110, 150, 255))


def _import_standalone_ui() -> object:
    """Import ovui's ``omni.ui``, preloading libglfw so its native backend resolves.

    Returns:
        The imported ``omni.ui`` module.

    Raises:
        RuntimeError: If ovui or glfw is missing, or libglfw cannot be loaded.

    """
    # glfw and omni.ui are optional extras, so they are imported here rather than at module scope.
    try:
        import glfw
    except ImportError as error:
        raise RuntimeError(INSTALL_HINT) from error

    # ovui's native backend links libglfw's symbols, so libglfw must be in the process with global
    # symbol visibility before omni.ui is imported, or the import fails on an undefined symbol.
    candidates = [name for name in [getattr(getattr(glfw, "_glfw", None), "_name", None)] if name]
    candidates += glob.glob(os.path.join(os.path.dirname(glfw.__file__), "**", "libglfw.so*"), recursive=True)
    for library in candidates:
        try:
            ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)
            break
        except OSError:
            continue
    else:
        raise RuntimeError(f"Could not load libglfw (tried: {candidates or 'none'}).\n{INSTALL_HINT}")

    try:
        import omni.ui as user_interface
    except ImportError as error:
        raise RuntimeError(INSTALL_HINT) from error
    return user_interface


def _project_to_view(point: Sequence[float]) -> tuple[float, float]:
    """Project a world point onto the viewer's isometric camera.

    Args:
        point: World-frame point (shape ``(3,)``).

    Returns:
        Pixel coordinates within the view image.

    """
    cosine, sine = math.cos(math.pi / 6.0), math.sin(math.pi / 6.0)
    x, y, z = (float(value) for value in point)
    horizontal = (x - y) * cosine
    vertical = z - (x + y) * sine
    return (
        VIEW_WIDTH * 0.5 + horizontal * PIXELS_PER_METER,
        VIEW_HEIGHT * 0.78 - vertical * PIXELS_PER_METER,
    )


def _draw_line(
    pixels: bytearray, start: tuple[float, float], end: tuple[float, float], color: tuple[int, int, int]
) -> None:
    """Draw a line into a row-major RGBA pixel buffer.

    Args:
        pixels: Destination buffer of ``VIEW_HEIGHT * VIEW_WIDTH * 4`` bytes.
        start: Start pixel coordinate.
        end: End pixel coordinate.
        color: Line color as ``rgb``.

    """
    span = max(abs(end[0] - start[0]), abs(end[1] - start[1]))
    steps = int(span) + 1
    red, green, blue = color
    for step in range(steps):
        fraction = step / (steps - 1) if steps > 1 else 0.0
        column = min(max(int(start[0] + (end[0] - start[0]) * fraction), 0), VIEW_WIDTH - 1)
        row = min(max(int(start[1] + (end[1] - start[1]) * fraction), 0), VIEW_HEIGHT - 1)
        offset = 4 * (row * VIEW_WIDTH + column)
        pixels[offset] = red
        pixels[offset + 1] = green
        pixels[offset + 2] = blue
        pixels[offset + 3] = 255


_BACKGROUND: bytes | None = None


def _ground_plane() -> bytes:
    """Build the static part of the view once and cache it.

    The ground grid and the opaque alpha channel never change, and redrawing them per frame costs
    far more than copying the finished buffer.

    Returns:
        An opaque RGBA image holding only the ground grid.

    """
    global _BACKGROUND
    if _BACKGROUND is None:
        pixels = bytearray(VIEW_HEIGHT * VIEW_WIDTH * 4)
        # Opaque alpha everywhere; every fourth byte is the alpha channel.
        pixels[3::4] = b"\xff" * (VIEW_HEIGHT * VIEW_WIDTH)
        extent, spacing = 2.5, 0.5
        line = -extent
        while line <= extent + 1e-9:
            _draw_line(
                pixels, _project_to_view((line, -extent, 0.0)), _project_to_view((line, extent, 0.0)), GROUND_COLOR
            )
            _draw_line(
                pixels, _project_to_view((-extent, line, 0.0)), _project_to_view((extent, line, 0.0)), GROUND_COLOR
            )
            line += spacing
        _BACKGROUND = bytes(pixels)
    return _BACKGROUND


def _render_scene(state: BodyState, reading: ImuReading | None, body_size: float) -> bytearray:
    """Rasterize a wireframe view of the body and the sensor frame.

    Args:
        state: Current world-frame body state.
        reading: Latest sensor reading, used to draw the sensor frame axes.
        body_size: Side length of the body.

    Returns:
        A row-major RGBA image of ``VIEW_HEIGHT * VIEW_WIDTH * 4`` bytes.

    """
    pixels = bytearray(_ground_plane())

    # The body, as a wireframe cube rotated by its current orientation.
    half = 0.5 * body_size
    corners = [
        state.position + wp.quat_rotate(state.orientation, wp.vec3d(x * half, y * half, z * half))
        for x in (-1, 1)
        for y in (-1, 1)
        for z in (-1, 1)
    ]
    for first in range(8):
        for second in range(first + 1, 8):
            # Cube edges connect corners differing in exactly one axis.
            if bin(first ^ second).count("1") == 1:
                _draw_line(pixels, _project_to_view(corners[first]), _project_to_view(corners[second]), BODY_COLOR)

    # The sensor frame: three axes drawn from the sensor origin, so the mount offset and rotation
    # are visible rather than implied.
    if reading is not None:
        origin = _project_to_view(reading.position)
        for axis, color in enumerate(SENSOR_AXIS_COLORS):
            direction = [0.0, 0.0, 0.0]
            direction[axis] = 0.35
            tip = reading.position + wp.quat_rotate(reading.orientation, wp.vec3d(direction))
            _draw_line(pixels, origin, _project_to_view(tip), color)
    return pixels


def run(
    *,
    sensor: ImuSensor,
    advance: Callable[[int], BodyState],
    restart: Callable[[], None],
    format_vector: Callable[[Sequence[float]], str],
    sensor_path: str,
    body_size: float,
    step_count: int,
) -> ImuReading | None:
    """Simulate with a live window showing the body and the sensor readings together.

    Physics is stepped from the frame callback, so both advance in one process and one thread.

    Args:
        sensor: Sensor whose readings are displayed.
        advance: Steps once for the given step index, samples, and returns the new body state.
        restart: Returns the body to its drop pose and discards buffered samples.
        format_vector: Formats a short sequence of numbers for display.
        sensor_path: Prim path shown in the window.
        body_size: Side length of the body.
        step_count: Steps in one drop, after which the body is dropped again.

    Returns:
        The last reading, or ``None`` if the window closed before one was available.

    Raises:
        RuntimeError: If the viewer packages are missing or no window could be opened.

    """
    import contextlib
    import tempfile

    ui = _import_standalone_ui()

    # ovui writes an imgui.ini of window layout state into the working directory, and that name is
    # fixed inside the library with no setting to disable it. Hold the whole viewer lifetime in a
    # scratch directory so the file is not left in the example source, which is expected to contain
    # no generated output.
    with tempfile.TemporaryDirectory() as scratch, contextlib.chdir(scratch):
        ui.standalone.init(title="Isaac Sim IMU sensor", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        # init() reports a failure to open a window through its log rather than an exception, leaving a
        # zero-sized surface that would render nothing at all. Fail here instead.
        if tuple(ui.standalone.get_window_size()) == (0, 0):
            ui.standalone.shutdown()
            raise RuntimeError(
                "ovui could not open a window. A display is required: this needs DISPLAY or a Wayland "
                "session, and neither was usable. Run without --ui for the headless report."
            )

        provider = ui.ByteImageProvider()
        labels: dict[str, object] = {}
        traces: dict[str, list[float]] = {"acceleration": [], "angular_velocity": []}
        plots: dict[str, object] = {}
        state = {"step": 0, "reading": None, "body": None, "run": 1}

        # Borderless and pinned to fill the OS window, so the content is not wrapped in a visible
        # inner sub-window with its own title bar. _sync_window_size keeps it filling on resize.
        window_width, window_height = ui.standalone.get_window_size()
        window = ui.Window(
            "Isaac Sim IMU sensor",
            width=window_width,
            height=window_height,
            flags=(
                ui.WINDOW_FLAGS_NO_TITLE_BAR
                | ui.WINDOW_FLAGS_NO_RESIZE
                | ui.WINDOW_FLAGS_NO_MOVE
                | ui.WINDOW_FLAGS_NO_COLLAPSE
                | ui.WINDOW_FLAGS_NO_SCROLLBAR
                | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS
            ),
        )
        window.position_x = 0
        window.position_y = 0

        def _sync_window_size() -> None:
            """Keep the borderless window filling the OS window as it is resized."""
            width, height = ui.standalone.get_window_size()
            if width != window.width or height != window.height:
                window.width = width
                window.height = height
                window.position_x = 0
                window.position_y = 0

        with window.frame:
            with ui.HStack(spacing=8):
                ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                with ui.VStack(spacing=6, width=460):
                    ui.Label(f"IMU at {sensor_path}", height=22)
                    ui.Label("Reporting specific force, the proper acceleration.", height=22)
                    for name in (
                        "time",
                        "world position",
                        "world orientation",
                        "sensor linear velocity",
                        "sensor angular velocity",
                        "sensor linear acceleration",
                    ):
                        labels[name] = ui.Label(f"{name}: --", height=20)
                    # ovui's standalone default stylesheet defines no Plot entry, so a plot drawn
                    # without an explicit style has no line color and appears empty. Style each one.
                    plot_style = {
                        "color": ui.color(120, 200, 255),
                        "background_color": ui.color(24, 26, 32),
                        "border_color": ui.color(70, 74, 86),
                        "border_width": 1,
                    }
                    ui.Label("sensor |acceleration|  m/s^2", height=20)
                    plots["acceleration"] = ui.Plot(
                        ui.Type.LINE, 0.0, ACCELERATION_SCALE, 0.0, height=90, style=plot_style
                    )
                    ui.Label("sensor |angular velocity|  rad/s", height=20)
                    plots["angular_velocity"] = ui.Plot(
                        ui.Type.LINE, 0.0, ANGULAR_VELOCITY_SCALE, 0.0, height=90, style=plot_style
                    )
                    labels["status"] = ui.Label("", height=22)

        def _restart() -> None:
            """Drop the body again and count the run.

            The cached reading is dropped with it. The frame update falls back to the previous
            reading while the sensor refills its window, which would otherwise show the last
            reading of the previous drop against the new one for the first few frames.
            """
            restart()
            state["step"] = 0
            state["reading"] = None
            state["run"] += 1

        # Start the first drop from the viewer's height rather than the height the stage was authored
        # at, so every drop the viewer shows is the same.
        _restart()
        state["run"] = 1

        def _on_frame() -> None:
            _sync_window_size()
            step = state["step"]
            if step > step_count:
                # The body has landed and settled: drop it again so there is always something moving.
                _restart()
                step = state["step"]
            body_state = advance(step)
            state["body"] = body_state
            state["reading"] = sensor.read() or state["reading"]
            state["step"] = step + 1
            labels["status"].text = (
                f"drop {state['run']} | step {state['step']}/{step_count} | close the window to exit"
            )

            body_state, reading = state["body"], state["reading"]
            if body_state is None:
                return
            provider.set_data_array(_render_scene(body_state, reading, body_size), [VIEW_WIDTH, VIEW_HEIGHT])
            if reading is None:
                return
            labels["time"].text = f"time: {reading.time:.3f} s"
            labels["world position"].text = f"world position: {format_vector(reading.position)} m"
            labels["world orientation"].text = f"world orientation (xyzw): {format_vector(reading.orientation)}"
            labels["sensor linear velocity"].text = (
                f"sensor linear velocity: {format_vector(reading.linear_velocity)} m/s"
            )
            labels["sensor angular velocity"].text = (
                f"sensor angular velocity: {format_vector(reading.angular_velocity)} rad/s"
            )
            labels["sensor linear acceleration"].text = (
                f"sensor linear acceleration: {format_vector(reading.linear_acceleration)} m/s^2"
            )
            for name, values in (
                ("acceleration", reading.linear_acceleration),
                ("angular_velocity", reading.angular_velocity),
            ):
                trace = traces[name]
                trace.append(float(wp.length(wp.vec3d([float(value) for value in values]))))
                del trace[:-TRACE_LENGTH]
                plots[name].set_data(*trace)

        async def _main_loop() -> None:
            while True:
                _on_frame()
                await ui.next_frame()

        try:
            ui.standalone.run(_main_loop())
        finally:
            ui.standalone.shutdown()

        # The viewer exits whenever the window is closed, which may be mid-fall, so the resting checks
        # the batch run makes do not apply. Hand the last reading back and let the caller report it.
        return state["reading"]
