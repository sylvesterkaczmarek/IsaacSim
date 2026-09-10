# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional live viewer for the joint state example.

Kept out of ``main.py`` so the simulation and its checks read on their own. `run()` takes the
simulation as callables rather than importing it, so this module knows nothing about the physics
engine or the scene.

The robot is drawn from the reading alone: two joint values place every link, which is the point a
joint state sensor makes. Nothing here reads a body pose.
"""

from __future__ import annotations

import ctypes
import glob
import math
import os
from collections.abc import Callable, Sequence

from joint_state import JointStateReading, JointStateSensor

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

# The robot's geometry, in metres, mirroring main.py's `SCENE` centimetres. The viewer draws links, and
# the joint state places them, so it needs the shapes the joint values move.
MAST_HEIGHT = 1.20
CARRIAGE_HEIGHT = 1.00
CARRIAGE_SIZE = 0.30
TURRET_OFFSET = 0.10
TURRET_LENGTH = 0.60

WINDOW_WIDTH = 1180
WINDOW_HEIGHT = 620
VIEW_WIDTH = 560
VIEW_HEIGHT = 520
PIXELS_PER_METER = 260.0
TRACE_LENGTH = 240
# The lift is driven at 0.5 m/s and the spin holds 3 rad/s, so these keep both traces readable
# while leaving headroom for the moment the carriage reaches its limit and stops.
LIFT_VELOCITY_SCALE = 1.0
SPIN_VELOCITY_SCALE = 5.0
GROUND_COLOR = (90, 90, 110)
MAST_COLOR = (150, 150, 170)
CARRIAGE_COLOR = (120, 190, 255)
TURRET_COLOR = (255, 170, 90)


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
        VIEW_HEIGHT * 0.88 - vertical * PIXELS_PER_METER,
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
        extent, spacing = 0.6, 0.15
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


def _draw_box(pixels: bytearray, center: Sequence[float], size: float, color: tuple[int, int, int]) -> None:
    """Draw a wireframe box.

    Args:
        pixels: Destination RGBA buffer.
        center: World-frame center of the box.
        size: Side length in the horizontal axes; the box is a third as tall.
        color: Wire color as ``rgb``.

    """
    half, half_height = 0.5 * size, size / 6.0
    corners = [
        (center[0] + x * half, center[1] + y * half, center[2] + z * half_height)
        for x in (-1, 1)
        for y in (-1, 1)
        for z in (-1, 1)
    ]
    for first in range(8):
        for second in range(first + 1, 8):
            # Box edges connect corners differing in exactly one axis.
            if bin(first ^ second).count("1") == 1:
                _draw_line(pixels, _project_to_view(corners[first]), _project_to_view(corners[second]), color)


def _render_scene(lift: float, spin: float) -> bytearray:
    """Rasterize the robot from its two joint values.

    Args:
        lift: Lift DOF position, in metres. Zero is the top of its travel.
        spin: Spin DOF position, in radians.

    Returns:
        A row-major RGBA image of ``VIEW_HEIGHT * VIEW_WIDTH * 4`` bytes.

    """
    pixels = bytearray(_ground_plane())

    _draw_line(pixels, _project_to_view((0.0, 0.0, 0.0)), _project_to_view((0.0, 0.0, MAST_HEIGHT)), MAST_COLOR)

    carriage_height = CARRIAGE_HEIGHT + lift
    _draw_box(pixels, (0.0, 0.0, carriage_height), CARRIAGE_SIZE, CARRIAGE_COLOR)

    # The turret bar spins about the vertical axis through the carriage, so the spin DOF alone
    # places both of its ends.
    turret_height = carriage_height + TURRET_OFFSET
    half = 0.5 * TURRET_LENGTH
    offset = (half * math.cos(spin), half * math.sin(spin), 0.0)
    first = (offset[0], offset[1], turret_height)
    second = (-offset[0], -offset[1], turret_height)
    _draw_line(pixels, _project_to_view(first), _project_to_view(second), TURRET_COLOR)
    _draw_line(
        pixels, _project_to_view((0.0, 0.0, carriage_height)), _project_to_view((0.0, 0.0, turret_height)), MAST_COLOR
    )
    return pixels


def run(
    *,
    sensor: JointStateSensor,
    advance: Callable[[int], None],
    restart: Callable[[], None],
    robot_path: str,
    lift_dof: str,
    spin_dof: str,
    lift_travel: float,
    step_count: int,
) -> JointStateReading | None:
    """Simulate with a live window showing the robot and its joint state together.

    Physics is stepped from the frame callback, so both advance in one process and one thread.

    Args:
        sensor: Sensor whose readings are displayed.
        advance: Steps once for the given step index and samples the sensor.
        restart: Returns the robot to its start state and discards buffered samples.
        robot_path: Prim path shown in the window.
        lift_dof: Name of the translation DOF, used to place the carriage.
        spin_dof: Name of the rotation DOF, used to place the turret.
        lift_travel: Metres from the lift DOF's zero to its lower limit, for the position bar.
        step_count: Steps in one drop, after which the robot starts again.

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
        ui.standalone.init(title="Isaac Sim joint state sensor", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
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
        traces: dict[str, list[float]] = {name: [] for name in sensor.names}
        plots: dict[str, object] = {}
        state = {"step": 0, "reading": None, "run": 1}

        # Borderless and pinned to fill the OS window, so the content is not wrapped in a visible
        # inner sub-window with its own title bar. _sync_window_size keeps it filling on resize.
        window_width, window_height = ui.standalone.get_window_size()
        window = ui.Window(
            "Isaac Sim joint state sensor",
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
                with ui.VStack(spacing=6, width=520):
                    ui.Label(f"Joint state of {robot_path}", height=22)
                    ui.Label("Reporting measured joint effort, not commanded actuation.", height=22)
                    labels["time"] = ui.Label("time: --", height=20)
                    for name in sensor.names:
                        labels[name] = ui.Label(f"{name}: --", height=20)
                    # ovui's standalone default stylesheet defines no Plot entry, so a plot drawn
                    # without an explicit style has no line color and appears empty. Style each one.
                    plot_style = {
                        "color": ui.color(120, 200, 255),
                        "background_color": ui.color(24, 26, 32),
                        "border_color": ui.color(70, 74, 86),
                        "border_width": 1,
                    }
                    ui.Label(f"|{lift_dof} velocity|  m/s", height=20)
                    plots[lift_dof] = ui.Plot(ui.Type.LINE, 0.0, LIFT_VELOCITY_SCALE, 0.0, height=90, style=plot_style)
                    ui.Label(f"|{spin_dof} velocity|  rad/s", height=20)
                    plots[spin_dof] = ui.Plot(ui.Type.LINE, 0.0, SPIN_VELOCITY_SCALE, 0.0, height=90, style=plot_style)
                    labels["status"] = ui.Label("", height=22)

        def _restart() -> None:
            """Start the robot again and count the run.

            The cached reading is dropped with it. The frame update falls back to the previous
            reading while the sensor refills its window, which would otherwise show the last
            reading of the previous run against the new one for the first few frames.
            """
            restart()
            state["step"] = 0
            state["reading"] = None
            state["run"] += 1

        _restart()
        state["run"] = 1

        units = {"rotation": ("rad", "rad/s", "N*m"), "translation": ("m", "m/s", "N")}

        def _on_frame() -> None:
            _sync_window_size()
            step = state["step"]
            if step > step_count:
                # The carriage has landed and settled: start again so there is always something moving.
                _restart()
                step = state["step"]
            advance(step)
            state["reading"] = sensor.read() or state["reading"]
            state["step"] = step + 1
            labels["status"].text = f"run {state['run']} | step {state['step']}/{step_count} | close the window to exit"

            reading = state["reading"]
            if reading is None:
                return
            lift = reading.positions[reading.names.index(lift_dof)]
            spin = reading.positions[reading.names.index(spin_dof)]
            provider.set_data_array(_render_scene(lift, spin), [VIEW_WIDTH, VIEW_HEIGHT])

            labels["time"].text = f"time: {reading.time:.3f} s"
            for index, name in enumerate(reading.names):
                position, velocity, effort = units[reading.types[index]]
                fraction = abs(lift) / lift_travel if name == lift_dof else 0.0
                bar = ("=" * round(20 * min(fraction, 1.0))).ljust(20, ".") if name == lift_dof else ""
                labels[name].text = (
                    f"{name} ({reading.types[index]}): "
                    f"{reading.positions[index]: .4f} {position}, "
                    f"{reading.velocities[index]: .4f} {velocity}, "
                    f"{reading.efforts[index]: .3f} {effort}  {bar}"
                )
                trace = traces[name]
                trace.append(abs(reading.velocities[index]))
                del trace[:-TRACE_LENGTH]
                # Only the two DOFs this robot has are plotted; anything else is labelled only.
                if name in plots:
                    plots[name].set_data(*trace)

        async def _main_loop() -> None:
            while True:
                _on_frame()
                await ui.next_frame()

        try:
            ui.standalone.run(_main_loop())
        finally:
            ui.standalone.shutdown()

        # The viewer exits whenever the window is closed, which may be mid-descent, so the resting
        # checks the batch run makes do not apply. Hand the last reading back and let the caller
        # report it.
        return state["reading"]
