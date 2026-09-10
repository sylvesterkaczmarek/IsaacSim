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

"""Carter warehouse teleop — standalone ZMQ server for the Isaac Sim ZMQ bridge.

Drives a Nova Carter around the Simple Warehouse over ZMQ and shows its front-camera color
stream in an omni.ui (ovui) window.

  Receive (PULL 5561): image (front camera), clock, pose (robot world pose).
  Send    (PUB  5557): joint_command (the two drive-wheel angular velocities).

Setup: run with the build's Python from the build/install root:

  ./python.sh -m pip install -r tools/zmq_bridge/requirements.txt

Run (first open /Isaac/Samples/ZeroMQ/carter_warehouse_zmq.usd in Isaac Sim and press Play):

  ./python.sh tools/zmq_bridge/zmq_server.py

Add ``--subscribe_only 1`` to receive and display without sending commands.

If the protobuf stubs cannot be located automatically, point ``ISAAC_ZMQ_PROTO_DIR`` at the
directory holding ``image_pb2.py`` (``<isaac_sim_root>/exts/isaacsim.zmq.protos/isaacsim/zmq/protos``).

Controls (click the window to focus it): W/S drive forward/back, A/D turn left/right. The window
shows the ZMQ camera-stream rate and the sim's render rate, plus a status bar with the sim clock,
robot world position, and heading; commands publish only on change.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
import traceback
from types import ModuleType

import numpy as np

# The isaacsim.zmq.protos extension holds ``image_pb2.py``. Two install layouts carry it, relative
# to an ancestor of this script: a source build (``_build/<platform>/<config>/release/exts/...``)
# and a packaged/installed Isaac Sim (``exts/...`` under the install root). We probe both so the
# tool works out of the box from either — ``ISAAC_ZMQ_PROTO_DIR`` is only a fallback override.
_PROTO_GLOBS = (
    os.path.join("*", "exts", "isaacsim.zmq.protos", "isaacsim", "zmq", "protos", "image_pb2.py"),
    os.path.join("exts", "isaacsim.zmq.protos", "isaacsim", "zmq", "protos", "image_pb2.py"),
)


def _add_bundled_packages_to_path() -> None:
    """Add the isaacsim.zmq.protos proto stubs, its bundled pip_prebundle (pyzmq, protobuf), and
    warp to sys.path from the build/install. ``ISAAC_ZMQ_PROTO_DIR`` overrides proto-dir discovery;
    otherwise both the source-build and installed layouts are auto-detected (see _PROTO_GLOBS), so
    setting the env var is not normally required."""
    import glob

    override = os.environ.get("ISAAC_ZMQ_PROTO_DIR")
    if override and os.path.isfile(os.path.join(override, "image_pb2.py")):
        sys.path.insert(0, override)
        return
    root = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        hits = [h for pattern in _PROTO_GLOBS for h in glob.glob(os.path.join(root, pattern))]
        if hits:
            proto_dir = os.path.dirname(hits[0])
            sys.path.insert(0, proto_dir)
            ext_root = os.path.normpath(os.path.join(proto_dir, "..", "..", ".."))  # exts/isaacsim.zmq.protos
            exts_dir = os.path.dirname(ext_root)
            prebundle = os.path.join(ext_root, "pip_prebundle")
            if os.path.isdir(prebundle):
                sys.path.insert(0, prebundle)
            release_dir = os.path.dirname(exts_dir)
            for warp_ext in sorted(glob.glob(os.path.join(release_dir, "extscache", "omni.warp.core-*"))):
                if os.path.isdir(os.path.join(warp_ext, "warp")):
                    sys.path.insert(0, warp_ext)
                    break
            return
        parent = os.path.dirname(root)
        if parent == root:
            break
        root = parent
    raise RuntimeError(
        "Could not find the built isaacsim.zmq.protos extension. Build it (./build.sh) or set "
        "ISAAC_ZMQ_PROTO_DIR to the directory containing image_pb2.py."
    )


def _preload_libglfw() -> None:
    """ovui's native backend links libglfw's symbols, so libglfw.so.3 must be loaded with global
    symbol visibility BEFORE omni.ui is imported. The ``glfw`` wheel (from requirements.txt) ships
    it; reuse the library pyGLFW already resolved, falling back to a search of its package dir."""
    import ctypes

    try:
        import glfw
    except ImportError:
        raise RuntimeError(
            "The 'ovui' and 'glfw' viewer packages are not installed. Install them with:\n"
            "  ./python.sh -m pip install -r "
            "tools/zmq_bridge/requirements.txt"
        )
    candidates = []
    resolved = getattr(getattr(glfw, "_glfw", None), "_name", None)
    if resolved:
        candidates.append(resolved)
    import glob

    candidates += glob.glob(os.path.join(os.path.dirname(glfw.__file__), "**", "libglfw.so*"), recursive=True)
    for lib in candidates:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue
    # Fail loudly here rather than let the omni.ui (ovui) import fail later with an opaque
    # "undefined symbol" error when libglfw was never loaded into the process.
    raise RuntimeError(
        f"Could not load libglfw.so needed by the omni.ui (ovui) viewer (tried: {candidates or 'none'}). "
        "Reinstall the viewer packages:\n"
        "  ./python.sh -m pip install -r tools/zmq_bridge/requirements.txt"
    )


def _import_standalone_omni_ui() -> ModuleType:
    """Import ovui's omni.ui in standalone mode: hide the ``carb`` / ``omni.kit.app`` specs during
    import so ovui uses its GLFW/OpenGL backend instead of the in-Kit path."""
    import importlib.util as _ilu

    real_find_spec = _ilu.find_spec

    def _find_spec_standalone(name, *args, **kwargs):
        if name in ("carb", "omni.kit.app"):
            return None
        return real_find_spec(name, *args, **kwargs)

    _ilu.find_spec = _find_spec_standalone
    try:
        import omni.ui as _ui
    finally:
        _ilu.find_spec = real_find_spec
    return _ui


_add_bundled_packages_to_path()
import clock_pb2  # noqa: E402
import image_pb2  # noqa: E402
import joint_command_pb2  # noqa: E402
import pose_pb2  # noqa: E402

_preload_libglfw()
ui = _import_standalone_omni_ui()
import warp as wp  # noqa: E402
from zmq_server.client import ZMQClient  # noqa: E402  (generic ZMQ PUB/PULL helper)
from zmq_server.gl_display import rgb_to_rgba  # noqa: E402
from zmq_server.ipc_image import IpcImageReader  # noqa: E402
from zmq_server.teleop import TeleopKeys  # noqa: E402

# ---------------------------------------------------------------------------
# Ports / topics (must match the scene's ZMQ graphs).
# ---------------------------------------------------------------------------
PORT_PUBLISH = 5561  # sim -> client: image, clock
PORT_SUBSCRIBE = 5557  # client -> sim: joint_command
TOPIC_IMAGE = "image"
TOPIC_CLOCK = "clock"
TOPIC_POSE = "pose"
TOPIC_JOINT_COMMAND = "joint_command"

# Nova Carter's two drive wheels (the controller resolves them by name). Casters are passive.
WHEEL_JOINT_NAMES = ("joint_wheel_left", "joint_wheel_right")

# Rate at which wheel velocity commands are published.
_SEND_HZ = 10.0

# Teleop speeds, in wheel angular velocity (rad/s). Nova Carter wheel radius ~0.14 m, so
# _DRIVE_SPEED=6 rad/s is ~0.85 m/s. Flip _DRIVE_SIGN / _TURN_SIGN if a key drives the wrong way.
_DRIVE_SPEED = 6.0
_TURN_SPEED = 4.0
_DRIVE_SIGN = 1.0
_TURN_SIGN = 1.0

# ovui's set_key_pressed_fn reports GLFW key codes (per ovui ImGuiKeyTranslation.h), which for
# letter keys equal ord("<UPPERCASE>"). (NOT ImGui's 1.87+ named-key values.)
_KEY_W, _KEY_A, _KEY_S, _KEY_D = ord("W"), ord("A"), ord("S"), ord("D")


class CarterWarehouse:
    """Standalone client: shows Carter's front camera and teleops it with WASD."""

    def __init__(self) -> None:
        parser = argparse.ArgumentParser(description="Carter warehouse teleop")
        parser.add_argument("--subscribe_only", type=int, default=0, help="1 = view only, send no commands")
        parser.add_argument("--debug", action="store_true", help="throttled telemetry to stdout")
        args = parser.parse_args()
        self.subscribe_only = bool(args.subscribe_only)
        self._debug = bool(args.debug)
        self._last_debug_t = 0.0

        self.zmq_server = ZMQClient()
        self.zmq_server.create_context()

        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._running = True
        self._recv_thread = None  # daemon threads, joined in _shutdown before releasing sockets
        self._send_thread = None

        # Latest received color frame (cuda wp.array HxWx4 uint8), consumed by the UI thread.
        self._pending_color = None
        self._ipc_reader = IpcImageReader()

        # Latest telemetry for the status readout.
        self._sim_time = None  # simulation clock (s)
        self._robot_pos = None  # robot world position [x, y, z]
        self._robot_quat = None  # robot world orientation quaternion [x, y, z, w]

        # omni.ui widgets (built in _build_ui).
        self._window = None
        self._color_provider = None
        self._status_label = None
        self._fps_label = None
        self._gpu_upload = False

        # Frame-rate readouts (see _update_rates / _on_clock).
        self._img_frames = 0  # camera frames received since the last camera-fps sample
        self._fps_window_t = None  # wall-clock time of the last camera-fps sample
        self._camera_fps = 0.0  # displayed camera-stream rate (Hz), measured at the client
        self._render_fps = 0.0  # real sim render rate (Hz), from the clock topic's sys_dt (EMA)
        self._prev_sys_time = None  # previous clock sys_time, for the sys_dt fallback

        # Keys currently held, integrated into wheel velocities each frame. Press-recency based,
        # so keyboard auto-repeat cannot read as a release (see zmq_server.teleop).
        self._keys = TeleopKeys()
        self._wheel_vel = [0.0, 0.0]  # [left, right] angular velocity command (rad/s)

    # ------------------------------------------------------------------
    # Networking
    # ------------------------------------------------------------------
    def create_network_iface(self) -> None:
        self.zmq_server.create_pull_socket(PORT_PUBLISH)
        if not self.subscribe_only:
            self.zmq_server.create_pub_socket(PORT_SUBSCRIBE)
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        if not self.subscribe_only:
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()

    def _recv_loop(self) -> None:
        dispatch = {TOPIC_IMAGE: self._on_image, TOPIC_CLOCK: self._on_clock, TOPIC_POSE: self._on_pose}
        while self._running:
            topic, data = self.zmq_server.receive_multipart(PORT_PUBLISH)
            if not data:
                continue
            handler = dispatch.get(topic)
            if handler is not None:
                try:
                    handler(data)
                except Exception:
                    # A malformed payload (protobuf DecodeError) or an unexpected image shape must
                    # not kill this daemon thread and freeze the viewer — log it and keep receiving.
                    traceback.print_exc()

    def _on_clock(self, data: bytes) -> None:
        msg = clock_pb2.Clock()
        if not msg.ParseFromString(data):
            return
        with self._lock:
            self._sim_time = msg.sim_time
            # Real sim render rate straight from the sim: sys_dt is the wall-clock time the sim
            # spent on that frame, so 1/sys_dt is its achieved fps — drop-immune, since each
            # message carries its own frame's delta. Fall back to the gap between successive clock
            # timestamps if a scene doesn't wire deltaSystemTime. Smoothed with an EMA.
            dt = msg.sys_dt
            if dt <= 0.0 and self._prev_sys_time is not None:
                dt = msg.sys_time - self._prev_sys_time
            self._prev_sys_time = msg.sys_time
            if dt > 0.0:
                inst = 1.0 / dt
                self._render_fps = inst if self._render_fps == 0.0 else 0.9 * self._render_fps + 0.1 * inst

    def _on_pose(self, data: bytes) -> None:
        msg = pose_pb2.Pose()
        if not msg.ParseFromString(data):
            return
        with self._lock:
            if len(msg.position) >= 3:
                self._robot_pos = [msg.position[0], msg.position[1], msg.position[2]]
            if len(msg.orientation) >= 4:
                self._robot_quat = [msg.orientation[i] for i in range(4)]

    def _on_image(self, data: bytes) -> None:
        msg = image_pb2.Image()
        if not msg.ParseFromString(data):
            return
        if msg.encoding not in ("rgba8", "rgb8"):  # color only (scene publishes rgb8)
            return
        transport = msg.WhichOneof("pixel_transport")
        if transport == "gpu":
            ipc = msg.gpu.array
            arr = self._ipc_reader.open_wp(
                bytes(ipc.mem_handle), ipc.dtype, tuple(ipc.shape), bytes(msg.gpu.ipc_event_handle)
            )
        elif transport == "data":
            ch = 4 if msg.encoding == "rgba8" else 3
            # bytes(msg.data) copies the protobuf buffer, avoiding the read-only numpy warning
            # and making ascontiguousarray a no-op (warp can then upload directly).
            host = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, ch)
            arr = wp.array(host, device="cuda")
        else:
            return
        if arr is None:
            return
        if tuple(arr.shape)[-1] == 3:  # pad rgb8 -> rgba8 on the GPU for the provider
            arr = rgb_to_rgba(arr)
        with self._lock:
            self._pending_color = arr
            self._img_frames += 1

    # ------------------------------------------------------------------
    # Send loop (wheel velocity teleop)
    # ------------------------------------------------------------------
    def _send(self, topic: str, payload: bytes) -> None:
        with self._send_lock:
            self.zmq_server.send_multipart(PORT_SUBSCRIBE, topic, payload)

    def _send_loop(self) -> None:
        # Send the current wheel velocity unconditionally at _SEND_HZ.
        while self._running:
            with self._lock:
                vel = list(self._wheel_vel)
            msg = joint_command_pb2.JointCommand()
            msg.joint_names.extend(WHEEL_JOINT_NAMES)
            msg.velocities.extend(vel)
            self._send(TOPIC_JOINT_COMMAND, msg.SerializeToString())
            if self._debug:
                self._debug_log(vel[0], vel[1])
            time.sleep(1.0 / _SEND_HZ)

    def _debug_log(self, vl: float, vr: float) -> None:
        now = time.time()
        if now - self._last_debug_t < 1.0:
            return
        self._last_debug_t = now
        # Report the streamed wheel command only. (Don't read self._keys here: it's mutated by
        # _on_key on the UI thread and iterating it from this send thread can raise.)
        print(f"[debug] wheel_vel [L, R] = [{vl:.2f}, {vr:.2f}] rad/s", flush=True)

    def _apply_held_keys(self) -> None:
        """WASD -> differential drive -> [left, right] wheel angular velocities."""
        now = time.monotonic()
        held = self._keys.is_held
        drive = (1.0 if held(_KEY_W, now) else 0.0) - (1.0 if held(_KEY_S, now) else 0.0)
        turn = (1.0 if held(_KEY_D, now) else 0.0) - (1.0 if held(_KEY_A, now) else 0.0)
        lin = _DRIVE_SIGN * drive * _DRIVE_SPEED
        ang = _TURN_SIGN * turn * _TURN_SPEED
        # Turn right (D) -> left wheel faster, right wheel slower.
        with self._lock:
            self._wheel_vel = [lin + ang, lin - ang]

    # ------------------------------------------------------------------
    # GUI (omni.ui / ovui)
    # ------------------------------------------------------------------
    def run(self) -> None:
        if wp.is_cuda_available():
            wp.init()
        ui.init("Carter Warehouse", width=1280, height=760)
        self._build_ui()
        self.create_network_iface()
        try:
            ui.run(self._main_loop())
        finally:
            self._shutdown()
            ui.shutdown()

    def _build_ui(self) -> None:
        self._gpu_upload = ui.has_gpu_byte_image()
        self._color_provider = ui.ByteImageProvider()
        # Borderless window pinned to fill the whole OS window, so the content is not wrapped in a
        # visible inner sub-window (no nested title bar). _sync_window_size keeps it filling on
        # resize (the flags disable the window's own move/resize/title chrome).
        win_w, win_h = ui.standalone.get_window_size()
        self._window = ui.Window(
            "Carter Warehouse",
            width=win_w,
            height=win_h,
            flags=(
                ui.WINDOW_FLAGS_NO_TITLE_BAR
                | ui.WINDOW_FLAGS_NO_RESIZE
                | ui.WINDOW_FLAGS_NO_MOVE
                | ui.WINDOW_FLAGS_NO_COLLAPSE
                | ui.WINDOW_FLAGS_NO_SCROLLBAR
                | ui.WINDOW_FLAGS_NO_SAVED_SETTINGS
            ),
        )
        self._window.position_x = 0
        self._window.position_y = 0
        with self._window.frame:
            with ui.VStack(spacing=6):
                # PRESERVE_ASPECT_FIT: the image scales to fit the widget (letterboxing as needed)
                # instead of the default crop, so the window stays responsive when resized.
                ui.ImageWithProvider(self._color_provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
                self._fps_label = ui.Label("", height=20)
                if self.subscribe_only:
                    self._status_label = ui.Label("Subscribe-only (no commands sent)", height=20)
                else:
                    ui.Label("Drive: W/S forward/back   A/D turn left/right", height=20)
                    self._status_label = ui.Label("", height=20)
        if not self.subscribe_only:
            self._window.set_key_pressed_fn(self._on_key)

    async def _main_loop(self):
        try:
            while self._running:
                self._on_frame()
                await ui.next_frame()
        finally:
            self._shutdown()

    def _on_frame(self) -> None:
        self._sync_window_size()
        with self._lock:
            color, self._pending_color = self._pending_color, None
        if color is not None:
            self._upload(self._color_provider, color)
        if not self.subscribe_only:
            self._apply_held_keys()
        self._update_rates()
        self._update_status()

    def _sync_window_size(self) -> None:
        """Keep the borderless window filling the OS window as it is resized."""
        if self._window is None:
            return
        w, h = ui.standalone.get_window_size()
        if w != self._window.width or h != self._window.height:
            self._window.width = w
            self._window.height = h
            self._window.position_x = 0
            self._window.position_y = 0

    def _update_rates(self) -> None:
        """Refresh the fps readout: the client-measured camera-stream rate (displayed ``image``
        frames per wall-clock second, over a ~0.5 s window) and the sim's real render rate (kept
        by _on_clock from the ``clock`` topic's ``sys_dt``)."""
        if self._fps_label is None:
            return
        now = time.time()
        with self._lock:
            if self._fps_window_t is None:
                self._fps_window_t = now
            elapsed = now - self._fps_window_t
            if elapsed >= 0.5:
                self._camera_fps = self._img_frames / elapsed
                self._img_frames = 0
                self._fps_window_t = now
            camera_fps, render_fps = self._camera_fps, self._render_fps
        self._fps_label.text = f"ZMQ camera stream {camera_fps:5.1f} fps   render {render_fps:5.1f} fps"

    def _update_status(self) -> None:
        if self._status_label is None:
            return
        with self._lock:
            t = self._sim_time
            pos = None if self._robot_pos is None else list(self._robot_pos)
            quat = None if self._robot_quat is None else list(self._robot_quat)
            vl, vr = self._wheel_vel
        t_s = "-" if t is None else f"{t:7.2f} s"
        if pos is None:
            pose_s = "pose -"
        else:
            # heading (yaw about world +Z) from the quaternion (x, y, z, w).
            x, y, z, w = quat if quat else (0.0, 0.0, 0.0, 1.0)
            yaw_deg = math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
            pose_s = f"pos ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) m   heading {yaw_deg:6.1f} deg"
        wheel_s = "" if self.subscribe_only else f"   wheel [L,R] [{vl:.1f}, {vr:.1f}] rad/s"
        self._status_label.text = f"sim time {t_s}   {pose_s}{wheel_s}"

    def _upload(self, provider, arr) -> None:
        if provider is None:
            return
        h, w = int(arr.shape[0]), int(arr.shape[1])
        try:
            if self._gpu_upload:
                provider.set_bytes_data_from_gpu(int(arr.ptr), [w, h])
            else:
                provider.set_data_array(arr.numpy(), [w, h])
        except Exception:
            print("[viewer] image upload failed:\n" + traceback.format_exc())

    def _on_key(self, key, modifier, pressed) -> None:
        if self._debug:
            print(f"[key] t={time.monotonic():.4f} code={int(key)} pressed={pressed}", flush=True)
        # Both edges are forwarded; TeleopKeys decides whether a release is real by waiting to see
        # whether an auto-repeat press follows it.
        now = time.monotonic()
        if pressed:
            self._keys.press(int(key), now)
        else:
            self._keys.release(int(key), now)

    def _shutdown(self) -> None:
        if not self._running:
            return  # idempotent: called from both _main_loop() and run()'s finally blocks
        self._running = False
        # Join the recv/send daemon threads so they leave their loops BEFORE we close the sockets
        # and terminate the context. The recv thread can be parked in a blocking recv (the PULL
        # socket's RCVTIMEO is 1 s), so joining it — rather than a short fixed sleep — avoids both
        # a "Context was terminated" (ETERM) traceback on exit and closing a socket from under a
        # blocked recv. It also ensures the bound ports (5561/5557) are freed promptly on restart.
        for thread in (self._recv_thread, self._send_thread):
            if thread is not None:
                thread.join(timeout=2.0)
        self.zmq_server.cleanup()


if __name__ == "__main__":
    CarterWarehouse().run()
