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

"""Publish camera images with rclpy at a high rate by keeping the simulation loop off the copy path.

Builds a small self-contained scene (ground, a field of colored cubes, and lights) and publishes
the camera's images over ROS 2 from ``rclpy`` without OmniGraph. The device-to-host copy and the
(GIL-holding) ROS 2 publish each stall the simulation thread by several ms/frame, so both are kept
off it: each frame is copied device-to-device into a rotating pool of GPU buffers on the simulation
thread, and a worker thread does the device-to-host copy (into page-locked host memory) and the
publish, overlapping them with the next frame's rendering. See ``AsyncImagePublisher``.

Compare the achieved rate with ``ros2 topic hz /rgb``.
"""

from __future__ import annotations

import argparse
import array
import contextlib
import ctypes

from isaacsim import SimulationApp

parser = argparse.ArgumentParser(description="Publish camera images asynchronously with rclpy.")
parser.add_argument("--headless", action="store_true", help="Run without a window.")
parser.add_argument("--frames", type=int, default=0, help="Number of frames to run (0 = run until interrupted).")
parser.add_argument("--test", default=False, action="store_true", help="Run in test mode.")
parser.add_argument(
    "--profile",
    action="store_true",
    help="Enable the Tracy profiler backend; attach the Tracy GUI or capture tool to port 8086.",
)
args, _ = parser.parse_known_args()

if args.test and args.frames == 0:
    args.frames = 10

RESOLUTION = (480, 640)  # (height, width)

# Stream the carb.profiler zones (the ``rclpy:*`` zones below, plus Isaac's own) to Tracy when
# profiling. This must be set in the SimulationApp config; it is not a runtime setting.
config = {"renderer": "RaytracedLighting", "headless": args.headless}
if args.profile:
    config["profiler_backend"] = ["tracy"]
simulation_app = SimulationApp(config)

import queue
import threading

import carb
import carb.profiler
import numpy as np
from isaacsim.core.api import SimulationContext
from isaacsim.core.experimental.objects import Cube, DistantLight, DomeLight
from isaacsim.core.experimental.utils.transform import euler_angles_to_quaternion, look_at_quaternion
from isaacsim.core.utils import extensions

# Enable the ROS 2 bridge extension (provides rclpy and the ROS 2 middleware)
if args.profile:
    extensions.enable_extension("omni.kit.profiler.tracy")
extensions.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

import rclpy
import warp as wp
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    _libcuda = ctypes.CDLL("libcuda.so.1")  # CUDA driver, for page-locking the host message buffers
except OSError:
    _libcuda = None


@contextlib.contextmanager
def _profile(name: str):
    """Emit a ``carb.profiler`` zone (visible in Tracy / the profiler window).

    A no-op unless a profiler backend is active, so it is free when the example is run normally.
    See the ROS 2 Python tutorial for how to run this example under Tracy.
    """
    carb.profiler.begin(0, name)
    try:
        yield
    finally:
        carb.profiler.end(0)


def _page_lock(buffer: array.array) -> bool:
    """Page-lock (pin) a host buffer via the CUDA driver so device-to-host copies are asynchronous.

    Pinned host memory lets ``warp.copy`` issue a non-blocking device-to-host transfer, so the
    worker thread can overlap the copy with the simulation's rendering. Returns False (leaving the
    buffer pageable, which forces a blocking copy) if registration is unavailable.
    """
    if _libcuda is None:
        return False
    address, num_bytes = buffer.buffer_info()  # itemsize is 1 ('B'), so num_bytes == length
    return _libcuda.cuMemHostRegister(ctypes.c_void_p(address), ctypes.c_size_t(num_bytes), ctypes.c_uint(0)) == 0


def _page_unlock(buffer: array.array) -> None:
    address, _ = buffer.buffer_info()
    _libcuda.cuMemHostUnregister(ctypes.c_void_p(address))


class AsyncImagePublisher:
    """Publish GPU camera images at a high rate by keeping the simulation loop off the copy path.

    Doing the device-to-host copy and the ROS 2 publish on the simulation thread stalls it by
    several ms/frame (the copy waits on the GPU; ``publish`` holds the GIL while it serializes and
    writes to DDS). This publisher moves both off that thread:

    - ``submit`` (simulation thread) issues only a cheap device-to-device copy of the frame into a
      rotating pool of GPU buffers, then hands the slot to the worker and returns.
    - The worker copies its GPU buffer into a page-locked host buffer, waits for that transfer, and
      publishes. Page-locking makes the copy asynchronous so it overlaps with the next render.

    A ``free`` event per slot keeps the simulation from overwriting a GPU buffer the worker is still
    reading; if no slot is free the frame is dropped so the simulation never stalls.

    Args:
        node: ROS 2 node used to create the publisher.
        topic: Topic name to publish images on.
        height: Image height in pixels.
        width: Image width in pixels.
        frame_id: TF frame id written to the message headers.
        queue_size: Publisher queue depth.
        num_slots: Number of GPU-buffer/message slots rotated across frames.
    """

    def __init__(
        self,
        node: Node,
        topic: str,
        height: int,
        width: int,
        frame_id: str = "sim_camera",
        queue_size: int = 10,
        num_slots: int = 8,
    ) -> None:
        self._node = node
        self._frame_id = frame_id
        self._publisher = node.create_publisher(Image, topic, queue_size)
        self.published_count = 0
        self.dropped_count = 0
        num_bytes = height * width * 3
        # Each slot pairs a GPU buffer (device-to-device copy target) with a preallocated, page-
        # locked message and a "free" event; the pool is rotated across frames.
        self._gpu_buffers = [wp.zeros((height, width, 3), dtype=wp.uint8, device="cuda:0") for _ in range(num_slots)]
        self._messages = []
        self._pinned_buffers = []  # host buffers actually page-locked, to unregister on close
        for _ in range(num_slots):
            message = Image()
            message.height, message.width = height, width
            message.encoding, message.step, message.is_bigendian = "rgb8", width * 3, 0
            message.data = array.array("B", bytes(num_bytes))
            if _page_lock(message.data):
                self._pinned_buffers.append(message.data)
            self._messages.append(message)
        if len(self._pinned_buffers) < num_slots:
            carb.log_warn("AsyncImagePublisher: some host buffers not page-locked; those device-to-host copies block")
        self._free = [threading.Event() for _ in range(num_slots)]
        for event in self._free:
            event.set()
        self._queue = queue.Queue()
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rclpy_image_publisher", daemon=True)
        self._thread.start()

    def submit(self, data: wp.array) -> bool:
        """Copy a GPU frame into a free slot (device-to-device) and queue it for publishing.

        Runs on the simulation thread but only issues a cheap device-to-device copy, so it never
        blocks on the host transfer or the publish.

        Args:
            data: Image data as a Warp array on a CUDA device.

        Returns:
            True if the frame was queued, False if it was dropped because no slot was free.
        """
        # The `rgb` annotator returns a contiguous device array, so we copy it directly. (A
        # non-contiguous source would need `.contiguous()`, whose temporary must be kept alive
        # until the async copy below runs — not just until this method returns.)
        for index, free in enumerate(self._free):
            if free.is_set():
                # Device-to-device copy on cuda:0's (process-global) current stream. The
                # queue handoff below orders this before the worker's device-to-host copy of the
                # same buffer (also on that stream), so the worker never reads it early.
                with _profile("rclpy:submit_d2d"):
                    wp.copy(self._gpu_buffers[index], data)
                free.clear()
                self._queue.put(index)
                return True
        self.dropped_count += 1
        return False

    def close(self) -> None:
        """Stop the worker thread and release the page-locked buffers."""
        self._stop_requested.set()
        self._queue.put(None)  # wake the worker
        self._thread.join(timeout=5.0)
        # Only unregister the pinned host buffers if the worker has actually stopped; releasing
        # host memory while a copy or DDS send is still using it is undefined behavior.
        if not self._thread.is_alive():
            for buffer in self._pinned_buffers:
                _page_unlock(buffer)

    def _run(self) -> None:
        while not self._stop_requested.is_set():
            index = self._queue.get()
            if index is None:
                break
            message = self._messages[index]
            gpu_buffer = self._gpu_buffers[index]
            try:
                # Device-to-host copy into the (page-locked) message buffer, then wait for it to
                # complete before publishing. Runs after the sim thread's device-to-device copy
                # (same stream, ordered by the queue handoff); both run off the simulation thread.
                destination = wp.array(np.frombuffer(message.data, dtype=np.uint8), device="cpu", copy=False)
                with _profile("rclpy:worker_d2h"):
                    wp.copy(destination, gpu_buffer.reshape((gpu_buffer.size,)))
                with _profile("rclpy:worker_sync"):
                    wp.synchronize()
                message.header.stamp = self._node.get_clock().now().to_msg()
                message.header.frame_id = self._frame_id
                with _profile("rclpy:worker_publish"):
                    self._publisher.publish(message)
                self.published_count += 1
            except Exception as e:
                carb.log_error(f"AsyncImagePublisher: failed to publish image: {e}")
            finally:
                self._free[index].set()


def build_scene() -> None:
    """Author a small self-contained scene: a ground plane, a field of colored cubes, and lights."""
    # Flat cube as a solid ground (top surface at z = 0).
    Cube(
        ["/World/ground"],
        sizes=1.0,
        scales=np.array([[100.0, 100.0, 0.1]]),
        translations=np.array([[0.0, 0.0, -0.05]]),
        colors=np.array([[0.3, 0.3, 0.35]]),
    )

    # A field of colored cubes scattered on the ground for the camera to see. Scatter them in
    # all directions around the origin so the camera frames content regardless of its heading.
    rng = np.random.default_rng(0)
    count = 25
    translations = np.column_stack(
        [rng.uniform(-8.0, 8.0, count), rng.uniform(-8.0, 8.0, count), rng.uniform(0.25, 1.0, count)]
    )
    Cube(
        [f"/World/box_{i}" for i in range(count)],
        sizes=0.5,
        translations=translations,
        colors=rng.uniform(0.1, 1.0, (count, 3)),
    )

    # Lights so the RaytracedLighting renderer produces a lit image.
    DomeLight(["/World/DomeLight"]).set_intensities([500.0])
    DistantLight(
        ["/World/DistantLight"],
        orientations=euler_angles_to_quaternion(np.array([45.0, 0.0, 0.0]), degrees=True, extrinsic=False).numpy(),
    ).set_intensities([2000.0])


def main() -> None:
    simulation_context = SimulationContext(stage_units_in_meters=1.0)

    build_scene()

    # Create a camera overlooking the scene, aimed at the cube field around the origin
    eye = np.array([10.0, 10.0, 5.0])
    camera = RtxCamera(
        "/World/camera",
        translations=eye,
        orientations=look_at_quaternion(eye=eye, target=np.array([0.0, 0.0, 0.5])).numpy(),
    )
    # The `rgb` annotator returns a contiguous 3-channel buffer ready for a direct device-to-host copy
    sensor = CameraSensor(camera, resolution=RESOLUTION, annotators=["rgb"])

    simulation_app.update()
    simulation_context.initialize_physics()
    simulation_context.play()

    rclpy.init()
    node = rclpy.create_node("image_publisher")
    height, width = RESOLUTION
    image_publisher = AsyncImagePublisher(node, "rgb", height, width)

    print("Publishing to /rgb...")
    frame_count = 0
    try:
        while simulation_app.is_running() and simulation_context.is_playing():
            simulation_context.step(render=True)
            if not simulation_context.is_playing():
                break

            with _profile("rclpy:get_data"):
                data, _ = sensor.get_data("rgb")
            if data is not None:
                image_publisher.submit(data)

            frame_count += 1
            if args.frames > 0 and frame_count >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.close()
        print(f"Published {image_publisher.published_count} images, dropped {image_publisher.dropped_count} frames")
        node.destroy_node()
        rclpy.shutdown()
        simulation_context.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()
