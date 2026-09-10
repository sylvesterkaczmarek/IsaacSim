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

"""GPU round-trip test for the OgnZMQPublishImage CUDA-IPC export path.

Runs the real C++ node in an OmniGraph (this process = producer), feeding it a
synthetic GPU buffer (no render product needed — the same `dataPtr` contract the
scene uses). The node D2D-copies into its IPC-export buffer, records its
interprocess event, and publishes an Image proto with the handles. We receive the
proto in-process, then open the handles in a SUBPROCESS (CUDA forbids opening an
IPC handle in the exporting process) and verify the pixels match.

This covers the seam the pure-warp round-trip in the standalone client's
zmq_server/tests/test_ipc_image.py cannot: a handle exported by raw
cudart (cudaIpcGetMemHandle + cudaEventCreateWithFlags(interprocess)) inside the
C++ node, marshalled through the proto, opened by warp.

Requires a CUDA device + warp + the built isaacsim.zmq.nodes plugin; skipped
otherwise. Only the linear-buffer copy path is exercised (the texture/mipmapped
path needs a real render product).
"""

import base64
import os
import socket as _socket
import subprocess
import sys
import textwrap
import unittest

import omni
import omni.graph.core as og
import omni.kit.test


def find_available_port(start: int = 16400) -> int:
    for port in range(start, start + 200):
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("No available port found in range")


def _cuda_available() -> bool:
    try:
        import warp as wp

        wp.init()
        return bool(wp.is_cuda_available())
    except Exception:
        return False


# Consumer: open the IPC handles exported by the node (in the parent process) and
# verify the pixels. Exit codes: 0 ok, 2 null mapping, 3 mismatch, 4 import error.
_CONSUMER_SRC = textwrap.dedent("""
    import base64, sys
    try:
        import numpy as np
        import warp as wp
    except Exception as e:
        sys.stderr.write(f"import failed: {e}\\n"); sys.exit(4)
    mem = base64.b64decode(sys.argv[1])
    evt = base64.b64decode(sys.argv[2])
    H, W = int(sys.argv[3]), int(sys.argv[4])
    wp.init()
    arr = wp.from_ipc_handle(mem, wp.uint8, (H, W, 4), device="cuda")
    if arr.ptr in (None, 0):
        sys.stderr.write("from_ipc_handle returned ptr=0\\n"); sys.exit(2)
    ev = wp.event_from_ipc_handle(evt, device="cuda")
    wp.get_stream("cuda").wait_event(ev)
    got = arr.numpy()
    expected = (np.arange(H * W * 4, dtype=np.uint64) % 251).astype(np.uint8).reshape(H, W, 4)
    if np.array_equal(got, expected):
        sys.exit(0)
    sys.stderr.write(
        "mismatch: got.shape=%s sum=%d allzero=%s ndiff=%d got[:8]=%s exp[:8]=%s\\n"
        % (got.shape, int(got.sum()), bool((got == 0).all()),
           int((got != expected).sum()), got.flatten()[:8].tolist(), expected.flatten()[:8].tolist())
    )
    sys.exit(3)
    """)


class TestPublishImageIpcGPU(omni.kit.test.AsyncTestCase):
    """Exercises the real OgnZMQPublishImage CUDA-IPC export path end-to-end."""

    async def setUp(self):
        super().setUp()
        await omni.usd.get_context().new_stage_async()
        await omni.kit.app.get_app().next_update_async()
        self.port = find_available_port()
        self._src = None  # keep the GPU input buffer alive for the test's lifetime

    async def tearDown(self):
        omni.timeline.get_timeline_interface().stop()
        for _ in range(5):
            await omni.kit.app.get_app().next_update_async()
        self._src = None
        super().tearDown()

    @unittest.skipUnless(_cuda_available(), "CUDA device + warp required")
    async def test_ipc_publish_roundtrip_via_subprocess(self):
        try:
            import zmq
        except ImportError:
            self.skipTest("zmq Python package not available")
        try:
            from isaacsim.zmq.protos import image_pb2
        except ImportError:
            self.skipTest("isaacsim.zmq.core proto modules not available")
        import numpy as np
        import warp as wp

        wp.init()
        H, W = 16, 16
        n = H * W * 4
        pattern = (np.arange(n, dtype=np.uint64) % 251).astype(np.uint8)
        # Synthetic GPU input buffer (stands in for the render-product annotator ptr).
        self._src = wp.array(pattern.reshape(H, W, 4), dtype=wp.uint8, device="cuda")
        wp.synchronize()

        ctx = zmq.Context()
        pull = ctx.socket(zmq.PULL)
        pull.setsockopt(zmq.RCVTIMEO, 5000)
        pull.bind(f"tcp://127.0.0.1:{self.port}")
        try:
            og.Controller.edit(
                {"graph_path": "/TestGraph", "evaluator_name": "execution"},
                {
                    og.Controller.Keys.CREATE_NODES: [
                        ("OnPlayback", "omni.graph.action.OnPlaybackTick"),
                        ("PublishImage", "isaacsim.zmq.nodes.ZMQPublishImage"),
                    ],
                    og.Controller.Keys.SET_VALUES: [
                        ("PublishImage.inputs:ip", "localhost"),
                        ("PublishImage.inputs:port", self.port),
                        ("PublishImage.inputs:encoding", "rgba8"),
                        ("PublishImage.inputs:width", W),
                        ("PublishImage.inputs:height", H),
                        ("PublishImage.inputs:useIpc", True),
                        ("PublishImage.inputs:cudaDeviceIndex", 0),
                        ("PublishImage.inputs:bufferSize", n),
                        ("PublishImage.inputs:dataPtr", int(self._src.ptr)),
                    ],
                    og.Controller.Keys.CONNECT: [
                        ("OnPlayback.outputs:tick", "PublishImage.inputs:execIn"),
                    ],
                },
            )

            omni.timeline.get_timeline_interface().play()
            for _ in range(30):
                await omni.kit.app.get_app().next_update_async()

            try:
                frames = pull.recv_multipart()
            except zmq.Again:
                self.fail("Timed out waiting for the published Image message")

            self.assertEqual(len(frames), 2)
            self.assertEqual(frames[0].decode(), "image")
            msg = image_pb2.Image()
            msg.ParseFromString(frames[1])

            # Confirm the node actually took the IPC path (not the bytes fallback).
            self.assertEqual(msg.WhichOneof("pixel_transport"), "gpu", "node published bytes, not a CUDA-IPC handle")
            self.assertTrue(msg.gpu.array.mem_handle, "missing CUDA-IPC mem handle")
            self.assertEqual(msg.gpu.array.dtype, "uint8")
            self.assertEqual(list(msg.gpu.array.shape), [H, W, 4])
            self.assertTrue(msg.gpu.ipc_event_handle, "missing interprocess event handle")

            # Freeze the export buffer (stop overwriting it) while the consumer reads.
            omni.timeline.get_timeline_interface().stop()
            await omni.kit.app.get_app().next_update_async()

            # Open the handles in a separate process (same-process IPC open is illegal).
            # The consumer must use the same interpreter as kit (warp ABI), so require
            # sys.executable to be a python; skip cleanly if the runner exposes the kit
            # binary instead (can't spawn a matching interpreter here).
            py_exe = sys.executable
            if not (py_exe and "python" in os.path.basename(py_exe).lower()):
                self.skipTest(f"no python interpreter available to spawn the IPC consumer (sys.executable={py_exe!r})")
            # Pass warp's location so the bare subprocess can import it.
            warp_parent = os.path.dirname(os.path.dirname(os.path.abspath(wp.__file__)))
            env = dict(os.environ)
            env["PYTHONPATH"] = warp_parent + os.pathsep + env.get("PYTHONPATH", "")
            result = subprocess.run(
                [
                    py_exe,
                    "-c",
                    _CONSUMER_SRC,
                    base64.b64encode(bytes(msg.gpu.array.mem_handle)).decode(),
                    base64.b64encode(bytes(msg.gpu.ipc_event_handle)).decode(),
                    str(H),
                    str(W),
                ],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                stderr = result.stderr or ""
                # Opening a CUDA IPC memory handle in another process is not supported on every
                # runner (containers without --ipc=host, MPS/driver constraints). When that is the
                # cause, skip rather than hard-fail — genuine failures (pixel mismatch, other
                # errors) still fail below.
                if "ipc_open_mem_handle" in stderr.lower() or "cudaipcopenmemhandle" in stderr.lower():
                    self.skipTest(
                        "cross-process CUDA IPC is unavailable on this runner "
                        f"(consumer rc={result.returncode}): {stderr.strip()[:300]}"
                    )
                self.fail(f"IPC consumer subprocess failed (rc={result.returncode}). stderr:\n{stderr[:1000]}")
        finally:
            pull.close()
            ctx.term()
