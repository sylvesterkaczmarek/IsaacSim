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

"""Tests for zmq_server.ipc_image.IpcImageReader.

Two layers:
  (a) Mock-warp tests of the epoch/cache/error logic — no GPU required.
  (b) A real cross-process CUDA-IPC round trip (GPU + warp required, else skipped):
      a producer subprocess exports a cudaMalloc-backed array + interprocess event,
      and the reader opens the handles and verifies the pixels match.

Run:  PYTHONPATH=src python -m unittest zmq_server.tests.test_ipc_image -v
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import types
import unittest

from zmq_server.ipc_image import IpcImageReader


# ---------------------------------------------------------------------------
# (a) Mock-warp tests — pure epoch/cache/error logic, no GPU.
# ---------------------------------------------------------------------------
class _FakeArray:
    def __init__(self, ptr, value):
        self.ptr = ptr
        self._value = value

    def numpy(self):
        return self._value


def _install_fake_warp(*, ptr=0xCAFE, wait_raises=False):
    """Insert a fake `warp` module and return a call-count recorder."""
    rec = {"from_ipc": 0, "event_from_ipc": 0, "wait": 0}
    fake = types.ModuleType("warp")
    fake.uint8 = "U8"
    fake.float32 = "F32"

    def from_ipc_handle(mem, dtype, shape, device=None):
        rec["from_ipc"] += 1
        return _FakeArray(ptr, ("ARR", mem, dtype, tuple(shape)))

    def event_from_ipc_handle(evt, device=None):
        rec["event_from_ipc"] += 1
        return ("EVT", evt)

    class _Stream:
        def wait_event(self, evt):
            rec["wait"] += 1
            if wait_raises:
                raise RuntimeError("simulated stale mapping")

    fake.from_ipc_handle = from_ipc_handle
    fake.event_from_ipc_handle = event_from_ipc_handle
    fake.get_stream = lambda device=None: _Stream()
    sys.modules["warp"] = fake
    return rec


class TestIpcImageReaderLogic(unittest.TestCase):
    def setUp(self):
        self._saved_warp = sys.modules.get("warp")

    def tearDown(self):
        if self._saved_warp is not None:
            sys.modules["warp"] = self._saved_warp
        else:
            sys.modules.pop("warp", None)

    def test_open_returns_array(self):
        _install_fake_warp()
        r = IpcImageReader()
        out = r.open(b"mem", "uint8", (8, 8, 4), b"evt")
        self.assertEqual(out, ("ARR", b"mem", "U8", (8, 8, 4)))

    def test_caches_array_and_event(self):
        rec = _install_fake_warp()
        r = IpcImageReader()
        r.open(b"mem", "uint8", (8, 8, 4), b"evt")
        r.open(b"mem", "uint8", (8, 8, 4), b"evt")
        self.assertEqual(rec["from_ipc"], 1, "array open should be cached")
        self.assertEqual(rec["event_from_ipc"], 1, "event open should be cached")
        self.assertEqual(rec["wait"], 2, "wait_event runs every frame")

    def test_epoch_change_drops_caches(self):
        rec = _install_fake_warp()
        r = IpcImageReader()
        r.open(b"mem", "uint8", (8, 8, 4), b"evtA")
        r.open(b"mem", "uint8", (8, 8, 4), b"evtB")  # same buffer, new event => recreated
        self.assertEqual(rec["from_ipc"], 2, "epoch change must re-open the array")
        self.assertEqual(rec["event_from_ipc"], 2)

    def test_two_streams_do_not_evict_each_other(self):
        # Color and depth are independent streams, each with its own mem + event
        # handle. Alternating between them must NOT thrash the cache: each buffer
        # opens once, then every subsequent frame is a cache hit (no re-open). A
        # re-open here would fail in practice (already-mapped) and drop the frame.
        rec = _install_fake_warp()
        r = IpcImageReader()
        for _ in range(5):
            r.open(b"mem_rgb", "uint8", (8, 8, 4), b"evt_rgb")
            r.open(b"mem_depth", "float32", (8, 8), b"evt_depth")
        self.assertEqual(rec["from_ipc"], 2, "each stream's buffer must open exactly once")
        self.assertEqual(rec["event_from_ipc"], 2, "each stream's event must open exactly once")
        self.assertEqual(rec["wait"], 10, "wait_event still runs every frame")

    def test_null_ptr_returns_none(self):
        _install_fake_warp(ptr=0)
        r = IpcImageReader()
        self.assertIsNone(r.open(b"mem", "uint8", (8, 8, 4), b"evt"))

    def test_unknown_dtype_returns_none(self):
        _install_fake_warp()
        r = IpcImageReader()
        self.assertIsNone(r.open(b"mem", "float64", (8, 8), b"evt"))

    def test_empty_handles_return_none(self):
        _install_fake_warp()
        r = IpcImageReader()
        self.assertIsNone(r.open(b"", "uint8", (8, 8, 4), b"evt"))
        self.assertIsNone(r.open(b"mem", "uint8", (8, 8, 4), b""))

    def test_exception_resets_epoch_and_recovers(self):
        rec = _install_fake_warp(wait_raises=True)
        r = IpcImageReader()
        self.assertIsNone(r.open(b"mem", "uint8", (8, 8, 4), b"evt"))
        # Epoch was reset, so the next call re-opens (caches were cleared).
        self.assertIsNone(r.open(b"mem", "uint8", (8, 8, 4), b"evt"))
        self.assertEqual(rec["from_ipc"], 2, "caches must be cleared after an error")

    def test_warp_unavailable_returns_none(self):
        sys.modules["warp"] = None  # makes `import warp` raise ImportError
        r = IpcImageReader()
        self.assertIsNone(r.open(b"mem", "uint8", (8, 8, 4), b"evt"))


# ---------------------------------------------------------------------------
# (b) Real cross-process CUDA-IPC round trip — GPU + warp required.
# ---------------------------------------------------------------------------
def _cuda_available() -> bool:
    try:
        import warp as wp

        wp.init()
        return bool(wp.is_cuda_available())
    except Exception:
        return False


# Producer: allocate a cudaMalloc-backed (IPC-exportable) array, fill a known
# pattern, export the mem + interprocess-event handles, then block until the
# parent has read so the allocation stays alive.
_PRODUCER_SRC = r"""
import base64, json, sys
import numpy as np
import warp as wp

wp.init()
H, W = 8, 8
n = H * W * 4
pattern = (np.arange(n, dtype=np.uint64) % 251).astype(np.uint8)
with wp.ScopedMempool("cuda", False):          # default cudaMalloc => IPC-exportable
    arr = wp.empty(n, dtype=wp.uint8, device="cuda")
wp.copy(arr, wp.array(pattern, dtype=wp.uint8, device="cuda"))
evt = wp.Event(device="cuda", interprocess=True)
wp.get_stream("cuda").record_event(evt)
wp.synchronize()
sys.stdout.write(json.dumps({
    "mem": base64.b64encode(bytes(arr.ipc_handle())).decode(),
    "evt": base64.b64encode(bytes(evt.ipc_handle())).decode(),
    "shape": [H, W, 4],
    "pattern": base64.b64encode(pattern.tobytes()).decode(),
}) + "\n")
sys.stdout.flush()
sys.stdin.readline()  # keep the allocation + event alive until the parent is done
"""


@unittest.skipUnless(_cuda_available(), "CUDA device + warp required for IPC round trip")
class TestIpcImageRoundTripGPU(unittest.TestCase):
    def test_cross_process_round_trip(self):
        import numpy as np

        proc = subprocess.Popen(
            [sys.executable, "-c", _PRODUCER_SRC],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            # warp may print an init banner before our JSON; scan for the JSON line.
            info = None
            for _ in range(100):
                line = proc.stdout.readline()
                if not line:
                    break
                line = line.strip()
                if line.startswith("{"):
                    info = json.loads(line)
                    break
            if info is None:
                err = proc.stderr.read()
                self.skipTest(f"IPC producer emitted no handle (env issue):\n{err[:800]}")
            mem = base64.b64decode(info["mem"])
            evt = base64.b64decode(info["evt"])
            expected = np.frombuffer(base64.b64decode(info["pattern"]), dtype=np.uint8).reshape(8, 8, 4)

            reader = IpcImageReader()
            arr = reader.open(mem, "uint8", tuple(info["shape"]), evt)
            self.assertIsNotNone(arr, "reader.open returned None for a live IPC handle")
            np.testing.assert_array_equal(arr, expected)

            # A second open with the same handles must hit the cache (no re-open error).
            arr2 = reader.open(mem, "uint8", tuple(info["shape"]), evt)
            np.testing.assert_array_equal(arr2, expected)
        finally:
            try:
                proc.stdin.write("done\n")
                proc.stdin.flush()
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    stream.close()
                except Exception:
                    pass


if __name__ == "__main__":
    unittest.main()
