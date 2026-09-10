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

"""Consumer-side reader for the CUDA-IPC image path. Standalone (no GUI / Isaac Sim deps) so it
can be unit-tested. Opens the mem/event handles from an ``Image`` proto via warp, waits on the
producer's event, and returns a host numpy (or device wp) array."""

from __future__ import annotations


class IpcImageReader:
    """Opens CUDA-IPC image handles and returns host numpy (or device wp) arrays. Caches open
    mappings per mem handle, re-opening only when a buffer reappears with a different event handle
    (its producer was recreated). The epoch is per-buffer, not global, so the independent color
    and depth streams don't evict each other's mapping. Same host only; needs ``warp`` + a CUDA
    device. ``open`` never raises — returns ``None`` (and resets caches) when warp is unavailable,
    the dtype is unsupported, or a mapping is stale this frame."""

    _DTYPE_TOKENS = ("uint8", "float32")

    def __init__(self) -> None:
        # mem_handle -> (wp.array, event_handle it was opened with)
        self._arr_cache: dict = {}
        # event_handle -> wp.event
        self._evt_cache: dict = {}
        self._ipc_warned = False

    def open(self, mem_handle: bytes, dtype_token: str, shape, event_handle: bytes):
        """Open the IPC image and return a host numpy array (None to skip the frame)."""
        arr = self.open_wp(mem_handle, dtype_token, shape, event_handle)
        return None if arr is None else arr.numpy()

    def open_wp(self, mem_handle: bytes, dtype_token: str, shape, event_handle: bytes):
        """Open the IPC image and return the device ``wp.array`` (zero-copy, stays on GPU), or
        ``None`` to skip the frame. ``event_handle`` doubles as the per-buffer epoch."""
        if not mem_handle or not event_handle:
            return None
        try:
            import warp as wp
        except ImportError:
            return None

        dtype = {"uint8": wp.uint8, "float32": wp.float32}.get(dtype_token)
        if dtype is None:
            return None
        shape = tuple(shape)

        try:
            cached = self._arr_cache.get(mem_handle)
            if cached is not None and cached[1] == event_handle:
                arr = cached[0]  # same buffer + epoch -> reuse the open mapping
            else:
                # First sighting of this buffer, or its producer was recreated (the
                # paired event handle changed). Drop any prior entry for this buffer,
                # then (re)open. Keyed per buffer so the color and depth streams don't
                # evict each other every frame.
                if cached is not None:
                    self._evt_cache.pop(cached[1], None)
                    self._arr_cache.pop(mem_handle, None)
                arr = wp.from_ipc_handle(mem_handle, dtype, shape, device="cuda")
                # warp.from_ipc_handle does not raise on a failed open; it returns ptr=0.
                if arr.ptr in (None, 0):
                    return None
                self._arr_cache[mem_handle] = (arr, event_handle)
            evt = self._evt_cache.get(event_handle)
            if evt is None:
                evt = wp.event_from_ipc_handle(event_handle, device="cuda")
                self._evt_cache[event_handle] = evt
            # Order our read after the producer's write completed (GPU-side wait).
            wp.get_stream("cuda").wait_event(evt)
            return arr
        except Exception as exc:
            if not self._ipc_warned:
                self._ipc_warned = True
                import traceback as _tb

                print(
                    "[zmq_server] CUDA IPC image transfer failed — this device may not support"
                    " CUDA IPC (common on DGX Spark / Jetson Orin).\n"
                    "  Fix: in Isaac Sim open the Action Graph and set useIpc=False on the\n"
                    "  ZMQ Camera Helper node, then restart Play.\n"
                    f"  First exception: {exc}\n" + _tb.format_exc(),
                    flush=True,
                )
            self._arr_cache.clear()
            self._evt_cache.clear()
            return None
