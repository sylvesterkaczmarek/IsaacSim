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

"""Verify compact indexed writes from NumPy and Warp buffers.

A NumPy array is a supported ``set_data`` buffer. Compact scatter coerces it to
a host Warp array, writes valid indexed rows into the full-size destination,
leaves other rows zero, and rejects invalid index metadata consistently with
Warp inputs.
"""

from __future__ import annotations

import _physics_setup  # noqa: F401
import isaacsim.physics.manager.impl.tensors as t
import numpy as np
import pytest
import warp as wp


class TestScatterCompactNumpy:
    """Verify compact scatter behavior for NumPy source data."""

    def test_numpy_data(self) -> None:
        """Verify that NumPy rows scatter into selected positions of a CPU Warp array."""
        # A plain numpy ndarray is a valid set_data buffer: it must coerce to a host warp
        # array (not raise). Indices follow the warp int32 contract (like the indexed read).
        data = np.array([[10.0, 11.0], [20.0, 21.0]], dtype=np.float32)
        indices = wp.array([1, 3], dtype=wp.int32, device="cpu")
        full = t._scatter_compact_to_full(data, indices, 5)

        assert str(full.device) == "cpu", "numpy input must scatter into a host buffer"
        arr = full.numpy()
        assert arr.shape == (5, 2), "result must be full [N, D]"
        assert (arr[1] == [10.0, 11.0]).all() and (arr[3] == [20.0, 21.0]).all(), "indexed rows must scatter"
        assert (arr[0] == 0).all() and (arr[2] == 0).all() and (arr[4] == 0).all(), "non-indexed rows must stay zero"


class TestScatterCompactValidation:
    """Index validation and out-of-range handling in the compact->full scatter.

    Every check is metadata-only (int32 / 1-D / contiguous / device) plus an on-device
    out-of-range skip -- there is no host readback, matching the indexed-read gather.
    """

    def _cuda_or_skip(self) -> str:
        if wp.get_cuda_device_count() == 0:
            pytest.skip("no CUDA device")
        return "cuda:0"

    def _assert_out_of_range_skipped(self, device: str) -> None:
        # Destinations [1, 100, 3, -1] into full N=5: 1 and 3 are valid, 100 is out of
        # range and -1 is negative. The out-of-range rows must be dropped on-device
        # (no write outside `full`, no illegal access), not corrupt or crash.
        data = wp.array([[10.0, 11.0], [20.0, 21.0], [30.0, 31.0], [40.0, 41.0]], dtype=wp.float32, device=device)
        indices = wp.array([1, 100, 3, -1], dtype=wp.int32, device=device)
        arr = t._scatter_compact_to_full(data, indices, 5).numpy()
        assert (arr[1] == [10.0, 11.0]).all(), "valid destination must scatter"
        assert (arr[3] == [30.0, 31.0]).all(), "valid destination must scatter"
        assert (
            (arr[0] == 0).all() and (arr[2] == 0).all() and (arr[4] == 0).all()
        ), "out-of-range / negative destinations must not write outside full"

    def test_out_of_range_skipped_cpu(self) -> None:
        """Verify that CPU scatter ignores negative and oversized destination indices."""
        self._assert_out_of_range_skipped("cpu")

    def test_out_of_range_skipped_cuda(self) -> None:
        """Verify that CUDA scatter ignores negative and oversized destination indices."""
        self._assert_out_of_range_skipped(self._cuda_or_skip())

    def test_rejects_non_warp_indices(self) -> None:
        """Verify that NumPy arrays and Python lists are rejected as index buffers."""
        # Indices follow the warp-only contract (like the indexed read): a numpy array or
        # a Python list is rejected, not silently coerced.
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="warp int32"):
            t._scatter_compact_to_full(data, np.array([0], dtype=np.int32), 5)
        with pytest.raises(ValueError, match="warp int32"):
            t._scatter_compact_to_full(data, [0], 5)

    def test_rejects_int64_indices(self) -> None:
        """Verify that Warp index buffers must use 32-bit integers."""
        # A warp non-int32 index is rejected, not silently converted (the int32 contract).
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="int32"):
            t._scatter_compact_to_full(data, wp.array([0], dtype=wp.int64, device="cpu"), 5)

    def test_rejects_2d_indices(self) -> None:
        """Verify that Warp index buffers must be one-dimensional."""
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="1-D"):
            t._scatter_compact_to_full(data, wp.zeros((1, 1), dtype=wp.int32, device="cpu"), 5)

    def test_rejects_noncontiguous_warp_indices(self) -> None:
        """Verify that strided Warp index views are rejected."""
        # A strided warp view is rejected -- the kernel reads a flat int32[K].
        strided = wp.array([0, 9, 3, 9], dtype=wp.int32, device="cpu")[::2]  # [0, 3], non-contiguous
        data = wp.array([[1.0, 2.0], [3.0, 4.0]], dtype=wp.float32, device="cpu")
        with pytest.raises(ValueError, match="contiguous"):
            t._scatter_compact_to_full(data, strided, 5)
