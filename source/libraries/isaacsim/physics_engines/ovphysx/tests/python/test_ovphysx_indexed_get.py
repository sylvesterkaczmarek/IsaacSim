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

"""Validate indexed reads through OvPhysX entity views.

The scenario checks index-sized results, row ordering, output buffers, device
placement, duplicate and empty selections, invalid index layouts, out-of-range
rows, retained-result ownership, and full reads without an index selection.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import _physics_setup  # noqa: F401
import numpy as np
import pytest
import warp as wp

_TENSORS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TENSORS_DIR not in sys.path:
    sys.path.insert(0, _TENSORS_DIR)

from _legacy_runner import cpu_device, gpu_device, gpu_only, run_scenario  # noqa: E402
from _scenario import DeviceParams, GridParams, GridTestBase, SimParams, Transform  # noqa: E402


def _to_numpy(t: Any) -> np.ndarray:
    """Convert a tensor-like value to a NumPy array.

    Args:
        t: Value that either exposes ``numpy()`` or supports NumPy conversion.

    Returns:
        NumPy representation of the input value.

    """
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


class _IndexedGetScenario(GridTestBase):
    """Exercise indexed rigid-body transform reads and input validation.

    Args:
        test_case: Test object that owns scenario assertions.
        device_params: Simulation and tensor device selection.

    """

    def __init__(self, test_case: object, device_params: DeviceParams) -> None:
        super().__init__(test_case, GridParams(num_envs=8, env_spacing=1.0), SimParams(), device_params)
        self.create_rigid_ball(self.env_template_path.AppendChild("ball"), Transform((0.0, 0.0, 0.5)), 0.15)

    def on_start(self, sim: Any) -> None:
        """Validate indexed transform reads and their buffer contract.

        The checks cover row order and device placement, caller-provided
        buffers, repeated and empty selections, invalid indices and outputs,
        boolean conversion, retained-result ownership, and full reads.

        Args:
            sim: Simulation view under test.

        """
        balls = sim.create_rigid_body_view("/envs/*/ball")
        n = int(balls.count)

        full = _to_numpy(balls.get_data("transforms")).reshape(n, 7)

        sel = np.array([0, 2, n - 1], dtype=np.int32)
        wp_sel = wp.from_numpy(sel, dtype=wp.int32, device=self.wp_device)
        indexed_raw = balls.get_data("transforms", wp_sel)
        indexed = _to_numpy(indexed_raw).reshape(len(sel), 7)

        # An indexed get returns index-size,
        assert indexed.shape[0] == len(sel), f"indexed get must return index-size {len(sel)}, got {indexed.shape[0]}"
        # with rows matching the full read at those indices,
        assert np.allclose(
            indexed, full[sel], rtol=1e-5, atol=1e-5
        ), "indexed get rows must match the full read at the same indices"
        # and stays on the request device (GPU gather via warp kernel, else
        # the CPU gather) -- no host round-trip.
        assert indexed_raw.device == wp_sel.device, "indexed get result must stay on the request device"
        # indexed EXTERNAL get: gather into a caller-provided device buffer
        # (a fresh one is allocated only when absent), staying on the device.
        out_buf = wp.zeros((len(sel), 7), dtype=wp.float32, device=self.wp_device)
        ext = balls.get_data("transforms", wp_sel, out_buf)
        assert ext.device == wp_sel.device, "indexed external get must stay on the request device"
        assert np.allclose(
            _to_numpy(ext).reshape(len(sel), 7), full[sel], rtol=1e-5, atol=1e-5
        ), "indexed external get must gather into the caller buffer"
        # Indexed BOOL reads convert uint8 storage to float32 and match the
        # corresponding rows from a full read. BOOL bindings convert on the host,
        # so this check applies to the CPU gather.
        if "cuda" not in str(self.wp_device):
            flags = np.zeros(n, dtype=np.uint8)
            flags[0] = 1
            flags[2] = 1
            balls.set_data("disable-simulations", wp.from_numpy(flags, dtype=wp.uint8, device="cpu"))
            full_bool = _to_numpy(balls.get_data("disable-simulations")).reshape(-1)
            bool_indexed = _to_numpy(balls.get_data("disable-simulations", wp_sel)).reshape(-1)
            assert bool_indexed.dtype == np.float32, "indexed bool read must return float32"
            assert bool_indexed.shape[0] == len(sel), "indexed bool read must be index-size"
            assert np.allclose(
                bool_indexed, full_bool[sel]
            ), "indexed bool read must gather + convert like the full read"
        # K > N (duplicate / repeat indices): result is index-size K, not
        # clamped to N -- guards the CPU gather buffer against overflow.
        dup = np.concatenate([np.arange(n), [0, 1]]).astype(np.int32)  # K = N + 2 > N
        wp_dup = wp.from_numpy(dup, dtype=wp.int32, device=self.wp_device)
        got = _to_numpy(balls.get_data("transforms", wp_dup)).reshape(len(dup), 7)
        assert got.shape[0] == len(dup), "K>N gather must return K rows"
        assert np.allclose(
            got, full[dup], rtol=1e-5, atol=1e-5
        ), "K>N gather (duplicate indices) must match the full read"
        # Out-of-range indices are dropped on both lanes rather than rejected: rejecting
        # them would need a readback of the index values. The dropped row carries the
        # all-bits-set marker -- NaN for this float binding -- so it cannot be mistaken
        # for data the way a zero row could. Valid rows still gather.
        oob = np.array([n + 100, sel[1], -1], dtype=np.int32)  # slot 0 positive, slot 2 negative
        oob_got = _to_numpy(
            balls.get_data("transforms", wp.from_numpy(oob, dtype=wp.int32, device=self.wp_device))
        ).reshape(len(oob), 7)
        assert np.all(np.isnan(oob_got[0])), "positive out-of-range index must yield the NaN marker"
        assert np.all(np.isnan(oob_got[2])), "negative out-of-range index must yield the NaN marker"
        assert np.allclose(oob_got[1], full[sel[1]], rtol=1e-5, atol=1e-5), "valid rows must still gather"
        # a mismatched external out (wrong row count) is rejected loudly, not
        # silently overflowed (guards the out validation on both lanes).
        bad_out = wp.zeros((len(sel) - 1, 7), dtype=wp.float32, device=self.wp_device)
        with pytest.raises(Exception, match="out buffer"):
            balls.get_data("transforms", wp_sel, bad_out)
        # A retained indexed result owns stable storage: a small-K gather held
        # across a later larger-K gather must preserve its original values.
        held = balls.get_data("transforms", wp.from_numpy(sel, dtype=wp.int32, device=self.wp_device))
        held_snapshot = _to_numpy(held).reshape(len(sel), 7).copy()
        big = np.arange(n, dtype=np.int32)  # larger K forces a bigger internal allocation
        balls.get_data("transforms", wp.from_numpy(big, dtype=wp.int32, device=self.wp_device))
        assert np.allclose(
            _to_numpy(held).reshape(len(sel), 7), held_snapshot, rtol=1e-5, atol=1e-5
        ), "a retained indexed result must keep stable storage across a later larger gather"
        # a non-1-D or strided index is rejected loudly on both lanes -- the CPU and GPU
        # gathers treat the index buffer as a flat int32[K], so a 2-D or strided view would
        # otherwise be misread (partial consume / adjacent values).
        idx_2d = wp.from_numpy(sel.reshape(-1, 1), dtype=wp.int32, device=self.wp_device)  # [K, 1]
        with pytest.raises(Exception):
            balls.get_data("transforms", idx_2d)
        strided_src = np.empty(2 * len(sel), dtype=np.int32)
        strided_src[::2] = sel  # logical (valid, in-range) values, non-contiguous storage
        strided = wp.from_numpy(strided_src, dtype=wp.int32, device=self.wp_device)[::2]
        with pytest.raises(Exception):
            balls.get_data("transforms", strided)
        # an explicitly empty selection returns K=0 rows -- distinct from None
        # (no selection -> full N). Empty subsets are valid (e.g. "update 0 envs").
        empty_got = balls.get_data("transforms", wp.zeros(0, dtype=wp.int32, device=self.wp_device))
        assert empty_got is not None, "empty selection must return a 0-row array, not None"
        assert _to_numpy(empty_got).reshape(-1, 7).shape[0] == 0, "empty selection must yield K=0 rows, not full N"
        # A supplied zero-size index is still validated: Warp gives (0,), (0, 1),
        # and (1, 0) a null pointer, so they must not slip past the dtype/rank
        # checks the way an omitted (None) index does -- empty int64 is a wrong
        # dtype, and (0, 1) / (1, 0) are non-1-D (the latter would otherwise report
        # K=1 and dereference the null buffer).
        for bad in (
            wp.zeros(0, dtype=wp.int64, device=self.wp_device),
            wp.zeros((0, 1), dtype=wp.int32, device=self.wp_device),
            wp.zeros((1, 0), dtype=wp.int32, device=self.wp_device),
        ):
            with pytest.raises(Exception):
                balls.get_data("transforms", bad)
        # The following guard the C++ CPU gather path; the GPU lane routes indexed
        # reads through the device gather kernel, which never reaches it.
        if str(self.wp_device) == "cpu":
            # A valid zero-row CPU out is recognized as supplied (validated,
            # not silently dropped) and yields a 0-row result -- isEmpty() must not
            # fold a null-data [0, cols] buffer into the no-out case.
            zero_out = wp.zeros((0, 7), dtype=wp.float32, device=self.wp_device)
            zero_ext = balls.get_data("transforms", wp.zeros(0, dtype=wp.int32, device=self.wp_device), zero_out)
            assert _to_numpy(zero_ext).reshape(-1, 7).shape[0] == 0, "zero-row external get must return 0 rows"
            # A supplied GPU out is rejected loudly (the host-side gather can't fill
            # device memory) rather than silently returning a different CPU buffer,
            # for both a normal and a zero-row selection -- the latter
            # confirms a zero-row out is recognized as supplied, not absent.
            if wp.get_cuda_device_count() > 0:
                gpu_out = wp.zeros((len(sel), 7), dtype=wp.float32, device="cuda:0")
                with pytest.raises(Exception, match="CPU out buffer"):
                    balls.get_data("transforms", wp_sel, gpu_out)
                zero_gpu_out = wp.zeros((0, 7), dtype=wp.float32, device="cuda:0")
                with pytest.raises(Exception, match="CPU out buffer"):
                    balls.get_data("transforms", wp.zeros(0, dtype=wp.int32, device=self.wp_device), zero_gpu_out)
        # a no-index get still returns full-size.
        assert _to_numpy(balls.get_data("transforms")).reshape(n, 7).shape[0] == n, "no-index get must return full-size"
        # int64 indices are rejected -- indexed GET enforces the same int32
        # convention as the SET impls (CPU gather and GPU gather both guard it).
        sel64 = wp.from_numpy(sel.astype(np.int64), dtype=wp.int64, device=self.wp_device)
        with pytest.raises(Exception, match="must be int32"):
            balls.get_data("transforms", sel64)
        self.finish()

    def on_physics_step(self, sim: object, stepno: int, dt: float) -> None:
        pass


class TestOvPhysxIndexedGet:
    """Validate indexed rigid-body reads, buffers, and input guards."""

    def test_indexed_get_returns_index_size_ovphysx_cc(self) -> None:
        """Check indexed-read contracts with CPU simulation and CPU tensors."""
        run_scenario(self, _IndexedGetScenario, "ovphysx", cpu_device())

    @gpu_only
    def test_indexed_get_returns_index_size_ovphysx_gg(self) -> None:
        """Check indexed-read contracts with GPU simulation and GPU tensors."""
        run_scenario(self, _IndexedGetScenario, "ovphysx", gpu_device())
