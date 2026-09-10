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

"""Exercise shared Warp tensor helpers on CPU and CUDA devices.

The range-generation check runs unconditionally on CPU. The CUDA variant
runs when ``UMBRELLA_TEST_GPU=1`` and a Warp CUDA device is available.
"""

from __future__ import annotations

import os
import sys

import _physics_setup  # noqa: F401
import pytest
import warp as wp

_TENSORS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TENSORS_DIR not in sys.path:
    sys.path.append(_TENSORS_DIR)

import warp_utils as wp_utils  # noqa: E402


class TestWarpUtils:
    """Verify shared Warp helpers on available CPU and CUDA devices."""

    def _check_arange(self, wp_device: str) -> None:
        assert wp.is_device_available(wp_device), f"Warp device {wp_device} is not available"
        with wp.ScopedDevice(wp_device):
            n = 10
            a = wp_utils.arange(n, device=wp_device)
            result = a.numpy().squeeze()
            expected = list(range(n))
            assert (result == expected).all(), "Warp arange() failed"

    def test_warp_cpu(self) -> None:
        """Verify Warp range generation on the CPU."""
        self._check_arange("cpu")

    @pytest.mark.skipif(
        os.environ.get("UMBRELLA_TEST_GPU", "0") != "1",
        reason="GPU codepath gated on UMBRELLA_TEST_GPU=1 (requires CUDA + warp GPU device)",
    )
    def test_warp_gpu(self) -> None:
        """Verify Warp range generation on the first CUDA device."""
        self._check_arange("cuda:0")
