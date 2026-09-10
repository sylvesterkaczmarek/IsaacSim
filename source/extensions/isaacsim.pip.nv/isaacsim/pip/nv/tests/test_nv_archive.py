# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Test suite for verifying the NVIDIA CUDA pip archive is available."""

import importlib.util

import omni.kit.test


class TestPipArchive(omni.kit.test.AsyncTestCase):
    """Test that all NVIDIA CUDA pip archive packages can be found."""

    async def test_find_all(self) -> None:
        """Test that all NVIDIA CUDA runtime packages are importable."""
        packages = [
            "cuda.bindings",
            "nvidia.cublas",
            "nvidia.cuda_cupti",
            "nvidia.cuda_nvrtc",
            "nvidia.cuda_runtime",
            "nvidia.cudnn",
            "nvidia.cufft",
            "nvidia.cufile",
            "nvidia.curand",
            "nvidia.cusolver",
            "nvidia.cusparse",
            "nvidia.cusparselt",
            "nvidia.nccl",
            "nvidia.nvjitlink",
            "nvidia.nvshmem",
            "nvidia.nvtx",
        ]
        for package in packages:
            self.assertIsNotNone(importlib.util.find_spec(package), f"{package} not found")
