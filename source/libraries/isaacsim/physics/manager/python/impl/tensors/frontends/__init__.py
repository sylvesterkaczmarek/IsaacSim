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

"""Provide the Warp tensor frontend.

The native layer returns DLPack-compatible arrays. These helpers convert them
to ``wp.array`` values, resolve devices, map data types, and validate
caller-provided buffers.
"""

from __future__ import annotations

from .warp_frontend import (
    as_contiguous,
    create_tensor,
    device_ordinal_from_warp,
    dtype_from_warp,
    dtype_to_warp,
    parse_device,
    unwrap_to_array,
    wrap_tensor,
)

__all__ = [
    "as_contiguous",
    "create_tensor",
    "device_ordinal_from_warp",
    "dtype_from_warp",
    "dtype_to_warp",
    "parse_device",
    "unwrap_to_array",
    "wrap_tensor",
]
