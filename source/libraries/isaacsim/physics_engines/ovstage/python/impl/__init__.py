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

"""Provide process-level setup for the private OVStage runtime.

Call :func:`setup` to expose the ctypes Python package and its native-library
loader hint. Engine facades call it in the platform-appropriate load order.
Setup does not import or modify the process-wide ``pxr`` package.

    >>> from isaacsim.physics_engines.ovstage import setup
    >>> setup()
"""

from .runtime import as_native_handle, get_native_handle, lookup_stage, setup

__all__ = ["as_native_handle", "get_native_handle", "lookup_stage", "setup"]
