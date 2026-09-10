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

"""Declare unimplemented Newton SDF-shape tensor scenarios.

The Newton tensor backend exposes no SDF-shape entity view, so these decorated
placeholders record the unsupported CPU and GPU operations.
"""

from __future__ import annotations

import os
import sys

import _physics_setup  # noqa: F401

_TENSORS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TENSORS_DIR not in sys.path:
    sys.path.append(_TENSORS_DIR)

from _legacy_runner import unimplemented_placeholder  # noqa: E402


@unimplemented_placeholder
class TestSdfShapeView:
    """Record unimplemented Newton SDF-shape view scenarios."""

    def test_sdf_shapes_newton_cc(self) -> None:
        """Record the unimplemented SDF shapes case for Newton."""

    def test_sdf_shapes_newton_gg(self) -> None:
        """Record the unimplemented SDF shapes case for Newton."""
