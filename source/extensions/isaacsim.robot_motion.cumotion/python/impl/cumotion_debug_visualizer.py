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

"""Kit-independent contract for optional cuMotion debug visualization."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CumotionDebugVisualizer(Protocol):
    """Interface used by the cuMotion world to visualize collision geometry."""

    def create_sphere(self, source_prim_path: str, geometry_index: int, radius: float, enabled: bool) -> str:
        """Create a sphere debug prim and return its path."""

    def create_cube(self, source_prim_path: str, geometry_index: int, side_lengths: np.ndarray, enabled: bool) -> str:
        """Create a cube debug prim and return its path."""

    def create_capsule(
        self,
        source_prim_path: str,
        geometry_index: int,
        radius: float,
        height: float,
        enabled: bool,
    ) -> str:
        """Create a capsule debug prim and return its path."""

    def update_poses(self, paths: list[str], positions: np.ndarray, orientations: np.ndarray) -> None:
        """Update world poses for debug prims."""

    def set_enabled(self, path: str, enabled: bool) -> None:
        """Update the visual state of a debug prim."""
