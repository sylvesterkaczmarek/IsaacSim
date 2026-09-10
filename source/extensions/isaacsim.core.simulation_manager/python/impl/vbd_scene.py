# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""vbd scene module."""

from __future__ import annotations

from pxr import Usd

from .physics_scene import PhysicsScene


class NewtonVbdScene(PhysicsScene):
    """Newton VBD solver-specific wrapper for manipulating a USD Physics Scene prim.

    This class extends PhysicsScene to provide VBD solver-specific functionality including
    relaxation parameters, compliance settings, and damping configuration.

    Args:
        prim: USD Physics Scene prim path or prim instance.
            If the input is a path, a new USD Physics Scene prim is created if it does not exist.

    Raises:
        ValueError: If the input prim exists and is not a USD Physics Scene prim.
    """

    def __init__(self, prim: str | Usd.Prim) -> None:
        super().__init__(prim)
        if not self._prim.HasAPI("NewtonVbdSceneAPI"):
            self._prim.ApplyAPI("NewtonVbdSceneAPI")

    def get_newton_solver_type(self) -> str:
        return "vbd"
