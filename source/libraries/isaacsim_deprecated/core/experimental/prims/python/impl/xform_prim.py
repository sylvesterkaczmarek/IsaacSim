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

"""Deprecated ``XformPrim`` wrapper that preserves experimental construction semantics."""

from __future__ import annotations

import numpy as np
import warp as wp
from isaacsim.foundation.objects import Xform

__all__ = ["XformPrim"]


class XformPrim(Xform):
    """Deprecated transform wrapper with non-destructive construction semantics.

    Args:
        paths: Prim path or paths to wrap.
        resolve_paths: Whether to resolve path expressions.
        positions: Positions in the world frame.
        translations: Translations in the local frame.
        orientations: Orientations in the world frame.
        scales: Scales to apply to the prims.
        reset_xform_op_properties: Whether to replace existing transform operations with the standard operation set.
    """

    def __init__(
        self,
        paths: str | list[str],
        *,
        resolve_paths: bool = True,
        positions: list | np.ndarray | wp.array | None = None,
        translations: list | np.ndarray | wp.array | None = None,
        orientations: list | np.ndarray | wp.array | None = None,
        scales: list | np.ndarray | wp.array | None = None,
        reset_xform_op_properties: bool = False,
    ) -> None:
        super().__init__(
            paths,
            resolve_paths=resolve_paths,
            positions=positions,
            translations=translations,
            orientations=orientations,
            scales=scales,
            reset_xform_op_properties=reset_xform_op_properties,
        )
