# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Provides utilities for computing bounding boxes and collision approximations for USD prims."""

import isaacsim.core.experimental.utils.bounds as core_bounds_utils
import isaacsim.core.experimental.utils.transform as transform_utils
from isaacsim.core.experimental.utils.bounds import create_bbox_cache as create_bbox_cache
from pxr import UsdGeom

from .bounding_geometries import AABB, OBB


def compute_obb(
    bbox_cache: UsdGeom.BBoxCache,
    prim_path: str,
) -> OBB:
    """Compute an oriented bounding box for a prim path.

    Args:
        bbox_cache: Bounding box cache for prim queries.
        prim_path: USD prim path to compute bounds for.

    Returns:
        Oriented bounding box for the prim.

    Example:

    .. code-block:: python

        >>> from pxr import Usd
        >>> from isaacsim.robot_motion.experimental.motion_generation.utils.collision_approximation import (
        ...     compute_obb,
        ...     create_bbox_cache,
        ... )
        >>> bbox_cache = create_bbox_cache(Usd.TimeCode.Default())
        >>> prim_path = "/World/SomePrim"
        >>> _ = compute_obb(bbox_cache=bbox_cache, prim_path=prim_path)  # doctest: +SKIP
    """
    center, axes, half_side_lengths = core_bounds_utils.compute_obb(
        prim_path, bbox_cache=bbox_cache, space="untransformed"
    )
    rotation = transform_utils.rotation_matrix_to_quaternion(axes).numpy()
    return OBB(rotation=rotation, center=center, half_side_lengths=half_side_lengths)


def compute_world_aabb(
    bbox_cache: UsdGeom.BBoxCache,
    prim_path: str,
) -> AABB:
    """Compute a world-space axis-aligned bounding box for a prim path.

    Args:
        bbox_cache: Bounding box cache for prim queries.
        prim_path: USD prim path to compute bounds for.

    Returns:
        Axis-aligned bounding box for the prim in world coordinates.

    Example:

    .. code-block:: python

        >>> from pxr import Usd
        >>> from isaacsim.robot_motion.experimental.motion_generation.utils.collision_approximation import (
        ...     compute_world_aabb,
        ...     create_bbox_cache,
        ... )
        >>> bbox_cache = create_bbox_cache(Usd.TimeCode.Default())
        >>> prim_path = "/World/SomePrim"
        >>> _ = compute_world_aabb(bbox_cache=bbox_cache, prim_path=prim_path)  # doctest: +SKIP
    """
    bounds = core_bounds_utils.compute_aabb(prim_path, bbox_cache=bbox_cache, space="world")
    return AABB(min_bounds=bounds[:3], max_bounds=bounds[3:])
