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

"""Frozen policy spec for the Unitree Go2 flat-terrain locomotion example.

The 48-wide locomotion observation and 12-wide joint-position action derive from the selected
engine's exported IO descriptor, including its 0.25 action scale. Joint names bind literally, so
each engine deploys the asset its policy was trained against.

The spawn orientation is pinned to identity because the hosted env configs mix quaternion
conventions; the spawn position falls back to the env config's initial root state.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import EllipsisType

from ..spec import PolicyArtifact, PolicySpec

__all__ = ["get_go2_spec"]


@functools.cache
def _get_default_go2_spec() -> PolicySpec:
    """Build the cached default Go2 flat-terrain policy spec.

    Returns:
        The resulting :class:`PolicySpec`.
    """
    from isaacsim.storage.native import get_assets_root_path

    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Could not resolve the assets root path; check the Isaac Sim asset configuration.")

    policy_dir = f"{assets_root_path}/Isaac/Samples/Policies/go2"
    return PolicySpec(
        name="go2_flat_terrain",
        engines={
            "physx": PolicyArtifact.from_files(
                f"{policy_dir}/physx_policy.pt",
                f"{policy_dir}/physx_env.yaml",
                f"{policy_dir}/go2_physx_IO_descriptors.yaml",
                model_sha256="984c802b8acd68c3199605ff36603f91ac9f3192eb1ebcfd80b0e764831abe0f",
            ),
            "newton": PolicyArtifact.from_files(
                f"{policy_dir}/newton_policy.pt",
                f"{policy_dir}/newton_env.yaml",
                f"{policy_dir}/go2_newton_IO_descriptors.yaml",
                model_sha256="af16c438d737a620c1c1d883a22d0395083df9061d58fe7c3db6133ac47d1510",
            ),
        },
        usd_path=f"{assets_root_path}/Isaac/IsaacLab/Robots/Unitree/Go2/go2.usd",
        default_spawn_orientation=(1.0, 0.0, 0.0, 0.0),
    )


def get_go2_spec(
    *,
    engines: Mapping[str, PolicyArtifact] | EllipsisType = ...,
    usd_path: str | Mapping[str, str] | None | EllipsisType = ...,
    name: str | None | EllipsisType = ...,
    default_spawn_position: tuple[float, float, float] | None | EllipsisType = ...,
    default_spawn_orientation: tuple[float, float, float, float] | None | EllipsisType = ...,
    zero_targets_on_initialize: bool | EllipsisType = ...,
    binding: Callable | None | EllipsisType = ...,
) -> PolicySpec:
    """Return the bundled Go2 spec; omitted :class:`PolicySpec` fields keep their defaults.

    Args:
        engines: Per-engine artifacts keyed by engine name.
        usd_path: Robot USD path, or an engine-keyed mapping of them.
        name: Diagnostic label used in runner error messages.
        default_spawn_position: World-frame spawn position used when the caller passes none.
        default_spawn_orientation: World-frame WXYZ spawn orientation for the same fallback.
        zero_targets_on_initialize: Reset to the default state and zero all targets on initialize.
        binding: Explicit binding hook used instead of descriptor derivation.

    Returns:
        The resulting :class:`PolicySpec`.
    """
    overrides = {field: value for field, value in locals().items() if value is not ...}
    spec = _get_default_go2_spec()
    return replace(spec, **overrides) if overrides else spec
