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

"""Frozen policy spec for the Cartpole balancing example.

The 4-wide observation (default-relative joint positions then velocities) and the 1-wide
joint-effort action on the cart joint derive from the IO descriptor, including its 100.0 effort
scale; the pole joint is passive and never commanded, and there is no command channel. One
descriptor serves both engines, since the trained interface does not depend on the training
engine.

The spec declares no robot USD -- the runner spawns from the env config's ``spawn.usd_path`` --
and ``zero_targets_on_initialize`` clears targets and efforts across the articulation before the
first control tick, which effort-mode start semantics require.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import EllipsisType

from ..spec import PolicyArtifact, PolicySpec

__all__ = ["get_cartpole_spec"]


@functools.cache
def _get_default_cartpole_spec() -> PolicySpec:
    """Build the cached default Cartpole balancing policy spec.

    Returns:
        The resulting :class:`PolicySpec`.
    """
    from isaacsim.storage.native import get_assets_root_path

    assets_root_path = get_assets_root_path()
    if not assets_root_path:
        raise RuntimeError("Could not resolve the assets root path; check the Isaac Sim asset configuration.")

    cartpole_dir = f"{assets_root_path}/Isaac/Samples/Policies/cartpole"
    policy_dir = f"{cartpole_dir}/onnx"
    descriptor_path = f"{cartpole_dir}/cartpole_IO_descriptors.yaml"
    return PolicySpec(
        name="cartpole_balance",
        engines={
            "physx": PolicyArtifact.from_files(
                f"{policy_dir}/physx/policy.onnx",
                f"{policy_dir}/physx/env.yaml",
                descriptor_path,
            ),
            "newton": PolicyArtifact.from_files(
                f"{policy_dir}/newton/policy.onnx",
                f"{policy_dir}/newton/env.yaml",
                descriptor_path,
            ),
        },
        usd_path=None,
        default_spawn_orientation=(1.0, 0.0, 0.0, 0.0),
        zero_targets_on_initialize=True,
    )


def get_cartpole_spec(
    *,
    engines: Mapping[str, PolicyArtifact] | EllipsisType = ...,
    usd_path: str | Mapping[str, str] | None | EllipsisType = ...,
    name: str | None | EllipsisType = ...,
    default_spawn_position: tuple[float, float, float] | None | EllipsisType = ...,
    default_spawn_orientation: tuple[float, float, float, float] | None | EllipsisType = ...,
    zero_targets_on_initialize: bool | EllipsisType = ...,
    binding: Callable | None | EllipsisType = ...,
) -> PolicySpec:
    """Return the bundled Cartpole spec; omitted :class:`PolicySpec` fields keep their defaults.

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
    spec = _get_default_cartpole_spec()
    return replace(spec, **overrides) if overrides else spec
