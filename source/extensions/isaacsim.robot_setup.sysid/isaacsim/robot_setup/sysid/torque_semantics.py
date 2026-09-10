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

"""Shared torque-channel semantics for telemetry and residual validation."""

from __future__ import annotations

from typing import Any

#: Transmitted joint torque compatible with the actuator-torque residual.
EFFORT_SEMANTICS_LINK_SIDE = "link_side"
#: Estimated contact/external torque, which is not an actuator command.
EFFORT_SEMANTICS_EXTERNAL = "external"
#: Torque data whose physical meaning was not explicitly declared.
EFFORT_SEMANTICS_UNSPECIFIED = "unspecified"


def resolve_effort_semantics(trajectory: Any, override: str | None = None) -> str:
    """Return the declared meaning of a trajectory's torque/effort channel.

    Missing semantics remain unspecified so torque-domain identification and
    residuals fail closed instead of guessing the channel's physical meaning.

    Args:
        trajectory: Trajectory whose metadata may define effort semantics.
        override: Optional explicit semantics selection.

    Returns:
        The normalized effort-semantics identifier.
    """
    if override:
        return str(override).strip().lower()
    metadata = getattr(trajectory, "metadata", None)
    extra = getattr(metadata, "extra", None)
    if isinstance(extra, dict):
        value = extra.get("torque_semantics")
        if value is None and isinstance(extra.get("topic_mapping"), dict):
            value = extra["topic_mapping"].get("torque_semantics")
        if value is None and isinstance(extra.get("column_mapping"), dict):
            value = extra["column_mapping"].get("torque_semantics")
        if value:
            return str(value).strip().lower()
    return EFFORT_SEMANTICS_UNSPECIFIED
