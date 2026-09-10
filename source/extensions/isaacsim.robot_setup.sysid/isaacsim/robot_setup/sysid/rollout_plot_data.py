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

"""Plot payloads passed from the optimizer to the SysId UI."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SysIdRolloutPlotData:
    """Measured vs simulated trajectory for the nominal rollout (env 0)."""

    times: list[float]
    measured_positions: list[list[float]]
    simulated_positions: list[list[float]]
    measured_velocities: list[list[float]]
    simulated_velocities: list[list[float]]
    measured_efforts: list[list[float]] | None = None
    simulated_efforts: list[list[float]] | None = None
