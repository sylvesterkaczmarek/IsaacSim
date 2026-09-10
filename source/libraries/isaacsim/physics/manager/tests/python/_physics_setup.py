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

"""Configure the shared physics runtime used by manager tests."""

from __future__ import annotations

import isaacsim.physics.registration as physics_registration

# Configure the private OVStage runtime and register the OvPhysX backend.
import isaacsim.physics_engines.ovphysx as _ovphysx

_ovphysx.activate()

try:
    import warp as _wp

    if not _wp.is_initialized():
        _wp.init()
except Exception:
    pass

OVPHYSX_SIM_NAME = "ovphysx"


def set_suppress_readback(enable: bool) -> None:
    """Select whether OvPhysX keeps simulation data on the GPU.

    Args:
        enable: Whether to suppress CPU readback.

    """
    _ovphysx.set_suppress_readback(enable)


def find_ovphysx_sim_id() -> physics_registration.SimulationId | None:
    """Find the registered OvPhysX simulation identifier.

    Returns:
        The OvPhysX identifier, or None when the backend is not registered.

    """
    physics = physics_registration
    for sim_id in physics.get_simulation_ids():
        if physics.get_simulation_name(sim_id) == OVPHYSX_SIM_NAME:
            return sim_id
    return None
