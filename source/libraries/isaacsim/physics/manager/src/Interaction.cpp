// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// clang-format off
#include <isaacsim/physics/manager/PhysicsInteraction.hpp>
// clang-format on

#include "SimulationSnapshot.hpp"

#include <isaacsim/physics/registration/Physics.hpp>


namespace isaacsim
{
namespace physics
{
namespace manager
{

using namespace registration;

// Called when a raycast request is executed - used for picking.
void handleRaycast(const Float3& origin, const Float3& direction, bool inputActive)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.interactionFunctions.handleRaycast)
        {
            simulation.second.simulation.interactionFunctions.handleRaycast(origin, direction, inputActive);
        }
    }
}

// Aggregate debug data from every active simulation. Later entries overwrite earlier ones
// on key collision (matches the previous carb::dictionary merge semantics with
// overwriteOriginal).
DebugDataDictionary getPrimDebugData(const std::string& primPath)
{
    DebugDataDictionary result;
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.interactionFunctions.getPrimDebugData)
        {
            DebugDataDictionary partial = simulation.second.simulation.interactionFunctions.getPrimDebugData(primPath);
            for (auto& entry : partial)
            {
                result[entry.first] = std::move(entry.second);
            }
        }
    }
    return result;
}
} // namespace manager
} // namespace physics
} // namespace isaacsim
