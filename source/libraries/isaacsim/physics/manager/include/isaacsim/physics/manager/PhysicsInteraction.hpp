// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <isaacsim/physics/manager/Export.h>
#include <isaacsim/physics/registration/Types.hpp>
#include <isaacsim/physics/registration/simulator/Interaction.hpp>
#include <isaacsim/physics/registration/simulator/Simulation.hpp>

#include <string>

namespace isaacsim
{
namespace physics
{
namespace manager
{

/**
 * @brief Dispatches an interaction raycast request to every active physics backend.
 *
 * @param[in] origin Ray origin in world coordinates.
 * @param[in] direction Ray direction in world coordinates.
 * @param[in] input Whether the associated input control is active, such as a pressed mouse button.
 *
 * @note Exceptions raised by a backend interaction callback propagate to the caller and prevent dispatch to later
 *       backends during that call.
 */
ISAACSIM_PHYSICS_MANAGER_API void handleRaycast(const registration::Float3& origin,
                                                const registration::Float3& direction,
                                                bool input);

/**
 * @brief Gets aggregated physics debug data for a prim.
 *
 * @param[in] primPath Absolute USD path of the prim, such as `"/World/Cube"`.
 * @return Debug entries contributed by all active physics backends, or an empty dictionary if no data is available.
 *         When multiple backends return the same key, the entry processed later replaces the earlier entry; backend
 *         iteration order is unspecified.
 *
 * @note Exceptions raised by a backend debug-data callback propagate to the caller and prevent aggregation from later
 *       backends during that call.
 */
ISAACSIM_PHYSICS_MANAGER_API registration::DebugDataDictionary getPrimDebugData(const std::string& primPath);
} // namespace manager
} // namespace physics
} // namespace isaacsim
