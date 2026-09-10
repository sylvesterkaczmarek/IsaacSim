// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <isaacsim/physics/ovsim/Export.h>
#include <ovsim/interfaces/control/simulation/Simulation.hpp>

namespace isaacsim
{
namespace physics
{
namespace ovsim
{
namespace control
{
namespace simulation
{

using ::ovsim::interfaces::control::simulation::InputParameterType;
using ::ovsim::interfaces::control::simulation::OutputParameterType;

/** @brief Start or resume automatic simulation stepping. */
ISAACSIM_PHYSICS_OVSIM_API void play();

/** @brief Pause automatic simulation stepping. */
ISAACSIM_PHYSICS_OVSIM_API void pause();

/** @brief Stop automatic simulation stepping. */
ISAACSIM_PHYSICS_OVSIM_API void stop();

/** @brief Initialize the simulation for manual stepping. */
ISAACSIM_PHYSICS_OVSIM_API void initialize();

/** @brief Invalidate the manually stepped simulation state. */
ISAACSIM_PHYSICS_OVSIM_API void invalidate();

/** @brief Advance a manually controlled simulation by one step. */
ISAACSIM_PHYSICS_OVSIM_API void step();

/** @brief Set a simulation parameter exposed by a provider.
 * @param[in] provider Name of the parameter provider.
 * @param[in] parameterName Name of the parameter to set.
 * @param[in] value Value to assign to the parameter.
 */
ISAACSIM_PHYSICS_OVSIM_API void setParameter(const std::string& provider,
                                             const std::string& parameterName,
                                             const InputParameterType& value);

/** @brief Get a simulation parameter exposed by a provider.
 * @param[in] provider Name of the parameter provider.
 * @param[in] parameterName Name of the parameter to retrieve.
 * @return Current parameter value.
 */
ISAACSIM_PHYSICS_OVSIM_API OutputParameterType getParameter(const std::string& provider,
                                                            const std::string& parameterName);

} // namespace simulation
} // namespace control
} // namespace ovsim
} // namespace physics
} // namespace isaacsim
