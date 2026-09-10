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

#include <isaacsim/ovsim/clients/local/Export.h>
#include <ovsim/interfaces/control/simulation/Simulation.hpp>

namespace isaacsim
{
namespace ovsim
{
namespace clients
{
namespace local
{
namespace control
{
namespace simulation
{

using ::ovsim::interfaces::control::simulation::InputParameterType;
using ::ovsim::interfaces::control::simulation::OutputParameterType;

/** @copydoc isaacsim::physics::ovsim::control::simulation::play */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API void play();

/** @copydoc isaacsim::physics::ovsim::control::simulation::pause */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API void pause();

/** @copydoc isaacsim::physics::ovsim::control::simulation::stop */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API void stop();

/** @copydoc isaacsim::physics::ovsim::control::simulation::initialize */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API void initialize();

/** @copydoc isaacsim::physics::ovsim::control::simulation::step */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API void step();

/** @copydoc isaacsim::physics::ovsim::control::simulation::invalidate */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API void invalidate();

/** @copydoc isaacsim::physics::ovsim::control::simulation::setParameter */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API void setParameter(const std::string& provider,
                                                   const std::string& parameterName,
                                                   const InputParameterType& value);

/** @copydoc isaacsim::physics::ovsim::control::simulation::getParameter */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API OutputParameterType getParameter(const std::string& provider,
                                                                  const std::string& parameterName);

} // namespace simulation
} // namespace control
} // namespace local
} // namespace clients
} // namespace ovsim
} // namespace isaacsim
