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

#include <isaacsim/foundation/ovsim/Export.h>
#include <ovsim/interfaces/control/simulation/Simulation.hpp>

namespace isaacsim
{
namespace foundation
{
namespace ovsim
{
namespace control
{
namespace simulation
{

using ::ovsim::interfaces::control::simulation::InputParameterType;
using ::ovsim::interfaces::control::simulation::OutputParameterType;

ISAACSIM_FOUNDATION_OVSIM_API void play();
ISAACSIM_FOUNDATION_OVSIM_API void pause();
ISAACSIM_FOUNDATION_OVSIM_API void stop();
ISAACSIM_FOUNDATION_OVSIM_API void step();

ISAACSIM_FOUNDATION_OVSIM_API void initialize();
ISAACSIM_FOUNDATION_OVSIM_API void invalidate();
ISAACSIM_FOUNDATION_OVSIM_API void step();

ISAACSIM_FOUNDATION_OVSIM_API void setParameter(const std::string& provider,
                                                const std::string& parameterName,
                                                const InputParameterType& value);
ISAACSIM_FOUNDATION_OVSIM_API OutputParameterType getParameter(const std::string& provider,
                                                               const std::string& parameterName);

} // namespace simulation
} // namespace control
} // namespace ovsim
} // namespace foundation
} // namespace isaacsim
