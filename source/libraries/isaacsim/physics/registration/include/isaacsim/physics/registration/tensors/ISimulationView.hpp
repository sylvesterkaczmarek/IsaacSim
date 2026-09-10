// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <isaacsim/physics/registration/Export.h>

namespace isaacsim
{
namespace physics
{
namespace tensors
{

/**
 * @brief Polymorphic base class for engine-specific tensor simulation views.
 *
 * Simulation-view factories return this type so callers can share ownership of engine-specific views without depending
 * on an engine's concrete implementation.
 */
class ISAACSIM_PHYSICS_REGISTRATION_API ISimulationView
{
public:
    /**
     * @brief Destroys the simulation view.
     */
    virtual ~ISimulationView() = default;
};

} // namespace tensors
} // namespace physics
} // namespace isaacsim
