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

#include "../Types.hpp"
#include "Benchmark.hpp"
#include "Interaction.hpp"
#include "SceneQuery.hpp"
#include "Simulation.hpp"

namespace isaacsim
{
namespace physics
{
namespace registration
{

/**
 * @brief Aggregates the function tables implemented by a physics simulation backend.
 */
struct Simulation
{
    /** @brief The physics profiling function table. */
    BenchmarkFunctions benchmarkFunctions;

    /** @brief The interactive debugging function table. */
    InteractionFunctions interactionFunctions;

    /** @brief The scene-query function table. */
    SceneQueryFunctions sceneQueryFunctions;

    /** @brief The simulation lifecycle and stepping function table. */
    SimulationFunctions simulationFunctions;
};

} // namespace registration
} // namespace physics
} // namespace isaacsim
