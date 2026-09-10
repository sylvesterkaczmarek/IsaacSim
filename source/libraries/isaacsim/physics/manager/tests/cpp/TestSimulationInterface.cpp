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

#include <doctest/doctest.h>
#include <isaacsim/physics/registration/Physics.hpp>


using namespace isaacsim::physics::registration;

//=============================================================================
// TEST: Simulation Interface
//=============================================================================
TEST_CASE("Simulation Interface")
{

    SUBCASE("Basic simulation structure")
    {
        Simulation simulation;

        // Test that all function pointers start as nullptr
        REQUIRE(simulation.simulationFunctions.initialize == nullptr);
        REQUIRE(simulation.simulationFunctions.simulateAsynchronously == nullptr);
        REQUIRE(simulation.simulationFunctions.simulate == nullptr);
        REQUIRE(simulation.sceneQueryFunctions.raycastClosest == nullptr);
        REQUIRE(simulation.interactionFunctions.handleRaycast == nullptr);
        REQUIRE(simulation.benchmarkFunctions.subscribeProfileStatisticsEvents == nullptr);
    }

    SUBCASE("Attach simulation functions")
    {
        Simulation simulation;

        // Attach a simple simulate function
        bool simulateCalled = false;
        simulation.simulationFunctions.simulate = [&simulateCalled](float /*timeStep*/, float /*t*/)
        { simulateCalled = true; };

        // Register and verify function is attached
        SimulationId simulationId = registerSimulation(simulation, "FunctionTest");
        REQUIRE(simulationId != g_kInvalidSimulationId);

        const Simulation* retrieved = getSimulation(simulationId);
        REQUIRE(retrieved != nullptr);
        REQUIRE(retrieved->simulationFunctions.simulate != nullptr);

        // Call the function
        retrieved->simulationFunctions.simulate(1.0f / 60.0f, 0.0f);
        REQUIRE(simulateCalled == true);

        unregisterSimulation(simulationId);
    }
}
