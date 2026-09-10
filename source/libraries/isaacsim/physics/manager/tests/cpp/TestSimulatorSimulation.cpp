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
// TEST: Simulation Functions
//=============================================================================
TEST_CASE("Simulation Functions")
{

    SUBCASE("Initialize and Close")
    {
        Simulation simulationDefinition;

        long attachedStageId = 0;
        simulationDefinition.simulationFunctions.initialize = [&attachedStageId](
                                                                  void* ovstage, const char* /*usdIdentifier*/) -> bool
        {
            attachedStageId = reinterpret_cast<long>(ovstage);
            return true;
        };

        simulationDefinition.simulationFunctions.close = [&attachedStageId]()
        {
            attachedStageId = 0;
            return true;
        };

        simulationDefinition.simulationFunctions.getAttachedStage = [&attachedStageId]() -> long
        { return attachedStageId; };

        SimulationId simulationId = registerSimulation(simulationDefinition, "StageTest");

        const Simulation* registeredSimulation = getSimulation(simulationId);
        REQUIRE(registeredSimulation);

        // Test initialize
        bool initialized =
            registeredSimulation->simulationFunctions.initialize(reinterpret_cast<void*>(12345), "test-stage");
        REQUIRE(initialized == true);
        REQUIRE(attachedStageId == 12345);

        long retrieved = registeredSimulation->simulationFunctions.getAttachedStage();
        REQUIRE(retrieved == 12345);

        // Test close
        registeredSimulation->simulationFunctions.close();
        REQUIRE(attachedStageId == 0);

        unregisterSimulation(simulationId);
    }

    SUBCASE("SimulateAsynchronously and FetchResults")
    {
        Simulation simulationDefinition;

        float lastTimeStep = 0.0f;
        bool resultsFetched = false;

        simulationDefinition.simulationFunctions.simulateAsynchronously =
            [&lastTimeStep](float timeStep, float /*simulationTime*/) { lastTimeStep = timeStep; };

        simulationDefinition.simulationFunctions.fetchResults = [&resultsFetched]() { resultsFetched = true; };

        simulationDefinition.simulationFunctions.checkResults = []() -> bool { return true; };

        SimulationId simulationId = registerSimulation(simulationDefinition, "SimulateTest");
        const Simulation* registeredSimulation = getSimulation(simulationId);

        // Test simulateAsynchronously
        registeredSimulation->simulationFunctions.simulateAsynchronously(1.0f / 60.0f, 0.0f);
        REQUIRE(lastTimeStep == 1.0f / 60.0f);

        // Test checkResults
        bool ready = registeredSimulation->simulationFunctions.checkResults();
        REQUIRE(ready == true);

        // Test fetchResults
        registeredSimulation->simulationFunctions.fetchResults();
        REQUIRE(resultsFetched == true);

        unregisterSimulation(simulationId);
    }
}
