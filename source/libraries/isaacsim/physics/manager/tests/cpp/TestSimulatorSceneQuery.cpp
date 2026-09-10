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

#include <vector>


using namespace isaacsim::physics::registration;

//=============================================================================
// TEST: Scene Query Functions
//=============================================================================
TEST_CASE("Scene Query Functions")
{

    SUBCASE("Raycast functions")
    {
        Simulation simulationDefinition;

        bool raycastCalled = false;
        simulationDefinition.sceneQueryFunctions.raycastClosest =
            [&raycastCalled](const isaacsim::physics::registration::Float3& /*origin*/,
                             const isaacsim::physics::registration::Float3& /*direction*/, float /*distance*/,
                             RaycastHit& hit, bool /*bothSides*/) -> bool
        {
            raycastCalled = true;
            hit.distance = 10.0f;
            hit.position = isaacsim::physics::registration::Float3{ 5.0f, 0.0f, 0.0f };
            return true;
        };

        simulationDefinition.sceneQueryFunctions.raycastAny =
            [](const isaacsim::physics::registration::Float3& /*origin*/,
               const isaacsim::physics::registration::Float3& /*direction*/, float /*distance*/,
               bool /*bothSides*/) -> bool { return true; };

        SimulationId simulationId = registerSimulation(simulationDefinition, "RaycastTest");
        const Simulation* registeredSimulation = getSimulation(simulationId);

        // Test raycastClosest
        RaycastHit hit;
        isaacsim::physics::registration::Float3 origin{ 0.0f, 0.0f, 0.0f };
        isaacsim::physics::registration::Float3 direction{ 1.0f, 0.0f, 0.0f };
        bool result = registeredSimulation->sceneQueryFunctions.raycastClosest(origin, direction, 100.0f, hit, false);

        REQUIRE(raycastCalled == true);
        REQUIRE(result == true);
        REQUIRE(hit.distance == 10.0f);
        REQUIRE(hit.position.x == 5.0f);

        // Test raycastAny
        result = registeredSimulation->sceneQueryFunctions.raycastAny(origin, direction, 100.0f, false);
        REQUIRE(result == true);

        unregisterSimulation(simulationId);
    }

    SUBCASE("Overlap functions")
    {
        Simulation simulationDefinition;

        simulationDefinition.sceneQueryFunctions.overlapSphere =
            [](float /*radius*/, const isaacsim::physics::registration::Float3& /*position*/,
               OverlapHitReportFunction reportFunction) -> uint32_t
        {
            // Simulate 2 overlaps
            OverlapHit hit1;
            hit1.collision = 123;
            reportFunction(hit1);

            OverlapHit hit2;
            hit2.collision = 456;
            reportFunction(hit2);

            return 2;
        };

        simulationDefinition.sceneQueryFunctions.overlapSphereAny =
            [](float /*radius*/, const isaacsim::physics::registration::Float3& /*position*/) -> bool { return true; };

        SimulationId simulationId = registerSimulation(simulationDefinition, "OverlapTest");
        const Simulation* registeredSimulation = getSimulation(simulationId);

        // Test overlapSphere
        std::vector<OverlapHit> hits;
        isaacsim::physics::registration::Float3 zero{ 0.0f, 0.0f, 0.0f };
        uint32_t count = registeredSimulation->sceneQueryFunctions.overlapSphere(5.0f, zero,
                                                                                 [&hits](const OverlapHit& hit)
                                                                                 {
                                                                                     hits.push_back(hit);
                                                                                     return true;
                                                                                 });

        REQUIRE(count == 2);
        REQUIRE(hits.size() == 2);
        REQUIRE(hits[0].collision == 123);
        REQUIRE(hits[1].collision == 456);

        // Test overlapSphereAny
        bool hasOverlap = registeredSimulation->sceneQueryFunctions.overlapSphereAny(5.0f, zero);
        REQUIRE(hasOverlap == true);

        unregisterSimulation(simulationId);
    }
}
