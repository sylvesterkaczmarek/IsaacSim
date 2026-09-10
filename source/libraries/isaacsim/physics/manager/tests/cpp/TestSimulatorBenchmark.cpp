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
// TEST: Benchmark Functions
//=============================================================================
TEST_CASE("Benchmark Functions")
{

    SUBCASE("Profile statistics subscription")
    {
        Simulation simulationDefinition;

        SubscriptionId nextId = 100;
        std::vector<ProfileStatisticsNotificationFunction> subscribers;

        simulationDefinition.benchmarkFunctions.subscribeProfileStatisticsEvents =
            [&](ProfileStatisticsNotificationFunction callback) -> SubscriptionId
        {
            subscribers.push_back(callback);
            SubscriptionId result = nextId;
            nextId = SubscriptionId(nextId.id + 1);
            return result;
        };

        simulationDefinition.benchmarkFunctions.unsubscribeProfileStatisticsEvents = [&](SubscriptionId subscriptionId)
        {
            if (subscriptionId.id >= 100 && subscriptionId.id < nextId.id)
            {
                size_t index = subscriptionId.id - 100;
                if (index < subscribers.size())
                {
                    subscribers.erase(subscribers.begin() + index);
                }
            }
        };

        SimulationId simulationId = registerSimulation(simulationDefinition, "BenchmarkTest");
        const Simulation* registeredSimulation = getSimulation(simulationId);

        // Test subscription
        bool statisticsReceived = false;
        SubscriptionId subscriptionId = registeredSimulation->benchmarkFunctions.subscribeProfileStatisticsEvents(
            [&statisticsReceived](const std::vector<PhysicsProfileStatistics>& /*statistics*/)
            { statisticsReceived = true; });

        REQUIRE(subscriptionId == 100);
        REQUIRE(subscribers.size() == 1);

        // Simulate statistics callback
        std::vector<PhysicsProfileStatistics> testStatistics;
        PhysicsProfileStatistics statistic;
        statistic.zoneName = "TestZone";
        statistic.elapsedMilliseconds = 1.5f;
        testStatistics.push_back(statistic);

        subscribers[0](testStatistics);
        REQUIRE(statisticsReceived == true);

        // Test unsubscription
        registeredSimulation->benchmarkFunctions.unsubscribeProfileStatisticsEvents(subscriptionId);
        REQUIRE(subscribers.size() == 0);

        unregisterSimulation(simulationId);
    }
}
