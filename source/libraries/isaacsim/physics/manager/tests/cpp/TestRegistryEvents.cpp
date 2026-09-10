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

#include <stdexcept>
#include <string>
#include <vector>


using namespace isaacsim::physics::registration;

//=============================================================================
// TEST: Registry Events
//=============================================================================
TEST_CASE("Registry Events")
{

    SUBCASE("Simulation registry event notifications")
    {
        struct EventRecord
        {
            SimulationRegistryEventType eventType;
            SimulationId simulationId;
            std::string simulationName;
        };

        std::vector<EventRecord> events;

        SubscriptionId subscriptionId = subscribeSimulationRegistryEvents(
            [&events](SimulationRegistryEventType eventType, const SimulationId& id, const std::string& name,
                      void* /*userData*/)
            {
                EventRecord record;
                record.eventType = eventType;
                record.simulationId = id;
                record.simulationName = name;
                events.push_back(record);
            },
            nullptr);

        REQUIRE(subscriptionId != g_kInvalidSubscriptionId);

        // Register simulation - should trigger event
        Simulation simulation;
        SimulationId simulationId = registerSimulation(simulation, "EventTest");

        REQUIRE(events.size() >= 1);
        REQUIRE(events.back().eventType == SimulationRegistryEventType::eSimulationRegistered);
        REQUIRE(events.back().simulationName == "EventTest");

        // Deactivate - should trigger event
        deactivateSimulation(simulationId);
        REQUIRE(events.back().eventType == SimulationRegistryEventType::eSimulationDeactivated);

        // Activate - should trigger event
        activateSimulation(simulationId);
        REQUIRE(events.back().eventType == SimulationRegistryEventType::eSimulationActivated);

        // Unregister - should trigger event
        unregisterSimulation(simulationId);
        REQUIRE(events.back().eventType == SimulationRegistryEventType::eSimulationUnregistered);

        unsubscribeSimulationRegistryEvents(subscriptionId);
    }

    SUBCASE("Event callback can unsubscribe itself")
    {
        size_t callbackCount = 0;
        SubscriptionId subscriptionId = g_kInvalidSubscriptionId;
        subscriptionId = subscribeSimulationRegistryEvents(
            [&callbackCount, &subscriptionId](SimulationRegistryEventType, const SimulationId&, const std::string&, void*)
            {
                ++callbackCount;
                unsubscribeSimulationRegistryEvents(subscriptionId);
            });

        REQUIRE(subscriptionId != g_kInvalidSubscriptionId);

        Simulation simulation;
        const SimulationId firstId = registerSimulation(simulation, "SelfUnsubscribeFirst");
        const SimulationId secondId = registerSimulation(simulation, "SelfUnsubscribeSecond");

        CHECK(callbackCount == 1);

        unregisterSimulation(firstId);
        unregisterSimulation(secondId);
    }

    SUBCASE("Throwing event callback does not interrupt registry operations")
    {
        const SubscriptionId throwingSubscriptionId =
            subscribeSimulationRegistryEvents([](SimulationRegistryEventType, const SimulationId&, const std::string&,
                                                 void*) { throw std::runtime_error("expected test exception"); });

        size_t observerCount = 0;
        const SubscriptionId observerSubscriptionId =
            subscribeSimulationRegistryEvents([&observerCount](SimulationRegistryEventType, const SimulationId&,
                                                               const std::string&, void*) { ++observerCount; });

        SimulationId simulationId = g_kInvalidSimulationId;
        CHECK_NOTHROW(simulationId = registerSimulation(Simulation{}, "ThrowingCallback"));
        CHECK(simulationId != g_kInvalidSimulationId);
        CHECK(observerCount == 1);

        unsubscribeSimulationRegistryEvents(throwingSubscriptionId);
        unsubscribeSimulationRegistryEvents(observerSubscriptionId);
        unregisterSimulation(simulationId);
    }
}

// TEST: getActiveSimulationId resolves by name, and refuses to choose when several carry one name
TEST_CASE("Registry: getActiveSimulationId")
{
    Simulation simulation;

    SUBCASE("matches the name exactly")
    {
        const SimulationId simulationId = registerSimulation(simulation, "ResolveByName");
        activateSimulation(simulationId);

        REQUIRE(getActiveSimulationId("ResolveByName") == simulationId);
        // Case is significant. Folding it would make two simulations that registered distinct names --
        // and hold distinct factories -- resolve to one ambiguous lookup that can answer for neither.
        REQUIRE(getActiveSimulationId("resolvebyname") == g_kInvalidSimulationId);

        unregisterSimulation(simulationId);
    }

    SUBCASE("reports no match when the only simulation with that name is inactive")
    {
        const SimulationId simulationId = registerSimulation(simulation, "InactiveByName");
        deactivateSimulation(simulationId);

        REQUIRE(getActiveSimulationId("InactiveByName") == g_kInvalidSimulationId);

        unregisterSimulation(simulationId);
    }

    SUBCASE("throws rather than choosing when several active simulations share a name")
    {
        // Activation is not exclusive, so this state is reachable; the registry is unordered, so picking one
        // would bind the caller to a different simulation between runs.
        const SimulationId firstId = registerSimulation(simulation, "AmbiguousByName");
        const SimulationId secondId = registerSimulation(simulation, "AmbiguousByName");
        activateSimulation(firstId);
        activateSimulation(secondId);

        REQUIRE_THROWS_AS(getActiveSimulationId("AmbiguousByName"), std::runtime_error);

        // Once the ambiguity is resolved the lookup answers again, and answers with the one left active.
        deactivateSimulation(firstId);
        REQUIRE(getActiveSimulationId("AmbiguousByName") == secondId);

        unregisterSimulation(firstId);
        unregisterSimulation(secondId);
    }
}
