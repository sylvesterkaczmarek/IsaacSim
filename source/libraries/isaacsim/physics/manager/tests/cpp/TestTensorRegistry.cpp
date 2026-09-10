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

#include <doctest/doctest.h>
#include <isaacsim/physics/manager/tensors/EntityView.hpp>
#include <isaacsim/physics/manager/tensors/SimulationView.hpp>
#include <isaacsim/physics/registration/tensors/TensorRegistry.hpp>

#include <memory>
#include <stdexcept>
#include <string>

using namespace isaacsim::physics::tensors;

namespace
{

// Minimal subclass — the registry tests don't exercise the impl callbacks.
class StubEntityView : public EntityView
{
public:
    explicit StubEntityView(std::vector<std::string> paths) : EntityView(std::move(paths))
    {
    }
};

class StubSimulationView : public SimulationView
{
public:
    StubSimulationView(const std::string& engine, const std::string& frontend, int64_t stageId)
        : SimulationView(engine, frontend, stageId)
    {
    }
};

EntityFactory makeStubEntityFactory()
{
    return [](const std::vector<std::string>& paths) -> std::shared_ptr<EntityView>
    { return std::make_shared<StubEntityView>(paths); };
}

SimulationViewFactory makeStubSimulationViewFactory(const std::string& engine)
{
    return [engine](const std::string& frontend, int64_t stageId) -> std::shared_ptr<SimulationView>
    { return std::make_shared<StubSimulationView>(engine, frontend, stageId); };
}

} // namespace

//=============================================================================
// TEST: getInstance is a singleton
//=============================================================================
TEST_CASE("TensorRegistry::getInstance is a singleton")
{
    TensorRegistry& firstRegistry = TensorRegistry::getInstance();
    TensorRegistry& secondRegistry = TensorRegistry::getInstance();
    REQUIRE(&firstRegistry == &secondRegistry);
}

//=============================================================================
// TEST: registerEntity / hasEntity / createEntity round-trip
//=============================================================================
TEST_CASE("TensorRegistry: entity registration round-trip")
{
    TensorRegistry& registry = TensorRegistry::getInstance();
    registry.clearForTesting();

    SUBCASE("Initial state — registry is empty")
    {
        REQUIRE_FALSE(registry.hasEntity("test", "articulation"));
        REQUIRE(registry.listEngines().empty());
    }

    SUBCASE("registerEntity makes the (engine, entity) addressable")
    {
        const bool first = registry.registerEntity("test", "articulation", makeStubEntityFactory());
        REQUIRE(first == true);
        REQUIRE(registry.hasEntity("test", "articulation"));
    }

    SUBCASE("createEntity dispatches to the registered factory")
    {
        registry.registerEntity("test", "articulation", makeStubEntityFactory());

        std::vector<std::string> paths{ "/World/robot_0", "/World/robot_1" };
        auto view = std::static_pointer_cast<EntityView>(registry.createEntity("test", "articulation", paths));
        REQUIRE(view.get() != nullptr);
        REQUIRE(view->getPaths().size() == 2);
        REQUIRE(view->getPaths()[0] == "/World/robot_0");
        REQUIRE(view->getPaths()[1] == "/World/robot_1");
    }

    SUBCASE("createEntity with missing engine throws std::out_of_range")
    {
        REQUIRE_THROWS_AS(registry.createEntity("nonexistent", "articulation", {}), std::out_of_range);
    }

    SUBCASE("createEntity with missing entity-name throws std::out_of_range")
    {
        registry.registerEntity("test", "articulation", makeStubEntityFactory());
        REQUIRE_THROWS_AS(registry.createEntity("test", "rigid-body", {}), std::out_of_range);
    }

    SUBCASE("re-registering the same (engine, entity) replaces the factory")
    {
        bool first = registry.registerEntity("test", "articulation", makeStubEntityFactory());
        REQUIRE(first == true); // `true` = first-time registration
        // Re-register with a DIFFERENT factory marker — the new one should
        // shadow the old. Use a factory that produces a distinguishable
        // view (zero paths regardless of input).
        EntityFactory replacement = [](const std::vector<std::string>& /*paths*/)
        { return std::make_shared<StubEntityView>(std::vector<std::string>{}); };
        bool second = registry.registerEntity("test", "articulation", replacement);
        REQUIRE(second == false); // `false` = replaced an existing entry

        std::vector<std::string> paths{ "/World/robot_0" };
        auto view = std::static_pointer_cast<EntityView>(registry.createEntity("test", "articulation", paths));
        REQUIRE(view.get() != nullptr);
        REQUIRE(view->getPaths().empty()); // replacement factory ignores input
    }
}

//=============================================================================
// TEST: registerSimulationView / createSimulationView
//=============================================================================
TEST_CASE("TensorRegistry: SimulationView registration round-trip")
{
    TensorRegistry& registry = TensorRegistry::getInstance();
    registry.clearForTesting();

    SUBCASE("createSimulationView with no factory throws std::out_of_range")
    {
        REQUIRE_THROWS_AS(registry.createSimulationView("test", "warp", /*stageId=*/0), std::out_of_range);
    }

    SUBCASE("registered factory is invoked with frontend and stage arguments")
    {
        registry.registerSimulationView("test", makeStubSimulationViewFactory("test"));

        auto simulation =
            std::static_pointer_cast<SimulationView>(registry.createSimulationView("test", "warp", /*stageId=*/12345));
        REQUIRE(simulation.get() != nullptr);
        REQUIRE(simulation->getEngine() == "test");
        REQUIRE(simulation->getFrontendName() == "warp");
        REQUIRE(simulation->getStageId() == 12345);
    }
}

//=============================================================================
// TEST: per-engine isolation — registrations don't leak across engine names
//=============================================================================
TEST_CASE("TensorRegistry: per-engine isolation")
{
    TensorRegistry& registry = TensorRegistry::getInstance();
    registry.clearForTesting();

    registry.registerEntity("ovphysx", "articulation", makeStubEntityFactory());
    registry.registerEntity("newton", "articulation", makeStubEntityFactory());
    registry.registerEntity("ovphysx", "rigid-body", makeStubEntityFactory());

    REQUIRE(registry.hasEntity("ovphysx", "articulation"));
    REQUIRE(registry.hasEntity("ovphysx", "rigid-body"));
    REQUIRE(registry.hasEntity("newton", "articulation"));
    REQUIRE_FALSE(registry.hasEntity("newton", "rigid-body"));

    // Registering factories makes a simulation name addressable; it does not declare an engine.
    auto engines = registry.listSimulations();
    REQUIRE(engines.size() == 2);
    bool foundOvphysx = false;
    bool foundNewton = false;
    for (const auto& engine : engines)
    {
        if (engine == "ovphysx")
        {
            foundOvphysx = true;
        }
        if (engine == "newton")
        {
            foundNewton = true;
        }
    }
    REQUIRE(foundOvphysx);
    REQUIRE(foundNewton);
}

//=============================================================================
// TEST: listEntities reports exactly what was registered for a given engine
//=============================================================================
TEST_CASE("TensorRegistry: listEntities")
{
    TensorRegistry& registry = TensorRegistry::getInstance();
    registry.clearForTesting();

    REQUIRE(registry.listEntities("ovphysx").empty());

    registry.registerEntity("ovphysx", "articulation", makeStubEntityFactory());
    registry.registerEntity("ovphysx", "rigid-body", makeStubEntityFactory());
    registry.registerEntity("ovphysx", "rigid-contact", makeStubEntityFactory());

    auto entities = registry.listEntities("ovphysx");
    REQUIRE(entities.size() == 3);

    bool foundArticulation = false;
    bool foundRigidBody = false;
    bool foundRigidContact = false;
    for (const auto& entityName : entities)
    {
        if (entityName == "articulation")
        {
            foundArticulation = true;
        }
        if (entityName == "rigid-body")
        {
            foundRigidBody = true;
        }
        if (entityName == "rigid-contact")
        {
            foundRigidContact = true;
        }
    }
    REQUIRE(foundArticulation);
    REQUIRE(foundRigidBody);
    REQUIRE(foundRigidContact);

    // A different engine has its own entity list.
    REQUIRE(registry.listEntities("newton").empty());
    // A non-registered engine returns an empty list, not an error.
    REQUIRE(registry.listEntities("nonexistent").empty());
}

//=============================================================================
// TEST: unregisterEntity / unregisterSimulationView
//=============================================================================
TEST_CASE("TensorRegistry: unregister")
{
    TensorRegistry& registry = TensorRegistry::getInstance();
    registry.clearForTesting();

    SUBCASE("unregisterEntity removes the (engine, entity) pair")
    {
        registry.registerEntity("test", "articulation", makeStubEntityFactory());
        REQUIRE(registry.hasEntity("test", "articulation"));

        bool removed = registry.unregisterEntity("test", "articulation");
        REQUIRE(removed == true);
        REQUIRE_FALSE(registry.hasEntity("test", "articulation"));
    }

    SUBCASE("unregisterEntity returns false for a missing entry")
    {
        bool removed = registry.unregisterEntity("nope", "articulation");
        REQUIRE(removed == false);
    }

    SUBCASE("unregisterSimulationView removes the engine factory")
    {
        registry.registerSimulationView("test", makeStubSimulationViewFactory("test"));
        REQUIRE(registry.createSimulationView("test", "warp", 0).get() != nullptr);

        bool removed = registry.unregisterSimulationView("test");
        REQUIRE(removed == true);
        REQUIRE_THROWS_AS(registry.createSimulationView("test", "warp", 0), std::out_of_range);
    }
}

//=============================================================================
// TEST: clearForTesting resets the registry
//=============================================================================
TEST_CASE("TensorRegistry::clearForTesting wipes all state")
{
    TensorRegistry& registry = TensorRegistry::getInstance();

    registry.registerEntity("test", "articulation", makeStubEntityFactory());
    registry.registerSimulationView("test", makeStubSimulationViewFactory("test"));
    REQUIRE(registry.hasEntity("test", "articulation"));
    REQUIRE(registry.createSimulationView("test", "warp", 0).get() != nullptr);

    registry.clearForTesting();

    REQUIRE_FALSE(registry.hasEntity("test", "articulation"));
    REQUIRE_THROWS_AS(registry.createSimulationView("test", "warp", 0), std::out_of_range);
    REQUIRE(registry.listEngines().empty());
}
