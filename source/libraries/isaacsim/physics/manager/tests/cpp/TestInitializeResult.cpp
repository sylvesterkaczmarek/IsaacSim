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
#include <isaacsim/physics/manager/PhysicsSimulation.hpp>
#include <isaacsim/physics/registration/Physics.hpp>

#include <stdexcept>
#include <string>
#include <vector>

using namespace isaacsim::physics::manager;
using namespace isaacsim::physics::registration;

namespace
{

// A backend whose initialize()/close()/hasAttachedStage() results the test controls.
struct MockBackend
{
    std::string name;
    std::vector<std::string>* order = nullptr;
    bool initializeResult = true;
    bool closeResult = true;
    bool hasStage = false;
    bool hasStageThrows = false;
    long stageId = 0;
};

Simulation makeSimulation(MockBackend& backend)
{
    Simulation simulation;
    simulation.simulationFunctions.initialize = [&backend](void*, const char*) -> bool
    {
        if (backend.order)
        {
            backend.order->push_back(backend.name);
        }
        return backend.initializeResult;
    };
    simulation.simulationFunctions.close = [&backend]() -> bool { return backend.closeResult; };
    simulation.simulationFunctions.getAttachedStage = [&backend]() -> long { return backend.stageId; };
    simulation.simulationFunctions.hasAttachedStage = [&backend]() -> bool
    {
        if (backend.hasStageThrows)
        {
            throw std::runtime_error("has_attached_stage boom from " + backend.name);
        }
        return backend.hasStage;
    };
    return simulation;
}

} // namespace

TEST_CASE("initialize: clean single-backend failure returns eFailed")
{
    MockBackend backend{ "clean" };
    backend.initializeResult = false;
    backend.hasStage = false; // nothing left attached
    Simulation engine = makeSimulation(backend);
    SimulationId id = registerSimulation(engine, "CleanFail");

    CHECK(initialize(nullptr, "0") == InitializeResult::eFailed);

    unregisterSimulation(id);
}

TEST_CASE("initialize: a retained stage with a zero StageCache id returns eFailedDirty")
{

    MockBackend backend{ "retained" };
    backend.initializeResult = false;
    backend.hasStage = true; // a stage is still attached ...
    backend.stageId = 0; // ... but its StageCache id is legally 0
    Simulation engine = makeSimulation(backend);
    SimulationId id = registerSimulation(engine, "RetainedZeroId");

    CHECK(initialize(nullptr, "") == InitializeResult::eFailedDirty);

    unregisterSimulation(id);
}

TEST_CASE("initialize: a throwing state query returns eFailedDirty and does not propagate")
{

    MockBackend backend{ "thrower" };
    backend.initializeResult = false;
    backend.hasStageThrows = true;
    Simulation engine = makeSimulation(backend);
    SimulationId id = registerSimulation(engine, "ThrowingQuery");

    InitializeResult result = InitializeResult::eOk;
    CHECK_NOTHROW(result = initialize(nullptr, "0"));
    CHECK(result == InitializeResult::eFailedDirty);

    unregisterSimulation(id);
}

TEST_CASE("initialize: a dirty rollback returns eFailedDirty")
{

    // The registry iterates unordered, so learn the order with a probe init (all succeed)
    // before forcing a failure that iterates after a succeeded backend.
    std::vector<std::string> order;
    std::vector<MockBackend> mocks;
    mocks.reserve(4);
    for (int i = 0; i < 4; ++i)
    {
        mocks.push_back(MockBackend{ "b" + std::to_string(i), &order });
    }
    std::vector<SimulationId> ids;
    for (auto& backend : mocks)
    {
        Simulation engine = makeSimulation(backend);
        ids.push_back(registerSimulation(engine, backend.name));
    }

    REQUIRE(initialize(nullptr, "0") == InitializeResult::eOk);
    close();
    REQUIRE(order.size() >= 2);

    auto find = [&](const std::string& name) -> MockBackend&
    {
        for (auto& backend : mocks)
        {
            if (backend.name == name)
            {
                return backend;
            }
        }
        REQUIRE(false);
        return mocks[0];
    };
    // First-iterated backend succeeds but its rollback close fails (dirty); the
    // second-iterated backend fails init, forcing the rollback of the first.
    find(order[0]).closeResult = false;
    find(order[1]).initializeResult = false;

    CHECK(initialize(nullptr, "0") == InitializeResult::eFailedDirty);

    for (SimulationId id : ids)
    {
        unregisterSimulation(id);
    }
}
