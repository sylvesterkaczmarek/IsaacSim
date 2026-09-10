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

#include <memory>
#include <string>

using namespace isaacsim::physics::tensors;

namespace
{

// Concrete subclass that overrides the factories with stub impls so we
// can exercise the dispatch surface that the umbrella's Python adapter
// relies on.
class OverridingSimulationView : public SimulationView
{
public:
    OverridingSimulationView(const std::string& engine, const std::string& frontend, int64_t stageId)
        : SimulationView(engine, frontend, stageId),
          gravity{ 0.0f, 0.0f, -9.81f },
          stepCount(0),
          clearCount(0),
          kinematicCount(0)
    {
    }

    void setGravity(Float3 gravityVector) override
    {
        gravity = gravityVector;
    }
    Float3 getGravity() const override
    {
        return gravity;
    }
    void clearForces() override
    {
        clearCount++;
    }
    void step(float timeStep) override
    {
        lastTimeStep = timeStep;
        stepCount++;
    }
    void updateArticulationsKinematic() override
    {
        kinematicCount++;
    }

    std::shared_ptr<EntityView> createArticulationView(const std::string& pattern) override
    {
        lastArticulationPattern = pattern;
        auto view = std::make_shared<EntityView>(std::vector<std::string>{ pattern });
        view->setCount(8);
        return view;
    }
    std::shared_ptr<EntityView> createRigidBodyView(const std::string& pattern) override
    {
        lastRigidBodyPattern = pattern;
        auto view = std::make_shared<EntityView>(std::vector<std::string>{ pattern });
        view->setCount(16);
        return view;
    }
    std::shared_ptr<EntityView> createRigidContactView(const std::string& pattern,
                                                       const std::vector<std::string>& filterPatterns,
                                                       int maximumContactDataCount) override
    {
        lastContactPattern = pattern;
        lastContactFilters = filterPatterns;
        lastContactDataCount = maximumContactDataCount;
        return std::make_shared<EntityView>(std::vector<std::string>{ pattern });
    }

    Float3 gravity;
    int stepCount;
    int clearCount;
    int kinematicCount;
    float lastTimeStep{ 0.0f };
    std::string lastArticulationPattern;
    std::string lastRigidBodyPattern;
    std::string lastContactPattern;
    std::vector<std::string> lastContactFilters;
    int lastContactDataCount{ 0 };
};

} // namespace

//=============================================================================
// TEST: identity members are stored
//=============================================================================
TEST_CASE("SimulationView: engine / frontend / stage_id stored at construction")
{
    OverridingSimulationView simulation("ovphysx", "warp", 42);
    REQUIRE(simulation.getEngine() == "ovphysx");
    REQUIRE(simulation.getFrontendName() == "warp");
    REQUIRE(simulation.getStageId() == 42);
}

//=============================================================================
// TEST: device-ordinal accessors
//=============================================================================
TEST_CASE("SimulationView: device ordinal accessors")
{
    OverridingSimulationView simulation("ovphysx", "warp", 0);

    SUBCASE("Default device ordinal is -1 (CPU)")
    {
        REQUIRE(simulation.getDeviceOrdinal() == -1);
        REQUIRE(simulation.getParameterDeviceOrdinal() == -1);
    }

    SUBCASE("setDeviceOrdinal updates the runtime device")
    {
        simulation.setDeviceOrdinal(0); // cuda:0
        REQUIRE(simulation.getDeviceOrdinal() == 0);
    }

    SUBCASE("setParameterDeviceOrdinal is independent of runtime ordinal")
    {
        simulation.setDeviceOrdinal(0);
        simulation.setParameterDeviceOrdinal(-1);
        REQUIRE(simulation.getDeviceOrdinal() == 0);
        REQUIRE(simulation.getParameterDeviceOrdinal() == -1);
    }
}

//=============================================================================
// TEST: validity flag
//=============================================================================
TEST_CASE("SimulationView: validity flag")
{
    OverridingSimulationView simulation("ovphysx", "warp", 0);

    SUBCASE("Default — view starts valid")
    {
        // The base class default is `m_valid{true}` — engines flip it
        // off via `invalidate()` when the underlying physics state goes
        // away (stage detach, scene reset).
        REQUIRE(simulation.isValid() == true);
    }

    SUBCASE("invalidate flips the flag, setValid restores it")
    {
        REQUIRE(simulation.isValid() == true);
        simulation.invalidate();
        REQUIRE(simulation.isValid() == false);
        simulation.setValid(true);
        REQUIRE(simulation.isValid() == true);
    }
}

//=============================================================================
// TEST: gravity round-trip
//=============================================================================
TEST_CASE("SimulationView: gravity round-trip")
{
    OverridingSimulationView simulation("ovphysx", "warp", 0);

    Float3 gravityVector = simulation.getGravity();
    REQUIRE(gravityVector.z == doctest::Approx(-9.81f));

    simulation.setGravity(Float3{ 1.0f, 2.0f, 3.0f });
    gravityVector = simulation.getGravity();
    REQUIRE(gravityVector.x == doctest::Approx(1.0f));
    REQUIRE(gravityVector.y == doctest::Approx(2.0f));
    REQUIRE(gravityVector.z == doctest::Approx(3.0f));
}

//=============================================================================
// TEST: step / clearForces / updateArticulationsKinematic dispatch through
// to the override.
//=============================================================================
TEST_CASE("SimulationView: scene-control overrides")
{
    OverridingSimulationView simulation("ovphysx", "warp", 0);

    SUBCASE("step forwards the timeStep argument")
    {
        simulation.step(1.0f / 60.0f);
        REQUIRE(simulation.stepCount == 1);
        REQUIRE(simulation.lastTimeStep == doctest::Approx(1.0f / 60.0f));

        simulation.step(1.0f / 120.0f);
        REQUIRE(simulation.stepCount == 2);
        REQUIRE(simulation.lastTimeStep == doctest::Approx(1.0f / 120.0f));
    }

    SUBCASE("clearForces increments the override counter")
    {
        simulation.clearForces();
        simulation.clearForces();
        REQUIRE(simulation.clearCount == 2);
    }

    SUBCASE("updateArticulationsKinematic increments the override counter")
    {
        simulation.updateArticulationsKinematic();
        REQUIRE(simulation.kinematicCount == 1);
    }
}

//=============================================================================
// TEST: view factory dispatch
//=============================================================================
TEST_CASE("SimulationView: view factories")
{
    OverridingSimulationView simulation("ovphysx", "warp", 0);

    SUBCASE("createArticulationView forwards the pattern")
    {
        auto view = simulation.createArticulationView("/World/envs/*/robot");
        REQUIRE(view.get() != nullptr);
        REQUIRE(simulation.lastArticulationPattern == "/World/envs/*/robot");
        REQUIRE(view->getCount() == 8);
    }

    SUBCASE("createRigidBodyView forwards the pattern")
    {
        auto view = simulation.createRigidBodyView("/World/balls/*");
        REQUIRE(view.get() != nullptr);
        REQUIRE(simulation.lastRigidBodyPattern == "/World/balls/*");
        REQUIRE(view->getCount() == 16);
    }

    SUBCASE("createRigidContactView forwards pattern + filters + max_data")
    {
        std::vector<std::string> filters{ "/World/groundPlane", "/World/wall" };
        auto view = simulation.createRigidContactView("/World/sensor", filters, 1024);
        REQUIRE(view.get() != nullptr);
        REQUIRE(simulation.lastContactPattern == "/World/sensor");
        REQUIRE(simulation.lastContactFilters.size() == 2);
        REQUIRE(simulation.lastContactFilters[0] == "/World/groundPlane");
        REQUIRE(simulation.lastContactFilters[1] == "/World/wall");
        REQUIRE(simulation.lastContactDataCount == 1024);
    }
}

//=============================================================================
// TEST: base-class default factories return nullptr
//
// Engines must override the factories they support; un-overridden ones
// return nullptr, which the umbrella adapter surfaces as "view type not
// available for this engine". This keeps the API surface uniform without
// engines having to register placeholder failures.
//=============================================================================
TEST_CASE("SimulationView: un-overridden factories return nullptr")
{
    SimulationView simulation("test", "warp", 0);

    REQUIRE(simulation.createArticulationView("/foo").get() == nullptr);
    REQUIRE(simulation.createRigidBodyView("/foo").get() == nullptr);
    REQUIRE(simulation.createVolumeDeformableBodyView("/foo").get() == nullptr);
    REQUIRE(simulation.createSurfaceDeformableBodyView("/foo").get() == nullptr);
    REQUIRE(simulation.createDeformableMaterialView("/foo").get() == nullptr);
    REQUIRE(simulation.createRigidContactView("/foo", {}, 0).get() == nullptr);
    REQUIRE(simulation.createSdfShapeView("/foo", 0).get() == nullptr);
}
