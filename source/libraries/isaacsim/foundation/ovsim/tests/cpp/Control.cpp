// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <isaacsim/common/exceptions/Exceptions.hpp>
#include <isaacsim/foundation/ovsim/control/authoring/Authoring.hpp>
#include <isaacsim/foundation/ovsim/control/simulation/Simulation.hpp>

#include <IsaacSimTest.hpp>
#include <cstdint>
#include <filesystem>
#include <stdexcept>

using namespace isaacsim::foundation::ovsim::control;

// Stage operations.

TEST_SUITE("authoring::stage")
{

    TEST_CASE("createStage")
    {
        REQUIRE_UNARY(authoring::createStage());
        CHECK_NE(std::get<int64_t>(authoring::getParameter("stage", "openusd-stage-id")), -1);
        CHECK_NE(std::get<uintptr_t>(authoring::getParameter("stage", "openusd-stage-ptr")), 0);
        CHECK_EQ(std::get<int64_t>(authoring::getParameter("stage", "ovstage-stage-id")), -1);
        CHECK_EQ(std::get<uintptr_t>(authoring::getParameter("stage", "ovstage-stage-ptr")), 0);
        REQUIRE_UNARY(authoring::closeStage());
    }

    TEST_CASE("openStage")
    {
        std::string usdPath = isaacsim::common::test::resolveResourcePath("isaacsim.foundation.ovsim", "variant.usda");
        REQUIRE_UNARY(authoring::openStage(usdPath));
        CHECK_NE(std::get<int64_t>(authoring::getParameter("stage", "openusd-stage-id")), -1);
        CHECK_NE(std::get<uintptr_t>(authoring::getParameter("stage", "openusd-stage-ptr")), 0);
        CHECK_EQ(std::get<int64_t>(authoring::getParameter("stage", "ovstage-stage-id")), -1);
        CHECK_EQ(std::get<uintptr_t>(authoring::getParameter("stage", "ovstage-stage-ptr")), 0);
        REQUIRE_UNARY(authoring::closeStage());
    }

    TEST_CASE("saveStage")
    {
        REQUIRE_UNARY(authoring::createStage());
        REQUIRE_UNARY(authoring::definePrim("/World", "Xform"));
        std::filesystem::path outputPath = std::filesystem::temp_directory_path() / "test_ovsim_save_stage.usda";
        CHECK_UNARY(authoring::saveStage(outputPath.string()));
        CHECK_UNARY(std::filesystem::exists(outputPath));
        std::filesystem::remove(outputPath);
        REQUIRE_UNARY(authoring::closeStage());
    }

    TEST_CASE("closeStage")
    {
        REQUIRE_UNARY(authoring::createStage());
        CHECK_NE(std::get<int64_t>(authoring::getParameter("stage", "openusd-stage-id")), -1);
        CHECK_NE(std::get<uintptr_t>(authoring::getParameter("stage", "openusd-stage-ptr")), 0);

        CHECK_UNARY(authoring::closeStage());
        CHECK_EQ(std::get<int64_t>(authoring::getParameter("stage", "openusd-stage-id")), -1);
        CHECK_EQ(std::get<uintptr_t>(authoring::getParameter("stage", "openusd-stage-ptr")), 0);

        CHECK_THROWS_AS(authoring::closeStage(), std::runtime_error);
    }

    TEST_CASE("addReferenceToStage")
    {
        REQUIRE_UNARY(authoring::createStage());

        REQUIRE_UNARY(authoring::definePrim("/World/Prim"));
        std::string usdPath = isaacsim::common::test::resolveResourcePath("isaacsim.foundation.ovsim", "variant.usda");
        CHECK_UNARY(authoring::addReferenceToStage(usdPath, "/World/Prim"));

        REQUIRE_UNARY(authoring::closeStage());
    }

    TEST_CASE("importStageFromString|exportStageToString")
    {
        REQUIRE_UNARY(authoring::createStage());
        REQUIRE_UNARY(authoring::definePrim("/World", "Xform"));

        std::string usdString = authoring::exportStageToString();
        REQUIRE_UNARY(authoring::closeStage());
        CHECK_UNARY(usdString.find("def Xform \"World\"") != std::string::npos);

        REQUIRE_UNARY(authoring::importStageFromString(usdString));
        REQUIRE_UNARY(authoring::closeStage());
    }
}

// Prim operations.

TEST_SUITE("authoring::prims")
{

    TEST_CASE("definePrim|movePrim|removePrim")
    {
        REQUIRE_UNARY(authoring::createStage());

        REQUIRE_UNARY(authoring::definePrim("/World/A", "Xform"));
        CHECK_UNARY(authoring::movePrim("/World/A", "/World/B"));
        CHECK_UNARY(authoring::removePrim("/World/B"));

        REQUIRE_UNARY(authoring::closeStage());
    }
}

// Attribute operations.

TEST_SUITE("authoring::attributes")
{

    TEST_CASE("createPrimAttribute|removePrimAttribute")
    {
        REQUIRE_UNARY(authoring::createStage());

        REQUIRE_UNARY(authoring::definePrim("/World/A", "Xform"));
        CHECK_UNARY(authoring::createPrimAttribute("/World/A", "attribute1", "float"));
        CHECK_UNARY(authoring::removePrimAttribute("/World/A", "attribute1"));

        CHECK_THROWS_AS(
            authoring::createPrimAttribute("/World/A", "attribute2", "nonExistentType"), std::invalid_argument);
        CHECK_THROWS_AS(
            authoring::removePrimAttribute("/World/A", "attribute2"), isaacsim::common::exceptions::AttributeNameError);

        REQUIRE_UNARY(authoring::closeStage());
    }
}

// Authoring parameter operations.

TEST_SUITE("authoring::parameters")
{

    TEST_CASE("setParameter|getParameter")
    {
        REQUIRE_UNARY(authoring::createStage());

        CHECK_NE(std::get<int64_t>(authoring::getParameter("stage", "openusd-stage-id")), -1);
        CHECK_NE(std::get<uintptr_t>(authoring::getParameter("stage", "openusd-stage-ptr")), 0);
        CHECK_EQ(std::get<int64_t>(authoring::getParameter("stage", "ovstage-stage-id")), -1);
        CHECK_EQ(std::get<uintptr_t>(authoring::getParameter("stage", "ovstage-stage-ptr")), 0);

        CHECK_NOTHROW(authoring::setParameter("stage", "openusd-stage-id", int64_t{ 1 }));
        CHECK_NOTHROW(authoring::setParameter("stage", "openusd-stage-ptr", double{ 1.0 }));
        CHECK_NOTHROW(authoring::setParameter("stage", "ovstage-stage-id", int64_t{ 1 }));
        CHECK_NOTHROW(authoring::setParameter("stage", "ovstage-stage-ptr", double{ 1.0 }));

        REQUIRE_UNARY(authoring::closeStage());
    }
}

// Simulation operations.

TEST_SUITE("simulation")
{

    TEST_CASE("automatic mode: play|pause|stop do not throw")
    {
        CHECK_NOTHROW(simulation::play());
        CHECK_NOTHROW(simulation::pause());
        CHECK_NOTHROW(simulation::stop());
    }

    TEST_CASE("manual mode: initialize|step|invalidate")
    {
        REQUIRE_UNARY(authoring::createStage());

        CHECK_NOTHROW(simulation::initialize());
        CHECK_NOTHROW(simulation::step());
        CHECK_NOTHROW(simulation::invalidate());
        CHECK_NOTHROW(simulation::invalidate()); // invalidate without a prior initialize is a no-op

        REQUIRE_UNARY(authoring::closeStage());
    }

    TEST_CASE("setParameter|getParameter are not implemented")
    {
        CHECK_THROWS_AS(simulation::setParameter("provider", "parameter", int64_t{ 1 }), std::logic_error);
        CHECK_THROWS_AS(simulation::getParameter("provider", "parameter"), std::logic_error);
    }
}
