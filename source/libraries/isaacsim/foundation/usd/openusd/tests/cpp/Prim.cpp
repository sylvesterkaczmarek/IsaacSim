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
#include <isaacsim/common/exceptions/Exceptions.hpp>
#include <isaacsim/foundation/usd/openusd/Usd.hpp>

#include <IsaacSimTest.hpp>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace isaacsim::foundation::usd::openusd;

TEST_SUITE("Prim")
{

    TEST_CASE("isPrimValid")
    {
        int64_t stageId = createStage();

        definePrim(stageId, "/World/A", "Xform");

        CHECK_UNARY(isPrimValid(stageId, "/World/A"));
        CHECK_UNARY(isPrimValid(stageId, "/World"));
        // non-existent prim
        CHECK_UNARY_FALSE(isPrimValid(stageId, "/NonExistent"));
        // invalid path string
        CHECK_UNARY_FALSE(isPrimValid(stageId, "Invalid"));

        closeStage(stageId);
    }

    TEST_CASE("getName")
    {
        int64_t stageId = createStage();

        definePrim(stageId, "/World/A/B", "Xform");

        CHECK_EQ(getName(stageId, "/World/A/B"), "B");
        CHECK_EQ(getName(stageId, "/World/A"), "A");
        CHECK_EQ(getName(stageId, "/World"), "World");

        CHECK_THROWS_AS(getName(stageId, "/NonExistent"), isaacsim::common::exceptions::PrimPathError);
        CHECK_THROWS_AS(getName(stageId, "Invalid"), isaacsim::common::exceptions::PrimPathError);

        closeStage(stageId);
    }

    TEST_CASE("getTypeName")
    {
        int64_t stageId = createStage();

        definePrim(stageId, "/World/Cube", "Cube");
        definePrim(stageId, "/World/Xform", "Xform");
        definePrim(stageId, "/World/Untyped", "");

        CHECK_EQ(getTypeName(stageId, "/World/Cube"), "Cube");
        CHECK_EQ(getTypeName(stageId, "/World/Xform"), "Xform");
        CHECK_EQ(getTypeName(stageId, "/World/Untyped"), "");

        CHECK_THROWS_AS(getTypeName(stageId, "/NonExistent"), isaacsim::common::exceptions::PrimPathError);
        CHECK_THROWS_AS(getTypeName(stageId, "Invalid"), isaacsim::common::exceptions::PrimPathError);

        closeStage(stageId);
    }

    TEST_CASE("getParent")
    {
        int64_t stageId = createStage();

        definePrim(stageId, "/World/A/B/C");

        CHECK_EQ(getParent(stageId, "/World/A/B/C"), "/World/A/B");
        CHECK_EQ(getParent(stageId, "/World/A/B"), "/World/A");
        CHECK_EQ(getParent(stageId, "/World/A"), "/World");
        CHECK_EQ(getParent(stageId, "/World"), "/");
        CHECK_EQ(getParent(stageId, "/"), "");

        CHECK_THROWS_AS(getParent(stageId, "Invalid"), isaacsim::common::exceptions::PrimPathError);

        closeStage(stageId);
    }

    TEST_CASE("getChildren")
    {
        int64_t stageId = createStage();

        definePrim(stageId, "/World/A");
        definePrim(stageId, "/World/B");
        definePrim(stageId, "/World/C");
        definePrim(stageId, "/World/A/X");
        definePrim(stageId, "/World/A/Y");

        CHECK_EQ(getChildren(stageId, "/World"), (std::vector<std::string>{ "/World/A", "/World/B", "/World/C" }));
        CHECK_EQ(getChildren(stageId, "/World/A"), (std::vector<std::string>{ "/World/A/X", "/World/A/Y" }));
        CHECK_EQ(getChildren(stageId, "/World/A/X"), std::vector<std::string>{});

        CHECK_THROWS_AS(getChildren(stageId, "/World/Z"), isaacsim::common::exceptions::PrimPathError);

        closeStage(stageId);
    }

    TEST_CASE("isA")
    {
        int64_t stageId = createStage();

        definePrim(stageId, "/World/Cube", "Cube");
        definePrim(stageId, "/World/Xform", "Xform");

        SUBCASE("Concrete schema type")
        {
            CHECK_UNARY(isA(stageId, "/World/Cube", "Cube"));
            CHECK_UNARY(isA(stageId, "/World/Xform", "Xform"));
        }

        SUBCASE("Inheritance")
        {
            // Cube
            CHECK_UNARY(isA(stageId, "/World/Cube", "Gprim"));
            CHECK_UNARY(isA(stageId, "/World/Cube", "Boundable"));
            CHECK_UNARY(isA(stageId, "/World/Cube", "Xformable"));
            CHECK_UNARY(isA(stageId, "/World/Cube", "Imageable"));
            // Xform
            CHECK_UNARY(isA(stageId, "/World/Xform", "Xformable"));
            CHECK_UNARY(isA(stageId, "/World/Xform", "Imageable"));
            CHECK_UNARY_FALSE(isA(stageId, "/World/Xform", "Gprim"));
            CHECK_UNARY_FALSE(isA(stageId, "/World/Xform", "Boundable"));
        }

        SUBCASE("Unrelated concrete schema type")
        {
            CHECK_UNARY_FALSE(isA(stageId, "/World/Cube", "Sphere"));
            CHECK_UNARY_FALSE(isA(stageId, "/World/Xform", "Cube"));
        }

        SUBCASE("Unknown schema type")
        {
            CHECK_THROWS_AS(isA(stageId, "/World/Cube", "NonExistentType"), std::invalid_argument);
        }

        SUBCASE("Invalid prim path")
        {
            CHECK_THROWS_AS(isA(stageId, "/NonExistent", "Cube"), isaacsim::common::exceptions::PrimPathError);
            CHECK_THROWS_AS(isA(stageId, "Invalid", "Cube"), isaacsim::common::exceptions::PrimPathError);
        }

        closeStage(stageId);
    }

    TEST_CASE("hasApi | applyApi | removeApi")
    {
        int64_t stageId = createStage();
        definePrim(stageId, "/World/Cube", "Cube");

        SUBCASE("Single-apply API: PhysicsRigidBodyAPI")
        {
            CHECK_UNARY_FALSE(hasApi(stageId, "/World/Cube", "PhysicsRigidBodyAPI"));

            CHECK_UNARY(applyApi(stageId, "/World/Cube", "PhysicsRigidBodyAPI"));
            CHECK_UNARY(hasApi(stageId, "/World/Cube", "PhysicsRigidBodyAPI"));

            CHECK_UNARY(removeApi(stageId, "/World/Cube", "PhysicsRigidBodyAPI"));
            CHECK_UNARY_FALSE(hasApi(stageId, "/World/Cube", "PhysicsRigidBodyAPI"));
        }

        SUBCASE("Multi-apply API: PhysicsDriveAPI")
        {
            CHECK_UNARY_FALSE(hasApi(stageId, "/World/Cube", "PhysicsDriveAPI", "linear"));

            CHECK_UNARY(applyApi(stageId, "/World/Cube", "PhysicsDriveAPI", "linear"));
            CHECK_UNARY(hasApi(stageId, "/World/Cube", "PhysicsDriveAPI", "linear"));
            CHECK_UNARY_FALSE(hasApi(stageId, "/World/Cube", "PhysicsDriveAPI", "angular"));

            CHECK_UNARY(removeApi(stageId, "/World/Cube", "PhysicsDriveAPI", "linear"));
            CHECK_UNARY_FALSE(hasApi(stageId, "/World/Cube", "PhysicsDriveAPI", "linear"));
        }

        SUBCASE("Unknown schema type")
        {
            CHECK_THROWS_AS(hasApi(stageId, "/World/Cube", "NonExistentAPI"), std::invalid_argument);
            CHECK_THROWS_AS(applyApi(stageId, "/World/Cube", "NonExistentAPI"), std::invalid_argument);
            CHECK_THROWS_AS(removeApi(stageId, "/World/Cube", "NonExistentAPI"), std::invalid_argument);
        }

        SUBCASE("Concrete typed schema")
        {
            CHECK_THROWS_AS(hasApi(stageId, "/World/Cube", "Cube", std::nullopt), std::invalid_argument);
            CHECK_THROWS_AS(applyApi(stageId, "/World/Cube", "Cube", std::nullopt), std::invalid_argument);
            CHECK_THROWS_AS(removeApi(stageId, "/World/Cube", "Cube", std::nullopt), std::invalid_argument);
        }

        closeStage(stageId);
    }

    TEST_CASE("getAppliedSchemas")
    {
        int64_t stageId = createStage();
        definePrim(stageId, "/World/Cube", "Cube");

        // No applied schemas
        CHECK_EQ(getAppliedSchemas(stageId, "/World/Cube"), std::vector<std::string>{});

        // Apply schemas and get them
        applyApi(stageId, "/World/Cube", "PhysicsRigidBodyAPI");
        applyApi(stageId, "/World/Cube", "PhysicsDriveAPI", "linear");
        applyApi(stageId, "/World/Cube", "PhysicsDriveAPI", "angular");
        const auto schemas = getAppliedSchemas(stageId, "/World/Cube");
        CHECK_EQ(schemas.size(), 3);
        CHECK_UNARY(std::find(schemas.begin(), schemas.end(), "PhysicsRigidBodyAPI") != schemas.end());
        CHECK_UNARY(std::find(schemas.begin(), schemas.end(), "PhysicsDriveAPI:linear") != schemas.end());
        CHECK_UNARY(std::find(schemas.begin(), schemas.end(), "PhysicsDriveAPI:angular") != schemas.end());

        closeStage(stageId);
    }

    TEST_CASE("getVariants | setVariants")
    {
        std::string usdPath =
            isaacsim::common::test::resolveResourcePath("isaacsim.foundation.usd.openusd", "variant.usda");
        int64_t stageId = openStage(usdPath);

        SUBCASE("getVariants(selection=true)")
        {
            auto result = getVariants(stageId, "/World/Sphere");
            REQUIRE_EQ(result.size(), 2);
            REQUIRE_UNARY(result.count("color"));
            REQUIRE_UNARY(result.count("radius"));
            CHECK_EQ(result.at("color"), std::vector<std::string>{ "red" });
            CHECK_EQ(result.at("radius"), std::vector<std::string>{ "small" });
        }

        SUBCASE("getVariants(selection=false)")
        {
            auto result = getVariants(stageId, "/World/Sphere", false);
            REQUIRE_EQ(result.size(), 2);
            REQUIRE_UNARY(result.count("color"));
            REQUIRE_UNARY(result.count("radius"));
            // GetVariantNames() returns names in alphabetical order.
            CHECK_EQ(result.at("color"), (std::vector<std::string>{ "blue", "red" }));
            CHECK_EQ(result.at("radius"), (std::vector<std::string>{ "large", "small" }));
        }

        SUBCASE("setVariants")
        {
            REQUIRE_NOTHROW(setVariants(stageId, "/World/Sphere", { { "color", { "blue" } }, { "radius", { "large" } } }));
            auto variants = getVariants(stageId, "/World/Sphere");
            CHECK_EQ(variants.at("color"), std::vector<std::string>{ "blue" });
            CHECK_EQ(variants.at("radius"), std::vector<std::string>{ "large" });
        }

        SUBCASE("setVariants: invalid variant set/selection")
        {
            CHECK_THROWS_AS(
                setVariants(stageId, "/World/Sphere", { { "NonExistent", { "red" } } }), std::invalid_argument);
            CHECK_THROWS_AS(
                setVariants(stageId, "/World/Sphere", { { "radius", { "Invalid" } } }), std::invalid_argument);
        }

        SUBCASE("getVariants | setVariants: invalid/non-existent prim")
        {
            CHECK_THROWS_AS(getVariants(stageId, "/NonExistent", true), isaacsim::common::exceptions::PrimPathError);
            CHECK_THROWS_AS(getVariants(stageId, "Invalid", false), isaacsim::common::exceptions::PrimPathError);
            CHECK_THROWS_AS(setVariants(stageId, "/NonExistent", { { "color", { "red" } } }),
                            isaacsim::common::exceptions::PrimPathError);
            CHECK_THROWS_AS(setVariants(stageId, "Invalid", { { "color", { "red" } } }),
                            isaacsim::common::exceptions::PrimPathError);
        }

        closeStage(stageId);
    }
}
