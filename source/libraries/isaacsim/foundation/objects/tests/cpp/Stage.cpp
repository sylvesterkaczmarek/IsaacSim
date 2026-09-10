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
#include <isaacsim/foundation/objects/Prim.hpp>
#include <isaacsim/foundation/objects/Stage.hpp>

#include <IsaacSimTest.hpp>
#include <algorithm>
#include <vector>

using namespace isaacsim::foundation::objects;

TEST_SUITE("Stage")
{

    TEST_CASE("Stage::getStagePtr")
    {
        auto backend = GENERATE("openusd", "ovstage");

        Stage stage = Stage(backend).createStage();
        CHECK_NE(stage.getStagePtr(), nullptr);
        REQUIRE_UNARY(stage.closeStage());
        CHECK_EQ(stage.getStagePtr(), nullptr);
        CHECK_EQ(Stage(backend).getStagePtr(), nullptr);
    }

    TEST_CASE("Stage::createStage")
    {
        auto backend = GENERATE("openusd", "ovstage");

        SUBCASE("Create and close a stage")
        {
            Stage stage(backend);
            Stage& reference = stage.createStage();
            CHECK_EQ(&reference, &stage);
            REQUIRE_NE(stage.getStageId(), -1);
            REQUIRE_UNARY(stage.isValid());
            REQUIRE_UNARY(stage.closeStage());
            REQUIRE_UNARY_FALSE(stage.isValid());
            REQUIRE_UNARY_FALSE(stage.closeStage());
        }

        SUBCASE("Successive create calls produce different IDs")
        {
            Stage stage1 = Stage(backend).createStage();
            Stage stage2 = Stage(backend).createStage();
            REQUIRE_NE(&stage1, &stage2);
            REQUIRE_NE(stage1.getStageId(), stage2.getStageId());
            REQUIRE_UNARY(stage1.closeStage());
            REQUIRE_UNARY(stage2.closeStage());
        }
    }

    TEST_CASE("Stage::openStage")
    {
        auto backend = GENERATE("openusd", "ovstage");

        std::string usdPath = isaacsim::common::test::resolveResourcePath("isaacsim.foundation.objects", "scene.usda");
        Stage stage(backend);
        Stage& reference = stage.openStage(usdPath);
        CHECK_EQ(&reference, &stage);
        REQUIRE_NE(stage.getStageId(), -1);
        REQUIRE_UNARY(stage.isValid());
        REQUIRE_UNARY(stage.closeStage());
        REQUIRE_UNARY_FALSE(stage.isValid());
        REQUIRE_UNARY_FALSE(stage.closeStage());
    }

    TEST_CASE("Stage::saveStage")
    {
        auto backend = GENERATE("openusd"); // TODO: unsupported by ovstage (save)

        SUBCASE("Save to a new file")
        {
            Stage stage = Stage(backend).createStage();
            stage.definePrim("/World", "Xform");
            std::filesystem::path outputPath = std::filesystem::temp_directory_path() / "test_save_stage.usda";
            REQUIRE_UNARY(stage.saveStage(outputPath.string()));
            REQUIRE_UNARY(std::filesystem::exists(outputPath));
            std::filesystem::remove(outputPath);
            REQUIRE_UNARY(stage.closeStage());
        }

        SUBCASE("Save and reopen")
        {
            Stage stage = Stage(backend).createStage();
            stage.definePrim("/World", "Xform");
            std::filesystem::path outputPath = std::filesystem::temp_directory_path() / "test_save_reopen.usda";
            REQUIRE_UNARY(stage.saveStage(outputPath.string()));
            REQUIRE_UNARY(stage.closeStage());

            Stage reopened = Stage(backend).openStage(outputPath.string());
            REQUIRE_UNARY(reopened.isValid());
            REQUIRE_UNARY(reopened.closeStage());
            std::filesystem::remove(outputPath);
        }

        SUBCASE("Save on closed stage throws")
        {
            Stage stage = Stage(backend).createStage();
            REQUIRE_UNARY(stage.closeStage());
            CHECK_THROWS_AS(stage.saveStage("/tmp/should_not_exist.usda"), std::runtime_error);
        }
    }

    TEST_CASE("Stage::addReference")
    {
        auto backend = GENERATE("openusd"); // TODO: unsupported by ovstage (add reference)

        std::string usdPath = isaacsim::common::test::resolveResourcePath("isaacsim.foundation.objects", "variant.usda");

        SUBCASE("Add reference")
        {
            Stage stage = Stage(backend).createStage();
            REQUIRE_NOTHROW(stage.addReference(usdPath, "/World/Reference"));
            REQUIRE_UNARY(stage.closeStage());
        }

        SUBCASE("Add reference with variant selection")
        {
            Stage stage = Stage(backend).createStage();
            REQUIRE_NOTHROW(stage.addReference(
                usdPath, "/World/Reference", "Xform",
                std::unordered_map<std::string, std::string>{ { "color", "blue" }, { "radius", "large" } }));
            auto result = Prim("/World/Reference").getVariantSelection();
            REQUIRE_EQ(result.size(), 1u);
            CHECK_EQ(result[0].at("color"), "blue");
            CHECK_EQ(result[0].at("radius"), "large");
            REQUIRE_UNARY(stage.closeStage());
        }

        SUBCASE("Exceptions")
        {
            Stage stage = Stage(backend).createStage();
            CHECK_THROWS_AS(stage.addReference(usdPath, "/World/"), isaacsim::common::exceptions::PrimPathStringError);
            CHECK_THROWS_AS(stage.addReference("/unknown/file.usda", "/World/Reference"), std::runtime_error);
            REQUIRE_UNARY(stage.closeStage());
        }
    }

    TEST_CASE("Stage::definePrim")
    {
        auto backend = GENERATE("openusd", "ovstage");

        Stage stage = Stage(backend).createStage();
        REQUIRE_UNARY(stage.isValid());

        const std::vector<std::string> typeNames = {
            "Camera",    "Capsule",      "Cone",      "Cube",      "Cylinder",    "Mesh",
            "Plane",     "Points",       "Scope",     "Sphere",    "Xform",       "CylinderLight",
            "DiskLight", "DistantLight", "DomeLight", "RectLight", "SphereLight", "PhysicsScene",
        };
        for (const auto& typeName : typeNames)
        {
            REQUIRE_NOTHROW(stage.definePrim("/" + typeName, typeName));
        }

        // Redefining a prim with the same type does not throw
        REQUIRE_NOTHROW(stage.definePrim("/Sphere", "Sphere"));

        // Non-absolute path
        CHECK_THROWS_AS(stage.definePrim("World", "Xform"), isaacsim::common::exceptions::PrimPathStringError);
        // Non-valid path (trailing slash)
        CHECK_THROWS_AS(stage.definePrim("/World/", "Xform"), isaacsim::common::exceptions::PrimPathStringError);
        // Prim already exists with a different type
        CHECK_THROWS_AS(stage.definePrim("/Sphere", "Cube"), std::runtime_error);

        REQUIRE_UNARY(stage.closeStage());
    }

    TEST_CASE("Stage::movePrim")
    {
        auto backend = GENERATE("openusd"); // TODO: unsupported by ovstage (move prim)

        Stage stage = Stage(backend).createStage();
        REQUIRE_UNARY(stage.isValid());

        stage.definePrim("/World/A", "Xform");
        stage.definePrim("/World/B", "Xform");

        SUBCASE("movePrim")
        {
            std::tuple<bool, std::string> result;

            result = stage.movePrim("/World/A", "/World/B");
            CHECK_UNARY(std::get<0>(result));
            CHECK_EQ(std::get<1>(result), "/World/B/A");

            result = stage.movePrim("/World/B/A", "/World");
            CHECK_UNARY(std::get<0>(result));
            CHECK_EQ(std::get<1>(result), "/World/A");

            result = stage.movePrim("/World/A", "/");
            CHECK_UNARY(std::get<0>(result));
            CHECK_EQ(std::get<1>(result), "/A");

            result = stage.movePrim("/A", "/World/C");
            CHECK_UNARY(std::get<0>(result));
            CHECK_EQ(std::get<1>(result), "/World/C");
        }

        SUBCASE("Exceptions")
        {
            CHECK_THROWS_AS(stage.movePrim("/NonExistent", "/"), isaacsim::common::exceptions::PrimPathError);
            CHECK_THROWS_AS(stage.movePrim("/World/B", "?"), isaacsim::common::exceptions::PrimPathStringError);
            CHECK_THROWS_AS(stage.movePrim("/World/B", "/World/X/Y"), std::invalid_argument);
        }

        REQUIRE_UNARY(stage.closeStage());
    }

    TEST_CASE("Stage::removePrim")
    {
        auto backend = GENERATE("openusd"); // TODO: unsupported by ovstage (add reference, used by this test)

        Stage stage = Stage(backend).createStage();
        REQUIRE_UNARY(stage.isValid());

        SUBCASE("Delete a locally defined prim")
        {
            stage.definePrim("/World/A", "Xform");
            CHECK_UNARY(stage.removePrim("/World/A"));
        }

        SUBCASE("Delete an ancestral prim (defined inside the reference)")
        {
            std::string usdPath =
                isaacsim::common::test::resolveResourcePath("isaacsim.foundation.objects", "variant.usda");
            stage.addReference(usdPath, "/World/Reference");
            CHECK_UNARY_FALSE(stage.removePrim("/World/Reference/Cube"));
            CHECK_UNARY(stage.removePrim("/World/Reference"));
        }

        SUBCASE("Delete non-existent prim")
        {
            CHECK_THROWS_AS(stage.removePrim("/NonExistent"), isaacsim::common::exceptions::PrimPathError);
        }

        REQUIRE_UNARY(stage.closeStage());
    }

    TEST_CASE("Stage::exportStageToString|importStageFromString")
    {
        auto backend = GENERATE("openusd"); // TODO: unsupported by ovstage (export)

        Stage stage = Stage(backend).createStage();
        REQUIRE_UNARY(stage.isValid());
        stage.definePrim("/World/Cube", "Cube");

        SUBCASE("Non-empty USDA string")
        {
            std::string usdaString = stage.exportStageToString();
            CHECK_UNARY_FALSE(usdaString.empty());
            CHECK_UNARY(usdaString.find("def Cube \"Cube\"") != std::string::npos);
        }

        SUBCASE("Round-trip stage content")
        {
            std::string usdaString = stage.exportStageToString();
            Stage newStage = Stage(backend).importStageFromString(usdaString);
            CHECK_UNARY(newStage.isValid());
            CHECK_UNARY(newStage.removePrim("/World/Cube"));
            REQUIRE_UNARY(newStage.closeStage());
        }

        SUBCASE("Invalid USDA")
        {
            Stage newStage = Stage(backend).importStageFromString("This is not valid USDA");
            CHECK_UNARY_FALSE(newStage.isValid());
        }

        REQUIRE_UNARY(stage.closeStage());
    }

    TEST_CASE("Stage::getUpAxis|setUpAxis")
    {
        auto backend = GENERATE("openusd"); // TODO: unsupported by ovstage (axis)

        Stage stage = Stage(backend).createStage();
        REQUIRE_UNARY(stage.isValid());

        SUBCASE("getUpAxis")
        {
            CHECK_EQ(stage.getUpAxis(), "Z");
        }

        SUBCASE("setUpAxis")
        {
            for (const auto& upAxis : std::vector<std::string>{ "Y", "Z", "y", "z" })
            {
                REQUIRE_NOTHROW(stage.setUpAxis(upAxis));
                std::string result = stage.getUpAxis();
                std::string expected = upAxis;
                std::transform(expected.begin(), expected.end(), expected.begin(),
                               [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
                CHECK_EQ(result, expected);
            }
        }

        SUBCASE("setUpAxis: invalid up axis")
        {
            CHECK_THROWS_AS(stage.setUpAxis("X"), std::invalid_argument);
            CHECK_THROWS_AS(stage.setUpAxis(""), std::invalid_argument);
        }

        REQUIRE_UNARY(stage.closeStage());
    }

    TEST_CASE("Stage::getUnits|setUnits")
    {
        auto backend = GENERATE("openusd"); // TODO: unsupported by ovstage (units)

        Stage stage = Stage(backend).createStage();
        REQUIRE_UNARY(stage.isValid());

        SUBCASE("getUnits")
        {
            auto [metersPerUnit, kilogramsPerUnit] = stage.getUnits();
            CHECK_EQ(metersPerUnit, 1.0f);
            CHECK_EQ(kilogramsPerUnit, 1.0f);
        }

        SUBCASE("setUnits(metersPerUnit)")
        {
            stage.setUnits(0.01f, std::nullopt);
            auto [metersPerUnit, kilogramsPerUnit] = stage.getUnits();
            CHECK_EQ(doctest::Approx(metersPerUnit), 0.01f);
            CHECK_EQ(doctest::Approx(kilogramsPerUnit), 1.0f);
        }

        SUBCASE("setUnits(kilogramsPerUnit)")
        {
            stage.setUnits(std::nullopt, 0.001f);
            auto [metersPerUnit, kilogramsPerUnit] = stage.getUnits();
            CHECK_EQ(doctest::Approx(metersPerUnit), 1.0f);
            CHECK_EQ(doctest::Approx(kilogramsPerUnit), 0.001f);
        }

        SUBCASE("setUnits(metersPerUnit, kilogramsPerUnit)")
        {
            stage.setUnits(0.0254f, 0.4536f);
            auto [metersPerUnit, kilogramsPerUnit] = stage.getUnits();
            CHECK_EQ(doctest::Approx(metersPerUnit), 0.0254f);
            CHECK_EQ(doctest::Approx(kilogramsPerUnit), 0.4536f);
        }

        REQUIRE_UNARY(stage.closeStage());
    }

    TEST_CASE("Stage::getTimeCode|setTimeCode")
    {
        auto backend = GENERATE("openusd"); // TODO: unsupported by ovstage (time code)

        Stage stage = Stage(backend).createStage();
        REQUIRE_UNARY(stage.isValid());

        SUBCASE("getTimeCode")
        {
            auto [startTimeCode, endTimeCode, timeCodesPerSecond] = stage.getTimeCode();
            CHECK_EQ(startTimeCode, 0.0f);
            CHECK_EQ(endTimeCode, 1000000.0f);
            CHECK_EQ(timeCodesPerSecond, 60.0f);
        }

        SUBCASE("setTimeCode(startTimeCode)")
        {
            stage.setTimeCode(10.0f, std::nullopt, std::nullopt);
            auto [startTimeCode, endTimeCode, timeCodesPerSecond] = stage.getTimeCode();
            CHECK_EQ(startTimeCode, 10.0f);
            CHECK_EQ(endTimeCode, 1000000.0f);
            CHECK_EQ(timeCodesPerSecond, 60.0f);
        }

        SUBCASE("setTimeCode(endTimeCode)")
        {
            stage.setTimeCode(std::nullopt, 200.0f, std::nullopt);
            auto [startTimeCode, endTimeCode, timeCodesPerSecond] = stage.getTimeCode();
            CHECK_EQ(startTimeCode, 0.0f);
            CHECK_EQ(endTimeCode, 200.0f);
            CHECK_EQ(timeCodesPerSecond, 60.0f);
        }

        SUBCASE("setTimeCode(timeCodesPerSecond)")
        {
            stage.setTimeCode(std::nullopt, std::nullopt, 30.0f);
            auto [startTimeCode, endTimeCode, timeCodesPerSecond] = stage.getTimeCode();
            CHECK_EQ(startTimeCode, 0.0f);
            CHECK_EQ(endTimeCode, 1000000.0f);
            CHECK_EQ(timeCodesPerSecond, 30.0f);
        }

        SUBCASE("setTimeCode(startTimeCode, endTimeCode, timeCodesPerSecond)")
        {
            stage.setTimeCode(20.0f, 300.0f, 40.0f);
            auto [startTimeCode, endTimeCode, timeCodesPerSecond] = stage.getTimeCode();
            CHECK_EQ(startTimeCode, 20.0f);
            CHECK_EQ(endTimeCode, 300.0f);
            CHECK_EQ(timeCodesPerSecond, 40.0f);
        }

        REQUIRE_UNARY(stage.closeStage());
    }
}
