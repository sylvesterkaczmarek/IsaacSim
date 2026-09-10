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
#include <isaacsim/foundation/objects/Stage.hpp>
#include <isaacsim/foundation/objects/Xform.hpp>

#include <IsaacSimTest.hpp>
#include <stdexcept>

using namespace isaacsim::foundation::objects;

TEST_SUITE("Xform")
{

    TEST_CASE("Xform::Xform (create)")
    {
        Stage stage = Stage("openusd").createStage();
        REQUIRE_UNARY(stage.isValid());

        // Non-existing paths are created as Xform prims
        Xform prim(std::vector<std::string>{ "/World/A", "/World/B" });
        CHECK_EQ(prim.size(), 2u);
        CHECK_EQ(prim.paths(), (std::vector<std::string>{ "/World/A", "/World/B" }));
        CHECK_EQ(prim.getTypeName(), (std::vector<std::string>{ "Xform", "Xform" }));
        CHECK_EQ(prim.isA("Xform").get<std::vector<bool>>(), (std::vector<bool>{ true, true }));

        REQUIRE_UNARY(stage.closeStage());
    }

    TEST_CASE("Xform::Xform (wrap existing)")
    {
        Stage stage = Stage("openusd").createStage();
        REQUIRE_UNARY(stage.isValid());

        stage.definePrim("/World/X0", "Xform");
        stage.definePrim("/World/X1", "Xform");

        SUBCASE("explicit paths")
        {
            Xform prim(std::vector<std::string>{ "/World/X0", "/World/X1" });
            CHECK_EQ(prim.size(), 2u);
            CHECK_EQ(prim.paths(), (std::vector<std::string>{ "/World/X0", "/World/X1" }));
            CHECK_EQ(prim.isA("Xform").get<std::vector<bool>>(), (std::vector<bool>{ true, true }));
        }

        SUBCASE("regex")
        {
            Xform prim("/World/X.*");
            CHECK_EQ(prim.size(), 2u);
            CHECK_EQ(prim.getTypeName(), (std::vector<std::string>{ "Xform", "Xform" }));
        }

        REQUIRE_UNARY(stage.closeStage());
    }

    TEST_CASE("Xform::Xform (non-Xform throws)")
    {
        Stage stage = Stage("openusd").createStage();
        REQUIRE_UNARY(stage.isValid());

        stage.definePrim("/World/Scope", "Scope");

        CHECK_THROWS_AS(Xform("/World/Scope"), std::runtime_error);
        // A mix that includes a non-Xformable existing prim also throws
        stage.definePrim("/World/Xform", "Xform");
        CHECK_THROWS_AS(Xform(std::vector<std::string>{ "/World/Xform", "/World/Scope" }), std::runtime_error);

        REQUIRE_UNARY(stage.closeStage());
    }
}
