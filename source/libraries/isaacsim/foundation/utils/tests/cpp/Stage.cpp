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
#include <isaacsim/foundation/utils/Stage.hpp>

#include <stdexcept>

using namespace isaacsim::foundation::utils;
namespace objects = isaacsim::foundation::objects;

TEST_SUITE("Stage")
{
    TEST_CASE("setDefaultStage | getDefaultStage roundtrip")
    {
        objects::Stage stage = objects::Stage("openusd");

        CHECK_THROWS_AS(getDefaultStage(), std::runtime_error);
        CHECK_THROWS_AS(setDefaultStage(stage), std::runtime_error);

        stage.createStage(std::nullopt, false);
        int64_t stageId = stage.getStageId();
        setDefaultStage(stage);
        CHECK_EQ(getDefaultStage().getStageId(), stageId);
        CHECK_EQ(getDefaultStage().getBackend(), "openusd");

        REQUIRE_UNARY(stage.closeStage());
        CHECK_THROWS_AS(getDefaultStage(), std::runtime_error);
    }

    TEST_CASE("getActiveStage falls back to default stage")
    {
        CHECK_THROWS_AS(getDefaultStage(), std::runtime_error);
        CHECK_THROWS_AS(getActiveStage(), std::runtime_error);

        objects::Stage stage = objects::Stage("openusd");
        stage.createStage(std::nullopt, false);
        int64_t stageId = stage.getStageId();
        setDefaultStage(stage);
        CHECK_EQ(getActiveStage().getStageId(), stageId);
        CHECK_EQ(getActiveStage().getBackend(), "openusd");

        REQUIRE_UNARY(stage.closeStage());
        CHECK_THROWS_AS(getDefaultStage(), std::runtime_error);
        CHECK_THROWS_AS(getActiveStage(), std::runtime_error);
    }

    TEST_CASE("StageGuard scopes the active stage")
    {
        CHECK_THROWS_AS(getDefaultStage(), std::runtime_error);
        CHECK_THROWS_AS(getActiveStage(), std::runtime_error);

        setDefaultStage(objects::Stage("openusd").createStage(std::nullopt, false));

        {
            StageGuard guard(objects::Stage("ovstage").createStage(std::nullopt, false));
            CHECK_EQ(getActiveStage().getBackend(), "ovstage");
            REQUIRE_UNARY(getActiveStage().closeStage());
        }

        // Thread-local active cleared by guard; falls back to default
        CHECK_EQ(getActiveStage().getBackend(), "openusd");

        REQUIRE_UNARY(getDefaultStage().closeStage());
        CHECK_THROWS_AS(getDefaultStage(), std::runtime_error);
        CHECK_THROWS_AS(getActiveStage(), std::runtime_error);
    }
}
