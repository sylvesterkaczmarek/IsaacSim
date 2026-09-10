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
#include <isaacsim/ovsim/api/Factory.hpp>

#include <stdexcept>
#include <unordered_map>
#include <variant>

using namespace isaacsim::ovsim::api;

TEST_SUITE("Factory::makeClient")
{
    TEST_CASE("Unknown client")
    {
        CHECK_THROWS_AS(makeClient("does-not-exist"), std::runtime_error);
    }

    TEST_CASE("Local client ignores configuration")
    {
        types::Implementation implementation =
            makeClient("local", std::unordered_map<std::string, std::string>{ { "label", "irrelevant-for-local" } });
        REQUIRE_UNARY(implementation.control.authoring.createStage());
        REQUIRE_UNARY(implementation.control.authoring.closeStage());
    }

    TEST_CASE("gRPC client requires configuration")
    {
        CHECK_THROWS_AS(makeClient("grpc"), std::invalid_argument);
    }
}
