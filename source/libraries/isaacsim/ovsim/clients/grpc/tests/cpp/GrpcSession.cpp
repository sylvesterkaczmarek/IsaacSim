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
#include <isaacsim/ovsim/clients/grpc/GrpcSession.hpp>

#include <stdexcept>
#include <unordered_map>

using isaacsim::ovsim::clients::grpc::GrpcSession;

namespace
{
using Configuration = std::unordered_map<std::string, std::string>;
}

TEST_SUITE("GrpcSession::construction")
{
    TEST_CASE("Constructor argument")
    {
        GrpcSession session(Configuration{ { "endpoint", "grpc://host:50051" } });
        CHECK_EQ(session.configuration().at("endpoint"), "grpc://host:50051");
    }

    TEST_CASE("Sessions hold independent configurations")
    {
        GrpcSession first(Configuration{ { "endpoint", "grpc://host-a:50051" } });
        GrpcSession second(Configuration{ { "endpoint", "grpc://host-b:50052" } });
        CHECK_EQ(first.configuration().at("endpoint"), "grpc://host-a:50051");
        CHECK_EQ(second.configuration().at("endpoint"), "grpc://host-b:50052");
    }
}

TEST_SUITE("GrpcSession::control::authoring")
{
    TEST_CASE("stub methods throw logic_error")
    {
        GrpcSession session(Configuration{ { "endpoint", "grpc://host:50051" } });
        CHECK_THROWS_AS(session.createStage(), std::logic_error);
        CHECK_THROWS_AS(session.openStage("/path.usd"), std::logic_error);
        CHECK_THROWS_AS(session.saveStage("/path.usd"), std::logic_error);
        CHECK_THROWS_AS(session.importStageFromString("usd"), std::logic_error);
        CHECK_THROWS_AS(session.exportStageToString(), std::logic_error);
        CHECK_THROWS_AS(session.closeStage(), std::logic_error);
        CHECK_THROWS_AS(session.addReferenceToStage("/path.usd", "/World/Ref", "Xform"), std::logic_error);
        CHECK_THROWS_AS(session.definePrim("/World", "Xform"), std::logic_error);
        CHECK_THROWS_AS(session.movePrim("/World/A", "/World/B"), std::logic_error);
        CHECK_THROWS_AS(session.removePrim("/World/A"), std::logic_error);
        CHECK_THROWS_AS(session.createPrimAttribute("/World/A", "mass", "float"), std::logic_error);
        CHECK_THROWS_AS(session.removePrimAttribute("/World/A", "mass"), std::logic_error);
        CHECK_THROWS_AS(session.authoringGetParameter("stage", "up-axis"), std::logic_error);
        CHECK_THROWS_AS(session.authoringSetParameter("stage", "up-axis", "Z"), std::logic_error);
    }
}

TEST_SUITE("GrpcSession::control::simulation")
{
    TEST_CASE("stub methods throw logic_error")
    {
        GrpcSession session(Configuration{ { "endpoint", "grpc://host:50051" } });
        CHECK_THROWS_AS(session.play(), std::logic_error);
        CHECK_THROWS_AS(session.pause(), std::logic_error);
        CHECK_THROWS_AS(session.stop(), std::logic_error);
        CHECK_THROWS_AS(session.initialize(), std::logic_error);
        CHECK_THROWS_AS(session.invalidate(), std::logic_error);
        CHECK_THROWS_AS(session.step(), std::logic_error);
        CHECK_THROWS_AS(session.simulationGetParameter("physics", "physics-engine"), std::logic_error);
        CHECK_THROWS_AS(session.simulationSetParameter("physics", "physics-engine", "my-engine"), std::logic_error);
    }
}

TEST_SUITE("GrpcSession::data")
{
    TEST_CASE("stub methods throw logic_error")
    {
        GrpcSession session(Configuration{ { "endpoint", "grpc://host:50051" } });
        CHECK_THROWS_AS(session.read("/World/Prim", "mass", std::nullopt), std::logic_error);
        CHECK_THROWS_AS(session.write("/World/Prim", "mass", std::string("1.0"), std::nullopt), std::logic_error);
    }
}
