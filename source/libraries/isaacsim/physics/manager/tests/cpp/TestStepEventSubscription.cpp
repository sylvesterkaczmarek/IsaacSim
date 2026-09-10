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

#include <atomic>

using namespace isaacsim::physics::manager;
using namespace isaacsim::physics::registration;

namespace
{

// State the stub simulation captures so the test can assert what the
// engine layer received.
struct CapturedSubscribe
{
    std::atomic<int> calls{ 0 };
    bool lastPreStep{ false };
    int lastOrder{ -1 };
    OnPhysicsStepEventFunction lastCallback;
    SubscriptionId nextId{ 1 };

    std::atomic<int> unsubscribeCallCount{ 0 };
    SubscriptionId lastUnsubscribedSubscriptionId{ g_kInvalidSubscriptionId };
};

CapturedSubscribe g_capture;

// Build a `Simulation` whose `simulationFunctions.subscribePhysicsOnStepEvents`
// captures the call into `g_capture` and returns a unique subscription id.
Simulation makeStubSimulation()
{
    Simulation simulation;
    simulation.simulationFunctions.subscribePhysicsOnStepEvents =
        [](bool preStep, int order, OnPhysicsStepEventFunction onUpdate) -> SubscriptionId
    {
        g_capture.calls.fetch_add(1);
        g_capture.lastPreStep = preStep;
        g_capture.lastOrder = order;
        g_capture.lastCallback = onUpdate;
        SubscriptionId id = g_capture.nextId;
        g_capture.nextId = SubscriptionId(g_capture.nextId.id + 1);
        return id;
    };
    simulation.simulationFunctions.unsubscribePhysicsOnStepEvents = [](SubscriptionId id)
    {
        g_capture.unsubscribeCallCount.fetch_add(1);
        g_capture.lastUnsubscribedSubscriptionId = id;
    };
    return simulation;
}

void resetCapture()
{
    g_capture.calls = 0;
    g_capture.lastPreStep = false;
    g_capture.lastOrder = -1;
    g_capture.lastCallback = nullptr;
    g_capture.nextId = SubscriptionId(1);
    g_capture.unsubscribeCallCount = 0;
    g_capture.lastUnsubscribedSubscriptionId = g_kInvalidSubscriptionId;
}

} // namespace

//=============================================================================
// TEST: `subscribePhysicsOnStepEvents` forwards `preStep=true` to the
// engine's `simulationFunctions.subscribePhysicsOnStepEvents`.
//=============================================================================
TEST_CASE("subscribePhysicsOnStepEvents forwards preStep=true")
{
    resetCapture();

    Simulation engine = makeStubSimulation();
    SimulationId simulationId = registerSimulation(engine, "PreStepTest");
    REQUIRE(simulationId != g_kInvalidSimulationId);

    OnPhysicsStepEventFunction callback = [](float /*elapsedTime*/, const PhysicsStepContext& /*context*/) {};
    SubscriptionId subscriptionId = subscribePhysicsOnStepEvents(/*preStep=*/true, /*order=*/5, callback);
    REQUIRE(subscriptionId != g_kInvalidSubscriptionId);

    REQUIRE(g_capture.calls == 1);
    REQUIRE(g_capture.lastPreStep == true);
    REQUIRE(g_capture.lastOrder == 5);
    REQUIRE(g_capture.lastCallback != nullptr);

    unsubscribePhysicsOnStepEvents(subscriptionId);
    REQUIRE(g_capture.unsubscribeCallCount == 1);

    unregisterSimulation(simulationId);
}

//=============================================================================
// TEST: `subscribePhysicsOnStepEvents` forwards `preStep=false`.
//=============================================================================
TEST_CASE("subscribePhysicsOnStepEvents forwards preStep=false")
{
    resetCapture();
    Simulation engine = makeStubSimulation();
    SimulationId simulationId = registerSimulation(engine, "PostStepTest");

    OnPhysicsStepEventFunction callback = [](float /*elapsedTime*/, const PhysicsStepContext& /*context*/) {};
    SubscriptionId subscriptionId = subscribePhysicsOnStepEvents(/*preStep=*/false, /*order=*/0, callback);
    REQUIRE(subscriptionId != g_kInvalidSubscriptionId);

    REQUIRE(g_capture.calls == 1);
    REQUIRE(g_capture.lastPreStep == false);
    REQUIRE(g_capture.lastOrder == 0);

    unsubscribePhysicsOnStepEvents(subscriptionId);
    unregisterSimulation(simulationId);
}

//=============================================================================
// TEST: `order` field passes through unchanged for every value the
// engine might receive.
//=============================================================================
TEST_CASE("subscribePhysicsOnStepEvents preserves the order integer")
{
    resetCapture();
    Simulation engine = makeStubSimulation();
    SimulationId simulationId = registerSimulation(engine, "OrderTest");

    OnPhysicsStepEventFunction callback = [](float /*elapsedTime*/, const PhysicsStepContext& /*context*/) {};
    for (int order : { 0, 10, 100, -7 })
    {
        g_capture.lastOrder = -999;
        SubscriptionId subscriptionId = subscribePhysicsOnStepEvents(/*preStep=*/true, order, callback);
        REQUIRE(subscriptionId != g_kInvalidSubscriptionId);
        REQUIRE(g_capture.lastOrder == order);
        unsubscribePhysicsOnStepEvents(subscriptionId);
    }

    unregisterSimulation(simulationId);
}

//=============================================================================
// TEST: pre+post subscriptions on the same simulation produce distinct
// subscription ids — the umbrella's `addOnStepSubscription` mints fresh
// ids for each forwarding round-trip.
//=============================================================================
TEST_CASE("subscribePhysicsOnStepEvents pre+post produce distinct subscription ids")
{
    resetCapture();
    Simulation engine = makeStubSimulation();
    SimulationId simulationId = registerSimulation(engine, "DistinctIdTest");

    OnPhysicsStepEventFunction callback = [](float /*elapsedTime*/, const PhysicsStepContext& /*context*/) {};
    SubscriptionId preStepSubscriptionId = subscribePhysicsOnStepEvents(/*preStep=*/true, /*order=*/0, callback);
    SubscriptionId postStepSubscriptionId = subscribePhysicsOnStepEvents(/*preStep=*/false, /*order=*/0, callback);

    REQUIRE(preStepSubscriptionId != g_kInvalidSubscriptionId);
    REQUIRE(postStepSubscriptionId != g_kInvalidSubscriptionId);
    REQUIRE(preStepSubscriptionId != postStepSubscriptionId);
    REQUIRE(g_capture.calls == 2);

    unsubscribePhysicsOnStepEvents(preStepSubscriptionId);
    unsubscribePhysicsOnStepEvents(postStepSubscriptionId);
    REQUIRE(g_capture.unsubscribeCallCount == 2);

    unregisterSimulation(simulationId);
}

//=============================================================================
// TEST: subscribe with no engine that supports the API returns
// `g_kInvalidSubscriptionId`. Mirrors the umbrella's "no eligible
// simulations" path — the consumer can detect this without crashing.
//=============================================================================
TEST_CASE("subscribePhysicsOnStepEvents returns invalid id when no simulation supports it")
{
    resetCapture();

    // Register a simulation whose simulationFunctions.subscribePhysicsOnStepEvents
    // is *not* set — the umbrella should skip it and ultimately return
    // invalid.
    Simulation engine; // default: all functions null
    SimulationId simulationId = registerSimulation(engine, "NoStepFunctionsTest");
    REQUIRE(simulationId != g_kInvalidSimulationId);

    OnPhysicsStepEventFunction callback = [](float /*elapsedTime*/, const PhysicsStepContext& /*context*/) {};
    SubscriptionId subscriptionId = subscribePhysicsOnStepEvents(/*preStep=*/true, /*order=*/0, callback);
    REQUIRE(subscriptionId == g_kInvalidSubscriptionId);
    REQUIRE(g_capture.calls == 0);

    unregisterSimulation(simulationId);
}
