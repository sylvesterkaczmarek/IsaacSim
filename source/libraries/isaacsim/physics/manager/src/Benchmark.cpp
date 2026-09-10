// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// clang-format off
#include <isaacsim/physics/manager/PhysicsBenchmark.hpp>
// clang-format on

#include "SimulationSnapshot.hpp"
#include "SubscriptionStore.hpp"

#include <isaacsim/physics/registration/Physics.hpp>

namespace isaacsim
{
namespace physics
{
namespace manager
{

using namespace registration;

namespace
{
details::SubscriptionStore g_benchmarkSubscriptions;
}

SubscriptionId subscribeProfileStatisticsEvents(ProfileStatisticsNotificationFunction onEvent)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();

    std::vector<details::SimulationSubscription> simulationSubscriptions;

    for (const auto& simulation : simulations)
    {
        if (simulation.second.simulation.benchmarkFunctions.subscribeProfileStatisticsEvents)
        {
            SubscriptionId simulationSubscriptionId =
                simulation.second.simulation.benchmarkFunctions.subscribeProfileStatisticsEvents(onEvent);
            if (simulationSubscriptionId != g_kInvalidSubscriptionId)
            {
                simulationSubscriptions.push_back({ simulation.first, simulationSubscriptionId });
            }
        }
    }

    if (simulationSubscriptions.empty())
    {
        return g_kInvalidSubscriptionId;
    }

    return g_benchmarkSubscriptions.add(std::move(simulationSubscriptions));
}

// Unsubscribes to simulation events.
void unsubscribeProfileStatisticsEvents(SubscriptionId subscriptionId)
{
    if (subscriptionId == g_kInvalidSubscriptionId)
    {
        return;
    }

    std::vector<details::SimulationSubscription> simulationSubscriptions =
        g_benchmarkSubscriptions.remove(subscriptionId);

    if (simulationSubscriptions.empty())
    {
        return;
    }

    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();

    for (const details::SimulationSubscription& simulationSubscription : simulationSubscriptions)
    {
        auto simulationIterator = simulations.find(simulationSubscription.simulationId);
        if (simulationIterator != simulations.end() &&
            simulationIterator->second.simulation.benchmarkFunctions.unsubscribeProfileStatisticsEvents)
        {
            simulationIterator->second.simulation.benchmarkFunctions.unsubscribeProfileStatisticsEvents(
                simulationSubscription.subscriptionId);
        }
    }
}

} // namespace manager
} // namespace physics
} // namespace isaacsim
