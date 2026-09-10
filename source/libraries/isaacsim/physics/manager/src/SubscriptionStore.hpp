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

#pragma once

#include <isaacsim/physics/registration/simulator/Simulation.hpp>

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace isaacsim
{
namespace physics
{
namespace manager
{
namespace details
{

struct SimulationSubscription
{
    registration::SimulationId simulationId;
    registration::SubscriptionId subscriptionId;
};

class SubscriptionStore
{
public:
    registration::SubscriptionId add(std::vector<SimulationSubscription>&& subscriptions)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const registration::SubscriptionId id = m_nextId;
        m_nextId = registration::SubscriptionId(m_nextId.id + 1);
        m_subscriptions.emplace(id, std::move(subscriptions));
        return id;
    }

    std::vector<SimulationSubscription> remove(registration::SubscriptionId id)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto iterator = m_subscriptions.find(id);
        if (iterator == m_subscriptions.end())
        {
            return {};
        }
        std::vector<SimulationSubscription> subscriptions = std::move(iterator->second);
        m_subscriptions.erase(iterator);
        return subscriptions;
    }

private:
    std::mutex m_mutex;
    registration::SubscriptionId m_nextId{ 1 };
    std::unordered_map<registration::SubscriptionId, std::vector<SimulationSubscription>, registration::SubscriptionIdHash>
        m_subscriptions;
};

} // namespace details
} // namespace manager
} // namespace physics
} // namespace isaacsim
