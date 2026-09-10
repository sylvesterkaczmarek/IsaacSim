// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../Types.hpp"
#include "Simulation.hpp"

#include <functional>
#include <string>
#include <vector>

namespace isaacsim
{
namespace physics
{
namespace registration
{


/**
 * @brief Timing statistics for a profiled physics zone.
 */
struct PhysicsProfileStatistics
{
    /** @brief The name of the profiled zone. */
    std::string zoneName;

    /** @brief The elapsed time, in milliseconds. */
    float elapsedMilliseconds;
};

/**
 * @brief Callback invoked when physics profiling statistics are available.
 *
 * @param[in] profileStatistics The profiling statistics reported for the event.
 */
using ProfileStatisticsNotificationFunction =
    std::function<void(const std::vector<PhysicsProfileStatistics>& profileStatistics)>;

/**
 * @brief Subscribes to physics profiling events.
 *
 * @param[in] onEvent The callback to invoke when profiling statistics are available.
 * @return A subscription identifier for use with @c UnsubscribeProfileStatisticsEventsFunction, or
 *         @c g_kInvalidSubscriptionId if the subscription fails.
 *
 * @note Profiling-event subscriptions must not be changed from within @p onEvent.
 */
using SubscribeProfileStatisticsEventsFunction =
    std::function<SubscriptionId(ProfileStatisticsNotificationFunction onEvent)>;

/**
 * @brief Unsubscribes from physics profiling events.
 *
 * @param[in] subscriptionId The identifier returned by @c SubscribeProfileStatisticsEventsFunction.
 *
 * @note Profiling-event subscriptions must not be changed from within a profiling callback.
 */
using UnsubscribeProfileStatisticsEventsFunction = std::function<void(SubscriptionId subscriptionId)>;

/**
 * @brief Function table for physics profiling operations.
 */
struct BenchmarkFunctions
{
    /** @brief Subscribes to profiling-statistics events. */
    SubscribeProfileStatisticsEventsFunction subscribeProfileStatisticsEvents{};

    /** @brief Unsubscribes from profiling-statistics events. */
    UnsubscribeProfileStatisticsEventsFunction unsubscribeProfileStatisticsEvents{};
};

} // namespace registration
} // namespace physics
} // namespace isaacsim
