// SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <isaacsim/physics/manager/Export.h>
#include <isaacsim/physics/registration/Types.hpp>
#include <isaacsim/physics/registration/simulator/Benchmark.hpp>

namespace isaacsim
{
namespace physics
{
namespace manager
{

/**
 * @brief Subscribes to physics simulation profiling events.
 *
 * The callback receives profiling statistics from each backend that supports profiling.
 *
 * @param[in] onEvent Callback invoked when profiling statistics are available.
 * @return A subscription identifier for use with `unsubscribeProfileStatisticsEvents()`, or
 *         `registration::g_kInvalidSubscriptionId` if no backend accepted the subscription.
 *
 * @note Do not modify profiling subscriptions from within `onEvent`.
 * @note Exceptions raised by a backend subscription callback propagate to the caller. If an exception occurs after
 *       another backend accepted the subscription, that earlier backend subscription may remain active without an
 *       aggregate subscription identifier being returned.
 */
ISAACSIM_PHYSICS_MANAGER_API registration::SubscriptionId subscribeProfileStatisticsEvents(
    registration::ProfileStatisticsNotificationFunction onEvent);

/**
 * @brief Unsubscribes from physics simulation profiling events.
 *
 * An invalid or unknown subscription identifier is ignored.
 *
 * @param[in] subscriptionId Subscription identifier returned by `subscribeProfileStatisticsEvents()`.
 *
 * @note Do not modify profiling subscriptions from within a profiling callback.
 * @note Exceptions raised by a backend unsubscription callback propagate to the caller and prevent later backends from
 *       being unsubscribed during that call.
 */
ISAACSIM_PHYSICS_MANAGER_API void unsubscribeProfileStatisticsEvents(registration::SubscriptionId subscriptionId);
} // namespace manager
} // namespace physics
} // namespace isaacsim
