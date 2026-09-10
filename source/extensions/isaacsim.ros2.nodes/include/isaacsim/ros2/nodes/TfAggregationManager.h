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

#include <isaacsim/ros2/core/Ros2Factory.hpp>
#include <isaacsim/ros2/core/Ros2QoS.hpp>
#include <isaacsim/ros2/core/Ros2Types.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace isaacsim
{
namespace ros2
{
namespace nodes
{

/**
 * @brief Aggregates TF submissions from multiple OGN publisher nodes into one ROS 2 TF publisher per topic group.
 *
 * Topic groups are keyed by ROS context, topic name, static/dynamic mode, and QoS profile. Dynamic groups are flushed
 * by the plugin's stage update node after OmniGraph evaluation. Manual OmniGraph evaluation callers can explicitly call
 * flushAll() if they need deterministic draining outside normal Kit playback.
 */
class TfAggregationManager
{
public:
    struct ContributorHandle
    {
        uint64_t id = 0;

        explicit operator bool() const
        {
            return id != 0;
        }
    };

    TfAggregationManager();
    ~TfAggregationManager();

    TfAggregationManager(const TfAggregationManager&) = delete;
    TfAggregationManager& operator=(const TfAggregationManager&) = delete;

    ContributorHandle registerContributor(const std::string& contributorName,
                                          isaacsim::ros2::core::Ros2Factory* factory,
                                          std::shared_ptr<isaacsim::ros2::core::Ros2NodeHandle> nodeHandle,
                                          isaacsim::ros2::core::Ros2ContextHandle* contextHandle,
                                          const std::string& topicName,
                                          bool staticPublisher,
                                          const isaacsim::ros2::core::Ros2QoSProfile& qos,
                                          bool publishWithoutVerification);

    void unregisterContributor(ContributorHandle handle);

    bool submit(ContributorHandle handle,
                double timeStamp,
                const std::vector<isaacsim::ros2::core::TfTransformStamped>& transforms);

    void flushAll();
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

TfAggregationManager* getTfAggregationManager();
TfAggregationManager* getEnabledTfAggregationManager();
void setTfAggregationManager(TfAggregationManager* manager);

} // namespace nodes
} // namespace ros2
} // namespace isaacsim
