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

#include <carb/logging/Log.h>
#include <carb/settings/ISettings.h>

#include <isaacsim/ros2/nodes/TfAggregationManager.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <unordered_map>

using namespace isaacsim::ros2::core;

namespace isaacsim
{
namespace ros2
{
namespace nodes
{

namespace
{

constexpr uint64_t kWarningIntervalFlushCycles = 120;
constexpr double kSecondsToNanoseconds = 1e9;
constexpr char kTfAggregationEnabledSetting[] = "/exts/isaacsim.ros2.nodes/tfAggregation/enabled";

TfAggregationManager* g_tfAggregationManager = nullptr;

template <typename T>
void hashCombine(size_t& seed, const T& value)
{
    seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

struct DomainKey
{
    bool hasDomainId = false;
    size_t domainId = 0;
    Ros2ContextHandle* contextHandle = nullptr;

    bool operator==(const DomainKey& other) const
    {
        if (hasDomainId != other.hasDomainId)
        {
            return false;
        }
        return hasDomainId ? domainId == other.domainId : contextHandle == other.contextHandle;
    }
};

struct QoSProfileKey
{
    Ros2QoSHistoryPolicy history = Ros2QoSHistoryPolicy::eSystemDefault;
    size_t depth = 0;
    Ros2QoSReliabilityPolicy reliability = Ros2QoSReliabilityPolicy::eSystemDefault;
    Ros2QoSDurabilityPolicy durability = Ros2QoSDurabilityPolicy::eSystemDefault;
    uint64_t deadlineSec = 0;
    uint64_t deadlineNsec = 0;
    uint64_t lifespanSec = 0;
    uint64_t lifespanNsec = 0;
    Ros2QoSLivelinessPolicy liveliness = Ros2QoSLivelinessPolicy::eSystemDefault;
    uint64_t livelinessLeaseDurationSec = 0;
    uint64_t livelinessLeaseDurationNsec = 0;
    bool avoidRosNamespaceConventions = false;

    bool operator==(const QoSProfileKey& other) const
    {
        return history == other.history && depth == other.depth && reliability == other.reliability &&
               durability == other.durability && deadlineSec == other.deadlineSec && deadlineNsec == other.deadlineNsec &&
               lifespanSec == other.lifespanSec && lifespanNsec == other.lifespanNsec &&
               liveliness == other.liveliness && livelinessLeaseDurationSec == other.livelinessLeaseDurationSec &&
               livelinessLeaseDurationNsec == other.livelinessLeaseDurationNsec &&
               avoidRosNamespaceConventions == other.avoidRosNamespaceConventions;
    }
};

struct GroupKey
{
    DomainKey domain;
    std::string topicName;
    bool staticPublisher = false;
    QoSProfileKey qos;

    bool operator==(const GroupKey& other) const
    {
        return domain == other.domain && topicName == other.topicName && staticPublisher == other.staticPublisher &&
               qos == other.qos;
    }
};

struct GroupKeyHash
{
    size_t operator()(const GroupKey& key) const
    {
        size_t seed = 0;
        hashCombine(seed, key.domain.hasDomainId);
        if (key.domain.hasDomainId)
        {
            hashCombine(seed, key.domain.domainId);
        }
        else
        {
            hashCombine(seed, key.domain.contextHandle);
        }
        hashCombine(seed, key.topicName);
        hashCombine(seed, key.staticPublisher);
        hashCombine(seed, static_cast<int>(key.qos.history));
        hashCombine(seed, key.qos.depth);
        hashCombine(seed, static_cast<int>(key.qos.reliability));
        hashCombine(seed, static_cast<int>(key.qos.durability));
        hashCombine(seed, key.qos.deadlineSec);
        hashCombine(seed, key.qos.deadlineNsec);
        hashCombine(seed, key.qos.lifespanSec);
        hashCombine(seed, key.qos.lifespanNsec);
        hashCombine(seed, static_cast<int>(key.qos.liveliness));
        hashCombine(seed, key.qos.livelinessLeaseDurationSec);
        hashCombine(seed, key.qos.livelinessLeaseDurationNsec);
        hashCombine(seed, key.qos.avoidRosNamespaceConventions);
        return seed;
    }
};

DomainKey rosDomainKey(Ros2ContextHandle* contextHandle)
{
    constexpr size_t kInvalidDomainId = (std::numeric_limits<size_t>::max)();
    if (contextHandle)
    {
        const size_t domainId = contextHandle->getDomainId();
        if (domainId != kInvalidDomainId)
        {
            return { true, domainId, nullptr };
        }
    }

    return { false, 0, contextHandle };
}

QoSProfileKey qosProfileKey(const Ros2QoSProfile& qos)
{
    return { qos.history,
             qos.depth,
             qos.reliability,
             qos.durability,
             qos.deadline.sec,
             qos.deadline.nsec,
             qos.lifespan.sec,
             qos.lifespan.nsec,
             qos.liveliness,
             qos.livelinessLeaseDuration.sec,
             qos.livelinessLeaseDuration.nsec,
             qos.avoidRosNamespaceConventions };
}

GroupKey groupKey(Ros2ContextHandle* contextHandle,
                  const std::string& topicName,
                  bool staticPublisher,
                  const Ros2QoSProfile& qos)
{
    return { rosDomainKey(contextHandle), topicName, staticPublisher, qosProfileKey(qos) };
}

int64_t timestampKey(double timeStamp)
{
    return static_cast<int64_t>(std::llround(timeStamp * kSecondsToNanoseconds));
}

bool transformEqual(const TfTransformStamped& lhs, const TfTransformStamped& rhs)
{
    return lhs.parentFrame == rhs.parentFrame && lhs.childFrame == rhs.childFrame &&
           lhs.translationX == rhs.translationX && lhs.translationY == rhs.translationY &&
           lhs.translationZ == rhs.translationZ && lhs.rotationX == rhs.rotationX && lhs.rotationY == rhs.rotationY &&
           lhs.rotationZ == rhs.rotationZ && lhs.rotationW == rhs.rotationW;
}

bool transformsEqual(const std::vector<TfTransformStamped>& lhs, const std::vector<TfTransformStamped>& rhs)
{
    if (lhs.size() != rhs.size())
    {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); i++)
    {
        if (!transformEqual(lhs[i], rhs[i]))
        {
            return false;
        }
    }

    return true;
}

struct Submission
{
    double timeStamp = 0.0;
    std::vector<TfTransformStamped> transforms;
};

struct TimestampBatch
{
    double timeStamp = 0.0;
    std::unordered_map<uint64_t, Submission> submissions;
};

struct Contributor
{
    uint64_t id = 0;
    std::string name;
    GroupKey groupKey;
    std::shared_ptr<Ros2NodeHandle> nodeHandle;
    uint64_t requiredFlushCycle = 0;
    uint64_t lastMissingWarningFlushCycle = 0;
};

struct Group
{
    Ros2Factory* factory = nullptr;
    std::shared_ptr<Ros2NodeHandle> publisherNodeHandle;
    std::vector<std::shared_ptr<Ros2NodeHandle>> nodeHandles;
    std::shared_ptr<Ros2Publisher> publisher;
    std::shared_ptr<Ros2TfTreeMessage> message;
    std::string topicName;
    Ros2QoSProfile qos;
    bool staticPublisher = false;
    bool publishWithoutVerification = false;
    std::vector<uint64_t> contributorOrder;
    // Dynamic submissions form the current post-OmniGraph epoch; later submissions from the same contributor win.
    std::unordered_map<uint64_t, Submission> currentDynamicSubmissions;
    std::unordered_map<uint64_t, std::vector<TfTransformStamped>> staticSubmissions;
    bool staticDirty = false;
    uint64_t lastConflictWarningFlushCycle = 0;
};

} // namespace

struct TfAggregationManager::Impl
{
    std::mutex mutex;
    uint64_t nextContributorId = 1;
    uint64_t flushCycle = 0;
    std::unordered_map<uint64_t, Contributor> contributors;
    std::unordered_map<GroupKey, Group, GroupKeyHash> groups;

    void logMissingContributors(Group& group)
    {
        for (uint64_t contributorId : group.contributorOrder)
        {
            auto contributorIt = contributors.find(contributorId);
            if (contributorIt == contributors.end())
            {
                continue;
            }

            Contributor& contributor = contributorIt->second;
            const bool hasPendingSubmission =
                group.currentDynamicSubmissions.find(contributorId) != group.currentDynamicSubmissions.end();

            if (contributor.requiredFlushCycle > flushCycle || hasPendingSubmission)
            {
                continue;
            }

            if (contributor.lastMissingWarningFlushCycle == 0 ||
                flushCycle - contributor.lastMissingWarningFlushCycle >= kWarningIntervalFlushCycles)
            {
                CARB_LOG_WARN(
                    "ROS2 TF aggregation contributor '%s' did not submit transforms for topic '%s' on flush cycle %llu",
                    contributor.name.c_str(), group.topicName.c_str(), static_cast<unsigned long long>(flushCycle));
                contributor.lastMissingWarningFlushCycle = flushCycle;
            }
        }
    }

    std::vector<TfTransformStamped> combineTransforms(Group& group,
                                                      const std::unordered_map<uint64_t, Submission>& submissions)
    {
        std::vector<TfTransformStamped> transforms;
        std::unordered_map<std::string, uint64_t> childFrameOwners;

        for (uint64_t contributorId : group.contributorOrder)
        {
            auto submissionIt = submissions.find(contributorId);
            if (submissionIt == submissions.end())
            {
                continue;
            }

            const Submission& submission = submissionIt->second;
            for (const TfTransformStamped& transform : submission.transforms)
            {
                auto ownerIt = childFrameOwners.find(transform.childFrame);
                if (ownerIt != childFrameOwners.end() && ownerIt->second != contributorId)
                {
                    if (group.lastConflictWarningFlushCycle == 0 ||
                        flushCycle - group.lastConflictWarningFlushCycle >= kWarningIntervalFlushCycles)
                    {
                        CARB_LOG_WARN(
                            "ROS2 TF aggregation topic '%s' received duplicate child frame '%s' from multiple "
                            "contributors; keeping the first registered contributor",
                            group.topicName.c_str(), transform.childFrame.c_str());
                        group.lastConflictWarningFlushCycle = flushCycle;
                    }
                    continue;
                }

                childFrameOwners[transform.childFrame] = contributorId;
                transforms.push_back(transform);
            }
        }

        return transforms;
    }

    void publishDynamicBatch(Group& group, TimestampBatch& batch)
    {
        std::vector<TfTransformStamped> transforms = combineTransforms(group, batch.submissions);
        if (transforms.empty())
        {
            return;
        }

        group.message->writeData(batch.timeStamp, transforms);
        group.publisher->publish(group.message->getPtr());
    }

    void flushDynamicGroup(Group& group)
    {
        logMissingContributors(group);

        if (group.currentDynamicSubmissions.empty())
        {
            return;
        }

        if (!group.publishWithoutVerification && group.publisher && !group.publisher->getSubscriptionCount())
        {
            group.currentDynamicSubmissions.clear();
            return;
        }

        // Ros2TfTreeMessage carries one timestamp, so split a completed epoch by timestamp only at publish time.
        std::map<int64_t, TimestampBatch> timestampBatches;
        for (auto& item : group.currentDynamicSubmissions)
        {
            const int64_t key = timestampKey(item.second.timeStamp);
            TimestampBatch& batch = timestampBatches[key];
            batch.timeStamp = item.second.timeStamp;
            batch.submissions.emplace(item.first, std::move(item.second));
        }
        group.currentDynamicSubmissions.clear();

        for (auto& item : timestampBatches)
        {
            publishDynamicBatch(group, item.second);
        }
    }

    void flushStaticGroup(Group& group)
    {
        if (!group.staticDirty)
        {
            return;
        }

        std::unordered_map<uint64_t, Submission> submissions;
        double timeStamp = 0.0;
        bool haveTimestamp = false;
        for (const auto& item : group.staticSubmissions)
        {
            Submission submission;
            submission.transforms = item.second;
            if (!submission.transforms.empty())
            {
                submission.timeStamp = submission.transforms.front().timeStamp;
                if (!haveTimestamp)
                {
                    timeStamp = submission.timeStamp;
                    haveTimestamp = true;
                }
            }
            submissions[item.first] = std::move(submission);
        }

        std::vector<TfTransformStamped> transforms = combineTransforms(group, submissions);
        if (transforms.empty())
        {
            group.staticDirty = false;
            return;
        }

        group.message->writeData(timeStamp, transforms);
        group.publisher->publish(group.message->getPtr());
        group.staticDirty = false;
    }
};

TfAggregationManager::TfAggregationManager() : m_impl(std::make_unique<Impl>())
{
}

TfAggregationManager::~TfAggregationManager()
{
    reset();
}

TfAggregationManager::ContributorHandle TfAggregationManager::registerContributor(const std::string& contributorName,
                                                                                  Ros2Factory* factory,
                                                                                  std::shared_ptr<Ros2NodeHandle> nodeHandle,
                                                                                  Ros2ContextHandle* contextHandle,
                                                                                  const std::string& topicName,
                                                                                  bool staticPublisher,
                                                                                  const Ros2QoSProfile& qos,
                                                                                  bool publishWithoutVerification)
{
    if (!factory || !nodeHandle)
    {
        CARB_LOG_ERROR("ROS2 TF aggregation cannot register contributor '%s' without a factory and node handle",
                       contributorName.c_str());
        return {};
    }

    std::lock_guard<std::mutex> lock(m_impl->mutex);

    const GroupKey key = groupKey(contextHandle, topicName, staticPublisher, qos);
    auto groupIt = m_impl->groups.find(key);
    if (groupIt == m_impl->groups.end())
    {
        Group group;
        group.factory = factory;
        group.publisherNodeHandle = nodeHandle;
        group.nodeHandles.push_back(nodeHandle);
        group.topicName = topicName;
        group.qos = qos;
        group.staticPublisher = staticPublisher;
        group.publishWithoutVerification = publishWithoutVerification;
        group.message = factory->createTfTreeMessage();
        if (!group.message)
        {
            CARB_LOG_ERROR("ROS2 TF aggregation failed to create TF message for topic '%s'", topicName.c_str());
            return {};
        }

        group.publisher =
            factory->createPublisher(nodeHandle.get(), topicName.c_str(), group.message->getTypeSupportHandle(), qos);
        if (!group.publisher)
        {
            CARB_LOG_ERROR("ROS2 TF aggregation failed to create publisher for topic '%s'", topicName.c_str());
            return {};
        }

        groupIt = m_impl->groups.emplace(key, std::move(group)).first;
    }
    else
    {
        groupIt->second.nodeHandles.push_back(nodeHandle);
        groupIt->second.publishWithoutVerification =
            groupIt->second.publishWithoutVerification || publishWithoutVerification;
    }

    const uint64_t contributorId = m_impl->nextContributorId++;
    Contributor contributor;
    contributor.id = contributorId;
    contributor.name = contributorName;
    contributor.groupKey = key;
    contributor.nodeHandle = nodeHandle;
    contributor.requiredFlushCycle = m_impl->flushCycle + 1;
    m_impl->contributors.emplace(contributorId, std::move(contributor));
    groupIt->second.contributorOrder.push_back(contributorId);

    return { contributorId };
}

void TfAggregationManager::unregisterContributor(ContributorHandle handle)
{
    if (!handle)
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_impl->mutex);

    auto contributorIt = m_impl->contributors.find(handle.id);
    if (contributorIt == m_impl->contributors.end())
    {
        return;
    }

    const GroupKey key = contributorIt->second.groupKey;
    auto groupIt = m_impl->groups.find(key);
    if (groupIt != m_impl->groups.end())
    {
        Group& group = groupIt->second;
        const std::shared_ptr<Ros2NodeHandle> unregisteredNodeHandle = contributorIt->second.nodeHandle;
        const auto nodeHandleIt = std::find(group.nodeHandles.begin(), group.nodeHandles.end(), unregisteredNodeHandle);
        if (nodeHandleIt != group.nodeHandles.end())
        {
            group.nodeHandles.erase(nodeHandleIt);
        }
        group.contributorOrder.erase(std::remove(group.contributorOrder.begin(), group.contributorOrder.end(), handle.id),
                                     group.contributorOrder.end());
        group.currentDynamicSubmissions.erase(handle.id);
        if (group.staticSubmissions.erase(handle.id))
        {
            group.staticDirty = true;
        }

        if (group.contributorOrder.empty())
        {
            m_impl->groups.erase(groupIt);
        }
        else if (group.publisherNodeHandle == unregisteredNodeHandle)
        {
            if (group.nodeHandles.empty())
            {
                CARB_LOG_ERROR("ROS2 TF aggregation topic '%s' has contributors but no remaining node handles",
                               group.topicName.c_str());
            }
            else
            {
                std::shared_ptr<Ros2NodeHandle> replacementNodeHandle = group.nodeHandles.front();
                std::shared_ptr<Ros2Publisher> replacementPublisher =
                    group.factory->createPublisher(replacementNodeHandle.get(), group.topicName.c_str(),
                                                   group.message->getTypeSupportHandle(), group.qos);
                if (!replacementPublisher)
                {
                    CARB_LOG_ERROR("ROS2 TF aggregation failed to recreate publisher for topic '%s' after unregister",
                                   group.topicName.c_str());
                }
                else
                {
                    group.publisher.reset();
                    group.publisherNodeHandle = replacementNodeHandle;
                    group.publisher = std::move(replacementPublisher);
                }
            }
        }
    }

    m_impl->contributors.erase(contributorIt);
}

bool TfAggregationManager::submit(ContributorHandle handle,
                                  double timeStamp,
                                  const std::vector<TfTransformStamped>& transforms)
{
    if (!handle)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_impl->mutex);

    auto contributorIt = m_impl->contributors.find(handle.id);
    if (contributorIt == m_impl->contributors.end())
    {
        return false;
    }

    auto groupIt = m_impl->groups.find(contributorIt->second.groupKey);
    if (groupIt == m_impl->groups.end())
    {
        return false;
    }

    Group& group = groupIt->second;
    if (group.staticPublisher)
    {
        auto staticIt = group.staticSubmissions.find(handle.id);
        if (staticIt == group.staticSubmissions.end() || !transformsEqual(staticIt->second, transforms))
        {
            group.staticSubmissions[handle.id] = transforms;
            group.staticDirty = true;
        }
        return true;
    }

    Submission submission;
    submission.timeStamp = timeStamp;
    submission.transforms = transforms;
    group.currentDynamicSubmissions[handle.id] = std::move(submission);
    return true;
}

void TfAggregationManager::flushAll()
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);

    for (auto& item : m_impl->groups)
    {
        Group& group = item.second;
        if (group.staticPublisher)
        {
            m_impl->flushStaticGroup(group);
        }
        else
        {
            m_impl->flushDynamicGroup(group);
        }
    }

    m_impl->flushCycle++;
}

void TfAggregationManager::reset()
{
    std::lock_guard<std::mutex> lock(m_impl->mutex);
    m_impl->contributors.clear();
    m_impl->groups.clear();
    m_impl->flushCycle = 0;
    m_impl->nextContributorId = 1;
}

TfAggregationManager* getTfAggregationManager()
{
    return g_tfAggregationManager;
}

TfAggregationManager* getEnabledTfAggregationManager()
{
    auto* settings = carb::getCachedInterface<carb::settings::ISettings>();
    if (!settings || !settings->getAsBool(kTfAggregationEnabledSetting))
    {
        return nullptr;
    }
    return getTfAggregationManager();
}

void setTfAggregationManager(TfAggregationManager* manager)
{
    g_tfAggregationManager = manager;
}

} // namespace nodes
} // namespace ros2
} // namespace isaacsim
