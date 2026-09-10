// SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <pch/UsdPCH.hpp>
// clang-format on

#include <isaacsim/ros2/core/Ros2Node.hpp>
#include <isaacsim/ros2/nodes/TfAggregationManager.h>

#include <OgnROS2PublishRawTransformTreeDatabase.h>
#include <vector>

using namespace isaacsim::ros2::core;

class OgnROS2PublishRawTransformTree : public Ros2Node
{
public:
    static void initInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnROS2PublishRawTransformTreeDatabase::sPerInstanceState<OgnROS2PublishRawTransformTree>(
            nodeObj, instanceId);

        state.m_firstIteration = true;
    }

    static bool compute(OgnROS2PublishRawTransformTreeDatabase& db)
    {
        auto& state = db.perInstanceState<OgnROS2PublishRawTransformTree>();

        // Spin once calls reset automatically if it was not successful
        const auto& nodeObj = db.abi_node();
        if (!state.isInitialized())
        {
            const GraphContextObj& context = db.abi_context();
            // Find our stage
            long stageId = context.iContext->getStageId(context);
            auto stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

            if (!state.initializeNodeHandle(
                    std::string(nodeObj.iNode->getPrimPath(nodeObj)),
                    collectNamespace(db.inputs.nodeNamespace(),
                                     stage->GetPrimAtPath(pxr::SdfPath(nodeObj.iNode->getPrimPath(nodeObj))), true),
                    db.inputs.context()))
            {
                db.logError("Unable to create ROS2 node, please check that namespace is valid");
                return false;
            }
        }

        // Either publisher was not valid, create a new direct publisher or aggregation contributor.
        if (!state.m_publisher && !state.m_tfAggregationHandle)
        {
            // Setup ROS TF publisher
            const std::string& topicName = db.inputs.topicName();
            std::string fullTopicName = addTopicPrefix(state.m_namespaceName, topicName);
            if (!state.m_factory->validateTopicName(fullTopicName))
            {
                db.logError("Unable to create ROS2 publisher, invalid topic name");
                return false;
            }

            Ros2QoSProfile qos;
            const std::string& qosProfile = db.inputs.qosProfile();
            if (db.inputs.staticPublisher())
            {
                qos.depth = 1;
                qos.durability = Ros2QoSDurabilityPolicy::eTransientLocal;
            }
            else if (qosProfile.empty())
            {
                qos.depth = db.inputs.queueSize();
            }
            else
            {
                if (!jsonToRos2QoSProfile(qos, qosProfile))
                {
                    return false;
                }
            }

            state.m_parentFrameId = db.inputs.parentFrameId();
            state.m_childFrameId = db.inputs.childFrameId();

            auto* tfAggregationManager = isaacsim::ros2::nodes::getEnabledTfAggregationManager();
            if (tfAggregationManager)
            {
                state.m_tfAggregationHandle = tfAggregationManager->registerContributor(
                    std::string(nodeObj.iNode->getPrimPath(nodeObj)), state.m_factory, state.m_nodeHandle,
                    state.m_contextHandle ? state.m_contextHandle->get() : nullptr, fullTopicName,
                    db.inputs.staticPublisher(), qos, state.m_publishWithoutVerification);
                if (!state.m_tfAggregationHandle)
                {
                    db.logError("Unable to register ROS2 TF aggregation contributor");
                    return false;
                }
                return true;
            }

            state.m_message = state.m_factory->createRawTfTreeMessage();
            state.m_publisher = state.m_factory->createPublisher(
                state.m_nodeHandle.get(), fullTopicName.c_str(), state.m_message->getTypeSupportHandle(), qos);

            return true;
        }

        return state.publishTF(db);
    }

    bool publishTF(OgnROS2PublishRawTransformTreeDatabase& db)
    {
        auto& state = db.perInstanceState<OgnROS2PublishRawTransformTree>();

        // If we're a static publisher we only publish once on the first iteration.
        // The message will persist as long as the simulation is playing.
        // If we're not a static publisher, we publish every tick only if
        // we have subscribers or m_publishWithoutVerification is true.
        const bool useAggregation = static_cast<bool>(state.m_tfAggregationHandle);
        bool isStaticPublisher = db.inputs.staticPublisher();
        if (!useAggregation && isStaticPublisher)
        {
            if (!state.m_firstIteration)
            {
                return false;
            }
            state.m_firstIteration = false;
        }
        else if (!useAggregation)
        {
            // Check if subscription count is 0
            if (!m_publishWithoutVerification && !state.m_publisher.get()->getSubscriptionCount())
            {
                return false;
            }
        }

        auto& translation = db.inputs.translation();
        auto& rotation = db.inputs.rotation();

        if (useAggregation)
        {
            TfTransformStamped transform;
            transform.timeStamp = db.inputs.timeStamp();
            transform.parentFrame = state.m_parentFrameId;
            transform.childFrame = state.m_childFrameId;
            transform.translationX = translation[0];
            transform.translationY = translation[1];
            transform.translationZ = translation[2];

            const pxr::GfVec3d& imaginary = rotation.GetImaginary();
            transform.rotationX = imaginary[0];
            transform.rotationY = imaginary[1];
            transform.rotationZ = imaginary[2];
            transform.rotationW = rotation.GetReal();

            auto* tfAggregationManager = isaacsim::ros2::nodes::getTfAggregationManager();
            if (!tfAggregationManager)
            {
                db.logError("ROS2 TF aggregation manager is unavailable");
                return false;
            }

            return tfAggregationManager->submit(state.m_tfAggregationHandle, transform.timeStamp, { transform });
        }

        state.m_message->writeData(
            db.inputs.timeStamp(), state.m_parentFrameId, state.m_childFrameId, translation, rotation);
        state.m_publisher.get()->publish(state.m_message->getPtr());

        return true;
    }

    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnROS2PublishRawTransformTreeDatabase::sPerInstanceState<OgnROS2PublishRawTransformTree>(
            nodeObj, instanceId);
        state.reset();
    }

    void reset() override
    {
        if (m_tfAggregationHandle)
        {
            if (auto* tfAggregationManager = isaacsim::ros2::nodes::getTfAggregationManager())
            {
                tfAggregationManager->unregisterContributor(m_tfAggregationHandle);
            }
            m_tfAggregationHandle = {};
        }
        m_publisher.reset(); // Publisher should be reset before we reset the handle.
        Ros2Node::reset();
        m_firstIteration = true;
    }

private:
    std::shared_ptr<Ros2Publisher> m_publisher = nullptr;
    std::shared_ptr<Ros2RawTfTreeMessage> m_message = nullptr;
    isaacsim::ros2::nodes::TfAggregationManager::ContributorHandle m_tfAggregationHandle;

    bool m_firstIteration = true;

    std::string m_parentFrameId = "odom";
    std::string m_childFrameId = "base_link";
};

REGISTER_OGN_NODE()
