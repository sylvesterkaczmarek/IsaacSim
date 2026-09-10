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

#include "isaacsim/core/includes/UsdUtilities.hpp"

#include <isaacsim/ros2/core/Ros2Node.hpp>
#include <isaacsim/ros2/nodes/Ros2OgnUtils.hpp>
#include <omni/fabric/FabricUSD.h>

#include <OgnROS2ServiceServerRequestDatabase.h>

using namespace isaacsim::ros2::core;

class OgnROS2ServiceServerRequest : public Ros2Node
{
public:
    static void initInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state =
            OgnROS2ServiceServerRequestDatabase::sPerInstanceState<OgnROS2ServiceServerRequest>(nodeObj, instanceId);
        state.m_nodeObj = nodeObj;
        // Register change event for message type
        AttributeObj attrMessagePackageObj = nodeObj.iNode->getAttribute(nodeObj, "inputs:messagePackage");
        AttributeObj attrMessageSubfolderObj = nodeObj.iNode->getAttribute(nodeObj, "inputs:messageSubfolder");
        AttributeObj attrMessageNameObj = nodeObj.iNode->getAttribute(nodeObj, "inputs:messageName");
        attrMessagePackageObj.iAttribute->registerValueChangedCallback(attrMessagePackageObj, onPackageChanged, true);
        attrMessageSubfolderObj.iAttribute->registerValueChangedCallback(attrMessageSubfolderObj, onPackageChanged, true);
        attrMessageNameObj.iAttribute->registerValueChangedCallback(attrMessageNameObj, onPackageChanged, true);
    }

    static bool compute(OgnROS2ServiceServerRequestDatabase& db)
    {
        const GraphContextObj& context = db.abi_context();
        auto& state = db.perInstanceState<OgnROS2ServiceServerRequest>();
        const auto& nodeObj = db.abi_node();

        // Spin once calls reset automatically if it was not successful
        if (!state.isInitialized())
        {
            // Find our stage
            long stageId = context.iContext->getStageId(context);
            auto stage = pxr::UsdUtilsStageCache::Get().Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

            if (!state.initializeNodeHandle(
                    std::string(nodeObj.iNode->getPrimPath(nodeObj)),
                    collectNamespace(db.inputs.nodeNamespace(),
                                     stage->GetPrimAtPath(pxr::SdfPath(nodeObj.iNode->getPrimPath(nodeObj)))),
                    db.inputs.context()))
            {
                db.logError("Unable to create ROS2 node, please check that namespace is valid");
                return false;
            }
        }

        auto messagePackage = std::string(db.inputs.messagePackage());
        auto messageSubfolder = std::string(db.inputs.messageSubfolder());
        auto messageName = std::string(db.inputs.messageName());
        if (messagePackage.empty() || messageSubfolder.empty() || messageName.empty())
        {
            db.logWarning("messagePackage [%s] or messageSubfolder [%s] or messageName [%s] empty, skipping compute",
                          messagePackage.c_str(), messageSubfolder.c_str(), messageName.c_str());
            return false;
        }
        if (messagePackage != state.m_messagePackage)
        {
            state.m_messageUpdateNeeded = true;
            state.m_messagePackage = messagePackage;
        }
        if (messageSubfolder != state.m_messageSubfolder)
        {
            state.m_messageUpdateNeeded = true;
            state.m_messageSubfolder = messageSubfolder;
        }
        if (messageName != state.m_messageName)
        {
            state.m_messageUpdateNeeded = true;
            state.m_messageName = messageName;
        }
        // Update message and node attributes
        if (state.m_messageUpdateNeeded)
        {
            state.m_messageRequest = state.m_factory->createDynamicMessage(
                state.m_messagePackage, state.m_messageSubfolder, state.m_messageName, BackendMessageType::eRequest);
            if (!isaacsim::ros2::omnigraph_utils::createOgAttributesForMessage<OgnROS2ServiceServerRequestDatabase, true>(
                    db, nodeObj, state.m_messagePackage, state.m_messageSubfolder, state.m_messageName,
                    state.m_messageRequest, "Request:"))
            {
                if (state.m_serverHandle != 0 && state.m_ros2Bridge != nullptr)
                {
                    state.m_ros2Bridge->removeHandle(state.m_serverHandle);
                    state.m_serverHandle = 0;
                    db.outputs.serverHandle() = 0;
                }
                state.m_serviceServer.reset();
                state.m_messageRequest.reset();
                return false;
            }
            state.m_messageUpdateNeeded = false;
        }

        std::string qosProfile = std::string(db.inputs.qosProfile());
        std::string serviceName = std::string(db.inputs.serviceName());
        if (qosProfile != state.m_qosProfile)
        {
            state.m_qosProfile = qosProfile;
            state.m_serviceUpdateNeeded = true;
        }
        if (serviceName != state.m_serviceName)
        {
            state.m_serviceName = serviceName;
            state.m_serviceUpdateNeeded = true;
        }

        if (!state.m_serviceServer || state.m_serviceUpdateNeeded)
        {
            // Setup ROS ServiceServer
            const std::string& currentServiceName = db.inputs.serviceName();
            std::string fullServiceName = addTopicPrefix(db.inputs.nodeNamespace(), currentServiceName);
            if (!state.m_factory->validateTopicName(fullServiceName))
            {
                db.logWarning("No Valid Topic : %s", fullServiceName.c_str());
                return false;
            }

            Ros2QoSProfile qos;
            if (qosProfile != "")
            {
                if (!jsonToRos2QoSProfile(qos, state.m_qosProfile))
                {
                    db.logWarning("No qos");
                    return false;
                }
            }

            CARB_LOG_INFO("Creating server for topic name %s", fullServiceName.c_str());
            if (!state.m_messageRequest)
            {
                db.logWarning("%s/%s/%s request message is invalid", state.m_messagePackage.c_str(),
                              state.m_messageSubfolder.c_str(), state.m_messageName.c_str());
                db.outputs.serverHandle() = 0;
                state.m_serviceUpdateNeeded = false;
                return false;
            }
            const void* typeSupport = state.m_messageRequest->getTypeSupportHandle();
            if (!typeSupport)
            {
                db.logWarning("%s/%s/%s service type support is unavailable", state.m_messagePackage.c_str(),
                              state.m_messageSubfolder.c_str(), state.m_messageName.c_str());
                db.outputs.serverHandle() = 0;
                state.m_serviceUpdateNeeded = false;
                return false;
            }

            state.m_serviceServer =
                state.m_factory->createService(state.m_nodeHandle.get(), fullServiceName.c_str(), typeSupport, qos);
            if (!state.m_serviceServer || !state.m_serviceServer->isValid())
            {
                db.logWarning("Unable to create ROS 2 service server");
                state.m_serviceServer.reset();
                db.outputs.serverHandle() = 0;
                state.m_serviceUpdateNeeded = false;
                return false;
            }

            state.m_serverHandle = state.m_ros2Bridge->addHandle(&state.m_serviceServer);
            db.outputs.serverHandle() = state.m_serverHandle;
            state.m_serviceUpdateNeeded = false;
        }

        return state.serviceServer(db, context);
    }

    bool serviceServer(OgnROS2ServiceServerRequestDatabase& db, const GraphContextObj& context)
    {
        auto& state = db.perInstanceState<OgnROS2ServiceServerRequest>();
        db.outputs.onReceived() = kExecutionAttributeStateDisabled;
        if (!state.m_serviceServer)
        {
            db.logWarning("service is invalid");
            return false;
        }
        if (!state.m_messageRequest)
        {
            db.logWarning("Request message is invalid");
            return false;
        }
        if (!state.m_serviceServer->isValid())
        {
            db.logWarning("service is invalid");
            return false;
        }
        if (state.m_serviceServer->takeRequest(state.m_messageRequest->getPtr()))
        {
            // Write incoming request data field/data to output
            isaacsim::ros2::omnigraph_utils::writeNodeAttributeFromMessage(db, state.m_messageRequest, "Request:", true);
            // Only if the server received  a request
            db.outputs.onReceived() = kExecutionAttributeStateEnabled;
        }
        return true;
    }

    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state =
            OgnROS2ServiceServerRequestDatabase::sPerInstanceState<OgnROS2ServiceServerRequest>(nodeObj, instanceId);
        state.reset();
    }

    virtual void reset()
    {
        if (m_serverHandle != 0 && m_ros2Bridge != nullptr)
        {
            m_ros2Bridge->removeHandle(m_serverHandle);
            m_serverHandle = 0;
        }
        m_serviceServer.reset(); // This should be reset before we reset the handle.
        m_messagePackage.clear();
        m_messageSubfolder.clear();
        m_messageName.clear();
        m_serviceName.clear();
        m_qosProfile.clear();
        Ros2Node::reset();
    }

private:
    std::shared_ptr<Ros2Service> m_serviceServer = nullptr;
    std::shared_ptr<Ros2Message> m_messageRequest = nullptr;

    bool m_serviceUpdateNeeded = true;
    bool m_messageUpdateNeeded = true;
    NodeObj m_nodeObj;
    uint64_t m_serverHandle = 0;

    std::string m_messagePackage;
    std::string m_messageSubfolder;
    std::string m_messageName;
    std::string m_serviceName;
    std::string m_qosProfile;
    static void onPackageChanged(AttributeObj const& attrObj, void const* userData)
    {
        // Get message package, subfolder and name
        NodeObj nodeObj = attrObj.iAttribute->getNode(attrObj);
        auto db = OgnROS2ServiceServerRequestDatabase(nodeObj);
        auto& state = db.perInstanceState<OgnROS2ServiceServerRequest>();
        state.m_messageUpdateNeeded = true;
        if (!state.isInitialized())
        {
            return;
        }

        std::string messagePackage = std::string(db.inputs.messagePackage());
        std::string messageSubfolder = std::string(db.inputs.messageSubfolder());
        std::string messageName = std::string(db.inputs.messageName());

        if (!isaacsim::ros2::omnigraph_utils::removeDynamicAttributes<true, true>(nodeObj))
        {
            db.logError("Unable to remove existing attributes from the node");
            return;
        }

        if (messagePackage.empty() || messageSubfolder.empty() || messageName.empty())
        {
            db.logWarning("messagePackage [%s] or messageSubfolder [%s] or messageName [%s] empty, skipping compute",
                          messagePackage.c_str(), messageSubfolder.c_str(), messageName.c_str());
            return;
        }

        // Build message attributes
        state.m_messageRequest = state.m_factory->createDynamicMessage(
            messagePackage, messageSubfolder, messageName, BackendMessageType::eRequest);
        if (!isaacsim::ros2::omnigraph_utils::createOgAttributesForMessage<OgnROS2ServiceServerRequestDatabase, true, false>(
                db, nodeObj, messagePackage, messageSubfolder, messageName, state.m_messageRequest, "Request:"))
        {
            if (state.m_serverHandle != 0 && state.m_ros2Bridge != nullptr)
            {
                state.m_ros2Bridge->removeHandle(state.m_serverHandle);
                state.m_serverHandle = 0;
            }
            state.m_serviceServer.reset();
            state.m_messageRequest.reset();
            state.m_messageUpdateNeeded = true;
            return;
        }
        state.m_serviceUpdateNeeded = true;
    }
};

REGISTER_OGN_NODE()
