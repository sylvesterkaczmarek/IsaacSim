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

#include <OgnROS2ServiceServerResponseDatabase.h>

using namespace isaacsim::ros2::core;

class OgnROS2ServiceServerResponse : public Ros2Node
{
public:
    static void initInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state =
            OgnROS2ServiceServerResponseDatabase::sPerInstanceState<OgnROS2ServiceServerResponse>(nodeObj, instanceId);
        state.m_nodeObj = nodeObj;
        AttributeObj attrMessagePackageObj = nodeObj.iNode->getAttribute(nodeObj, "inputs:messagePackage");
        AttributeObj attrMessageSubfolderObj = nodeObj.iNode->getAttribute(nodeObj, "inputs:messageSubfolder");
        AttributeObj attrMessageNameObj = nodeObj.iNode->getAttribute(nodeObj, "inputs:messageName");
        AttributeObj attrHandle = nodeObj.iNode->getAttribute(nodeObj, "inputs:serverHandle");
        attrMessagePackageObj.iAttribute->registerValueChangedCallback(attrMessagePackageObj, onPackageChanged, true);
        attrMessageSubfolderObj.iAttribute->registerValueChangedCallback(attrMessageSubfolderObj, onPackageChanged, true);
        attrMessageNameObj.iAttribute->registerValueChangedCallback(attrMessageNameObj, onPackageChanged, true);
        attrHandle.iAttribute->registerValueChangedCallback(attrHandle, onServiceChanged, true);
    }

    static bool compute(OgnROS2ServiceServerResponseDatabase& db)
    {
        const GraphContextObj& context = db.abi_context();
        auto& state = db.perInstanceState<OgnROS2ServiceServerResponse>();
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
        uint64_t serverHandle = db.inputs.serverHandle();

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
        if (serverHandle != state.m_serverHandle || !state.m_serviceServer)
        {
            if (db.inputs.serverHandle())
            {
                if (state.m_ros2Bridge == nullptr)
                {
                    return false;
                }
                void* voidPtr = state.m_ros2Bridge->getHandle(db.inputs.serverHandle());
                if (voidPtr == nullptr)
                {
                    return false;
                }
                state.m_messageUpdateNeeded = true;
                state.m_serverHandle = serverHandle;
                state.m_serviceServer = *reinterpret_cast<std::shared_ptr<Ros2Service>*>(voidPtr);
                if (!state.m_serviceServer)
                {
                    return false;
                }
            }
        }

        // Update message and node attributes
        if (state.m_messageUpdateNeeded)
        {
            if (!state.updateNodeState<false>(
                    db, nodeObj, state.m_messagePackage, state.m_messageSubfolder, state.m_messageName))
            {
                return false;
            }
            state.m_messageUpdateNeeded = false;
        }

        // ServiceServer was not valid, create a new one
        if (state.m_serviceServer)
        {
            return state.serviceServer(db, context);
        }
        return false;
    }


    bool serviceServer(OgnROS2ServiceServerResponseDatabase& db, const GraphContextObj& context)
    {
        auto& state = db.perInstanceState<OgnROS2ServiceServerResponse>();

        if (!state.m_serviceServer || !state.m_messageResponse)
        {
            return false;
        }
        // Check if all sub-message size match size of actuators before setting data
        if (!state.m_serviceServer->isValid())
        {
            db.logWarning("service is invalid");
            return false;
        }
        if (!state.m_messageResponse)
        {
            db.logWarning("Response message is invalid");
            return false;
        }
        // Write response of the node from the input to the message
        if (!isaacsim::ros2::omnigraph_utils::writeMessageDataFromNode(db, state.m_messageResponse, "Response:", false))
        {
            return false;
        }
        state.m_serviceServer->sendResponse(state.m_messageResponse->getPtr());

        // Only if the server received a request
        db.outputs.execOut() = kExecutionAttributeStateEnabled;
        return true;
    }

    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state =
            OgnROS2ServiceServerResponseDatabase::sPerInstanceState<OgnROS2ServiceServerResponse>(nodeObj, instanceId);
        state.reset();
    }

    virtual void reset()
    {
        m_serviceServer.reset(); // This should be reset before we reset the handle.
        Ros2Node::reset();
    }

private:
    std::shared_ptr<Ros2Service> m_serviceServer = nullptr;
    std::shared_ptr<Ros2Message> m_messageResponse = nullptr;

    bool m_messageUpdateNeeded = true;
    NodeObj m_nodeObj;
    uint64_t m_serverHandle = 0;
    std::string m_messagePackage;
    std::string m_messageSubfolder;
    std::string m_messageName;

    template <bool removeAttributes>
    bool updateNodeState(OgnROS2ServiceServerResponseDatabase& db,
                         const NodeObj& nodeObj,
                         std::string messagePackage,
                         std::string messageSubfolder,
                         std::string messageName)
    {
        CARB_LOG_WARN("service package information changed. Updating OgnROS2ServiceServerResponse node interface.");
        auto& state = db.perInstanceState<OgnROS2ServiceServerResponse>();

        if (removeAttributes)
        {
            if (!isaacsim::ros2::omnigraph_utils::removeDynamicAttributes<true, true>(nodeObj))
            {
                db.logError("Unable to remove existing attributes from the node");
                return false;
            }
        }

        if (messagePackage.empty() || messageSubfolder.empty() || messageName.empty())
        {
            db.logWarning("messagePackage [%s] or messageSubfolder [%s] or messageName [%s] empty, skipping compute",
                          messagePackage.c_str(), messageSubfolder.c_str(), messageName.c_str());
            return false;
        }

        // Build message attributes
        state.m_messageResponse = state.m_factory->createDynamicMessage(
            messagePackage, messageSubfolder, messageName, BackendMessageType::eResponse);
        if (!isaacsim::ros2::omnigraph_utils::createOgAttributesForMessage<OgnROS2ServiceServerResponseDatabase, false, false>(
                db, nodeObj, messagePackage, messageSubfolder, messageName, state.m_messageResponse, "Response:"))
        {
            state.m_messageResponse.reset();
            return false;
        }
        return true;
    }

    static void onPackageChanged(AttributeObj const& attrObj, void const* userData)
    {
        // Get message package, subfolder and name
        NodeObj nodeObj = attrObj.iAttribute->getNode(attrObj);
        auto db = OgnROS2ServiceServerResponseDatabase(nodeObj);
        auto& state = db.perInstanceState<OgnROS2ServiceServerResponse>();
        state.m_messageUpdateNeeded = true;
        if (!state.isInitialized())
        {
            return;
        }

        std::string messagePackage = std::string(db.inputs.messagePackage());
        std::string messageSubfolder = std::string(db.inputs.messageSubfolder());
        std::string messageName = std::string(db.inputs.messageName());
        if (!state.updateNodeState<true>(db, nodeObj, messagePackage, messageSubfolder, messageName))
        {
            state.m_messageResponse.reset();
        }
        state.m_messageUpdateNeeded = true;
    }

    static void onServiceChanged(AttributeObj const& attrObj, void const* userData)
    {
        // Get message package, subfolder and name
        NodeObj nodeObj = attrObj.iAttribute->getNode(attrObj);
        auto db = OgnROS2ServiceServerResponseDatabase(nodeObj);
        auto& state = db.perInstanceState<OgnROS2ServiceServerResponse>();
        uint64_t serverHandle = db.inputs.serverHandle();
        if (!state.isInitialized())
        {
            return;
        }

        if (serverHandle != state.m_serverHandle)
        {
            if (serverHandle)
            {
                if (state.m_ros2Bridge == nullptr)
                {
                    return;
                }
                void* voidPtr = state.m_ros2Bridge->getHandle(serverHandle);
                if (voidPtr == nullptr)
                {
                    return;
                }

                state.m_serviceServer = *reinterpret_cast<std::shared_ptr<Ros2Service>*>(voidPtr);
                if (!state.m_serviceServer)
                {
                    return;
                }
                state.m_serverHandle = serverHandle;
                if (!state.updateNodeState<true>(
                        db, nodeObj, state.m_messagePackage, state.m_messageSubfolder, state.m_messageName))
                {
                    state.m_messageResponse.reset();
                    state.m_messageUpdateNeeded = true;
                }
            }
        }
    }
};

REGISTER_OGN_NODE()
