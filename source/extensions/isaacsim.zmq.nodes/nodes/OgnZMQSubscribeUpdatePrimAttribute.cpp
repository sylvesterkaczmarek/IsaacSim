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

// clang-format off
#include <pch/UsdPCH.hpp>
// clang-format on

#include <isaacsim/zmq/nodes/ZmqSubscribeNode.hpp>

#include <OgnZMQSubscribeUpdatePrimAttributeDatabase.h>
#include <update_prim_attribute.pb.h>

class OgnZMQSubscribeUpdatePrimAttribute : public isaacsim::zmq::nodes::ZmqSubscribeNode
{
public:
    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnZMQSubscribeUpdatePrimAttributeDatabase::sPerInstanceState<OgnZMQSubscribeUpdatePrimAttribute>(
            nodeObj, instanceId);
        state.reset();
    }

    static bool compute(OgnZMQSubscribeUpdatePrimAttributeDatabase& db)
    {
        auto& state = db.template perInstanceState<OgnZMQSubscribeUpdatePrimAttribute>();

        isaacsim::zmq::UpdatePrimAttribute proto;
        const std::string ip = db.inputs.ip();
        const uint16_t port = static_cast<uint16_t>(db.inputs.port());
        const std::string topicName = db.inputs.topicName();
        if (!state.ensureSocketReady(db, ip, port, topicName))
        {
            return false;
        }
        if (!state.tryReceiveProto(db, proto))
        {
            // No message this tick — not an error
            return true;
        }

        // Decode into outputs that feed an omni.graph.nodes.WritePrimAttribute (usePath=true).
        db.outputs.primPath() = db.stringToToken(proto.prim_path().c_str());
        db.outputs.attributeName() = db.stringToToken(proto.attribute().c_str());
        db.outputs.value() = proto.value();

        db.outputs.execOut() = kExecutionAttributeStateEnabled;
        return true;
    }
};

REGISTER_OGN_NODE()
