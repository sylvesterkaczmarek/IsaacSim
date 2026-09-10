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

#include <OgnZMQSubscribeJointCommandDatabase.h>
#include <joint_command.pb.h>

class OgnZMQSubscribeJointCommand : public isaacsim::zmq::nodes::ZmqSubscribeNode
{
public:
    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state =
            OgnZMQSubscribeJointCommandDatabase::sPerInstanceState<OgnZMQSubscribeJointCommand>(nodeObj, instanceId);
        state.reset();
    }

    static bool compute(OgnZMQSubscribeJointCommandDatabase& db)
    {
        auto& state = db.template perInstanceState<OgnZMQSubscribeJointCommand>();

        isaacsim::zmq::JointCommand proto;
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

        // Populate joint names
        auto& jointNames = db.outputs.jointNames();
        jointNames.resize(proto.joint_names_size());
        for (int i = 0; i < proto.joint_names_size(); ++i)
        {
            jointNames[i] = db.stringToToken(proto.joint_names(i).c_str());
        }

        // Populate positions
        auto& positions = db.outputs.positions();
        positions.resize(proto.positions_size());
        for (int i = 0; i < proto.positions_size(); ++i)
        {
            positions[i] = proto.positions(i);
        }

        // Populate velocities
        auto& velocities = db.outputs.velocities();
        velocities.resize(proto.velocities_size());
        for (int i = 0; i < proto.velocities_size(); ++i)
        {
            velocities[i] = proto.velocities(i);
        }

        db.outputs.execOut() = kExecutionAttributeStateEnabled;
        return true;
    }
};

REGISTER_OGN_NODE()
