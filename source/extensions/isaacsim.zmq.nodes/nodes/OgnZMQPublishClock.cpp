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

#include <isaacsim/zmq/nodes/ZmqPublishNode.hpp>

#include <OgnZMQPublishClockDatabase.h>
#include <clock.pb.h>

class OgnZMQPublishClock : public isaacsim::zmq::nodes::ZmqPublishNode
{
public:
    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnZMQPublishClockDatabase::sPerInstanceState<OgnZMQPublishClock>(nodeObj, instanceId);
        state.reset();
    }

    static bool compute(OgnZMQPublishClockDatabase& db)
    {
        auto& state = db.template perInstanceState<OgnZMQPublishClock>();

        isaacsim::zmq::Clock proto;
        proto.set_sim_dt(db.inputs.deltaSimulationTime());
        proto.set_sys_dt(db.inputs.deltaSystemTime());
        proto.set_sim_time(db.inputs.simulationTime());
        proto.set_sys_time(db.inputs.systemTime());

        const std::string ip = db.inputs.ip();
        const uint16_t port = static_cast<uint16_t>(db.inputs.port());
        const std::string topicName = db.inputs.topicName();
        if (!state.ensureSocketReady(db, ip, port))
        {
            return false;
        }
        if (!state.publishProto(db, topicName, proto))
        {
            return false;
        }

        db.outputs.execOut() = kExecutionAttributeStateEnabled;
        return true;
    }
};

REGISTER_OGN_NODE()
