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

#include <OgnZMQPublishPoseDatabase.h>
#include <pose.pb.h>

class OgnZMQPublishPose : public isaacsim::zmq::nodes::ZmqPublishNode
{
public:
    static void releaseInstance(NodeObj const& nodeObj, GraphInstanceID instanceId)
    {
        auto& state = OgnZMQPublishPoseDatabase::sPerInstanceState<OgnZMQPublishPose>(nodeObj, instanceId);
        state.reset();
    }

    static bool compute(OgnZMQPublishPoseDatabase& db)
    {
        auto& state = db.template perInstanceState<OgnZMQPublishPose>();

        const std::string ip = db.inputs.ip();
        const uint16_t port = static_cast<uint16_t>(db.inputs.port());
        const std::string topicName = db.inputs.topicName();

        if (!state.ensureSocketReady(db, ip, port))
        {
            return false;
        }

        isaacsim::zmq::Pose proto;
        proto.set_timestamp(db.inputs.timeStamp());

        // World position (x, y, z)
        const pxr::GfVec3d& position = db.inputs.position();
        proto.add_position(position[0]);
        proto.add_position(position[1]);
        proto.add_position(position[2]);

        // World orientation quaternion as (x, y, z, w) — GfQuatd stores imaginary (i,j,k) + real (w).
        const pxr::GfQuatd& orientation = db.inputs.orientation();
        const pxr::GfVec3d& imaginary = orientation.GetImaginary();
        proto.add_orientation(imaginary[0]);
        proto.add_orientation(imaginary[1]);
        proto.add_orientation(imaginary[2]);
        proto.add_orientation(orientation.GetReal());

        if (!state.publishProto(db, topicName, proto))
        {
            return false;
        }

        db.outputs.execOut() = kExecutionAttributeStateEnabled;
        return true;
    }
};

REGISTER_OGN_NODE()
