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

#include <isaacsim/core/includes/BaseResetNode.hpp>
#include <isaacsim/zmq/core/ZmqPublishSocket.hpp>

#include <cstring>
#include <memory>
#include <string>

namespace isaacsim
{
namespace zmq
{
namespace nodes
{

/**
 * @class ZmqPublishNode
 * @brief Base class for ZMQ OmniGraph publish nodes.
 * @details
 * Mirrors UcxNode from isaacsim.ucx.nodes.
 * Owns a ZmqPublishSocket — one PUSH socket per node. These are connect sockets, so multiple
 * nodes targeting the same endpoint is fine; reset() closes this node's socket.
 */
class ZmqPublishNode : public isaacsim::core::includes::BaseResetNode
{
public:
    ZmqPublishNode() = default;

    virtual ~ZmqPublishNode()
    {
        reset();
    }

    /**
     * @brief Resets the node: releases socket reference.
     */
    virtual void reset() override
    {
        m_socket.reset();
    }

protected:
    /**
     * @brief Get or create socket for ip:port; re-initialize if parameters changed.
     * @tparam DatabaseT OGN database type (for logging)
     * @param db   OGN database
     * @param ip   Remote IP string
     * @param port Remote port number
     * @return true on success
     */
    template <typename DatabaseT>
    bool ensureSocketReady(DatabaseT& db, const std::string& ip, uint16_t port)
    {
        if (m_socket && m_socket->getIp() == ip && m_socket->getPort() == port)
        {
            return true;
        }

        // Recreate the socket (first call, or ip/port changed).
        m_socket.reset();
        try
        {
            m_socket = std::make_shared<isaacsim::zmq::core::ZmqPublishSocket>(ip, port);
            return m_socket != nullptr;
        }
        catch (const std::exception& e)
        {
            db.logError("ZmqPublishNode: failed to create socket for %s:%u: %s", ip.c_str(), port, e.what());
            return false;
        }
    }

    /**
     * @brief Serialize proto and send via socket.
     * @tparam DatabaseT OGN database type
     * @tparam ProtoT    Protobuf message type
     * @param db    OGN database
     * @param topic Topic string constant (e.g. kZmqTopicClock)
     * @param proto Protobuf message to serialize and send
     * @return true on success
     */
    template <typename DatabaseT, typename ProtoT>
    bool publishProto(DatabaseT& db, const std::string& topic, const ProtoT& proto)
    {
        if (!m_socket)
        {
            db.logError("ZmqPublishNode::publishProto: socket not initialized");
            return false;
        }
        std::string serialized;
        if (!proto.SerializeToString(&serialized))
        {
            db.logError("ZmqPublishNode::publishProto: failed to serialize proto message");
            return false;
        }
        return m_socket->sendMultipart(topic, serialized);
    }

    std::shared_ptr<isaacsim::zmq::core::ZmqPublishSocket> m_socket; //!< Shared PUSH socket
};

} // namespace nodes
} // namespace zmq
} // namespace isaacsim
