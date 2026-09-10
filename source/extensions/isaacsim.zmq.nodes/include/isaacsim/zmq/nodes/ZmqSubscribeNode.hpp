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
#include <isaacsim/zmq/core/ZmqSubscribeSocket.hpp>

#include <memory>
#include <string>

namespace isaacsim
{
namespace zmq
{
namespace nodes
{

/**
 * @class ZmqSubscribeNode
 * @brief Base class for ZMQ OmniGraph subscribe (inbound) nodes.
 * @details
 * Mirrors ZmqPublishNode but for SUB sockets.
 * Owns a ZmqSubscribeSocket — one per node; reset() closes it.
 * tryReceiveProto() performs a non-blocking recv and deserializes the payload.
 */
class ZmqSubscribeNode : public isaacsim::core::includes::BaseResetNode
{
public:
    ZmqSubscribeNode() = default;

    virtual ~ZmqSubscribeNode()
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
     * @brief Get or create subscribe socket for ip:port:topic; re-initialize if parameters changed.
     * @tparam DatabaseT OGN database type (for logging)
     * @param db    OGN database
     * @param ip    Remote server IP string
     * @param port  Remote server port number
     * @param topic Topic string to subscribe to
     * @return true on success
     */
    template <typename DatabaseT>
    bool ensureSocketReady(DatabaseT& db, const std::string& ip, uint16_t port, const std::string& topic)
    {
        if (m_socket && m_socket->getIp() == ip && m_socket->getPort() == port && m_socket->getTopic() == topic)
        {
            return true;
        }

        // Recreate the socket (first call, or ip/port/topic changed).
        m_socket.reset();
        try
        {
            m_socket = std::make_shared<isaacsim::zmq::core::ZmqSubscribeSocket>(ip, port, topic);
            return m_socket != nullptr;
        }
        catch (const std::exception& e)
        {
            db.logError("ZmqSubscribeNode: failed to create socket for %s:%u (topic: %s): %s", ip.c_str(), port,
                        topic.c_str(), e.what());
            return false;
        }
    }

    /**
     * @brief Attempt a non-blocking receive and deserialize into a proto message.
     * @tparam DatabaseT OGN database type
     * @tparam ProtoT    Protobuf message type
     * @param db    OGN database (for logging)
     * @param proto Protobuf message to fill on success
     * @return true if a message was received and parsed; false if no message available this tick
     */
    template <typename DatabaseT, typename ProtoT>
    bool tryReceiveProto(DatabaseT& db, ProtoT& proto)
    {
        if (!m_socket)
        {
            db.logError("ZmqSubscribeNode::tryReceiveProto: socket not initialized");
            return false;
        }
        std::string payload;
        if (!m_socket->tryRecv(payload))
        {
            return false;
        }
        if (!proto.ParseFromString(payload))
        {
            db.logWarning("ZmqSubscribeNode::tryReceiveProto: failed to parse proto message");
            return false;
        }
        return true;
    }

    std::shared_ptr<isaacsim::zmq::core::ZmqSubscribeSocket> m_socket; //!< Shared SUB socket
};

} // namespace nodes
} // namespace zmq
} // namespace isaacsim
