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

#include <mutex>
#include <string>
#include <zmq.hpp>

namespace isaacsim
{
namespace zmq
{
namespace core
{

/**
 * @class ZmqPublishSocket
 * @brief PUSH socket wrapper with HWM=1, LINGER=0, non-blocking multipart send.
 * @details
 * Wraps a ZMQ PUSH socket connected to a remote tcp://ip:port endpoint.
 * sendMultipart() sends [topic_bytes, serialized_proto] as a two-frame multipart.
 * The socket is non-blocking; if HWM is reached the message is dropped.
 *
 * Thread-safety: sendMultipart() is serialized by an internal mutex. ZeroMQ sockets are
 * not thread-safe and are thread-affine, so a mutex is required for the shared_ptr exposed
 * to Python to be safely callable from more than one thread.
 */
class ZmqPublishSocket
{
public:
    /**
     * @brief Construct and connect a PUSH socket.
     * @param ip   Remote IP address
     * @param port Remote port number
     */
    ZmqPublishSocket(const std::string& ip, uint16_t port);

    ~ZmqPublishSocket();

    /**
     * @brief Send a two-frame multipart message [topic, payload].
     * @param topic   Topic string (e.g. kZmqTopicClock)
     * @param payload Serialized protobuf bytes
     * @return true on success, false if dropped (HWM) or socket error
     */
    bool sendMultipart(const std::string& topic, const std::string& payload);

    const std::string& getIp() const
    {
        return m_ip;
    }
    uint16_t getPort() const
    {
        return m_port;
    }

private:
    std::string m_ip;
    uint16_t m_port;
    ::zmq::socket_t m_socket;
    std::mutex m_mutex; //!< Serializes access to m_socket (ZMQ sockets are not thread-safe)
};

} // namespace core
} // namespace zmq
} // namespace isaacsim
