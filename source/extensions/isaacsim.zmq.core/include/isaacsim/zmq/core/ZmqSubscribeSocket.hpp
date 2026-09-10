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
 * @class ZmqSubscribeSocket
 * @brief SUB socket that connects to a remote server and receives topic-filtered messages non-blocking.
 * @details
 * Connects to tcp://ip:port where the external server has bound a PUB socket.
 * Subscribes to a specific topic string; messages are two-frame multipart: [topic, payload].
 * tryRecv() uses ZMQ_DONTWAIT so it never blocks the simulation loop.
 *
 * Thread-safety: tryRecv() is serialized by an internal mutex. ZeroMQ sockets are not
 * thread-safe and are thread-affine, so a mutex is required for the shared_ptr exposed to
 * Python to be safely callable from more than one thread.
 */
class ZmqSubscribeSocket
{
public:
    /**
     * @brief Construct and connect a SUB socket to the remote server.
     * @param ip    Remote server IP address
     * @param port  Remote server port
     * @param topic Topic string to subscribe to
     */
    ZmqSubscribeSocket(const std::string& ip, uint16_t port, const std::string& topic);

    ~ZmqSubscribeSocket();

    /**
     * @brief Attempt a non-blocking receive.
     * @param payload Filled with the payload frame bytes on success.
     * @return true if a message was received; false if no message was available (EAGAIN) or on error.
     */
    bool tryRecv(std::string& payload);

    const std::string& getIp() const
    {
        return m_ip;
    }
    uint16_t getPort() const
    {
        return m_port;
    }
    const std::string& getTopic() const
    {
        return m_topic;
    }

private:
    std::string m_ip;
    uint16_t m_port;
    std::string m_topic;
    ::zmq::socket_t m_socket;
    std::mutex m_mutex; //!< Serializes access to m_socket (ZMQ sockets are not thread-safe)
};

} // namespace core
} // namespace zmq
} // namespace isaacsim
