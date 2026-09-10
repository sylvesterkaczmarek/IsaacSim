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

#include <carb/logging/Log.h>

#include <isaacsim/zmq/core/ZmqContext.hpp>
#include <isaacsim/zmq/core/ZmqSubscribeSocket.hpp>

namespace isaacsim
{
namespace zmq
{
namespace core
{

// rcvhwm=1: bound the inbound queue so a slow consumer stays at most ~one message behind the
// sender rather than draining a long backlog of stale commands (the control channel only cares
// about the freshest command; the ZMQ default of 1000 could buffer that many stale ones).
// LINGER=0: don't block on close waiting for pending messages.
constexpr int kRecvHighWaterMark = 1;
constexpr int kLingerMs = 0;

ZmqSubscribeSocket::ZmqSubscribeSocket(const std::string& ip, uint16_t port, const std::string& topic)
    : m_ip(ip), m_port(port), m_topic(topic), m_socket(ZmqContext::get(), ::zmq::socket_type::sub)
{
    m_socket.set(::zmq::sockopt::rcvhwm, kRecvHighWaterMark);
    m_socket.set(::zmq::sockopt::linger, kLingerMs);
    m_socket.set(::zmq::sockopt::subscribe, topic);

    const std::string address = "tcp://" + ip + ":" + std::to_string(port);
    try
    {
        m_socket.connect(address);
        CARB_LOG_INFO("ZmqSubscribeSocket: connected to %s (topic: %s)", address.c_str(), topic.c_str());
    }
    catch (const ::zmq::error_t& e)
    {
        CARB_LOG_ERROR("ZmqSubscribeSocket: failed to connect to %s: %s", address.c_str(), e.what());
        throw;
    }
}

ZmqSubscribeSocket::~ZmqSubscribeSocket()
{
    try
    {
        m_socket.close();
    }
    catch (const ::zmq::error_t& e)
    {
        CARB_LOG_ERROR("ZmqSubscribeSocket: error closing socket: %s", e.what());
    }
}

bool ZmqSubscribeSocket::tryRecv(std::string& payload)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    try
    {
        // First frame: topic
        ::zmq::message_t topicMsg;
        const auto result = m_socket.recv(topicMsg, ::zmq::recv_flags::dontwait);
        if (!result.has_value())
        {
            return false;
        }
        // Second frame: payload
        if (m_socket.get(::zmq::sockopt::rcvmore))
        {
            // dontwait: stay consistent with the non-blocking contract. ZMQ delivers all frames
            // of a multipart message atomically, so with rcvmore set the payload is already
            // available; dontwait just guards against ever blocking the sim loop.
            ::zmq::message_t payloadMsg;
            const auto payloadResult = m_socket.recv(payloadMsg, ::zmq::recv_flags::dontwait);
            if (!payloadResult.has_value())
            {
                CARB_LOG_ERROR("ZmqSubscribeSocket::tryRecv: failed to receive payload frame");
                return false;
            }
            payload.assign(static_cast<char*>(payloadMsg.data()), payloadMsg.size());
        }
        else
        {
            // Malformed message: expected [topic, payload] but got a single frame.
            // Returning the topic bytes as payload would cause silent proto parse failures.
            CARB_LOG_ERROR("ZmqSubscribeSocket::tryRecv: received single-frame message (expected [topic, payload])");
            return false;
        }
        return true;
    }
    catch (const ::zmq::error_t& e)
    {
        if (e.num() == EAGAIN)
        {
            return false;
        }
        CARB_LOG_ERROR("ZmqSubscribeSocket::tryRecv: error: %s", e.what());
        return false;
    }
}

} // namespace core
} // namespace zmq
} // namespace isaacsim
