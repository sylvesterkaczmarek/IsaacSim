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
#include <isaacsim/zmq/core/ZmqPublishSocket.hpp>

#include <array>
#include <zmq_addon.hpp>

namespace isaacsim
{
namespace zmq
{
namespace core
{

// sndhwm=1 + non-blocking send: bound the outbound queue to a single message and never block
// the sim loop. When the consumer hasn't drained the previous frame, the new send is dropped
// (sendMultipart returns false) rather than blocking — i.e. the in-flight frame is kept and the
// newer one is discarded. (ZMQ_CONFLATE would give true "latest wins" but is incompatible with
// multipart messaging, which the [topic, payload] framing requires.)
// LINGER=0: don't block on close waiting for pending messages.
constexpr int kSendHighWaterMark = 1;
constexpr int kLingerMs = 0;

ZmqPublishSocket::ZmqPublishSocket(const std::string& ip, uint16_t port)
    : m_ip(ip), m_port(port), m_socket(ZmqContext::get(), ::zmq::socket_type::push)
{
    m_socket.set(::zmq::sockopt::sndhwm, kSendHighWaterMark);
    m_socket.set(::zmq::sockopt::linger, kLingerMs);

    const std::string address = "tcp://" + ip + ":" + std::to_string(port);
    try
    {
        m_socket.connect(address);
        CARB_LOG_INFO("ZmqPublishSocket: connected to %s", address.c_str());
    }
    catch (const ::zmq::error_t& e)
    {
        CARB_LOG_ERROR("ZmqPublishSocket: failed to connect to %s: %s", address.c_str(), e.what());
        throw;
    }
}

ZmqPublishSocket::~ZmqPublishSocket()
{
    try
    {
        m_socket.close();
    }
    catch (const ::zmq::error_t& e)
    {
        CARB_LOG_ERROR("ZmqPublishSocket: error closing socket: %s", e.what());
    }
}

bool ZmqPublishSocket::sendMultipart(const std::string& topic, const std::string& payload)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    try
    {
        // Build both frames before sending so neither is partially committed.
        // zmq::send_multipart sends them atomically with respect to the HWM check.
        std::array<::zmq::message_t, 2> frames;
        frames[0] = ::zmq::message_t(topic.data(), topic.size());
        frames[1] = ::zmq::message_t(payload.data(), payload.size());

        // dontwait: drop message if HWM reached rather than blocking the sim loop
        const auto res = ::zmq::send_multipart(m_socket, frames, ::zmq::send_flags::dontwait);
        return res.has_value();
    }
    catch (const ::zmq::error_t& e)
    {
        CARB_LOG_ERROR("ZmqPublishSocket::sendMultipart: error: %s", e.what());
        return false;
    }
}

} // namespace core
} // namespace zmq
} // namespace isaacsim
