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

#include <zmq.hpp>

namespace isaacsim
{
namespace zmq
{
namespace core
{

/**
 * @class ZmqContext
 * @brief Singleton wrapper around zmq::context_t.
 * @details
 * Provides a single shared ZMQ context for the process. Creating multiple
 * ZMQ contexts is valid but wasteful; a single context with multiple sockets
 * is the recommended pattern.
 */
class ZmqContext
{
public:
    ZmqContext(const ZmqContext&) = delete;
    ZmqContext& operator=(const ZmqContext&) = delete;

    /**
     * @brief Get the singleton ZMQ context.
     * @return zmq::context_t& Reference to the shared context
     */
    static ::zmq::context_t& get()
    {
        static ZmqContext instance;
        return instance.m_context;
    }

private:
    // Number of background I/O threads for the shared context; one is plenty for the
    // bridge's traffic (a few topics, HWM=1, image payloads carried via CUDA-IPC handles).
    static constexpr int kIoThreadCount = 1;

    ZmqContext() : m_context(kIoThreadCount)
    {
    }

    ::zmq::context_t m_context;
};

} // namespace core
} // namespace zmq
} // namespace isaacsim
