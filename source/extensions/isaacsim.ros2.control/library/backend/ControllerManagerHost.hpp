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

#include <controller_manager/controller_manager.hpp>
#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace isaacsim
{
namespace ros2
{
namespace control
{
namespace backend
{

struct CmInstance
{
    std::shared_ptr<controller_manager::ControllerManager> cm;
    std::string articulationPath;
    double lastUpdateTimeSec = 0.0;
    double updatePeriodSec = 0.0;
    bool warnedRateTooHigh = false;
    bool useSimTime = true;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr robotDescriptionPub;
};

class ControllerManagerHost
{
public:
    static ControllerManagerHost& instance();

    bool init();
    void shutdown();

    int createCm(const std::string& articulationPath,
                 const std::string& urdfXml,
                 const std::string& yamlPath,
                 const std::string& nsName,
                 bool publishRobotDescription,
                 bool useSimTime);

    void destroyCm(const std::string& articulationPath);
    void destroyAll();

    void onPhysicsStep(double simTimeSec, double physicsDtSec);
    void setProfilingEnabled(bool enabled);
    void resetProfiling();
    std::string profilingJson();

private:
    ControllerManagerHost() = default;
    ControllerManagerHost(const ControllerManagerHost&) = delete;
    void startExecutorThread();
    void stopExecutorThread();
    void recreateExecutor();

    struct ProfilingTotals
    {
        uint64_t physicsStepCalls = 0;
        uint64_t updatedInstances = 0;
        uint64_t skippedInstances = 0;
        uint64_t readCalls = 0;
        uint64_t updateCalls = 0;
        uint64_t writeCalls = 0;
        double totalUs = 0.0;
        double lockedSectionUs = 0.0;
        double readUs = 0.0;
        double controllerUpdateUs = 0.0;
        double writeUs = 0.0;
        double otherUs = 0.0;
    };

    // Single rclcpp context shared by CM nodes and hardware_interface's
    // internal lifecycle nodes. rclcpp cannot adopt core's rcl_context_t, so we
    // unify on rclcpp's global default context (see init()).
    rclcpp::Context::SharedPtr m_rclcppContext;
    // True only if init() initialised m_rclcppContext; shutdown() shuts it down
    // only when we own it, never a context core may have brought up.
    bool m_ownsRclcppContext = false;
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> m_executor;
    std::thread m_executorThread;
    std::atomic<bool> m_running{ false };
    std::atomic<bool> m_profilingEnabled{ false };
    // Serializes CM lifecycle changes with physics read/update/write.
    std::mutex m_lifecycleMutex;
    std::mutex m_registryMutex;
    std::unordered_map<std::string, std::shared_ptr<CmInstance>> m_registry;
    std::mutex m_profilingMutex;
    ProfilingTotals m_profilingTotals;
};

}
}
}
}
