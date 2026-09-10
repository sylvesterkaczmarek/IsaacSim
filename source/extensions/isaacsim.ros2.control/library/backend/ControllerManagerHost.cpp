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

#include "ControllerManagerHost.hpp"

#include <carb/logging/Log.h>
#include <carb/profiler/Profile.h>

#include <rclcpp/contexts/default_context.hpp>
#include <rcutils/logging.h>
#if defined(ROS2_BACKEND_HUMBLE)
#    include <rcl_yaml_param_parser/parser.h>
#    include <rclcpp/parameter_map.hpp>
#    include <rcutils/allocator.h>
#endif

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <memory>
#include <sstream>
#if defined(ROS2_BACKEND_HUMBLE)
#    include <unordered_map>
#    include <unordered_set>
#endif
#include <vector>

namespace isaacsim
{
namespace ros2
{
namespace control
{
namespace backend
{

namespace
{
using Clock = std::chrono::steady_clock;

double elapsedUs(Clock::time_point start, Clock::time_point end)
{
    return std::chrono::duration<double, std::micro>(end - start).count();
}

double pct(double value, double total)
{
    return total > 0.0 ? (100.0 * value / total) : 0.0;
}

void appendMetricObject(std::ostringstream& os, const char* name, double totalUs, uint64_t samples)
{
    const double meanUs = samples > 0 ? totalUs / static_cast<double>(samples) : 0.0;
    os << "\"" << name << "\":{\"total_us\":" << totalUs << ",\"mean_us\":" << meanUs << ",\"samples\":" << samples
       << "}";
}

#if defined(ROS2_BACKEND_HUMBLE)
std::string controllerManagerFqn(const std::string& nsName)
{
    const size_t first = nsName.find_first_not_of('/');
    if (first == std::string::npos)
        return "/controller_manager";
    const size_t last = nsName.find_last_not_of('/');
    return "/" + nsName.substr(first, last - first + 1) + "/controller_manager";
}

void addHumbleControllerParameterFiles(rclcpp::NodeOptions& options,
                                       const std::string& yamlPath,
                                       const std::string& managerFqn)
{
    // Humble loads dynamically created controller parameters through
    // <controller>.params_file. Point each controller at the shared YAML unless
    // explicitly configured otherwise.
    struct ParamsDeleter
    {
        void operator()(rcl_params_t* params) const
        {
            if (params)
                rcl_yaml_node_struct_fini(params);
        }
    };
    const rcutils_allocator_t allocator = rcutils_get_default_allocator();
    std::unique_ptr<rcl_params_t, ParamsDeleter> raw(rcl_yaml_node_struct_init(allocator));
    if (!raw || !rcl_parse_yaml_file(yamlPath.c_str(), raw.get()))
        throw std::runtime_error("failed to parse controller YAML");

    const auto parameterMap = rclcpp::parameter_map_from(raw.get(), managerFqn.c_str());
    std::unordered_map<std::string, rclcpp::Parameter> effectiveParameters;
    for (const auto& nodeParameters : parameterMap)
        for (const auto& parameter : nodeParameters.second)
            effectiveParameters.insert_or_assign(parameter.get_name(), parameter);

    std::unordered_set<std::string> controllerNames;
    std::unordered_set<std::string> controllersWithParamFile;
    constexpr const char* kTypeSuffix = ".type";
    constexpr const char* kParamFileSuffix = ".params_file";
    const size_t typeSuffixLength = std::char_traits<char>::length(kTypeSuffix);
    const size_t paramFileSuffixLength = std::char_traits<char>::length(kParamFileSuffix);

    for (const auto& entry : effectiveParameters)
    {
        const std::string& name = entry.first;
        const rclcpp::Parameter& parameter = entry.second;
        if (name.size() > typeSuffixLength &&
            name.compare(name.size() - typeSuffixLength, typeSuffixLength, kTypeSuffix) == 0 &&
            parameter.get_type() == rclcpp::ParameterType::PARAMETER_STRING && !parameter.as_string().empty())
        {
            controllerNames.insert(name.substr(0, name.size() - typeSuffixLength));
        }
        else if (name.size() > paramFileSuffixLength &&
                 name.compare(name.size() - paramFileSuffixLength, paramFileSuffixLength, kParamFileSuffix) == 0)
        {
            controllersWithParamFile.insert(name.substr(0, name.size() - paramFileSuffixLength));
        }
    }
    for (const auto& parameter : options.parameter_overrides())
    {
        const std::string& name = parameter.get_name();
        if (name.size() > paramFileSuffixLength &&
            name.compare(name.size() - paramFileSuffixLength, paramFileSuffixLength, kParamFileSuffix) == 0)
        {
            controllersWithParamFile.insert(name.substr(0, name.size() - paramFileSuffixLength));
        }
    }
    for (const auto& controllerName : controllerNames)
    {
        if (!controllersWithParamFile.count(controllerName))
            options.parameter_overrides().emplace_back(controllerName + kParamFileSuffix, yamlPath);
    }
}
#endif

bool allHardwareComponentsActive(hardware_interface::ResourceManager& resourceManager, const std::string& articulationPath)
{
    auto statuses = resourceManager.get_components_status();
    if (statuses.empty())
    {
        CARB_LOG_ERROR("[ros2.control] ResourceManager loaded no hardware components for %s", articulationPath.c_str());
        return false;
    }
    for (const auto& kv : statuses)
    {
        const std::string state = kv.second.state.label();
        if (state != "active")
        {
            CARB_LOG_ERROR("[ros2.control] hardware component %s for %s is in state '%s', expected 'active'",
                           kv.first.c_str(), articulationPath.c_str(), state.c_str());
            return false;
        }
    }
    return true;
}

class CheckedControllerManager : public controller_manager::ControllerManager
{
public:
    using controller_manager::ControllerManager::ControllerManager;

    bool areHardwareComponentsActive(const std::string& articulationPath)
    {
        if (!resource_manager_)
        {
            CARB_LOG_ERROR("[ros2.control] ControllerManager has no resource manager for %s", articulationPath.c_str());
            return false;
        }
        return allHardwareComponentsActive(*resource_manager_, articulationPath);
    }
};

class ScopedLoggerLevel
{
public:
    ScopedLoggerLevel(const char* name, int level) : m_name(name), m_previous(rcutils_logging_get_logger_level(name))
    {
        const rcutils_ret_t result = rcutils_logging_set_logger_level(name, level);
        if (result != RCUTILS_RET_OK)
            CARB_LOG_WARN("[ros2.control] Failed to set logger level for %s", name);
    }
    ~ScopedLoggerLevel()
    {
        const rcutils_ret_t result = rcutils_logging_set_logger_level(m_name, m_previous);
        if (result != RCUTILS_RET_OK)
            CARB_LOG_WARN("[ros2.control] Failed to restore logger level for %s", m_name);
    }

private:
    const char* m_name;
    int m_previous;
};

}

ControllerManagerHost& ControllerManagerHost::instance()
{
    static ControllerManagerHost s;
    return s;
}

void ControllerManagerHost::startExecutorThread()
{
    if (!m_executor || m_executorThread.joinable())
        return;

    // Guard spin: a throwing service/action callback (e.g. a controller load that
    // throws in on_init) would otherwise terminate the whole process in-proc.
    m_executorThread = std::thread(
        [this]()
        {
            while (m_running.load())
            {
                try
                {
                    m_executor->spin();
                    return; // normal return on cancel() during teardown/shutdown
                }
                catch (const std::exception& ex)
                {
                    CARB_LOG_ERROR("[ros2.control] executor callback threw: %s; resuming", ex.what());
                }
            }
        });
}

void ControllerManagerHost::stopExecutorThread()
{
    if (m_executor)
        m_executor->cancel();
    if (m_executorThread.joinable())
    {
        if (std::this_thread::get_id() == m_executorThread.get_id())
        {
            CARB_LOG_ERROR("[ros2.control] refusing to join executor thread from itself");
            return;
        }
        m_executorThread.join();
    }
}

void ControllerManagerHost::recreateExecutor()
{
    if (!m_rclcppContext || !m_rclcppContext->is_valid())
    {
        m_executor.reset();
        return;
    }
    rclcpp::ExecutorOptions options;
    options.context = m_rclcppContext;
    m_executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>(options);
}

bool ControllerManagerHost::init()
{
    if (m_running)
        return true;
    // rclcpp can't adopt core's existing rcl_context_t, so unify on rclcpp's
    // global default context for both the CM nodes and hardware_interface's
    // internal LifecycleNodes (which default to it); DDS-graph sharing with core
    // happens at the rmw layer via the shared domain.
    m_rclcppContext = rclcpp::contexts::get_global_default_context();
    rclcpp::InitOptions initOpts;
    initOpts.shutdown_on_signal = false; // isaacsim.ros2.core owns signal lifecycle.
    if (!m_rclcppContext->is_valid())
    {
        try
        {
            m_rclcppContext->init(0, nullptr, initOpts);
            m_ownsRclcppContext = true; // we brought it up, so we shut it down.
        }
        catch (const std::exception& ex)
        {
            CARB_LOG_ERROR("[ros2.control] rclcpp default context init failed: %s", ex.what());
            m_rclcppContext.reset();
            return false;
        }
    }

    recreateExecutor();

    m_running.store(true);
    startExecutorThread();
    return true;
}

void ControllerManagerHost::shutdown()
{
    m_running.store(false);
    destroyAll();
    m_executor.reset();

    // Only shut down the context if we initialised it; never tear down a
    // context core (or anyone else) may have brought up.
    if (m_rclcppContext && m_ownsRclcppContext)
        m_rclcppContext->shutdown("isaacsim.ros2.control plugin shutdown");
    m_rclcppContext.reset();
    m_ownsRclcppContext = false;
}

int ControllerManagerHost::createCm(const std::string& articulationPath,
                                    const std::string& urdfXml,
                                    const std::string& yamlPath,
                                    const std::string& nsName,
                                    bool publishRobotDescription,
                                    bool useSimTime)
{
    std::lock_guard<std::mutex> lifecycleLk(m_lifecycleMutex);
    if (!m_running.load() || !m_executor || !m_rclcppContext || !m_rclcppContext->is_valid())
        return -1;
    std::lock_guard<std::mutex> lk(m_registryMutex);
    if (m_registry.count(articulationPath))
        return -2;

    // Declares YAML overrides automatically; required by Humble's dynamic controllers.
    rclcpp::NodeOptions opts = controller_manager::get_cm_node_options();
    opts.context(m_rclcppContext);
    opts.arguments({ "--ros-args", "--params-file", yamlPath });
#if defined(ROS2_BACKEND_HUMBLE)
    try
    {
        addHumbleControllerParameterFiles(opts, yamlPath, controllerManagerFqn(nsName));
    }
    catch (const std::exception& ex)
    {
        CARB_LOG_ERROR(
            "[ros2.control] failed to parse Humble controller parameters from %s: %s", yamlPath.c_str(), ex.what());
        return -3;
    }
#endif
    // Avoid encoding multi-line XML in ROS arguments.
    opts.parameter_overrides().emplace_back("robot_description", urdfXml);
    opts.parameter_overrides().emplace_back("use_sim_time", useSimTime);

    std::shared_ptr<CheckedControllerManager> checkedCmNode;
    try
    {
#ifdef ROS2_BACKEND_JAZZY
        ScopedLoggerLevel quietPalStatistics("pal_statistics", RCUTILS_LOG_SEVERITY_ERROR);
        checkedCmNode = std::make_shared<CheckedControllerManager>(
            m_executor, urdfXml, /*activate_all_hw_components=*/true, "controller_manager", nsName, opts);
#else
        checkedCmNode = std::make_shared<CheckedControllerManager>(m_executor, "controller_manager", nsName, opts);
#endif
    }
    catch (const std::exception& ex)
    {
        CARB_LOG_ERROR("[ros2.control] ControllerManager ctor failed for %s: %s", articulationPath.c_str(), ex.what());
        return -3;
    }
    catch (...)
    {
        CARB_LOG_ERROR(
            "[ros2.control] ControllerManager ctor failed for %s with a non-std exception", articulationPath.c_str());
        return -3;
    }
#ifdef ROS2_BACKEND_JAZZY
    if (!checkedCmNode->is_resource_manager_initialized())
    {
        CARB_LOG_ERROR(
            "[ros2.control] ControllerManager resource manager failed to initialize for %s", articulationPath.c_str());
        return -3;
    }
#endif
    if (!checkedCmNode->areHardwareComponentsActive(articulationPath))
        return -3;
    std::shared_ptr<controller_manager::ControllerManager> cmNode = checkedCmNode;
    m_executor->add_node(cmNode->get_node_base_interface());

    auto inst = std::make_shared<CmInstance>();
    inst->cm = cmNode;
    inst->articulationPath = articulationPath;
    inst->useSimTime = useSimTime;

    if (publishRobotDescription)
    {
        // Use an absolute topic to bypass the controller_manager namespace.
        std::string ns = nsName;
        while (!ns.empty() && ns.front() == '/')
            ns.erase(ns.begin());
        const std::string topic = ns.empty() ? "/robot_description" : ("/" + ns + "/robot_description");
        // Held on CmInstance so transient_local QoS serves late subscribers.
        auto qos = rclcpp::QoS(1).transient_local().reliable();
        inst->robotDescriptionPub = cmNode->create_publisher<std_msgs::msg::String>(topic, qos);
        std_msgs::msg::String msg;
        msg.data = urdfXml;
        inst->robotDescriptionPub->publish(msg);
    }

    unsigned int hz = cmNode->get_update_rate();
    inst->updatePeriodSec = (hz > 0) ? (1.0 / static_cast<double>(hz)) : 0.0;
    m_registry[articulationPath] = std::move(inst);
    // destroyAll() leaves the executor stopped until another manager is created.
    startExecutorThread();
    return 0;
}

void ControllerManagerHost::destroyCm(const std::string& articulationPath)
{
    std::lock_guard<std::mutex> lifecycleLk(m_lifecycleMutex);
    const bool restartExecutor = m_running.load() && m_executor != nullptr;
    std::shared_ptr<CmInstance> inst;
    {
        std::lock_guard<std::mutex> lk(m_registryMutex);
        auto it = m_registry.find(articulationPath);
        if (it == m_registry.end())
            return;
        inst = it->second;
    }
    if (m_executor)
        stopExecutorThread();

    std::lock_guard<std::mutex> lk(m_registryMutex);
    if (m_executor)
        m_executor->remove_node(inst->cm->get_node_base_interface());
    m_registry.erase(articulationPath);
    if (restartExecutor)
    {
        if (m_registry.empty())
            recreateExecutor();
        else
            startExecutorThread();
    }
}

void ControllerManagerHost::destroyAll()
{
    std::lock_guard<std::mutex> lifecycleLk(m_lifecycleMutex);
    if (m_executor)
        stopExecutorThread();

    std::lock_guard<std::mutex> lk(m_registryMutex);
    for (auto& kv : m_registry)
    {
        if (m_executor)
            m_executor->remove_node(kv.second->cm->get_node_base_interface());
    }
    m_registry.clear();
    if (m_running.load())
        recreateExecutor();
}

void ControllerManagerHost::onPhysicsStep(double simTimeSec, double physicsDtSec)
{
    CARB_PROFILE_ZONE(0, "[ros2.control] ControllerManagerHost::onPhysicsStep");

    const bool profile = m_profilingEnabled.load(std::memory_order_relaxed);
    const auto stepStart = profile ? Clock::now() : Clock::time_point{};
    std::lock_guard<std::mutex> lifecycleLk(m_lifecycleMutex);

    // Snapshot the instances due this step under a short-held lock.
    const auto lockStart = profile ? Clock::now() : Clock::time_point{};
    std::vector<std::shared_ptr<CmInstance>> dueInstances;
    uint64_t skippedInstances = 0;
    {
        std::lock_guard<std::mutex> lk(m_registryMutex);
        dueInstances.reserve(m_registry.size());
        for (auto& kv : m_registry)
        {
            auto& inst = kv.second;
            // update_rate above the physics rate is clamped to it; warn once.
            if (inst->updatePeriodSec > 0.0 && inst->updatePeriodSec < physicsDtSec - 1e-9 && !inst->warnedRateTooHigh)
            {
                CARB_LOG_WARN(
                    "[ros2.control] %s: update_rate %.1f Hz exceeds physics step rate %.1f Hz; "
                    "clamped to the physics rate.",
                    inst->articulationPath.c_str(), 1.0 / inst->updatePeriodSec, 1.0 / physicsDtSec);
                inst->warnedRateTooHigh = true;
            }
            // Half-step tolerance: float dt is marginally below updatePeriodSec, so a
            // strict < comparison skips the boundary step and halves the rate.
            if (inst->updatePeriodSec > 0.0 &&
                (simTimeSec - inst->lastUpdateTimeSec) < (inst->updatePeriodSec - 0.5 * physicsDtSec))
            {
                ++skippedInstances;
                continue;
            }
            // Advance on the period grid (not to the actual fire time) so the
            // long-run average rate matches update_rate even when it isn't an
            // integer divisor of the physics rate. Resync if we fall more than a
            // period behind, so an unattainably high rate can't accumulate lag.
            inst->lastUpdateTimeSec += inst->updatePeriodSec;
            if (simTimeSec - inst->lastUpdateTimeSec > inst->updatePeriodSec)
                inst->lastUpdateTimeSec = simTimeSec;
            dueInstances.push_back(inst);
        }
    }
    const auto lockAcquired = profile ? Clock::now() : Clock::time_point{};

    uint64_t updatedInstances = 0;
    double readUs = 0.0;
    double controllerUpdateUs = 0.0;
    double writeUs = 0.0;

    for (auto& inst : dueInstances)
    {
#if defined(ROS2_BACKEND_JAZZY)
        // Match ros2_control_node: controller activation and updates must use the same trigger clock.
        const rclcpp::Time t = inst->cm->get_trigger_clock()->now();
#else
        const rclcpp::Time t =
            inst->useSimTime ? rclcpp::Time(static_cast<int64_t>(simTimeSec * 1e9), RCL_ROS_TIME) : inst->cm->now();
#endif
        const rclcpp::Duration dt = rclcpp::Duration::from_seconds(std::max(inst->updatePeriodSec, physicsDtSec));
        ++updatedInstances;
        // Isolate each CM: a throw must not cross the extern "C"/physics boundary.
        try
        {
            {
                CARB_PROFILE_ZONE(0, "[ros2.control] ControllerManagerHost::read");
                const auto start = profile ? Clock::now() : Clock::time_point{};
                inst->cm->read(t, dt);
                if (profile)
                    readUs += elapsedUs(start, Clock::now());
            }
            {
                CARB_PROFILE_ZONE(0, "[ros2.control] ControllerManagerHost::update");
                const auto start = profile ? Clock::now() : Clock::time_point{};
                inst->cm->update(t, dt);
                if (profile)
                    controllerUpdateUs += elapsedUs(start, Clock::now());
            }
            {
                CARB_PROFILE_ZONE(0, "[ros2.control] ControllerManagerHost::write");
                const auto start = profile ? Clock::now() : Clock::time_point{};
                inst->cm->write(t, dt);
                if (profile)
                    writeUs += elapsedUs(start, Clock::now());
            }
        }
        catch (const std::exception& ex)
        {
            CARB_LOG_ERROR(
                "[ros2.control] read/update/write threw for %s: %s", inst->articulationPath.c_str(), ex.what());
        }
    }

    if (profile)
    {
        const auto stepEnd = Clock::now();
        const double totalUs = elapsedUs(stepStart, stepEnd);
        const double lockedSectionUs = elapsedUs(lockStart, lockAcquired);
        const double categorizedUs = lockedSectionUs + readUs + controllerUpdateUs + writeUs;
        std::lock_guard<std::mutex> profileLk(m_profilingMutex);
        m_profilingTotals.physicsStepCalls += 1;
        m_profilingTotals.updatedInstances += updatedInstances;
        m_profilingTotals.skippedInstances += skippedInstances;
        m_profilingTotals.readCalls += updatedInstances;
        m_profilingTotals.updateCalls += updatedInstances;
        m_profilingTotals.writeCalls += updatedInstances;
        m_profilingTotals.totalUs += totalUs;
        m_profilingTotals.lockedSectionUs += lockedSectionUs;
        m_profilingTotals.readUs += readUs;
        m_profilingTotals.controllerUpdateUs += controllerUpdateUs;
        m_profilingTotals.writeUs += writeUs;
        m_profilingTotals.otherUs += std::max(0.0, totalUs - categorizedUs);
    }
}

void ControllerManagerHost::setProfilingEnabled(bool enabled)
{
    m_profilingEnabled.store(enabled, std::memory_order_relaxed);
}

void ControllerManagerHost::resetProfiling()
{
    std::lock_guard<std::mutex> lk(m_profilingMutex);
    m_profilingTotals = ProfilingTotals{};
}

std::string ControllerManagerHost::profilingJson()
{
    ProfilingTotals totals;
    {
        std::lock_guard<std::mutex> lk(m_profilingMutex);
        totals = m_profilingTotals;
    }

    const uint64_t stepSamples = totals.physicsStepCalls;
    const uint64_t instanceSamples = totals.updatedInstances;
    const double meanStepTotalUs = stepSamples > 0 ? totals.totalUs / static_cast<double>(stepSamples) : 0.0;
    const double meanInstanceTotalUs = instanceSamples > 0 ? totals.totalUs / static_cast<double>(instanceSamples) : 0.0;

    std::ostringstream os;
    os << std::fixed << std::setprecision(3);
    os << "{";
    os << "\"enabled\":" << (m_profilingEnabled.load(std::memory_order_relaxed) ? "true" : "false") << ",";
    os << "\"physics_step_calls\":" << totals.physicsStepCalls << ",";
    os << "\"updated_instances\":" << totals.updatedInstances << ",";
    os << "\"skipped_instances\":" << totals.skippedInstances << ",";
    os << "\"mean_per_physics_step_us\":{\"total\":" << meanStepTotalUs << ",";
    os << "\"locked_section\":" << (stepSamples > 0 ? totals.lockedSectionUs / static_cast<double>(stepSamples) : 0.0)
       << ",";
    os << "\"read\":" << (stepSamples > 0 ? totals.readUs / static_cast<double>(stepSamples) : 0.0) << ",";
    os << "\"controller_update\":"
       << (stepSamples > 0 ? totals.controllerUpdateUs / static_cast<double>(stepSamples) : 0.0) << ",";
    os << "\"write\":" << (stepSamples > 0 ? totals.writeUs / static_cast<double>(stepSamples) : 0.0) << ",";
    os << "\"other\":" << (stepSamples > 0 ? totals.otherUs / static_cast<double>(stepSamples) : 0.0) << "},";
    os << "\"mean_per_updated_instance_us\":{\"total\":" << meanInstanceTotalUs << ",";
    os << "\"read\":" << (instanceSamples > 0 ? totals.readUs / static_cast<double>(instanceSamples) : 0.0) << ",";
    os << "\"controller_update\":"
       << (instanceSamples > 0 ? totals.controllerUpdateUs / static_cast<double>(instanceSamples) : 0.0) << ",";
    os << "\"write\":" << (instanceSamples > 0 ? totals.writeUs / static_cast<double>(instanceSamples) : 0.0) << "},";
    os << "\"percent_of_total\":{\"locked_section\":" << pct(totals.lockedSectionUs, totals.totalUs) << ",";
    os << "\"read\":" << pct(totals.readUs, totals.totalUs) << ",";
    os << "\"controller_update\":" << pct(totals.controllerUpdateUs, totals.totalUs) << ",";
    os << "\"write\":" << pct(totals.writeUs, totals.totalUs) << ",";
    os << "\"other\":" << pct(totals.otherUs, totals.totalUs) << "},";
    os << "\"totals\":{";
    appendMetricObject(os, "total", totals.totalUs, stepSamples);
    os << ",";
    appendMetricObject(os, "locked_section", totals.lockedSectionUs, stepSamples);
    os << ",";
    appendMetricObject(os, "read", totals.readUs, totals.readCalls);
    os << ",";
    appendMetricObject(os, "controller_update", totals.controllerUpdateUs, totals.updateCalls);
    os << ",";
    appendMetricObject(os, "write", totals.writeUs, totals.writeCalls);
    os << ",";
    appendMetricObject(os, "other", totals.otherUs, stepSamples);
    os << "}";
    os << "}";
    return os.str();
}

}
}
}
}
