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

#define CARB_EXPORTS
#include <carb/Framework.h>
#include <carb/PluginUtils.h>
#include <carb/events/EventsUtils.h>
#include <carb/logging/Log.h>
#include <carb/tokens/ITokens.h>
#include <carb/tokens/TokensUtils.h>

#include <isaacsim/core/includes/LibraryLoader.hpp>
#include <isaacsim/ros2/control/IControlBackend.hpp>
#include <isaacsim/ros2/control/IRos2Control.hpp>
#include <isaacsim/ros2/core/IRos2Core.hpp>
#include <isaacsim/ros2/core/Ros2Distro.hpp>
#include <isaacsim/ros2/core/Ros2Types.hpp>
#include <omni/physics/simulation/IPhysics.h>
#include <omni/physics/simulation/IPhysicsSimulation.h>
#include <omni/physics/tensors/TensorApi.h>
#include <omni/timeline/ITimeline.h>
#include <omni/timeline/TimelineTypes.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace
{
std::shared_ptr<isaacsim::ros2::core::Ros2ContextHandle>* g_sharedContext = nullptr;
bool g_ready = false;

std::shared_ptr<isaacsim::core::includes::LibraryLoader> g_backendLoader;
isaacsim::ros2::control::details::IControlBackend* g_backend = nullptr;
std::mutex g_backendMutex;
std::mutex g_subscriptionMutex;
bool g_shuttingDown = false;

omni::physics::IPhysics* g_physics = nullptr;
omni::physics::IPhysicsSimulation* g_physicsSimulation = nullptr;
omni::physics::SubscriptionId g_physicsSub = omni::physics::kInvalidSubscriptionId;
omni::physics::SubscriptionId g_simulationRegistrySub = omni::physics::kInvalidSubscriptionId;
// Accumulated sim time for the update-rate gate. Atomic: written on the physics
// step thread, reset on the timeline/shutdown thread; relaxed suffices.
std::atomic<double> g_simTime{ 0.0 };
carb::events::ISubscriptionPtr g_timelineSub;

static void onPhysicsStep(float dt, const omni::physics::PhysicsStepContext&)
{
    const double simTime = g_simTime.load(std::memory_order_relaxed) + static_cast<double>(dt);
    g_simTime.store(simTime, std::memory_order_relaxed);
    std::lock_guard<std::mutex> lock(g_backendMutex);
    if (g_backend)
        g_backend->onPhysicsStep(simTime, static_cast<double>(dt));
}

static void unsubscribePhysicsStep()
{
    if (g_physicsSimulation && g_physicsSub != omni::physics::kInvalidSubscriptionId)
        g_physicsSimulation->unsubscribePhysicsOnStepEvents(g_physicsSub);
    g_physicsSub = omni::physics::kInvalidSubscriptionId;
}

static void subscribePhysicsStep()
{
    if (!g_physicsSimulation)
        g_physicsSimulation = carb::getCachedInterface<omni::physics::IPhysicsSimulation>();
    if (!g_physicsSimulation)
    {
        CARB_LOG_WARN_ONCE("[ros2.control] omni::physics::IPhysicsSimulation unavailable; control loop is not driven");
        return;
    }

    unsubscribePhysicsStep();
    g_physicsSub = g_physicsSimulation->subscribePhysicsOnStepEvents(false, 0, onPhysicsStep);
    if (g_physicsSub == omni::physics::kInvalidSubscriptionId)
        CARB_LOG_WARN("[ros2.control] failed to subscribe to physics post-step events");
}

static void onSimulationRegistryEvent(omni::physics::SimulationRegistryEventType::Enum eventType,
                                      omni::physics::SimulationId,
                                      const char*,
                                      void*)
{
    if (eventType == omni::physics::SimulationRegistryEventType::eSIMULATION_REGISTERED ||
        eventType == omni::physics::SimulationRegistryEventType::eSIMULATION_ACTIVATED)
    {
        std::lock_guard<std::mutex> lock(g_subscriptionMutex);
        if (g_shuttingDown)
            return;
        subscribePhysicsStep();
    }
}

static void onTimelineEvent(carb::events::IEvent* e)
{
    if (e->type == static_cast<carb::events::EventType>(omni::timeline::TimelineEventType::eStop))
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        if (g_backend)
            g_backend->destroyAllControllerManagers();
        g_simTime.store(0.0, std::memory_order_relaxed);
    }
}
}

using namespace isaacsim::ros2::control;

static std::optional<std::string> getControlDistro()
{
    const char* distro = std::getenv("ROS_DISTRO");
    const std::string distroName = distro && *distro ? distro : "jazzy";
    if (!isaacsim::ros2::core::stringToRos2Distro(distroName))
    {
        CARB_LOG_ERROR(
            "[ros2.control] unsupported ROS_DISTRO='%s'; supported distributions are 'humble' and 'jazzy'. "
            "ROS 2 Control uses distribution-specific C++ libraries and cannot use the ros2.core Jazzy fallback.",
            distroName.c_str());
        return std::nullopt;
    }
    return distroName;
}

static bool loadBackend(const std::string& distroName)
{
    const std::string libName = std::string("isaacsim.ros2.control.") + distroName;
    auto* tokens = carb::getCachedInterface<carb::tokens::ITokens>();
    const std::string controlRoot =
        tokens ? carb::tokens::resolveString(tokens, "${isaacsim.ros2.control}") : std::string();
    if (controlRoot.empty())
    {
        CARB_LOG_ERROR("[ros2.control] could not resolve ${isaacsim.ros2.control}; backend load will fail");
        return false;
    }
    const std::string prefix = controlRoot + "/" + distroName + "/lib/";
    auto loader = std::make_shared<isaacsim::core::includes::LibraryLoader>(libName, prefix);
    if (!loader->isValid())
    {
        CARB_LOG_ERROR("[ros2.control] failed to load backend %s from prefix '%s'", libName.c_str(), prefix.c_str());
        return false;
    }
    auto createBackend = loader->getSymbol<details::CreateControlBackend>("createControlBackend");
    if (!createBackend)
    {
        CARB_LOG_ERROR("[ros2.control] backend factory not found in %s", libName.c_str());
        return false;
    }
    auto* backend = createBackend(details::kControlBackendAbiVersion);
    if (!backend)
    {
        CARB_LOG_ERROR("[ros2.control] backend ABI %u is unsupported", details::kControlBackendAbiVersion);
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        g_backendLoader = std::move(loader);
        g_backend = backend;
    }
    return true;
}

static std::string pathSegmentKey(std::string value)
{
#ifdef _WIN32
    std::replace(value.begin(), value.end(), '\\', '/');
    std::transform(
        value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
#endif
    while (value.size() > 1 && (value.back() == '/' || value.back() == '\\'))
        value.pop_back();
    return value;
}

// Prepend `value` to path-list env var `name`, preserving priority without duplicates.
static void prependEnvPath(const char* name, const std::string& value)
{
    constexpr char kPathSeparator =
#ifdef _WIN32
        ';';
#else
        ':';
#endif
    const char* existing = std::getenv(name);
    std::string full = value;
    const std::string wanted = pathSegmentKey(value);
    if (existing && *existing)
    {
        const std::string paths(existing);
        size_t start = 0;
        while (start <= paths.size())
        {
            const size_t end = paths.find(kPathSeparator, start);
            const std::string segment = paths.substr(start, end == std::string::npos ? std::string::npos : end - start);
            if (!segment.empty() && pathSegmentKey(segment) != wanted)
                full += std::string(1, kPathSeparator) + segment;
            if (end == std::string::npos)
                break;
            start = end + 1;
        }
    }
#ifdef _WIN32
    _putenv_s(name, full.c_str());
#else
    ::setenv(name, full.c_str(), 1);
#endif
}

// Resolve extension roots via carb tokens so both the build tree and packaged
// installs work. ROS package metadata is staged under each extension's
// <distro>/share prefix; native libraries remain on the platform loader path.
static void augmentRuntimePaths(const std::string& distro)
{
    auto* tokens = carb::getCachedInterface<carb::tokens::ITokens>();
    const std::string controlRoot =
        tokens ? carb::tokens::resolveString(tokens, "${isaacsim.ros2.control}") : std::string();
    if (controlRoot.empty())
    {
        CARB_LOG_ERROR("[ros2.control] could not resolve ${isaacsim.ros2.control}; pluginlib will fail");
        return;
    }
    const std::string coreRoot = tokens ? carb::tokens::resolveString(tokens, "${isaacsim.ros2.core}") : std::string();
    if (coreRoot.empty())
    {
        CARB_LOG_ERROR("[ros2.control] could not resolve ${isaacsim.ros2.core}; ROS package lookup will fail");
        return;
    }

    const std::string controlPrefix = controlRoot + "/" + distro;
    const std::string corePrefix = coreRoot + "/" + distro;

    prependEnvPath("AMENT_PREFIX_PATH", corePrefix);
    prependEnvPath("AMENT_PREFIX_PATH", controlPrefix);
#ifdef _WIN32
    prependEnvPath("PATH", corePrefix + "/lib");
    prependEnvPath("PATH", controlPrefix + "/lib");
#else
    prependEnvPath("LD_LIBRARY_PATH", corePrefix + "/lib");
    prependEnvPath("LD_LIBRARY_PATH", controlPrefix + "/lib");
#endif
}

// Gate backend init on isaacsim.ros2.core being up: its rcl_context being valid
// means the shared rmw/DDS domain exists. core may initialize it lazily.
static bool ensureBackendInitialized()
{
    std::lock_guard<std::mutex> lock(g_backendMutex);
    if (g_ready)
        return true;
    if (!g_sharedContext || !*g_sharedContext || !(*g_sharedContext)->isValid())
    {
        CARB_LOG_WARN_ONCE("[ros2.control] isaacsim.ros2.core context not valid yet; deferring init");
        return false;
    }
    if (!g_backend)
        return false;
    auto* tensorApi = carb::getCachedInterface<omni::physics::tensors::TensorApi>();
    if (!tensorApi)
    {
        CARB_LOG_ERROR("[ros2.control] omni::physics::tensors::TensorApi not available");
        return false;
    }
    if (!g_backend->initialize(tensorApi))
    {
        CARB_LOG_ERROR("[ros2.control] backend init failed");
        return false;
    }
    g_ready = true;
    CARB_LOG_INFO("[ros2.control] backend ready");
    return true;
}

static int setupCm(
    const char* articulationPath, const char* urdfXml, const char* yaml, const char* ns, bool pub, bool useSimTime)
{
    if (!ensureBackendInitialized())
        return -1;
    std::lock_guard<std::mutex> lock(g_backendMutex);
    if (!g_backend)
        return -1;
    return g_backend->createControllerManager(articulationPath, urdfXml, yaml, ns, pub, useSimTime);
}
static void teardownCm(const char* p)
{
    std::lock_guard<std::mutex> lock(g_backendMutex);
    if (g_backend)
        g_backend->destroyControllerManager(p);
}
static void teardownAllCms()
{
    std::lock_guard<std::mutex> lock(g_backendMutex);
    if (g_backend)
        g_backend->destroyAllControllerManagers();
}
static bool isReady()
{
    return ensureBackendInitialized();
}
static void setProfilingEnabled(bool enabled)
{
    std::lock_guard<std::mutex> lock(g_backendMutex);
    if (g_backend)
        g_backend->setProfilingEnabled(enabled);
}
static void resetProfiling()
{
    std::lock_guard<std::mutex> lock(g_backendMutex);
    if (g_backend)
        g_backend->resetProfiling();
}
static const char* getProfilingJson()
{
    static thread_local std::string json;
    try
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        json = g_backend ? g_backend->getProfilingJson() : "{}";
        return json.c_str();
    }
    catch (...)
    {
        return "{}";
    }
}

static const carb::PluginImplDesc g_kPluginDesc = { "isaacsim.ros2.control.plugin", "Isaac ROS2 Control", "NVIDIA",
                                                    carb::PluginHotReload::eDisabled, "dev" };

CARB_PLUGIN_IMPL(g_kPluginDesc, Ros2Control)
CARB_PLUGIN_IMPL_DEPS(omni::physics::tensors::TensorApi)

void fillInterface(Ros2Control& iface)
{
    iface.setupCm = setupCm;
    iface.teardownCm = teardownCm;
    iface.teardownAllCms = teardownAllCms;
    iface.isReady = isReady;
    iface.setProfilingEnabled = setProfilingEnabled;
    iface.resetProfiling = resetProfiling;
    iface.getProfilingJson = getProfilingJson;
}

CARB_EXPORT void carbOnPluginStartup()
{
    auto* core = carb::getCachedInterface<isaacsim::ros2::core::Ros2Bridge>();
    if (!core)
    {
        CARB_LOG_ERROR("[ros2.control] isaacsim.ros2.core not available");
        return;
    }
    g_sharedContext =
        reinterpret_cast<std::shared_ptr<isaacsim::ros2::core::Ros2ContextHandle>*>(core->getDefaultContextHandleAddr());
    if (!g_sharedContext)
        return;

    const auto distro = getControlDistro();
    if (!distro)
        return;

    augmentRuntimePaths(*distro);
    if (!loadBackend(*distro))
        return;

    ensureBackendInitialized();

    g_physics = carb::getCachedInterface<omni::physics::IPhysics>();
    g_physicsSimulation = carb::getCachedInterface<omni::physics::IPhysicsSimulation>();
    {
        std::lock_guard<std::mutex> lock(g_subscriptionMutex);
        g_shuttingDown = false;
    }
    subscribePhysicsStep();
    if (g_physics)
        g_simulationRegistrySub = g_physics->subscribeSimulationRegistryEvents(onSimulationRegistryEvent, nullptr);

    // On Stop, tear down all CMs; OG node re-runs setup() on next Play.
    if (auto* timelineIface = carb::getCachedInterface<omni::timeline::ITimeline>())
    {
        if (auto timeline = timelineIface->getTimeline(nullptr))
        {
            CARB_IGNORE_DEPRECATION_BEGIN
            auto* stream = timeline->getTimelineEventStream();
            CARB_IGNORE_DEPRECATION_END
            if (stream)
                g_timelineSub = carb::events::createSubscriptionToPop(stream, onTimelineEvent);
        }
    }
}

CARB_EXPORT void carbOnPluginShutdown()
{
    g_timelineSub = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_subscriptionMutex);
        g_shuttingDown = true;
    }
    if (g_physics && g_simulationRegistrySub != omni::physics::kInvalidSubscriptionId)
    {
        g_physics->unsubscribeSimulationRegistryEvents(g_simulationRegistrySub);
        g_simulationRegistrySub = omni::physics::kInvalidSubscriptionId;
    }
    unsubscribePhysicsStep();
    g_simTime.store(0.0, std::memory_order_relaxed);
    details::IControlBackend* backend;
    {
        std::lock_guard<std::mutex> lock(g_backendMutex);
        g_ready = false;
        backend = g_backend;
        g_backend = nullptr;
        g_sharedContext = nullptr;
    }
    if (backend)
        backend->release();
    g_backendLoader.reset();
    g_physics = nullptr;
    g_physicsSimulation = nullptr;
}
