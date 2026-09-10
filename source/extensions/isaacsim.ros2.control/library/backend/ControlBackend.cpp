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

#include <carb/Framework.h>
#include <carb/PluginUtils.h>
#include <carb/logging/Log.h>

#include <isaacsim/ros2/control/IControlBackend.hpp>
#include <omni/physics/tensors/TensorApi.h>

#include <atomic>
#include <exception>
#include <new>
#include <string>

// LibraryLoader bypasses Carbonite plugin setup, so acquire the process-wide Framework.
CARB_GLOBALS("isaacsim.ros2.control.backend")

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
std::atomic<omni::physics::tensors::TensorApi*> g_tensorApi{ nullptr };

template <typename Function, typename Result>
Result invoke(const char* operation, Function&& function, Result failure) noexcept
{
    try
    {
        return function();
    }
    catch (const std::exception& exception)
    {
        CARB_LOG_ERROR("[ros2.control] backend %s failed: %s", operation, exception.what());
    }
    catch (...)
    {
        CARB_LOG_ERROR("[ros2.control] backend %s failed with an unknown exception", operation);
    }
    return failure;
}

template <typename Function>
void invoke(const char* operation, Function&& function) noexcept
{
    invoke(
        operation,
        [&function]()
        {
            function();
            return true;
        },
        false);
}
}

void setTensorApi(omni::physics::tensors::TensorApi* tensorApi)
{
    g_tensorApi.store(tensorApi, std::memory_order_release);
}

omni::physics::tensors::TensorApi* getInjectedTensorApi()
{
    return g_tensorApi.load(std::memory_order_acquire);
}

class ControlBackend final : public details::IControlBackend
{
public:
    bool initialize(omni::physics::tensors::TensorApi* tensorApi) noexcept override
    {
        if (!tensorApi)
            return false;
        const bool initialized = invoke(
            "initialization",
            [tensorApi]()
            {
                setTensorApi(tensorApi);
                if (!::g_carbFramework)
                    ::g_carbFramework = carb::acquireFramework("isaacsim.ros2.control.backend");
                return ::g_carbFramework && ControllerManagerHost::instance().init();
            },
            false);
        if (!initialized)
            _shutdown();
        return initialized;
    }

    int createControllerManager(const char* articulationPath,
                                const char* urdfXml,
                                const char* controllerYamlPath,
                                const char* namespaceName,
                                bool publishRobotDescription,
                                bool useSimTime) noexcept override
    {
        if (!articulationPath || !urdfXml || !controllerYamlPath)
            return -1;
        return invoke(
            "controller manager creation",
            [=]()
            {
                return ControllerManagerHost::instance().createCm(articulationPath, urdfXml, controllerYamlPath,
                                                                  namespaceName ? namespaceName : "",
                                                                  publishRobotDescription, useSimTime);
            },
            -1);
    }

    void destroyControllerManager(const char* articulationPath) noexcept override
    {
        if (!articulationPath)
            return;
        invoke("controller manager destruction",
               [articulationPath]() { ControllerManagerHost::instance().destroyCm(articulationPath); });
    }

    void destroyAllControllerManagers() noexcept override
    {
        invoke("controller manager destruction", []() { ControllerManagerHost::instance().destroyAll(); });
    }

    void onPhysicsStep(double simulationTime, double stepSize) noexcept override
    {
        invoke("physics step", [=]() { ControllerManagerHost::instance().onPhysicsStep(simulationTime, stepSize); });
    }

    void setProfilingEnabled(bool enabled) noexcept override
    {
        invoke(
            "profiling configuration", [enabled]() { ControllerManagerHost::instance().setProfilingEnabled(enabled); });
    }

    void resetProfiling() noexcept override
    {
        invoke("profiling reset", []() { ControllerManagerHost::instance().resetProfiling(); });
    }

    const char* getProfilingJson() noexcept override
    {
        static thread_local std::string json;
        return invoke(
            "profiling query",
            []()
            {
                json = ControllerManagerHost::instance().profilingJson();
                return json.c_str();
            },
            "{}");
    }

    void release() noexcept override
    {
        _shutdown();
        delete this;
    }

private:
    void _shutdown() noexcept
    {
        invoke("shutdown", []() { ControllerManagerHost::instance().shutdown(); });
        setTensorApi(nullptr);
        if (::g_carbFramework)
        {
            carb::releaseFramework();
            ::g_carbFramework = nullptr;
        }
    }
};

}
}
}
}

extern "C" CARB_EXPORT isaacsim::ros2::control::details::IControlBackend* CARB_ABI
createControlBackend(std::uint32_t requestedAbi) noexcept
{
    if (requestedAbi != isaacsim::ros2::control::details::kControlBackendAbiVersion)
        return nullptr;
    return new (std::nothrow) isaacsim::ros2::control::backend::ControlBackend;
}
