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

#include <carb/PluginUtils.h>
#include <carb/logging/Log.h>
#include <carb/profiler/IProfiler.h>

#include <isaacsim/common/logging/Logging.hpp>
#include <isaacsim/common/logging/LoggingHost.h>
#include <isaacsim/common/profiling/ProfilingHost.h>
#include <omni/ext/IExt.h>

#include <memory>
#include <string_view>

namespace isaacsim
{
namespace common
{
namespace kit
{

namespace
{
IsaacSimCommonLoggingHostToken g_hostToken{ 0 };
IsaacSimCommonProfilingHostToken g_profilingHostToken{ 0 };
std::unique_ptr<logging::Logger> g_logger;

void detachHosts()
{
    if (g_hostToken != 0)
    {
        const auto result = isaacsimCommonLoggingDetachCarboniteHost(g_hostToken);
        g_hostToken = 0;
        if (result != ISAACSIM_COMMON_LOGGING_HOST_SUCCESS)
        {
            CARB_LOG_ERROR("Could not detach isaacsim.common.logging from Kit: result %d", static_cast<int>(result));
        }
    }
    if (g_profilingHostToken != 0)
    {
        const auto result = isaacsimCommonProfilingDetachCarboniteHost(g_profilingHostToken);
        g_profilingHostToken = 0;
        if (result != ISAACSIM_COMMON_PROFILING_HOST_SUCCESS)
        {
            CARB_LOG_ERROR("Could not detach isaacsim.common.profiling from Kit: result %d", static_cast<int>(result));
        }
    }
}
} // namespace

class Extension final : public omni::ext::IExt
{
public:
    void onStartup(const char* extensionId) override
    {
        const std::string_view effectiveExtensionId = extensionId == nullptr ? "isaacsim.common" : extensionId;
        if (g_hostToken != 0 || g_profilingHostToken != 0)
        {
            CARB_LOG_ERROR("%.*s is already attached to the isaacsim_common runtime",
                           static_cast<int>(effectiveExtensionId.size()), effectiveExtensionId.data());
            return;
        }

        const auto loggingResult = isaacsimCommonLoggingAttachCarboniteHost(carb::logging::getLogging(), &g_hostToken);
        if (loggingResult != ISAACSIM_COMMON_LOGGING_HOST_SUCCESS)
        {
            CARB_LOG_ERROR("%.*s could not attach isaacsim.common.logging to Kit: result %d",
                           static_cast<int>(effectiveExtensionId.size()), effectiveExtensionId.data(),
                           static_cast<int>(loggingResult));
            detachHosts();
            return;
        }

        try
        {
            g_logger = std::make_unique<logging::Logger>("isaacsim.common.extension");
            ISAACSIM_LOG_INFO(*g_logger, "{} attached the isaacsim_common runtime to Kit", effectiveExtensionId);
        }
        catch (...)
        {
            detachHosts();
            CARB_LOG_ERROR("%.*s failed to initialize the isaacsim_common runtime",
                           static_cast<int>(effectiveExtensionId.size()), effectiveExtensionId.data());
            return;
        }

        auto* profiler = carb::getCachedInterface<carb::profiler::IProfiler>();
        if (profiler == nullptr)
        {
            ISAACSIM_LOG_WARN(*g_logger, "{} could not acquire the optional Kit profiling service", effectiveExtensionId);
            return;
        }

        const auto profilingResult = isaacsimCommonProfilingAttachCarboniteHost(profiler, &g_profilingHostToken);
        if (profilingResult != ISAACSIM_COMMON_PROFILING_HOST_SUCCESS)
        {
            ISAACSIM_LOG_WARN(*g_logger, "{} could not attach isaacsim.common.profiling to Kit: result {}",
                              effectiveExtensionId, static_cast<int>(profilingResult));
        }
    }

    void onShutdown() override
    {
        if (g_hostToken == 0 && g_profilingHostToken == 0)
        {
            return;
        }
        if (g_logger)
        {
            ISAACSIM_LOG_INFO(*g_logger, "Detaching the isaacsim_common runtime from Kit");
        }
        g_logger.reset();
        detachHosts();
    }
};

} // namespace kit
} // namespace common
} // namespace isaacsim

const carb::PluginImplDesc g_kPluginDesc = { "isaacsim.common.plugin", "Isaac Sim common library Kit adapter", "NVIDIA",
                                             carb::PluginHotReload::eDisabled, "dev" };

CARB_PLUGIN_IMPL(g_kPluginDesc, isaacsim::common::kit::Extension)
CARB_PLUGIN_IMPL_DEPS(carb::logging::ILogging)

void fillInterface(isaacsim::common::kit::Extension& extension)
{
    extension = {};
}
