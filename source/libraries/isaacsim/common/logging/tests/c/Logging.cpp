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

#include "isaacsim/common/logging/Logging.h"

#include "isaacsim/common/logging/LoggingHost.h"

#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>

TEST_SUITE("C API")
{
    TEST_CASE("Global configuration accepts the legacy structure size")
    {
        IsaacSimCommonLoggingGlobalConfig config = ISAACSIM_COMMON_LOGGING_GLOBAL_CONFIG_INIT;
        config.fields = ISAACSIM_COMMON_LOGGING_CONFIG_ELAPSED_TIME;
        config.elapsedTime = ISAACSIM_COMMON_LOGGING_ELAPSED_TIME_MILLISECONDS;

        CHECK(isaacsimCommonLoggingConfigureGlobal(&config) == ISAACSIM_COMMON_LOGGING_CONFIGURE_SUCCESS);

        config.structSize = offsetof(IsaacSimCommonLoggingGlobalConfig, channelConfigs);
        CHECK(isaacsimCommonLoggingConfigureGlobal(&config) == ISAACSIM_COMMON_LOGGING_CONFIGURE_SUCCESS);
    }

    TEST_CASE("Channel enablement can be overridden and restored")
    {
        constexpr const char* kChannel = "isaacsim.test.c";
        IsaacSimCommonLoggingChannelConfig channel = ISAACSIM_COMMON_LOGGING_CHANNEL_CONFIG_INIT;
        channel.channel = kChannel;
        channel.enabledBehavior = ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_OVERRIDE;
        channel.enabled = 0;

        IsaacSimCommonLoggingGlobalConfig config = ISAACSIM_COMMON_LOGGING_GLOBAL_CONFIG_INIT;
        config.fields = ISAACSIM_COMMON_LOGGING_CONFIG_CHANNELS;
        config.channelConfigs = &channel;
        config.channelConfigCount = 1;

        REQUIRE(isaacsimCommonLoggingConfigureGlobal(&config) == ISAACSIM_COMMON_LOGGING_CONFIGURE_SUCCESS);
        ISAACSIM_COMMON_LOGGING_ERROR(kChannel, "disabled channel record");

        channel.enabledBehavior = ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_INHERIT;
        CHECK(isaacsimCommonLoggingConfigureGlobal(&config) == ISAACSIM_COMMON_LOGGING_CONFIGURE_SUCCESS);
    }

    TEST_CASE("Logging macros submit C API records")
    {
        constexpr const char* kChannel = "isaacsim.test.c";
        ISAACSIM_COMMON_LOGGING_VERBOSE(kChannel, "verbose record");
        ISAACSIM_COMMON_LOGGING_INFO(kChannel, "info record");
        ISAACSIM_COMMON_LOGGING_WARNING(kChannel, "isaacsim-c-api-backend-probe");
        ISAACSIM_COMMON_LOGGING_ERROR(kChannel, "error record");
    }

    TEST_CASE("Report writes to standard output and submits to Carbonite")
    {
        ISAACSIM_COMMON_LOGGING_REPORT("isaacsim.test.c", "isaacsim-c-report-probe");
    }

    TEST_CASE("Invalid supplemental logging input is ignored")
    {
        constexpr const char* kChannel = "isaacsim.test.c";
        isaacsimCommonLoggingWrite(ISAACSIM_COMMON_LOGGING_LOG_INFO, nullptr, "message", nullptr, nullptr, 0);
        isaacsimCommonLoggingWrite(ISAACSIM_COMMON_LOGGING_LOG_INFO, kChannel, nullptr, nullptr, nullptr, 0);
        isaacsimCommonLoggingWrite(
            static_cast<IsaacSimCommonLoggingLogLevel>(99), kChannel, "message", nullptr, nullptr, 0);
        isaacsimCommonLoggingReport(nullptr, "message", nullptr, nullptr, 0);
        isaacsimCommonLoggingReport(kChannel, nullptr, nullptr, nullptr, 0);
    }

    TEST_CASE("Invalid global configuration is rejected")
    {
        CHECK(isaacsimCommonLoggingConfigureGlobal(nullptr) == ISAACSIM_COMMON_LOGGING_CONFIGURE_INVALID_ARGUMENT);

        IsaacSimCommonLoggingChannelConfig channel = ISAACSIM_COMMON_LOGGING_CHANNEL_CONFIG_INIT;
        channel.channel = "isaacsim.test.c";
        IsaacSimCommonLoggingGlobalConfig config = ISAACSIM_COMMON_LOGGING_GLOBAL_CONFIG_INIT;

        config.fields = UINT64_C(1) << 63;
        CHECK(isaacsimCommonLoggingConfigureGlobal(&config) == ISAACSIM_COMMON_LOGGING_CONFIGURE_INVALID_ARGUMENT);

        config.fields = ISAACSIM_COMMON_LOGGING_CONFIG_CHANNELS;
        config.channelConfigs = &channel;
        config.channelConfigCount = 1;
        channel.enabledBehavior = 99;
        CHECK(isaacsimCommonLoggingConfigureGlobal(&config) == ISAACSIM_COMMON_LOGGING_CONFIGURE_INVALID_ARGUMENT);

        channel.enabledBehavior = ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_INHERIT;
        config.channelConfigs = nullptr;
        CHECK(isaacsimCommonLoggingConfigureGlobal(&config) == ISAACSIM_COMMON_LOGGING_CONFIGURE_INVALID_ARGUMENT);

        config.channelConfigs = &channel;
        channel.structSize = 0;
        CHECK(isaacsimCommonLoggingConfigureGlobal(&config) == ISAACSIM_COMMON_LOGGING_CONFIGURE_INVALID_ARGUMENT);
    }

    TEST_CASE("Flush drains the process backend")
    {
        isaacsimCommonLoggingFlush();
    }

    TEST_CASE("Host integration rejects invalid ownership input")
    {
        IsaacSimCommonLoggingHostToken token = 0;
        CHECK(isaacsimCommonLoggingAttachCarboniteHost(nullptr, &token) == ISAACSIM_COMMON_LOGGING_HOST_INVALID_ARGUMENT);
        CHECK(isaacsimCommonLoggingAttachCarboniteHost(reinterpret_cast<void*>(1), nullptr) ==
              ISAACSIM_COMMON_LOGGING_HOST_INVALID_ARGUMENT);
        CHECK(isaacsimCommonLoggingDetachCarboniteHost(0) == ISAACSIM_COMMON_LOGGING_HOST_INVALID_ARGUMENT);
    }
}
