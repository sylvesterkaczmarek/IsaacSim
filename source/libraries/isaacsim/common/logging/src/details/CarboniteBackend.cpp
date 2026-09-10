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

#include "isaacsim/common/logging/details/CarboniteBackend.hpp"

#include "LoggingHostInternal.hpp"

#include <carb/ClientUtils.h>
#include <carb/logging/Log.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <condition_variable>
#include <cstddef>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>

CARB_STATIC_BINARY_GLOBALS("isaacsim.common.logging")

namespace isaacsim
{
namespace common
{
namespace logging
{
namespace details
{

class CarboniteChannel
{
public:
    explicit CarboniteChannel(std::string_view channel) : name(channel)
    {
    }

    std::string name;
    int32_t level{ carb::logging::kLevelWarn };
};

namespace
{

constexpr uint64_t kAllGlobalConfigFields = (UINT64_C(1) << 28) - 1;
constexpr size_t kGlobalConfigBaseSize = offsetof(IsaacSimCommonLoggingGlobalConfig, channelConfigs);

bool isValidLevel(int32_t level)
{
    return level >= ISAACSIM_COMMON_LOGGING_LOG_VERBOSE && level <= ISAACSIM_COMMON_LOGGING_LOG_ERROR;
}

bool isValidBoolean(uint32_t value)
{
    return value <= 1;
}

bool isSelected(uint64_t fields, uint64_t field)
{
    return (fields & field) != 0;
}

bool isValidChannelSettingBehavior(int32_t behavior)
{
    return behavior >= ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_UNCHANGED &&
           behavior <= ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_OVERRIDE;
}

bool isValidChannelConfig(const IsaacSimCommonLoggingChannelConfig& config)
{
    if (config.structSize < sizeof(config) || config.channel == nullptr || config.channel[0] == '\0' ||
        !isValidChannelSettingBehavior(config.enabledBehavior) ||
        !isValidChannelSettingBehavior(config.minimumLevelBehavior))
    {
        return false;
    }
    if (config.enabledBehavior == ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_OVERRIDE && !isValidBoolean(config.enabled))
    {
        return false;
    }
    return config.minimumLevelBehavior != ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_OVERRIDE ||
           isValidLevel(config.minimumLevel);
}

bool isValidGlobalConfig(const IsaacSimCommonLoggingGlobalConfig& config)
{
    if (config.structSize < kGlobalConfigBaseSize || (config.fields & ~kAllGlobalConfigFields) != 0)
    {
        return false;
    }

    if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_CHANNELS))
    {
        if (config.structSize < sizeof(config) || (config.channelConfigCount != 0 && config.channelConfigs == nullptr))
        {
            return false;
        }
        for (size_t index = 0; index < config.channelConfigCount; ++index)
        {
            if (!isValidChannelConfig(config.channelConfigs[index]))
            {
                return false;
            }
        }
    }

    if ((isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_MINIMUM_LEVEL) && !isValidLevel(config.minimumLevel)) ||
        (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_STANDARD_STREAM_LEVEL) &&
         !isValidLevel(config.standardStreamLevel)) ||
        (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_DEBUG_CONSOLE_LEVEL) &&
         !isValidLevel(config.debugConsoleLevel)) ||
        (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_LEVEL) && !isValidLevel(config.fileLevel)) ||
        (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_FLUSH_LEVEL) &&
         !isValidLevel(config.fileFlushLevel)))
    {
        return false;
    }
    const struct
    {
        uint64_t field;
        uint32_t value;
    } booleans[] = {
        { ISAACSIM_COMMON_LOGGING_CONFIG_ENABLED, config.enabled },
        { ISAACSIM_COMMON_LOGGING_CONFIG_ASYNCHRONOUS, config.asynchronous },
        { ISAACSIM_COMMON_LOGGING_CONFIG_STANDARD_STREAM_ENABLED, config.standardStreamEnabled },
        { ISAACSIM_COMMON_LOGGING_CONFIG_STANDARD_STREAM_FLUSH, config.standardStreamFlush },
        { ISAACSIM_COMMON_LOGGING_CONFIG_DEBUG_CONSOLE_ENABLED, config.debugConsoleEnabled },
        { ISAACSIM_COMMON_LOGGING_CONFIG_FILE_APPEND, config.fileAppend },
        { ISAACSIM_COMMON_LOGGING_CONFIG_FILENAME_INCLUDED, config.filenameIncluded },
        { ISAACSIM_COMMON_LOGGING_CONFIG_LINE_NUMBER_INCLUDED, config.lineNumberIncluded },
        { ISAACSIM_COMMON_LOGGING_CONFIG_FUNCTION_NAME_INCLUDED, config.functionNameIncluded },
        { ISAACSIM_COMMON_LOGGING_CONFIG_TIMESTAMP_INCLUDED, config.timestampIncluded },
        { ISAACSIM_COMMON_LOGGING_CONFIG_UTC_TIMESTAMPS, config.utcTimestamps },
        { ISAACSIM_COMMON_LOGGING_CONFIG_MICROSECOND_TIMESTAMPS, config.microsecondTimestamps },
        { ISAACSIM_COMMON_LOGGING_CONFIG_THREAD_ID_INCLUDED, config.threadIdIncluded },
        { ISAACSIM_COMMON_LOGGING_CONFIG_SOURCE_INCLUDED, config.sourceIncluded },
        { ISAACSIM_COMMON_LOGGING_CONFIG_PROCESS_ID_INCLUDED, config.processIdIncluded },
        { ISAACSIM_COMMON_LOGGING_CONFIG_TRACE_ID_INCLUDED, config.traceIdIncluded },
        { ISAACSIM_COMMON_LOGGING_CONFIG_COLOR_INCLUDED, config.colorIncluded },
        { ISAACSIM_COMMON_LOGGING_CONFIG_FORCE_ANSI_COLOR, config.forceAnsiColor },
    };
    for (const auto& boolean : booleans)
    {
        if (isSelected(config.fields, boolean.field) && !isValidBoolean(boolean.value))
        {
            return false;
        }
    }

    if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_OUTPUT_STREAM) &&
        config.outputStream != ISAACSIM_COMMON_LOGGING_OUTPUT_STREAM_DEFAULT &&
        config.outputStream != ISAACSIM_COMMON_LOGGING_OUTPUT_STREAM_STDERR)
    {
        return false;
    }
    if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_ELAPSED_TIME) &&
        (config.elapsedTime < ISAACSIM_COMMON_LOGGING_ELAPSED_TIME_DISABLED ||
         config.elapsedTime > ISAACSIM_COMMON_LOGGING_ELAPSED_TIME_NANOSECONDS))
    {
        return false;
    }
    return true;
}

const char* getElapsedTimeUnits(int32_t unit)
{
    switch (unit)
    {
    case ISAACSIM_COMMON_LOGGING_ELAPSED_TIME_MILLISECONDS:
        return "ms";
    case ISAACSIM_COMMON_LOGGING_ELAPSED_TIME_MICROSECONDS:
        return "us";
    case ISAACSIM_COMMON_LOGGING_ELAPSED_TIME_NANOSECONDS:
        return "ns";
    case ISAACSIM_COMMON_LOGGING_ELAPSED_TIME_DISABLED:
    default:
        return "";
    }
}

bool convertToCarboniteLevel(LogLevel severity, int32_t& level)
{
    switch (severity)
    {
    case LogLevel::eVerbose:
        level = carb::logging::kLevelVerbose;
        return true;
    case LogLevel::eInfo:
        level = carb::logging::kLevelInfo;
        return true;
    case LogLevel::eWarning:
        level = carb::logging::kLevelWarn;
        return true;
    case LogLevel::eError:
        level = carb::logging::kLevelError;
        return true;
    default:
        return false;
    }
}

carb::logging::LogSettingBehavior convertToCarboniteBehavior(int32_t behavior)
{
    return behavior == ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_INHERIT ? carb::logging::LogSettingBehavior::eInherit :
                                                                         carb::logging::LogSettingBehavior::eOverride;
}

void writeToStandardOutput(std::string_view channel, std::string_view message) noexcept
{
    static std::mutex standardOutputMutex;
    const std::lock_guard<std::mutex> lock(standardOutputMutex);
    std::fputc('[', stdout);
    if (!channel.empty())
    {
        std::fwrite(channel.data(), sizeof(char), channel.size(), stdout);
    }
    std::fwrite("] ", sizeof(char), 2, stdout);
    if (!message.empty())
    {
        std::fwrite(message.data(), sizeof(char), message.size(), stdout);
    }
    std::fputc('\n', stdout);
    std::fflush(stdout);
}

class CarboniteBackend
{
public:
    CarboniteBackend() noexcept = default;

    ~CarboniteBackend() noexcept
    {
        carb::logging::ILogging* logging = nullptr;
        carb::Framework* framework = nullptr;
        {
            std::unique_lock<std::mutex> backendLock(m_backendMutex);
            if (m_mode != BackendMode::eStandalone)
            {
                return;
            }
            m_stopping = true;
            m_available.store(false, std::memory_order_release);
            logging = m_logging;
            framework = m_framework;
            m_logging = nullptr;
            m_framework = nullptr;
            m_mode = BackendMode::eDisabled;
            m_backendCondition.wait(backendLock, [this] { return m_activeBackendCalls == 0; });
        }

        _flush(logging);
        _removeChannels(logging);
        if (framework != nullptr)
        {
            try
            {
                carb::releaseFrameworkAndDeregisterBuiltins();
            }
            catch (...)
            {
            }
        }
    }

    std::shared_ptr<CarboniteChannel> acquireChannel(std::string_view name)
    {
        {
            const std::shared_lock<std::shared_mutex> channelLock(m_channelMutex);
            const auto found = m_channels.find(name);
            if (found != m_channels.end())
            {
                return found->second;
            }
        }

        carb::logging::ILogging* logging = nullptr;
        {
            const std::lock_guard<std::mutex> backendLock(m_backendMutex);
            if (m_mode == BackendMode::eUninitialized || m_logging == nullptr || m_stopping)
            {
                // Channel construction is backend-neutral. Holding the backend lock while inserting the channel
                // ensures that host attachment cannot miss a channel created concurrently.
                return _acquireChannel(name, nullptr);
            }
            logging = m_logging;
            ++m_activeBackendCalls;
            ++g_backendCallDepth;
        }
        try
        {
            std::shared_ptr<CarboniteChannel> channel = _acquireChannel(name, logging);
            _endBackendCall();
            return channel;
        }
        catch (...)
        {
            _endBackendCall();
            throw;
        }
    }

    IsaacSimCommonLoggingHostResult attachHost(void* carboniteLogging, IsaacSimCommonLoggingHostToken& token) noexcept
    {
        if (carboniteLogging == nullptr || g_backendCallDepth != 0)
        {
            return ISAACSIM_COMMON_LOGGING_HOST_INVALID_ARGUMENT;
        }
        const std::lock_guard<std::mutex> backendLock(m_backendMutex);
        if (m_mode != BackendMode::eUninitialized)
        {
            return ISAACSIM_COMMON_LOGGING_HOST_ALREADY_INITIALIZED;
        }
        m_logging = static_cast<carb::logging::ILogging*>(carboniteLogging);
        m_mode = BackendMode::eHost;
        _registerChannels(m_logging);
        m_available.store(true, std::memory_order_release);
        m_hostToken = ++m_nextHostToken;
        token = m_hostToken;
        return ISAACSIM_COMMON_LOGGING_HOST_SUCCESS;
    }

    IsaacSimCommonLoggingHostResult detachHost(IsaacSimCommonLoggingHostToken token) noexcept
    {
        if (token == 0 || g_backendCallDepth != 0)
        {
            return ISAACSIM_COMMON_LOGGING_HOST_INVALID_ARGUMENT;
        }
        carb::logging::ILogging* logging = nullptr;
        {
            std::unique_lock<std::mutex> backendLock(m_backendMutex);
            if (m_mode != BackendMode::eHost || token != m_hostToken)
            {
                return ISAACSIM_COMMON_LOGGING_HOST_NOT_OWNER;
            }
            m_stopping = true;
            m_available.store(false, std::memory_order_release);
            logging = m_logging;
            m_logging = nullptr;
            m_hostToken = 0;
            m_mode = BackendMode::eDisabled;
            m_backendCondition.wait(backendLock, [this] { return m_activeBackendCalls == 0; });
        }
        _flush(logging);
        _removeChannels(logging);
        return ISAACSIM_COMMON_LOGGING_HOST_SUCCESS;
    }

    bool isEnabled(LogLevel severity, const std::shared_ptr<CarboniteChannel>& channel) noexcept
    {
        _ensureInitialized();
        int32_t carboniteLevel = 0;
        return m_available.load(std::memory_order_acquire) && channel != nullptr &&
               convertToCarboniteLevel(severity, carboniteLevel) && channel->level <= carboniteLevel;
    }

    void emit(LogLevel severity,
              std::string_view message,
              const std::shared_ptr<CarboniteChannel>& channel,
              SourceLocation location) noexcept
    {
        if (channel == nullptr)
        {
            return;
        }
        _ensureInitialized();
        int32_t carboniteLevel = 0;
        if (!convertToCarboniteLevel(severity, carboniteLevel))
        {
            return;
        }
        // Carbonite writes the effective policy into each registered channel. `ILogging::log()`
        // expects its caller to apply the same threshold check as the `CARB_LOG_*` macros.
        if (!m_available.load(std::memory_order_acquire) || channel->level > carboniteLevel)
        {
            return;
        }
        carb::logging::ILogging* logging = _beginBackendCall();
        if (logging == nullptr)
        {
            return;
        }
        try
        {
            const size_t boundedSize = std::min(message.size(), static_cast<size_t>(INT_MAX));
            const char* messageData = message.empty() ? "" : message.data();
            logging->log(channel->name.c_str(), carboniteLevel, location.file, location.function,
                         static_cast<int>(location.line), "%.*s", static_cast<int>(boundedSize), messageData);
        }
        catch (...)
        {
        }
        _endBackendCall();
    }

    void flush() noexcept
    {
        carb::logging::ILogging* logging = _beginBackendCall();
        if (logging == nullptr)
        {
            return;
        }
        _flush(logging);
        _endBackendCall();
    }

    IsaacSimCommonLoggingConfigureResult configureGlobal(const IsaacSimCommonLoggingGlobalConfig& config) noexcept
    {
        if (g_backendCallDepth != 0)
        {
            return ISAACSIM_COMMON_LOGGING_CONFIGURE_BACKEND_UNAVAILABLE;
        }
        if (!isValidGlobalConfig(config))
        {
            return ISAACSIM_COMMON_LOGGING_CONFIGURE_INVALID_ARGUMENT;
        }
        _ensureInitialized();
        const std::lock_guard<std::mutex> lock(m_configurationMutex);
        carb::logging::ILogging* logging = _beginBackendCall();
        if (logging == nullptr)
        {
            return ISAACSIM_COMMON_LOGGING_CONFIGURE_BACKEND_UNAVAILABLE;
        }

        carb::logging::StandardLogger2* logger = logging->getDefaultLogger();
        if (logger == nullptr)
        {
            _endBackendCall();
            return ISAACSIM_COMMON_LOGGING_CONFIGURE_BACKEND_UNAVAILABLE;
        }

        IsaacSimCommonLoggingConfigureResult result = ISAACSIM_COMMON_LOGGING_CONFIGURE_SUCCESS;
        try
        {
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_ENABLED))
            {
                logging->setLogEnabled(config.enabled != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_MINIMUM_LEVEL))
            {
                int32_t level = 0;
                convertToCarboniteLevel(static_cast<LogLevel>(config.minimumLevel), level);
                logging->setLevelThreshold(level);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_ASYNCHRONOUS))
            {
                logging->setLogAsync(config.asynchronous != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_STANDARD_STREAM_ENABLED))
            {
                logger->setStandardStreamOutput(config.standardStreamEnabled != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_STANDARD_STREAM_LEVEL))
            {
                int32_t level = 0;
                convertToCarboniteLevel(static_cast<LogLevel>(config.standardStreamLevel), level);
                logger->setStandardStreamOutputLevelThreshold(level);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_STANDARD_STREAM_FLUSH))
            {
                logger->setFlushStandardStreamOutput(config.standardStreamFlush != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_OUTPUT_STREAM))
            {
                const carb::logging::OutputStream stream =
                    config.outputStream == ISAACSIM_COMMON_LOGGING_OUTPUT_STREAM_STDERR ?
                        carb::logging::OutputStream::eStderr :
                        carb::logging::OutputStream::eDefault;
                logger->setOutputStream(stream);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_DEBUG_CONSOLE_ENABLED))
            {
                logger->setDebugConsoleOutput(config.debugConsoleEnabled != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_DEBUG_CONSOLE_LEVEL))
            {
                int32_t level = 0;
                convertToCarboniteLevel(static_cast<LogLevel>(config.debugConsoleLevel), level);
                logger->setDebugConsoleOutputLevelThreshold(level);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_PATH) ||
                isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_APPEND))
            {
                carb::logging::LogFileConfiguration fileConfig{};
                logger->getFileConfiguration(nullptr, 0, &fileConfig);
                if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_APPEND))
                {
                    fileConfig.append = config.fileAppend != 0;
                }
                const char* filePath = carb::logging::kKeepSameFile;
                if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_PATH))
                {
                    filePath = config.filePath != nullptr && config.filePath[0] != '\0' ? config.filePath : nullptr;
                }
                logger->setFileConfiguration(filePath, &fileConfig);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_LEVEL))
            {
                int32_t level = 0;
                convertToCarboniteLevel(static_cast<LogLevel>(config.fileLevel), level);
                logger->setFileOutputLevelThreshold(level);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_FLUSH_LEVEL))
            {
                int32_t level = 0;
                convertToCarboniteLevel(static_cast<LogLevel>(config.fileFlushLevel), level);
                logger->setFileOuputFlushLevel(level);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FILENAME_INCLUDED))
            {
                logger->setFilenameIncluded(config.filenameIncluded != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_LINE_NUMBER_INCLUDED))
            {
                logger->setLineNumberIncluded(config.lineNumberIncluded != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FUNCTION_NAME_INCLUDED))
            {
                logger->setFunctionNameIncluded(config.functionNameIncluded != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_TIMESTAMP_INCLUDED))
            {
                logger->setTimestampIncluded(config.timestampIncluded != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_UTC_TIMESTAMPS))
            {
                logger->setUtcTimestamps(config.utcTimestamps != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_MICROSECOND_TIMESTAMPS))
            {
                logger->setMicrosecondTimestamp(config.microsecondTimestamps != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_ELAPSED_TIME))
            {
                logger->setElapsedTimeUnits(getElapsedTimeUnits(config.elapsedTime));
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_THREAD_ID_INCLUDED))
            {
                logger->setThreadIdIncluded(config.threadIdIncluded != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_SOURCE_INCLUDED))
            {
                logger->setSourceIncluded(config.sourceIncluded != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_PROCESS_ID_INCLUDED))
            {
                logger->setProcessIdIncluded(config.processIdIncluded != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_TRACE_ID_INCLUDED))
            {
                logger->setTraceIdIncluded(config.traceIdIncluded != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_COLOR_INCLUDED))
            {
                logger->setColorOutputIncluded(config.colorIncluded != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_FORCE_ANSI_COLOR))
            {
                logger->setForceAnsiColor(config.forceAnsiColor != 0);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_MULTIPROCESS_GROUP_ID))
            {
                logger->setMultiProcessGroupId(config.multiprocessGroupId);
            }
            if (isSelected(config.fields, ISAACSIM_COMMON_LOGGING_CONFIG_CHANNELS))
            {
                for (size_t index = 0; index < config.channelConfigCount; ++index)
                {
                    const IsaacSimCommonLoggingChannelConfig& channel = config.channelConfigs[index];
                    _acquireChannel(channel.channel, logging);
                    if (channel.enabledBehavior != ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_UNCHANGED)
                    {
                        logging->setLogEnabledForSource(
                            channel.channel, convertToCarboniteBehavior(channel.enabledBehavior), channel.enabled != 0);
                    }
                    if (channel.minimumLevelBehavior != ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_UNCHANGED)
                    {
                        int32_t level = 0;
                        if (channel.minimumLevelBehavior == ISAACSIM_COMMON_LOGGING_CHANNEL_SETTING_OVERRIDE)
                        {
                            convertToCarboniteLevel(static_cast<LogLevel>(channel.minimumLevel), level);
                        }
                        logging->setLevelThresholdForSource(
                            channel.channel, convertToCarboniteBehavior(channel.minimumLevelBehavior), level);
                    }
                }
            }
        }
        catch (...)
        {
            result = ISAACSIM_COMMON_LOGGING_CONFIGURE_BACKEND_UNAVAILABLE;
        }
        _endBackendCall();
        return result;
    }

private:
    enum class BackendMode
    {
        eUninitialized,
        eStandalone,
        eHost,
        eDisabled,
    };

    std::shared_ptr<CarboniteChannel> _acquireChannel(std::string_view name, carb::logging::ILogging* logging)
    {
        {
            const std::shared_lock<std::shared_mutex> lock(m_channelMutex);
            const auto found = m_channels.find(name);
            if (found != m_channels.end())
            {
                return found->second;
            }
        }

        const std::lock_guard<std::shared_mutex> lock(m_channelMutex);
        const auto found = m_channels.find(name);
        if (found != m_channels.end())
        {
            return found->second;
        }

        auto channel = std::make_shared<CarboniteChannel>(name);
        if (logging != nullptr)
        {
            logging->addChannel(channel->name.c_str(), &channel->level, "Isaac Sim library logging channel");
        }
        m_channels.emplace(channel->name, channel);
        return channel;
    }

    void _registerChannels(carb::logging::ILogging* logging) noexcept
    {
        if (logging == nullptr)
        {
            return;
        }
        const std::lock_guard<std::shared_mutex> channelLock(m_channelMutex);
        for (const auto& entry : m_channels)
        {
            try
            {
                logging->addChannel(
                    entry.second->name.c_str(), &entry.second->level, "Isaac Sim library logging channel");
            }
            catch (...)
            {
            }
        }
    }

    carb::logging::ILogging* _beginBackendCall() noexcept
    {
        const std::lock_guard<std::mutex> backendLock(m_backendMutex);
        if (m_logging == nullptr || m_stopping)
        {
            return nullptr;
        }
        ++m_activeBackendCalls;
        ++g_backendCallDepth;
        return m_logging;
    }

    void _endBackendCall() noexcept
    {
        const std::lock_guard<std::mutex> backendLock(m_backendMutex);
        --g_backendCallDepth;
        --m_activeBackendCalls;
        if (m_stopping && m_activeBackendCalls == 0)
        {
            m_backendCondition.notify_all();
        }
    }

    void _ensureInitialized() noexcept
    {
        {
            const std::lock_guard<std::mutex> backendLock(m_backendMutex);
            if (m_mode != BackendMode::eUninitialized)
            {
                return;
            }
        }

        const std::lock_guard<std::mutex> backendLock(m_backendMutex);
        if (m_mode != BackendMode::eUninitialized)
        {
            return;
        }
        try
        {
            OmniCoreStartArgs coreArgs{};
            coreArgs.flags = fStartFlagDisableIStructuredLog;
            m_framework = carb::acquireFrameworkAndRegisterBuiltins(&coreArgs);
            if (m_framework != nullptr)
            {
                m_logging = carb::logging::getLogging();
                if (m_logging != nullptr)
                {
                    _registerChannels(m_logging);
                    carb::logging::StandardLogger2* defaultLogger = m_logging->getDefaultLogger();
                    if (defaultLogger != nullptr)
                    {
                        defaultLogger->setTimestampIncluded(false);
                        defaultLogger->setElapsedTimeUnits("ms");
                    }
                }
            }
        }
        catch (...)
        {
            m_logging = nullptr;
        }
        m_mode = m_logging == nullptr ? BackendMode::eDisabled : BackendMode::eStandalone;
        m_available.store(m_logging != nullptr, std::memory_order_release);
        if (m_mode == BackendMode::eDisabled && m_framework != nullptr)
        {
            try
            {
                carb::releaseFrameworkAndDeregisterBuiltins();
            }
            catch (...)
            {
            }
            m_framework = nullptr;
        }
    }

    static void _flush(carb::logging::ILogging* logging) noexcept
    {
        if (logging != nullptr)
        {
            try
            {
                logging->flushLogs();
            }
            catch (...)
            {
            }
        }
    }

    void _removeChannels(carb::logging::ILogging* logging) noexcept
    {
        const std::lock_guard<std::shared_mutex> channelLock(m_channelMutex);
        if (logging != nullptr)
        {
            for (const auto& entry : m_channels)
            {
                try
                {
                    logging->removeChannel(entry.second->name.c_str(), &entry.second->level);
                }
                catch (...)
                {
                }
            }
        }
        m_channels.clear();
    }

    static thread_local size_t g_backendCallDepth;

    std::mutex m_backendMutex;
    std::condition_variable m_backendCondition;
    carb::Framework* m_framework{ nullptr };
    carb::logging::ILogging* m_logging{ nullptr };
    BackendMode m_mode{ BackendMode::eUninitialized };
    bool m_stopping{ false };
    std::atomic<bool> m_available{ false };
    size_t m_activeBackendCalls{ 0 };
    IsaacSimCommonLoggingHostToken m_hostToken{ 0 };
    IsaacSimCommonLoggingHostToken m_nextHostToken{ 0 };
    std::mutex m_configurationMutex;
    std::shared_mutex m_channelMutex;
    std::map<std::string, std::shared_ptr<CarboniteChannel>, std::less<>> m_channels;
};

thread_local size_t CarboniteBackend::g_backendCallDepth = 0;

CarboniteBackend& getBackend()
{
    static CarboniteBackend backend;
    return backend;
}

} // namespace

std::shared_ptr<CarboniteChannel> acquireCarboniteChannel(std::string_view channel)
{
    return getBackend().acquireChannel(channel);
}

std::string_view getCarboniteChannelName(const std::shared_ptr<CarboniteChannel>& channel) noexcept
{
    return channel == nullptr ? std::string_view{} : std::string_view(channel->name);
}

bool isCarboniteLevelEnabled(LogLevel severity, const std::shared_ptr<CarboniteChannel>& channel) noexcept
{
    return getBackend().isEnabled(severity, channel);
}

void emitToCarbonite(LogLevel severity,
                     std::string_view message,
                     const std::shared_ptr<CarboniteChannel>& channel,
                     SourceLocation location) noexcept
{
    getBackend().emit(severity, message, channel, location);
}

void reportToStandardOutputAndCarbonite(std::string_view message,
                                        const std::shared_ptr<CarboniteChannel>& channel,
                                        SourceLocation location) noexcept
{
    writeToStandardOutput(getCarboniteChannelName(channel), message);
    emitToCarbonite(LogLevel::eInfo, message, channel, location);
}

void emitToCarbonite(LogLevel severity, std::string_view message, const char* channel, SourceLocation location) noexcept
{
    try
    {
        getBackend().emit(severity, message, getBackend().acquireChannel(channel), location);
    }
    catch (...)
    {
    }
}

void reportToStandardOutputAndCarbonite(std::string_view message, const char* channel, SourceLocation location) noexcept
{
    writeToStandardOutput(channel == nullptr ? std::string_view{} : std::string_view(channel), message);
    emitToCarbonite(LogLevel::eInfo, message, channel, location);
}

IsaacSimCommonLoggingConfigureResult configureCarboniteGlobal(const IsaacSimCommonLoggingGlobalConfig& config) noexcept
{
    return getBackend().configureGlobal(config);
}

void flushCarboniteBackend() noexcept
{
    getBackend().flush();
}

IsaacSimCommonLoggingHostResult attachCarboniteHost(void* carboniteLogging, IsaacSimCommonLoggingHostToken& token) noexcept
{
    return getBackend().attachHost(carboniteLogging, token);
}

IsaacSimCommonLoggingHostResult detachCarboniteHost(IsaacSimCommonLoggingHostToken token) noexcept
{
    return getBackend().detachHost(token);
}

} // namespace details
} // namespace logging
} // namespace common
} // namespace isaacsim
