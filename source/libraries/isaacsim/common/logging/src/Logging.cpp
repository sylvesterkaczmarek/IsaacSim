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

#include "isaacsim/common/logging/Logging.hpp"

#include "isaacsim/common/logging/details/CarboniteBackend.hpp"

#include <stdexcept>

namespace isaacsim
{
namespace common
{
namespace logging
{
namespace
{

template <typename Source, typename Destination>
void assignOptional(const std::optional<Source>& source, uint64_t field, uint64_t& fields, Destination& destination)
{
    if (source.has_value())
    {
        fields |= field;
        destination = static_cast<Destination>(*source);
    }
}

std::shared_ptr<details::CarboniteChannel> acquireLoggerChannel(std::string_view channel)
{
    if (channel.empty())
    {
        throw std::invalid_argument("A logging channel must not be empty.");
    }
    return details::acquireCarboniteChannel(channel);
}

} // namespace

ConfigureResult configureGlobalLogging(const GlobalLoggingConfig& config) noexcept
{
    IsaacSimCommonLoggingGlobalConfig nativeConfig{};
    nativeConfig.structSize = sizeof(nativeConfig);
    std::vector<IsaacSimCommonLoggingChannelConfig> nativeChannels;
    assignOptional(config.enabled, ISAACSIM_COMMON_LOGGING_CONFIG_ENABLED, nativeConfig.fields, nativeConfig.enabled);
    assignOptional(config.minimumLevel, ISAACSIM_COMMON_LOGGING_CONFIG_MINIMUM_LEVEL, nativeConfig.fields,
                   nativeConfig.minimumLevel);
    assignOptional(config.asynchronous, ISAACSIM_COMMON_LOGGING_CONFIG_ASYNCHRONOUS, nativeConfig.fields,
                   nativeConfig.asynchronous);
    assignOptional(config.standardStreamEnabled, ISAACSIM_COMMON_LOGGING_CONFIG_STANDARD_STREAM_ENABLED,
                   nativeConfig.fields, nativeConfig.standardStreamEnabled);
    assignOptional(config.standardStreamLevel, ISAACSIM_COMMON_LOGGING_CONFIG_STANDARD_STREAM_LEVEL,
                   nativeConfig.fields, nativeConfig.standardStreamLevel);
    assignOptional(config.standardStreamFlush, ISAACSIM_COMMON_LOGGING_CONFIG_STANDARD_STREAM_FLUSH,
                   nativeConfig.fields, nativeConfig.standardStreamFlush);
    assignOptional(config.outputStream, ISAACSIM_COMMON_LOGGING_CONFIG_OUTPUT_STREAM, nativeConfig.fields,
                   nativeConfig.outputStream);
    assignOptional(config.debugConsoleEnabled, ISAACSIM_COMMON_LOGGING_CONFIG_DEBUG_CONSOLE_ENABLED,
                   nativeConfig.fields, nativeConfig.debugConsoleEnabled);
    assignOptional(config.debugConsoleLevel, ISAACSIM_COMMON_LOGGING_CONFIG_DEBUG_CONSOLE_LEVEL, nativeConfig.fields,
                   nativeConfig.debugConsoleLevel);
    if (config.filePath.has_value())
    {
        nativeConfig.fields |= ISAACSIM_COMMON_LOGGING_CONFIG_FILE_PATH;
        nativeConfig.filePath = config.filePath->empty() ? nullptr : config.filePath->c_str();
    }
    assignOptional(
        config.fileAppend, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_APPEND, nativeConfig.fields, nativeConfig.fileAppend);
    assignOptional(
        config.fileLevel, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_LEVEL, nativeConfig.fields, nativeConfig.fileLevel);
    assignOptional(config.fileFlushLevel, ISAACSIM_COMMON_LOGGING_CONFIG_FILE_FLUSH_LEVEL, nativeConfig.fields,
                   nativeConfig.fileFlushLevel);
    assignOptional(config.filenameIncluded, ISAACSIM_COMMON_LOGGING_CONFIG_FILENAME_INCLUDED, nativeConfig.fields,
                   nativeConfig.filenameIncluded);
    assignOptional(config.lineNumberIncluded, ISAACSIM_COMMON_LOGGING_CONFIG_LINE_NUMBER_INCLUDED, nativeConfig.fields,
                   nativeConfig.lineNumberIncluded);
    assignOptional(config.functionNameIncluded, ISAACSIM_COMMON_LOGGING_CONFIG_FUNCTION_NAME_INCLUDED,
                   nativeConfig.fields, nativeConfig.functionNameIncluded);
    assignOptional(config.timestampIncluded, ISAACSIM_COMMON_LOGGING_CONFIG_TIMESTAMP_INCLUDED, nativeConfig.fields,
                   nativeConfig.timestampIncluded);
    assignOptional(config.utcTimestamps, ISAACSIM_COMMON_LOGGING_CONFIG_UTC_TIMESTAMPS, nativeConfig.fields,
                   nativeConfig.utcTimestamps);
    assignOptional(config.microsecondTimestamps, ISAACSIM_COMMON_LOGGING_CONFIG_MICROSECOND_TIMESTAMPS,
                   nativeConfig.fields, nativeConfig.microsecondTimestamps);
    assignOptional(
        config.elapsedTime, ISAACSIM_COMMON_LOGGING_CONFIG_ELAPSED_TIME, nativeConfig.fields, nativeConfig.elapsedTime);
    assignOptional(config.threadIdIncluded, ISAACSIM_COMMON_LOGGING_CONFIG_THREAD_ID_INCLUDED, nativeConfig.fields,
                   nativeConfig.threadIdIncluded);
    assignOptional(config.sourceIncluded, ISAACSIM_COMMON_LOGGING_CONFIG_SOURCE_INCLUDED, nativeConfig.fields,
                   nativeConfig.sourceIncluded);
    assignOptional(config.processIdIncluded, ISAACSIM_COMMON_LOGGING_CONFIG_PROCESS_ID_INCLUDED, nativeConfig.fields,
                   nativeConfig.processIdIncluded);
    assignOptional(config.traceIdIncluded, ISAACSIM_COMMON_LOGGING_CONFIG_TRACE_ID_INCLUDED, nativeConfig.fields,
                   nativeConfig.traceIdIncluded);
    assignOptional(config.colorIncluded, ISAACSIM_COMMON_LOGGING_CONFIG_COLOR_INCLUDED, nativeConfig.fields,
                   nativeConfig.colorIncluded);
    assignOptional(config.forceAnsiColor, ISAACSIM_COMMON_LOGGING_CONFIG_FORCE_ANSI_COLOR, nativeConfig.fields,
                   nativeConfig.forceAnsiColor);
    assignOptional(config.multiprocessGroupId, ISAACSIM_COMMON_LOGGING_CONFIG_MULTIPROCESS_GROUP_ID,
                   nativeConfig.fields, nativeConfig.multiprocessGroupId);
    if (!config.channels.empty())
    {
        nativeChannels.reserve(config.channels.size());
        for (const ChannelLoggingConfig& channel : config.channels)
        {
            IsaacSimCommonLoggingChannelConfig nativeChannel = ISAACSIM_COMMON_LOGGING_CHANNEL_CONFIG_INIT;
            nativeChannel.channel = channel.channel.c_str();
            nativeChannel.enabledBehavior = static_cast<int32_t>(channel.enabledBehavior);
            nativeChannel.enabled = channel.enabled ? 1u : 0u;
            nativeChannel.minimumLevelBehavior = static_cast<int32_t>(channel.minimumLevelBehavior);
            nativeChannel.minimumLevel = static_cast<int32_t>(channel.minimumLevel);
            nativeChannels.push_back(nativeChannel);
        }
        nativeConfig.fields |= ISAACSIM_COMMON_LOGGING_CONFIG_CHANNELS;
        nativeConfig.channelConfigs = nativeChannels.data();
        nativeConfig.channelConfigCount = nativeChannels.size();
    }

    return static_cast<ConfigureResult>(isaacsimCommonLoggingConfigureGlobal(&nativeConfig));
}

Logger::Logger(std::string_view channel) : m_channel(acquireLoggerChannel(channel))
{
}

Logger::~Logger() = default;

std::string_view Logger::getChannel() const noexcept
{
    return details::getCarboniteChannelName(m_channel);
}

bool Logger::isEnabled(LogLevel severity) const noexcept
{
    return details::isCarboniteLevelEnabled(severity, m_channel);
}

void Logger::log(LogLevel severity, std::string_view message, SourceLocation location) const noexcept
{
    details::emitToCarbonite(severity, message, m_channel, location);
}

void Logger::report(std::string_view message, SourceLocation location) const noexcept
{
    details::reportToStandardOutputAndCarbonite(message, m_channel, location);
}

void flush() noexcept
{
    details::flushCarboniteBackend();
}

} // namespace logging
} // namespace common
} // namespace isaacsim
