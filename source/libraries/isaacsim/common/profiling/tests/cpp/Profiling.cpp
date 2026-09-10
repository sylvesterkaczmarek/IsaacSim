// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaacsim/common/profiling/Profiling.hpp"

#include "isaacsim/common/profiling/ProfilingHost.h"

#include <carb/profiler/IProfiler.h>

#include <doctest/doctest.h>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>

using namespace isaacsim::common::profiling;

namespace
{

struct Recorder
{
    std::uint64_t captureMask{ carb::profiler::kCaptureMaskDefault };
    std::size_t beginCount{ 0 };
    std::size_t endCount{ 0 };
    std::size_t frameCount{ 0 };
    std::size_t integerValueCount{ 0 };
    std::size_t floatValueCount{ 0 };
    std::size_t instantCount{ 0 };
    std::size_t flowBeginCount{ 0 };
    std::size_t flowEndCount{ 0 };
    std::size_t threadNameCount{ 0 };
    int instantSourceLine{ 0 };
    int flowBeginSourceLine{ 0 };
    int flowEndSourceLine{ 0 };
    carb::profiler::ZoneId lastEndedZone{ carb::profiler::kNoZoneId };
    std::string lastZoneName;
};

Recorder* g_recorder = nullptr;

std::uint64_t CARB_ABI getCaptureMask()
{
    return g_recorder->captureMask;
}

bool CARB_ABI supportsDynamicSourceLocations()
{
    return true;
}

carb::profiler::ZoneId CARB_ABI recordBegin(
    std::uint64_t, carb::profiler::StaticStringType, carb::profiler::StaticStringType, int, const char* nameFormat, ...)
{
    ++g_recorder->beginCount;
    va_list arguments;
    va_start(arguments, nameFormat);
    const char* name = va_arg(arguments, const char*);
    va_end(arguments);
    g_recorder->lastZoneName = name == nullptr ? "" : name;
    return 17;
}

void CARB_ABI recordEnd(std::uint64_t, carb::profiler::ZoneId zone)
{
    ++g_recorder->endCount;
    g_recorder->lastEndedZone = zone;
}

void CARB_ABI recordFrame(std::uint64_t, const char*, ...)
{
    ++g_recorder->frameCount;
}

void CARB_ABI recordIntegerValue(std::uint64_t, std::int32_t, const char*, ...)
{
    ++g_recorder->integerValueCount;
}

void CARB_ABI recordFloatValue(std::uint64_t, float, const char*, ...)
{
    ++g_recorder->floatValueCount;
}

void CARB_ABI recordInstant(std::uint64_t,
                            carb::profiler::StaticStringType,
                            carb::profiler::StaticStringType,
                            int line,
                            carb::profiler::InstantType,
                            const char*,
                            ...)
{
    ++g_recorder->instantCount;
    g_recorder->instantSourceLine = line;
}

void CARB_ABI recordFlow(std::uint64_t,
                         carb::profiler::StaticStringType,
                         carb::profiler::StaticStringType,
                         int line,
                         carb::profiler::FlowType type,
                         std::uint64_t,
                         const char*,
                         ...)
{
    if (type == carb::profiler::FlowType::Begin)
    {
        ++g_recorder->flowBeginCount;
        g_recorder->flowBeginSourceLine = line;
    }
    else
    {
        ++g_recorder->flowEndCount;
        g_recorder->flowEndSourceLine = line;
    }
}

void CARB_ABI recordThreadName(std::uint64_t, const char*, ...)
{
    ++g_recorder->threadNameCount;
}

carb::profiler::IProfiler makeProfiler()
{
    carb::profiler::IProfiler profiler{};
    profiler.getCaptureMask = getCaptureMask;
    profiler.beginDynamic = recordBegin;
    profiler.endEx = recordEnd;
    profiler.frameDynamic = recordFrame;
    profiler.valueIntDynamic = recordIntegerValue;
    profiler.valueFloatDynamic = recordFloatValue;
    profiler.nameThreadDynamic = recordThreadName;
    profiler.supportsDynamicSourceLocations = supportsDynamicSourceLocations;
    profiler.emitInstantDynamic = recordInstant;
    profiler.emitFlowDynamic = recordFlow;
    return profiler;
}

} // namespace

TEST_SUITE("C++ profiling facade")
{
    TEST_CASE("Disabled lazy zones do not construct names")
    {
        bool constructed = false;
        Zone zone = Zone::lazy(
            [&]()
            {
                constructed = true;
                return std::string("formatted");
            });
        CHECK_FALSE(zone);
        CHECK_FALSE(constructed);

        ISAACSIM_COMMON_PROFILE_ZONE(
            [&]()
            {
                constructed = true;
                return std::string("macro");
            }());
        CHECK_FALSE(constructed);
    }

    TEST_CASE("Disabled event macros do not evaluate event arguments")
    {
        int evaluations = 0;
        const auto makeName = [&]()
        {
            ++evaluations;
            return std::string("disabled");
        };

        ISAACSIM_COMMON_PROFILE_FRAME(makeName());
        ISAACSIM_COMMON_PROFILE_VALUE(makeName(), ++evaluations);
        ISAACSIM_COMMON_PROFILE_INSTANT(makeName());
        ISAACSIM_COMMON_PROFILE_INSTANT_TYPE(InstantType::eProcess, makeName());
        ISAACSIM_COMMON_PROFILE_FLOW_BEGIN(++evaluations, makeName());
        ISAACSIM_COMMON_PROFILE_FLOW_END(++evaluations);

        CHECK(evaluations == 0);
    }

    TEST_CASE("Inactive zones are movable and close is idempotent")
    {
        Zone first("inactive");
        CHECK_FALSE(first);
        Zone second(std::move(first));
        CHECK_FALSE(first);
        CHECK_FALSE(second);
        second.close();
        second.close();
    }

    TEST_CASE("All event helpers are harmless without an application host")
    {
        const SourceLocation location = makeSourceLocation(__FILE__, __func__, __LINE__);
        frame("simulation/frame");
        value("simulation/step", 3);
        value("simulation/duration_ms", 1.25f);
        instant("simulation/ready", InstantType::eProcess, 0, location);
        flow(true, 7, "simulation/handoff", 0, location);
        flow(false, 7, {}, 0, location);
        setThreadName("simulation-worker");
        CHECK_FALSE(isEnabled());
    }

    TEST_CASE("C frame events honor the selected capture mask")
    {
        Recorder recorder;
        g_recorder = &recorder;
        carb::profiler::IProfiler profiler = makeProfiler();
        IsaacSimCommonProfilingHostToken token = 0;
        REQUIRE(isaacsimCommonProfilingAttachCarboniteHost(&profiler, &token) == ISAACSIM_COMMON_PROFILING_HOST_SUCCESS);

        isaacsimCommonProfilingFrame(carb::profiler::kCaptureMaskDefault << 1, "filtered-frame");
        CHECK(recorder.frameCount == 0);
        isaacsimCommonProfilingFrame(0, "admitted-frame");
        CHECK(recorder.frameCount == 1);

        CHECK(isaacsimCommonProfilingDetachCarboniteHost(token) == ISAACSIM_COMMON_PROFILING_HOST_SUCCESS);
        g_recorder = nullptr;
    }

    TEST_CASE("Attached Carbonite profiler receives every facade event")
    {
        Recorder recorder;
        g_recorder = &recorder;
        carb::profiler::IProfiler profiler = makeProfiler();
        IsaacSimCommonProfilingHostToken token = 0;
        REQUIRE(isaacsimCommonProfilingAttachCarboniteHost(&profiler, &token) == ISAACSIM_COMMON_PROFILING_HOST_SUCCESS);

        const SourceLocation location = makeSourceLocation(__FILE__, __func__, __LINE__);
        {
            Zone zone("simulation/step", 0, location);
            REQUIRE(zone);
            ISAACSIM_COMMON_PROFILE_FRAME("simulation/frame");
            ISAACSIM_COMMON_PROFILE_VALUE("simulation/count", 3);
            ISAACSIM_COMMON_PROFILE_VALUE_MASK(0, "simulation/duration_ms", 1.25f);
            ISAACSIM_COMMON_PROFILE_INSTANT_TYPE(InstantType::eProcess, "simulation/ready");
            ISAACSIM_COMMON_PROFILE_FLOW_BEGIN_MASK(0, 7, "simulation/handoff");
            ISAACSIM_COMMON_PROFILE_FLOW_END_MASK(0, 7);
            setThreadName("simulation-worker");
        }

        CHECK(recorder.beginCount == 1);
        CHECK(recorder.endCount == 1);
        CHECK(recorder.frameCount == 1);
        CHECK(recorder.integerValueCount == 1);
        CHECK(recorder.floatValueCount == 1);
        CHECK(recorder.instantCount == 1);
        CHECK(recorder.flowBeginCount == 1);
        CHECK(recorder.flowEndCount == 1);
        CHECK(recorder.threadNameCount == 1);
        CHECK(recorder.instantSourceLine != 0);
        CHECK(recorder.flowBeginSourceLine != 0);
        CHECK(recorder.flowEndSourceLine != 0);
        CHECK(recorder.lastZoneName == "simulation/step");
        CHECK(recorder.lastEndedZone == 17);

        CHECK(isaacsimCommonProfilingDetachCarboniteHost(token) == ISAACSIM_COMMON_PROFILING_HOST_SUCCESS);
        g_recorder = nullptr;
    }

    TEST_CASE("Detaching the Carbonite profiler closes outstanding zones once")
    {
        Recorder recorder;
        g_recorder = &recorder;
        carb::profiler::IProfiler profiler = makeProfiler();
        IsaacSimCommonProfilingHostToken token = 0;
        REQUIRE(isaacsimCommonProfilingAttachCarboniteHost(&profiler, &token) == ISAACSIM_COMMON_PROFILING_HOST_SUCCESS);

        Zone zone("simulation/outstanding");
        REQUIRE(zone);
        CHECK(recorder.beginCount == 1);
        CHECK(recorder.endCount == 0);

        CHECK(isaacsimCommonProfilingDetachCarboniteHost(token) == ISAACSIM_COMMON_PROFILING_HOST_SUCCESS);
        CHECK(recorder.endCount == 1);
        CHECK(recorder.lastEndedZone == 17);

        zone.close();
        CHECK_FALSE(zone);
        CHECK(recorder.endCount == 1);
        g_recorder = nullptr;
    }
}
