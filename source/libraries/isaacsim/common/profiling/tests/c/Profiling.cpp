// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "isaacsim/common/profiling/Profiling.h"

#include "isaacsim/common/profiling/ProfilingHost.h"

#include <doctest/doctest.h>

TEST_SUITE("C profiling ABI")
{
    TEST_CASE("No attached Carbonite profiler is a semantic no-op")
    {
        CHECK(isaacsimCommonProfilingHasCarboniteHost() == 0);
        CHECK(isaacsimCommonProfilingIsEnabled(0) == 0);
        CHECK(isaacsimCommonProfilingBegin(0, "disabled", NULL) == 0);
        isaacsimCommonProfilingEnd(0);
        isaacsimCommonProfilingFrame(0, "frame");
        isaacsimCommonProfilingValueInt(0, "integer", 1);
        isaacsimCommonProfilingValueFloat(0, "floating_point", 1.0f);
        isaacsimCommonProfilingInstant(0, ISAACSIM_COMMON_PROFILING_INSTANT_THREAD, "instant", NULL);
        isaacsimCommonProfilingFlow(0, 1, 1, "flow", NULL);
        isaacsimCommonProfilingFlow(0, 0, 1, NULL, NULL);
        isaacsimCommonProfilingSetThreadName("worker");
    }

    TEST_CASE("Source location construction and host argument validation are stable")
    {
        IsaacSimCommonProfilingSourceLocation location = isaacsimCommonProfilingMakeSourceLocation();
        CHECK(location.structSize == sizeof(location));
        CHECK(location.file == NULL);
        CHECK(location.function == NULL);
        CHECK(location.line == 0);

        IsaacSimCommonProfilingHostToken token = 0;
        CHECK(isaacsimCommonProfilingAttachCarboniteHost(NULL, &token) == ISAACSIM_COMMON_PROFILING_HOST_INVALID_ARGUMENT);
        CHECK(isaacsimCommonProfilingAttachCarboniteHost((void*)1, NULL) ==
              ISAACSIM_COMMON_PROFILING_HOST_INVALID_ARGUMENT);
        CHECK(isaacsimCommonProfilingDetachCarboniteHost(0) == ISAACSIM_COMMON_PROFILING_HOST_INVALID_ARGUMENT);
        CHECK(isaacsimCommonProfilingDetachCarboniteHost(1) == ISAACSIM_COMMON_PROFILING_HOST_NOT_OWNER);
    }
}
