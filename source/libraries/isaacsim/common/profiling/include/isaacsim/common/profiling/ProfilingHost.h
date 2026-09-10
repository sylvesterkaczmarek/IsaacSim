// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isaacsim/common/profiling/Profiling.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief Result returned by Carbonite host ownership operations. */
    typedef enum IsaacSimCommonProfilingHostResult
    {
        ISAACSIM_COMMON_PROFILING_HOST_SUCCESS = 0,
        ISAACSIM_COMMON_PROFILING_HOST_INVALID_ARGUMENT = 1,
        ISAACSIM_COMMON_PROFILING_HOST_ALREADY_ATTACHED = 2,
        ISAACSIM_COMMON_PROFILING_HOST_NOT_OWNER = 3,
        ISAACSIM_COMMON_PROFILING_HOST_UNAVAILABLE = 4,
    } IsaacSimCommonProfilingHostResult;

    /** @brief Token proving ownership of the attached Carbonite profiler. */
    typedef uint64_t IsaacSimCommonProfilingHostToken;

    /**
     * @brief Attach an application-owned Carbonite `IProfiler`.
     *
     * @param[in] carboniteProfiler Non-null borrowed `carb::profiler::IProfiler` pointer passed as `void*`.
     * @param[out] token Receives a nonzero ownership token on success.
     * @return Host attachment result.
     *
     * The application must keep the profiler alive until it detaches the returned token. The public ABI does not
     * expose Carbonite types. This function does not start or stop the profiler.
     */
    ISAACSIM_COMMON_PROFILING_API IsaacSimCommonProfilingHostResult
    isaacsimCommonProfilingAttachCarboniteHost(void* carboniteProfiler, IsaacSimCommonProfilingHostToken* token);

    /**
     * @brief Detach the owned Carbonite profiler and close any remaining facade zones.
     *
     * @param[in] token Token returned by `isaacsimCommonProfilingAttachCarboniteHost()`.
     * @return Host detachment result.
     */
    ISAACSIM_COMMON_PROFILING_API IsaacSimCommonProfilingHostResult
    isaacsimCommonProfilingDetachCarboniteHost(IsaacSimCommonProfilingHostToken token);

    /** @brief Return whether an application currently owns the Carbonite profiler attachment. */
    ISAACSIM_COMMON_PROFILING_API uint32_t isaacsimCommonProfilingHasCarboniteHost(void);

#ifdef __cplusplus
}
#endif
