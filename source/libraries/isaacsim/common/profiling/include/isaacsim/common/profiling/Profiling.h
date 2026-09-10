// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isaacsim/common/profiling/Export.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief Carbonite capture-mask bits assigned to an event. Zero selects the default mask. */
    typedef uint64_t IsaacSimCommonProfilingMask;

    /** @brief Opaque ownership for one active CPU zone. */
    typedef uint64_t IsaacSimCommonProfilingZone;

    /** @brief Timeline occupied by an instant event. */
    typedef enum IsaacSimCommonProfilingInstantType
    {
        ISAACSIM_COMMON_PROFILING_INSTANT_THREAD = 0,
        ISAACSIM_COMMON_PROFILING_INSTANT_PROCESS = 1,
    } IsaacSimCommonProfilingInstantType;

    /** @brief Borrowed source metadata for one profiling event. */
    typedef struct IsaacSimCommonProfilingSourceLocation
    {
        size_t structSize; /**< Size of this structure for ABI validation. */
        const char* file; /**< Null-terminated source filename, or NULL. */
        const char* function; /**< Null-terminated function name, or NULL. */
        uint32_t line; /**< One-based source line, or zero. */
    } IsaacSimCommonProfilingSourceLocation;

    /** @brief Create empty source metadata with the ABI size initialized. */
    static inline IsaacSimCommonProfilingSourceLocation isaacsimCommonProfilingMakeSourceLocation(void)
    {
#ifdef __cplusplus
        IsaacSimCommonProfilingSourceLocation location{};
#else
    IsaacSimCommonProfilingSourceLocation location = { 0 };
#endif
        location.structSize = sizeof(location);
        return location;
    }

    /** @brief Return whether the attached Carbonite profiler admits the selected mask. */
    ISAACSIM_COMMON_PROFILING_API uint32_t isaacsimCommonProfilingIsEnabled(IsaacSimCommonProfilingMask mask);

    /** @brief Begin a CPU zone and return zero when profiling is disabled or the input is invalid. */
    ISAACSIM_COMMON_PROFILING_API IsaacSimCommonProfilingZone isaacsimCommonProfilingBegin(
        IsaacSimCommonProfilingMask mask, const char* name, const IsaacSimCommonProfilingSourceLocation* location);

    /** @brief End a zone returned by `isaacsimCommonProfilingBegin()`. Zero is ignored. */
    ISAACSIM_COMMON_PROFILING_API void isaacsimCommonProfilingEnd(IsaacSimCommonProfilingZone zone);

    /** @brief Emit a frame marker for the calling thread. */
    ISAACSIM_COMMON_PROFILING_API void isaacsimCommonProfilingFrame(IsaacSimCommonProfilingMask mask, const char* name);

    /** @brief Emit a signed integer value. */
    ISAACSIM_COMMON_PROFILING_API void isaacsimCommonProfilingValueInt(IsaacSimCommonProfilingMask mask,
                                                                       const char* name,
                                                                       int32_t value);

    /** @brief Emit a floating-point value. */
    ISAACSIM_COMMON_PROFILING_API void isaacsimCommonProfilingValueFloat(IsaacSimCommonProfilingMask mask,
                                                                         const char* name,
                                                                         float value);

    /** @brief Emit an instant event. */
    ISAACSIM_COMMON_PROFILING_API void isaacsimCommonProfilingInstant(IsaacSimCommonProfilingMask mask,
                                                                      int32_t type,
                                                                      const char* name,
                                                                      const IsaacSimCommonProfilingSourceLocation* location);

    /** @brief Emit one endpoint of a cross-thread flow. */
    ISAACSIM_COMMON_PROFILING_API void isaacsimCommonProfilingFlow(IsaacSimCommonProfilingMask mask,
                                                                   uint32_t begin,
                                                                   uint64_t identifier,
                                                                   const char* name,
                                                                   const IsaacSimCommonProfilingSourceLocation* location);

    /** @brief Assign a profiler-visible name to the calling thread. */
    ISAACSIM_COMMON_PROFILING_API void isaacsimCommonProfilingSetThreadName(const char* name);

#ifdef __cplusplus
}
#endif
