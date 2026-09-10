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

#pragma once

#include "isaacsim/common/logging/Export.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /** @brief Result of changing the process host integration. */
    typedef enum IsaacSimCommonLoggingHostResult
    {
        /** @brief The host integration change completed successfully. */
        ISAACSIM_COMMON_LOGGING_HOST_SUCCESS = 0,
        /** @brief A required pointer or token was invalid. */
        ISAACSIM_COMMON_LOGGING_HOST_INVALID_ARGUMENT = 1,
        /** @brief Logging had already selected a backend. */
        ISAACSIM_COMMON_LOGGING_HOST_ALREADY_INITIALIZED = 2,
        /** @brief The token does not own the current host registration. */
        ISAACSIM_COMMON_LOGGING_HOST_NOT_OWNER = 3,
    } IsaacSimCommonLoggingHostResult;

    /** @brief Opaque token identifying one borrowed host registration. */
    typedef uint64_t IsaacSimCommonLoggingHostToken;

    /**
     * @brief Use an existing Carbonite logging interface without taking framework ownership.
     *
     * This integration API is intended for a native host adapter that attaches before any logging,
     * admission query, or global configuration operation selects the standalone backend. Logger and
     * channel construction do not select a backend and may occur before host attachment. The pointed-to
     * Carbonite interface remains owned by the host. The logging library does not acquire, release, or
     * configure the host framework during registration.
     *
     * @param[in] carboniteLogging Borrowed `carb::logging::ILogging*` represented as an opaque pointer.
     * @param[out] token Token required to detach this registration.
     * @return Host integration result.
     */
    ISAACSIM_COMMON_LOGGING_API IsaacSimCommonLoggingHostResult
    isaacsimCommonLoggingAttachCarboniteHost(void* carboniteLogging, IsaacSimCommonLoggingHostToken* token);

    /**
     * @brief Detach the borrowed Carbonite interface owned by `token`.
     *
     * Detaching waits for active submissions, removes channels registered by this library, and
     * disables subsequent submissions. It never releases the Carbonite framework.
     *
     * @param[in] token Token returned by `isaacsimCommonLoggingAttachCarboniteHost`.
     * @return Host integration result.
     */
    ISAACSIM_COMMON_LOGGING_API IsaacSimCommonLoggingHostResult
    isaacsimCommonLoggingDetachCarboniteHost(IsaacSimCommonLoggingHostToken token);

#ifdef __cplusplus
}
#endif
