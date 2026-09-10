// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isaacsim/common/profiling/Profiling.h"

#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

namespace isaacsim
{
namespace common
{
namespace profiling
{

using Mask = IsaacSimCommonProfilingMask;
using SourceLocation = IsaacSimCommonProfilingSourceLocation;

/** @brief Timeline occupied by an instant event. */
enum class InstantType
{
    eThread = ISAACSIM_COMMON_PROFILING_INSTANT_THREAD,
    eProcess = ISAACSIM_COMMON_PROFILING_INSTANT_PROCESS,
};

/** @brief Return whether the attached Carbonite profiler admits the selected mask. */
ISAACSIM_COMMON_PROFILING_API bool isEnabled(Mask mask = 0) noexcept;

/** @brief Create source metadata for one event. */
inline SourceLocation makeSourceLocation(const char* file = nullptr,
                                         const char* function = nullptr,
                                         std::uint32_t line = 0) noexcept
{
    SourceLocation location = isaacsimCommonProfilingMakeSourceLocation();
    location.file = file;
    location.function = function;
    location.line = line;
    return location;
}

/** @brief Movable RAII ownership for one admitted CPU zone. */
class ISAACSIM_COMMON_PROFILING_API Zone final
{
public:
    Zone() noexcept = default;

    /**
     * @brief Begin a CPU zone if the attached profiler admits the selected mask.
     *
     * @param[in] name Profiler-visible zone name.
     * @param[in] mask Carbonite capture mask. Zero selects the default mask.
     * @param[in] location Source metadata for the zone.
     */
    Zone(std::string_view name, Mask mask = 0, SourceLocation location = makeSourceLocation()) noexcept;
    ~Zone() noexcept;

    /**
     * @brief Transfer ownership of an active zone.
     *
     * @param[in,out] other Zone to move from. It is inactive after the transfer.
     */
    Zone(Zone&& other) noexcept;

    /**
     * @brief End the current zone and transfer ownership from another zone.
     *
     * @param[in,out] other Zone to move from. It is inactive after the transfer.
     * @return Reference to this zone.
     */
    Zone& operator=(Zone&& other) noexcept;
    Zone(const Zone&) = delete;
    Zone& operator=(const Zone&) = delete;

    /** @brief End the zone if it is active. Repeated calls are harmless. */
    void close() noexcept;

    /** @brief Return whether a Carbonite profiler admitted this zone. */
    explicit operator bool() const noexcept;

    /** @brief Create a zone without evaluating the name factory when profiling is disabled. */
    template <typename NameFactory>
    static Zone lazy(NameFactory&& nameFactory, Mask mask = 0, SourceLocation location = makeSourceLocation()) noexcept
    {
        if (!isEnabled(mask))
        {
            return {};
        }
        try
        {
            std::string name(std::forward<NameFactory>(nameFactory)());
            return Zone(name, mask, location);
        }
        catch (...)
        {
            return {};
        }
    }

private:
    explicit Zone(IsaacSimCommonProfilingZone zone) noexcept;

    IsaacSimCommonProfilingZone m_zone{ 0 };
};

/** @brief Emit a frame marker for the calling thread. */
ISAACSIM_COMMON_PROFILING_API void frame(std::string_view name, Mask mask = 0) noexcept;

/** @brief Emit a signed integer value. */
ISAACSIM_COMMON_PROFILING_API void value(std::string_view name, std::int32_t selected, Mask mask = 0) noexcept;

/** @brief Emit a floating-point value. */
ISAACSIM_COMMON_PROFILING_API void value(std::string_view name, float selected, Mask mask = 0) noexcept;

/** @brief Emit an instant event. */
ISAACSIM_COMMON_PROFILING_API void instant(std::string_view name,
                                           InstantType type = InstantType::eThread,
                                           Mask mask = 0,
                                           SourceLocation location = makeSourceLocation()) noexcept;

/** @brief Emit one endpoint of a cross-thread flow. */
ISAACSIM_COMMON_PROFILING_API void flow(bool begin,
                                        std::uint64_t identifier,
                                        std::string_view name = {},
                                        Mask mask = 0,
                                        SourceLocation location = makeSourceLocation()) noexcept;

/** @brief Assign a profiler-visible name to the calling thread. */
ISAACSIM_COMMON_PROFILING_API void setThreadName(std::string_view name) noexcept;

} // namespace profiling
} // namespace common
} // namespace isaacsim

#include "isaacsim/common/profiling/details/ProfileMacros.hpp"
