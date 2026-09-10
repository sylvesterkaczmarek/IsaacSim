// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define ISAACSIM_COMMON_PROFILE_DETAIL_JOIN_INNER(a, b) a##b
#define ISAACSIM_COMMON_PROFILE_DETAIL_JOIN(a, b) ISAACSIM_COMMON_PROFILE_DETAIL_JOIN_INNER(a, b)
#define ISAACSIM_COMMON_PROFILE_DETAIL_LOCATION                                                                        \
    ::isaacsim::common::profiling::makeSourceLocation(__FILE__, __func__, static_cast<std::uint32_t>(__LINE__))

/** @brief Create a lazily admitted CPU zone for the current lexical scope. */
#define ISAACSIM_COMMON_PROFILE_ZONE(nameExpression)                                                                   \
    auto ISAACSIM_COMMON_PROFILE_DETAIL_JOIN(_isaacsimProfileZone_, __COUNTER__) =                                     \
        ::isaacsim::common::profiling::Zone::lazy(                                                                     \
            [&]() { return (nameExpression); }, 0, ISAACSIM_COMMON_PROFILE_DETAIL_LOCATION)

/** @brief Create a lazily admitted CPU zone with an explicit Carbonite capture mask. */
#define ISAACSIM_COMMON_PROFILE_ZONE_MASK(mask, nameExpression)                                                        \
    auto ISAACSIM_COMMON_PROFILE_DETAIL_JOIN(_isaacsimProfileZone_, __COUNTER__) =                                     \
        ::isaacsim::common::profiling::Zone::lazy(                                                                     \
            [&]() { return (nameExpression); }, (mask), ISAACSIM_COMMON_PROFILE_DETAIL_LOCATION)

/** @brief Profile the current function with its source name. */
#define ISAACSIM_COMMON_PROFILE_FUNCTION() ISAACSIM_COMMON_PROFILE_ZONE(__func__)

/** @brief Profile the current function with its source name and an explicit Carbonite capture mask. */
#define ISAACSIM_COMMON_PROFILE_FUNCTION_MASK(mask) ISAACSIM_COMMON_PROFILE_ZONE_MASK(mask, __func__)

/** @brief Emit a lazily admitted frame marker. */
#define ISAACSIM_COMMON_PROFILE_FRAME(nameExpression) ISAACSIM_COMMON_PROFILE_FRAME_MASK(0, nameExpression)

/** @brief Emit a lazily admitted frame marker with an explicit Carbonite capture mask. */
#define ISAACSIM_COMMON_PROFILE_FRAME_MASK(maskExpression, nameExpression)                                             \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto _isaacsimProfileMask = (maskExpression);                                                            \
        if (::isaacsim::common::profiling::isEnabled(_isaacsimProfileMask))                                            \
        {                                                                                                              \
            ::isaacsim::common::profiling::frame((nameExpression), _isaacsimProfileMask);                              \
        }                                                                                                              \
    } while (false)

/** @brief Emit a lazily admitted numeric value. */
#define ISAACSIM_COMMON_PROFILE_VALUE(nameExpression, selectedExpression)                                              \
    ISAACSIM_COMMON_PROFILE_VALUE_MASK(0, nameExpression, selectedExpression)

/** @brief Emit a lazily admitted numeric value with an explicit Carbonite capture mask. */
#define ISAACSIM_COMMON_PROFILE_VALUE_MASK(maskExpression, nameExpression, selectedExpression)                         \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto _isaacsimProfileMask = (maskExpression);                                                            \
        if (::isaacsim::common::profiling::isEnabled(_isaacsimProfileMask))                                            \
        {                                                                                                              \
            ::isaacsim::common::profiling::value((nameExpression), (selectedExpression), _isaacsimProfileMask);        \
        }                                                                                                              \
    } while (false)

/** @brief Emit a lazily admitted instant event on the calling thread timeline. */
#define ISAACSIM_COMMON_PROFILE_INSTANT(nameExpression) ISAACSIM_COMMON_PROFILE_INSTANT_MASK(0, nameExpression)

/** @brief Emit a lazily admitted thread instant with an explicit Carbonite capture mask. */
#define ISAACSIM_COMMON_PROFILE_INSTANT_MASK(maskExpression, nameExpression)                                           \
    ISAACSIM_COMMON_PROFILE_INSTANT_TYPE_MASK(                                                                         \
        maskExpression, ::isaacsim::common::profiling::InstantType::eThread, nameExpression)

/** @brief Emit a lazily admitted instant event on the selected timeline. */
#define ISAACSIM_COMMON_PROFILE_INSTANT_TYPE(typeExpression, nameExpression)                                           \
    ISAACSIM_COMMON_PROFILE_INSTANT_TYPE_MASK(0, typeExpression, nameExpression)

/** @brief Emit a lazily admitted instant on the selected timeline and capture mask. */
#define ISAACSIM_COMMON_PROFILE_INSTANT_TYPE_MASK(maskExpression, typeExpression, nameExpression)                      \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto _isaacsimProfileMask = (maskExpression);                                                            \
        if (::isaacsim::common::profiling::isEnabled(_isaacsimProfileMask))                                            \
        {                                                                                                              \
            ::isaacsim::common::profiling::instant(                                                                    \
                (nameExpression), (typeExpression), _isaacsimProfileMask, ISAACSIM_COMMON_PROFILE_DETAIL_LOCATION);    \
        }                                                                                                              \
    } while (false)

/** @brief Emit the lazily admitted beginning of a cross-thread flow. */
#define ISAACSIM_COMMON_PROFILE_FLOW_BEGIN(identifierExpression, nameExpression)                                       \
    ISAACSIM_COMMON_PROFILE_FLOW_BEGIN_MASK(0, identifierExpression, nameExpression)

/** @brief Emit a lazily admitted flow beginning with an explicit Carbonite capture mask. */
#define ISAACSIM_COMMON_PROFILE_FLOW_BEGIN_MASK(maskExpression, identifierExpression, nameExpression)                  \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto _isaacsimProfileMask = (maskExpression);                                                            \
        if (::isaacsim::common::profiling::isEnabled(_isaacsimProfileMask))                                            \
        {                                                                                                              \
            ::isaacsim::common::profiling::flow(true, (identifierExpression), (nameExpression), _isaacsimProfileMask,  \
                                                ISAACSIM_COMMON_PROFILE_DETAIL_LOCATION);                              \
        }                                                                                                              \
    } while (false)

/** @brief Emit the lazily admitted end of a cross-thread flow. */
#define ISAACSIM_COMMON_PROFILE_FLOW_END(identifierExpression)                                                         \
    ISAACSIM_COMMON_PROFILE_FLOW_END_MASK(0, identifierExpression)

/** @brief Emit a lazily admitted flow ending with an explicit Carbonite capture mask. */
#define ISAACSIM_COMMON_PROFILE_FLOW_END_MASK(maskExpression, identifierExpression)                                    \
    do                                                                                                                 \
    {                                                                                                                  \
        const auto _isaacsimProfileMask = (maskExpression);                                                            \
        if (::isaacsim::common::profiling::isEnabled(_isaacsimProfileMask))                                            \
        {                                                                                                              \
            ::isaacsim::common::profiling::flow(                                                                       \
                false, (identifierExpression), {}, _isaacsimProfileMask, ISAACSIM_COMMON_PROFILE_DETAIL_LOCATION);     \
        }                                                                                                              \
    } while (false)
