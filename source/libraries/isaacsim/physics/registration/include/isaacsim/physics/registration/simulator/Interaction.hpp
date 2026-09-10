// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "../Types.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>

namespace isaacsim
{
namespace physics
{
namespace registration
{

/**
 * @brief Identifies the value stored in a @ref DebugDataItem.
 *
 * Each enumerator specifies which value member of @ref DebugDataItem is meaningful.
 */
enum class DebugDataItemType : int
{
    /** @brief A floating-point value stored in @ref DebugDataItem::doubleValue. */
    eFloat,

    /** @brief A three-dimensional vector stored in @ref DebugDataItem::vector3Value. */
    eVector,

    /** @brief A three-dimensional point stored in @ref DebugDataItem::vector3Value. */
    ePoint,

    /** @brief A quaternion in x, y, z, w order stored in @ref DebugDataItem::vector4Value. */
    eQuaternion,

    /** @brief A string value stored in @ref DebugDataItem::stringValue. */
    eString,

    /** @brief A Boolean value stored in @ref DebugDataItem::booleanValue. */
    eBoolean,

    /** @brief An integer value stored in @ref DebugDataItem::integerValue. */
    eInteger,

    /** @brief An undefined or unsupported value for which no value member is meaningful. */
    eUndefined,
};

/**
 * @brief Describes one debug-data entry produced by @c GetPrimDebugDataFunction.
 *
 * Only the value member selected by @ref type is meaningful. Consumers must ignore all other value members.
 */
struct DebugDataItem
{
    /** @brief The type of value stored in this item. */
    DebugDataItemType type{ DebugDataItemType::eUndefined };

    /** @brief The human-readable description. */
    std::string description;

    /** @brief The Boolean value when @ref type is @c DebugDataItemType::eBoolean. */
    bool booleanValue{ false };

    /** @brief The integer value when @ref type is @c DebugDataItemType::eInteger. */
    int32_t integerValue{ 0 };

    /** @brief The floating-point value when @ref type is @c DebugDataItemType::eFloat. */
    double doubleValue{ 0.0 };

    /** @brief The string value when @ref type is @c DebugDataItemType::eString. */
    std::string stringValue;

    /** @brief The x, y, and z components of a vector or point. */
    double vector3Value[3]{ 0.0, 0.0, 0.0 };

    /** @brief The x, y, z, and w components of a quaternion. */
    double vector4Value[4]{ 0.0, 0.0, 0.0, 0.0 };
};

/**
 * @brief Debug-data entries keyed by their display names, such as "Position" or "Velocity".
 */
using DebugDataDictionary = std::unordered_map<std::string, DebugDataItem>;

/**
 * @brief Handles a raycast request from the interaction system.
 *
 * @param[in] origin The ray origin in world space.
 * @param[in] direction The ray direction in world space.
 * @param[in] inputActive @c true when the input control is engaged, such as while a mouse button is down; otherwise,
 *                        @c false.
 */
using HandleRaycastFunction = std::function<void(const Float3& origin, const Float3& direction, bool inputActive)>;

/**
 * @brief Retrieves simulation debug data for a USD prim.
 *
 * @param[in] primPath The USD prim path, such as @c "/World/Cube".
 * @return The prim's simulation debug data, or an empty dictionary when no data is available.
 */
using GetPrimDebugDataFunction = std::function<DebugDataDictionary(const std::string& primPath)>;

/**
 * @brief Function table for interaction operations.
 */
struct InteractionFunctions
{
    /** @brief Handles interaction raycast requests. */
    HandleRaycastFunction handleRaycast{};

    /** @brief Retrieves debug data for a USD prim. */
    GetPrimDebugDataFunction getPrimDebugData{};
};

} // namespace registration
} // namespace physics
} // namespace isaacsim
