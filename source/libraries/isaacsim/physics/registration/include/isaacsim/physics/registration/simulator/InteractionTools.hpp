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

#include "Interaction.hpp"

#include <string>

namespace isaacsim
{
namespace physics
{
namespace registration
{

/**
 * @brief Creates a double-precision debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The value to store.
 * @return A debug-data item of type @c DebugDataItemType::eFloat containing @p value.
 */
inline DebugDataItem makeDebugDataDouble(const std::string& description, double value)
{
    DebugDataItem item;
    item.type = DebugDataItemType::eFloat;
    item.description = description;
    item.doubleValue = value;
    return item;
}

/**
 * @brief Creates a single-precision debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The value to store.
 * @return A debug-data item of type @c DebugDataItemType::eFloat containing @p value.
 */
inline DebugDataItem makeDebugDataFloat(const std::string& description, float value)
{
    return makeDebugDataDouble(description, static_cast<double>(value));
}

/**
 * @brief Creates a double-precision three-dimensional vector debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The vector to store.
 * @return A debug-data item of type @c DebugDataItemType::eVector containing @p value.
 */
inline DebugDataItem makeDebugDataVector(const std::string& description, const Double3& value)
{
    DebugDataItem item;
    item.type = DebugDataItemType::eVector;
    item.description = description;
    item.vector3Value[0] = value.x;
    item.vector3Value[1] = value.y;
    item.vector3Value[2] = value.z;
    return item;
}

/**
 * @brief Creates a single-precision three-dimensional vector debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The vector to store.
 * @return A debug-data item of type @c DebugDataItemType::eVector containing @p value.
 */
inline DebugDataItem makeDebugDataVector(const std::string& description, const Float3& value)
{
    return makeDebugDataVector(description, Double3{ value.x, value.y, value.z });
}

/**
 * @brief Creates a double-precision three-dimensional point debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The point to store.
 * @return A debug-data item of type @c DebugDataItemType::ePoint containing @p value.
 */
inline DebugDataItem makeDebugDataPoint(const std::string& description, const Double3& value)
{
    DebugDataItem item = makeDebugDataVector(description, value);
    item.type = DebugDataItemType::ePoint;
    return item;
}

/**
 * @brief Creates a single-precision three-dimensional point debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The point to store.
 * @return A debug-data item of type @c DebugDataItemType::ePoint containing @p value.
 */
inline DebugDataItem makeDebugDataPoint(const std::string& description, const Float3& value)
{
    return makeDebugDataPoint(description, Double3{ value.x, value.y, value.z });
}

/**
 * @brief Creates a double-precision quaternion debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The quaternion to store in x, y, z, w order.
 * @return A debug-data item of type @c DebugDataItemType::eQuaternion containing @p value.
 */
inline DebugDataItem makeDebugDataQuaternion(const std::string& description, const Double4& value)
{
    DebugDataItem item;
    item.type = DebugDataItemType::eQuaternion;
    item.description = description;
    item.vector4Value[0] = value.x;
    item.vector4Value[1] = value.y;
    item.vector4Value[2] = value.z;
    item.vector4Value[3] = value.w;
    return item;
}

/**
 * @brief Creates a single-precision quaternion debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The quaternion to store in x, y, z, w order.
 * @return A debug-data item of type @c DebugDataItemType::eQuaternion containing @p value.
 */
inline DebugDataItem makeDebugDataQuaternion(const std::string& description, const Float4& value)
{
    return makeDebugDataQuaternion(description, Double4{ value.x, value.y, value.z, value.w });
}

/**
 * @brief Creates a string debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The string to store.
 * @return A debug-data item of type @c DebugDataItemType::eString containing @p value.
 */
inline DebugDataItem makeDebugDataString(const std::string& description, const std::string& value)
{
    DebugDataItem item;
    item.type = DebugDataItemType::eString;
    item.description = description;
    item.stringValue = value;
    return item;
}

/**
 * @brief Creates a Boolean debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The Boolean value to store.
 * @return A debug-data item of type @c DebugDataItemType::eBoolean containing @p value.
 */
inline DebugDataItem makeDebugDataBoolean(const std::string& description, bool value)
{
    DebugDataItem item;
    item.type = DebugDataItemType::eBoolean;
    item.description = description;
    item.booleanValue = value;
    return item;
}

/**
 * @brief Creates an integer debug-data item.
 *
 * @param[in] description The human-readable description of the value.
 * @param[in] value The integer value to store.
 * @return A debug-data item of type @c DebugDataItemType::eInteger containing @p value.
 */
inline DebugDataItem makeDebugDataInteger(const std::string& description, int32_t value)
{
    DebugDataItem item;
    item.type = DebugDataItemType::eInteger;
    item.description = description;
    item.integerValue = value;
    return item;
}

} // namespace registration
} // namespace physics
} // namespace isaacsim
