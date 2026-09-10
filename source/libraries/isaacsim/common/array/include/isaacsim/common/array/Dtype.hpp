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

#include "isaacsim/common/array/Export.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <type_traits>

namespace isaacsim
{
namespace common
{
namespace array
{

/**
 * @class DType
 * @brief Represents the element data type of an array.
 * @details
 * A DType identifies the scalar kind (boolean, signed/unsigned integer, or
 * floating-point) together with its storage width. Instances are value types
 * and can be compared for equality.
 *
 * Instances can be constructed from a @ref Kind enumerator, a string name
 * (e.g. `"float32"`, `"int8"`), or via the named factory methods such as
 * @ref Float32() and @ref Int32(). The template factory @ref fromType() maps a
 * C++ type directly to its corresponding DType at compile time.
 */
class ISAACSIM_COMMON_ARRAY_API DType
{
public:
    /**
     * @brief Enumerates all scalar kinds supported as array element types.
     */
    enum class Kind
    {
        /**
         * @brief Boolean (1 byte).
         */
        eBool,

        /**
         * @brief Signed 8-bit integer.
         */
        eInt8,

        /**
         * @brief Signed 16-bit integer.
         */
        eInt16,

        /**
         * @brief Signed 32-bit integer.
         */
        eInt32,

        /**
         * @brief Signed 64-bit integer.
         */
        eInt64,

        /**
         * @brief Unsigned 8-bit integer.
         */
        eUInt8,

        /**
         * @brief Unsigned 16-bit integer.
         */
        eUInt16,

        /**
         * @brief Unsigned 32-bit integer.
         */
        eUInt32,

        /**
         * @brief Unsigned 64-bit integer.
         */
        eUInt64,

        /**
         * @brief IEEE 754 single-precision (32-bit) floating-point.
         */
        eFloat32,

        /**
         * @brief IEEE 754 double-precision (64-bit) floating-point.
         */
        eFloat64,
    };

    /**
     * @brief Constructs a DType from the given kind enumerator.
     * @param[in] kind The scalar kind to represent.
     */
    constexpr DType(Kind kind) : m_kind(kind)
    {
        // fromType() needs DType to be a literal type, which MSVC enforces strictly.
        // Therefore, it has to be defined in .hpp file.
    }

    /**
     * @brief Constructs a DType from a string name.
     * @details
     * Accepted names match the string produced by @ref toString(), e.g.
     * `"bool"`, `"int8"`, `"float32"`.
     *
     * @param[in] name String name of the data type.
     * @throws std::invalid_argument if @p name is not a recognized type name.
     */
    DType(const std::string& name);

    DType() = delete;
    /** @brief Copy-constructs a data type descriptor. */
    DType(const DType& other) = default;
    /** @brief Move-constructs a data type descriptor. */
    DType(DType&& other) = default;
    ~DType() = default;

    /** @brief Copy-assigns a data type descriptor. */
    DType& operator=(const DType& other) = default;
    /** @brief Move-assigns a data type descriptor. */
    DType& operator=(DType&& other) = default;

    /**
     * @brief Checks whether two DTypes represent the same scalar kind.
     * @param[in] other The DType to compare against.
     * @return `true` if both DTypes have equal kinds.
     */
    bool operator==(const DType& other) const;

    /**
     * @brief Checks whether two DTypes represent different scalar kinds.
     * @param[in] other The DType to compare against.
     * @return `true` if the DTypes have different kinds.
     */
    bool operator!=(const DType& other) const;

    /**
     * @brief Returns the scalar kind enumerator for this DType.
     * @return The @ref Kind value identifying the element type.
     */
    Kind kind() const;

    /**
     * @brief Returns the storage size of a single element.
     * @return Size in bytes.
     */
    size_t size() const;

    /**
     * @brief Returns whether this DType is a floating-point type.
     * @return `true` for @ref Kind::eFloat32 and @ref Kind::eFloat64.
     */
    bool isFloating() const;

    /**
     * @brief Returns whether this DType is an integral type.
     * @return `true` for all integer kinds.
     */
    bool isIntegral() const;

    /**
     * @brief Returns whether this DType is a signed type.
     * @return `true` for signed integers and floating-point types.
     */
    bool isSigned() const;

    /**
     * @brief Returns whether this DType is an unsigned type.
     * @return `true` for unsigned integer kinds.
     */
    bool isUnsigned() const;

    /**
     * @brief Returns the canonical string name for this DType.
     * @return A lowercase string such as `"bool"`, `"int8"`, or `"float32"`.
     */
    std::string toString() const;

    /**
     * @brief Constructs a DType from a string name.
     * @details
     * Equivalent to the string constructor. Accepted names match the output of
     * @ref toString().
     *
     * @param[in] name String name of the data type.
     * @return The corresponding DType.
     * @throws std::invalid_argument if @p name is not a recognized type name.
     */
    static DType fromString(const std::string& name);

    /**
     * @brief Constructs the DType corresponding to the C++ type @p T at compile time.
     * @tparam T A C++ scalar type that maps to a supported @ref Kind.
     * @return The DType for @p T.
     */
    template <typename T>
    static constexpr DType fromType()
    {
        if constexpr (std::is_same_v<T, bool>)
            return DType(Kind::eBool);
        else if constexpr (std::is_same_v<T, int8_t>)
            return DType(Kind::eInt8);
        else if constexpr (std::is_same_v<T, int16_t>)
            return DType(Kind::eInt16);
        else if constexpr (std::is_same_v<T, int32_t>)
            return DType(Kind::eInt32);
        else if constexpr (std::is_same_v<T, int64_t>)
            return DType(Kind::eInt64);
        else if constexpr (std::is_same_v<T, uint8_t>)
            return DType(Kind::eUInt8);
        else if constexpr (std::is_same_v<T, uint16_t>)
            return DType(Kind::eUInt16);
        else if constexpr (std::is_same_v<T, uint32_t>)
            return DType(Kind::eUInt32);
        else if constexpr (std::is_same_v<T, uint64_t>)
            return DType(Kind::eUInt64);
        else if constexpr (std::is_same_v<T, float>)
            return DType(Kind::eFloat32);
        else if constexpr (std::is_same_v<T, double>)
            return DType(Kind::eFloat64);
        else
            static_assert(sizeof(T) == 0, "DType::fromType(): unsupported scalar type");
    }

    /**
     * @brief Returns the DType for boolean values.
     * @return A DType with kind @ref Kind::eBool.
     */
    static DType Bool();

    /**
     * @brief Returns the DType for signed 8-bit integers.
     * @return A DType with kind @ref Kind::eInt8.
     */
    static DType Int8();

    /**
     * @brief Returns the DType for signed 16-bit integers.
     * @return A DType with kind @ref Kind::eInt16.
     */
    static DType Int16();

    /**
     * @brief Returns the DType for signed 32-bit integers.
     * @return A DType with kind @ref Kind::eInt32.
     */
    static DType Int32();

    /**
     * @brief Returns the DType for signed 64-bit integers.
     * @return A DType with kind @ref Kind::eInt64.
     */
    static DType Int64();

    /**
     * @brief Returns the DType for unsigned 8-bit integers.
     * @return A DType with kind @ref Kind::eUInt8.
     */
    static DType UInt8();

    /**
     * @brief Returns the DType for unsigned 16-bit integers.
     * @return A DType with kind @ref Kind::eUInt16.
     */
    static DType UInt16();

    /**
     * @brief Returns the DType for unsigned 32-bit integers.
     * @return A DType with kind @ref Kind::eUInt32.
     */
    static DType UInt32();

    /**
     * @brief Returns the DType for unsigned 64-bit integers.
     * @return A DType with kind @ref Kind::eUInt64.
     */
    static DType UInt64();

    /**
     * @brief Returns the DType for single-precision floating-point values.
     * @return A DType with kind @ref Kind::eFloat32.
     */
    static DType Float32();

    /**
     * @brief Returns the DType for double-precision floating-point values.
     * @return A DType with kind @ref Kind::eFloat64.
     */
    static DType Float64();

private:
    /**
     * @brief The scalar kind this DType represents.
     */
    Kind m_kind;
};

} // namespace array
} // namespace common
} // namespace isaacsim
