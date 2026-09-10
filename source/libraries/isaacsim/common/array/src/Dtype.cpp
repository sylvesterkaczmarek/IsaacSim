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

#include "isaacsim/common/array/Dtype.hpp"

#include <stdexcept>

namespace isaacsim
{
namespace common
{
namespace array
{

DType::DType(const std::string& name) : m_kind(fromString(name).kind())
{
}

bool DType::operator==(const DType& other) const
{
    return m_kind == other.m_kind;
}

bool DType::operator!=(const DType& other) const
{
    return m_kind != other.m_kind;
}

DType::Kind DType::kind() const
{
    return m_kind;
}

size_t DType::size() const
{
    switch (m_kind)
    {
    case Kind::eBool:
        return sizeof(bool);
    case Kind::eInt8:
        return sizeof(int8_t);
    case Kind::eInt16:
        return sizeof(int16_t);
    case Kind::eInt32:
        return sizeof(int32_t);
    case Kind::eInt64:
        return sizeof(int64_t);
    case Kind::eUInt8:
        return sizeof(uint8_t);
    case Kind::eUInt16:
        return sizeof(uint16_t);
    case Kind::eUInt32:
        return sizeof(uint32_t);
    case Kind::eUInt64:
        return sizeof(uint64_t);
    case Kind::eFloat32:
        return sizeof(float);
    case Kind::eFloat64:
        return sizeof(double);
    }
    return 0;
}

bool DType::isFloating() const
{
    return m_kind == Kind::eFloat32 || m_kind == Kind::eFloat64;
}

bool DType::isIntegral() const
{
    return !isFloating() && m_kind != Kind::eBool;
}

bool DType::isSigned() const
{
    switch (m_kind)
    {
    case Kind::eInt8:
    case Kind::eInt16:
    case Kind::eInt32:
    case Kind::eInt64:
    case Kind::eFloat32:
    case Kind::eFloat64:
        return true;
    default:
        return false;
    }
}

bool DType::isUnsigned() const
{
    switch (m_kind)
    {
    case Kind::eUInt8:
    case Kind::eUInt16:
    case Kind::eUInt32:
    case Kind::eUInt64:
        return true;
    default:
        return false;
    }
}

std::string DType::toString() const
{
    switch (m_kind)
    {
    case Kind::eBool:
        return "bool";
    case Kind::eInt8:
        return "int8";
    case Kind::eInt16:
        return "int16";
    case Kind::eInt32:
        return "int32";
    case Kind::eInt64:
        return "int64";
    case Kind::eUInt8:
        return "uint8";
    case Kind::eUInt16:
        return "uint16";
    case Kind::eUInt32:
        return "uint32";
    case Kind::eUInt64:
        return "uint64";
    case Kind::eFloat32:
        return "float32";
    case Kind::eFloat64:
        return "float64";
    }
    return "";
}

DType DType::fromString(const std::string& name)
{
    if (name == "bool")
        return DType(Kind::eBool);
    else if (name == "int8")
        return DType(Kind::eInt8);
    else if (name == "int16")
        return DType(Kind::eInt16);
    else if (name == "int32")
        return DType(Kind::eInt32);
    else if (name == "int64")
        return DType(Kind::eInt64);
    else if (name == "uint8")
        return DType(Kind::eUInt8);
    else if (name == "uint16")
        return DType(Kind::eUInt16);
    else if (name == "uint32")
        return DType(Kind::eUInt32);
    else if (name == "uint64")
        return DType(Kind::eUInt64);
    else if (name == "float32")
        return DType(Kind::eFloat32);
    else if (name == "float64")
        return DType(Kind::eFloat64);
    throw std::invalid_argument("Unknown dtype: '" + std::string(name) + "'");
}

DType DType::Bool()
{
    return DType(Kind::eBool);
}

DType DType::Int8()
{
    return DType(Kind::eInt8);
}

DType DType::Int16()
{
    return DType(Kind::eInt16);
}

DType DType::Int32()
{
    return DType(Kind::eInt32);
}

DType DType::Int64()
{
    return DType(Kind::eInt64);
}

DType DType::UInt8()
{
    return DType(Kind::eUInt8);
}

DType DType::UInt16()
{
    return DType(Kind::eUInt16);
}

DType DType::UInt32()
{
    return DType(Kind::eUInt32);
}

DType DType::UInt64()
{
    return DType(Kind::eUInt64);
}

DType DType::Float32()
{
    return DType(Kind::eFloat32);
}

DType DType::Float64()
{
    return DType(Kind::eFloat64);
}

} // namespace array
} // namespace common
} // namespace isaacsim
