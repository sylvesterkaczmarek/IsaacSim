// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace ovsim
{
namespace interfaces
{
namespace details
{

inline std::string _formatAttributeErrorMessage(const std::string& attributeName,
                                                const std::optional<std::vector<std::string>>& validAttributeNames)
{
    std::string message = "Invalid attribute name: '" + attributeName + "'";
    if (validAttributeNames && !validAttributeNames->empty())
    {
        message += ". Valid attribute names: ";
        for (size_t i = 0; i < validAttributeNames->size(); ++i)
        {
            message += (i == 0 ? "'" : ", '") + (*validAttributeNames)[i] + "'";
        }
    }
    return message;
}

class InitializationError : public std::runtime_error
{
public:
    InitializationError() : std::runtime_error("Not initialized")
    {
    }

    explicit InitializationError(const std::string& component)
        : std::runtime_error("Not initialized: " + component), m_component(component)
    {
    }

    [[nodiscard]] const std::string& component() const noexcept
    {
        return m_component;
    }

private:
    std::string m_component;
};

class AttributeError : public std::runtime_error
{
public:
    explicit AttributeError(const std::string& attributeName,
                            const std::optional<std::vector<std::string>>& validAttributeNames = std::nullopt)
        : std::runtime_error(_formatAttributeErrorMessage(attributeName, validAttributeNames)),
          m_attributeName(attributeName),
          m_validAttributeNames(validAttributeNames ? *validAttributeNames : std::vector<std::string>())
    {
    }

    [[nodiscard]] const std::string& attributeName() const noexcept
    {
        return m_attributeName;
    }

    [[nodiscard]] const std::vector<std::string>& validAttributeNames() const noexcept
    {
        return m_validAttributeNames;
    }

private:
    std::string m_attributeName;
    std::vector<std::string> m_validAttributeNames;
};

} // namespace details
} // namespace interfaces
} // namespace ovsim
