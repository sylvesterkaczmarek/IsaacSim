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

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace isaacsim
{
namespace common
{
namespace test
{

/**
 * @brief Resolve a resource path below the test-resource directory for a specific module.
 *
 * Each module's TEST_RESOURCES.md must begin with a line of the form
 * `module: <module-name>` so this helper can locate the correct directory when
 * multiple modules' source trees are visible under the current working directory.
 *
 * @param[in] moduleName Module identifier that must match the first line of TEST_RESOURCES.md.
 * @param[in] path       Path relative to the directory containing `TEST_RESOURCES.md`.
 * @return Resolved resource path.
 * @throws std::runtime_error If no matching marker or the requested resource cannot be found.
 */
inline std::string resolveResourcePath(const std::string& moduleName, const std::string& path)
{
    const std::string expectedFirstLine = "module: " + moduleName;
    const std::filesystem::path cwd = std::filesystem::current_path();
    for (const auto& entry :
         std::filesystem::recursive_directory_iterator(cwd, std::filesystem::directory_options::skip_permission_denied))
    {
        if (!entry.is_regular_file() || entry.path().filename() != "TEST_RESOURCES.md")
        {
            continue;
        }
        std::ifstream file(entry.path());
        std::string firstLine;
        if (!std::getline(file, firstLine) || firstLine != expectedFirstLine)
        {
            continue;
        }
        std::string resourcePath = (entry.path().parent_path() / path).string();
        if (std::filesystem::exists(resourcePath))
        {
            return resourcePath;
        }
        throw std::runtime_error("Resource path '" + resourcePath + "' does not exist");
    }
    throw std::runtime_error("Unable to resolve test resources path: no TEST_RESOURCES.md with 'module: " + moduleName +
                             "' was found under '" + cwd.string() + "'");
}

} // namespace test
} // namespace common
} // namespace isaacsim
