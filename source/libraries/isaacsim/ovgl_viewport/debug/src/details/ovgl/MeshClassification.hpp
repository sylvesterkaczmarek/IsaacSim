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

#include <cctype>
#include <cstring>

namespace isaacsim
{
namespace ovgl_viewport
{
namespace debug
{
namespace details
{
namespace ovgl
{

inline bool hasPathNameToken(const char* path, const char* token)
{
    if (!path || !token || !token[0])
        return false;
    const size_t token_size = std::strlen(token);
    for (const char* candidate = path; *candidate; ++candidate)
    {
        size_t index = 0;
        while (index < token_size && candidate[index] &&
               std::tolower(static_cast<unsigned char>(candidate[index])) ==
                   std::tolower(static_cast<unsigned char>(token[index])))
        {
            ++index;
        }
        if (index != token_size)
            continue;

        const unsigned char first = static_cast<unsigned char>(*candidate);
        const unsigned char previous = candidate == path ? 0 : static_cast<unsigned char>(candidate[-1]);
        const unsigned char next = static_cast<unsigned char>(candidate[token_size]);
        const bool starts_token =
            candidate == path || !std::isalpha(previous) || (std::isupper(first) && std::islower(previous));
        const bool ends_token =
            next == 0 || !std::isalpha(next) ||
            (std::isupper(next) && std::islower(static_cast<unsigned char>(candidate[token_size - 1])));
        if (starts_token && ends_token)
            return true;
    }
    return false;
}

/* The optional procedural floor checker matches complete path-name tokens, so
 * an asset name such as "biplane" is not mistaken for a floor named "plane". */
inline bool isFloorMesh(const char* path)
{
    return hasPathNameToken(path, "floor") || hasPathNameToken(path, "ground") || hasPathNameToken(path, "plane");
}

} // namespace ovgl
} // namespace details
} // namespace debug
} // namespace ovgl_viewport
} // namespace isaacsim
