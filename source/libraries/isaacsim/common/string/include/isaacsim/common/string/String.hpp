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

#include "isaacsim/common/string/Export.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace isaacsim
{
namespace common
{
namespace string
{

/**
 * @brief Hash functor for `std::vector<std::string>`, e.g. for use as an `std::unordered_map` key hasher.
 */
struct ISAACSIM_COMMON_STRING_API VectorStringHasher
{
    /**
     * @brief Compute a combined hash of all strings in @p paths.
     * @param[in] paths The vector of strings to hash.
     * @return The combined hash value.
     */
    std::size_t operator()(const std::vector<std::string>& paths) const;
};

/**
 * @brief Convert all characters in a string to lowercase.
 * @param[in] text The string to convert.
 * @return A copy of @p text with all characters converted to lowercase.
 */
ISAACSIM_COMMON_STRING_API std::string toLower(const std::string& text);

/**
 * @brief Convert all characters in a string to uppercase.
 * @param[in] text The string to convert.
 * @return A copy of @p text with all characters converted to uppercase.
 */
ISAACSIM_COMMON_STRING_API std::string toUpper(const std::string& text);

/**
 * @brief Join a vector of strings with a separator.
 * @param[in] strings The vector of strings to join.
 * @param[in] separator The separator to use between strings.
 * @return A string with the joined strings.
 */
ISAACSIM_COMMON_STRING_API std::string join(const std::vector<std::string>& strings, const std::string& separator);

/**
 * @brief Split a string into a vector of strings using a separator.
 * @param[in] text The string to split.
 * @param[in] separator The separator to use to split the string.
 * @return A vector of strings with the split strings.
 */
ISAACSIM_COMMON_STRING_API std::vector<std::string> split(const std::string& text, const std::string& separator);

/**
 * @brief Sort a vector of strings.
 * @param[in] strings The vector of strings to sort.
 * @param[in] ascending Whether to sort in ascending order.
 * @return A vector of strings with the sorted strings.
 */
ISAACSIM_COMMON_STRING_API std::vector<std::string> sort(const std::vector<std::string>& strings, bool ascending = true);

/**
 * @brief Resolve a matplotlib-style color specification to a vector of 3 normalized RGB float components.
 * @details
 * Supported inputs (mirroring matplotlib's `to_rgb`):
 *  - `#rgb` / `#rrggbb` short and full hex (case-insensitive; `#rgba` / `#rrggbbaa` accepted, alpha discarded)
 *  - grayscale strings such as `"0.5"` (a float in the 0-1 range)
 *  - single-letter base colors such as `"k"`
 *  - X11/CSS4 named colors such as `"AquaMarine"` (case-insensitive)
 *  - Tableau colors such as `"tab:Green"` (case-insensitive)
 *  - the `CN` cycle specification such as `"C2"`
 *  - `"none"` (case-insensitive), the fully transparent color, resolved to black `{0, 0, 0}`
 *
 * @param[in] color The color specification to resolve.
 * @return RGB components, each in the 0-1 range (e.g. `{0.2f, 0.5f, 1.0f}`).
 * @throws std::invalid_argument if @p color is not a supported/recognized specification.
 */
ISAACSIM_COMMON_STRING_API std::vector<float> resolveColor(const std::string& color);

} // namespace string
} // namespace common
} // namespace isaacsim
