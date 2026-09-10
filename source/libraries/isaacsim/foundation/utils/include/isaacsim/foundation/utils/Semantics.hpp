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

#include <isaacsim/foundation/utils/Export.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>


namespace isaacsim
{
namespace foundation
{
namespace utils
{

/**
 * @brief Adds semantic labels to a prim under the given taxonomy.
 * @details
 * Applies the @c SemanticsLabelsAPI schema to the prim at @p path (if not already applied),
 * then merges @p labels into the @c semantics:labels:taxonomy attribute. Labels already
 * present on the prim are not duplicated; only new, unique entries are appended.
 *
 * Only the prim at @p path is modified; descendants are not affected.
 *
 * @param[in] path USD path of the target prim on the default stage.
 * @param[in] labels A single label string or a list of label strings to add.
 * @param[in] taxonomy Name of the semantic taxonomy under which labels are stored (e.g.
 *            @c "class", @c "type"). Defaults to @c "class".
 *
 * @see getLabels(), removeLabels()
 */
ISAACSIM_FOUNDATION_UTILS_API void addLabels(const std::string& path,
                                             const std::variant<std::string, std::vector<std::string>>& labels,
                                             const std::string& taxonomy = "class");

/**
 * @brief Returns all semantic labels on a prim, grouped by taxonomy.
 * @details
 * Inspects the applied schemas of the prim at @p path (and, when @p includeDescendants is @c true,
 * all prims in its subtree) for @c SemanticsLabelsAPI instances. For each found taxonomy the
 * corresponding label values are read and collected.
 *
 * When @p includeDescendants is @c true, labels from the prim and all its descendants are merged
 * into the same result map: for each taxonomy the returned vector is the concatenation of labels
 * from all visited prims in traversal order, which may contain duplicates across prims.
 *
 * @param[in] path USD path of the target prim on the default stage.
 * @param[in] includeDescendants When @c true, the subtree rooted at @p path is traversed and
 *            labels from all prims are aggregated.
 *
 * @return Map from taxonomy name to the list of label strings for that taxonomy.
 *
 * @see addLabels(), removeLabels()
 */
ISAACSIM_FOUNDATION_UTILS_API std::unordered_map<std::string, std::vector<std::string>> getLabels(
    const std::string& path, bool includeDescendants = false);

/**
 * @brief Removes specific semantic labels from a prim.
 * @details
 * For each taxonomy that has @c SemanticsLabelsAPI applied, any label in @p labels is removed
 * from the corresponding @c semantics:labels:taxonomy attribute. When @p taxonomy is set only
 * that taxonomy is affected; otherwise all taxonomies on the prim are examined.
 *
 * The @c SemanticsLabelsAPI schema itself is not removed from the prim, even if the label list
 * becomes empty after the operation. Use removeAllLabels() with @p removeTaxonomies set to remove
 * the schema as well.
 *
 * @param[in] path USD path of the target prim on the default stage.
 * @param[in] labels A single label string or a list of label strings to remove.
 * @param[in] taxonomy When set, restricts removal to the named taxonomy. When @c std::nullopt,
 *            the labels are removed from every taxonomy found on the prim.
 * @param[in] includeDescendants When @c true, the same removal is applied to all prims in the
 *            subtree rooted at @p path.
 *
 * @see addLabels(), removeAllLabels()
 */
ISAACSIM_FOUNDATION_UTILS_API void removeLabels(const std::string& path,
                                                const std::variant<std::string, std::vector<std::string>>& labels,
                                                const std::optional<std::string>& taxonomy = std::nullopt,
                                                bool includeDescendants = false);

/**
 * @brief Removes all semantic labels from a prim.
 * @details
 * Iterates over every @c SemanticsLabelsAPI instance applied to the prim at @p path and either
 * clears the label list or removes the schema entirely, depending on @p removeTaxonomies.
 *
 * When @p removeTaxonomies is @c false the @c semantics:labels:taxonomy attribute is set to an
 * empty list but the @c SemanticsLabelsAPI schema remains applied. When @p removeTaxonomies is
 * @c true the schema itself is removed from the prim via @c removeApi.
 *
 * The applied-schema list is snapshotted before any modification to avoid iterator invalidation
 * during schema removal.
 *
 * @param[in] path USD path of the target prim on the default stage.
 * @param[in] removeTaxonomies When @c true, also removes the @c SemanticsLabelsAPI schema for
 *            each taxonomy. When @c false, only clears the label values.
 * @param[in] includeDescendants When @c true, the operation is applied to all prims in the
 *            subtree rooted at @p path.
 *
 * @see addLabels(), removeLabels()
 */
ISAACSIM_FOUNDATION_UTILS_API void removeAllLabels(const std::string& path,
                                                   bool removeTaxonomies = false,
                                                   bool includeDescendants = false);

} // namespace utils
} // namespace foundation
} // namespace isaacsim
