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

#include <unordered_map>
#include <utility>
#include <vector>

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

/* A read group may return the exact path-list handle used to create the query, in which case
 * its prim indices address the caller's original path order directly.  Cardinality is not an
 * identity check: a backend may return a different, equally-sized list in a different order.
 * Resolve and cache every distinct non-query handle so those indices address the right paths. */
template <typename ListHandle, typename Path, typename Resolver>
const std::vector<Path>* resolveBatchGroupPaths(ListHandle groupList,
                                                ListHandle originalQueryList,
                                                const std::vector<Path>& originalQueryPaths,
                                                std::unordered_map<ListHandle, std::vector<Path>>& resolved,
                                                Resolver&& resolver)
{
    if (groupList == originalQueryList)
        return &originalQueryPaths;

    auto it = resolved.find(groupList);
    if (it == resolved.end())
    {
        it = resolved.emplace(groupList, std::forward<Resolver>(resolver)(groupList)).first;
    }
    return &it->second;
}

} // namespace ovgl
} // namespace details
} // namespace debug
} // namespace ovgl_viewport
} // namespace isaacsim
