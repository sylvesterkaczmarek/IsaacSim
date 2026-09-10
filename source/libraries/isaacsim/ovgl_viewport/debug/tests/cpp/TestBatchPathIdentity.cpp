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

#include "BatchPaths.hpp"

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#define CHECK(condition)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        if (!(condition))                                                                                              \
        {                                                                                                              \
            std::fprintf(stderr, "CHECK failed at line %d: %s\n", __LINE__, #condition);                               \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main()
{
    using Handle = uint64_t;
    const Handle query_list = 11;
    const Handle reordered_list = 22;
    const Handle same_order_other_list = 33;
    const std::vector<std::string> query_paths{ "/A", "/B", "/C" };
    const std::vector<std::string> reordered_paths{ "/C", "/A", "/B" };

    std::unordered_map<Handle, std::vector<std::string>> resolved;
    int resolve_calls = 0;
    bool resolver_valid = true;
    auto resolver = [&](Handle handle)
    {
        ++resolve_calls;
        if (handle == reordered_list)
            return reordered_paths;
        if (handle == same_order_other_list)
            return query_paths;
        resolver_valid = false;
        return std::vector<std::string>{};
    };

    /* Equal cardinality must not select query order when the returned handle differs. */
    const auto* reordered = isaacsim::ovgl_viewport::debug::details::ovgl::resolveBatchGroupPaths(
        reordered_list, query_list, query_paths, resolved, resolver);
    CHECK(resolver_valid);
    CHECK(reordered != &query_paths);
    CHECK(*reordered == reordered_paths);
    CHECK(resolve_calls == 1);

    /* The distinct list is resolved once, then served from the per-read cache. */
    const auto* reordered_again = isaacsim::ovgl_viewport::debug::details::ovgl::resolveBatchGroupPaths(
        reordered_list, query_list, query_paths, resolved, resolver);
    CHECK(reordered_again == reordered);
    CHECK(resolve_calls == 1);

    /* Even an equal-cardinality, same-order list needs resolution when its handle differs. */
    const auto* same_order_other = isaacsim::ovgl_viewport::debug::details::ovgl::resolveBatchGroupPaths(
        same_order_other_list, query_list, query_paths, resolved, resolver);
    CHECK(same_order_other != &query_paths);
    CHECK(*same_order_other == query_paths);
    CHECK(resolve_calls == 2);

    /* Only exact query-list identity may bypass path resolution. */
    const auto* original = isaacsim::ovgl_viewport::debug::details::ovgl::resolveBatchGroupPaths(
        query_list, query_list, query_paths, resolved, resolver);
    CHECK(original == &query_paths);
    CHECK(resolve_calls == 2);
    return 0;
}
