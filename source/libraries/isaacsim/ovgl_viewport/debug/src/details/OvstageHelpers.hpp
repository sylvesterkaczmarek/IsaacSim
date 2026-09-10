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

#include <ovstage/ovstage.h>
#include <ovx/path_dictionary/path_dictionary.h>
#include <ovx/path_dictionary/path_dictionary_utils.h>

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

namespace isaacsim
{
namespace ovgl_viewport
{
namespace debug
{
namespace details
{

inline bool waitAndRelease(ovstage_instance_t* stage, const ovstage_enqueue_result_t& operation)
{
    if (!stage || operation.status != OVSTAGE_OK || operation.op_index == OVSTAGE_INVALID_OP_ID)
        return false;
    ovstage_op_wait_result_t waitResult{};
    const ovstage_api_status_t waitStatus =
        ovstage_wait_op(stage, operation.op_index, OVSTAGE_TIMEOUT_INFINITE, &waitResult);
    const ovstage_api_status_t releaseStatus = ovstage_release_op(stage, operation.op_index);
    return waitStatus == OVSTAGE_OK && waitResult.error_op_id_count == 0 && releaseStatus == OVSTAGE_OK;
}

// Resolve a dictionary-owned path list through the public path-dictionary vtable.
inline std::vector<std::string> resolvePrimPaths(path_dictionary_instance_t* dictionary, ovx_primpath_list_t list)
{
    std::vector<std::string> result;
    if (!dictionary || !dictionary->vtable || list == OVX_INVALID_PRIMPATH_LIST)
        return result;

    auto* vtable = dictionary->vtable;
    auto* context = dictionary->context;
    size_t pathCount = 0;
    if (vtable->get_num_paths_from_path_list(context, list, &pathCount).status != OVX_API_SUCCESS || pathCount == 0)
        return result;

    std::vector<ovx_primpath_t> handles(pathCount);
    size_t filled = 0;
    while (filled < pathCount)
    {
        size_t count = 0;
        if (vtable->get_paths_from_path_list(context, list, filled, pathCount - filled, handles.data() + filled, &count)
                    .status != OVX_API_SUCCESS ||
            count == 0)
            return {};
        filled += count;
    }

    std::vector<ovx_token_t> tokenBuffer(pathCount * 8 + 8);
    std::vector<ovx_token_t*> pathTokens(pathCount, nullptr);
    std::vector<size_t> pathTokenCounts(pathCount, 0);
    size_t processed = 0;
    for (;;)
    {
        if (vtable
                ->get_tokens_from_paths(context, handles.data(), pathCount, tokenBuffer.data(), tokenBuffer.size(),
                                        pathTokens.data(), pathTokenCounts.data(), &processed)
                .status != OVX_API_SUCCESS)
            return {};
        if (processed >= pathCount)
            break;
        tokenBuffer.resize(tokenBuffer.size() * 2);
    }

    result.reserve(pathCount);
    for (size_t i = 0; i < pathCount; ++i)
    {
        std::vector<ovx_string_t> components(pathTokenCounts[i]);
        if (pathTokenCounts[i] != 0 &&
            vtable->get_strings_from_tokens(context, pathTokens[i], pathTokenCounts[i], components.data()).status !=
                OVX_API_SUCCESS)
            return {};
        std::string path;
        for (const ovx_string_t& component : components)
        {
            path.push_back('/');
            path.append(component.ptr, component.length);
        }
        result.push_back(std::move(path));
    }
    return result;
}

// Enumerate the paths matched by a query using readable value groups. Query
// result handles and path-list handles intentionally belong to different ID spaces.
inline std::optional<std::vector<std::string>> resolveQueryPrimPaths(ovstage_instance_t* stage,
                                                                     ovstage_query_handle_t query,
                                                                     const ovx_token_t* attributes,
                                                                     size_t attributeCount,
                                                                     ovstage_ordinal_t ordinal)
{
    std::vector<std::string> result;
    std::unordered_set<std::string> seen;
    if (!stage || query == OVSTAGE_INVALID_QUERY_HANDLE || (attributeCount != 0 && !attributes))
        return std::nullopt;
    path_dictionary_instance_t* dictionary = ovstage_get_path_dictionary(stage);
    if (!dictionary)
        return std::nullopt;

    ovstage_ordinal_range_t range{};
    range.has_start_ordinal = false;
    range.end_ordinal = ordinal;

    const auto readAttribute = [&](ovx_token_t token) -> bool
    {
        if (token == OVX_INVALID_TOKEN)
            return false;
        ovstage_read_handle_t read = OVSTAGE_INVALID_READ_HANDLE;
        const ovstage_enqueue_result_t operation = ovstage_read_attributes(stage, query, &token, 1, range, &read);
        if (operation.status != OVSTAGE_OK || read == OVSTAGE_INVALID_READ_HANDLE)
            return false;

        bool succeeded = true;

        while (succeeded)
        {
            ovstage_read_group_t group{};
            const ovstage_api_status_t fetched = ovstage_fetch_read_next(stage, read, OVSTAGE_TIMEOUT_INFINITE, &group);
            if (fetched == OVSTAGE_ERROR_END_OF_ITERATION)
                break;
            if (fetched != OVSTAGE_OK)
            {
                succeeded = false;
                break;
            }
            if (!group.is_delete && group.prims.count != 0 && group.prims.list != OVX_INVALID_PRIMPATH_LIST)
            {
                std::vector<std::string> groupPaths = resolvePrimPaths(dictionary, group.prims.list);
                for (size_t i = 0; i < group.prims.count; ++i)
                {
                    const size_t pathIndex = group.prims.index_map ? static_cast<size_t>(group.prims.index_map[i]) :
                                                                     static_cast<size_t>(group.prims.offset) + i;
                    if (pathIndex >= groupPaths.size() || groupPaths[pathIndex].empty())
                    {
                        succeeded = false;
                        break;
                    }
                    if (seen.insert(groupPaths[pathIndex]).second)
                        result.push_back(std::move(groupPaths[pathIndex]));
                }
            }
            if (ovstage_release_group(stage, &group) != OVSTAGE_OK)
                succeeded = false;
        }
        if (!waitAndRelease(stage, operation))
            succeeded = false;
        if (!waitAndRelease(stage, ovstage_release_read(stage, read)))
            succeeded = false;
        return succeeded;
    };

    const ovx_string_t typeName{ "usd-prim-type", sizeof("usd-prim-type") - 1 };
    ovx_token_t typeToken = OVX_INVALID_TOKEN;
    if (path_dictionary_create_tokens_from_strings(dictionary, &typeName, 1, &typeToken).status != OVX_API_SUCCESS ||
        typeToken == OVX_INVALID_TOKEN || !readAttribute(typeToken))
        return std::nullopt;
    if (result.empty())
    {
        for (size_t i = 0; i < attributeCount; ++i)
            if (attributes[i] != typeToken)
                if (!readAttribute(attributes[i]))
                    return std::nullopt;
    }
    return result;
}

} // namespace details
} // namespace debug
} // namespace ovgl_viewport
} // namespace isaacsim
