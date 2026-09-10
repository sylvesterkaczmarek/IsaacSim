// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <carb/tasking/ITasking.h>

#include <isaacsim/ros2/nodes/FillPointCloudBufferHost.hpp>

#include <cstring>
#include <tuple>
#include <vector>

namespace isaacsim
{
namespace ros2
{
namespace nodes
{

void fillPointCloudBufferHost(uint8_t* buffer,
                              const float* pointCloudData,
                              const std::vector<std::tuple<void*, size_t, size_t>>& orderedFields,
                              const size_t pointWidth,
                              const size_t numPoints)
{
    if (numPoints == 0)
    {
        return;
    }

    // With no metadata and a packed layout the interleave degenerates to one contiguous copy
    if (orderedFields.empty() && pointWidth == kPointCloudXyzBytes)
    {
        memcpy(buffer, pointCloudData, numPoints * kPointCloudXyzBytes);
        return;
    }

    auto tasking = carb::getCachedInterface<carb::tasking::ITasking>();

    // Parallel batches of points: each invocation handles a contiguous span so the per-point
    // copy loop stays local and inlinable (parallelFor would dispatch per index through the
    // C ABI). Writes are cache-friendly since each point writes a contiguous pointWidth block.
    tasking->applyRangeBatch(
        numPoints, 0,
        [buffer, pointCloudData, &orderedFields, pointWidth](size_t begin, size_t end)
        {
            for (size_t i = begin; i < end; ++i)
            {
                uint8_t* dst = buffer + i * pointWidth;
                // Copy xyz
                memcpy(dst, reinterpret_cast<const uint8_t*>(pointCloudData + i * 3), kPointCloudXyzBytes);
                // Copy each metadata field for this point
                for (const auto& [data, dataSize, offset] : orderedFields)
                {
                    memcpy(dst + offset, reinterpret_cast<const uint8_t*>(data) + i * dataSize, dataSize);
                }
            }
        });
}

} // namespace nodes
} // namespace ros2
} // namespace isaacsim
