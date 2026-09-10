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

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

namespace isaacsim
{
namespace ros2
{
namespace nodes
{

/// Number of bytes the xyz position occupies at the start of every interleaved point (3 x float32).
constexpr size_t kPointCloudXyzBytes = 3 * sizeof(float);

/**
 * @brief Host-side interleave of xyz + per-point metadata fields into a contiguous
 *        PointCloud2-style buffer using carb::tasking for parallelism.
 * @details
 * Each point occupies @p pointWidth bytes in @p buffer. The first 12 bytes of every
 * point are the xyz position (3 consecutive floats); every entry of @p orderedFields
 * is gathered from its own per-point source array into the field's byte offset
 * within the point.
 *
 * @param[out] buffer Host buffer to fill; must hold at least `pointWidth * numPoints` bytes.
 * @param[in] pointCloudData Host pointer to the point positions, arranged as x, y, z per point.
 * @param[in] orderedFields Metadata fields as tuples of (source data pointer, per-point size in
 *                          bytes, byte offset within a point).
 * @param[in] pointWidth Width of a point in bytes (the PointCloud2 `point_step`).
 * @param[in] numPoints Number of points.
 *
 * @note This header is deliberately CUDA-free so the Python bindings can include it.
 *       The device-side twin for CUDA source buffers is \ref fillPointCloudBuffer in
 *       cuda/FillPointCloudBuffer.cuh.
 */
void fillPointCloudBufferHost(uint8_t* buffer,
                              const float* pointCloudData,
                              const std::vector<std::tuple<void*, size_t, size_t>>& orderedFields,
                              size_t pointWidth,
                              size_t numPoints);

} // namespace nodes
} // namespace ros2
} // namespace isaacsim
