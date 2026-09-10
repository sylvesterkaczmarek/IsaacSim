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

#include <cstddef>
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

void buildMeshSoup(const float* points,
                   size_t pointCount,
                   const int* faceCounts,
                   size_t faceCount,
                   const int* faceIndices,
                   size_t faceIndexCount,
                   const float* normals,
                   size_t normalCount,
                   const float* texcoords,
                   size_t texcoordCount,
                   const int* texcoordIndices,
                   size_t texcoordIndexCount,
                   std::vector<float>& outputPoints,
                   std::vector<float>& outputNormals,
                   std::vector<float>& outputTexcoords);

} // namespace ovgl
} // namespace details
} // namespace debug
} // namespace ovgl_viewport
} // namespace isaacsim
