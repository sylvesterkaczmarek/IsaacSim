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

#include "gl/Scene.h"

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

/* Pack a triangle-soup SceneMesh into the 15-float PBR vertex layout. This is
 * the OpenGL convention boundary: source USD UVs stay unchanged in SceneMesh,
 * while packed V is flipped for stb_image's top-down image rows. */
void interleaveMeshVertices(const SceneMesh& mesh, std::vector<float>& interleaved);

} // namespace ovgl
} // namespace details
} // namespace debug
} // namespace ovgl_viewport
} // namespace isaacsim
