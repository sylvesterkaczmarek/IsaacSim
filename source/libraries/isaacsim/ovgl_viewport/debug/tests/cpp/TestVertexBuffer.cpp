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

#include "VertexBuffer.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{

bool near(float actual, float expected)
{
    return std::fabs(actual - expected) <= 1e-6f;
}

bool expect(const char* label, float actual, float expected)
{
    if (near(actual, expected))
        return true;
    // logging: allow printf - standalone Carb/Omni-free native test failure output.
    std::fprintf(stderr, "%s: got %.9g, expected %.9g\n", label, actual, expected);
    return false;
}

} // namespace

int main()
{
    float positions[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
    float normals[] = { 0, 0, 1, 0, 0, 1, 0, 0, 1 };
    float texcoords[] = { 0, 0, 1, 0, 0, 1 };
    uint32_t indices[] = { 0, 1, 2 };
    SceneMesh mesh;
    std::memset(&mesh, 0, sizeof(mesh));
    mesh.positions = positions;
    mesh.normals = normals;
    mesh.texcoords = texcoords;
    mesh.indices = indices;
    mesh.nvertices = 3;
    mesh.nindices = 3;

    std::vector<float> packed;
    isaacsim::ovgl_viewport::debug::details::ovgl::interleaveMeshVertices(mesh, packed);
    bool ok = packed.size() == 45;
    if (!ok)
    {
        // logging: allow printf - standalone Carb/Omni-free native test failure output.
        std::fprintf(stderr, "packed size: got %zu, expected 45\n", packed.size());
    }
    if (!ok)
        return 1;

    ok &= expect("v0.v", packed[10], 1.0f);
    ok &= expect("v1.v", packed[25], 1.0f);
    ok &= expect("v2.v", packed[40], 0.0f);
    for (int vertex = 0; vertex < 3; ++vertex)
    {
        const size_t offset = static_cast<size_t>(vertex) * 15;
        ok &= expect("tangent.x", packed[offset + 11], 1.0f);
        ok &= expect("tangent.y", packed[offset + 12], 0.0f);
        ok &= expect("tangent.z", packed[offset + 13], 0.0f);
        ok &= expect("tangent.sign", packed[offset + 14], -1.0f);
    }
    return ok ? 0 : 1;
}
