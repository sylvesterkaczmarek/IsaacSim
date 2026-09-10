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

void interleaveMeshVertices(const SceneMesh& mesh, std::vector<float>& interleaved)
{
    interleaved.assign(static_cast<size_t>(mesh.nvertices) * 15, 0.0f);
    for (int vertex = 0; vertex < mesh.nvertices; ++vertex)
    {
        float* destination = &interleaved[static_cast<size_t>(vertex) * 15];
        destination[0] = mesh.positions[vertex * 3 + 0];
        destination[1] = mesh.positions[vertex * 3 + 1];
        destination[2] = mesh.positions[vertex * 3 + 2];
        if (mesh.normals)
        {
            destination[3] = mesh.normals[vertex * 3 + 0];
            destination[4] = mesh.normals[vertex * 3 + 1];
            destination[5] = mesh.normals[vertex * 3 + 2];
        }
        else
        {
            destination[5] = 1.0f;
        }
        if (mesh.colors)
        {
            destination[6] = mesh.colors[vertex * 3 + 0];
            destination[7] = mesh.colors[vertex * 3 + 1];
            destination[8] = mesh.colors[vertex * 3 + 2];
        }
        if (mesh.texcoords)
        {
            destination[9] = mesh.texcoords[vertex * 2 + 0];
            /* stb_image returns the first image row at the start of the upload,
             * while USD Preview Surface UVs use the opposite V convention for
             * these OpenGL textures. NanoUSD's GLES path performs this same
             * conversion while packing its material vertex buffer. Keep the
             * source SceneMesh values untouched and convert only at the GPU
             * boundary so every texture slot sees one consistent convention. */
            destination[10] = 1.0f - mesh.texcoords[vertex * 2 + 1];
        }
        destination[14] = 1.0f;
    }

    /* Per-triangle tangents from the UV gradient (Lengyel). The geometry is a
     * triangle soup, so each triangle writes directly to its three vertices. */
    if (mesh.texcoords)
    {
        for (int triangle = 0; triangle + 2 < mesh.nvertices; triangle += 3)
        {
            float* v0 = &interleaved[static_cast<size_t>(triangle + 0) * 15];
            float* v1 = &interleaved[static_cast<size_t>(triangle + 1) * 15];
            float* v2 = &interleaved[static_cast<size_t>(triangle + 2) * 15];
            const float edge1[3] = { v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2] };
            const float edge2[3] = { v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2] };
            const float du1 = v1[9] - v0[9];
            const float dv1 = v1[10] - v0[10];
            const float du2 = v2[9] - v0[9];
            const float dv2 = v2[10] - v0[10];
            const float determinant = du1 * dv2 - du2 * dv1;
            float tangent[3];
            float bitangent[3] = { 0.0f, 0.0f, 0.0f };
            if (std::fabs(determinant) > 1e-12f)
            {
                const float reciprocal = 1.0f / determinant;
                tangent[0] = (edge1[0] * dv2 - edge2[0] * dv1) * reciprocal;
                tangent[1] = (edge1[1] * dv2 - edge2[1] * dv1) * reciprocal;
                tangent[2] = (edge1[2] * dv2 - edge2[2] * dv1) * reciprocal;
                bitangent[0] = (-edge1[0] * du2 + edge2[0] * du1) * reciprocal;
                bitangent[1] = (-edge1[1] * du2 + edge2[1] * du1) * reciprocal;
                bitangent[2] = (-edge1[2] * du2 + edge2[2] * du1) * reciprocal;
            }
            else
            {
                const float* normal = &v0[3];
                const float axis[3] = { std::fabs(normal[0]) < 0.9f ? 1.0f : 0.0f,
                                        std::fabs(normal[0]) < 0.9f ? 0.0f : 1.0f, 0.0f };
                tangent[0] = axis[1] * normal[2] - axis[2] * normal[1];
                tangent[1] = axis[2] * normal[0] - axis[0] * normal[2];
                tangent[2] = axis[0] * normal[1] - axis[1] * normal[0];
            }
            const float length = std::sqrt(tangent[0] * tangent[0] + tangent[1] * tangent[1] + tangent[2] * tangent[2]);
            const float reciprocal_length = length > 1e-12f ? 1.0f / length : 0.0f;
            for (float* destination : { v0, v1, v2 })
            {
                destination[11] = tangent[0] * reciprocal_length;
                destination[12] = tangent[1] * reciprocal_length;
                destination[13] = tangent[2] * reciprocal_length;
                if (reciprocal_length > 0.0f && std::fabs(determinant) > 1e-12f)
                {
                    const float cross_nt[3] = { destination[4] * destination[13] - destination[5] * destination[12],
                                                destination[5] * destination[11] - destination[3] * destination[13],
                                                destination[3] * destination[12] - destination[4] * destination[11] };
                    const float handedness =
                        cross_nt[0] * bitangent[0] + cross_nt[1] * bitangent[1] + cross_nt[2] * bitangent[2];
                    destination[14] = handedness < 0.0f ? -1.0f : 1.0f;
                }
            }
        }
    }
}

} // namespace ovgl
} // namespace details
} // namespace debug
} // namespace ovgl_viewport
} // namespace isaacsim
