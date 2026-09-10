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

#include "MeshClassification.hpp"

#include <cstdio>

namespace
{

bool expect_floor(const char* path, bool expected)
{
    const bool actual = isaacsim::ovgl_viewport::debug::details::ovgl::isFloorMesh(path);
    if (actual == expected)
        return true;
    // logging: allow printf - standalone Carb/Omni-free native test failure output.
    std::fprintf(stderr, "isFloorMesh(%s) returned %d, expected %d\n", path ? path : "<null>", actual, expected);
    return false;
}

} // namespace

int main()
{
    bool ok = true;
    ok &= expect_floor("/World/Floor", true);
    ok &= expect_floor("/World/ground_plane", true);
    ok &= expect_floor("/World/GroundPlane", true);
    ok &= expect_floor("/World/Plane001", true);
    ok &= expect_floor("/toy_biplane_idle/rig/biplane_mesh", false);
    ok &= expect_floor("/World/Airplane", false);
    ok &= expect_floor("/World/playground", false);
    ok &= expect_floor(nullptr, false);
    return ok ? 0 : 1;
}
