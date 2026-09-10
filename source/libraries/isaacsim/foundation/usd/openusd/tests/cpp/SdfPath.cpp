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

#include <doctest/doctest.h>
#include <isaacsim/foundation/usd/openusd/Usd.hpp>

using namespace isaacsim::foundation::usd::openusd;

TEST_SUITE("SdfPath")
{

    TEST_CASE("isValidPathString")
    {
        // Absolute paths
        CHECK_UNARY(isValidPathString("/World"));
        CHECK_UNARY(isValidPathString("/World/A/B"));
        CHECK_UNARY(isValidPathString("/"));
        // Relative paths are syntactically valid
        CHECK_UNARY(isValidPathString("World"));
        CHECK_UNARY(isValidPathString("A/B/C"));
        // Invalid paths
        CHECK_UNARY_FALSE(isValidPathString("/World/"));
        CHECK_UNARY_FALSE(isValidPathString("?"));
        CHECK_UNARY_FALSE(isValidPathString(""));
    }
}
