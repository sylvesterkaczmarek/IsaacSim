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

// Shared ovphysx result -> exception helper for the tensors backend TUs.
// Kept separate from OvPhysxTensorHelpers.hpp, which must remain free of ovphysx
// headers so the in-process doctest suite can compile those helpers alone.

#include <ovphysx/ovphysx.h>

#include <stdexcept>
#include <string>

namespace isaacsim
{
namespace physics_engines
{
namespace ovphysx
{

inline void checkOvphysxResult(ovphysx_result_t result, const char* context)
{
    if (result.status != OVPHYSX_API_SUCCESS)
    {
        ovphysx_string_t error = ovphysx_get_last_error();
        std::string message =
            (error.ptr && error.length > 0) ? std::string(error.ptr, error.length) : std::string("unknown error");
        throw std::runtime_error(std::string(context) + ": " + message);
    }
}

} // namespace ovphysx
} // namespace physics_engines
} // namespace isaacsim
