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

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

namespace isaacsim
{
namespace physics
{
namespace tensors
{

/**
 * @brief Flat metadata value returned by tensor views.
 *
 * A metadata value may be empty, a Boolean, a signed integer, a floating-point number, a string, an ordered list of
 * strings, or an ordered list of signed integers.
 */
using Metadata =
    std::variant<std::monostate, bool, int64_t, double, std::string, std::vector<std::string>, std::vector<int64_t>>;

} // namespace tensors
} // namespace physics
} // namespace isaacsim
