// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <functional>
#include <string>
#include <variant>

namespace ovsim
{
namespace interfaces
{
namespace control
{
namespace simulation
{

using InputParameterType = std::variant<bool, uintptr_t, int64_t, double, std::string>;
using OutputParameterType = std::variant<bool, uintptr_t, int64_t, double, std::string>;

using PlayFn = std::function<void()>;
using PauseFn = std::function<void()>;
using StopFn = std::function<void()>;

using InitializeFn = std::function<void()>;
using InvalidateFn = std::function<void()>;
using StepFn = std::function<void()>;

using SetParameterFn = std::function<void(
    const std::string& /*provider*/, const std::string& /*parameterName*/, const InputParameterType& /*value*/)>;
using GetParameterFn =
    std::function<OutputParameterType(const std::string& /*provider*/, const std::string& /*parameterName*/)>;

} // namespace simulation
} // namespace control
} // namespace interfaces
} // namespace ovsim
