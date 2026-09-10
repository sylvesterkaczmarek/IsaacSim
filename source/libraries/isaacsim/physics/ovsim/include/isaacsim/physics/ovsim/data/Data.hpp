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

#include <isaacsim/physics/ovsim/Export.h>
#include <ovsim/interfaces/data/Data.hpp>

namespace isaacsim
{
namespace physics
{
namespace ovsim
{
namespace data
{

using ::ovsim::interfaces::data::InputValueType;
using ::ovsim::interfaces::data::OutputValueType;
using ::ovsim::interfaces::data::PathType;

/** @brief Read an attribute from one or more prims.
 * @param[in] paths Stage path or paths of the prims to read.
 * @param[in] attributeName Name of the attribute to read.
 * @param[in] timeStamp Optional USD time code for the read operation.
 * @return Attribute values for the requested prim paths.
 */
ISAACSIM_PHYSICS_OVSIM_API OutputValueType read(const PathType& paths,
                                                const std::string& attributeName,
                                                std::optional<double> timeStamp = std::nullopt);

/** @brief Write an attribute on one or more prims.
 * @param[in] paths Stage path or paths of the prims to update.
 * @param[in] attributeName Name of the attribute to write.
 * @param[in] values Attribute values to write.
 * @param[in] timeStamp Optional USD time code for the write operation.
 */
ISAACSIM_PHYSICS_OVSIM_API void write(const PathType& paths,
                                      const std::string& attributeName,
                                      const InputValueType& values,
                                      std::optional<double> timeStamp = std::nullopt);

} // namespace data
} // namespace ovsim
} // namespace physics
} // namespace isaacsim
