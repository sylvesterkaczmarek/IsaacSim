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

#include <isaacsim/foundation/ovsim/Export.h>
#include <ovsim/interfaces/control/authoring/Authoring.hpp>

#include <string>

namespace isaacsim
{
namespace foundation
{
namespace ovsim
{
namespace control
{
namespace authoring
{

using ::ovsim::interfaces::control::authoring::InputParameterType;
using ::ovsim::interfaces::control::authoring::OutputParameterType;

ISAACSIM_FOUNDATION_OVSIM_API bool createStage();
ISAACSIM_FOUNDATION_OVSIM_API bool openStage(const std::string& usdPath);
ISAACSIM_FOUNDATION_OVSIM_API bool saveStage(const std::string& usdPath);
ISAACSIM_FOUNDATION_OVSIM_API bool importStageFromString(const std::string& usdString);
ISAACSIM_FOUNDATION_OVSIM_API std::string exportStageToString();
ISAACSIM_FOUNDATION_OVSIM_API bool closeStage();
ISAACSIM_FOUNDATION_OVSIM_API bool addReferenceToStage(const std::string& usdPath,
                                                       const std::string& path,
                                                       const std::string& typeName = "Xform");

ISAACSIM_FOUNDATION_OVSIM_API bool definePrim(const std::string& path, const std::string& typeName = "Xform");
ISAACSIM_FOUNDATION_OVSIM_API bool movePrim(const std::string& targetPath, const std::string& destinationPath);
ISAACSIM_FOUNDATION_OVSIM_API bool removePrim(const std::string& path);

ISAACSIM_FOUNDATION_OVSIM_API bool createPrimAttribute(const std::string& path,
                                                       const std::string& attributeName,
                                                       const std::string& typeName);
ISAACSIM_FOUNDATION_OVSIM_API bool removePrimAttribute(const std::string& path, const std::string& attributeName);

ISAACSIM_FOUNDATION_OVSIM_API void setParameter(const std::string& provider,
                                                const std::string& parameterName,
                                                const InputParameterType& value);
ISAACSIM_FOUNDATION_OVSIM_API OutputParameterType getParameter(const std::string& provider,
                                                               const std::string& parameterName);

} // namespace authoring
} // namespace control
} // namespace ovsim
} // namespace foundation
} // namespace isaacsim
