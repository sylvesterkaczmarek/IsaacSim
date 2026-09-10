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

#include <isaacsim/ovsim/clients/local/Export.h>
#include <ovsim/interfaces/control/authoring/Authoring.hpp>

#include <string>

namespace isaacsim
{
namespace ovsim
{
namespace clients
{
namespace local
{
namespace control
{
namespace authoring
{

using ::ovsim::interfaces::control::authoring::InputParameterType;
using ::ovsim::interfaces::control::authoring::OutputParameterType;

/** @copydoc isaacsim::physics::ovsim::control::authoring::createStage */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool createStage();

/** @copydoc isaacsim::physics::ovsim::control::authoring::openStage */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool openStage(const std::string& usdPath);

/** @copydoc isaacsim::physics::ovsim::control::authoring::saveStage */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool saveStage(const std::string& usdPath);

/** @copydoc isaacsim::physics::ovsim::control::authoring::importStageFromString */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool importStageFromString(const std::string& usdString);

/** @copydoc isaacsim::physics::ovsim::control::authoring::exportStageToString */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API std::string exportStageToString();

/** @copydoc isaacsim::physics::ovsim::control::authoring::closeStage */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool closeStage();

/** @copydoc isaacsim::physics::ovsim::control::authoring::addReferenceToStage */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool addReferenceToStage(const std::string& usdPath,
                                                          const std::string& path,
                                                          const std::string& typeName = "Xform");

/** @copydoc isaacsim::physics::ovsim::control::authoring::definePrim */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool definePrim(const std::string& path, const std::string& typeName = "Xform");

/** @copydoc isaacsim::physics::ovsim::control::authoring::movePrim */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool movePrim(const std::string& targetPath, const std::string& destinationPath);

/** @copydoc isaacsim::physics::ovsim::control::authoring::removePrim */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool removePrim(const std::string& path);

/** @copydoc isaacsim::physics::ovsim::control::authoring::createPrimAttribute */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool createPrimAttribute(const std::string& path,
                                                          const std::string& attributeName,
                                                          const std::string& typeName);

/** @copydoc isaacsim::physics::ovsim::control::authoring::removePrimAttribute */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API bool removePrimAttribute(const std::string& path, const std::string& attributeName);

/** @copydoc isaacsim::physics::ovsim::control::authoring::setParameter */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API void setParameter(const std::string& provider,
                                                   const std::string& parameterName,
                                                   const InputParameterType& value);

/** @copydoc isaacsim::physics::ovsim::control::authoring::getParameter */
ISAACSIM_OVSIM_CLIENTS_LOCAL_API OutputParameterType getParameter(const std::string& provider,
                                                                  const std::string& parameterName);

} // namespace authoring
} // namespace control
} // namespace local
} // namespace clients
} // namespace ovsim
} // namespace isaacsim
