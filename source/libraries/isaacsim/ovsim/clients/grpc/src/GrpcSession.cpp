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

#include <isaacsim/ovsim/clients/grpc/GrpcSession.hpp>

#include <stdexcept>

namespace isaacsim
{
namespace ovsim
{
namespace clients
{
namespace grpc
{

GrpcSession::GrpcSession(const std::optional<std::unordered_map<std::string, std::string>>& configuration)
{
    if (!configuration || configuration->empty())
    {
        throw std::invalid_argument("GrpcSession requires a non-empty configuration.");
    }
    m_configuration = *configuration;
}

const std::unordered_map<std::string, std::string>& GrpcSession::configuration() const
{
    return m_configuration;
}

bool GrpcSession::createStage()
{
    throw std::logic_error("GrpcSession::createStage is not implemented yet");
}

bool GrpcSession::openStage(const std::string& /*usdPath*/)
{
    throw std::logic_error("GrpcSession::openStage is not implemented yet");
}

bool GrpcSession::saveStage(const std::string& /*usdPath*/)
{
    throw std::logic_error("GrpcSession::saveStage is not implemented yet");
}

bool GrpcSession::importStageFromString(const std::string& /*usdString*/)
{
    throw std::logic_error("GrpcSession::importStageFromString is not implemented yet");
}

std::string GrpcSession::exportStageToString()
{
    throw std::logic_error("GrpcSession::exportStageToString is not implemented yet");
}

bool GrpcSession::closeStage()
{
    throw std::logic_error("GrpcSession::closeStage is not implemented yet");
}

bool GrpcSession::addReferenceToStage(const std::string& /*usdPath*/,
                                      const std::string& /*path*/,
                                      const std::string& /*typeName*/)
{
    throw std::logic_error("GrpcSession::addReferenceToStage is not implemented yet");
}

bool GrpcSession::definePrim(const std::string& /*path*/, const std::string& /*typeName*/)
{
    throw std::logic_error("GrpcSession::definePrim is not implemented yet");
}

bool GrpcSession::movePrim(const std::string& /*targetPath*/, const std::string& /*destinationPath*/)
{
    throw std::logic_error("GrpcSession::movePrim is not implemented yet");
}

bool GrpcSession::removePrim(const std::string& /*path*/)
{
    throw std::logic_error("GrpcSession::removePrim is not implemented yet");
}

bool GrpcSession::createPrimAttribute(const std::string& /*path*/,
                                      const std::string& /*attributeName*/,
                                      const std::string& /*typeName*/)
{
    throw std::logic_error("GrpcSession::createPrimAttribute is not implemented yet");
}

bool GrpcSession::removePrimAttribute(const std::string& /*path*/, const std::string& /*attributeName*/)
{
    throw std::logic_error("GrpcSession::removePrimAttribute is not implemented yet");
}

void GrpcSession::authoringSetParameter(const std::string& /*provider*/,
                                        const std::string& /*parameterName*/,
                                        const InputParameterType& /*value*/)
{
    throw std::logic_error("GrpcSession::authoringSetParameter is not implemented yet");
}

OutputParameterType GrpcSession::authoringGetParameter(const std::string& /*provider*/,
                                                       const std::string& /*parameterName*/)
{
    throw std::logic_error("GrpcSession::authoringGetParameter is not implemented yet");
}

void GrpcSession::play()
{
    throw std::logic_error("GrpcSession::play is not implemented yet");
}

void GrpcSession::pause()
{
    throw std::logic_error("GrpcSession::pause is not implemented yet");
}

void GrpcSession::stop()
{
    throw std::logic_error("GrpcSession::stop is not implemented yet");
}

void GrpcSession::initialize()
{
    throw std::logic_error("GrpcSession::initialize is not implemented yet");
}

void GrpcSession::invalidate()
{
    throw std::logic_error("GrpcSession::invalidate is not implemented yet");
}

void GrpcSession::step()
{
    throw std::logic_error("GrpcSession::step is not implemented yet");
}

void GrpcSession::simulationSetParameter(const std::string& /*provider*/,
                                         const std::string& /*parameterName*/,
                                         const InputParameterType& /*value*/)
{
    throw std::logic_error("GrpcSession::simulationSetParameter is not implemented yet");
}

OutputParameterType GrpcSession::simulationGetParameter(const std::string& /*provider*/,
                                                        const std::string& /*parameterName*/)
{
    throw std::logic_error("GrpcSession::simulationGetParameter is not implemented yet");
}

OutputValueType GrpcSession::read(const PathType& /*paths*/,
                                  const std::string& /*attributeName*/,
                                  std::optional<double> /*timeStamp*/)
{
    throw std::logic_error("GrpcSession::read is not implemented yet");
}

void GrpcSession::write(const PathType& /*paths*/,
                        const std::string& /*attributeName*/,
                        const InputValueType& /*values*/,
                        std::optional<double> /*timeStamp*/)
{
    throw std::logic_error("GrpcSession::write is not implemented yet");
}

} // namespace grpc
} // namespace clients
} // namespace ovsim
} // namespace isaacsim
