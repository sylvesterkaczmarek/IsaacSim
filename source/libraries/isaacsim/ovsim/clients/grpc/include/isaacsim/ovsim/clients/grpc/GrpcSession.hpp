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

#include <isaacsim/ovsim/clients/grpc/Export.h>
#include <ovsim/interfaces/control/authoring/Authoring.hpp>
#include <ovsim/interfaces/control/simulation/Simulation.hpp>
#include <ovsim/interfaces/data/Data.hpp>

#include <optional>
#include <string>

namespace isaacsim
{
namespace ovsim
{
namespace clients
{
namespace grpc
{

using PathType = ::ovsim::interfaces::data::PathType;
using InputValueType = ::ovsim::interfaces::data::InputValueType;
using OutputValueType = ::ovsim::interfaces::data::OutputValueType;
using InputParameterType = ::ovsim::interfaces::control::authoring::InputParameterType;
using OutputParameterType = ::ovsim::interfaces::control::authoring::OutputParameterType;

/** @brief Remote OV SIM client session.
 * @warning Remote operations are not implemented and currently throw `std::logic_error`.
 */
class ISAACSIM_OVSIM_CLIENTS_GRPC_API GrpcSession
{
public:
    /** @brief Create a remote OV SIM session.
     * @param[in] configuration Non-empty implementation-specific connection configuration.
     * @throws std::invalid_argument If `configuration` is absent or empty.
     */
    explicit GrpcSession(const std::optional<std::unordered_map<std::string, std::string>>& configuration = std::nullopt);

    /** @brief Get the connection configuration supplied at construction.
     * @return Session connection configuration.
     */
    const std::unordered_map<std::string, std::string>& configuration() const;

    // authoring
    /** @copydoc isaacsim::physics::ovsim::control::authoring::createStage
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool createStage();

    /** @copydoc isaacsim::physics::ovsim::control::authoring::openStage
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool openStage(const std::string& usdPath);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::saveStage
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool saveStage(const std::string& usdPath);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::importStageFromString
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool importStageFromString(const std::string& usdString);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::exportStageToString
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    std::string exportStageToString();

    /** @copydoc isaacsim::physics::ovsim::control::authoring::closeStage
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool closeStage();

    /** @copydoc isaacsim::physics::ovsim::control::authoring::addReferenceToStage
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool addReferenceToStage(const std::string& usdPath, const std::string& path, const std::string& typeName);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::definePrim
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool definePrim(const std::string& path, const std::string& typeName);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::movePrim
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool movePrim(const std::string& targetPath, const std::string& destinationPath);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::removePrim
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool removePrim(const std::string& path);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::createPrimAttribute
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool createPrimAttribute(const std::string& path, const std::string& attributeName, const std::string& typeName);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::removePrimAttribute
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    bool removePrimAttribute(const std::string& path, const std::string& attributeName);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::setParameter
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    void authoringSetParameter(const std::string& provider,
                               const std::string& parameterName,
                               const InputParameterType& value);

    /** @copydoc isaacsim::physics::ovsim::control::authoring::getParameter
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    OutputParameterType authoringGetParameter(const std::string& provider, const std::string& parameterName);

    // simulation
    /** @copydoc isaacsim::physics::ovsim::control::simulation::play
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    void play();

    /** @copydoc isaacsim::physics::ovsim::control::simulation::pause
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    void pause();

    /** @copydoc isaacsim::physics::ovsim::control::simulation::stop
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    void stop();

    /** @copydoc isaacsim::physics::ovsim::control::simulation::initialize
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    void initialize();

    /** @copydoc isaacsim::physics::ovsim::control::simulation::invalidate
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    void invalidate();

    /** @copydoc isaacsim::physics::ovsim::control::simulation::step
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    void step();

    /** @copydoc isaacsim::physics::ovsim::control::simulation::setParameter
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    void simulationSetParameter(const std::string& provider,
                                const std::string& parameterName,
                                const InputParameterType& value);

    /** @copydoc isaacsim::physics::ovsim::control::simulation::getParameter
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    OutputParameterType simulationGetParameter(const std::string& provider, const std::string& parameterName);

    // data
    /** @copydoc isaacsim::physics::ovsim::data::read
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    OutputValueType read(const PathType& paths, const std::string& attributeName, std::optional<double> timeStamp);

    /** @copydoc isaacsim::physics::ovsim::data::write
     * @throws std::logic_error Always, because the remote operation is not implemented.
     */
    void write(const PathType& paths,
               const std::string& attributeName,
               const InputValueType& values,
               std::optional<double> timeStamp);

private:
    std::unordered_map<std::string, std::string> m_configuration;
};

} // namespace grpc
} // namespace clients
} // namespace ovsim
} // namespace isaacsim
