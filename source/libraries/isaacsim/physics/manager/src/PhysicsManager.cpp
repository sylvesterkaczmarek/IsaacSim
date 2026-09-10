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

#include <isaacsim/physics/manager/PhysicsManager.hpp>
#include <isaacsim/physics/manager/PhysicsSimulation.hpp>
#include <isaacsim/physics/registration/Physics.hpp>
#include <isaacsim/physics/registration/tensors/TensorRegistry.hpp>

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <vector>

namespace isaacsim
{
namespace physics
{
namespace manager
{

namespace registration = isaacsim::physics::registration;

namespace
{

std::string _toLower(std::string value)
{
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

} // namespace

PhysicsManager& PhysicsManager::getInstance()
{
    static PhysicsManager instance;
    return instance;
}

bool PhysicsManager::isInitialized() const
{
    return m_initialized;
}

void PhysicsManager::setup(float dt)
{
    m_dt = dt;
    m_simulatedTime = 0.0f;
    m_simulatedPhysicsSteps = 0;
}

bool PhysicsManager::initialize(void* ovstageInstancePtr, int64_t usdStageId)
{
    // Check if there are any simulations registered
    const size_t numSimulations = registration::getNumberOfSimulations();
    if (!numSimulations)
    {
        return false;
    }
    // Check if there are any active simulations
    std::vector<registration::SimulationId> simulationIds(numSimulations);
    simulationIds.resize(registration::getSimulationIds(simulationIds.data(), simulationIds.size()));
    const bool anyActiveSimulation =
        std::any_of(simulationIds.begin(), simulationIds.end(),
                    [](const registration::SimulationId& id) { return registration::isSimulationActive(id); });
    if (!anyActiveSimulation)
    {
        return false;
    }

    const std::string usdIdentifier = std::to_string(usdStageId);
    const InitializeResult result = isaacsim::physics::manager::initialize(ovstageInstancePtr, usdIdentifier.c_str());
    if (result == InitializeResult::eOk)
    {
        m_initialized = true;
        m_simulatedTime = 0.0f;
        m_simulatedPhysicsSteps = 0;
        return true;
    }
    return false;
}

bool PhysicsManager::invalidate()
{
    if (!isaacsim::physics::manager::close())
    {
        return false;
    }
    m_initialized = false;
    m_simulatedTime = 0.0f;
    m_simulatedPhysicsSteps = 0;
    return true;
}

int PhysicsManager::step(int steps, std::function<bool(int, int)> callback)
{
    if (!m_initialized)
    {
        throw std::runtime_error("PhysicsManager::step called before initialize()");
    }
    // Step the physics simulation
    int step = 0;
    for (; step < steps; ++step)
    {
        isaacsim::physics::manager::simulate(m_dt, m_simulatedTime);
        m_simulatedTime += m_dt;
        ++m_simulatedPhysicsSteps;

        if (callback && !callback(step + 1, steps))
        {
            return step + 1;
        }
    }
    return step;
}

float PhysicsManager::getSimulatedTime() const
{
    return m_simulatedTime;
}

int PhysicsManager::getSimulatedPhysicsSteps() const
{
    return m_simulatedPhysicsSteps;
}

size_t PhysicsManager::registerCallback(SimulationEventFunction callback, PhysicsEvent event, int order)
{
    const size_t uid = ++m_callbackUid;

    switch (event)
    {
    case PhysicsEvent::ePhysicsPreStep:
    case PhysicsEvent::ePhysicsPostStep:
    {
        auto onStep = [callback = std::move(callback)](float elapsedTime, const registration::PhysicsStepContext& context)
        {
            SimulationEventPayload payload;
            payload.emplace("elapsed_time", elapsedTime);
            payload.emplace("scene_path", static_cast<size_t>(context.scenePath.path));
            payload.emplace("simulation_id", context.simulationId.id);
            callback(std::move(payload));
        };
        registration::SubscriptionId subscriptionId = isaacsim::physics::manager::subscribePhysicsOnStepEvents(
            event == PhysicsEvent::ePhysicsPreStep, order, std::move(onStep));
        if (subscriptionId != registration::g_kInvalidSubscriptionId)
        {
            m_callbacks[uid] = subscriptionId;
            return uid;
        }
        break;
    }
    default:
        break;
    }

    return 0;
}

bool PhysicsManager::deregisterCallback(size_t uid) noexcept
{
    auto iterator = m_callbacks.find(uid);
    if (iterator == m_callbacks.end())
    {
        return false;
    }
    try
    {
        isaacsim::physics::manager::unsubscribePhysicsOnStepEvents(iterator->second);
    }
    catch (...)
    {
        // Backend threw during unsubscription; remove the tracked entry so this
        // uid is not retried, but report failure to the caller.
        m_callbacks.erase(iterator);
        return false;
    }
    m_callbacks.erase(iterator);
    return true;
}

void PhysicsManager::deregisterAllCallbacks() noexcept
{
    std::vector<size_t> uids;
    uids.reserve(m_callbacks.size());
    for (const auto& [uid, _] : m_callbacks)
    {
        uids.push_back(uid);
    }
    for (size_t uid : uids)
    {
        this->deregisterCallback(uid);
    }
    m_callbacks.clear();
}

std::vector<std::pair<std::string, bool>> PhysicsManager::getRegisteredPhysicsEngines() const
{
    std::vector<std::pair<std::string, bool>> engines;

    const size_t numSimulations = registration::getNumberOfSimulations();
    if (!numSimulations)
    {
        return engines;
    }

    std::vector<registration::SimulationId> simulationIds(numSimulations);
    const size_t copiedCount = registration::getSimulationIds(simulationIds.data(), simulationIds.size());
    simulationIds.resize(copiedCount);
    if (simulationIds.empty())
    {
        return engines;
    }

    engines.reserve(simulationIds.size());
    for (const registration::SimulationId simulationId : simulationIds)
    {
        std::string simulationName = registration::getSimulationName(simulationId);
        if (simulationName.empty())
        {
            continue;
        }
        const bool isActive = registration::isSimulationActive(simulationId);
        engines.emplace_back(_toLower(simulationName), isActive);
    }
    return engines;
}

bool PhysicsManager::switchPhysicsEngine(const std::string& engine)
{

    const size_t numSimulations = registration::getNumberOfSimulations();
    if (!numSimulations)
    {
        return false;
    }

    std::vector<registration::SimulationId> simulationIds(numSimulations);
    const size_t copiedCount = registration::getSimulationIds(simulationIds.data(), simulationIds.size());
    simulationIds.resize(copiedCount);
    if (simulationIds.empty())
    {
        return false;
    }

    const std::string targetEngine = _toLower(engine);
    registration::SimulationId targetSimulationId = registration::g_kInvalidSimulationId;
    for (const registration::SimulationId simulationId : simulationIds)
    {
        std::string simulationName = registration::getSimulationName(simulationId);
        if (simulationName.empty())
        {
            continue;
        }
        if (_toLower(simulationName) == targetEngine)
        {
            targetSimulationId = simulationId;
            break;
        }
    }

    if (targetSimulationId == registration::g_kInvalidSimulationId)
    {
        return false;
    }

    // deactivate all other engines for mutual exclusivity
    for (const registration::SimulationId simulationId : simulationIds)
    {
        if (simulationId == targetSimulationId)
        {
            continue;
        }
        if (registration::isSimulationActive(simulationId))
        {
            registration::deactivateSimulation(simulationId);
        }
    }

    // activate the target engine
    const bool isActive = registration::isSimulationActive(targetSimulationId);
    if (!isActive)
    {
        registration::activateSimulation(targetSimulationId);
    }
    return true;
}

std::shared_ptr<isaacsim::physics::tensors::IEntityView> PhysicsManager::createEntity(
    const std::string& engine,
    const std::string& entity,
    const std::variant<std::string, std::vector<std::string>>& paths) const
{
    const std::vector<std::string> pathList = std::holds_alternative<std::string>(paths) ?
                                                  std::vector<std::string>{ std::get<std::string>(paths) } :
                                                  std::get<std::vector<std::string>>(paths);
    return isaacsim::physics::tensors::TensorRegistry::getInstance().createEntity(engine, entity, pathList);
}

} // namespace manager
} // namespace physics
} // namespace isaacsim
