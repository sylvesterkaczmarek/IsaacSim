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

#include "isaacsim/physics/manager/Export.h"
#include "isaacsim/physics/manager/PhysicsEvent.hpp"

#include <isaacsim/physics/registration/Physics.hpp>
#include <isaacsim/physics/registration/tensors/IEntityView.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace isaacsim
{
namespace physics
{
namespace manager
{

/**
 * @brief Variant value type used to deliver simulation event payload entries.
 */
using SimulationEventPayload = std::unordered_map<std::string, std::variant<float, size_t>>;

/**
 * @brief Callback signature receiving a dictionary of named event payload values.
 */
using SimulationEventFunction = std::function<void(SimulationEventPayload)>;

/**
 * @class PhysicsManager
 * @brief Singleton that manages physics simulations.
 */
class ISAACSIM_PHYSICS_MANAGER_API PhysicsManager
{
public:
    PhysicsManager(const PhysicsManager&) = delete;
    PhysicsManager& operator=(const PhysicsManager&) = delete;

    /**
     * @brief Get the singleton instance.
     *
     * @return Singleton instance.
     */
    static PhysicsManager& getInstance();

    /**
     * @brief Get whether the physics manager is initialized.
     *
     * @return True if the physics manager is initialized, false otherwise.
     */
    bool isInitialized() const;

    /**
     * @brief Configure the simulation.
     *
     * @param[in] dt Physics time step in seconds (default 1/60 s).
     */
    void setup(float dt = 1.0f / 60.0f);

    /**
     * @brief Attach a USD stage to all active physics simulations.
     * @details
     * Broadcasts the stage to every active backend. If any backend fails, previously
     * succeeded backends are rolled back before returning.
     *
     * @param[in] ovstageInstancePtr Pointer to the caller-owned ovstage instance, or nullptr.
     * @param[in] usdStageId USD stage-cache identifier of the stage to attach.
     * @return True if every active backend attached the stage successfully, false otherwise.
     */
    bool initialize(void* ovstageInstancePtr, int64_t usdStageId);

    /**
     * @brief Close all attached stages and mark the manager as uninitialized.
     *
     * Delegates to the underlying engine close call and propagates its result. If the engine close call returns `false`
     * one or more engines failed to detach and the manager remains in the initialized state so the caller
     * can retry. The caller must keep any stage pointer alive until this method returns `true`.
     *
     * @return `true` if every engine closed successfully, `false` if a engine reported a close failure.
     */
    bool invalidate();

    /**
     * @brief Steps the physics simulation one or more times.
     * @details
     * Each iteration calls simulate() and fetchResults() on the underlying physics engine,
     * then invokes the optional callback. If the callback returns false the loop exits early.
     *
     * @param[in] steps Number of physics steps to perform (default 1).
     * @param[in] callback Callback called after each step as callback(step, steps).
     * Return false to stop stepping early.
     *
     * @return Number of steps actually simulated.
     * @throws std::runtime_error if the manager has not been initialized via initialize().
     */
    int step(int steps = 1, std::function<bool(int, int)> callback = nullptr);

    /**
     * @brief Get the accumulated simulation time.
     *
     * @return Simulation time in seconds since the last initialize() call.
     */
    float getSimulatedTime() const;

    /**
     * @brief Get the total number of physics steps simulated since the last initialize() call.
     *
     * @return Number of physics steps simulated.
     */
    int getSimulatedPhysicsSteps() const;

    // --------------------------------------------------------------------------------
    // Callback (subscription to simulation events) management
    // --------------------------------------------------------------------------------

    /**
     * @brief Register a callback to be triggered when a specific simulation event occurs.
     * @details
     * The callback is invoked with a dictionary populated with the event payload.
     *
     * For ::PhysicsEvent::ePhysicsPreStep and ::PhysicsEvent::ePhysicsPostStep events:
     *   - "elapsed_time" (float): elapsed time since the previous physics step.
     *   - "scene_path" (size_t): USD scene path encoded as size_t.
     *   - "simulation_id" (size_t): identifier of the simulation that produced the event.
     *
     * @param[in] callback Callback invoked with the event payload dictionary.
     * @param[in] event Event to subscribe to.
     * @param[in] order Subscription order. Lower values run earlier; callbacks within the
     * same order are triggered in registration order.
     *
     * @return Unique callback identifier on success, or 0 if the event is unsupported,
     * the physics simulation is unavailable, or the subscription failed.
     */
    size_t registerCallback(SimulationEventFunction callback, PhysicsEvent event, int order = 0);

    /**
     * @brief Deregister a callback.
     * @param[in] uid Callback identifier.
     * @return True if the callback was deregistered, false if the identifier is invalid.
     */
    bool deregisterCallback(size_t uid) noexcept;

    /**
     * @brief Deregister every callback previously registered via registerCallback().
     * @details
     * Unsubscribes each tracked physics subscription and clears the internal callback map.
     * Identifiers issued in the future continue to be strictly increasing; previously
     * issued identifiers are not reused.
     */
    void deregisterAllCallbacks() noexcept;

    // --------------------------------------------------------------------------------
    // Physics engine management
    // --------------------------------------------------------------------------------

    /**
     * @brief Get all registered physics engines and their active state.
     *
     * @return Vector of (engine, isActive).
     */
    std::vector<std::pair<std::string, bool>> getRegisteredPhysicsEngines() const;

    /**
     * @brief Switch to a specific physics engine.
     * @details
     * The requested engine name is matched case-insensitively. All other active engines are
     * deactivated before activating the target engine.
     *
     * @param[in] engine Target engine name.
     * @return True if the target engine exists and switch logic ran; false otherwise.
     */
    bool switchPhysicsEngine(const std::string& engine);

    // --------------------------------------------------------------------------------
    // Physics tensor management
    // --------------------------------------------------------------------------------

    /**
     * @brief Create a physics tensor entity.
     *
     * @param[in] engine Target engine name.
     * @param[in] entity Type of the entity to create.
     * @param[in] paths Prim path(s). Can include regular expression for matching multiple prims.
     * @return Shared pointer to the entity view.
     */
    std::shared_ptr<isaacsim::physics::tensors::IEntityView> createEntity(
        const std::string& engine,
        const std::string& entity,
        const std::variant<std::string, std::vector<std::string>>& paths) const;

private:
    /** @brief Constructor*/
    PhysicsManager() = default;

    /** @brief Whether the physics manager is initialized. */
    bool m_initialized = false;

    /** @brief Physics time step in seconds. */
    float m_dt = 1.0f / 60.0f;

    /** @brief Accumulated simulation time in seconds. */
    float m_simulatedTime = 0.0f;

    /** @brief Number of physics steps simulated since the last initialize() call. */
    int m_simulatedPhysicsSteps = 0;

    /** @brief Callback registration map. */
    std::unordered_map<size_t, isaacsim::physics::registration::SubscriptionId> m_callbacks;

    /** @brief Next unique identifier to assign to a registered callback. */
    size_t m_callbackUid = 0;
};

} // namespace manager
} // namespace physics
} // namespace isaacsim
