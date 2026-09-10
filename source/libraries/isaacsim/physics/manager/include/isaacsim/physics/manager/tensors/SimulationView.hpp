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

#include <isaacsim/physics/manager/Export.h>
#include <isaacsim/physics/manager/tensors/EntityView.hpp>
#include <isaacsim/physics/registration/Types.hpp>
#include <isaacsim/physics/registration/tensors/ISimulationView.hpp>
#include <isaacsim/physics/registration/tensors/PhysicsEnums.hpp>

#include <memory>
#include <string>
#include <vector>

namespace isaacsim
{
namespace physics
{
namespace tensors
{

/**
 * @brief Three-component floating-point vector used for gravity.
 */
using Float3 = isaacsim::physics::registration::Float3;

/**
 * @class SimulationView
 * @brief Provides scene-level access to an engine's tensor simulation data.
 *
 * Engine implementations override scene-control operations and entity-view factories. Factory methods return shared
 * ownership so entity views can outlive the @ref SimulationView that created them.
 */
class ISAACSIM_PHYSICS_MANAGER_API SimulationView : public ISimulationView
{
public:
    /**
     * @brief Constructs an unconfigured simulation view.
     */
    SimulationView() = default;

    /**
     * @brief Constructs a simulation view for an engine, tensor frontend, and USD stage.
     *
     * @param[in] engine Physics-engine name.
     * @param[in] frontendName Tensor-frontend name.
     * @param[in] stageId USD stage-cache identifier.
     */
    explicit SimulationView(std::string engine, std::string frontendName, int64_t stageId);

    /**
     * @brief Destroys the simulation view.
     */
    ~SimulationView() override = default;

    /**
     * @brief Simulation views cannot be copied.
     */
    SimulationView(const SimulationView&) = delete;

    /**
     * @brief Simulation views cannot be copy-assigned.
     */
    SimulationView& operator=(const SimulationView&) = delete;

    /**
     * @brief Gets the physics-engine name.
     *
     * @return A reference to the engine name. The reference remains valid until the view is destroyed.
     */
    const std::string& getEngine() const noexcept
    {
        return m_engine;
    }

    /**
     * @brief Gets the tensor-frontend name.
     *
     * @return A reference to the frontend name. The reference remains valid until the view is destroyed.
     */
    const std::string& getFrontendName() const noexcept
    {
        return m_frontendName;
    }

    /**
     * @brief Gets the USD stage-cache identifier.
     *
     * @return The stage-cache identifier, or `-1` for an unconfigured view.
     */
    int64_t getStageId() const noexcept
    {
        return m_stageId;
    }

    /**
     * @brief Gets the ordinal of the device on which simulation tensors reside.
     *
     * @return The simulation-device ordinal, or `-1` if no device is configured.
     */
    virtual int getDeviceOrdinal() const
    {
        return m_deviceOrdinal;
    }

    /**
     * @brief Gets the ordinal of the device on which parameter tensors reside.
     *
     * @return The parameter-device ordinal, or `-1` if no device is configured.
     */
    virtual int getParameterDeviceOrdinal() const
    {
        return m_parameterDeviceOrdinal;
    }

    /**
     * @brief Checks whether the simulation view can be used.
     *
     * @return `true` if the view is valid; otherwise, `false`.
     */
    virtual bool isValid() const
    {
        return m_valid;
    }

    /**
     * @brief Sets the simulation-device ordinal.
     *
     * @param[in] ordinal Device ordinal, or `-1` to indicate that no device is configured.
     */
    void setDeviceOrdinal(int ordinal) noexcept
    {
        m_deviceOrdinal = ordinal;
    }

    /**
     * @brief Sets the parameter-device ordinal.
     *
     * @param[in] ordinal Device ordinal, or `-1` to indicate that no device is configured.
     */
    void setParameterDeviceOrdinal(int ordinal) noexcept
    {
        m_parameterDeviceOrdinal = ordinal;
    }

    /**
     * @brief Sets whether the simulation view can be used.
     *
     * @param[in] valid `true` to mark the view as valid; `false` to mark it as invalid.
     */
    void setValid(bool valid) noexcept
    {
        m_valid = valid;
    }

    /**
     * @brief Marks this simulation view as invalid.
     *
     * Engine overrides may also invalidate entity views created by this object.
     */
    virtual void invalidate()
    {
        m_valid = false;
    }

    /**
     * @brief Sets the gravity vector.
     *
     * The base implementation stores the value locally.
     *
     * @param[in] gravity Gravity acceleration in world coordinates.
     */
    virtual void setGravity(Float3 gravity)
    {
        m_gravity = gravity;
    }

    /**
     * @brief Gets the gravity vector.
     *
     * @return Gravity acceleration in world coordinates.
     */
    virtual Float3 getGravity() const
    {
        return m_gravity;
    }

    /**
     * @brief Clears externally applied forces.
     *
     * The base implementation has no effect.
     */
    virtual void clearForces()
    {
    }

    /**
     * @brief Advances the simulation by one time step.
     *
     * The base implementation has no effect.
     *
     * @param[in] timeStep Time-step duration, in seconds.
     */
    virtual void step([[maybe_unused]] float timeStep)
    {
    }

    /**
     * @brief Writes current articulation poses to kinematic targets.
     *
     * The base implementation has no effect.
     */
    virtual void updateArticulationsKinematic()
    {
    }

    /**
     * @brief Initializes kinematic bodies from their current poses.
     *
     * The base implementation has no effect.
     */
    virtual void initializeKinematicBodies()
    {
    }

    /**
     * @brief Gets the physics object type for a prim path.
     *
     * The base implementation reports an invalid object.
     *
     * @param[in] primPath Absolute path of the prim to classify.
     * @return The object classification, or @c ObjectType::eInvalid when the engine cannot classify the path.
     */
    virtual ObjectType getObjectType([[maybe_unused]] const std::string& primPath) const
    {
        return ObjectType::eInvalid;
    }

    /**
     * @brief Creates an articulation view from one prim-path pattern.
     *
     * @param[in] pattern Glob pattern or explicit prim path selecting articulations.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createArticulationView([[maybe_unused]] const std::string& pattern)
    {
        return nullptr;
    }

    /**
     * @brief Creates an articulation view from multiple prim-path patterns.
     *
     * Matches are concatenated in pattern-list order.
     *
     * @param[in] patterns Glob patterns or explicit prim paths selecting articulations.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createArticulationView([[maybe_unused]] const std::vector<std::string>& patterns)
    {
        return nullptr;
    }

    /**
     * @brief Creates a rigid-body view from one prim-path pattern.
     *
     * @param[in] pattern Glob pattern or explicit prim path selecting rigid bodies.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createRigidBodyView([[maybe_unused]] const std::string& pattern)
    {
        return nullptr;
    }

    /**
     * @brief Creates a rigid-body view from multiple prim-path patterns.
     *
     * Matches are concatenated in pattern-list order.
     *
     * @param[in] patterns Glob patterns or explicit prim paths selecting rigid bodies.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createRigidBodyView([[maybe_unused]] const std::vector<std::string>& patterns)
    {
        return nullptr;
    }

    /**
     * @brief Creates a volume-deformable-body view from one prim-path pattern.
     *
     * @param[in] pattern Glob pattern or explicit prim path selecting volume-deformable bodies.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createVolumeDeformableBodyView([[maybe_unused]] const std::string& pattern)
    {
        return nullptr;
    }

    /**
     * @brief Creates a volume-deformable-body view from multiple prim-path patterns.
     *
     * Matches are concatenated in pattern-list order.
     *
     * @param[in] patterns Glob patterns or explicit prim paths selecting volume-deformable bodies.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createVolumeDeformableBodyView(
        [[maybe_unused]] const std::vector<std::string>& patterns)
    {
        return nullptr;
    }

    /**
     * @brief Creates a surface-deformable-body view from one prim-path pattern.
     *
     * @param[in] pattern Glob pattern or explicit prim path selecting surface-deformable bodies.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createSurfaceDeformableBodyView([[maybe_unused]] const std::string& pattern)
    {
        return nullptr;
    }

    /**
     * @brief Creates a surface-deformable-body view from multiple prim-path patterns.
     *
     * Matches are concatenated in pattern-list order.
     *
     * @param[in] patterns Glob patterns or explicit prim paths selecting surface-deformable bodies.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createSurfaceDeformableBodyView(
        [[maybe_unused]] const std::vector<std::string>& patterns)
    {
        return nullptr;
    }

    /**
     * @brief Creates a deformable-material view from one prim-path pattern.
     *
     * @param[in] pattern Glob pattern or explicit prim path selecting deformable materials.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createDeformableMaterialView([[maybe_unused]] const std::string& pattern)
    {
        return nullptr;
    }

    /**
     * @brief Creates a deformable-material view from multiple prim-path patterns.
     *
     * Matches are concatenated in pattern-list order.
     *
     * @param[in] patterns Glob patterns or explicit prim paths selecting deformable materials.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createDeformableMaterialView([[maybe_unused]] const std::vector<std::string>& patterns)
    {
        return nullptr;
    }

    /**
     * @brief Creates a rigid-contact view.
     *
     * @param[in] pattern Glob pattern or explicit prim path selecting rigid bodies whose contacts are reported.
     * @param[in] filterPatterns Glob patterns or explicit prim paths selecting contact counterparts.
     * @param[in] maximumContactDataCount Maximum number of detailed contact records retained by the view. Must be
     *                                nonnegative.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createRigidContactView([[maybe_unused]] const std::string& pattern,
                                                               [[maybe_unused]] const std::vector<std::string>& filterPatterns,
                                                               [[maybe_unused]] int maximumContactDataCount)
    {
        return nullptr;
    }

    /**
     * @brief Creates a signed-distance-field shape view.
     *
     * @param[in] pattern Glob pattern or explicit prim path selecting shapes.
     * @param[in] numberOfPoints Maximum number of query points allocated for each selected shape. Must be nonnegative.
     * @return Shared ownership of the entity view, or `nullptr` if the engine does not provide the factory.
     */
    virtual std::shared_ptr<EntityView> createSdfShapeView([[maybe_unused]] const std::string& pattern,
                                                           [[maybe_unused]] int numberOfPoints)
    {
        return nullptr;
    }

private:
    std::string m_engine;
    std::string m_frontendName;
    int64_t m_stageId{ -1 };
    int m_deviceOrdinal{ -1 };
    int m_parameterDeviceOrdinal{ -1 };
    bool m_valid{ true };
    Float3 m_gravity{ 0.0f, 0.0f, -9.81f };
};

} // namespace tensors
} // namespace physics
} // namespace isaacsim
