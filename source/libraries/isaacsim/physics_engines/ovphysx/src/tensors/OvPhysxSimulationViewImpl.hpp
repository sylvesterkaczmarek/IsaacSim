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

#include <isaacsim/physics/manager/tensors/SimulationView.hpp>
#include <isaacsim/physics/registration/tensors/TensorRegistry.hpp>
#include <ovphysx/ovphysx_types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace isaacsim
{
namespace physics_engines
{
namespace ovphysx
{

class OvPhysxAdapter;

// C++ subclass of SimulationView for the OvPhysX backend.
// Overrides view factories to return C++ EntityView objects that call the
// ovphysx C API directly -- no Python on the tensor data hot path.
class OvPhysxSimulationViewImpl : public isaacsim::physics::tensors::SimulationView
{
public:
    OvPhysxSimulationViewImpl(ovphysx_handle_t handle, const std::string& frontendName, int64_t stageId);
    ~OvPhysxSimulationViewImpl() override = default;

    void updateArticulationsKinematic() override;

    std::shared_ptr<isaacsim::physics::tensors::EntityView> createArticulationView(const std::string& pattern) override;

    std::shared_ptr<isaacsim::physics::tensors::EntityView> createArticulationView(
        const std::vector<std::string>& patterns) override;

    std::shared_ptr<isaacsim::physics::tensors::EntityView> createRigidBodyView(const std::string& pattern) override;

    std::shared_ptr<isaacsim::physics::tensors::EntityView> createRigidBodyView(
        const std::vector<std::string>& patterns) override;

    std::shared_ptr<isaacsim::physics::tensors::EntityView> createVolumeDeformableBodyView(const std::string& pattern) override;
    std::shared_ptr<isaacsim::physics::tensors::EntityView> createVolumeDeformableBodyView(
        const std::vector<std::string>& patterns) override;

    std::shared_ptr<isaacsim::physics::tensors::EntityView> createSurfaceDeformableBodyView(
        const std::string& pattern) override;
    std::shared_ptr<isaacsim::physics::tensors::EntityView> createSurfaceDeformableBodyView(
        const std::vector<std::string>& patterns) override;

    std::shared_ptr<isaacsim::physics::tensors::EntityView> createDeformableMaterialView(const std::string& pattern) override;
    std::shared_ptr<isaacsim::physics::tensors::EntityView> createDeformableMaterialView(
        const std::vector<std::string>& patterns) override;

    std::shared_ptr<isaacsim::physics::tensors::EntityView> createRigidContactView(
        const std::string& pattern, const std::vector<std::string>& filterPatterns, int maximumContactDataCount) override;

    std::shared_ptr<isaacsim::physics::tensors::EntityView> createSdfShapeView(const std::string& pattern,
                                                                               int numberOfPoints) override;

    isaacsim::physics::tensors::ObjectType getObjectType(const std::string& primPath) const override;

    // Device ordinal is set by the Python caller via set_device_ordinal() (the
    // caller knows the configured tensor device); the base SimulationView stores
    // and returns it. No USD/scene readback needed here.

private:
    ovphysx_handle_t m_handle;
};

// Register an adapter so the SimulationView factory can read the handle
// and device ordinal. Called from the bindings register() function.
void storeAdapter(int64_t simulationId, std::shared_ptr<OvPhysxAdapter> adapter);
void removeAdapterHandle(int64_t simulationId);

// Construct the view for an entity type against an already-resolved handle. Shared by the
// TensorRegistry entity factories and the dual-path invariant test so both describe one implementation.
// An invalid handle yields an unsupported view rather than a broken one.
std::shared_ptr<isaacsim::physics::tensors::EntityView> makeOvPhysxEntityView(const std::string& entityType,
                                                                              ovphysx_handle_t handle,
                                                                              const std::vector<std::string>& paths);

// Register the OvPhysX SimulationView and entity factories with TensorRegistry under a simulation name.
// The name is what callers pass to createSimulationView / createEntity, and what the entity factories
// resolve their handle from, so registering distinct names is how several simulations coexist.
void registerOvPhysxSimulationViewFactory(const std::string& simulationName = "ovphysx");

// Unregister the factories registered under a simulation name.
void unregisterOvPhysxSimulationViewFactory(const std::string& simulationName = "ovphysx");

} // namespace ovphysx
} // namespace physics_engines
} // namespace isaacsim
