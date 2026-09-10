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

#include "OvPhysxSimulationViewImpl.hpp"

#include "../OvPhysxAdapter.hpp"
#include "OvPhysxEntityViews.hpp"
#include "OvPhysxResultHelpers.hpp"

#include <isaacsim/physics/registration/Physics.hpp>
#include <isaacsim/physics/registration/tensors/TensorRegistry.hpp>
#include <isaacsim/physics/registration/tensors/TensorSpec.hpp>
#include <ovphysx/ovphysx.h>
#include <ovphysx/ovphysx_types.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaacsim
{
namespace physics_engines
{
namespace ovphysx
{

// Process-global map: simulationId -> OvPhysxAdapter.
// Stores the adapter so the SimulationView factory can read the ovphysx C handle
// and the caller-configured device ordinal.
static std::mutex g_adapterMapMutex;
static std::unordered_map<int64_t, std::shared_ptr<OvPhysxAdapter>> g_adapterMap;


void storeAdapter(int64_t simulationId, std::shared_ptr<OvPhysxAdapter> adapter)
{
    std::lock_guard<std::mutex> lock(g_adapterMapMutex);
    g_adapterMap[simulationId] = std::move(adapter);
}

void removeAdapterHandle(int64_t simulationId)
{
    std::lock_guard<std::mutex> lock(g_adapterMapMutex);
    g_adapterMap.erase(simulationId);
}

static std::shared_ptr<OvPhysxAdapter> lookupAdapter(int64_t simulationId)
{
    std::lock_guard<std::mutex> lock(g_adapterMapMutex);
    auto adapterIterator = g_adapterMap.find(simulationId);
    return (adapterIterator != g_adapterMap.end()) ? adapterIterator->second : nullptr;
}

// Entity types this engine offers through TensorRegistry::createEntity.
static const char* s_kEntityTypes[] = {
    "articulation",        "rigid-body", "rigid-contact", "volume-deformable-body", "surface-deformable-body",
    "deformable-material", "sdf-shape",
};

// Capacities used when the caller cannot state one. createEntity carries no construction arguments, so a
// contact or SDF view built through it starts here and is resized by the first read that asks for a
// different extent.
static constexpr int s_kDefaultContactDataCount = 1000;
static constexpr int s_kDefaultQueryPointCount = 1;

// Entity factories receive paths only, so the handle comes from the simulation registered under the name
// the factory was registered with. Naming each simulation distinctly is what lets several coexist: the
// lookup is exact, so nothing is inferred. Returns OVPHYSX_INVALID_HANDLE when no simulation of that name
// is active, which yields an unsupported view rather than a broken one.
static ovphysx_handle_t activeSimulationHandle(const std::string& simulationName)
{
    namespace registration = isaacsim::physics::registration;
    const registration::SimulationId activeSimulationId = registration::getActiveSimulationId(simulationName);
    if (activeSimulationId == registration::g_kInvalidSimulationId)
    {
        return OVPHYSX_INVALID_HANDLE;
    }
    const std::shared_ptr<OvPhysxAdapter> adapter = lookupAdapter(static_cast<int64_t>(activeSimulationId.id));
    return adapter ? adapter->getHandle() : OVPHYSX_INVALID_HANDLE;
}

static std::shared_ptr<isaacsim::physics::tensors::EntityView> makeOvPhysxEntityViewForHandle(
    const std::string& entityType, ovphysx_handle_t handle, const std::vector<std::string>& paths);

std::shared_ptr<isaacsim::physics::tensors::EntityView> makeOvPhysxEntityView(const std::string& entityType,
                                                                              ovphysx_handle_t handle,
                                                                              const std::vector<std::string>& paths)
{
    std::shared_ptr<isaacsim::physics::tensors::EntityView> view =
        makeOvPhysxEntityViewForHandle(entityType, handle, paths);
    // A view whose patterns matched nothing binds no operations at all, and a caller cannot tell that from a
    // registration someone forgot to write. Say "no support" instead, as the contact and SDF views already do
    // when their own construction cannot proceed.
    if (view && view->listImpls(isaacsim::physics::tensors::ImplKind::eGet).empty() &&
        view->listImpls(isaacsim::physics::tensors::ImplKind::eSet).empty())
    {
        return makeUnsupportedView(paths.empty() ? std::string() : paths.front(), entityType);
    }
    return view;
}

static std::shared_ptr<isaacsim::physics::tensors::EntityView> makeOvPhysxEntityViewForHandle(
    const std::string& entityType, ovphysx_handle_t handle, const std::vector<std::string>& paths)
{
    if (handle == OVPHYSX_INVALID_HANDLE)
    {
        return makeUnsupportedView(paths.empty() ? std::string() : paths.front(), entityType);
    }
    if (entityType == "articulation")
    {
        return std::make_shared<OvPhysxArticulationEntityView>(handle, paths);
    }
    if (entityType == "rigid-body")
    {
        return std::make_shared<OvPhysxRigidBodyEntityView>(handle, paths);
    }
    if (entityType == "volume-deformable-body")
    {
        return std::make_shared<OvPhysxVolumeDeformableBodyEntityView>(handle, paths);
    }
    if (entityType == "surface-deformable-body")
    {
        return std::make_shared<OvPhysxSurfaceDeformableBodyEntityView>(handle, paths);
    }
    if (entityType == "deformable-material")
    {
        return std::make_shared<OvPhysxDeformableMaterialEntityView>(handle, paths);
    }
    // Contact and SDF views take a single pattern; there is no path-list form to fall back on, so a
    // multi-path request is rejected rather than silently narrowed to the first entry.
    if (paths.size() > 1)
    {
        throw std::invalid_argument("createEntity: entity '" + entityType + "' accepts a single path (received " +
                                    std::to_string(paths.size()) + ")");
    }
    const std::string pattern = paths.empty() ? std::string() : paths.front();
    if (entityType == "rigid-contact")
    {
        return std::make_shared<OvPhysxRigidContactEntityView>(
            handle, pattern, std::vector<std::string>{}, s_kDefaultContactDataCount);
    }
    if (entityType == "sdf-shape")
    {
        return std::make_shared<OvPhysxSdfShapeEntityView>(handle, pattern, s_kDefaultQueryPointCount);
    }
    throw std::out_of_range("createEntity: unknown ovphysx entity type '" + entityType + "'");
}

// ----------------------------------------------------------------------------
// OvPhysxSimulationViewImpl
// ----------------------------------------------------------------------------

OvPhysxSimulationViewImpl::OvPhysxSimulationViewImpl(ovphysx_handle_t handle,
                                                     const std::string& frontendName,
                                                     int64_t stageId)
    : SimulationView("ovphysx", frontendName, stageId), m_handle(handle)
{
}

isaacsim::physics::tensors::ObjectType OvPhysxSimulationViewImpl::getObjectType(const std::string& primPath) const
{
    if (m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return isaacsim::physics::tensors::ObjectType::eInvalid;
    }

    ovphysx_object_type_t objectType = OVPHYSX_OBJECT_TYPE_INVALID;
    if (ovphysx_get_object_type(m_handle, ovphysx_cstr(primPath.c_str()), &objectType).status != OVPHYSX_API_SUCCESS)
    {
        return isaacsim::physics::tensors::ObjectType::eInvalid;
    }

    return static_cast<isaacsim::physics::tensors::ObjectType>(objectType);
}

void OvPhysxSimulationViewImpl::updateArticulationsKinematic()
{
    if (m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return; // no live scene -- nothing to update (base-class no-op semantics)
    }
    checkOvphysxResult(ovphysx_update_articulations_kinematic(m_handle), "ovphysx_update_articulations_kinematic");
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createArticulationView(
    const std::string& pattern)
{
    if (m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return makeUnsupportedView(pattern, "articulation");
    }
    return std::make_shared<OvPhysxArticulationEntityView>(m_handle, pattern);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createArticulationView(
    const std::vector<std::string>& patterns)
{
    return makeOvPhysxEntityView("articulation", m_handle, patterns);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createRigidBodyView(
    const std::string& pattern)
{
    if (m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return makeUnsupportedView(pattern, "rigid-body");
    }
    return std::make_shared<OvPhysxRigidBodyEntityView>(m_handle, pattern);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createRigidBodyView(
    const std::vector<std::string>& patterns)
{
    return makeOvPhysxEntityView("rigid-body", m_handle, patterns);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createVolumeDeformableBodyView(
    const std::string& pattern)
{
    if (m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return makeUnsupportedView(pattern, "volume-deformable-body");
    }
    return std::make_shared<OvPhysxVolumeDeformableBodyEntityView>(m_handle, pattern);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createVolumeDeformableBodyView(
    const std::vector<std::string>& patterns)
{
    return makeOvPhysxEntityView("volume-deformable-body", m_handle, patterns);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createSurfaceDeformableBodyView(
    const std::string& pattern)
{
    if (m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return makeUnsupportedView(pattern, "surface-deformable-body");
    }
    return std::make_shared<OvPhysxSurfaceDeformableBodyEntityView>(m_handle, pattern);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createSurfaceDeformableBodyView(
    const std::vector<std::string>& patterns)
{
    return makeOvPhysxEntityView("surface-deformable-body", m_handle, patterns);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createDeformableMaterialView(
    const std::string& pattern)
{
    if (m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return makeUnsupportedView(pattern, "deformable-material");
    }
    return std::make_shared<OvPhysxDeformableMaterialEntityView>(m_handle, pattern);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createDeformableMaterialView(
    const std::vector<std::string>& patterns)
{
    return makeOvPhysxEntityView("deformable-material", m_handle, patterns);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createRigidContactView(
    const std::string& pattern, const std::vector<std::string>& filterPatterns, int maximumContactDataCount)
{
    if (m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return makeUnsupportedView(pattern, "rigid-contact");
    }
    return std::make_shared<OvPhysxRigidContactEntityView>(m_handle, pattern, filterPatterns, maximumContactDataCount);
}

std::shared_ptr<isaacsim::physics::tensors::EntityView> OvPhysxSimulationViewImpl::createSdfShapeView(
    const std::string& pattern, int numberOfPoints)
{
    if (m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return makeUnsupportedView(pattern, "sdf-shape");
    }
    return std::make_shared<OvPhysxSdfShapeEntityView>(m_handle, pattern, numberOfPoints);
}

// ----------------------------------------------------------------------------
// TensorRegistry factory registration
// ----------------------------------------------------------------------------

// Registered under "ovphysx" (no longer under "_ovphysx_cpp") since entity
// Views expose all supported methods directly from C++. The Python tensors
// layer only applies _attach_warp_dispatch to wrap returns as wp.array.
void registerOvPhysxSimulationViewFactory(const std::string& simulationName)
{
    using namespace isaacsim::physics::tensors;
    // The engine is a kind of simulator and is declared once; the factories below are keyed by simulation
    // name, of which there may be several.
    TensorRegistry::getInstance().registerEngine("ovphysx");
    TensorRegistry::getInstance().registerSimulationView(
        simulationName,
        [](const std::string& frontendName, int64_t stageId) -> std::shared_ptr<SimulationView>
        {
            auto adapter = lookupAdapter(stageId);
            ovphysx_handle_t handle = adapter ? adapter->getHandle() : OVPHYSX_INVALID_HANDLE;
            return std::make_shared<OvPhysxSimulationViewImpl>(handle, frontendName, stageId);
        });

    for (const char* entityType : s_kEntityTypes)
    {
        TensorRegistry::getInstance().registerEntity(
            simulationName, entityType,
            [entityType, simulationName](const std::vector<std::string>& paths) -> std::shared_ptr<EntityView>
            { return makeOvPhysxEntityView(entityType, activeSimulationHandle(simulationName), paths); });
    }
}

void unregisterOvPhysxSimulationViewFactory(const std::string& simulationName)
{
    using namespace isaacsim::physics::tensors;
    for (const char* entityType : s_kEntityTypes)
    {
        TensorRegistry::getInstance().unregisterEntity(simulationName, entityType);
    }
    TensorRegistry::getInstance().unregisterSimulationView(simulationName);
}

} // namespace ovphysx
} // namespace physics_engines
} // namespace isaacsim
