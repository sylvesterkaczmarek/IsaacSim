// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// clang-format off
#include <pch/UsdPCH.hpp>
// clang-format on

#include "ContactSensorImpl.hpp"

#include "SensorImplUtils.hpp"

#include <carb/events/EventsUtils.h>
#include <carb/logging/Log.h>

#include <isaacsim/core/experimental/prims/IPrimDataReader.hpp>
#include <isaacsim/core/experimental/prims/IPrimDataReaderManager.hpp>
#include <isaacsim/core/experimental/prims/SdfPathToken.hpp>
#include <isaacsim/core/includes/Buffer.hpp>
#include <isaacsim/core/includes/PhysicsEngine.hpp>
#include <isaacsim/core/includes/UsdUtilities.hpp>
#include <isaacsim/core/simulation_manager/ISimulationManager.hpp>
#include <isaacsim/robot/schema/sensor_tokens.hpp>
#include <omni/fabric/FabricUSD.h>
#include <omni/physics/simulation/IPhysicsSimulation.h>
#include <omni/physics/simulation/IPhysicsStageUpdate.h>
#include <omni/physics/tensors/IRigidContactView.h>
#include <omni/physics/tensors/ISimulationView.h>
#include <omni/physics/tensors/TensorApi.h>
#include <omni/usd/UsdContext.h>
#include <pxr/usd/usdPhysics/rigidBodyAPI.h>

#if defined(_WIN32)
#    include <usdrt/scenegraph/usd/usd/stage.h>
#else
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-variable"
#    pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#    include <usdrt/scenegraph/usd/usd/stage.h>

#    include <strings.h>
#    pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace isaacsim
{
namespace sensors
{
namespace experimental
{
namespace physics
{
namespace
{

using core::experimental::prims::IPrimDataReader;
using core::experimental::prims::IPrimDataReaderManager;
using core::experimental::prims::IXformDataView;
using core::simulation_manager::ISimulationManager;
using omni::physics::tensors::IRigidContactView;
using omni::physics::tensors::ISimulationView;
using omni::physics::tensors::TensorApi;
using omni::physics::tensors::TensorDataType;
using omni::physics::tensors::TensorDesc;

static constexpr uint32_t kDefaultMaxContactDataCount = 256;

static std::string findParentRigidBody(pxr::UsdStageRefPtr stage, const pxr::SdfPath& sensorPath)
{
    pxr::UsdPrim prim = stage->GetPrimAtPath(sensorPath);
    if (!prim.IsValid())
    {
        return {};
    }

    prim = prim.GetParent();
    while (prim.IsValid() && prim.GetPath() != pxr::SdfPath::AbsoluteRootPath())
    {
        bool enabled = false;
        pxr::UsdAttribute attr = prim.GetAttribute(pxr::TfToken("physics:rigidBodyEnabled"));
        bool hasRigidBodyAPI = prim.HasAPI<pxr::UsdPhysicsRigidBodyAPI>();
        bool attrValid = attr.IsValid();
        if (attrValid)
        {
            attr.Get(&enabled);
        }

        if (enabled)
        {
            return prim.GetPath().GetString();
        }

        if (hasRigidBodyAPI && !attrValid)
        {
            return prim.GetPath().GetString();
        }

        prim = prim.GetParent();
    }
    return {};
}

using isaacsim::core::experimental::prims::sdfPathToToken;
using isaacsim::core::includes::GenericBufferBase;

bool isNewtonEngine(const char* engineName)
{
    if (!engineName || engineName[0] == '\0')
    {
        return false;
    }
#if defined(_WIN32)
    return _stricmp(engineName, "newton") == 0;
#else
    return strcasecmp(engineName, "newton") == 0;
#endif
}

void setTensorDesc1D(TensorDesc& desc, void* data, int size, TensorDataType dtype, int device)
{
    desc = {};
    desc.device = device;
    desc.dtype = dtype;
    desc.numDims = 1;
    desc.dims[0] = size;
    desc.data = data;
    desc.ownData = false;
}

void fillWorldPosOriFromMatrix(const pxr::GfMatrix4d& worldMat, float* outPos3, float* outOriWxyz)
{
    const pxr::GfVec3d translation = worldMat.ExtractTranslation();
    outPos3[0] = static_cast<float>(translation[0]);
    outPos3[1] = static_cast<float>(translation[1]);
    outPos3[2] = static_cast<float>(translation[2]);

    const pxr::GfQuatd quat = worldMat.ExtractRotationQuat();
    outOriWxyz[0] = static_cast<float>(quat.GetReal());
    const pxr::GfVec3d imag = quat.GetImaginary();
    outOriWxyz[1] = static_cast<float>(imag[0]);
    outOriWxyz[2] = static_cast<float>(imag[1]);
    outOriWxyz[3] = static_cast<float>(imag[2]);
}

// Rotate vector v by quaternion q (wxyz). Pass the conjugate to apply the inverse rotation.
void quatRotateVec(const float* q, const float* v, float* out)
{
    const float qw = q[0];
    const float qx = q[1];
    const float qy = q[2];
    const float qz = q[3];
    const float tx = 2.0f * (qy * v[2] - qz * v[1]);
    const float ty = 2.0f * (qz * v[0] - qx * v[2]);
    const float tz = 2.0f * (qx * v[1] - qy * v[0]);
    out[0] = v[0] + qw * tx + (qy * tz - qz * ty);
    out[1] = v[1] + qw * ty + (qz * tx - qx * tz);
    out[2] = v[2] + qw * tz + (qx * ty - qy * tx);
}

struct ContactTensorScratch
{
    int deviceOrdinal = -1;
    uint32_t maxContactDataCount = 0;
    std::unique_ptr<GenericBufferBase<float>> forces;
    std::unique_ptr<GenericBufferBase<float>> points;
    std::unique_ptr<GenericBufferBase<float>> normals;
    std::unique_ptr<GenericBufferBase<float>> separations;
    std::unique_ptr<GenericBufferBase<int32_t>> counts;
    std::unique_ptr<GenericBufferBase<int32_t>> startIndices;
    std::unique_ptr<GenericBufferBase<int64_t>> otherActorIds;
    std::unique_ptr<GenericBufferBase<float>> netForces;

    void reset()
    {
        deviceOrdinal = -1;
        maxContactDataCount = 0;
        forces.reset();
        points.reset();
        normals.reset();
        separations.reset();
        counts.reset();
        startIndices.reset();
        otherActorIds.reset();
        netForces.reset();
    }

    void ensure(uint32_t maxContacts, int device)
    {
        if (deviceOrdinal == device && maxContactDataCount == maxContacts && forces)
        {
            forces->resize(maxContacts);
            points->resize(maxContacts * 3);
            normals->resize(maxContacts * 3);
            separations->resize(maxContacts);
            counts->resize(1);
            startIndices->resize(1);
            otherActorIds->resize(maxContacts);
            netForces->resize(3);
            return;
        }

        deviceOrdinal = device;
        maxContactDataCount = maxContacts;
        forces = std::make_unique<GenericBufferBase<float>>(maxContacts, device);
        points = std::make_unique<GenericBufferBase<float>>(maxContacts * 3, device);
        normals = std::make_unique<GenericBufferBase<float>>(maxContacts * 3, device);
        separations = std::make_unique<GenericBufferBase<float>>(maxContacts, device);
        counts = std::make_unique<GenericBufferBase<int32_t>>(1, device);
        startIndices = std::make_unique<GenericBufferBase<int32_t>>(1, device);
        otherActorIds = std::make_unique<GenericBufferBase<int64_t>>(maxContacts, device);
        netForces = std::make_unique<GenericBufferBase<float>>(3, device);
    }
};

} // namespace

struct ContactSensorImpl::SensorData
{
    std::string sensorPrimPath;
    std::string parentRigidBodyPath;
    std::string viewId;
    IXformDataView* xformView = nullptr;
    IRigidContactView* contactView = nullptr;
    uint64_t parentToken = 0;

    float radius = -1.0f;
    float minThreshold = 0.0f;
    float maxThreshold = 100000.0f;
    bool enabled = true;
    bool previousEnabled = true;
    pxr::GfVec3f sensorLocalOffset{ 0.0f, 0.0f, 0.0f };
    bool hasSensorLocalOffset = false;
    bool configLockedForRun = false;

    ContactSensorReading latestReading;
    std::vector<ContactRawData> latestRawContacts;

    uint32_t maxContactDataCount = kDefaultMaxContactDataCount;
    ContactTensorScratch tensorScratch;
    std::vector<float> hostForces;
    std::vector<float> hostPoints;
    std::vector<float> hostNormals;
    std::vector<uint64_t> hostOtherActorIds;
    std::array<float, 3> hostNetForces{};
    int32_t hostContactCount = 0;
    int32_t hostStartIndex = 0;
    double lastProcessedSimTime = -1.0;
    double lastRawProcessedSimTime = -1.0;

    void refreshConfig(pxr::UsdStageRefPtr stage)
    {
        using namespace isaacsim::robot::schema::sensors;

        pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(sensorPrimPath));
        if (!prim.IsValid())
        {
            return;
        }

        pxr::UsdAttribute enabledAttr = prim.GetAttribute(kEnabledAttr);
        if (enabledAttr.IsValid())
        {
            bool val = true;
            enabledAttr.Get(&val);
            enabled = val;
        }
        else
        {
            enabled = true;
        }

        pxr::GfVec2f thresholdAttr(0.0f, 100000.0f);
        isaacsim::core::includes::safeGetAttribute(prim.GetAttribute(kThresholdAttr), thresholdAttr);
        const float* t = thresholdAttr.GetArray();
        minThreshold = t[0];
        maxThreshold = t[1];

        float r = -1.0f;
        isaacsim::core::includes::safeGetAttribute(prim.GetAttribute(kRadiusAttr), r);
        radius = r;
    }

    // Compute the constant sensor origin in the parent body's rigid frame, in meters, from the
    // supplied parent world pose (from the runtime reader, so its rotation matches the frame the
    // solver writes back) and the sensor prim's authored world position. Must be called on the first
    // simulated step, before the body has moved from its rest pose, so the reader pose still matches
    // the sensor's authored world. Differencing the world positions bakes in any scale on the prim
    // hierarchy; the result is expressed in the parent's rotation frame so it composes with the
    // simulated body pose at runtime. Computed lazily rather than at bind time because the sensor's
    // authored local transform may not be set when the sensor is first created.
    void computeSensorLocalOffset(pxr::UsdStageRefPtr stage)
    {
        if (hasSensorLocalOffset)
        {
            return;
        }
        pxr::UsdPrim prim = stage->GetPrimAtPath(pxr::SdfPath(sensorPrimPath));
        pxr::UsdPrim parentPrim = stage->GetPrimAtPath(pxr::SdfPath(parentRigidBodyPath));
        if (!prim.IsValid() || !parentPrim.IsValid())
        {
            return;
        }
        pxr::UsdGeomXformCache xformCache;
        xformCache.SetTime(pxr::UsdTimeCode::Default());
        bool resetsXformStack = false;
        // Sensor origin expressed in the parent body's local (authored) frame, rotation-correct
        // regardless of how the body is oriented in the world.
        const pxr::GfVec3d local =
            xformCache.ComputeRelativeTransform(prim, parentPrim, &resetsXformStack).ExtractTranslation();
        // The runtime reader reports the body pose with unit scale, but the authored hierarchy may
        // scale the body's local frame (e.g. a centimeter asset composed at meter units). Scale the
        // local offset to metric world units per axis so it composes with the reader's body rotation.
        const pxr::GfMatrix4d parentWorld = xformCache.GetLocalToWorldTransform(parentPrim);
        const double sx = parentWorld.TransformDir(pxr::GfVec3d(1.0, 0.0, 0.0)).GetLength();
        const double sy = parentWorld.TransformDir(pxr::GfVec3d(0.0, 1.0, 0.0)).GetLength();
        const double sz = parentWorld.TransformDir(pxr::GfVec3d(0.0, 0.0, 1.0)).GetLength();
        sensorLocalOffset = pxr::GfVec3f(
            static_cast<float>(local[0] * sx), static_cast<float>(local[1] * sy), static_cast<float>(local[2] * sz));
        hasSensorLocalOffset = true;
    }

    void ensureTensorBuffers(uint32_t maxContacts, int device)
    {
        maxContactDataCount = maxContacts;
        tensorScratch.ensure(maxContacts, device);
        hostForces.resize(maxContacts);
        hostPoints.resize(maxContacts * 3);
        hostNormals.resize(maxContacts * 3);
        hostOtherActorIds.resize(maxContacts);
    }

    void syncContactCountsToHost(int device)
    {
        if (device >= 0)
        {
            tensorScratch.counts->copyTo(&hostContactCount, 1);
            tensorScratch.startIndices->copyTo(&hostStartIndex, 1);
        }
        else
        {
            hostContactCount = tensorScratch.counts->data()[0];
            hostStartIndex = tensorScratch.startIndices->data()[0];
        }
    }

    void syncContactDataToHost(int device, size_t packedContactCount)
    {
        packedContactCount = std::min(packedContactCount, static_cast<size_t>(maxContactDataCount));
        if (packedContactCount == 0)
        {
            return;
        }

        if (device >= 0)
        {
            tensorScratch.forces->copyTo(hostForces.data(), packedContactCount);
            tensorScratch.points->copyTo(hostPoints.data(), packedContactCount * 3);
            tensorScratch.normals->copyTo(hostNormals.data(), packedContactCount * 3);
            tensorScratch.otherActorIds->copyTo(reinterpret_cast<int64_t*>(hostOtherActorIds.data()), packedContactCount);
        }
        else
        {
            std::memcpy(hostForces.data(), tensorScratch.forces->data(), packedContactCount * sizeof(float));
            std::memcpy(hostPoints.data(), tensorScratch.points->data(), packedContactCount * 3 * sizeof(float));
            std::memcpy(hostNormals.data(), tensorScratch.normals->data(), packedContactCount * 3 * sizeof(float));
            std::memcpy(
                hostOtherActorIds.data(), tensorScratch.otherActorIds->data(), packedContactCount * sizeof(int64_t));
        }
    }
};

struct ContactSensorImpl::ImplData
{
    long stageId = 0;
    float lastDt = 0.0f;
    int stepCount = 0;
    uint64_t readerGeneration = 0;
    std::string engineType = "physx";

    ISimulationManager* simManager = nullptr;
    IPrimDataReaderManager* readerManager = nullptr;
    IPrimDataReader* reader = nullptr;
    TensorApi* tensorApi = nullptr;
    ISimulationView* simView = nullptr;
    int simDeviceOrdinal = -1;
    omni::physics::IPhysicsSimulation* physicsSimulation = nullptr;
    omni::physics::SubscriptionId physicsStepSub = omni::physics::kInvalidSubscriptionId;
    carb::events::ISubscriptionPtr physicsEventSub;

    pxr::UsdStageRefPtr usdStage;
    usdrt::UsdStageRefPtr usdrtStage;
    std::unordered_map<std::string, SensorData> sensors;

    // Caches encoded SdfPath tokens for the "other actor" of Newton contacts.
    // Retaining the SdfPath objects keeps their interned tokens valid for the
    // lifetime of the sensor, so the encoded identifiers do not dangle.
    std::unordered_map<std::string, pxr::SdfPath> otherActorPathCache;
};

ContactSensorImpl::ContactSensorImpl() : m_impl(std::make_unique<ImplData>())
{
    m_impl->simManager = carb::getCachedInterface<ISimulationManager>();
    m_impl->readerManager = carb::getCachedInterface<IPrimDataReaderManager>();
    m_impl->reader = m_impl->readerManager ? m_impl->readerManager->getReader() : nullptr;
    m_impl->tensorApi = carb::getCachedInterface<TensorApi>();
    _subscribeToPhysicsStepEvents();
    _subscribeToPhysicsEvents();
}

ContactSensorImpl::~ContactSensorImpl()
{
    shutdown();
}

void ContactSensorImpl::shutdown()
{
    _unsubscribeFromPhysicsStepEvents();
    m_impl->physicsEventSub.reset();
    _clearSensors();
    m_impl->readerManager = nullptr;
    m_impl->reader = nullptr;
    m_impl->simManager = nullptr;
    m_impl->tensorApi = nullptr;
    m_impl->physicsSimulation = nullptr;
    m_impl->usdStage = nullptr;
    m_impl->usdrtStage = nullptr;
    m_impl->stageId = 0;
    m_impl->stepCount = 0;
    m_impl->lastDt = 0.0f;
    m_impl->readerGeneration = 0;
}

void ContactSensorImpl::_initializeFromContext()
{
    auto* usdContext = omni::usd::UsdContext::getContext();
    if (!usdContext)
    {
        return;
    }

    pxr::UsdStageRefPtr stage = usdContext->getStage();
    if (!stage)
    {
        return;
    }

    pxr::UsdStageCache& cache = pxr::UsdUtilsStageCache::Get();
    const long stageId = cache.GetId(stage).ToLongInt();
    if (stageId == 0)
    {
        return;
    }

    // Prefer the live active engine name so that Newton is detected correctly
    // even when the default_engine setting still reads "physx".
    const char* activeEngine = isaacsim::core::includes::getActivePhysicsEngineName();
    if (activeEngine && activeEngine[0] != '\0')
    {
        m_impl->engineType = activeEngine;
    }
    else
    {
        const std::string engineFromSettings = utils::getEngineTypeFromSettings();
        if (!engineFromSettings.empty())
        {
            m_impl->engineType = engineFromSettings;
        }
    }

    _initializeStage(stageId);
    _discoverSensorsFromStage();
}

void ContactSensorImpl::_initializeStage(long stageId)
{
    if (m_impl->stageId == stageId && m_impl->usdStage && m_impl->readerManager && m_impl->reader)
    {
        return;
    }

    if (m_impl->stageId != 0 && m_impl->stageId != stageId)
    {
        _clearSensors();
    }

    m_impl->stageId = stageId;
    m_impl->stepCount = 0;
    m_impl->lastDt = 0.0f;

    m_impl->simManager = carb::getCachedInterface<ISimulationManager>();
    m_impl->readerManager = carb::getCachedInterface<IPrimDataReaderManager>();
    m_impl->tensorApi = carb::getCachedInterface<TensorApi>();
    if (m_impl->readerManager)
    {
        m_impl->readerManager->ensureInitialized(stageId, -1);
        m_impl->reader = m_impl->readerManager->getReader();
    }
    else
    {
        m_impl->reader = nullptr;
    }

    pxr::UsdStageCache& cache = pxr::UsdUtilsStageCache::Get();
    m_impl->usdStage = cache.Find(pxr::UsdStageCache::Id::FromLongInt(stageId));

    if (m_impl->usdStage)
    {
        omni::fabric::UsdStageId fabricStageId = { static_cast<uint64_t>(stageId) };
        omni::fabric::IStageReaderWriter* iStageReaderWriter =
            carb::getCachedInterface<omni::fabric::IStageReaderWriter>();
        if (iStageReaderWriter)
        {
            omni::fabric::StageReaderWriterId stageInProgress = iStageReaderWriter->get(fabricStageId);
            m_impl->usdrtStage = usdrt::UsdStage::Attach(fabricStageId, stageInProgress);
        }
    }
}

bool ContactSensorImpl::_ensureSimulationView()
{
    if (m_impl->simView && !m_impl->simView->getValid())
    {
        for (auto& [id, sensor] : m_impl->sensors)
        {
            (void)id;
            _releaseContactView(sensor);
        }
        m_impl->simView->release(true);
        m_impl->simView = nullptr;
        m_impl->simDeviceOrdinal = -1;
    }

    if (m_impl->simView)
    {
        return true;
    }

    for (auto& [id, sensor] : m_impl->sensors)
    {
        (void)id;
        _releaseContactView(sensor);
    }

    if (!m_impl->tensorApi)
    {
        m_impl->tensorApi = carb::getCachedInterface<TensorApi>();
    }
    if (!m_impl->tensorApi || m_impl->stageId == 0)
    {
        return false;
    }

    m_impl->simView =
        m_impl->tensorApi->createSimulationView(m_impl->stageId, isaacsim::core::includes::getActivePhysicsEngineName());
    if (m_impl->simView)
    {
        m_impl->simDeviceOrdinal = m_impl->simView->getDeviceOrdinal();
    }
    return m_impl->simView != nullptr;
}

bool ContactSensorImpl::_ensureContactView(SensorData& sensor)
{
    if (sensor.contactView)
    {
        if (m_impl->simView && m_impl->simView->getValid() && sensor.contactView->check())
        {
            return true;
        }
        else
        {
            _releaseContactView(sensor);
        }
    }

    // Tensor backends publish the stable simulation model after the first
    // completed physics callback. Delay initial binding instead of creating and
    // unconditionally replacing an otherwise valid view during warm-up.
    if (m_impl->stepCount < 2)
    {
        return false;
    }

    if (!_ensureSimulationView())
    {
        return false;
    }

    sensor.contactView = m_impl->simView->createRigidContactView(
        sensor.parentRigidBodyPath, std::vector<std::string>{}, kDefaultMaxContactDataCount);
    if (!sensor.contactView)
    {
        CARB_LOG_WARN(
            "ContactSensorImpl: failed to create rigid contact view for '%s'", sensor.parentRigidBodyPath.c_str());
        return false;
    }

    sensor.ensureTensorBuffers(sensor.contactView->getMaxContactDataCount(), m_impl->simDeviceOrdinal);
    return true;
}

void ContactSensorImpl::_releaseContactView(SensorData& sensor)
{
    if (sensor.contactView)
    {
        sensor.contactView->release();
        sensor.contactView = nullptr;
    }
    sensor.tensorScratch.reset();
}

bool ContactSensorImpl::createSensor(const char* primPath)
{
    if (!m_impl->usdStage)
    {
        _initializeFromContext();
    }
    if (!m_impl->usdStage)
    {
        return false;
    }

    const char* activeEngine = isaacsim::core::includes::getActivePhysicsEngineName();
    if (activeEngine && activeEngine[0] != '\0')
    {
        m_impl->engineType = activeEngine;
    }

    std::string key(primPath);
    pxr::SdfPath sdfPath(primPath);
    pxr::UsdPrim prim = m_impl->usdStage->GetPrimAtPath(sdfPath);

    auto existing = m_impl->sensors.find(key);
    if (existing != m_impl->sensors.end())
    {
        if (prim.IsValid() && prim.GetTypeName() == "IsaacContactSensor")
        {
            std::string currentParent = findParentRigidBody(m_impl->usdStage, sdfPath);
            if (currentParent == existing->second.parentRigidBodyPath)
            {
                existing->second.refreshConfig(m_impl->usdStage);
                return true;
            }
        }
        if (m_impl->reader && !existing->second.viewId.empty())
        {
            m_impl->reader->removeView(existing->second.viewId.c_str());
        }
        _releaseContactView(existing->second);
        m_impl->sensors.erase(existing);
    }

    if (!prim.IsValid())
    {
        return false;
    }

    if (prim.GetTypeName() != "IsaacContactSensor")
    {
        return false;
    }

    std::string parentPath = findParentRigidBody(m_impl->usdStage, sdfPath);
    if (parentPath.empty())
    {
        return false;
    }

    if (!isNewtonEngine(activeEngine) && !isNewtonEngine(m_impl->engineType.c_str()))
    {
        if (m_impl->reader)
        {
            if (!m_impl->reader->enableContactReporting(parentPath.c_str()))
            {
                CARB_LOG_WARN("ContactSensorImpl: failed to enable contact reporting for '%s'", parentPath.c_str());
            }
        }
    }

    SensorData& sensor = m_impl->sensors[key];
    sensor.sensorPrimPath = primPath;
    sensor.parentRigidBodyPath = parentPath;
    sensor.parentToken = sdfPathToToken(pxr::SdfPath(parentPath));
    sensor.viewId = "contact_xform_" + key;

    if (m_impl->reader)
    {
        // Bind the view to the parent rigid body: its world pose is the one solvers write back
        // (including Newton, which does not update child sensor prims). The constant sensor offset
        // in the body frame is applied separately when filtering by radius.
        const char* parentPathPtr = parentPath.c_str();
        sensor.xformView =
            m_impl->reader->createXformView(sensor.viewId.c_str(), &parentPathPtr, 1, m_impl->engineType.c_str());
        m_impl->readerGeneration = m_impl->reader->getGeneration();
    }

    sensor.refreshConfig(m_impl->usdStage);
    sensor.configLockedForRun = true;

    return true;
}

void ContactSensorImpl::removeSensor(const char* primPath)
{
    auto it = m_impl->sensors.find(std::string(primPath));
    if (it == m_impl->sensors.end())
    {
        return;
    }
    if (m_impl->reader && !it->second.viewId.empty())
    {
        m_impl->reader->removeView(it->second.viewId.c_str());
    }
    _releaseContactView(it->second);
    m_impl->sensors.erase(it);
}

void ContactSensorImpl::_refreshSensorReadingIfNeeded(const std::string& primPath, bool requireRawContacts)
{
    auto it = m_impl->sensors.find(primPath);
    if (it == m_impl->sensors.end())
    {
        return;
    }

    if (!m_impl->usdStage)
    {
        _initializeFromContext();
    }

    if (m_impl->readerManager && m_impl->stageId != 0)
    {
        if (m_impl->readerManager->ensureInitialized(m_impl->stageId, -1))
        {
            m_impl->reader = m_impl->readerManager->getReader();
        }
    }

    if (m_impl->reader && m_impl->reader->getGeneration() != m_impl->readerGeneration)
    {
        _recreateSensorViews();
    }

    if (!m_impl->simManager)
    {
        return;
    }
    const double simTime = m_impl->simManager->getSimulationTime();
    if (m_impl->usdStage)
    {
        const float dt = m_impl->lastDt > 0.0f ? m_impl->lastDt : (1.0f / 60.0f);
        _processSensorIfNeeded(*m_impl, primPath, dt, simTime, requireRawContacts);
    }
}

ContactSensorReading ContactSensorImpl::getSensorReading(const char* primPath)
{
    const std::string key(primPath);
    auto it = m_impl->sensors.find(key);
    if (it == m_impl->sensors.end())
    {
        return ContactSensorReading();
    }

    if (m_impl->usdStage)
    {
        pxr::UsdPrim prim = m_impl->usdStage->GetPrimAtPath(pxr::SdfPath(primPath));
        if (!prim.IsValid())
        {
            if (m_impl->reader && !it->second.viewId.empty())
            {
                m_impl->reader->removeView(it->second.viewId.c_str());
            }
            _releaseContactView(it->second);
            m_impl->sensors.erase(it);
            return ContactSensorReading();
        }
    }

    _refreshSensorReadingIfNeeded(key, false);
    it = m_impl->sensors.find(key);
    if (it == m_impl->sensors.end())
    {
        return ContactSensorReading();
    }

    return it->second.latestReading;
}

void ContactSensorImpl::getRawContacts(const char* primPath, const ContactRawData** outData, int32_t* outCount)
{
    if (!outData || !outCount)
    {
        return;
    }

    *outData = nullptr;
    *outCount = 0;

    auto it = m_impl->sensors.find(std::string(primPath));
    if (it == m_impl->sensors.end())
    {
        return;
    }

    if (m_impl->usdStage)
    {
        pxr::UsdPrim prim = m_impl->usdStage->GetPrimAtPath(pxr::SdfPath(primPath));
        if (!prim.IsValid())
        {
            if (m_impl->reader && !it->second.viewId.empty())
            {
                m_impl->reader->removeView(it->second.viewId.c_str());
            }
            _releaseContactView(it->second);
            m_impl->sensors.erase(it);
            return;
        }
    }

    _refreshSensorReadingIfNeeded(std::string(primPath), true);
    it = m_impl->sensors.find(std::string(primPath));
    if (it == m_impl->sensors.end())
    {
        return;
    }

    const auto& contacts = it->second.latestRawContacts;
    if (!contacts.empty())
    {
        *outData = contacts.data();
        *outCount = static_cast<int32_t>(contacts.size());
    }
}

void ContactSensorImpl::_discoverSensorsFromStage()
{
    if (!m_impl->usdStage)
    {
        return;
    }

    for (auto prim : m_impl->usdStage->Traverse())
    {
        if (prim.GetTypeName() == "IsaacContactSensor")
        {
            (void)createSensor(prim.GetPath().GetString().c_str());
        }
    }
}

void ContactSensorImpl::_clearSensors()
{
    for (auto& [id, sensor] : m_impl->sensors)
    {
        (void)id;
        if (m_impl->reader && !sensor.viewId.empty())
        {
            m_impl->reader->removeView(sensor.viewId.c_str());
        }
        _releaseContactView(sensor);
    }
    m_impl->sensors.clear();

    if (m_impl->simView)
    {
        m_impl->simView->release(true);
        m_impl->simView = nullptr;
        m_impl->simDeviceOrdinal = -1;
    }
}

void ContactSensorImpl::_recreateSensorViews()
{
    if (!m_impl->reader)
    {
        return;
    }

    for (auto& [id, sensor] : m_impl->sensors)
    {
        (void)id;
        sensor.xformView = nullptr;
        _releaseContactView(sensor);
        if (sensor.viewId.empty() || sensor.sensorPrimPath.empty() || sensor.parentRigidBodyPath.empty())
        {
            continue;
        }

        const char* parentPathPtr = sensor.parentRigidBodyPath.c_str();
        sensor.xformView =
            m_impl->reader->createXformView(sensor.viewId.c_str(), &parentPathPtr, 1, m_impl->engineType.c_str());
    }
    m_impl->readerGeneration = m_impl->reader->getGeneration();

    if (m_impl->simView)
    {
        m_impl->simView->release(true);
        m_impl->simView = nullptr;
        m_impl->simDeviceOrdinal = -1;
    }
}

void ContactSensorImpl::_subscribeToPhysicsEvents()
{
    if (m_impl->physicsEventSub)
    {
        return;
    }

    auto* physicsStageUpdate = carb::getCachedInterface<omni::physics::IPhysicsStageUpdate>();
    if (!physicsStageUpdate)
    {
        return;
    }

    m_impl->physicsEventSub = carb::events::createSubscriptionToPop(
        physicsStageUpdate->getSimulationEventStream().get(),
        [this](carb::events::IEvent* e)
        {
            if (e->type == omni::physics::SimulationEvent::eStopped)
            {
                _clearSensors();
                m_impl->usdStage = nullptr;
                m_impl->usdrtStage = nullptr;
                m_impl->stageId = 0;
                m_impl->stepCount = 0;
                m_impl->lastDt = 0.0f;
            }
            else if (e->type == omni::physics::SimulationEvent::eResumed)
            {
                _initializeFromContext();
            }
        },
        0, "IsaacSim.Sensors.Experimental.Physics.ContactSensor.SimulationEvent");
}

void ContactSensorImpl::_subscribeToPhysicsStepEvents()
{
    if (m_impl->physicsStepSub != omni::physics::kInvalidSubscriptionId)
    {
        return;
    }

    m_impl->physicsSimulation = carb::getCachedInterface<omni::physics::IPhysicsSimulation>();
    if (!m_impl->physicsSimulation)
    {
        return;
    }

    m_impl->physicsStepSub = m_impl->physicsSimulation->subscribePhysicsOnStepEvents(
        false, 1,
        [this](float elapsedTime, const omni::physics::PhysicsStepContext& /*context*/) { _stepSensors(elapsedTime); });
}

void ContactSensorImpl::_unsubscribeFromPhysicsStepEvents()
{
    if (m_impl->physicsSimulation && m_impl->physicsStepSub != omni::physics::kInvalidSubscriptionId)
    {
        m_impl->physicsSimulation->unsubscribePhysicsOnStepEvents(m_impl->physicsStepSub);
        m_impl->physicsStepSub = omni::physics::kInvalidSubscriptionId;
    }
}

void ContactSensorImpl::_stepSensors(float dt)
{
    m_impl->lastDt = dt;
    m_impl->stepCount++;

    if (!m_impl->simManager)
    {
        return;
    }

    if (!m_impl->usdStage)
    {
        _initializeFromContext();
    }

    if (!m_impl->usdStage)
    {
        return;
    }

    if (m_impl->sensors.empty())
    {
        return;
    }

    if (m_impl->readerManager && m_impl->stageId != 0)
    {
        if (!m_impl->readerManager->ensureInitialized(m_impl->stageId, -1))
        {
            return;
        }
        m_impl->reader = m_impl->readerManager->getReader();
    }

    if (m_impl->reader && m_impl->reader->getGeneration() != m_impl->readerGeneration)
    {
        _recreateSensorViews();
    }

    const double simTime = m_impl->simManager->getSimulationTime();
    for (auto& [id, sensor] : m_impl->sensors)
    {
        (void)sensor;
        _processSensorIfNeeded(*m_impl, id, dt, simTime, false);
    }
}

void ContactSensorImpl::_processSensorIfNeeded(
    ImplData& impl, const std::string& primPath, float dt, double simTime, bool requireRawContacts)
{
    auto it = impl.sensors.find(primPath);
    if (it == impl.sensors.end())
    {
        return;
    }

    const bool updateSummary = it->second.lastProcessedSimTime != simTime;
    const bool updateRawContacts = requireRawContacts && it->second.lastRawProcessedSimTime != simTime;
    if (!updateSummary && !updateRawContacts)
    {
        return;
    }

    _processSensor(impl, primPath, dt, simTime, updateSummary, updateRawContacts);
    it = impl.sensors.find(primPath);
    if (it != impl.sensors.end())
    {
        if (updateSummary)
        {
            it->second.lastProcessedSimTime = simTime;
        }
        if (updateRawContacts || (updateSummary && it->second.radius > 0.0f))
        {
            it->second.lastRawProcessedSimTime = simTime;
        }
    }
}

void ContactSensorImpl::_processSensor(
    ImplData& impl, const std::string& primPath, float dt, double simTime, bool updateSummary, bool updateRawContacts)
{
    auto it = impl.sensors.find(primPath);
    if (it == impl.sensors.end())
    {
        return;
    }
    SensorData& sensor = it->second;

    // USD configuration is static while the simulation is running. Reading it
    // on every physics step performs several USD lookups per sensor, so sample
    // it once at the start of each run. Sensors are recreated on stop/play,
    // and createSensor() refreshes this cache for explicit re-creation.
    if (!sensor.configLockedForRun)
    {
        sensor.refreshConfig(impl.usdStage);
        sensor.configLockedForRun = true;
    }

    if (sensor.previousEnabled != sensor.enabled)
    {
        if (!sensor.enabled)
        {
            sensor.latestReading = ContactSensorReading();
            sensor.latestRawContacts.clear();
        }
        sensor.previousEnabled = sensor.enabled;
    }

    if (!sensor.enabled)
    {
        return;
    }

    ContactSensorReading reading;
    reading.time = static_cast<float>(simTime);

    const bool needsRawData = updateRawContacts || (updateSummary && sensor.radius > 0.0f);
    if (needsRawData)
    {
        sensor.latestRawContacts.clear();
    }

    if (!_ensureContactView(sensor))
    {
        if (updateSummary)
        {
            sensor.latestReading = reading;
        }
        return;
    }

    const uint32_t maxContacts = sensor.maxContactDataCount;
    const int device = impl.simDeviceOrdinal;
    auto& scratch = sensor.tensorScratch;
    const float contactDt = dt > 0.0f ? dt : (impl.lastDt > 0.0f ? impl.lastDt : 1.0f / 60.0f);

    double rawForceMagnitude = 0.0;
    if (needsRawData)
    {
        TensorDesc forceDesc;
        TensorDesc pointDesc;
        TensorDesc normalDesc;
        TensorDesc separationDesc;
        TensorDesc countDesc;
        TensorDesc startIndexDesc;
        TensorDesc otherActorDesc;

        setTensorDesc1D(
            forceDesc, scratch.forces->data(), static_cast<int>(maxContacts), TensorDataType::eFloat32, device);
        setTensorDesc1D(
            pointDesc, scratch.points->data(), static_cast<int>(maxContacts * 3), TensorDataType::eFloat32, device);
        setTensorDesc1D(
            normalDesc, scratch.normals->data(), static_cast<int>(maxContacts * 3), TensorDataType::eFloat32, device);
        setTensorDesc1D(separationDesc, scratch.separations->data(), static_cast<int>(maxContacts),
                        TensorDataType::eFloat32, device);
        setTensorDesc1D(countDesc, scratch.counts->data(), 1, TensorDataType::eInt32, device);
        setTensorDesc1D(startIndexDesc, scratch.startIndices->data(), 1, TensorDataType::eInt32, device);
        setTensorDesc1D(otherActorDesc, scratch.otherActorIds->data(), static_cast<int>(maxContacts),
                        TensorDataType::eInt64, device);

        const bool rawQuerySucceeded =
            sensor.contactView->getRawContactData(&forceDesc, &pointDesc, &normalDesc, &separationDesc, &countDesc,
                                                  &startIndexDesc, &otherActorDesc, contactDt);
        if (!rawQuerySucceeded)
        {
            CARB_LOG_WARN_ONCE(
                "Contact sensor raw tensor query failed. The current raw reading is unavailable and the contact "
                "view will be recreated on the next sample.");
            _releaseContactView(sensor);
            if (updateSummary)
            {
                sensor.latestReading = reading;
            }
            return;
        }

        sensor.syncContactCountsToHost(device);
        const int32_t contactCount = std::max(sensor.hostContactCount, 0);
        const int32_t startIndex = std::max(sensor.hostStartIndex, 0);
        const size_t packedContactCount = std::min(
            static_cast<size_t>(maxContacts), static_cast<size_t>(startIndex) + static_cast<size_t>(contactCount));
        sensor.syncContactDataToHost(device, packedContactCount);

        // Newton reports the "other actor" as a body index, whereas PhysX reports an
        // encoded SdfPath token. Resolve the Newton indices to paths so the sensor's
        // raw contact data exposes a backend-consistent, decodable identifier.
        const bool normalizeOtherActorIds = isNewtonEngine(impl.engineType.c_str());
        std::vector<std::string> otherActorPaths;
        if (normalizeOtherActorIds && contactCount > 0)
        {
            sensor.contactView->getOtherActorPathsFromIds(&otherActorDesc, otherActorPaths);
        }

        float parentPos[3] = {};
        float parentOri[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
        bool hasParentTransform = false;
        if (sensor.radius > 0.0f)
        {
            if (sensor.xformView != nullptr)
            {
                hasParentTransform =
                    sensor.xformView->getPrimWorldTransform(sensor.parentRigidBodyPath.c_str(), parentPos, parentOri);
            }
            if (!hasParentTransform && impl.usdStage)
            {
                pxr::UsdPrim parentPrim = impl.usdStage->GetPrimAtPath(pxr::SdfPath(sensor.parentRigidBodyPath));
                if (parentPrim.IsValid())
                {
                    pxr::UsdGeomXformCache xformCache;
                    xformCache.SetTime(pxr::UsdTimeCode::Default());
                    fillWorldPosOriFromMatrix(xformCache.GetLocalToWorldTransform(parentPrim), parentPos, parentOri);
                    hasParentTransform = true;
                }
            }
            if (impl.usdStage)
            {
                sensor.computeSensorLocalOffset(impl.usdStage);
            }

            if (!hasParentTransform || !sensor.hasSensorLocalOffset)
            {
                CARB_LOG_WARN_ONCE(
                    "Contact sensor radius filtering requires a valid parent transform and sensor local "
                    "offset. The current reading is invalid.");
                if (updateSummary)
                {
                    sensor.latestReading = reading;
                }
                return;
            }
        }

        double totalForceX = 0.0;
        double totalForceY = 0.0;
        double totalForceZ = 0.0;
        if (contactCount > 0)
        {
            sensor.latestRawContacts.reserve(static_cast<size_t>(contactCount));
            for (int32_t ci = 0; ci < contactCount; ++ci)
            {
                const int32_t idx = startIndex + ci;
                if (idx < 0 || static_cast<uint32_t>(idx) >= maxContacts)
                {
                    continue;
                }

                const float px = sensor.hostPoints[static_cast<size_t>(idx) * 3 + 0];
                const float py = sensor.hostPoints[static_cast<size_t>(idx) * 3 + 1];
                const float pz = sensor.hostPoints[static_cast<size_t>(idx) * 3 + 2];

                if (sensor.radius > 0.0f)
                {
                    const float offset[3] = { sensor.sensorLocalOffset[0], sensor.sensorLocalOffset[1],
                                              sensor.sensorLocalOffset[2] };
                    float rotated[3] = {};
                    quatRotateVec(parentOri, offset, rotated);
                    const double dx = static_cast<double>(parentPos[0] + rotated[0]) - static_cast<double>(px);
                    const double dy = static_cast<double>(parentPos[1] + rotated[1]) - static_cast<double>(py);
                    const double dz = static_cast<double>(parentPos[2] + rotated[2]) - static_cast<double>(pz);
                    const double distance = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (distance >= static_cast<double>(sensor.radius))
                    {
                        continue;
                    }
                }

                const float forceScalar = sensor.hostForces[static_cast<size_t>(idx)];
                const float nx = sensor.hostNormals[static_cast<size_t>(idx) * 3 + 0];
                const float ny = sensor.hostNormals[static_cast<size_t>(idx) * 3 + 1];
                const float nz = sensor.hostNormals[static_cast<size_t>(idx) * 3 + 2];
                const double forceX = static_cast<double>(forceScalar) * static_cast<double>(nx);
                const double forceY = static_cast<double>(forceScalar) * static_cast<double>(ny);
                const double forceZ = static_cast<double>(forceScalar) * static_cast<double>(nz);

                totalForceX += forceX;
                totalForceY += forceY;
                totalForceZ += forceZ;

                ContactRawData entry;
                entry.body0 = sensor.parentToken;
                entry.body1 = sensor.hostOtherActorIds[static_cast<size_t>(idx)];
                if (normalizeOtherActorIds && static_cast<size_t>(idx) < otherActorPaths.size())
                {
                    const std::string& otherPath = otherActorPaths[static_cast<size_t>(idx)];
                    const std::string absPath =
                        (!otherPath.empty() && otherPath[0] == '/') ? otherPath : ("/" + otherPath);
                    auto cached = impl.otherActorPathCache.find(absPath);
                    if (cached == impl.otherActorPathCache.end())
                    {
                        cached = impl.otherActorPathCache.emplace(absPath, pxr::SdfPath(absPath)).first;
                    }
                    entry.body1 = sdfPathToToken(cached->second);
                }
                else if (normalizeOtherActorIds)
                {
                    entry.body1 = sdfPathToToken(pxr::SdfPath::AbsoluteRootPath());
                }
                entry.positionX = px;
                entry.positionY = py;
                entry.positionZ = pz;
                entry.normalX = nx;
                entry.normalY = ny;
                entry.normalZ = nz;
                entry.impulseX = static_cast<float>(forceX * static_cast<double>(contactDt));
                entry.impulseY = static_cast<float>(forceY * static_cast<double>(contactDt));
                entry.impulseZ = static_cast<float>(forceZ * static_cast<double>(contactDt));
                entry.time = reading.time;
                entry.dt = contactDt;
                sensor.latestRawContacts.push_back(entry);
            }
        }

        rawForceMagnitude = std::sqrt(totalForceX * totalForceX + totalForceY * totalForceY + totalForceZ * totalForceZ);
    }

    if (!updateSummary)
    {
        return;
    }

    double forceMagnitude = 0.0;
    if (sensor.radius <= 0.0f)
    {
        TensorDesc netDesc;
        setTensorDesc1D(netDesc, scratch.netForces->data(), 3, TensorDataType::eFloat32, device);
        const bool netQuerySucceeded = sensor.contactView->getNetContactForces(&netDesc, contactDt);
        if (!netQuerySucceeded)
        {
            sensor.latestReading = reading;
            return;
        }

        if (device >= 0)
        {
            scratch.netForces->copyTo(sensor.hostNetForces.data(), 3);
        }
        else
        {
            std::memcpy(sensor.hostNetForces.data(), scratch.netForces->data(), 3 * sizeof(float));
        }

        forceMagnitude =
            std::sqrt(static_cast<double>(sensor.hostNetForces[0]) * static_cast<double>(sensor.hostNetForces[0]) +
                      static_cast<double>(sensor.hostNetForces[1]) * static_cast<double>(sensor.hostNetForces[1]) +
                      static_cast<double>(sensor.hostNetForces[2]) * static_cast<double>(sensor.hostNetForces[2]));
    }
    else
    {
        forceMagnitude = rawForceMagnitude;
    }

    if (forceMagnitude <= 0.0)
    {
        reading.isValid = true;
        sensor.latestReading = reading;
        return;
    }

    float forceValue = static_cast<float>(forceMagnitude);
    forceValue = std::min(forceValue, sensor.maxThreshold);
    if (forceValue < sensor.minThreshold)
    {
        reading.isValid = true;
        sensor.latestReading = reading;
        return;
    }

    reading.value = forceValue;
    reading.inContact = true;
    reading.isValid = true;
    sensor.latestReading = reading;
}

} // namespace physics
} // namespace experimental
} // namespace sensors
} // namespace isaacsim
