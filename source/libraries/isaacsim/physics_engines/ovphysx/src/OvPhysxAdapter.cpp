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

#include "OvPhysxAdapter.hpp"

#include <isaacsim/common/logging/Logging.hpp>
#include <isaacsim/physics/registration/Physics.hpp>
#include <isaacsim/physics/registration/simulator/Benchmark.hpp>
#include <isaacsim/physics/registration/simulator/Interaction.hpp>
#include <isaacsim/physics/registration/simulator/Simulator.hpp>
#include <ovphysx/ovphysx.h>
#include <ovstage/ovstage.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

namespace isaacsim
{
namespace physics_engines
{
namespace ovphysx
{

namespace
{

const isaacsim::common::logging::Logger g_kLogger("isaacsim.physics_engines.ovphysx");

// Process-global lifecycle guard so ovphysx_initialize() / ovphysx_shutdown()
// are called exactly once per process even with multiple OvPhysxAdapter instances.
std::mutex g_lifecycleMutex;
int g_instanceCount{ 0 };
// ovphysx's lifecycle is a single latch (initialize() errors if already set), not a
// refcount. Track whether OUR initialize() set it so we never shut down a lifecycle
// another consumer in this process owns.
bool g_weInitialized{ false };

void initializeGlobalState()
{
    std::lock_guard<std::mutex> lock(g_lifecycleMutex);
    if (g_instanceCount == 0)
    {
        // A non-success status means the latch was already set -- the only failure
        // ovphysx_initialize() documents -- i.e. another consumer owns the lifecycle.
        g_weInitialized = (ovphysx_initialize().status == OVPHYSX_API_SUCCESS);
    }
    ++g_instanceCount;
}

void globalRelease()
{
    std::lock_guard<std::mutex> lock(g_lifecycleMutex);
    --g_instanceCount;
    if (g_instanceCount == 0)
    {
        // Only release the latch if we set it; otherwise the consumer that
        // initialized ovphysx still owns it.
        if (g_weInitialized)
        {
            ovphysx_shutdown();
        }
        g_weInitialized = false;
    }
}

void checkResult(ovphysx_result_t result, const char* context)
{
    if (result.status != OVPHYSX_API_SUCCESS)
    {
        ovphysx_string_t errorString = ovphysx_get_last_error();
        std::string message(errorString.ptr ? errorString.ptr : "", errorString.ptr ? errorString.length : 0);
        throw std::runtime_error(std::string(context) + ": ovphysx error: " + message);
    }
}

// The manager parses each scene once at this ordinal and steps it; it never
// authors control edits into the Stage (state is written via the tensor API),
// so no ovphysx_update_from_ovstage / higher ordinals are needed. ovstage ordinals
// are application-owned (see ovstage_population.h): the caller-owned Stage must be
// populated at this ordinal, or the attach reads an empty scene. Part of the
// dual-input contract documented on PhysicsSimulation::initialize.
constexpr ovstage_ordinal_t g_kAttachOrdinal = 1;

// ovphysx reports the contact-pair event kind as 0 = found, 1 = lost, 2 = persist. Map the three
// known values explicitly rather than casting, and report anything else instead of passing it off
// as a persist -- a renumbering on either side would otherwise mislabel every event silently. The
// warning fires once per process: this runs per contact pair per step.
isaacsim::physics::registration::ContactEventType toContactEventType(int32_t eventType)
{
    using isaacsim::physics::registration::ContactEventType;
    switch (eventType)
    {
    case 0:
        return ContactEventType::eContactFound;
    case 1:
        return ContactEventType::eContactLost;
    case 2:
        return ContactEventType::eContactPersist;
    default:
        break;
    }

    static std::atomic<bool> s_reportedUnknownEventType{ false };
    if (!s_reportedUnknownEventType.exchange(true))
    {
        ISAACSIM_LOG_WARN(g_kLogger,
                          "OvPhysxAdapter: unknown ovphysx contact event type {}; reporting it as persist. "
                          "Further occurrences are not logged.",
                          eventType);
    }
    return ContactEventType::eContactPersist;
}

// SweepHit and RaycastHit share the same set of fields; fill them from one
// templated helper so the two converters can't drift out of sync.
template <class HitType>
HitType toHit(const ovphysx_scene_query_hit_t& sourceHit)
{
    HitType outputHit{};
    outputHit.collision = static_cast<isaacsim::physics::registration::PathToken>(sourceHit.collision);
    outputHit.rigidBody = static_cast<isaacsim::physics::registration::PathToken>(sourceHit.rigid_body);
    outputHit.prototypeIndex = sourceHit.proto_index;
    outputHit.normal = { sourceHit.normal[0], sourceHit.normal[1], sourceHit.normal[2] };
    outputHit.position = { sourceHit.position[0], sourceHit.position[1], sourceHit.position[2] };
    outputHit.distance = sourceHit.distance;
    outputHit.faceIndex = sourceHit.face_index;
    outputHit.material = static_cast<isaacsim::physics::registration::PathToken>(sourceHit.material);
    return outputHit;
}

isaacsim::physics::registration::SweepHit toSweepHit(const ovphysx_scene_query_hit_t& sourceHit)
{
    return toHit<isaacsim::physics::registration::SweepHit>(sourceHit);
}

isaacsim::physics::registration::RaycastHit toRaycastHit(const ovphysx_scene_query_hit_t& sourceHit)
{
    return toHit<isaacsim::physics::registration::RaycastHit>(sourceHit);
}

isaacsim::physics::registration::OverlapHit toOverlapHit(const ovphysx_scene_query_hit_t& sourceHit)
{
    isaacsim::physics::registration::OverlapHit outputHit{};
    outputHit.collision = static_cast<isaacsim::physics::registration::PathToken>(sourceHit.collision);
    outputHit.rigidBody = static_cast<isaacsim::physics::registration::PathToken>(sourceHit.rigid_body);
    outputHit.prototypeIndex = sourceHit.proto_index;
    return outputHit;
}

ovphysx_scene_query_geometry_desc_t makeSphereGeometry(float radius,
                                                       const isaacsim::physics::registration::Float3& position)
{
    ovphysx_scene_query_geometry_desc_t geometry{};
    geometry.type = OVPHYSX_SCENE_QUERY_GEOMETRY_SPHERE;
    geometry.sphere.radius = radius;
    geometry.sphere.position[0] = position.x;
    geometry.sphere.position[1] = position.y;
    geometry.sphere.position[2] = position.z;
    return geometry;
}

ovphysx_scene_query_geometry_desc_t makeBoxGeometry(const isaacsim::physics::registration::Float3& halfExtent,
                                                    const isaacsim::physics::registration::Float3& position,
                                                    const isaacsim::physics::registration::Float4& rotation)
{
    ovphysx_scene_query_geometry_desc_t geometry{};
    geometry.type = OVPHYSX_SCENE_QUERY_GEOMETRY_BOX;
    geometry.box.half_extent[0] = halfExtent.x;
    geometry.box.half_extent[1] = halfExtent.y;
    geometry.box.half_extent[2] = halfExtent.z;
    geometry.box.position[0] = position.x;
    geometry.box.position[1] = position.y;
    geometry.box.position[2] = position.z;
    geometry.box.rotation[0] = rotation.x;
    geometry.box.rotation[1] = rotation.y;
    geometry.box.rotation[2] = rotation.z;
    geometry.box.rotation[3] = rotation.w;
    return geometry;
}

} // namespace

// ----------------------------------------------------------------------------
// Construction / destruction
// ----------------------------------------------------------------------------

std::shared_ptr<OvPhysxAdapter> OvPhysxAdapter::create(const Configuration& configuration)
{
    return std::shared_ptr<OvPhysxAdapter>(new OvPhysxAdapter(configuration));
}

OvPhysxAdapter::GlobalLifecycleGuard::GlobalLifecycleGuard()
{
    initializeGlobalState();
}

OvPhysxAdapter::GlobalLifecycleGuard::~GlobalLifecycleGuard()
{
    globalRelease();
}

OvPhysxAdapter::OvPhysxAdapter(const Configuration& configuration)
{
    ovphysx_register_schema_paths();

    ovphysx_create_args arguments = OVPHYSX_CREATE_ARGS_DEFAULT;
    // Non-static (per-call, so concurrent constructions don't share it) but declared here so it
    // outlives ovphysx_create_instance below -- active_cuda_gpus borrows the pointer, not a copy.
    char gpuOrdinalBuffer[32];
    if (configuration.gpuOrdinal > 0)
    {
        snprintf(gpuOrdinalBuffer, sizeof(gpuOrdinalBuffer), "%d", configuration.gpuOrdinal);
        arguments.active_cuda_gpus = ovphysx_cstr(gpuOrdinalBuffer);
    }

    if (configuration.useGpuPipeline)
    {
        // DirectGPU: suppress readback before instance creation.
        ovphysx_config_entry_t suppressReadbackEntry{};
        suppressReadbackEntry.key_type = OVPHYSX_CONFIG_KEY_TYPE_CARBONITE;
        suppressReadbackEntry.key.carbonite_key = OVPHYSX_LITERAL("/physics/suppressReadback");
        suppressReadbackEntry.value.string_value = OVPHYSX_LITERAL("true");
        ovphysx_set_global_config(suppressReadbackEntry);

        ovphysx_config_entry_t suppressFabricUpdateEntry{};
        suppressFabricUpdateEntry.key_type = OVPHYSX_CONFIG_KEY_TYPE_CARBONITE;
        suppressFabricUpdateEntry.key.carbonite_key = OVPHYSX_LITERAL("/physics/suppressFabricUpdate");
        suppressFabricUpdateEntry.value.string_value = OVPHYSX_LITERAL("true");
        ovphysx_set_global_config(suppressFabricUpdateEntry);
    }

    checkResult(ovphysx_create_instance(&arguments, &m_handle), "OvPhysxAdapter::create");
}

OvPhysxAdapter::~OvPhysxAdapter()
{
    unregisterFromManager();
    // Defensive: detach any borrowed Stage still attached if _doClose wasn't called.
    // The Stage is caller-owned, so it is only detached here, never destroyed.
    _detachOvstage();
    if (m_handle != OVPHYSX_INVALID_HANDLE)
    {
        ovphysx_destroy_instance(m_handle);
        m_handle = OVPHYSX_INVALID_HANDLE;
    }
}

// ----------------------------------------------------------------------------
// Registration
// ----------------------------------------------------------------------------

size_t OvPhysxAdapter::registerWithManager(const std::string& name)
{
    isaacsim::physics::registration::Simulation simulation;
    _buildSimulation(simulation);

    const isaacsim::physics::registration::SimulationId simulationId =
        isaacsim::physics::registration::registerSimulation(simulation, name);
    if (simulationId == isaacsim::physics::registration::g_kInvalidSimulationId)
    {
        throw std::runtime_error("OvPhysxAdapter: registerSimulation returned g_kInvalidSimulationId");
    }

    m_simulationId = simulationId.id;
    return m_simulationId;
}

void OvPhysxAdapter::unregisterFromManager()
{
    if (m_simulationId == isaacsim::physics::registration::g_kInvalidSimulationId.id)
    {
        return;
    }
    isaacsim::physics::registration::unregisterSimulation(isaacsim::physics::registration::SimulationId(m_simulationId));
    m_simulationId = isaacsim::physics::registration::g_kInvalidSimulationId.id;
}

// ----------------------------------------------------------------------------
// Simulation struct construction
// ----------------------------------------------------------------------------

void OvPhysxAdapter::_buildSimulation(isaacsim::physics::registration::Simulation& simulation)
{
    auto& simulationFunctions = simulation.simulationFunctions;

    simulationFunctions.initialize = [this](void* ovstage, const char* usdIdentifier)
    { return _doInitialize(ovstage, usdIdentifier); };
    simulationFunctions.close = [this]() { return _doClose(); };
    simulationFunctions.getAttachedStage = [this]() { return _doGetAttachedStage(); };
    simulationFunctions.hasAttachedStage = [this]() { return _doHasAttachedStage(); };
    simulationFunctions.simulateAsynchronously = [this](float elapsed, float current)
    { _doSimulateAsynchronously(elapsed, current); };
    simulationFunctions.simulate = [this](float elapsed, float current) { _doSimulate(elapsed, current); };
    simulationFunctions.fetchResults = [this]() { _doFetchResults(); };
    simulationFunctions.checkResults = [this]() { return _doCheckResults(); };
    simulationFunctions.flushChanges = [this]() { _doFlushChanges(); };
    simulationFunctions.pauseChangeTracking = [this](bool pause) { _doPauseChangeTracking(pause); };
    simulationFunctions.isChangeTrackingPaused = [this]() { return _doIsChangeTrackingPaused(); };
    simulationFunctions.subscribePhysicsContactReportEvents =
        [this](isaacsim::physics::registration::OnContactReportEventFunction callback)
    { return _doSubscribeContactReport(std::move(callback)); };
    simulationFunctions.unsubscribePhysicsContactReportEvents =
        [this](isaacsim::physics::registration::SubscriptionId subscriptionId)
    { _doUnsubscribeContactReport(subscriptionId); };
    simulationFunctions.getSimulationTimeStepsPerSecond =
        [this](long stageId, isaacsim::physics::registration::PathToken scenePath)
    { return _doGetTimeStepsPerSecond(stageId, scenePath); };
    simulationFunctions.getSimulationTimestamp = [this]() { return _doGetTimestamp(); };
    simulationFunctions.getSimulationStepCount = [this]() { return _doGetStepCount(); };
    simulationFunctions.subscribePhysicsOnStepEvents =
        [this](bool preStep, int order, isaacsim::physics::registration::OnPhysicsStepEventFunction callback)
    { return _doSubscribeStepEvents(preStep, order, std::move(callback)); };
    simulationFunctions.unsubscribePhysicsOnStepEvents =
        [this](isaacsim::physics::registration::SubscriptionId subscriptionId)
    { _doUnsubscribeStepEvents(subscriptionId); };
    simulationFunctions.isCapableOfSimulating = [this](const char** schemaNames, size_t count, bool* isCapable)
    { return _doIsCapableOfSimulating(schemaNames, count, isCapable); };

    _fillSceneQueryFunctions(simulation.sceneQueryFunctions);

    // InteractionFunctions: wire no-op stubs so callers don't crash on a null function pointer.
    simulation.interactionFunctions.handleRaycast = [](const isaacsim::physics::registration::Float3&,
                                                       const isaacsim::physics::registration::Float3&, bool) {};
    simulation.interactionFunctions.getPrimDebugData =
        [](const std::string&) -> isaacsim::physics::registration::DebugDataDictionary { return {}; };

    // BenchmarkFunctions: no-op stubs -- profile stats not yet plumbed to ovphysx.
    simulation.benchmarkFunctions.subscribeProfileStatisticsEvents =
        [](isaacsim::physics::registration::ProfileStatisticsNotificationFunction) -> isaacsim::physics::registration::SubscriptionId
    { return isaacsim::physics::registration::SubscriptionId(0); };
    simulation.benchmarkFunctions.unsubscribeProfileStatisticsEvents =
        [](isaacsim::physics::registration::SubscriptionId) {};
}

// ----------------------------------------------------------------------------
// ovstage attach/detach (the Stage is caller-owned; the adapter borrows it)
// ----------------------------------------------------------------------------

bool OvPhysxAdapter::_detachOvstage()
{
    if (m_ovstage == nullptr)
    {
        return true;
    }
    // The Stage is caller-owned, so only detach it -- never destroy it. On a failed
    // detach ovphysx still references the Stage, so keep m_ovstage set and report the
    // failure so the caller retries rather than proceeds with inconsistent state.
    ovphysx_result_t result = ovphysx_detach_ovstage(m_handle);
    if (result.status != OVPHYSX_API_SUCCESS)
    {
        ovphysx_string_t errorString = ovphysx_get_last_error();
        const std::string_view error(errorString.ptr ? errorString.ptr : "", errorString.ptr ? errorString.length : 0);
        ISAACSIM_LOG_ERROR(g_kLogger, "OvPhysxAdapter::_detachOvstage: ovphysx_detach_ovstage failed: {}", error);
        return false;
    }
    m_ovstage = nullptr;
    return true;
}

// ----------------------------------------------------------------------------
// SimulationFunctions implementations
// ----------------------------------------------------------------------------

bool OvPhysxAdapter::_doInitialize(void* callerOvstage, const char* usdIdentifier)
{
    // The adapter is process-global and may be re-initialized for a new stage (each
    // test/scene calls initialize on the same instance). Detach any borrowed Stage
    // from a previous initialize first; if detach fails the old Stage is still
    // attached, so abort rather than attach a new one over it. m_attachedStageId is
    // left untouched on that abort, so getAttachedStage() keeps reporting the Stage
    // that is still attached.
    if (!_detachOvstage())
    {
        return false;
    }

    // The detach succeeded (or nothing was attached): nothing is attached now, so
    // getAttachedStage() must read 0 until a new attach succeeds below. The id is
    // committed only on a success path, so a failed attach cannot leave a stale id
    // (getAttachedStage() == 0 after a failed initialize).
    m_attachedStageId = 0;
    m_initialized = false;
    m_stepCount = 0;
    m_timestamp = 0;
    // Drop any step dispatched against the previous stage: its op index does not carry over,
    // and its post-step event must not fire against the new one.
    m_hasPendingStep = false;
    m_postStepPending = false;

    // ovphysx parses physics from the caller-owned ovstage. The USD identifier (a
    // StageCache id as a string, or "0") is only republished via getAttachedStage();
    // the tensor views no longer read USD metadata through it.
    const long stageId = usdIdentifier ? std::strtol(usdIdentifier, nullptr, 10) : 0L;

    // Dual-input contract: ovphysx consumes the caller-owned ovstage directly. A
    // null Stage is an explicit empty scene; a non-null Stage is attached and
    // borrowed (never owned/destroyed here).
    ovstage_instance_t* stage = static_cast<ovstage_instance_t*>(callerOvstage);
    if (stage == nullptr)
    {
        m_ovstage = nullptr;
        m_attachedStageId = stageId;
        m_initialized = true;
        return true;
    }

    if (ovphysx_attach_ovstage(m_handle, stage, g_kAttachOrdinal).status != OVPHYSX_API_SUCCESS)
    {
        ovphysx_string_t errorString = ovphysx_get_last_error();
        const std::string_view error(errorString.ptr ? errorString.ptr : "", errorString.ptr ? errorString.length : 0);
        ISAACSIM_LOG_ERROR(g_kLogger, "OvPhysxAdapter::_doInitialize: ovphysx_attach_ovstage failed: {}", error);
        return false;
    }
    m_ovstage = stage;
    m_attachedStageId = stageId;
    m_initialized = true;
    return true;
}

bool OvPhysxAdapter::_doClose()
{
    if (!m_initialized)
    {
        return true;
    }
    // Keep the instance's state intact if detach fails so the Stage is not lost
    // while ovphysx still owns it; a later _doClose/destructor can retry. Report
    // the failure so the caller keeps the Stage alive rather than freeing it.
    if (!_detachOvstage())
    {
        return false;
    }
    m_initialized = false;
    m_attachedStageId = 0;
    return true;
}

long OvPhysxAdapter::_doGetAttachedStage()
{
    return m_attachedStageId;
}

void OvPhysxAdapter::_doSimulateAsynchronously(float elapsed, float /*current*/)
{
    if (!m_initialized)
    {
        return;
    }

    m_lastStepElapsed = elapsed;
    _fireStepSubscribers(/*preStep=*/true, elapsed);

    // A dispatched step owes its post-step notification even when the engine enqueues
    // nothing: an instance initialized without a Stage still steps an empty scene, and the
    // manager's contract is one post-step event per step call. Only a fetch with nothing
    // dispatched stays silent.
    m_postStepPending = true;

    if (m_handle != OVPHYSX_INVALID_HANDLE)
    {
        ovphysx_enqueue_result_t result = ovphysx_step(m_handle, elapsed);
        if (result.status == OVPHYSX_API_SUCCESS)
        {
            m_pendingStepOperation = result.op_index;
            m_hasPendingStep = true;
        }
    }

    ++m_stepCount;
    ++m_timestamp;
    // Post-step subscribers fire in _doFetchResults after ovphysx_wait_op completes,
    // so they observe settled physics state rather than the pre-wait snapshot.
}

void OvPhysxAdapter::_doSimulate(float elapsed, float current)
{
    static_cast<void>(current);
    if (!m_initialized)
    {
        return;
    }

    _fireStepSubscribers(/*preStep=*/true, elapsed);

    if (m_handle != OVPHYSX_INVALID_HANDLE)
    {
        checkResult(ovphysx_step_sync(m_handle, elapsed), "OvPhysxAdapter::_doSimulate");
    }

    ++m_stepCount;
    ++m_timestamp;

    _fireStepSubscribers(/*preStep=*/false, elapsed);
    _fireContactReportSubscribers();
}

void OvPhysxAdapter::_doFetchResults()
{
    if (!m_initialized)
    {
        return;
    }

    if (m_hasPendingStep && m_handle != OVPHYSX_INVALID_HANDLE)
    {
        ovphysx_op_wait_result_t waitResult{};
        ovphysx_wait_op(m_handle, m_pendingStepOperation, UINT64_MAX, &waitResult);
        ovphysx_destroy_wait_result(&waitResult);
        m_hasPendingStep = false;
    }

    // Post-step subscribers fire here -- once the step is complete -- so they observe settled
    // physics state. Gated on the owed notification, not on m_hasPendingStep: a _doCheckResults
    // poll may already have consumed the op index. fetchResults() is public and may also run
    // with nothing dispatched, or after a synchronous simulate() that fired its post-step
    // inline, and neither owes an event here.
    if (m_postStepPending)
    {
        m_postStepPending = false;
        _fireStepSubscribers(/*preStep=*/false, m_lastStepElapsed);
        _fireContactReportSubscribers();
    }
}

bool OvPhysxAdapter::_doCheckResults()
{
    // Non-blocking readiness poll. The synchronous path completes inside _doSimulate, so
    // only an in-flight asynchronous step can report not-ready.
    if (!m_hasPendingStep || m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return true;
    }

    // A zero timeout makes this a poll rather than a wait.
    ovphysx_op_wait_result_t waitResult{};
    const ovphysx_result_t result = ovphysx_wait_op(m_handle, m_pendingStepOperation, 0, &waitResult);
    ovphysx_destroy_wait_result(&waitResult);

    if (result.status == OVPHYSX_API_TIMEOUT)
    {
        return false;
    }
    if (result.status == OVPHYSX_API_SUCCESS || result.status == OVPHYSX_API_NOT_FOUND)
    {
        // The step finished and the poll consumed the op index (single-use), so
        // _doFetchResults must not wait on it again. It still owes the post-step
        // event, which m_postStepPending carries.
        m_hasPendingStep = false;
    }
    // On an internal error keep the pending op so _doFetchResults' blocking wait resolves it.
    return true;
}

void OvPhysxAdapter::_doFlushChanges()
{
    // No-op: ovphysx processes changes internally on step.
}

void OvPhysxAdapter::_doPauseChangeTracking(bool pause)
{
    m_changeTrackingPaused = pause;
}

bool OvPhysxAdapter::_doIsChangeTrackingPaused()
{
    return m_changeTrackingPaused;
}

isaacsim::physics::registration::SubscriptionId OvPhysxAdapter::_doSubscribeContactReport(
    isaacsim::physics::registration::OnContactReportEventFunction callback)
{
    std::lock_guard<std::mutex> lock(m_subscriberMutex);
    size_t subscriptionId = m_nextSubscriptionId++;
    m_contactSubscribers[subscriptionId] = std::move(callback);
    return isaacsim::physics::registration::SubscriptionId(subscriptionId);
}

void OvPhysxAdapter::_doUnsubscribeContactReport(isaacsim::physics::registration::SubscriptionId subscriptionId)
{
    std::lock_guard<std::mutex> lock(m_subscriberMutex);
    m_contactSubscribers.erase(subscriptionId.id);
}

uint32_t OvPhysxAdapter::_doGetTimeStepsPerSecond(long /*stageId*/,
                                                  isaacsim::physics::registration::PathToken /*scenePath*/)
{
    return 60u;
}

uint64_t OvPhysxAdapter::_doGetTimestamp()
{
    return m_timestamp.load();
}

uint64_t OvPhysxAdapter::_doGetStepCount()
{
    return m_stepCount.load();
}

isaacsim::physics::registration::SubscriptionId OvPhysxAdapter::_doSubscribeStepEvents(
    bool preStep, int order, isaacsim::physics::registration::OnPhysicsStepEventFunction callback)
{
    std::lock_guard<std::mutex> lock(m_subscriberMutex);
    size_t subscriptionId = m_nextSubscriptionId++;
    m_stepSubscribers[subscriptionId] = { preStep, order, std::move(callback) };
    return isaacsim::physics::registration::SubscriptionId(subscriptionId);
}

void OvPhysxAdapter::_doUnsubscribeStepEvents(isaacsim::physics::registration::SubscriptionId subscriptionId)
{
    std::lock_guard<std::mutex> lock(m_subscriberMutex);
    m_stepSubscribers.erase(subscriptionId.id);
}

bool OvPhysxAdapter::_doIsCapableOfSimulating(const char** schemaNames, size_t count, bool* isCapable)
{
    static const char* s_kSupportedSchemas[] = {
        "PhysicsRigidBodyAPI",  "PhysicsCollisionAPI",   "PhysicsArticulationRootAPI",
        "PhysicsRevoluteJoint", "PhysicsPrismaticJoint", "PhysicsSphericalJoint",
        "PhysicsFixedJoint",    "PhysicsDistanceJoint",  "PhysicsD6Joint",
    };
    for (size_t i = 0; i < count; ++i)
    {
        isCapable[i] = false;
        for (auto* supportedSchema : s_kSupportedSchemas)
        {
            if (schemaNames[i] && std::string(schemaNames[i]) == supportedSchema)
            {
                isCapable[i] = true;
                break;
            }
        }
    }
    return true;
}

// ----------------------------------------------------------------------------
// Step subscriber dispatch
// ----------------------------------------------------------------------------

void OvPhysxAdapter::_fireStepSubscribers(bool preStep, float elapsed)
{
    // Collect matching subscribers under the lock, then fire outside it.
    std::vector<std::pair<int, isaacsim::physics::registration::OnPhysicsStepEventFunction>> subscribers;
    {
        std::lock_guard<std::mutex> lock(m_subscriberMutex);
        for (auto& subscriberEntry : m_stepSubscribers)
        {
            if (subscriberEntry.second.preStep == preStep)
            {
                subscribers.emplace_back(subscriberEntry.second.order, subscriberEntry.second.callback);
            }
        }
    }
    std::sort(subscribers.begin(), subscribers.end(),
              [](const auto& left, const auto& right) { return left.first < right.first; });

    isaacsim::physics::registration::PhysicsStepContext context{};
    context.simulationId = isaacsim::physics::registration::SimulationId(m_simulationId);
    context.scenePath = 0;

    for (auto& subscriber : subscribers)
    {
        try
        {
            subscriber.second(elapsed, context);
        }
        catch (...)
        {
        }
    }
}

void OvPhysxAdapter::_fireContactReportSubscribers()
{
    // Collect matching subscribers under the lock, then fire outside it (mirrors
    // _fireStepSubscribers). Skip the engine query entirely when nobody is listening.
    std::vector<isaacsim::physics::registration::OnContactReportEventFunction> subscribers;
    {
        std::lock_guard<std::mutex> lock(m_subscriberMutex);
        subscribers.reserve(m_contactSubscribers.size());
        for (auto& subscriberEntry : m_contactSubscribers)
        {
            subscribers.push_back(subscriberEntry.second);
        }
    }
    if (subscribers.empty() || m_handle == OVPHYSX_INVALID_HANDLE)
    {
        return;
    }

    const ovphysx_contact_event_header_t* eventHeaders = nullptr;
    uint32_t eventHeaderCount = 0;
    const ovphysx_contact_point_t* contactPoints = nullptr;
    uint32_t contactPointCount = 0;
    const ovphysx_friction_anchor_t* frictionAnchors = nullptr;
    uint32_t frictionAnchorCount = 0;
    const ovphysx_result_t result =
        ovphysx_get_contact_report(m_handle, &eventHeaders, &eventHeaderCount, &contactPoints, &contactPointCount,
                                   &frictionAnchors, &frictionAnchorCount);
    if (result.status != OVPHYSX_API_SUCCESS)
    {
        ovphysx_string_t errorString = ovphysx_get_last_error();
        const std::string_view error(errorString.ptr ? errorString.ptr : "", errorString.ptr ? errorString.length : 0);
        ISAACSIM_LOG_ERROR(
            g_kLogger, "OvPhysxAdapter: ovphysx_get_contact_report failed, contact report dropped: {}", error);
        return;
    }

    // Copy out of the engine's buffers: they are read-only views invalidated by the next step,
    // and a subscriber may outlive this call's scope in its own bookkeeping.
    isaacsim::physics::registration::ContactEventHeaderVector headers;
    headers.reserve(eventHeaderCount);
    for (uint32_t i = 0; i < eventHeaderCount; ++i)
    {
        const ovphysx_contact_event_header_t& source = eventHeaders[i];
        isaacsim::physics::registration::ContactEventHeader header{};
        header.type = toContactEventType(source.type);
        header.stageId = source.stageId;
        header.actor0 = static_cast<isaacsim::physics::registration::PathToken>(source.actor0);
        header.actor1 = static_cast<isaacsim::physics::registration::PathToken>(source.actor1);
        header.collider0 = static_cast<isaacsim::physics::registration::PathToken>(source.collider0);
        header.collider1 = static_cast<isaacsim::physics::registration::PathToken>(source.collider1);
        header.contactDataOffset = source.contactDataOffset;
        header.contactDataCount = source.numContactData;
        header.frictionAnchorsDataOffset = source.frictionAnchorsDataOffset;
        header.frictionAnchorDataCount = source.numfrictionAnchorsData;
        header.prototypeIndex0 = source.protoIndex0;
        header.prototypeIndex1 = source.protoIndex1;
        headers.push_back(header);
    }

    isaacsim::physics::registration::ContactDataVector contactData;
    contactData.reserve(contactPointCount);
    for (uint32_t i = 0; i < contactPointCount; ++i)
    {
        const ovphysx_contact_point_t& source = contactPoints[i];
        isaacsim::physics::registration::ContactData point{};
        point.position = { source.position[0], source.position[1], source.position[2] };
        point.normal = { source.normal[0], source.normal[1], source.normal[2] };
        point.impulse = { source.impulse[0], source.impulse[1], source.impulse[2] };
        point.separation = source.separation;
        point.faceIndex0 = source.faceIndex0;
        point.faceIndex1 = source.faceIndex1;
        point.material0 = static_cast<isaacsim::physics::registration::PathToken>(source.material0);
        point.material1 = static_cast<isaacsim::physics::registration::PathToken>(source.material1);
        contactData.push_back(point);
    }

    isaacsim::physics::registration::FrictionAnchorsDataVector anchors;
    anchors.reserve(frictionAnchorCount);
    for (uint32_t i = 0; i < frictionAnchorCount; ++i)
    {
        const ovphysx_friction_anchor_t& source = frictionAnchors[i];
        isaacsim::physics::registration::FrictionAnchor anchor{};
        anchor.position = { source.position[0], source.position[1], source.position[2] };
        anchor.impulse = { source.impulse[0], source.impulse[1], source.impulse[2] };
        anchors.push_back(anchor);
    }

    for (auto& callback : subscribers)
    {
        try
        {
            callback(headers, contactData, anchors);
        }
        catch (...)
        {
            ISAACSIM_LOG_ERROR(g_kLogger, "OvPhysxAdapter: a contact-report subscriber threw; continuing");
        }
    }
}

// ----------------------------------------------------------------------------
// SceneQueryFunctions
// ----------------------------------------------------------------------------

void OvPhysxAdapter::_fillSceneQueryFunctions(isaacsim::physics::registration::SceneQueryFunctions& sceneQueryFunctions)
{
    sceneQueryFunctions.raycastClosest = [this](const isaacsim::physics::registration::Float3& origin,
                                                const isaacsim::physics::registration::Float3& direction, float distance,
                                                isaacsim::physics::registration::RaycastHit& hit, bool both) -> bool
    {
        const float originCoordinates[3]{ origin.x, origin.y, origin.z };
        const float directionCoordinates[3]{ direction.x, direction.y, direction.z };
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_raycast(m_handle, originCoordinates, directionCoordinates, distance, both,
                        OVPHYSX_SCENE_QUERY_MODE_CLOSEST, &hits, &count);
        if (count == 0)
        {
            return false;
        }
        hit = toRaycastHit(hits[0]);
        return true;
    };

    sceneQueryFunctions.raycastAny = [this](const isaacsim::physics::registration::Float3& origin,
                                            const isaacsim::physics::registration::Float3& direction, float distance,
                                            bool both) -> bool
    {
        const float originCoordinates[3]{ origin.x, origin.y, origin.z };
        const float directionCoordinates[3]{ direction.x, direction.y, direction.z };
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_raycast(m_handle, originCoordinates, directionCoordinates, distance, both, OVPHYSX_SCENE_QUERY_MODE_ANY,
                        &hits, &count);
        return count > 0;
    };

    sceneQueryFunctions.raycastAll = [this](const isaacsim::physics::registration::Float3& origin,
                                            const isaacsim::physics::registration::Float3& direction, float distance,
                                            isaacsim::physics::registration::RaycastHitReportFunction callback, bool both)
    {
        const float originCoordinates[3]{ origin.x, origin.y, origin.z };
        const float directionCoordinates[3]{ direction.x, direction.y, direction.z };
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_raycast(m_handle, originCoordinates, directionCoordinates, distance, both, OVPHYSX_SCENE_QUERY_MODE_ALL,
                        &hits, &count);
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!callback(toRaycastHit(hits[i])))
            {
                break;
            }
        }
    };

    sceneQueryFunctions.sweepSphereClosest = [this](float radius, const isaacsim::physics::registration::Float3& origin,
                                                    const isaacsim::physics::registration::Float3& direction,
                                                    float distance, isaacsim::physics::registration::SweepHit& hit,
                                                    bool both) -> bool
    {
        auto geometry = makeSphereGeometry(radius, origin);
        const float directionCoordinates[3]{ direction.x, direction.y, direction.z };
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_sweep(
            m_handle, &geometry, directionCoordinates, distance, both, OVPHYSX_SCENE_QUERY_MODE_CLOSEST, &hits, &count);
        if (count == 0)
        {
            return false;
        }
        hit = toSweepHit(hits[0]);
        return true;
    };

    sceneQueryFunctions.sweepSphereAny = [this](float radius, const isaacsim::physics::registration::Float3& origin,
                                                const isaacsim::physics::registration::Float3& direction,
                                                float distance, bool both) -> bool
    {
        auto geometry = makeSphereGeometry(radius, origin);
        const float directionCoordinates[3]{ direction.x, direction.y, direction.z };
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_sweep(
            m_handle, &geometry, directionCoordinates, distance, both, OVPHYSX_SCENE_QUERY_MODE_ANY, &hits, &count);
        return count > 0;
    };

    sceneQueryFunctions.sweepSphereAll = [this](float radius, const isaacsim::physics::registration::Float3& origin,
                                                const isaacsim::physics::registration::Float3& direction, float distance,
                                                isaacsim::physics::registration::SweepHitReportFunction callback,
                                                bool both)
    {
        auto geometry = makeSphereGeometry(radius, origin);
        const float directionCoordinates[3]{ direction.x, direction.y, direction.z };
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_sweep(
            m_handle, &geometry, directionCoordinates, distance, both, OVPHYSX_SCENE_QUERY_MODE_ALL, &hits, &count);
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!callback(toSweepHit(hits[i])))
            {
                break;
            }
        }
    };

    sceneQueryFunctions.sweepBoxClosest = [this](const isaacsim::physics::registration::Float3& halfExtent,
                                                 const isaacsim::physics::registration::Float3& position,
                                                 const isaacsim::physics::registration::Float4& rotation,
                                                 const isaacsim::physics::registration::Float3& direction, float distance,
                                                 isaacsim::physics::registration::SweepHit& hit, bool both) -> bool
    {
        auto geometry = makeBoxGeometry(halfExtent, position, rotation);
        const float directionCoordinates[3]{ direction.x, direction.y, direction.z };
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_sweep(
            m_handle, &geometry, directionCoordinates, distance, both, OVPHYSX_SCENE_QUERY_MODE_CLOSEST, &hits, &count);
        if (count == 0)
        {
            return false;
        }
        hit = toSweepHit(hits[0]);
        return true;
    };

    sceneQueryFunctions.sweepBoxAny = [this](const isaacsim::physics::registration::Float3& halfExtent,
                                             const isaacsim::physics::registration::Float3& position,
                                             const isaacsim::physics::registration::Float4& rotation,
                                             const isaacsim::physics::registration::Float3& direction, float distance,
                                             bool both) -> bool
    {
        auto geometry = makeBoxGeometry(halfExtent, position, rotation);
        const float directionCoordinates[3]{ direction.x, direction.y, direction.z };
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_sweep(
            m_handle, &geometry, directionCoordinates, distance, both, OVPHYSX_SCENE_QUERY_MODE_ANY, &hits, &count);
        return count > 0;
    };

    sceneQueryFunctions.sweepBoxAll = [this](const isaacsim::physics::registration::Float3& halfExtent,
                                             const isaacsim::physics::registration::Float3& position,
                                             const isaacsim::physics::registration::Float4& rotation,
                                             const isaacsim::physics::registration::Float3& direction, float distance,
                                             isaacsim::physics::registration::SweepHitReportFunction callback, bool both)
    {
        auto geometry = makeBoxGeometry(halfExtent, position, rotation);
        const float directionCoordinates[3]{ direction.x, direction.y, direction.z };
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_sweep(
            m_handle, &geometry, directionCoordinates, distance, both, OVPHYSX_SCENE_QUERY_MODE_ALL, &hits, &count);
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!callback(toSweepHit(hits[i])))
            {
                break;
            }
        }
    };

    // Shape-based sweep/overlap: the C API takes a prim path string. The manager
    // passes an encoded PathToken (uint64 SdfPath). Decode to string is not available
    // without USD; leave these as no-ops for now -- they match the Python behavior
    // which also required a USD prim path but was never exercised by the manager tests.
    sceneQueryFunctions.sweepShapeClosest = nullptr;
    sceneQueryFunctions.sweepShapeAny = nullptr;
    sceneQueryFunctions.sweepShapeAll = nullptr;

    sceneQueryFunctions.overlapSphere = [this](float radius, const isaacsim::physics::registration::Float3& position,
                                               isaacsim::physics::registration::OverlapHitReportFunction callback) -> uint32_t
    {
        auto geometry = makeSphereGeometry(radius, position);
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_overlap(m_handle, &geometry, OVPHYSX_SCENE_QUERY_MODE_ALL, &hits, &count);
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!callback(toOverlapHit(hits[i])))
            {
                break;
            }
        }
        return count;
    };

    sceneQueryFunctions.overlapSphereAny = [this](float radius,
                                                  const isaacsim::physics::registration::Float3& position) -> bool
    {
        auto geometry = makeSphereGeometry(radius, position);
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_overlap(m_handle, &geometry, OVPHYSX_SCENE_QUERY_MODE_ANY, &hits, &count);
        return count > 0;
    };

    sceneQueryFunctions.overlapBox = [this](const isaacsim::physics::registration::Float3& halfExtent,
                                            const isaacsim::physics::registration::Float3& position,
                                            const isaacsim::physics::registration::Float4& rotation,
                                            isaacsim::physics::registration::OverlapHitReportFunction callback) -> uint32_t
    {
        auto geometry = makeBoxGeometry(halfExtent, position, rotation);
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_overlap(m_handle, &geometry, OVPHYSX_SCENE_QUERY_MODE_ALL, &hits, &count);
        for (uint32_t i = 0; i < count; ++i)
        {
            if (!callback(toOverlapHit(hits[i])))
            {
                break;
            }
        }
        return count;
    };

    sceneQueryFunctions.overlapBoxAny = [this](const isaacsim::physics::registration::Float3& halfExtent,
                                               const isaacsim::physics::registration::Float3& position,
                                               const isaacsim::physics::registration::Float4& rotation) -> bool
    {
        auto geometry = makeBoxGeometry(halfExtent, position, rotation);
        const ovphysx_scene_query_hit_t* hits = nullptr;
        uint32_t count = 0;
        ovphysx_overlap(m_handle, &geometry, OVPHYSX_SCENE_QUERY_MODE_ANY, &hits, &count);
        return count > 0;
    };

    sceneQueryFunctions.overlapShape = nullptr;
    sceneQueryFunctions.overlapShapeAny = nullptr;
}

} // namespace ovphysx
} // namespace physics_engines
} // namespace isaacsim
