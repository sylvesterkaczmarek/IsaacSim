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

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include "IsaacSimSystem.hpp"

#include <carb/Framework.h>
#include <carb/InterfaceUtils.h>
#include <carb/profiler/Profile.h>

#include <isaacsim/core/includes/PhysicsEngine.hpp>
#include <isaacsim/sensors/experimental/physics/IImuSensor.hpp>
// clang-format off
#include <omni/usd/UsdContextIncludes.h>
#include <omni/usd/UsdContext.h>
#include <pxr/usd/sdf/path.h>
// clang-format on
#include <pluginlib/class_list_macros.hpp>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdUtils/stageCache.h>
#include <rclcpp/logging.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <limits>
#include <mutex>
#include <unordered_map>

#ifdef min
#    undef min
#endif
#ifdef max
#    undef max
#endif
#ifdef ERROR
#    undef ERROR
#endif

using hardware_interface::CallbackReturn;
using hardware_interface::return_type;

namespace isaacsim
{
namespace ros2
{
namespace control
{
namespace backend
{

namespace
{

struct GainBaseline
{
    pxr::UsdStageWeakPtr stage;
    std::vector<std::string> jointNames;
    std::vector<float> stiffness;
    std::vector<float> damping;
};

std::mutex g_gainBaselineMutex;
std::unordered_map<std::string, GainBaseline> g_gainBaselines;

inline void fillCpuFloat1D(omni::physics::tensors::TensorDesc& desc, void* data, int n)
{
    desc.device = -1;
    desc.dtype = omni::physics::tensors::TensorDataType::eFloat32;
    desc.numDims = 1;
    desc.dims[0] = n;
    desc.data = data;
    desc.ownData = false;
}

bool extractImuValue(const std::string& name,
                     const isaacsim::sensors::experimental::physics::ImuSensorReading& r,
                     double& out)
{
    using Reading = isaacsim::sensors::experimental::physics::ImuSensorReading;
    static const struct
    {
        const char* suffix;
        float Reading::*field;
    } kMap[] = {
        { "orientation.x", &Reading::orientationX },
        { "orientation.y", &Reading::orientationY },
        { "orientation.z", &Reading::orientationZ },
        { "orientation.w", &Reading::orientationW },
        { "angular_velocity.x", &Reading::angularVelocityX },
        { "angular_velocity.y", &Reading::angularVelocityY },
        { "angular_velocity.z", &Reading::angularVelocityZ },
        { "linear_acceleration.x", &Reading::linearAccelerationX },
        { "linear_acceleration.y", &Reading::linearAccelerationY },
        { "linear_acceleration.z", &Reading::linearAccelerationZ },
    };
    for (auto& m : kMap)
    {
        if (name == m.suffix)
        {
            out = static_cast<double>(r.*m.field);
            return true;
        }
    }
    return false;
}

// Maps a force_torque_sensor_broadcaster interface name to its offset in the
// per-link 6-vector returned by getLinkIncomingJointForce (force.xyz, torque.xyz).
bool extractFtComponent(const std::string& name, int& offset)
{
    static const struct
    {
        const char* suffix;
        int offset;
    } kMap[] = {
        { "force.x", 0 }, { "force.y", 1 }, { "force.z", 2 }, { "torque.x", 3 }, { "torque.y", 4 }, { "torque.z", 5 },
    };
    for (auto& m : kMap)
        if (name == m.suffix)
        {
            offset = m.offset;
            return true;
        }
    return false;
}

const rclcpp::Logger& getIsaacSimSystemLogger()
{
    static const rclcpp::Logger logger = rclcpp::get_logger("isaacsim.ros2.control.IsaacSimSystem");
    return logger;
}

}

omni::physics::tensors::TensorApi* getInjectedTensorApi();

IsaacSimSystem::~IsaacSimSystem()
{
    if (m_physicsSimulation && m_preStepSub != omni::physics::kInvalidSubscriptionId)
        m_physicsSimulation->unsubscribePhysicsOnStepEvents(m_preStepSub);
}

#if defined(ROS2_BACKEND_JAZZY)
CallbackReturn IsaacSimSystem::on_init(const hardware_interface::HardwareComponentInterfaceParams& params)
{
    if (SystemInterface::on_init(params) != CallbackReturn::SUCCESS)
        return CallbackReturn::ERROR;
    const hardware_interface::HardwareInfo& info = params.hardware_info;
#else
CallbackReturn IsaacSimSystem::on_init(const hardware_interface::HardwareInfo& info)
{
    if (SystemInterface::on_init(info) != CallbackReturn::SUCCESS)
        return CallbackReturn::ERROR;
#endif
    auto it = info.hardware_parameters.find("prim_path");
    if (it == info.hardware_parameters.end())
    {
        RCLCPP_ERROR(getIsaacSimSystemLogger(), "IsaacSimSystem: <hardware> missing <param name=\"prim_path\">");
        return CallbackReturn::ERROR;
    }
    m_articulationPath = it->second;

    m_jointNames.reserve(info.joints.size());
    for (auto& j : info.joints)
        m_jointNames.push_back(j.name);
    m_position.assign(m_jointNames.size(), 0.0);
    m_velocity.assign(m_jointNames.size(), 0.0);
    m_effort.assign(m_jointNames.size(), 0.0);
    // NaN marks an unwritten command; position/velocity skip it, while effort
    // treats it as zero.
    const double cmdInit = std::numeric_limits<double>::quiet_NaN();
    m_positionCmd.assign(m_jointNames.size(), cmdInit);
    m_velocityCmd.assign(m_jointNames.size(), cmdInit);
    m_effortCmd.assign(m_jointNames.size(), cmdInit);

    // A joint may declare several command interfaces; the active one is chosen at
    // runtime by perform_command_mode_switch. No interface drives until claimed.
    m_cmdMask.assign(m_jointNames.size(), 0);
    m_activeMode.assign(m_jointNames.size(), kNone);
    m_appliedMode.assign(m_jointNames.size(), kNone);
    for (size_t i = 0; i < info.joints.size(); ++i)
    {
        for (auto& ci : info.joints[i].command_interfaces)
        {
            if (ci.name == "position")
                m_cmdMask[i] |= kBitPosition;
            else if (ci.name == "velocity")
                m_cmdMask[i] |= kBitVelocity;
            else if (ci.name == "effort")
                m_cmdMask[i] |= kBitEffort;
        }
    }

    // Mimic joints: urdf_synth forwards the relationship as joint <param>s because
    // ros2_control parses <mimic> natively on jazzy but not humble. We read it here and
    // enforce it in write() on any physics backend. The follower exports no command
    // interface (m_cmdMask stays 0), so only this mapping drives it.
    const size_t numJoints = m_jointNames.size();
    m_mimicLeader.assign(numJoints, -1);
    m_mimicMultiplier.assign(numJoints, 1.0);
    m_mimicOffset.assign(numJoints, 0.0);
    std::unordered_map<std::string, size_t> jointNameToIndex;
    for (size_t i = 0; i < numJoints; ++i)
        jointNameToIndex[m_jointNames[i]] = i;
    for (size_t i = 0; i < info.joints.size(); ++i)
    {
        auto mit = info.joints[i].parameters.find("mimic");
        if (mit == info.joints[i].parameters.end())
            continue;
        auto lit = jointNameToIndex.find(mit->second);
        if (lit == jointNameToIndex.end())
        {
            RCLCPP_ERROR(getIsaacSimSystemLogger(),
                         "IsaacSimSystem: mimic joint '%s' references unknown leader joint '%s'.",
                         m_jointNames[i].c_str(), mit->second.c_str());
            return CallbackReturn::ERROR;
        }
        m_mimicLeader[i] = static_cast<int32_t>(lit->second);
        auto mul = info.joints[i].parameters.find("multiplier");
        if (mul != info.joints[i].parameters.end())
            m_mimicMultiplier[i] = std::strtod(mul->second.c_str(), nullptr);
        auto off = info.joints[i].parameters.find("offset");
        if (off != info.joints[i].parameters.end())
            m_mimicOffset[i] = std::strtod(off->second.c_str(), nullptr);
    }

    m_sensors.clear();
    m_sensors.reserve(info.sensors.size());
    for (auto& s : info.sensors)
    {
        SensorBinding b;
        b.sensorName = s.name;
        b.interfaceNames.reserve(s.state_interfaces.size());
        for (auto& si : s.state_interfaces)
            b.interfaceNames.push_back(si.name);
        b.values.assign(b.interfaceNames.size(), 0.0);
        auto pit = s.parameters.find("prim_path");
        if (pit != s.parameters.end())
            b.primPath = pit->second;
        auto lit = s.parameters.find("link");
        if (lit != s.parameters.end())
            b.linkName = lit->second;
        m_sensors.push_back(std::move(b));
    }

    return CallbackReturn::SUCCESS;
}

CallbackReturn IsaacSimSystem::on_configure(const rclcpp_lifecycle::State&)
{
    CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::on_configure");

    m_tensorApi = getInjectedTensorApi();
    if (!m_tensorApi)
        m_tensorApi = carb::getCachedInterface<omni::physics::tensors::TensorApi>();
    if (!m_tensorApi)
    {
        RCLCPP_ERROR(getIsaacSimSystemLogger(), "IsaacSimSystem: failed to acquire omni::physics::tensors::TensorApi");
        return CallbackReturn::ERROR;
    }

    auto* usdContext = omni::usd::UsdContext::getContext();
    if (!usdContext)
    {
        RCLCPP_ERROR(getIsaacSimSystemLogger(), "IsaacSimSystem: omni::usd::UsdContext::getContext() returned null");
        return CallbackReturn::ERROR;
    }
    pxr::UsdStageRefPtr stage = usdContext->getStage();
    if (!stage)
    {
        RCLCPP_ERROR(getIsaacSimSystemLogger(), "IsaacSimSystem: no USD stage attached to UsdContext");
        return CallbackReturn::ERROR;
    }
    long stageId = pxr::UsdUtilsStageCache::Get().GetId(stage).ToLongInt();

    // Bind to the active physics engine; nullptr lets the tensor API choose the
    // default backend when no simulation is active yet.
    const char* engine = isaacsim::core::includes::getActivePhysicsEngineName();
    m_simView = m_tensorApi->createSimulationView(stageId, engine);
    if (!m_simView)
    {
        RCLCPP_ERROR(getIsaacSimSystemLogger(), "IsaacSimSystem: createSimulationView(stageId=%ld, engine=\"%s\") failed",
                     stageId, engine ? engine : "auto");
        return CallbackReturn::ERROR;
    }

    m_articulationView = m_simView->createArticulationView(m_articulationPath.c_str());
    if (!m_articulationView)
    {
        RCLCPP_ERROR(getIsaacSimSystemLogger(),
                     "IsaacSimSystem: createArticulationView(\"%s\") failed; prim is not an articulation?",
                     m_articulationPath.c_str());
        m_simView = nullptr;
        return CallbackReturn::ERROR;
    }

    m_numDofs = static_cast<size_t>(m_articulationView->getMaxDofs());
    m_readBuf.assign(m_numDofs, 0.0f);
    m_writeBuf.assign(m_numDofs, 0.0f);
    m_maxLinks = m_articulationView->getMaxLinks();
    m_linkForceBuf.assign(static_cast<size_t>(m_maxLinks) * 6, 0.0f);
    m_hasFtSensors = false;
    m_warnedFtReadFailure = false;

    const omni::physics::tensors::IArticulationMetatype* meta = m_articulationView->getSharedMetatype();
    if (!meta)
    {
        RCLCPP_ERROR(getIsaacSimSystemLogger(),
                     "IsaacSimSystem: articulation '%s' has no metatype; cannot map joints to DOFs.",
                     m_articulationPath.c_str());
        m_articulationView = nullptr;
        m_simView = nullptr;
        return CallbackReturn::ERROR;
    }
    m_dofIndex.assign(m_jointNames.size(), -1);
    for (size_t i = 0; i < m_jointNames.size(); ++i)
    {
        int32_t d = meta->findDofIndex(m_jointNames[i].c_str());
        if (d < 0 || static_cast<size_t>(d) >= m_numDofs)
        {
            RCLCPP_ERROR(getIsaacSimSystemLogger(),
                         "IsaacSimSystem: URDF joint '%s' has no matching DOF in articulation '%s'. "
                         "Check that joint names match the USD joint prim names.",
                         m_jointNames[i].c_str(), m_articulationPath.c_str());
            m_articulationView = nullptr;
            m_simView = nullptr;
            return CallbackReturn::ERROR;
        }
        m_dofIndex[i] = d;
    }

    // Verify all DOF indices are distinct. Physics backends may deduplicate short joint names;
    // two URDF joints mapping to the same DOF index would silently alias.
    for (size_t i = 0; i < m_dofIndex.size(); ++i)
    {
        for (size_t j = i + 1; j < m_dofIndex.size(); ++j)
        {
            if (m_dofIndex[i] == m_dofIndex[j])
            {
                RCLCPP_ERROR(getIsaacSimSystemLogger(),
                             "IsaacSimSystem: joints '%s' and '%s' both map to DOF index %d in "
                             "articulation '%s'. Joint names must be unique in the physics DOF table.",
                             m_jointNames[i].c_str(), m_jointNames[j].c_str(), m_dofIndex[i], m_articulationPath.c_str());
                m_articulationView = nullptr;
                m_simView = nullptr;
                return CallbackReturn::ERROR;
            }
        }
    }

    m_authoredStiffness.assign(m_jointNames.size(), 0.0f);
    m_authoredDamping.assign(m_jointNames.size(), 0.0f);
    bool reusedBaseline = false;
    {
        std::lock_guard<std::mutex> lock(g_gainBaselineMutex);
        auto it = g_gainBaselines.find(m_articulationPath);
        if (it != g_gainBaselines.end() && it->second.stage == stage && it->second.jointNames == m_jointNames)
        {
            m_authoredStiffness = it->second.stiffness;
            m_authoredDamping = it->second.damping;
            reusedBaseline = true;
        }
    }
    if (!reusedBaseline)
    {
        omni::physics::tensors::TensorDesc desc;
        fillCpuFloat1D(desc, m_readBuf.data(), static_cast<int>(m_numDofs));
        if (!m_articulationView->getDofStiffnesses(&desc))
        {
            RCLCPP_ERROR(getIsaacSimSystemLogger(), "IsaacSimSystem: failed to read authored stiffness for '%s'",
                         m_articulationPath.c_str());
            m_articulationView = nullptr;
            m_simView = nullptr;
            return CallbackReturn::ERROR;
        }
        for (size_t i = 0; i < m_jointNames.size(); ++i)
            m_authoredStiffness[i] = m_readBuf[m_dofIndex[i]];
        if (!m_articulationView->getDofDampings(&desc))
        {
            RCLCPP_ERROR(getIsaacSimSystemLogger(), "IsaacSimSystem: failed to read authored damping for '%s'",
                         m_articulationPath.c_str());
            m_articulationView = nullptr;
            m_simView = nullptr;
            return CallbackReturn::ERROR;
        }
        for (size_t i = 0; i < m_jointNames.size(); ++i)
            m_authoredDamping[i] = m_readBuf[m_dofIndex[i]];
        std::lock_guard<std::mutex> lock(g_gainBaselineMutex);
        g_gainBaselines[m_articulationPath] = { stage, m_jointNames, m_authoredStiffness, m_authoredDamping };
    }
    {
        std::lock_guard<std::mutex> lock(m_modeMutex);
        std::fill(m_activeMode.begin(), m_activeMode.end(), kNone);
        std::fill(m_appliedMode.begin(), m_appliedMode.end(), kNone);
        m_gainsDirty = reusedBaseline;
    }

    // A mimic follower is driven to a position target each step, so it needs a position
    // drive (stiffness > 0) to track its leader. Warn once if one is missing.
    for (size_t i = 0; i < m_jointNames.size(); ++i)
        if (m_mimicLeader[i] >= 0 && m_authoredStiffness[i] <= 0.0f && !m_warnedMimicNoDrive)
        {
            RCLCPP_WARN(getIsaacSimSystemLogger(),
                        "IsaacSimSystem: mimic joint '%s' has no position drive (stiffness=0); it "
                        "cannot track leader '%s'. Author a position DriveAPI on the mimic joint.",
                        m_jointNames[i].c_str(), m_jointNames[m_mimicLeader[i]].c_str());
            m_warnedMimicNoDrive = true;
        }

    if (!m_sensors.empty())
    {
        m_imuApi = carb::getCachedInterface<isaacsim::sensors::experimental::physics::IImuSensor>();
        if (!m_imuApi)
        {
            RCLCPP_WARN(getIsaacSimSystemLogger(),
                        "IsaacSimSystem: IImuSensor interface unavailable; "
                        "is isaacsim.sensors.experimental.physics loaded? IMU readings will be zero.");
        }
    }

    pxr::SdfPath articulationSdfPath(m_articulationPath);
    pxr::UsdPrim articulationPrim = stage->GetPrimAtPath(articulationSdfPath);

    for (auto& s : m_sensors)
    {
        // A sensor whose interfaces are all force.*/torque.* is a force-torque
        // sensor: it reads the joint reaction wrench by link, so it binds to a link
        // (required <param name="link">) and skips USD sensor-prim resolution.
        bool allFt = !s.interfaceNames.empty();
        for (auto& iname : s.interfaceNames)
        {
            int off;
            if (!extractFtComponent(iname, off))
            {
                allFt = false;
                break;
            }
        }
        if (allFt)
        {
            if (s.linkName.empty())
            {
                RCLCPP_ERROR(getIsaacSimSystemLogger(),
                             "IsaacSimSystem: force-torque sensor '%s' requires <param name=\"link\"> naming "
                             "the joint's child link.",
                             s.sensorName.c_str());
                return CallbackReturn::ERROR;
            }
            int32_t li = meta->findLinkIndex(s.linkName.c_str());
            if (li < 0 || static_cast<uint32_t>(li) >= m_maxLinks)
            {
                RCLCPP_ERROR(getIsaacSimSystemLogger(),
                             "IsaacSimSystem: force-torque sensor '%s' link '%s' has no matching link in "
                             "articulation '%s'.",
                             s.sensorName.c_str(), s.linkName.c_str(), m_articulationPath.c_str());
                return CallbackReturn::ERROR;
            }
            s.kind = SensorBinding::Kind::ForceTorque;
            s.linkIndex = li;
            m_hasFtSensors = true;
            continue;
        }

        if (s.primPath.empty())
        {
            if (!articulationPrim)
            {
                RCLCPP_ERROR(getIsaacSimSystemLogger(),
                             "IsaacSimSystem: sensor '%s' requires by-name lookup but articulation prim '%s' "
                             "is not on the stage.",
                             s.sensorName.c_str(), m_articulationPath.c_str());
                return CallbackReturn::ERROR;
            }
            std::vector<std::string> matches;
            for (const pxr::UsdPrim& p : pxr::UsdPrimRange(articulationPrim))
            {
                if (p.GetName().GetString() == s.sensorName)
                {
                    matches.push_back(p.GetPath().GetString());
                }
            }
            if (matches.empty())
            {
                RCLCPP_ERROR(getIsaacSimSystemLogger(),
                             "IsaacSimSystem: no USD prim named '%s' under articulation '%s' for <sensor> "
                             "block. Either add the sensor prim to USD or supply "
                             "<param name=\"prim_path\"> on the <sensor>.",
                             s.sensorName.c_str(), m_articulationPath.c_str());
                return CallbackReturn::ERROR;
            }
            if (matches.size() > 1)
            {
                std::string list;
                for (size_t i = 0; i < matches.size(); ++i)
                {
                    if (i)
                        list += ", ";
                    list += matches[i];
                }
                RCLCPP_ERROR(getIsaacSimSystemLogger(),
                             "IsaacSimSystem: sensor name '%s' matches %zu prims under '%s': [%s]. "
                             "Disambiguate with <param name=\"prim_path\"> on the <sensor>.",
                             s.sensorName.c_str(), matches.size(), m_articulationPath.c_str(), list.c_str());
                return CallbackReturn::ERROR;
            }
            s.primPath = matches[0];
        }

        pxr::UsdPrim sensorPrim = stage->GetPrimAtPath(pxr::SdfPath(s.primPath));
        if (!sensorPrim)
        {
            RCLCPP_ERROR(getIsaacSimSystemLogger(),
                         "IsaacSimSystem: sensor '%s' resolved to '%s' but that prim does not exist.",
                         s.sensorName.c_str(), s.primPath.c_str());
            return CallbackReturn::ERROR;
        }
        const std::string typeName = sensorPrim.GetTypeName().GetString();
        if (typeName != "IsaacImuSensor")
        {
            RCLCPP_WARN(getIsaacSimSystemLogger(),
                        "IsaacSimSystem: sensor '%s' at '%s' has unsupported USD type '%s'; "
                        "only IsaacImuSensor is supported. Values will remain zero.",
                        s.sensorName.c_str(), s.primPath.c_str(), typeName.c_str());
            continue;
        }
        s.kind = SensorBinding::Kind::IMU;

        for (auto& iname : s.interfaceNames)
        {
            double dummy;
            isaacsim::sensors::experimental::physics::ImuSensorReading r{};
            if (!extractImuValue(iname, r, dummy) && !s.warnedUnknownInterface)
            {
                RCLCPP_WARN(getIsaacSimSystemLogger(),
                            "IsaacSimSystem: sensor '%s' declares unrecognised state_interface '%s'; "
                            "value will stay 0.0.",
                            s.sensorName.c_str(), iname.c_str());
                s.warnedUnknownInterface = true;
            }
        }

        if (m_imuApi)
            s.sensorCreated = m_imuApi->createSensor(s.primPath.c_str());
    }

    m_physicsSimulation = carb::getCachedInterface<omni::physics::IPhysicsSimulation>();
    if (m_physicsSimulation && m_preStepSub == omni::physics::kInvalidSubscriptionId)
    {
        m_preStepSub = m_physicsSimulation->subscribePhysicsOnStepEvents(
            true, 0, [this](float, const omni::physics::PhysicsStepContext&) { applyPendingGains(); });
    }
    if (!m_physicsSimulation || m_preStepSub == omni::physics::kInvalidSubscriptionId)
    {
        RCLCPP_WARN(getIsaacSimSystemLogger(),
                    "IsaacSimSystem: omni::physics::IPhysicsSimulation unavailable; command-mode drive gains "
                    "will not update on physics pre-step.");
    }

    return CallbackReturn::SUCCESS;
}

void IsaacSimSystem::releaseResources()
{
    for (auto& s : m_sensors)
    {
        if (!s.sensorCreated)
            continue;
        if (s.kind == SensorBinding::Kind::IMU && m_imuApi)
            m_imuApi->removeSensor(s.primPath.c_str());
        s.sensorCreated = false;
    }
    if (m_physicsSimulation && m_preStepSub != omni::physics::kInvalidSubscriptionId)
    {
        m_physicsSimulation->unsubscribePhysicsOnStepEvents(m_preStepSub);
        m_preStepSub = omni::physics::kInvalidSubscriptionId;
    }
    m_physicsSimulation = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_modeMutex);
        std::fill(m_activeMode.begin(), m_activeMode.end(), kNone);
        std::fill(m_appliedMode.begin(), m_appliedMode.end(), kNone);
        const double nan = std::numeric_limits<double>::quiet_NaN();
        std::fill(m_positionCmd.begin(), m_positionCmd.end(), nan);
        std::fill(m_velocityCmd.begin(), m_velocityCmd.end(), nan);
        std::fill(m_effortCmd.begin(), m_effortCmd.end(), nan);
        m_gainsDirty = false;
    }
    // TensorApi owns the views; drop our handles so on_configure can re-acquire
    // them after a stage close/reload.
    m_articulationView = nullptr;
    m_simView = nullptr;
}

CallbackReturn IsaacSimSystem::on_deactivate(const rclcpp_lifecycle::State&)
{
    releaseResources();
    return CallbackReturn::SUCCESS;
}

CallbackReturn IsaacSimSystem::on_cleanup(const rclcpp_lifecycle::State&)
{
    releaseResources();
    return CallbackReturn::SUCCESS;
}

CallbackReturn IsaacSimSystem::on_shutdown(const rclcpp_lifecycle::State&)
{
    releaseResources();
    return CallbackReturn::SUCCESS;
}

CallbackReturn IsaacSimSystem::on_error(const rclcpp_lifecycle::State&)
{
    // Roll back partial state from a failed transition (e.g. an IMU sensor created before a
    // later sensor failed in on_configure) so the component can be reconfigured cleanly.
    releaseResources();
    return CallbackReturn::SUCCESS;
}

std::vector<hardware_interface::StateInterface> IsaacSimSystem::export_state_interfaces()
{
    std::vector<hardware_interface::StateInterface> out;
    for (size_t i = 0; i < m_jointNames.size(); ++i)
    {
        out.emplace_back(m_jointNames[i], "position", &m_position[i]);
        out.emplace_back(m_jointNames[i], "velocity", &m_velocity[i]);
        out.emplace_back(m_jointNames[i], "effort", &m_effort[i]);
    }
    for (auto& s : m_sensors)
        for (size_t i = 0; i < s.interfaceNames.size(); ++i)
            out.emplace_back(s.sensorName, s.interfaceNames[i], &s.values[i]);
    return out;
}

std::vector<hardware_interface::CommandInterface> IsaacSimSystem::export_command_interfaces()
{
    std::vector<hardware_interface::CommandInterface> out;
    for (size_t i = 0; i < m_jointNames.size(); ++i)
    {
        if (m_cmdMask[i] & kBitPosition)
            out.emplace_back(m_jointNames[i], "position", &m_positionCmd[i]);
        if (m_cmdMask[i] & kBitVelocity)
            out.emplace_back(m_jointNames[i], "velocity", &m_velocityCmd[i]);
        if (m_cmdMask[i] & kBitEffort)
            out.emplace_back(m_jointNames[i], "effort", &m_effortCmd[i]);
    }
    return out;
}

return_type IsaacSimSystem::read(const rclcpp::Time&, const rclcpp::Duration&)
{
    CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::read");

    if (!m_articulationView || m_numDofs == 0)
    {
        return return_type::OK;
    }

    const size_t n = m_jointNames.size();
    omni::physics::tensors::TensorDesc desc;
    fillCpuFloat1D(desc, m_readBuf.data(), static_cast<int>(m_numDofs));

    {
        CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::read_positions");
        if (m_articulationView->getDofPositions(&desc))
        {
            for (size_t i = 0; i < n; ++i)
                m_position[i] = static_cast<double>(m_readBuf[m_dofIndex[i]]);
        }
        else if (!m_warnedReadFailure)
        {
            RCLCPP_WARN(getIsaacSimSystemLogger(),
                        "IsaacSimSystem: getDofPositions failed for '%s'; joint states will be stale.",
                        m_articulationPath.c_str());
            m_warnedReadFailure = true;
        }
    }
    {
        CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::read_velocities");
        if (m_articulationView->getDofVelocities(&desc))
        {
            for (size_t i = 0; i < n; ++i)
                m_velocity[i] = static_cast<double>(m_readBuf[m_dofIndex[i]]);
        }
    }
    {
        CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::read_efforts");
        if (m_articulationView->getDofProjectedJointForces(&desc))
        {
            for (size_t i = 0; i < n; ++i)
                m_effort[i] = static_cast<double>(m_readBuf[m_dofIndex[i]]);
        }
    }

    // Force-torque sensors share one articulation-wide wrench fetch, then each
    // slices its link's 6 components by interface name.
    bool ftOk = false;
    if (m_hasFtSensors)
    {
        CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::read_link_forces");
        omni::physics::tensors::TensorDesc fdesc;
        fillCpuFloat1D(fdesc, m_linkForceBuf.data(), static_cast<int>(m_linkForceBuf.size()));
        ftOk = m_articulationView->getLinkIncomingJointForce(&fdesc);
        if (!ftOk && !m_warnedFtReadFailure)
        {
            RCLCPP_WARN(getIsaacSimSystemLogger(),
                        "IsaacSimSystem: getLinkIncomingJointForce failed for '%s'; force-torque values stay 0 "
                        "(unsupported on the Newton backend).",
                        m_articulationPath.c_str());
            m_warnedFtReadFailure = true;
        }
    }

    for (auto& s : m_sensors)
    {
        if (s.kind == SensorBinding::Kind::ForceTorque)
        {
            if (!ftOk)
                continue;
            const float* w = &m_linkForceBuf[static_cast<size_t>(s.linkIndex) * 6];
            for (size_t i = 0; i < s.interfaceNames.size(); ++i)
            {
                int off;
                if (extractFtComponent(s.interfaceNames[i], off))
                    s.values[i] = static_cast<double>(w[off]);
            }
            continue;
        }
        if (s.kind != SensorBinding::Kind::IMU || !m_imuApi)
            continue;
        if (!s.sensorCreated && !(s.sensorCreated = m_imuApi->createSensor(s.primPath.c_str())))
            continue;
        // readGravity=true: sensor_msgs/Imu requires linear_acceleration to include
        // the gravity reaction force (a stationary IMU reads ~9.81 m/s^2 up).
        auto reading = m_imuApi->getSensorReading(s.primPath.c_str(), /*readGravity=*/true);
        if (!reading.isValid)
            continue;
        for (size_t i = 0; i < s.interfaceNames.size(); ++i)
        {
            double v = s.values[i];
            if (extractImuValue(s.interfaceNames[i], reading, v))
                s.values[i] = v;
        }
    }
    return return_type::OK;
}

return_type IsaacSimSystem::write(const rclcpp::Time&, const rclcpp::Duration&)
{
    CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::write");

    if (!m_articulationView || m_numDofs == 0)
    {
        return return_type::OK;
    }

    using View = omni::physics::tensors::IArticulationView;
    using Desc = omni::physics::tensors::TensorDesc;
    const size_t n = m_jointNames.size();

    std::vector<int8_t> activeMode;
    std::vector<double> positionCmd, velocityCmd, effortCmd;
    {
        std::lock_guard<std::mutex> lk(m_modeMutex);
        // Delay new commands until pre-step commits their drive gains.
        activeMode = m_appliedMode;
        positionCmd = m_positionCmd;
        velocityCmd = m_velocityCmd;
        effortCmd = m_effortCmd;
    }

    Desc desc;
    fillCpuFloat1D(desc, m_writeBuf.data(), static_cast<int>(m_numDofs));

    auto writeBuf = [&](bool (View::*get)(const Desc*) const, bool (View::*set)(const Desc*, const Desc*),
                        const std::function<bool(size_t, float&)>& valueFor)
    {
        if (!(m_articulationView->*get)(&desc))
        {
            if (!m_warnedWriteFailure)
            {
                RCLCPP_WARN(getIsaacSimSystemLogger(),
                            "IsaacSimSystem: reading current targets failed for '%s'; commands dropped this step.",
                            m_articulationPath.c_str());
                m_warnedWriteFailure = true;
            }
            return;
        }
        bool wrote = false;
        for (size_t i = 0; i < n; ++i)
        {
            float v;
            if (valueFor(i, v))
            {
                m_writeBuf[m_dofIndex[i]] = v;
                wrote = true;
            }
        }
        if (wrote && !(m_articulationView->*set)(&desc, /*indexTensor=*/nullptr) && !m_warnedWriteFailure)
        {
            RCLCPP_WARN(getIsaacSimSystemLogger(), "IsaacSimSystem: writing command targets failed for '%s'.",
                        m_articulationPath.c_str());
            m_warnedWriteFailure = true;
        }
    };

    {
        CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::write_position_targets");
        writeBuf(&View::getDofPositionTargets, &View::setDofPositionTargets,
                 [&](size_t i, float& v) -> bool
                 {
                     if (m_mimicLeader[i] >= 0) // follower tracks its leader's measured position
                     {
                         v = static_cast<float>(m_mimicMultiplier[i] * m_position[m_mimicLeader[i]] + m_mimicOffset[i]);
                         return true;
                     }
                     if (activeMode[i] != kPosition || !std::isfinite(positionCmd[i]))
                         return false;
                     v = static_cast<float>(positionCmd[i]);
                     return true;
                 });
    }
    {
        CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::write_velocity_targets");
        writeBuf(&View::getDofVelocityTargets, &View::setDofVelocityTargets,
                 [&](size_t i, float& v) -> bool
                 {
                     if (m_mimicLeader[i] >= 0) // position-tracked follower; damp velocity toward rest
                     {
                         v = 0.0f;
                         return true;
                     }
                     if (activeMode[i] == kVelocity)
                     {
                         if (!std::isfinite(velocityCmd[i]))
                             return false;
                         v = static_cast<float>(velocityCmd[i]);
                         return true;
                     }
                     if (activeMode[i] == kPosition) // damp toward rest
                     {
                         v = 0.0f;
                         return true;
                     }
                     return false;
                 });
    }
    {
        CARB_PROFILE_ZONE(0, "[ros2.control] IsaacSimSystem::write_effort");
        writeBuf(&View::getDofActuationForces, &View::setDofActuationForces,
                 [&](size_t i, float& v) -> bool
                 {
                     if (m_cmdMask[i] == 0)
                         return false; // not a joint we manage
                     if (activeMode[i] == kEffort)
                         v = std::isfinite(effortCmd[i]) ? static_cast<float>(effortCmd[i]) : 0.0f;
                     else
                         v = 0.0f;
                     return true;
                 });
    }
    return return_type::OK;
}

return_type IsaacSimSystem::prepare_command_mode_switch(const std::vector<std::string>& start,
                                                        const std::vector<std::string>& stop)
{
    (void)stop;
    // Reject simultaneous multi-interface (impedance) on one joint: only one
    // command interface per joint is supported.
    std::unordered_map<std::string, int> startsPerJoint;
    for (auto& s : start)
    {
        auto pos = s.rfind('/');
        if (pos != std::string::npos)
            startsPerJoint[s.substr(0, pos)]++;
    }
    for (auto& kv : startsPerJoint)
        if (kv.second > 1)
        {
            RCLCPP_ERROR(getIsaacSimSystemLogger(),
                         "IsaacSimSystem: joint '%s' would activate %d command interfaces at once; only one "
                         "command interface per joint is supported.",
                         kv.first.c_str(), kv.second);
            return return_type::ERROR;
        }
    return return_type::OK;
}

return_type IsaacSimSystem::perform_command_mode_switch(const std::vector<std::string>& start,
                                                        const std::vector<std::string>& stop)
{
    auto parse = [&](const std::string& s, size_t& jointIdx, int8_t& kind) -> bool
    {
        auto pos = s.rfind('/');
        if (pos == std::string::npos)
            return false;
        const std::string iface = s.substr(pos + 1);
        if (iface == "position")
            kind = kPosition;
        else if (iface == "velocity")
            kind = kVelocity;
        else if (iface == "effort")
            kind = kEffort;
        else
            return false;
        const std::string joint = s.substr(0, pos);
        for (size_t i = 0; i < m_jointNames.size(); ++i)
            if (m_jointNames[i] == joint)
            {
                jointIdx = i;
                return true;
            }
        return false;
    };

    size_t i = 0;
    int8_t kind = kNone;
    std::lock_guard<std::mutex> lk(m_modeMutex);
    for (auto& s : stop)
        if (parse(s, i, kind) && m_activeMode[i] == kind)
        {
            m_activeMode[i] = kNone;
            const double nan = std::numeric_limits<double>::quiet_NaN();
            m_positionCmd[i] = m_velocityCmd[i] = m_effortCmd[i] = nan;
        }
    for (auto& s : start)
        if (parse(s, i, kind))
            m_activeMode[i] = kind;
        else
            RCLCPP_WARN(
                getIsaacSimSystemLogger(), "IsaacSimSystem: ignoring unrecognized command interface '%s'", s.c_str());
    m_gainsDirty = true; // pre-step applyPendingGains() reconfigures the drive
    return return_type::OK;
}

void IsaacSimSystem::applyPendingGains()
{
    if (!m_articulationView)
        return;
    std::vector<int8_t> activeMode;
    {
        std::lock_guard<std::mutex> lk(m_modeMutex);
        if (!m_gainsDirty)
            return;
        activeMode = m_activeMode;
    }
    omni::physics::tensors::TensorDesc desc;
    fillCpuFloat1D(desc, m_writeBuf.data(), static_cast<int>(m_numDofs));
    bool success = m_articulationView->getDofStiffnesses(&desc);
    if (success)
    {
        for (size_t i = 0; i < m_jointNames.size(); ++i)
            m_writeBuf[m_dofIndex[i]] =
                (activeMode[i] == kVelocity || activeMode[i] == kEffort) ? 0.0f : m_authoredStiffness[i];
        success = m_articulationView->setDofStiffnesses(&desc, nullptr);
    }
    if (success)
        success = m_articulationView->getDofDampings(&desc);
    if (success)
    {
        for (size_t i = 0; i < m_jointNames.size(); ++i)
            m_writeBuf[m_dofIndex[i]] = (activeMode[i] == kEffort) ? 0.0f : m_authoredDamping[i];
        success = m_articulationView->setDofDampings(&desc, nullptr);
    }
    if (!success)
    {
        if (!m_warnedGainFailure)
        {
            RCLCPP_WARN(getIsaacSimSystemLogger(),
                        "IsaacSimSystem: applying drive gains failed for '%s'; retrying next pre-step.",
                        m_articulationPath.c_str());
            m_warnedGainFailure = true;
        }
        return;
    }
    {
        std::lock_guard<std::mutex> lk(m_modeMutex);
        // A concurrent switch leaves m_gainsDirty set for the next pre-step.
        if (m_activeMode == activeMode)
        {
            m_appliedMode = activeMode;
            m_gainsDirty = false;
            m_warnedGainFailure = false;
        }
    }
}

}
}
}
}

PLUGINLIB_EXPORT_CLASS(isaacsim::ros2::control::backend::IsaacSimSystem, hardware_interface::SystemInterface)
