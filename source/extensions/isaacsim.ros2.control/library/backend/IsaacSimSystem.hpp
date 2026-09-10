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
#include <hardware_interface/system_interface.hpp>
#include <hardware_interface/types/hardware_interface_return_values.hpp>
#include <omni/physics/simulation/IPhysicsSimulation.h>
#include <omni/physics/tensors/IArticulationMetatype.h>
#include <omni/physics/tensors/IArticulationView.h>
#include <omni/physics/tensors/ISimulationView.h>
#include <omni/physics/tensors/TensorApi.h>
#include <omni/physics/tensors/TensorDesc.h>
#include <rclcpp_lifecycle/state.hpp>

#include <mutex>
#include <string>
#include <vector>

namespace isaacsim
{
namespace sensors
{
namespace experimental
{
namespace physics
{
struct IImuSensor;
}
}
}
}

namespace isaacsim
{
namespace ros2
{
namespace control
{
namespace backend
{

class IsaacSimSystem : public hardware_interface::SystemInterface
{
public:
    ~IsaacSimSystem() override;

#if defined(ROS2_BACKEND_JAZZY)
    hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareComponentInterfaceParams&) override;
#else
    hardware_interface::CallbackReturn on_init(const hardware_interface::HardwareInfo&) override;
#endif
    hardware_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State&) override;
    hardware_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override;
    hardware_interface::CallbackReturn on_cleanup(const rclcpp_lifecycle::State&) override;
    hardware_interface::CallbackReturn on_shutdown(const rclcpp_lifecycle::State&) override;
    hardware_interface::CallbackReturn on_error(const rclcpp_lifecycle::State&) override;

    std::vector<hardware_interface::StateInterface> export_state_interfaces() override;
    std::vector<hardware_interface::CommandInterface> export_command_interfaces() override;

    hardware_interface::return_type read(const rclcpp::Time&, const rclcpp::Duration&) override;
    hardware_interface::return_type write(const rclcpp::Time&, const rclcpp::Duration&) override;

    hardware_interface::return_type prepare_command_mode_switch(const std::vector<std::string>& start,
                                                                const std::vector<std::string>& stop) override;
    hardware_interface::return_type perform_command_mode_switch(const std::vector<std::string>& start,
                                                                const std::vector<std::string>& stop) override;

private:
    enum CommandKind : int8_t
    {
        kNone = -1,
        kPosition = 0,
        kVelocity = 1,
        kEffort = 2,
    };

    // Bit flags for which command interfaces a joint declares (export side).
    enum CmdBit : uint8_t
    {
        kBitPosition = 1 << 0,
        kBitVelocity = 1 << 1,
        kBitEffort = 1 << 2,
    };

    std::string m_articulationPath;
    std::vector<std::string> m_jointNames;
    std::vector<double> m_position, m_velocity, m_effort;
    std::vector<double> m_positionCmd, m_velocityCmd, m_effortCmd;
    std::vector<uint8_t> m_cmdMask; // declared command interfaces (CmdBit bits)
    std::vector<int8_t> m_activeMode; // CommandKind currently claimed, kNone if unclaimed
    std::vector<int8_t> m_appliedMode; // mode whose drive gains have reached the physics backend
    std::vector<int32_t> m_dofIndex; // URDF joint i -> physics DOF index
    std::vector<float> m_authoredStiffness, m_authoredDamping; // USD drive gains, per joint

    // Mimic joints: follower's position target = multiplier * leader_measured_position + offset,
    // applied in write() so it holds on any physics backend (no reliance on PhysX mimic
    // constraints). Per joint; m_mimicLeader[i] == -1 if joint i is not a mimic follower.
    std::vector<int32_t> m_mimicLeader;
    std::vector<double> m_mimicMultiplier;
    std::vector<double> m_mimicOffset;
    bool m_warnedMimicNoDrive = false;

    // Drive gains can't be set from the post-step callback, so a mode switch only
    // marks them dirty; applyPendingGains() flushes them from the pre-step boundary.
    bool m_gainsDirty = false;
    // Guards commands, modes, and gain transitions.
    std::mutex m_modeMutex;
    omni::physics::IPhysicsSimulation* m_physicsSimulation = nullptr;
    omni::physics::SubscriptionId m_preStepSub = omni::physics::kInvalidSubscriptionId;
    void applyPendingGains();
    // Idempotent teardown shared by on_deactivate/on_cleanup/on_shutdown/on_error: removes
    // created sensors, unsubscribes the pre-step callback, and drops the tensor views.
    void releaseResources();

    omni::physics::tensors::TensorApi* m_tensorApi = nullptr;
    omni::physics::tensors::ISimulationView* m_simView = nullptr;
    omni::physics::tensors::IArticulationView* m_articulationView = nullptr;
    size_t m_numDofs = 0;
    std::vector<float> m_readBuf;
    std::vector<float> m_writeBuf;

    // Warn-once guards so a persistent tensor-API failure logs without spamming.
    bool m_warnedReadFailure = false;
    bool m_warnedWriteFailure = false;
    bool m_warnedGainFailure = false;

    // One SensorBinding per URDF <sensor> block. `values` backs the raw pointers
    // handed out by export_state_interfaces(), so it must not be reallocated
    // after on_init.
    struct SensorBinding
    {
        enum class Kind
        {
            Unsupported,
            IMU,
            ForceTorque
        };
        std::string sensorName;
        std::string primPath;
        std::string linkName; // ForceTorque only: target link (<param name="link">)
        int32_t linkIndex = -1; // ForceTorque only: link index in the wrench buffer
        Kind kind = Kind::Unsupported;
        std::vector<std::string> interfaceNames;
        std::vector<double> values;
        bool warnedUnknownInterface = false;
        // Sensor managers self-initialize on physics resume, so we retry
        // createSensor() lazily from read() until it sticks.
        bool sensorCreated = false;
    };
    std::vector<SensorBinding> m_sensors;

    isaacsim::sensors::experimental::physics::IImuSensor* m_imuApi = nullptr;

    // Force-torque sensors: one [maxLinks*6] wrench fetch per read(), then sliced
    // per sensor by linkIndex. Backend-dependent; warn once if unavailable.
    std::vector<float> m_linkForceBuf;
    uint32_t m_maxLinks = 0;
    bool m_hasFtSensors = false;
    bool m_warnedFtReadFailure = false;
};

}
}
}
}
