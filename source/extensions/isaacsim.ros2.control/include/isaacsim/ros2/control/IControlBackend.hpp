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

#include <carb/Defines.h>

#include <cstdint>

namespace omni
{
namespace physics
{
namespace tensors
{
struct TensorApi;
}
}
}

namespace isaacsim
{
namespace ros2
{
namespace control
{
namespace details
{

constexpr std::uint32_t kControlBackendAbiVersion = 1;

class IControlBackend
{
public:
    virtual bool initialize(omni::physics::tensors::TensorApi* tensorApi) noexcept = 0;
    virtual int createControllerManager(const char* articulationPath,
                                        const char* urdfXml,
                                        const char* controllerYamlPath,
                                        const char* namespaceName,
                                        bool publishRobotDescription,
                                        bool useSimTime) noexcept = 0;
    virtual void destroyControllerManager(const char* articulationPath) noexcept = 0;
    virtual void destroyAllControllerManagers() noexcept = 0;
    virtual void onPhysicsStep(double simulationTime, double stepSize) noexcept = 0;
    virtual void setProfilingEnabled(bool enabled) noexcept = 0;
    virtual void resetProfiling() noexcept = 0;
    virtual const char* getProfilingJson() noexcept = 0;
    virtual void release() noexcept = 0;

protected:
    ~IControlBackend() = default;
};

using CreateControlBackend = IControlBackend*(CARB_ABI*)(std::uint32_t requestedAbi) noexcept;

}
}
}
}
