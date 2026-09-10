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

#pragma once

#include <carb/Interface.h>

#include <cstdint>

namespace isaacsim
{
namespace ros2
{
namespace control
{

/// Carbonite plugin interface for the ROS 2 Control extension.
/// Other extensions retrieve this via carb::getCachedInterface<Ros2Control>().
struct Ros2Control
{
    CARB_PLUGIN_INTERFACE("isaacsim::ros2::control::Ros2Control", 0, 6);

    /// Configure a CM for the given articulation. A second call with the same
    /// articulationPath returns an error until the first CM is torn down.
    /// urdfXml: pre-synthesized URDF (live-exported by the Python OG node).
    /// Returns 0 on success, non-zero on failure.
    int (*setupCm)(const char* articulationPath,
                   const char* urdfXml,
                   const char* controllerYamlPath,
                   const char* nsName,
                   bool publishRobotDescription,
                   bool useSimTime);

    /// Tear down the CM associated with articulationPath. No-op if not registered.
    void (*teardownCm)(const char* articulationPath);

    /// Tear down all CMs (called on timeline Stop and available for explicit cleanup).
    void (*teardownAllCms)();

    /// Returns true if extension fully initialized and shared rcl_context is valid.
    bool (*isReady)();

    /// Enable or disable low-level ros2_control timing accumulation.
    /// Disabled by default so normal use only pays Carbonite profiler-zone cost.
    void (*setProfilingEnabled)(bool enabled);

    /// Clear accumulated timing samples.
    void (*resetProfiling)();

    /// Return accumulated timing data as a JSON object. Pointer is valid until
    /// the next call on the same thread.
    const char* (*getProfilingJson)();
};

}
}
}
