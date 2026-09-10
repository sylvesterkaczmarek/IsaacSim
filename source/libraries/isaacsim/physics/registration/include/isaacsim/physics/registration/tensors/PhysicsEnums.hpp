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

#include <cstdint>

namespace isaacsim
{
namespace physics
{
namespace tensors
{

/**
 * @brief Types of physics objects represented by tensor views.
 */
enum class ObjectType : int32_t
{
    /** @brief An invalid or unspecified physics object. */
    eInvalid = 0,

    /** @brief A rigid body. */
    eRigidBody,

    /** @brief An articulation. */
    eArticulation,

    /** @brief A non-root articulation link. */
    eArticulationLink,

    /** @brief The root link of an articulation. */
    eArticulationRootLink,

    /** @brief An articulation joint. */
    eArticulationJoint,
};

/**
 * @brief Types of joints represented by tensor views.
 */
enum class JointType : int32_t
{
    /** @brief An invalid or unspecified joint. */
    eInvalid = 0,

    /** @brief A joint with no relative degrees of freedom. */
    eFixed,

    /** @brief A joint that rotates about one axis. */
    eRevolute,

    /** @brief A joint that translates along one axis. */
    ePrismatic,

    /** @brief A joint that rotates about three axes. */
    eSpherical,
};

/**
 * @brief Types of joint degrees of freedom.
 */
enum class DofType : int32_t
{
    /** @brief An invalid or unspecified degree of freedom. */
    eInvalid = 0,

    /** @brief A rotational degree of freedom. */
    eRotation,

    /** @brief A translational degree of freedom. */
    eTranslation,
};

/**
 * @brief Motion constraints for a joint degree of freedom.
 */
enum class DofMotion : int32_t
{
    /** @brief An invalid or unspecified motion constraint. */
    eInvalid = 0,

    /** @brief Motion is unconstrained. */
    eFree,

    /** @brief Motion is constrained to a finite range. */
    eLimited,

    /** @brief Motion is disabled. */
    eLocked,
};

/**
 * @brief Drive modes for a joint degree of freedom.
 */
enum class DofDriveType : int32_t
{
    /** @brief No drive is applied. */
    eNone = 0,

    /** @brief The drive output is interpreted as force or torque. */
    eForce,

    /** @brief The drive output is interpreted as linear or angular acceleration. */
    eAcceleration,
};

} // namespace tensors
} // namespace physics
} // namespace isaacsim
