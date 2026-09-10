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

#include "WarpCompat.hpp"

namespace isaacsim
{
namespace physics
{
namespace newton
{
namespace tensors
{
namespace details
{

WP_CUDA_CALLABLE inline wp::vec3 makeVec3(float x, float y, float z)
{
    wp::vec3 result;
    result[0] = x;
    result[1] = y;
    result[2] = z;
    return result;
}

WP_CUDA_CALLABLE inline wp::vec3 addVec3(const wp::vec3& a, const wp::vec3& b)
{
    return makeVec3(a[0] + b[0], a[1] + b[1], a[2] + b[2]);
}

WP_CUDA_CALLABLE inline wp::vec3 subtractVec3(const wp::vec3& a, const wp::vec3& b)
{
    return makeVec3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

WP_CUDA_CALLABLE inline wp::vec3 crossVec3(const wp::vec3& a, const wp::vec3& b)
{
    return makeVec3(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]);
}

WP_CUDA_CALLABLE inline wp::vec3 rotateVec3(const wp::quat& rotation, const wp::vec3& vector)
{
    const wp::vec3 imaginary = makeVec3(rotation[0], rotation[1], rotation[2]);
    const wp::vec3 twiceCross = makeVec3(2.0f * (imaginary[1] * vector[2] - imaginary[2] * vector[1]),
                                         2.0f * (imaginary[2] * vector[0] - imaginary[0] * vector[2]),
                                         2.0f * (imaginary[0] * vector[1] - imaginary[1] * vector[0]));
    return addVec3(
        vector, addVec3(makeVec3(rotation[3] * twiceCross[0], rotation[3] * twiceCross[1], rotation[3] * twiceCross[2]),
                        crossVec3(imaginary, twiceCross)));
}

WP_CUDA_CALLABLE inline void computeWorldWrench(const float* force,
                                                const float* torque,
                                                const float* position,
                                                const wp::transform& bodyTransform,
                                                const wp::vec3& bodyCenterOfMass,
                                                bool isGlobal,
                                                bool hasForce,
                                                bool hasTorque,
                                                bool hasPosition,
                                                wp::vec3& forceWorld,
                                                wp::vec3& torqueWorld)
{
    forceWorld = makeVec3(0.0f, 0.0f, 0.0f);
    torqueWorld = makeVec3(0.0f, 0.0f, 0.0f);

    if (hasForce)
    {
        const wp::vec3 inputForce = makeVec3(force[0], force[1], force[2]);
        forceWorld = isGlobal ? inputForce : rotateVec3(bodyTransform.q, inputForce);

        if (hasPosition)
        {
            const wp::vec3 inputPosition = makeVec3(position[0], position[1], position[2]);
            const wp::vec3 positionWorld =
                isGlobal ? inputPosition : addVec3(bodyTransform.p, rotateVec3(bodyTransform.q, inputPosition));
            const wp::vec3 centerOfMassWorld = addVec3(bodyTransform.p, rotateVec3(bodyTransform.q, bodyCenterOfMass));
            torqueWorld = crossVec3(subtractVec3(positionWorld, centerOfMassWorld), forceWorld);
        }
    }

    if (hasTorque)
    {
        const wp::vec3 inputTorque = makeVec3(torque[0], torque[1], torque[2]);
        torqueWorld = addVec3(torqueWorld, isGlobal ? inputTorque : rotateVec3(bodyTransform.q, inputTorque));
    }
}

} // namespace details
} // namespace tensors
} // namespace newton
} // namespace physics
} // namespace isaacsim
