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

#include <cmath>
#include <cstddef>

namespace isaacsim
{
namespace ros2
{
namespace nodes
{

static constexpr float kFullCircleDegrees = 360.0f;
static constexpr float kLaserScanTickRoundingTolerance = 1e-3f;

/// Return the number of sampled azimuth ticks in a LaserScan interval.
///
/// RTX rotary lidar first truncates firing-rate / scan-rate to the number of
/// full-rotation ticks, then clips with ceil(FOV / (360 / fullTicks)). Recover
/// that full-tick count from the beam resolution. In particular, a
/// non-integral or sub-resolution valid arc still contains the final (or only)
/// tick.
inline size_t getLaserScanOutputElementCount(float horizontalFov, float horizontalResolution)
{
    if (horizontalFov <= 0.0f || horizontalResolution <= 0.0f)
    {
        return 0;
    }
    const float rawFullRotationTicks = kFullCircleDegrees / horizontalResolution;
    const float nearestFullRotationTicks = std::round(rawFullRotationTicks);
    const size_t fullRotationTicks =
        static_cast<size_t>(std::abs(rawFullRotationTicks - nearestFullRotationTicks) <= kLaserScanTickRoundingTolerance ?
                                nearestFullRotationTicks :
                                std::floor(rawFullRotationTicks));
    if (fullRotationTicks == 0)
    {
        return 0;
    }
    if (horizontalFov >= kFullCircleDegrees - kLaserScanTickRoundingTolerance)
    {
        return fullRotationTicks;
    }
    const float clippedResolution = kFullCircleDegrees / static_cast<float>(fullRotationTicks);
    const float clippedTicks = horizontalFov / clippedResolution;
    const float nearestClippedTicks = std::round(clippedTicks);
    const size_t clippedTickCount = static_cast<size_t>(
        std::abs(clippedTicks - nearestClippedTicks) <= kLaserScanTickRoundingTolerance ? nearestClippedTicks :
                                                                                          std::ceil(clippedTicks));
    return clippedTickCount == 0 ? 1 : clippedTickCount;
}

/// Return the inclusive angle of the final discrete LaserScan sample.
inline float getLaserScanOutputAngleMax(float angleMin, float horizontalResolution, size_t numOutputElements)
{
    return numOutputElements == 0 ? angleMin :
                                    angleMin + static_cast<float>(numOutputElements - 1) * horizontalResolution;
}

/// Return whether the supplied interval includes the authored FOV endpoint.
///
/// Solid-state metadata uses its maximum emitter bearing as both angleMax and
/// an inclusive sample. Rotary metadata instead uses the final discrete tick,
/// one resolution interval before the exclusive authored valid-end boundary.
inline bool isLaserScanFovEndpointInclusive(
    float angleMin, float angleMax, float horizontalFov, float horizontalResolution, size_t numOutputElements)
{
    if (horizontalResolution <= 0.0f || horizontalFov <= 0.0f || numOutputElements == 0)
    {
        return false;
    }
    const float intervalSpan = angleMax - angleMin;
    const float sampledSpan = static_cast<float>(numOutputElements - 1) * horizontalResolution;
    return std::abs(intervalSpan - horizontalFov) < std::abs(intervalSpan - sampledSpan);
}

/// Map a signed GMO azimuth into an ascending LaserScan interval.
///
/// LaserScan intervals may extend beyond +180 degrees when a partial scan
/// crosses the signed GMO boundary. Taking the positive modulo relative to
/// angleMin keeps both sides of that boundary contiguous in the output.
/// angleMax distinguishes an inclusive solid-state maximum emitter from a
/// rotary interval whose authored valid-end/FOV boundary remains exclusive.
inline bool getLaserScanOutputIndex(float azimuth,
                                    float angleMin,
                                    float angleMax,
                                    float horizontalFov,
                                    float horizontalResolution,
                                    size_t numOutputElements,
                                    size_t& outputIndex)
{
    if (horizontalFov <= 0.0f || horizontalResolution <= 0.0f || numOutputElements == 0)
    {
        return false;
    }

    const float endpointTolerance = horizontalResolution * kLaserScanTickRoundingTolerance;
    const float unwrappedRelativeAzimuth = azimuth - angleMin;
    const bool includeFovEndpoint =
        isLaserScanFovEndpointInclusive(angleMin, angleMax, horizontalFov, horizontalResolution, numOutputElements);
    const float unwrappedEndpoint = angleMax - angleMin;
    if (includeFovEndpoint && std::abs(unwrappedRelativeAzimuth - unwrappedEndpoint) <= endpointTolerance)
    {
        outputIndex = numOutputElements - 1;
        return true;
    }
    if (includeFovEndpoint && std::abs(unwrappedRelativeAzimuth) <= endpointTolerance)
    {
        outputIndex = 0;
        return true;
    }

    float relativeAzimuth = std::fmod(unwrappedRelativeAzimuth, kFullCircleDegrees);
    if (relativeAzimuth < 0.0f)
    {
        relativeAzimuth += kFullCircleDegrees;
    }

    float wrappedEndpoint = std::fmod(unwrappedEndpoint, kFullCircleDegrees);
    if (wrappedEndpoint < 0.0f)
    {
        wrappedEndpoint += kFullCircleDegrees;
    }
    if (includeFovEndpoint && std::abs(relativeAzimuth - wrappedEndpoint) <= endpointTolerance)
    {
        outputIndex = numOutputElements - 1;
        return true;
    }

    // A rotary authored valid-end/FOV boundary is exclusive of an additional
    // tick. Discard it and points outside the partial scan instead of
    // collapsing them into the final bin.
    if (relativeAzimuth >= horizontalFov)
    {
        return false;
    }

    // Solid-state sensors can have non-uniform bearings and retain the
    // established containing-bin (truncation) behavior.
    const float rawOutputIndex = relativeAzimuth / horizontalResolution;
    if (includeFovEndpoint)
    {
        outputIndex = static_cast<size_t>(std::floor(rawOutputIndex));
        return outputIndex < numOutputElements;
    }

    // Rotary GMO bearings are generated from a discrete tick grid, but
    // coordinate conversion and atan2 can put a value a few ulps below its
    // ideal integer index. Snap values close to that grid.
    const float nearestOutputIndex = std::round(rawOutputIndex);
    outputIndex = static_cast<size_t>(std::abs(rawOutputIndex - nearestOutputIndex) <= kLaserScanTickRoundingTolerance ?
                                          nearestOutputIndex :
                                          std::floor(rawOutputIndex));
    if (outputIndex >= numOutputElements)
    {
        return false;
    }
    return true;
}

} // namespace nodes
} // namespace ros2
} // namespace isaacsim
