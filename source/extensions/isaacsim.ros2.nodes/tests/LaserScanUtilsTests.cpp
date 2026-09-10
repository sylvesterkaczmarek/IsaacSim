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

#include <carb/BindingsUtils.h>

#include <doctest/doctest.h>
#include <isaacsim/ros2/nodes/LaserScanUtils.hpp>

CARB_BINDINGS("isaacsim.ros2.nodes.tests")

namespace
{

using isaacsim::ros2::nodes::getLaserScanOutputAngleMax;
using isaacsim::ros2::nodes::getLaserScanOutputElementCount;
using isaacsim::ros2::nodes::getLaserScanOutputIndex;
using isaacsim::ros2::nodes::isLaserScanFovEndpointInclusive;

TEST_SUITE("isaacsim.ros2.nodes.tests.laser_scan_utils")
{
    TEST_CASE("laser_scan_utils: output count matches RTX clipped rotary ticks")
    {
        constexpr float resolution = 1.0f / 3.0f;
        CHECK(getLaserScanOutputElementCount(360.0f, resolution) == 1080);
        CHECK(getLaserScanOutputElementCount(270.0f, resolution) == 810);
        CHECK(getLaserScanOutputElementCount(125.34f, resolution) == 377);
        CHECK(getLaserScanOutputElementCount(0.1f, resolution) == 1);
        CHECK(getLaserScanOutputElementCount(0.0000001f, resolution) == 1);
        CHECK(getLaserScanOutputElementCount(0.0f, resolution) == 0);

        // Float32 USD values in the shipped Hesai AT360 profile must not
        // introduce a spurious extra bin just above an integral tick count.
        constexpr float at360Fov = 154.0200042725f - 28.6800003052f;
        CHECK(getLaserScanOutputElementCount(at360Fov, 0.06f) == 2089);

        // Kit truncates 32000 / 30 to 1066 full-rotation ticks before
        // calculating the clipped resolution.
        CHECK(getLaserScanOutputElementCount(360.0f, 0.3375f) == 1066);
    }

    TEST_CASE("laser_scan_utils: inclusive angle max follows the allocated discrete samples")
    {
        CHECK(getLaserScanOutputAngleMax(-20.0f, 1.0f, 40) == doctest::Approx(19.0f));
        CHECK(getLaserScanOutputAngleMax(-20.0f, 1.0f, 0) == doctest::Approx(-20.0f));
    }

    TEST_CASE("laser_scan_utils: GXO CW GMO endpoints map to all 810 LaserScan bins")
    {
        constexpr float resolution = 1.0f / 3.0f;
        constexpr float angleMin = -134.6666667f;
        constexpr float angleMax = 135.0f;
        constexpr float fov = 270.0f;
        constexpr size_t count = 810;
        size_t outputIndex = 0;

        CHECK(getLaserScanOutputIndex(-134.6666667f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 0);
        CHECK(getLaserScanOutputIndex(135.0f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 809);
        CHECK_FALSE(getLaserScanOutputIndex(-135.0f, angleMin, angleMax, fov, resolution, count, outputIndex));
    }

    TEST_CASE("laser_scan_utils: GXO CCW GMO endpoints map to all 810 LaserScan bins")
    {
        constexpr float resolution = 1.0f / 3.0f;
        constexpr float angleMin = 135.0f;
        constexpr float angleMax = 404.6666667f;
        constexpr float fov = 270.0f;
        constexpr size_t count = 810;
        size_t outputIndex = 0;

        CHECK(getLaserScanOutputIndex(135.0f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 0);
        // The inclusive angle_max is 404 2/3 degrees, represented by GMO as
        // its signed equivalent, +44 2/3 degrees.
        CHECK(getLaserScanOutputIndex(44.6666667f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 809);
        CHECK_FALSE(getLaserScanOutputIndex(45.0f, angleMin, angleMax, fov, resolution, count, outputIndex));
    }

    TEST_CASE("laser_scan_utils: signed GMO boundary stays contiguous")
    {
        constexpr float resolution = 1.0f;
        constexpr float angleMin = 170.0f;
        constexpr float angleMax = 289.0f;
        constexpr float fov = 120.0f;
        constexpr size_t count = 120;
        size_t outputIndex = 0;

        CHECK(getLaserScanOutputIndex(170.0f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 0);
        CHECK(getLaserScanOutputIndex(-180.0f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 10);
        CHECK(getLaserScanOutputIndex(-71.0f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 119);
        CHECK_FALSE(getLaserScanOutputIndex(-70.0f, angleMin, angleMax, fov, resolution, count, outputIndex));
    }

    TEST_CASE("laser_scan_utils: non-grid bearings retain containing-bin behavior")
    {
        constexpr float resolution = 1.0f;
        constexpr float angleMin = 0.0f;
        constexpr float angleMax = 10.0f;
        constexpr float fov = 10.0f;
        constexpr size_t count = 10;
        size_t outputIndex = 0;

        CHECK(getLaserScanOutputIndex(0.6f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 0);
        CHECK(getLaserScanOutputIndex(1.9999f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 1);

        constexpr float rotaryAngleMax = angleMax - resolution;
        CHECK(getLaserScanOutputIndex(1.9999f, angleMin, rotaryAngleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 2);

        CHECK(isLaserScanFovEndpointInclusive(angleMin, angleMax, fov, resolution, count));
        CHECK(getLaserScanOutputIndex(fov, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == count - 1);
        CHECK_FALSE(getLaserScanOutputIndex(fov, angleMin, rotaryAngleMax, fov, resolution, count, outputIndex));
        CHECK_FALSE(isLaserScanFovEndpointInclusive(angleMin, rotaryAngleMax, fov, resolution, count));
        CHECK_FALSE(isLaserScanFovEndpointInclusive(angleMin, angleMin, 0.0000001f, resolution, 1));
        CHECK_FALSE(getLaserScanOutputIndex(0.0000001f, angleMin, angleMin, 0.0000001f, resolution, 1, outputIndex));

        constexpr float narrowAngleMin = 179.9f;
        constexpr float narrowAngleMax = 180.0f;
        constexpr float narrowFov = 0.1f;
        constexpr float narrowResolution = narrowFov / 100.0f;
        CHECK(isLaserScanFovEndpointInclusive(narrowAngleMin, narrowAngleMax, narrowFov, narrowResolution, 100));
        CHECK(getLaserScanOutputIndex(
            narrowAngleMax, narrowAngleMin, narrowAngleMax, narrowFov, narrowResolution, 100, outputIndex));
        CHECK(outputIndex == 99);

        CHECK(getLaserScanOutputIndex(-180.0f, -180.0f, 180.0f, 360.0f, resolution, 360, outputIndex));
        CHECK(outputIndex == 0);
        CHECK(getLaserScanOutputIndex(180.0f, -180.0f, 180.0f, 360.0f, resolution, 360, outputIndex));
        CHECK(outputIndex == 359);
    }

    TEST_CASE("laser_scan_utils: non-integral field of view retains its final sampled tick")
    {
        constexpr float resolution = 1.0f / 3.0f;
        constexpr float angleMin = -64.0133333f;
        constexpr float angleMax = 61.32f;
        constexpr float fov = 125.34f;
        constexpr size_t count = 377;
        size_t outputIndex = 0;

        CHECK(getLaserScanOutputIndex(61.32f, angleMin, angleMax, fov, resolution, count, outputIndex));
        CHECK(outputIndex == 376);
        CHECK_FALSE(getLaserScanOutputIndex(61.3266667f, angleMin, angleMax, fov, resolution, count, outputIndex));
    }
}

} // namespace
