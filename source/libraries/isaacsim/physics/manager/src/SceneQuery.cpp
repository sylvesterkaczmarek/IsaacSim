// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <isaacsim/physics/manager/PhysicsSceneQuery.hpp>
// clang-format on

#include "SimulationSnapshot.hpp"

#include <isaacsim/physics/registration/Physics.hpp>


namespace isaacsim
{
namespace physics
{
namespace manager
{

using namespace registration;

// Raycast closest
bool raycastClosest(const isaacsim::physics::registration::Float3& origin,
                    const isaacsim::physics::registration::Float3& unitDirection,
                    float distance,
                    RaycastHit& hit,
                    bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    bool foundHit = false;
    float closestDistance = distance;

    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.raycastClosest)
        {
            RaycastHit currentHit;
            if (simulation.second.simulation.sceneQueryFunctions.raycastClosest(
                    origin, unitDirection, closestDistance, currentHit, bothSides))
            {
                if (!foundHit || currentHit.distance < closestDistance)
                {
                    hit = currentHit;
                    closestDistance = currentHit.distance;
                    foundHit = true;
                }
            }
        }
    }
    return foundHit;
}

// Raycast any
bool raycastAny(const isaacsim::physics::registration::Float3& origin,
                const isaacsim::physics::registration::Float3& unitDirection,
                float distance,
                bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.raycastAny)
        {
            if (simulation.second.simulation.sceneQueryFunctions.raycastAny(origin, unitDirection, distance, bothSides))
            {
                return true;
            }
        }
    }
    return false;
}

// Raycast all
void raycastAll(const isaacsim::physics::registration::Float3& origin,
                const isaacsim::physics::registration::Float3& unitDirection,
                float distance,
                RaycastHitReportFunction reportFunction,
                bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.raycastAll)
        {
            simulation.second.simulation.sceneQueryFunctions.raycastAll(
                origin, unitDirection, distance, reportFunction, bothSides);
        }
    }
}

// Sweep sphere closest
bool sweepSphereClosest(float radius,
                        const isaacsim::physics::registration::Float3& origin,
                        const isaacsim::physics::registration::Float3& unitDirection,
                        float distance,
                        SweepHit& hit,
                        bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    bool foundHit = false;
    float closestDistance = distance;

    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.sweepSphereClosest)
        {
            SweepHit currentHit;
            if (simulation.second.simulation.sceneQueryFunctions.sweepSphereClosest(
                    radius, origin, unitDirection, closestDistance, currentHit, bothSides))
            {
                if (!foundHit || currentHit.distance < closestDistance)
                {
                    hit = currentHit;
                    closestDistance = currentHit.distance;
                    foundHit = true;
                }
            }
        }
    }
    return foundHit;
}

// Sweep sphere any
bool sweepSphereAny(float radius,
                    const isaacsim::physics::registration::Float3& origin,
                    const isaacsim::physics::registration::Float3& unitDirection,
                    float distance,
                    bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.sweepSphereAny)
        {
            if (simulation.second.simulation.sceneQueryFunctions.sweepSphereAny(
                    radius, origin, unitDirection, distance, bothSides))
            {
                return true;
            }
        }
    }
    return false;
}

// Sweep sphere all
void sweepSphereAll(float radius,
                    const isaacsim::physics::registration::Float3& origin,
                    const isaacsim::physics::registration::Float3& unitDirection,
                    float distance,
                    SweepHitReportFunction reportFunction,
                    bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.sweepSphereAll)
        {
            simulation.second.simulation.sceneQueryFunctions.sweepSphereAll(
                radius, origin, unitDirection, distance, reportFunction, bothSides);
        }
    }
}

// Overlap test of a sphere against objects in the physics scene
uint32_t overlapSphere(float radius,
                       const isaacsim::physics::registration::Float3& position,
                       OverlapHitReportFunction reportFunction)
{
    uint32_t totalOverlaps = 0;
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.overlapSphere)
        {
            totalOverlaps +=
                simulation.second.simulation.sceneQueryFunctions.overlapSphere(radius, position, reportFunction);
        }
    }
    return totalOverlaps;
}

// Overlap test of a sphere against objects in the physics scene, reports only boolean
bool overlapSphereAny(float radius, const isaacsim::physics::registration::Float3& position)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.overlapSphereAny)
        {
            if (simulation.second.simulation.sceneQueryFunctions.overlapSphereAny(radius, position))
            {
                return true;
            }
        }
    }
    return false;
}

// Overlap test of a box against objects in the physics scene
uint32_t overlapBox(const isaacsim::physics::registration::Float3& halfExtent,
                    const isaacsim::physics::registration::Float3& position,
                    const isaacsim::physics::registration::Float4& rotation,
                    OverlapHitReportFunction reportFunction)
{
    uint32_t totalOverlaps = 0;
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.overlapBox)
        {
            totalOverlaps += simulation.second.simulation.sceneQueryFunctions.overlapBox(
                halfExtent, position, rotation, reportFunction);
        }
    }
    return totalOverlaps;
}

// Overlap test of a box against objects in the physics scene, reports only boolean
bool overlapBoxAny(const isaacsim::physics::registration::Float3& halfExtent,
                   const isaacsim::physics::registration::Float3& position,
                   const isaacsim::physics::registration::Float4& rotation)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.overlapBoxAny)
        {
            if (simulation.second.simulation.sceneQueryFunctions.overlapBoxAny(halfExtent, position, rotation))
            {
                return true;
            }
        }
    }
    return false;
}

// Overlap test of a UsdGeomGPrim against objects in the physics scene
uint32_t overlapShape(PathToken geometryPrimitivePath, OverlapHitReportFunction reportFunction)
{
    uint32_t totalOverlaps = 0;
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.overlapShape)
        {
            totalOverlaps +=
                simulation.second.simulation.sceneQueryFunctions.overlapShape(geometryPrimitivePath, reportFunction);
        }
    }
    return totalOverlaps;
}

// Overlap test of a mesh against objects in the physics scene, reports only boolean
bool overlapShapeAny(PathToken geometryPrimitivePath)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.overlapShapeAny)
        {
            if (simulation.second.simulation.sceneQueryFunctions.overlapShapeAny(geometryPrimitivePath))
            {
                return true;
            }
        }
    }
    return false;
}

// Sweep test of a box against all objects in the physics scene, returning the closest hit found.
bool sweepBoxClosest(const isaacsim::physics::registration::Float3& halfExtent,
                     const isaacsim::physics::registration::Float3& position,
                     const isaacsim::physics::registration::Float4& rotation,
                     const isaacsim::physics::registration::Float3& unitDirection,
                     float distance,
                     SweepHit& hit,
                     bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    bool foundHit = false;
    float closestDistance = distance;

    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.sweepBoxClosest)
        {
            SweepHit currentHit;
            if (simulation.second.simulation.sceneQueryFunctions.sweepBoxClosest(
                    halfExtent, position, rotation, unitDirection, closestDistance, currentHit, bothSides))
            {
                if (!foundHit || currentHit.distance < closestDistance)
                {
                    hit = currentHit;
                    closestDistance = currentHit.distance;
                    foundHit = true;
                }
            }
        }
    }
    return foundHit;
}

// Sweep test of a UsdGeomGPrim against all objects in the physics scene, returning the closest hit found.
bool sweepShapeClosest(PathToken geometryPrimitivePath,
                       const isaacsim::physics::registration::Float3& unitDirection,
                       float distance,
                       SweepHit& hit,
                       bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    bool foundHit = false;
    float closestDistance = distance;

    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.sweepShapeClosest)
        {
            SweepHit currentHit;
            if (simulation.second.simulation.sceneQueryFunctions.sweepShapeClosest(
                    geometryPrimitivePath, unitDirection, closestDistance, currentHit, bothSides))
            {
                if (!foundHit || currentHit.distance < closestDistance)
                {
                    hit = currentHit;
                    closestDistance = currentHit.distance;
                    foundHit = true;
                }
            }
        }
    }
    return foundHit;
}

// Sweep test of a box against all objects in the physics scene, returning whether any hit was found.
bool sweepBoxAny(const isaacsim::physics::registration::Float3& halfExtent,
                 const isaacsim::physics::registration::Float3& position,
                 const isaacsim::physics::registration::Float4& rotation,
                 const isaacsim::physics::registration::Float3& unitDirection,
                 float distance,
                 bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.sweepBoxAny)
        {
            if (simulation.second.simulation.sceneQueryFunctions.sweepBoxAny(
                    halfExtent, position, rotation, unitDirection, distance, bothSides))
            {
                return true;
            }
        }
    }
    return false;
}

// Sweep test of a UsdGeomGPrim against all objects in the physics scene, returning whether any hit was found.
bool sweepShapeAny(PathToken geometryPrimitivePath,
                   const isaacsim::physics::registration::Float3& unitDirection,
                   float distance,
                   bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.sweepShapeAny)
        {
            if (simulation.second.simulation.sceneQueryFunctions.sweepShapeAny(
                    geometryPrimitivePath, unitDirection, distance, bothSides))
            {
                return true;
            }
        }
    }
    return false;
}

// Sweep test of a box against all objects in the physics scene, returning all the hits found.
void sweepBoxAll(const isaacsim::physics::registration::Float3& halfExtent,
                 const isaacsim::physics::registration::Float3& position,
                 const isaacsim::physics::registration::Float4& rotation,
                 const isaacsim::physics::registration::Float3& unitDirection,
                 float distance,
                 SweepHitReportFunction reportFunction,
                 bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.sweepBoxAll)
        {
            simulation.second.simulation.sceneQueryFunctions.sweepBoxAll(
                halfExtent, position, rotation, unitDirection, distance, reportFunction, bothSides);
        }
    }
}

// Sweep test of a UsdGeomGPrim against all objects in the physics scene, returning all the hits found.
void sweepShapeAll(PathToken geometryPrimitivePath,
                   const isaacsim::physics::registration::Float3& unitDirection,
                   float distance,
                   SweepHitReportFunction reportFunction,
                   bool bothSides)
{
    const details::SimulationSnapshots simulations = details::getSimulationSnapshots();
    for (const auto& simulation : simulations)
    {
        if (simulation.second.isActive && simulation.second.simulation.sceneQueryFunctions.sweepShapeAll)
        {
            simulation.second.simulation.sceneQueryFunctions.sweepShapeAll(
                geometryPrimitivePath, unitDirection, distance, reportFunction, bothSides);
        }
    }
}

} // namespace manager
} // namespace physics
} // namespace isaacsim
