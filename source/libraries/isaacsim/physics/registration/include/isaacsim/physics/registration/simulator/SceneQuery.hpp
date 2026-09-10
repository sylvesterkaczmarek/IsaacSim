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

#include "../Types.hpp"

#include <functional>


namespace isaacsim
{
namespace physics
{
namespace registration
{

/**
 * @brief Identifies the scene objects associated with a scene-query hit.
 */
struct SceneQueryHitObject
{
    /** @brief The encoded USD path of the collision shape. */
    PathToken collision;

    /** @brief The encoded USD path of the rigid body. */
    PathToken rigidBody;

    /**
     * @brief The point-instancer prototype index.
     *
     * A value of @c 0xFFFFFFFF indicates that the hit object is not part of a point instancer.
     */
    uint32_t prototypeIndex;
};

/**
 * @brief Describes the location and surface properties of a scene-query hit.
 */
struct SceneQueryHitLocation : SceneQueryHitObject
{
    /** @brief The hit-surface normal in world space. */
    isaacsim::physics::registration::Float3 normal;

    /** @brief The hit position in world space, in scene length units. */
    isaacsim::physics::registration::Float3 position;

    /**
     * @brief The query travel distance to the hit, in scene length units.
     *
     * For a raycast, this is the distance along the ray. For a sweep, this is the distance traveled by the query shape
     * along the sweep direction.
     */
    float distance;

    /** @brief The hit face index when the collision shape is a triangle mesh. */
    uint32_t faceIndex;

    /** @brief The encoded USD path of the material assigned to the hit surface. */
    PathToken material;
};

/**
 * @brief Describes an overlap-query hit.
 */
struct OverlapHit : SceneQueryHitObject
{
};

/**
 * @brief Describes a raycast-query hit.
 */
struct RaycastHit : SceneQueryHitLocation
{
};

/**
 * @brief Describes a sweep-query hit.
 */
struct SweepHit : SceneQueryHitLocation
{
};

/**
 * @brief Callback invoked for each overlap-query hit.
 *
 * @param[in] hit The reported overlap hit.
 * @return @c true to continue traversal; @c false to stop traversal.
 *
 * @note The callback borrows @p hit. The reference remains valid only for the duration of the callback invocation and
 *       must not be retained.
 * @note Exceptions raised by the callback may propagate through the query operation.
 */
using OverlapHitReportFunction = std::function<bool(const OverlapHit& hit)>;

/**
 * @brief Callback invoked for each raycast-query hit.
 *
 * @param[in] hit The reported raycast hit.
 * @return @c true to continue traversal; @c false to stop traversal.
 *
 * @note The callback borrows @p hit. The reference remains valid only for the duration of the callback invocation and
 *       must not be retained.
 * @note Exceptions raised by the callback may propagate through the query operation.
 */
using RaycastHitReportFunction = std::function<bool(const RaycastHit& hit)>;

/**
 * @brief Callback invoked for each sweep-query hit.
 *
 * @param[in] hit The reported sweep hit.
 * @return @c true to continue traversal; @c false to stop traversal.
 *
 * @note The callback borrows @p hit. The reference remains valid only for the duration of the callback invocation and
 *       must not be retained.
 * @note Exceptions raised by the callback may propagate through the query operation.
 */
using SweepHitReportFunction = std::function<bool(const SweepHit& hit)>;

/**
 * @brief Raycasts against the physics scene and returns the closest hit.
 *
 * @param[in] origin The ray origin in world space, expressed in scene length units.
 * @param[in] unitDirection The normalized ray direction.
 * @param[in] distance The maximum ray distance, in scene length units. Must be nonnegative.
 * @param[out] hit The closest hit. The value is modified only when the function returns @c true.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 * @return @c true if a hit was found; otherwise, @c false.
 */
using RaycastClosestFunction = std::function<bool(const isaacsim::physics::registration::Float3& origin,
                                                  const isaacsim::physics::registration::Float3& unitDirection,
                                                  float distance,
                                                  RaycastHit& hit,
                                                  bool bothSidesEnabled)>;

/**
 * @brief Tests whether a ray hits any object in the physics scene.
 *
 * @param[in] origin The ray origin in world space, expressed in scene length units.
 * @param[in] unitDirection The normalized ray direction.
 * @param[in] distance The maximum ray distance, in scene length units. Must be nonnegative.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 * @return @c true if a hit was found; otherwise, @c false.
 */
using RaycastAnyFunction = std::function<bool(const isaacsim::physics::registration::Float3& origin,
                                              const isaacsim::physics::registration::Float3& unitDirection,
                                              float distance,
                                              bool bothSidesEnabled)>;

/**
 * @brief Raycasts against the physics scene and reports each hit through a callback.
 *
 * @param[in] origin The ray origin in world space, expressed in scene length units.
 * @param[in] unitDirection The normalized ray direction.
 * @param[in] distance The maximum ray distance, in scene length units. Must be nonnegative.
 * @param[in] reportFunction The callback to invoke for each hit.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 */
using RaycastAllFunction = std::function<void(const isaacsim::physics::registration::Float3& origin,
                                              const isaacsim::physics::registration::Float3& unitDirection,
                                              float distance,
                                              RaycastHitReportFunction reportFunction,
                                              bool bothSidesEnabled)>;

/**
 * @brief Sweeps a sphere through the physics scene and returns the closest hit.
 *
 * @param[in] radius The sphere radius, in scene length units. Must be nonnegative.
 * @param[in] origin The initial sphere-center position in world space, expressed in scene length units.
 * @param[in] unitDirection The normalized sweep direction.
 * @param[in] distance The maximum sweep distance, in scene length units. Must be nonnegative.
 * @param[out] hit The closest hit. The value is modified only when the function returns @c true.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 * @return @c true if a hit was found; otherwise, @c false.
 */
using SweepSphereClosestFunction = std::function<bool(float radius,
                                                      const isaacsim::physics::registration::Float3& origin,
                                                      const isaacsim::physics::registration::Float3& unitDirection,
                                                      float distance,
                                                      SweepHit& hit,
                                                      bool bothSidesEnabled)>;

/**
 * @brief Tests whether a swept sphere hits any object in the physics scene.
 *
 * @param[in] radius The sphere radius, in scene length units. Must be nonnegative.
 * @param[in] origin The initial sphere-center position in world space, expressed in scene length units.
 * @param[in] unitDirection The normalized sweep direction.
 * @param[in] distance The maximum sweep distance, in scene length units. Must be nonnegative.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 * @return @c true if a hit was found; otherwise, @c false.
 */
using SweepSphereAnyFunction = std::function<bool(float radius,
                                                  const isaacsim::physics::registration::Float3& origin,
                                                  const isaacsim::physics::registration::Float3& unitDirection,
                                                  float distance,
                                                  bool bothSidesEnabled)>;

/**
 * @brief Sweeps a sphere through the physics scene and reports each hit through a callback.
 *
 * @param[in] radius The sphere radius, in scene length units. Must be nonnegative.
 * @param[in] origin The initial sphere-center position in world space, expressed in scene length units.
 * @param[in] unitDirection The normalized sweep direction.
 * @param[in] distance The maximum sweep distance, in scene length units. Must be nonnegative.
 * @param[in] reportFunction The callback to invoke for each hit.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 */
using SweepSphereAllFunction = std::function<void(float radius,
                                                  const isaacsim::physics::registration::Float3& origin,
                                                  const isaacsim::physics::registration::Float3& unitDirection,
                                                  float distance,
                                                  SweepHitReportFunction reportFunction,
                                                  bool bothSidesEnabled)>;

/**
 * @brief Sweeps a box through the physics scene and returns the closest hit.
 *
 * @param[in] halfExtents The box half-extents, in scene length units. Each component must be nonnegative.
 * @param[in] position The initial box-center position in world space, expressed in scene length units.
 * @param[in] rotation The initial box orientation as a quaternion in x, y, z, w order.
 * @param[in] unitDirection The normalized sweep direction.
 * @param[in] distance The maximum sweep distance, in scene length units. Must be nonnegative.
 * @param[out] hit The closest hit. The value is modified only when the function returns @c true.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 * @return @c true if a hit was found; otherwise, @c false.
 */
using SweepBoxClosestFunction = std::function<bool(const isaacsim::physics::registration::Float3& halfExtents,
                                                   const isaacsim::physics::registration::Float3& position,
                                                   const isaacsim::physics::registration::Float4& rotation,
                                                   const isaacsim::physics::registration::Float3& unitDirection,
                                                   float distance,
                                                   SweepHit& hit,
                                                   bool bothSidesEnabled)>;

/**
 * @brief Sweeps a USD geometric primitive through the physics scene and returns the closest hit.
 *
 * @param[in] geometryPrimPath The encoded USD path of the geometric primitive to sweep.
 * @param[in] unitDirection The normalized sweep direction.
 * @param[in] distance The maximum sweep distance, in scene length units. Must be nonnegative.
 * @param[out] hit The closest hit. The value is modified only when the function returns @c true.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 * @return @c true if a hit was found; otherwise, @c false.
 */
using SweepShapeClosestFunction = std::function<bool(PathToken geometryPrimPath,
                                                     const isaacsim::physics::registration::Float3& unitDirection,
                                                     float distance,
                                                     SweepHit& hit,
                                                     bool bothSidesEnabled)>;

/**
 * @brief Tests whether a swept box hits any object in the physics scene.
 *
 * @param[in] halfExtents The box half-extents, in scene length units. Each component must be nonnegative.
 * @param[in] position The initial box-center position in world space, expressed in scene length units.
 * @param[in] rotation The initial box orientation as a quaternion in x, y, z, w order.
 * @param[in] unitDirection The normalized sweep direction.
 * @param[in] distance The maximum sweep distance, in scene length units. Must be nonnegative.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 * @return @c true if a hit was found; otherwise, @c false.
 */
using SweepBoxAnyFunction = std::function<bool(const isaacsim::physics::registration::Float3& halfExtents,
                                               const isaacsim::physics::registration::Float3& position,
                                               const isaacsim::physics::registration::Float4& rotation,
                                               const isaacsim::physics::registration::Float3& unitDirection,
                                               float distance,
                                               bool bothSidesEnabled)>;

/**
 * @brief Tests whether a swept USD geometric primitive hits any object in the physics scene.
 *
 * @param[in] geometryPrimPath The encoded USD path of the geometric primitive to sweep.
 * @param[in] unitDirection The normalized sweep direction.
 * @param[in] distance The maximum sweep distance, in scene length units. Must be nonnegative.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 * @return @c true if a hit was found; otherwise, @c false.
 */
using SweepShapeAnyFunction = std::function<bool(
    PathToken geometryPrimPath, const isaacsim::physics::registration::Float3& unitDirection, float distance, bool bothSidesEnabled)>;

/**
 * @brief Sweeps a box through the physics scene and reports each hit through a callback.
 *
 * @param[in] halfExtents The box half-extents, in scene length units. Each component must be nonnegative.
 * @param[in] position The initial box-center position in world space, expressed in scene length units.
 * @param[in] rotation The initial box orientation as a quaternion in x, y, z, w order.
 * @param[in] unitDirection The normalized sweep direction.
 * @param[in] distance The maximum sweep distance, in scene length units. Must be nonnegative.
 * @param[in] reportFunction The callback to invoke for each hit.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 */
using SweepBoxAllFunction = std::function<void(const isaacsim::physics::registration::Float3& halfExtents,
                                               const isaacsim::physics::registration::Float3& position,
                                               const isaacsim::physics::registration::Float4& rotation,
                                               const isaacsim::physics::registration::Float3& unitDirection,
                                               float distance,
                                               SweepHitReportFunction reportFunction,
                                               bool bothSidesEnabled)>;

/**
 * @brief Sweeps a USD geometric primitive through the physics scene and reports each hit through a callback.
 *
 * @param[in] geometryPrimPath The encoded USD path of the geometric primitive to sweep.
 * @param[in] unitDirection The normalized sweep direction.
 * @param[in] distance The maximum sweep distance, in scene length units. Must be nonnegative.
 * @param[in] reportFunction The callback to invoke for each hit.
 * @param[in] bothSidesEnabled Whether to test both sides of mesh triangles.
 */
using SweepShapeAllFunction = std::function<void(PathToken geometryPrimPath,
                                                 const isaacsim::physics::registration::Float3& unitDirection,
                                                 float distance,
                                                 SweepHitReportFunction reportFunction,
                                                 bool bothSidesEnabled)>;

/**
 * @brief Tests a sphere for overlaps and reports each overlap through a callback.
 *
 * @param[in] radius The sphere radius, in scene length units. Must be nonnegative.
 * @param[in] position The sphere-center position in world space, expressed in scene length units.
 * @param[in] reportFunction The callback to invoke for each overlap.
 * @return The number of overlaps found, or zero when no object overlaps the sphere.
 */
using OverlapSphereFunction = std::function<uint32_t(
    float radius, const isaacsim::physics::registration::Float3& position, OverlapHitReportFunction reportFunction)>;

/**
 * @brief Tests whether a sphere overlaps any object in the physics scene.
 *
 * @param[in] radius The sphere radius, in scene length units. Must be nonnegative.
 * @param[in] position The sphere-center position in world space, expressed in scene length units.
 * @return @c true if an overlap was found; otherwise, @c false.
 */
using OverlapSphereAnyFunction =
    std::function<bool(float radius, const isaacsim::physics::registration::Float3& position)>;

/**
 * @brief Tests a box for overlaps and reports each overlap through a callback.
 *
 * @param[in] halfExtents The box half-extents, in scene length units. Each component must be nonnegative.
 * @param[in] position The box-center position in world space, expressed in scene length units.
 * @param[in] rotation The box orientation as a quaternion in x, y, z, w order.
 * @param[in] reportFunction The callback to invoke for each overlap.
 * @return The number of overlaps found, or zero when no object overlaps the box.
 */
using OverlapBoxFunction = std::function<uint32_t(const isaacsim::physics::registration::Float3& halfExtents,
                                                  const isaacsim::physics::registration::Float3& position,
                                                  const isaacsim::physics::registration::Float4& rotation,
                                                  OverlapHitReportFunction reportFunction)>;

/**
 * @brief Tests whether a box overlaps any object in the physics scene.
 *
 * @param[in] halfExtents The box half-extents, in scene length units. Each component must be nonnegative.
 * @param[in] position The box-center position in world space, expressed in scene length units.
 * @param[in] rotation The box orientation as a quaternion in x, y, z, w order.
 * @return @c true if an overlap was found; otherwise, @c false.
 */
using OverlapBoxAnyFunction = std::function<bool(const isaacsim::physics::registration::Float3& halfExtents,
                                                 const isaacsim::physics::registration::Float3& position,
                                                 const isaacsim::physics::registration::Float4& rotation)>;

/**
 * @brief Tests a USD geometric primitive for overlaps and reports each overlap through a callback.
 *
 * @param[in] geometryPrimPath The encoded USD path of the geometric primitive to test.
 * @param[in] reportFunction The callback to invoke for each overlap.
 * @return The number of overlaps found, or zero when no object overlaps the primitive.
 */
using OverlapShapeFunction = std::function<uint32_t(PathToken geometryPrimPath, OverlapHitReportFunction reportFunction)>;

/**
 * @brief Tests whether a USD geometric primitive overlaps any object in the physics scene.
 *
 * @param[in] geometryPrimPath The encoded USD path of the geometric primitive to test.
 * @return @c true if an overlap was found; otherwise, @c false.
 */
using OverlapShapeAnyFunction = std::function<bool(PathToken geometryPrimPath)>;

/**
 * @brief Function table for physics scene-query operations.
 */
struct SceneQueryFunctions
{
    /** @brief Raycasts and returns the closest hit. */
    RaycastClosestFunction raycastClosest{};

    /** @brief Tests whether a ray hits any object. */
    RaycastAnyFunction raycastAny{};

    /** @brief Raycasts and reports each hit. */
    RaycastAllFunction raycastAll{};

    /** @brief Sweeps a sphere and returns the closest hit. */
    SweepSphereClosestFunction sweepSphereClosest{};

    /** @brief Tests whether a swept sphere hits any object. */
    SweepSphereAnyFunction sweepSphereAny{};

    /** @brief Sweeps a sphere and reports each hit. */
    SweepSphereAllFunction sweepSphereAll{};

    /** @brief Sweeps a box and returns the closest hit. */
    SweepBoxClosestFunction sweepBoxClosest{};

    /** @brief Tests whether a swept box hits any object. */
    SweepBoxAnyFunction sweepBoxAny{};

    /** @brief Sweeps a box and reports each hit. */
    SweepBoxAllFunction sweepBoxAll{};

    /** @brief Sweeps a USD geometric primitive and returns the closest hit. */
    SweepShapeClosestFunction sweepShapeClosest{};

    /** @brief Tests whether a swept USD geometric primitive hits any object. */
    SweepShapeAnyFunction sweepShapeAny{};

    /** @brief Sweeps a USD geometric primitive and reports each hit. */
    SweepShapeAllFunction sweepShapeAll{};

    /** @brief Tests a sphere and reports each overlap. */
    OverlapSphereFunction overlapSphere{};

    /** @brief Tests whether a sphere overlaps any object. */
    OverlapSphereAnyFunction overlapSphereAny{};

    /** @brief Tests a box and reports each overlap. */
    OverlapBoxFunction overlapBox{};

    /** @brief Tests whether a box overlaps any object. */
    OverlapBoxAnyFunction overlapBoxAny{};

    /** @brief Tests a USD geometric primitive and reports each overlap. */
    OverlapShapeFunction overlapShape{};

    /** @brief Tests whether a USD geometric primitive overlaps any object. */
    OverlapShapeAnyFunction overlapShapeAny{};
};

} // namespace registration
} // namespace physics
} // namespace isaacsim
