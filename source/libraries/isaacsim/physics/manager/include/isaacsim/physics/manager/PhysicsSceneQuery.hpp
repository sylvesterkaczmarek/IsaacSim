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

#include <isaacsim/physics/manager/Export.h>
#include <isaacsim/physics/registration/Types.hpp>
#include <isaacsim/physics/registration/simulator/SceneQuery.hpp>

namespace isaacsim
{
namespace physics
{
namespace manager
{

/**
 * @brief Finds the closest raycast hit across all active physics backends.
 *
 * @param[in] origin Ray origin in world coordinates.
 * @param[in] unitDirection Normalized ray direction.
 * @param[in] distance Maximum ray distance in scene length units. Must be nonnegative.
 * @param[out] hit Closest hit. The value is modified only when the function returns `true`.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 * @return `true` if the ray hit an object; otherwise, `false`.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API bool raycastClosest(const isaacsim::physics::registration::Float3& origin,
                                                 const isaacsim::physics::registration::Float3& unitDirection,
                                                 float distance,
                                                 registration::RaycastHit& hit,
                                                 bool bothSides);

/**
 * @brief Tests whether a ray hits any object in an active physics backend.
 *
 * @param[in] origin Ray origin in world coordinates.
 * @param[in] unitDirection Normalized ray direction.
 * @param[in] distance Maximum ray distance in scene length units. Must be nonnegative.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 * @return `true` if the ray hit an object; otherwise, `false`.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller.
 */
ISAACSIM_PHYSICS_MANAGER_API bool raycastAny(const isaacsim::physics::registration::Float3& origin,
                                             const isaacsim::physics::registration::Float3& unitDirection,
                                             float distance,
                                             bool bothSides);

/**
 * @brief Reports raycast hits from all active physics backends.
 *
 * @param[in] origin Ray origin in world coordinates.
 * @param[in] unitDirection Normalized ray direction.
 * @param[in] distance Maximum ray distance in scene length units. Must be nonnegative.
 * @param[in] reportFunction Callback invoked for each hit. Returning `true` continues the current backend's traversal;
 *                     returning `false` stops it.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API void raycastAll(const isaacsim::physics::registration::Float3& origin,
                                             const isaacsim::physics::registration::Float3& unitDirection,
                                             float distance,
                                             registration::RaycastHitReportFunction reportFunction,
                                             bool bothSides);

/**
 * @brief Finds the closest hit produced by sweeping a sphere through the scene.
 *
 * @param[in] radius Sphere radius in scene length units. Must be nonnegative.
 * @param[in] origin Initial sphere-center position in world coordinates.
 * @param[in] unitDirection Normalized sweep direction.
 * @param[in] distance Maximum sweep distance in scene length units. Must be nonnegative.
 * @param[out] hit Closest hit. The value is modified only when the function returns `true`.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 * @return `true` if the sphere encountered an object; otherwise, `false`.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API bool sweepSphereClosest(float radius,
                                                     const isaacsim::physics::registration::Float3& origin,
                                                     const isaacsim::physics::registration::Float3& unitDirection,
                                                     float distance,
                                                     registration::SweepHit& hit,
                                                     bool bothSides);

/**
 * @brief Tests whether a swept sphere encounters any object in the scene.
 *
 * @param[in] radius Sphere radius in scene length units. Must be nonnegative.
 * @param[in] origin Initial sphere-center position in world coordinates.
 * @param[in] unitDirection Normalized sweep direction.
 * @param[in] distance Maximum sweep distance in scene length units. Must be nonnegative.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 * @return `true` if the sphere encountered an object; otherwise, `false`.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller.
 */
ISAACSIM_PHYSICS_MANAGER_API bool sweepSphereAny(float radius,
                                                 const isaacsim::physics::registration::Float3& origin,
                                                 const isaacsim::physics::registration::Float3& unitDirection,
                                                 float distance,
                                                 bool bothSides);

/**
 * @brief Reports hits produced by sweeping a sphere through the scene.
 *
 * @param[in] radius Sphere radius in scene length units. Must be nonnegative.
 * @param[in] origin Initial sphere-center position in world coordinates.
 * @param[in] unitDirection Normalized sweep direction.
 * @param[in] distance Maximum sweep distance in scene length units. Must be nonnegative.
 * @param[in] reportFunction Callback invoked for each hit. Returning `true` continues the current backend's traversal;
 *                     returning `false` stops it.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API void sweepSphereAll(float radius,
                                                 const isaacsim::physics::registration::Float3& origin,
                                                 const isaacsim::physics::registration::Float3& unitDirection,
                                                 float distance,
                                                 registration::SweepHitReportFunction reportFunction,
                                                 bool bothSides);

/**
 * @brief Reports objects that overlap a sphere.
 *
 * @param[in] radius Sphere radius in scene length units. Must be nonnegative.
 * @param[in] position Sphere-center position in world coordinates.
 * @param[in] reportFunction Callback invoked for each overlap. Returning `true` continues the current backend's
 * traversal; returning `false` stops it.
 * @return Sum of the overlap counts returned by all active physics backends. This value may exceed the number of
 *         `reportFunction` invocations, including when the callback stops a backend's traversal early.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API uint32_t overlapSphere(float radius,
                                                    const isaacsim::physics::registration::Float3& position,
                                                    registration::OverlapHitReportFunction reportFunction);

/**
 * @brief Tests whether a sphere overlaps any object in the scene.
 *
 * @param[in] radius Sphere radius in scene length units. Must be nonnegative.
 * @param[in] position Sphere-center position in world coordinates.
 * @return `true` if the sphere overlaps an object; otherwise, `false`.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller.
 */
ISAACSIM_PHYSICS_MANAGER_API bool overlapSphereAny(float radius, const isaacsim::physics::registration::Float3& position);

/**
 * @brief Reports objects that overlap an oriented box.
 *
 * @param[in] halfExtent Box half-extents in scene length units. Each component must be nonnegative.
 * @param[in] position Box-center position in world coordinates.
 * @param[in] rotation Box orientation as an `(x, y, z, w)` quaternion.
 * @param[in] reportFunction Callback invoked for each overlap. Returning `true` continues the current backend's
 * traversal; returning `false` stops it.
 * @return Sum of the overlap counts returned by all active physics backends. This value may exceed the number of
 *         `reportFunction` invocations, including when the callback stops a backend's traversal early.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API uint32_t overlapBox(const isaacsim::physics::registration::Float3& halfExtent,
                                                 const isaacsim::physics::registration::Float3& position,
                                                 const isaacsim::physics::registration::Float4& rotation,
                                                 registration::OverlapHitReportFunction reportFunction);

/**
 * @brief Tests whether an oriented box overlaps any object in the scene.
 *
 * @param[in] halfExtent Box half-extents in scene length units. Each component must be nonnegative.
 * @param[in] position Box-center position in world coordinates.
 * @param[in] rotation Box orientation as an `(x, y, z, w)` quaternion.
 * @return `true` if the box overlaps an object; otherwise, `false`.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller.
 */
ISAACSIM_PHYSICS_MANAGER_API bool overlapBoxAny(const isaacsim::physics::registration::Float3& halfExtent,
                                                const isaacsim::physics::registration::Float3& position,
                                                const isaacsim::physics::registration::Float4& rotation);

/**
 * @brief Reports objects that overlap a USD geometric primitive.
 *
 * @param[in] geometryPrimitivePath Encoded USD path of the `UsdGeomGPrim` used as the query shape.
 * @param[in] reportFunction Callback invoked for each overlap. Returning `true` continues the current backend's
 * traversal; returning `false` stops it.
 * @return Sum of the overlap counts returned by all active physics backends. This value may exceed the number of
 *         `reportFunction` invocations, including when the callback stops a backend's traversal early.
 *
 * @note A backend may use and cache a convex approximation for a mesh query shape. This behavior is backend-specific.
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API uint32_t overlapShape(registration::PathToken geometryPrimitivePath,
                                                   registration::OverlapHitReportFunction reportFunction);

/**
 * @brief Tests whether a USD geometric primitive overlaps any object in the scene.
 *
 * @param[in] geometryPrimitivePath Encoded USD path of the `UsdGeomGPrim` used as the query shape.
 * @return `true` if the shape overlaps an object; otherwise, `false`.
 *
 * @note A backend may use and cache a convex approximation for a mesh query shape. This behavior is backend-specific.
 * @note Exceptions that escape a backend query callback propagate to the caller.
 */
ISAACSIM_PHYSICS_MANAGER_API bool overlapShapeAny(registration::PathToken geometryPrimitivePath);

/**
 * @brief Finds the closest hit produced by sweeping an oriented box through the scene.
 *
 * @param[in] halfExtent Box half-extents in scene length units. Each component must be nonnegative.
 * @param[in] position Initial box-center position in world coordinates.
 * @param[in] rotation Box orientation as an `(x, y, z, w)` quaternion.
 * @param[in] unitDirection Normalized sweep direction.
 * @param[in] distance Maximum sweep distance in scene length units. Must be nonnegative.
 * @param[out] hit Closest hit. The value is modified only when the function returns `true`.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 * @return `true` if the box encountered an object; otherwise, `false`.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API bool sweepBoxClosest(const isaacsim::physics::registration::Float3& halfExtent,
                                                  const isaacsim::physics::registration::Float3& position,
                                                  const isaacsim::physics::registration::Float4& rotation,
                                                  const isaacsim::physics::registration::Float3& unitDirection,
                                                  float distance,
                                                  registration::SweepHit& hit,
                                                  bool bothSides);

/**
 * @brief Finds the closest hit produced by sweeping a USD geometric primitive through the scene.
 *
 * @param[in] geometryPrimitivePath Encoded USD path of the `UsdGeomGPrim` used as the sweep shape.
 * @param[in] unitDirection Normalized sweep direction.
 * @param[in] distance Maximum sweep distance in scene length units. Must be nonnegative.
 * @param[out] hit Closest hit. The value is modified only when the function returns `true`.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 * @return `true` if the shape encountered an object; otherwise, `false`.
 *
 * @note A backend may use and cache a convex approximation for a mesh query shape. This behavior is backend-specific.
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API bool sweepShapeClosest(registration::PathToken geometryPrimitivePath,
                                                    const isaacsim::physics::registration::Float3& unitDirection,
                                                    float distance,
                                                    registration::SweepHit& hit,
                                                    bool bothSides);

/**
 * @brief Tests whether a swept oriented box encounters any object in the scene.
 *
 * @param[in] halfExtent Box half-extents in scene length units. Each component must be nonnegative.
 * @param[in] position Initial box-center position in world coordinates.
 * @param[in] rotation Box orientation as an `(x, y, z, w)` quaternion.
 * @param[in] unitDirection Normalized sweep direction.
 * @param[in] distance Maximum sweep distance in scene length units. Must be nonnegative.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 * @return `true` if the box encountered an object; otherwise, `false`.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller.
 */
ISAACSIM_PHYSICS_MANAGER_API bool sweepBoxAny(const isaacsim::physics::registration::Float3& halfExtent,
                                              const isaacsim::physics::registration::Float3& position,
                                              const isaacsim::physics::registration::Float4& rotation,
                                              const isaacsim::physics::registration::Float3& unitDirection,
                                              float distance,
                                              bool bothSides);

/**
 * @brief Tests whether a swept USD geometric primitive encounters any object in the scene.
 *
 * @param[in] geometryPrimitivePath Encoded USD path of the `UsdGeomGPrim` used as the sweep shape.
 * @param[in] unitDirection Normalized sweep direction.
 * @param[in] distance Maximum sweep distance in scene length units. Must be nonnegative.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 * @return `true` if the shape encountered an object; otherwise, `false`.
 *
 * @note A backend may use and cache a convex approximation for a mesh query shape. This behavior is backend-specific.
 * @note Exceptions that escape a backend query callback propagate to the caller.
 */
ISAACSIM_PHYSICS_MANAGER_API bool sweepShapeAny(registration::PathToken geometryPrimitivePath,
                                                const isaacsim::physics::registration::Float3& unitDirection,
                                                float distance,
                                                bool bothSides);

/**
 * @brief Reports hits produced by sweeping an oriented box through the scene.
 *
 * @param[in] halfExtent Box half-extents in scene length units. Each component must be nonnegative.
 * @param[in] position Initial box-center position in world coordinates.
 * @param[in] rotation Box orientation as an `(x, y, z, w)` quaternion.
 * @param[in] unitDirection Normalized sweep direction.
 * @param[in] distance Maximum sweep distance in scene length units. Must be nonnegative.
 * @param[in] reportFunction Callback invoked for each hit. Returning `true` continues the current backend's traversal;
 *                     returning `false` stops it.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 *
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API void sweepBoxAll(const isaacsim::physics::registration::Float3& halfExtent,
                                              const isaacsim::physics::registration::Float3& position,
                                              const isaacsim::physics::registration::Float4& rotation,
                                              const isaacsim::physics::registration::Float3& unitDirection,
                                              float distance,
                                              registration::SweepHitReportFunction reportFunction,
                                              bool bothSides);

/**
 * @brief Reports hits produced by sweeping a USD geometric primitive through the scene.
 *
 * @param[in] geometryPrimitivePath Encoded USD path of the `UsdGeomGPrim` used as the sweep shape.
 * @param[in] unitDirection Normalized sweep direction.
 * @param[in] distance Maximum sweep distance in scene length units. Must be nonnegative.
 * @param[in] reportFunction Callback invoked for each hit. Returning `true` continues the current backend's traversal;
 *                     returning `false` stops it.
 * @param[in] bothSides Whether both sides of mesh triangles participate in the query.
 *
 * @note A backend may use and cache a convex approximation for a mesh query shape. This behavior is backend-specific.
 * @note Exceptions that escape a backend query callback propagate to the caller and prevent dispatch to later
 *       backends.
 */
ISAACSIM_PHYSICS_MANAGER_API void sweepShapeAll(registration::PathToken geometryPrimitivePath,
                                                const isaacsim::physics::registration::Float3& unitDirection,
                                                float distance,
                                                registration::SweepHitReportFunction reportFunction,
                                                bool bothSides);
} // namespace manager
} // namespace physics
} // namespace isaacsim
