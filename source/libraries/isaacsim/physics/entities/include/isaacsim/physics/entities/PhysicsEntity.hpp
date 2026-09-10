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

#include <isaacsim/common/array/Array.hpp>
#include <isaacsim/physics/entities/Export.h>
#include <isaacsim/physics/manager/tensors/EntityView.hpp>

#include <cstddef>
#include <tuple>

namespace isaacsim
{
namespace physics
{
namespace entities
{

namespace array = isaacsim::common::array;

/**
 * @class PhysicsEntity
 * @brief Base wrapper over a batch of simulated physics prims exposed through the tensor API.
 * @details
 * Binds a set of USD prim paths to a physics engine entity view and exposes named data buffers
 * (positions, velocities, masses, ...) as batched arrays. All methods that accept an @p indices
 * parameter operate only on the prims at those positions; when @p indices is omitted, all wrapped
 * prims are processed.
 *
 * The set of valid buffer names depends on the entity kind and the physics engine backing it.
 *
 * @note The wrapped prims must be part of a started simulation for data access to succeed.
 */
class ISAACSIM_PHYSICS_ENTITIES_API PhysicsEntity
{
public:
    /**
     * @brief Construct a physics entity wrapper for one or more USD prim paths.
     * @param[in] engine Name of the physics engine that simulates the prims.
     * @param[in] entity Kind of physics entity to create (e.g. @c "rigid-body", @c "articulation").
     * @param[in] paths  Single path string or list of path strings. May include regular expressions.
     * @throws std::runtime_error if the entity cannot be created for the given engine and paths.
     */
    PhysicsEntity(const std::string& engine,
                  const std::string& entity,
                  const std::variant<std::string, std::vector<std::string>>& paths);
    ~PhysicsEntity() = default;

    /**
     * @brief Return the number of wrapped prims.
     * @return Count of prims managed by this entity.
     */
    size_t numPrims() const noexcept;

    /**
     * @brief Read a named data buffer for the selected prims.
     * @param[in] name    Name of the data buffer to read.
     * @param[in] indices Indices of prims to process (shape @c (N,)). If omitted, all wrapped prims are processed.
     * @return Buffer contents, with the leading dimension matching the number of selected prims.
     * @throws std::runtime_error if the buffer name is not supported by the entity.
     */
    array::Array getData(const std::string& name, const std::optional<array::Array>& indices = std::nullopt);

    /**
     * @brief Write a named data buffer for the selected prims.
     * @param[in] name    Name of the data buffer to write.
     * @param[in] data    New buffer contents, with the leading dimension matching the number of selected prims.
     * @param[in] indices Indices of prims to process (shape @c (N,)). If omitted, all wrapped prims are processed.
     * @throws std::runtime_error if the buffer name is not supported by the entity.
     */
    void setData(const std::string& name,
                 const array::Array& data,
                 const std::optional<array::Array>& indices = std::nullopt);

    /**
     * @brief Read a named data buffer that is composed of several arrays for the selected prims.
     * @param[in] name    Name of the data buffer to read.
     * @param[in] indices Indices of prims to process (shape @c (N,)). If omitted, all wrapped prims are processed.
     * @return Ordered list of buffer components.
     * @throws std::runtime_error if the buffer name is not supported by the entity.
     */
    std::vector<array::Array> getMultiData(const std::string& name,
                                           const std::optional<array::Array>& indices = std::nullopt);

    /**
     * @brief Write a named data buffer that is composed of several arrays for the selected prims.
     * @param[in] name    Name of the data buffer to write.
     * @param[in] data    Ordered list of buffer components to write.
     * @param[in] indices Indices of prims to process (shape @c (N,)). If omitted, all wrapped prims are processed.
     * @throws std::runtime_error if the buffer name is not supported by the entity.
     */
    void setMultiData(const std::string& name,
                      const std::vector<array::Array>& data,
                      const std::optional<array::Array>& indices = std::nullopt);

protected:
    /**
     * @brief View of the wrapped prims held by the physics engine.
     */
    std::shared_ptr<isaacsim::physics::tensors::EntityView> m_entityView;
};

} // namespace entities
} // namespace physics
} // namespace isaacsim
