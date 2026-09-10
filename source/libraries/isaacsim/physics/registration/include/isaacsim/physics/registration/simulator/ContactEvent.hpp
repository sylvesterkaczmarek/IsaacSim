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

#include <cstdint>
#include <functional>
#include <vector>

namespace isaacsim
{
namespace physics
{
namespace registration
{

/**
 * @brief Types of reported contact-pair events.
 */
enum class ContactEventType : uint32_t
{
    /** @brief A contact pair was newly detected. */
    eContactFound,

    /** @brief A previously detected contact pair was lost. */
    eContactLost,

    /** @brief A previously detected contact pair remains in contact. */
    eContactPersist
};

/**
 * @brief Describes one contact-pair event and its associated data ranges.
 *
 * The offset and count members identify slices in the contact and friction-anchor buffers supplied to
 * @c OnContactReportEventFunction.
 */
struct ContactEventHeader
{
    /** @brief The contact event type. */
    ContactEventType type;

    /** @brief The identifier of the simulated USD stage. */
    int64_t stageId;

    /** @brief The encoded USD path of the first actor in the contact pair. */
    PathToken actor0;

    /** @brief The encoded USD path of the second actor in the contact pair. */
    PathToken actor1;

    /** @brief The encoded USD path of the first collider in the contact pair. */
    PathToken collider0;

    /** @brief The encoded USD path of the second collider in the contact pair. */
    PathToken collider1;

    /** @brief The starting index of this pair's data in the contact-data buffer. */
    uint32_t contactDataOffset;

    /** @brief The number of contact-data entries associated with this pair. */
    uint32_t contactDataCount;

    /** @brief The starting index of this pair's data in the friction-anchor buffer. */
    uint32_t frictionAnchorsDataOffset;

    /** @brief The number of friction-anchor entries associated with this pair. */
    uint32_t frictionAnchorDataCount;

    /**
     * @brief The first collider's point-instancer prototype index.
     *
     * A value of @c 0xFFFFFFFF indicates that the collider is not part of a point instancer.
     */
    uint32_t prototypeIndex0;

    /**
     * @brief The second collider's point-instancer prototype index.
     *
     * A value of @c 0xFFFFFFFF indicates that the collider is not part of a point instancer.
     */
    uint32_t prototypeIndex1;
};

/**
 * @brief Describes one contact point in a contact pair.
 */
struct ContactData
{
    /** @brief The contact position in world space, in scene length units. */
    isaacsim::physics::registration::Float3 position;

    /** @brief The contact normal in world space. */
    isaacsim::physics::registration::Float3 normal;

    /** @brief The contact impulse in world space, in mass times scene length per second. */
    isaacsim::physics::registration::Float3 impulse;

    /** @brief The signed separation distance between the contact shapes, in scene length units. */
    float separation;

    /** @brief The first collider's contacted face index when it is a triangle mesh. */
    uint32_t faceIndex0;

    /** @brief The second collider's contacted face index when it is a triangle mesh. */
    uint32_t faceIndex1;

    /** @brief The encoded USD path of the material assigned to the first collider. */
    PathToken material0;

    /** @brief The encoded USD path of the material assigned to the second collider. */
    PathToken material1;
};

/**
 * @brief Describes one friction anchor in a contact pair.
 */
struct FrictionAnchor
{
    /** @brief The friction-anchor position in world space, in scene length units. */
    isaacsim::physics::registration::Float3 position;

    /**
     * @brief The impulse applied at the friction anchor in world space.
     *
     * The impulse has units of mass times scene length per second. Divide it by the simulation time step to obtain
     * force.
     */
    isaacsim::physics::registration::Float3 impulse;
};

/**
 * @brief Sequence of contact-event headers delivered to a contact-report callback.
 */
using ContactEventHeaderVector = std::vector<isaacsim::physics::registration::ContactEventHeader>;

/**
 * @brief Sequence of contact-point data delivered to a contact-report callback.
 */
using ContactDataVector = std::vector<isaacsim::physics::registration::ContactData>;

/**
 * @brief Sequence of friction-anchor data delivered to a contact-report callback.
 */
using FrictionAnchorsDataVector = std::vector<isaacsim::physics::registration::FrictionAnchor>;

/**
 * @brief Callback invoked for a physics contact report.
 *
 * @param[in] eventHeaders The contact-pair event headers.
 * @param[in] contactData The contact points referenced by @p eventHeaders.
 * @param[in] frictionAnchors The friction anchors referenced by @p eventHeaders.
 *
 * @note The callback borrows all three vectors. They remain valid only for the duration of the callback invocation and
 *       must not be retained by reference.
 */
using OnContactReportEventFunction = std::function<void(const ContactEventHeaderVector& eventHeaders,
                                                        const ContactDataVector& contactData,
                                                        const FrictionAnchorsDataVector& frictionAnchors)>;

} // namespace registration
} // namespace physics
} // namespace isaacsim
