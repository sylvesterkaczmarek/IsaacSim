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

#include <isaacsim/foundation/objects/Xform.hpp>

namespace isaacsim
{
namespace foundation
{
namespace objects
{
namespace shapes
{

/// @brief Variant type for display colors: a single color token, a list of color tokens, or a numeric RGB array.
using ColorType = std::variant<std::string, std::vector<std::string>, array::Array>;

/**
 * @class Shape
 * @brief Base class for USD geometry shape prim wrappers.
 * @details
 * Extends Xform with display color control and the @c updateExtents interface that concrete shape subclasses
 * must implement to keep USD extents synchronized with their geometry attributes.
 */
class ISAACSIM_FOUNDATION_OBJECTS_API Shape : public Xform
{
public:
    ~Shape() = default;

    /**
     * @brief Set the display colors of the selected prims.
     * @param[in] colors  Color values as RGB triples (shape @c (N,3)), a single color token string,
     *                    or a list of color token strings. Broadcast rules apply for smaller inputs.
     * @param[in] indices Indices of prims to process. If omitted, all wrapped prims are processed.
     */
    void setDisplayColors(const ColorType& colors, const std::optional<array::Array>& indices = std::nullopt);

    /**
     * @brief Get the display colors of the selected prims.
     * @param[in] indices Indices of prims to process. If omitted, all wrapped prims are processed.
     * @return Display color values as RGB triples (shape @c (N,3)).
     */
    array::Array getDisplayColors(const std::optional<array::Array>& indices = std::nullopt);

    /**
     * @brief Update the USD extent attribute of the wrapped prims to match their current geometry.
     * @details Must be called after changing shape-specific geometry attributes (e.g. radii, sizes)
     *          to keep the USD stage consistent.
     */
    void updateExtents();

protected:
    /**
     * @brief Construct a Shape wrapper.
     * @param[in] paths                  Single path or list of paths to USD shape prims.
     * @param[in] shapeType              USD geometry type name used when creating new prims.
     * @param[in] colors                 Initial display colors. Optional.
     * @param[in] positions              Initial world-frame positions (shape @c (N,3)). Optional.
     * @param[in] translations           Initial local-frame translations (shape @c (N,3)). Optional.
     * @param[in] orientations           Initial orientations as quaternions @c wxyz (shape @c (N,4)). Optional.
     * @param[in] scales                 Initial local scales (shape @c (N,3)). Optional.
     * @param[in] resetXformOpProperties Whether to normalize the xformOp stack before applying the
     *                                   initial transform.
     */
    Shape(const std::variant<std::string, std::vector<std::string>>& paths,
          // Shape (internal)
          const std::string& shapeType,
          // Shape
          const std::optional<ColorType>& colors = std::nullopt,
          // Xform
          const std::optional<array::Array>& positions = std::nullopt,
          const std::optional<array::Array>& translations = std::nullopt,
          const std::optional<array::Array>& orientations = std::nullopt,
          const std::optional<array::Array>& scales = std::nullopt,
          bool resetXformOpProperties = true);

    /** @brief Expands and validates one or more requested shape axes. */
    std::vector<std::string> _resolveAxes(const std::variant<std::string, std::vector<std::string>>& axes);
};

} // namespace shapes
} // namespace objects
} // namespace foundation
} // namespace isaacsim
