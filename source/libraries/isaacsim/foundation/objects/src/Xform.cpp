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

#include <isaacsim/common/logging/Logging.hpp>
#include <isaacsim/foundation/objects/Xform.hpp>
#include <isaacsim/foundation/usd/openusd/Usd.hpp>
#include <isaacsim/foundation/usd/ovstage/Usd.hpp>

#include <stdexcept>

namespace isaacsim
{
namespace foundation
{
namespace objects
{

namespace openusd = isaacsim::foundation::usd::openusd;
namespace ovstage = isaacsim::foundation::usd::ovstage;

Xform::Xform(const std::variant<std::string, std::vector<std::string>>& paths,
             const std::optional<array::Array>& positions,
             const std::optional<array::Array>& translations,
             const std::optional<array::Array>& orientations,
             const std::optional<array::Array>& scales,
             bool resetXformOpProperties,
             bool resolvePaths)
    : Prim(paths, /*resolvePaths=*/false)
{
    m_xformOps = _populateXformOps(this->getStage().getBackend());
    // Get or create Xform prims.
    if (resolvePaths)
    {
        auto [existentPaths, nonexistentPaths] = this->resolvePaths(paths);
        // Get Xform prims.
        if (!existentPaths.empty())
        {
            m_paths = std::move(existentPaths);
            const std::vector<bool> isXformable = this->isA("Xformable").get<std::vector<bool>>();
            for (std::size_t i = 0; i < m_paths.size(); ++i)
            {
                if (!isXformable[i])
                {
                    throw std::runtime_error("The wrapped prim at path '" + m_paths[i] + "' is not a USD Xformable");
                }
            }
        }
        // Create Xform prims.
        else
        {
            m_paths = std::move(nonexistentPaths);
            for (const auto& path : m_paths)
            {
                this->getStage().definePrim(path, "Xform");
            }
        }
    }
    // Initialize instance from arguments.
    _initialize(positions, translations, orientations, scales, resetXformOpProperties);
}

Xform::Xform() : Prim()
{
    m_xformOps = _populateXformOps(this->getStage().getBackend());
}

Xform::XformOps Xform::_populateXformOps(const std::string& backend)
{
    if (backend == "openusd")
    {
        return {
            openusd::resetXformOpProperties, openusd::getXformLocalScales, openusd::setXformLocalScales,
            openusd::getXformLocalPoses,     openusd::setXformLocalPoses,  openusd::getXformWorldPoses,
            openusd::setXformWorldPoses,
        };
    }
    else if (backend == "ovstage")
    {
        return {
            ovstage::resetXformOpProperties, ovstage::getXformLocalScales, ovstage::setXformLocalScales,
            ovstage::getXformLocalPoses,     ovstage::setXformLocalPoses,  ovstage::getXformWorldPoses,
            ovstage::setXformWorldPoses,
        };
    }
    throw std::runtime_error("Unknown backend '" + backend + "'. Expected 'openusd' or 'ovstage'.");
}

void Xform::_initialize(const std::optional<array::Array>& positions,
                        const std::optional<array::Array>& translations,
                        const std::optional<array::Array>& orientations,
                        const std::optional<array::Array>& scales,
                        bool resetXformOpProperties)
{
    // Reset xformOp properties.
    if (!m_nonRootArticulationLink && resetXformOpProperties)
    {
        this->resetXformOpProperties();
    }
    // Set specified values.
    if (positions.has_value() || translations.has_value() || orientations.has_value() || scales.has_value())
    {
        if (positions.has_value() && translations.has_value())
        {
            throw std::invalid_argument("Both 'positions' and 'translations' are specified. Specify only one of them");
        }
        if (m_nonRootArticulationLink)
        {
            throw std::invalid_argument(
                "The prim is a non-root link in an articulation. "
                "Specified values (positions, translations, orientations and/or scales) cannot be set");
        }
        if (positions.has_value() || orientations.has_value())
        {
            this->setWorldPoses(positions, orientations);
        }
        if (translations.has_value() || orientations.has_value())
        {
            this->setLocalPoses(translations, orientations);
        }
        if (scales.has_value())
        {
            this->setLocalScales(*scales);
        }
    }
}

void Xform::setVisibilities(const array::Array& visibilities, const std::optional<array::Array>& indices)
{
    // Note: this sets the authored visibility token, not the computed visibility.
    // MakeVisible/MakeInvisible is not exposed through the base-class Prim interface. Use USD APIs directly if needed.
    const int64_t batchSize = _resolveIndexedSize(indices);
    auto bools = visibilities.broadcastTo(array::Shape({ batchSize, int64_t{ 1 } }))
                     .reshape(array::Shape({ int64_t{ -1 } }))
                     .get<std::vector<bool>>();
    std::vector<std::string> tokens(bools.size());
    for (std::size_t i = 0; i < bools.size(); ++i)
    {
        tokens[i] = bools[i] ? "inherited" : "invisible";
    }
    this->setAttributeValues("visibility", tokens, indices);
}

array::Array Xform::getVisibilities(const std::optional<array::Array>& indices)
{
    // Note: this gets the authored visibility token, not the computed visibility.
    // Requires UsdGeomImageable::ComputeVisibility, which is not available through the
    // base-class Prim interface. Use USD APIs directly if needed.
    const auto tokens = std::get<std::vector<std::string>>(this->getAttributeValues("visibility", indices));
    std::vector<bool> bools(tokens.size());
    for (std::size_t i = 0; i < tokens.size(); ++i)
    {
        bools[i] = tokens[i] != "invisible";
    }
    return array::Array(bools).reshape(array::Shape({ int64_t{ -1 }, int64_t{ 1 } }));
}

// TODO: applyVisualMaterials
// TODO: getAppliedVisualMaterials

std::tuple<array::Array, array::Array> Xform::getWorldPoses(const std::optional<array::Array>& indices)
{
    const int64_t stageId = this->getStage().getStageId();
    return m_xformOps.getXformWorldPoses(stageId, _resolveIndexedPaths(indices));
}

void Xform::setWorldPoses(const std::optional<array::Array>& positions,
                          const std::optional<array::Array>& orientations,
                          const std::optional<array::Array>& indices)
{
    if (!positions.has_value() && !orientations.has_value())
    {
        throw std::invalid_argument("Both `positions` and `orientations` are not defined. Define at least one of them");
    }
    const int64_t stageId = this->getStage().getStageId();
    const int64_t batchSize = _resolveIndexedSize(indices);
    std::optional<array::Array> broadcastedPositions = positions;
    std::optional<array::Array> broadcastedOrientations = orientations;
    if (positions.has_value())
    {
        broadcastedPositions = positions->reshape(array::Shape({ int64_t{ -1 }, int64_t{ 3 } }))
                                   .broadcastTo(array::Shape({ batchSize, int64_t{ 3 } }));
    }
    if (orientations.has_value())
    {
        broadcastedOrientations = orientations->reshape(array::Shape({ int64_t{ -1 }, int64_t{ 4 } }))
                                      .broadcastTo(array::Shape({ batchSize, int64_t{ 4 } }));
    }
    m_xformOps.setXformWorldPoses(stageId, _resolveIndexedPaths(indices), broadcastedPositions, broadcastedOrientations);
}

std::tuple<array::Array, array::Array> Xform::getLocalPoses(const std::optional<array::Array>& indices)
{
    const int64_t stageId = this->getStage().getStageId();
    return m_xformOps.getXformLocalPoses(stageId, _resolveIndexedPaths(indices));
}

void Xform::setLocalPoses(const std::optional<array::Array>& translations,
                          const std::optional<array::Array>& orientations,
                          const std::optional<array::Array>& indices)
{
    if (!translations.has_value() && !orientations.has_value())
    {
        throw std::invalid_argument("Both `translations` and `orientations` are not defined. Define at least one of them");
    }
    const int64_t stageId = this->getStage().getStageId();
    const int64_t batchSize = _resolveIndexedSize(indices);
    std::optional<array::Array> broadcastedTranslations = translations;
    std::optional<array::Array> broadcastedOrientations = orientations;
    if (translations.has_value())
    {
        broadcastedTranslations = translations->reshape(array::Shape({ int64_t{ -1 }, int64_t{ 3 } }))
                                      .broadcastTo(array::Shape({ batchSize, int64_t{ 3 } }));
    }
    if (orientations.has_value())
    {
        broadcastedOrientations = orientations->reshape(array::Shape({ int64_t{ -1 }, int64_t{ 4 } }))
                                      .broadcastTo(array::Shape({ batchSize, int64_t{ 4 } }));
    }
    m_xformOps.setXformLocalPoses(
        stageId, _resolveIndexedPaths(indices), broadcastedTranslations, broadcastedOrientations);
}

array::Array Xform::getLocalScales(const std::optional<array::Array>& indices)
{
    const int64_t stageId = this->getStage().getStageId();
    return m_xformOps.getXformLocalScales(stageId, _resolveIndexedPaths(indices));
}

void Xform::setLocalScales(const array::Array& scales, const std::optional<array::Array>& indices)
{
    const int64_t stageId = this->getStage().getStageId();
    const int64_t batchSize = _resolveIndexedSize(indices);
    m_xformOps.setXformLocalScales(stageId, _resolveIndexedPaths(indices),
                                   scales.reshape(array::Shape({ int64_t{ -1 }, int64_t{ 3 } }))
                                       .broadcastTo(array::Shape({ batchSize, int64_t{ 3 } })));
}

void Xform::resetXformOpProperties()
{
    const int64_t stageId = this->getStage().getStageId();
    for (const auto& path : m_paths)
    {
        m_xformOps.resetXformOpProperties(stageId, path);
    }
}

} // namespace objects
} // namespace foundation
} // namespace isaacsim
