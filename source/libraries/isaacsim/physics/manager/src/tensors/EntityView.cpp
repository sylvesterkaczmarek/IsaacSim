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

#include <isaacsim/physics/manager/tensors/EntityView.hpp>

#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace isaacsim
{
namespace physics
{
namespace tensors
{

struct EntityView::Impl
{
    struct GetEntry
    {
        GetImplFunction callback;
        TensorSpec specification;
    };
    struct SetEntry
    {
        SetImplFunction callback;
        TensorSpec specification;
    };
    struct GetMultiEntry
    {
        GetMultiImplFunction callback;
        TensorSpec specification;
        std::vector<TensorSpec> outputSpecifications;
    };
    struct SetMultiEntry
    {
        SetMultiImplFunction callback;
        TensorSpec specification;
    };

    std::unordered_map<std::string, GetEntry> getImplementations;
    std::unordered_map<std::string, SetEntry> setImplementations;
    std::unordered_map<std::string, GetMultiEntry> getMultiImplementations;
    std::unordered_map<std::string, SetMultiEntry> setMultiImplementations;
    std::unordered_map<std::string, MetadataImplFunction> metadataImplementations;
};

EntityView::EntityView() : m_impl(std::make_unique<Impl>())
{
}

EntityView::EntityView(std::vector<std::string> paths) : m_paths(std::move(paths)), m_impl(std::make_unique<Impl>())
{
}

EntityView::~EntityView() = default;

bool EntityView::registerImpl(const std::string& operationName,
                              ImplKind kind,
                              GetImplFunction callback,
                              TensorSpec specification)
{
    if (!m_impl)
    {
        m_impl = std::make_unique<Impl>();
    }
    if (kind != ImplKind::eGet)
    {
        throw std::invalid_argument("registerImpl(GetImplFunction): kind must be ImplKind::eGet for impl '" +
                                    operationName + "'");
    }
    if (m_impl->getMultiImplementations.find(operationName) != m_impl->getMultiImplementations.end())
    {
        throw std::invalid_argument("EntityView::registerImpl: GET impl '" + operationName +
                                    "' is already registered as multi-buffer");
    }
    const bool isNew = m_impl->getImplementations.find(operationName) == m_impl->getImplementations.end();
    m_impl->getImplementations[operationName] = Impl::GetEntry{ std::move(callback), std::move(specification) };
    return isNew;
}

bool EntityView::registerImpl(const std::string& operationName,
                              ImplKind kind,
                              SetImplFunction callback,
                              TensorSpec specification)
{
    if (!m_impl)
    {
        m_impl = std::make_unique<Impl>();
    }
    if (kind != ImplKind::eSet)
    {
        throw std::invalid_argument("registerImpl(SetImplFunction): kind must be ImplKind::eSet for impl '" +
                                    operationName + "'");
    }
    if (m_impl->setMultiImplementations.find(operationName) != m_impl->setMultiImplementations.end())
    {
        throw std::invalid_argument("EntityView::registerImpl: SET impl '" + operationName +
                                    "' is already registered as multi-buffer");
    }
    const bool isNew = m_impl->setImplementations.find(operationName) == m_impl->setImplementations.end();
    m_impl->setImplementations[operationName] = Impl::SetEntry{ std::move(callback), std::move(specification) };
    return isNew;
}

bool EntityView::registerImpl(const std::string& operationName,
                              ImplKind kind,
                              GetMultiImplFunction callback,
                              TensorSpec specification)
{
    if (!m_impl)
    {
        m_impl = std::make_unique<Impl>();
    }
    if (kind != ImplKind::eGet)
    {
        throw std::invalid_argument("registerImpl(GetMultiImplFunction): kind must be ImplKind::eGet for impl '" +
                                    operationName + "'");
    }
    if (m_impl->getImplementations.find(operationName) != m_impl->getImplementations.end())
    {
        throw std::invalid_argument("EntityView::registerImpl: GET impl '" + operationName +
                                    "' is already registered as single-buffer");
    }
    const bool isNew = m_impl->getMultiImplementations.find(operationName) == m_impl->getMultiImplementations.end();
    m_impl->getMultiImplementations[operationName] =
        Impl::GetMultiEntry{ std::move(callback), std::move(specification), {} };
    return isNew;
}

bool EntityView::registerImpl(const std::string& operationName,
                              ImplKind kind,
                              GetMultiImplFunction callback,
                              std::vector<TensorSpec> outputSpecifications)
{
    if (!m_impl)
    {
        m_impl = std::make_unique<Impl>();
    }
    if (kind != ImplKind::eGet)
    {
        throw std::invalid_argument(
            "registerImpl(GetMultiImplFunction, outputSpecifications): kind must be ImplKind::eGet for impl '" +
            operationName + "'");
    }
    if (m_impl->getImplementations.find(operationName) != m_impl->getImplementations.end())
    {
        throw std::invalid_argument("EntityView::registerImpl: GET impl '" + operationName +
                                    "' is already registered as single-buffer");
    }
    // The aggregate specification drives the supports and indexed-read gates in getDataMulti. The per-output
    // specifications drive framework-side device allocation.
    TensorSpec aggregateSpecification;
    aggregateSpecification.supports = !outputSpecifications.empty() && outputSpecifications.front().supports;
    if (!outputSpecifications.empty())
    {
        aggregateSpecification.dtype = outputSpecifications.front().dtype;
        aggregateSpecification.shapeHint = outputSpecifications.front().shapeHint;
        aggregateSpecification.supportsIndexedRead = outputSpecifications.front().supportsIndexedRead;
        aggregateSpecification.requiresHostData = outputSpecifications.front().requiresHostData;
    }
    const bool isNew = m_impl->getMultiImplementations.find(operationName) == m_impl->getMultiImplementations.end();
    m_impl->getMultiImplementations[operationName] =
        Impl::GetMultiEntry{ std::move(callback), aggregateSpecification, std::move(outputSpecifications) };
    return isNew;
}

bool EntityView::registerImpl(const std::string& operationName,
                              ImplKind kind,
                              SetMultiImplFunction callback,
                              TensorSpec specification)
{
    if (!m_impl)
    {
        m_impl = std::make_unique<Impl>();
    }
    if (kind != ImplKind::eSet)
    {
        throw std::invalid_argument("registerImpl(SetMultiImplFunction): kind must be ImplKind::eSet for impl '" +
                                    operationName + "'");
    }
    if (m_impl->setImplementations.find(operationName) != m_impl->setImplementations.end())
    {
        throw std::invalid_argument("EntityView::registerImpl: SET impl '" + operationName +
                                    "' is already registered as single-buffer");
    }
    const bool isNew = m_impl->setMultiImplementations.find(operationName) == m_impl->setMultiImplementations.end();
    m_impl->setMultiImplementations[operationName] = Impl::SetMultiEntry{ std::move(callback), std::move(specification) };
    return isNew;
}

bool EntityView::registerMetadata(const std::string& operationName, MetadataImplFunction callback)
{
    if (!m_impl)
    {
        m_impl = std::make_unique<Impl>();
    }
    const bool isNew = m_impl->metadataImplementations.find(operationName) == m_impl->metadataImplementations.end();
    m_impl->metadataImplementations[operationName] = std::move(callback);
    return isNew;
}

std::vector<std::string> EntityView::listImpls(ImplKind kind) const
{
    std::vector<std::string> implementationNames;
    if (!m_impl)
    {
        return implementationNames;
    }
    if (kind == ImplKind::eGet)
    {
        implementationNames.reserve(m_impl->getImplementations.size() + m_impl->getMultiImplementations.size());
        for (const auto& entry : m_impl->getImplementations)
        {
            implementationNames.push_back(entry.first);
        }
        for (const auto& entry : m_impl->getMultiImplementations)
        {
            implementationNames.push_back(entry.first);
        }
    }
    else
    {
        implementationNames.reserve(m_impl->setImplementations.size() + m_impl->setMultiImplementations.size());
        for (const auto& entry : m_impl->setImplementations)
        {
            implementationNames.push_back(entry.first);
        }
        for (const auto& entry : m_impl->setMultiImplementations)
        {
            implementationNames.push_back(entry.first);
        }
    }
    return implementationNames;
}

bool EntityView::hasImpl(const std::string& operationName, ImplKind kind) const
{
    if (!m_impl)
    {
        return false;
    }

    const auto isSupported = [](const TensorSpec& specification) { return specification.supports; };

    if (kind == ImplKind::eGet)
    {
        const auto implementationIterator = m_impl->getImplementations.find(operationName);
        if (implementationIterator != m_impl->getImplementations.end())
        {
            return isSupported(implementationIterator->second.specification);
        }
        const auto multiImplementationIterator = m_impl->getMultiImplementations.find(operationName);
        if (multiImplementationIterator != m_impl->getMultiImplementations.end())
        {
            return isSupported(multiImplementationIterator->second.specification);
        }
        return false;
    }
    else
    {
        const auto implementationIterator = m_impl->setImplementations.find(operationName);
        if (implementationIterator != m_impl->setImplementations.end())
        {
            return isSupported(implementationIterator->second.specification);
        }
        const auto multiImplementationIterator = m_impl->setMultiImplementations.find(operationName);
        if (multiImplementationIterator != m_impl->setMultiImplementations.end())
        {
            return isSupported(multiImplementationIterator->second.specification);
        }
        return false;
    }
}

TensorSpec EntityView::getImplSpec(const std::string& operationName, ImplKind kind) const
{
    if (!m_impl)
    {
        throw std::out_of_range("EntityView::getImplSpec: no impls registered for '" + operationName + "'");
    }

    if (kind == ImplKind::eGet)
    {
        const auto implementationIterator = m_impl->getImplementations.find(operationName);
        if (implementationIterator != m_impl->getImplementations.end())
        {
            return implementationIterator->second.specification;
        }
        const auto multiImplementationIterator = m_impl->getMultiImplementations.find(operationName);
        if (multiImplementationIterator != m_impl->getMultiImplementations.end())
        {
            return multiImplementationIterator->second.specification;
        }
    }
    else
    {
        const auto implementationIterator = m_impl->setImplementations.find(operationName);
        if (implementationIterator != m_impl->setImplementations.end())
        {
            return implementationIterator->second.specification;
        }
        const auto multiImplementationIterator = m_impl->setMultiImplementations.find(operationName);
        if (multiImplementationIterator != m_impl->setMultiImplementations.end())
        {
            return multiImplementationIterator->second.specification;
        }
    }
    throw std::out_of_range("EntityView::getImplSpec: impl '" + operationName + "' not registered for the requested kind");
}

std::vector<TensorSpec> EntityView::getImplSpecMulti(const std::string& operationName, ImplKind kind) const
{
    if (!m_impl || kind != ImplKind::eGet)
    {
        return {};
    }
    const auto multiImplementationIterator = m_impl->getMultiImplementations.find(operationName);
    if (multiImplementationIterator != m_impl->getMultiImplementations.end())
    {
        return multiImplementationIterator->second.outputSpecifications;
    }
    return {};
}

bool EntityView::_setImplShapeHint(const std::string& operationName, ImplKind kind, const std::vector<int64_t>& shapeHint)
{
    if (!m_impl)
    {
        return false;
    }
    if (kind == ImplKind::eGet)
    {
        const auto implementationIterator = m_impl->getImplementations.find(operationName);
        if (implementationIterator == m_impl->getImplementations.end())
        {
            return false;
        }
        implementationIterator->second.specification.shapeHint = shapeHint;
        return true;
    }
    const auto implementationIterator = m_impl->setImplementations.find(operationName);
    if (implementationIterator == m_impl->setImplementations.end())
    {
        return false;
    }
    implementationIterator->second.specification.shapeHint = shapeHint;
    return true;
}

bool EntityView::_setImplOutputShapeHints(const std::string& operationName,
                                          ImplKind kind,
                                          const std::vector<std::vector<int64_t>>& shapeHints)
{
    if (!m_impl || kind != ImplKind::eGet)
    {
        return false;
    }
    const auto multiImplementationIterator = m_impl->getMultiImplementations.find(operationName);
    if (multiImplementationIterator == m_impl->getMultiImplementations.end())
    {
        return false;
    }
    std::vector<TensorSpec>& outputSpecifications = multiImplementationIterator->second.outputSpecifications;
    if (shapeHints.size() != outputSpecifications.size())
    {
        throw std::invalid_argument("EntityView::_setImplOutputShapeHints: impl '" + operationName + "' has " +
                                    std::to_string(outputSpecifications.size()) + " outputs, received " +
                                    std::to_string(shapeHints.size()));
    }
    for (size_t outputIndex = 0; outputIndex < outputSpecifications.size(); ++outputIndex)
    {
        outputSpecifications[outputIndex].shapeHint = shapeHints[outputIndex];
    }
    return true;
}

Metadata EntityView::getMetadata(const std::string& operationName) const
{
    if (!m_impl)
    {
        return Metadata{};
    }
    const auto implementationIterator = m_impl->metadataImplementations.find(operationName);
    if (implementationIterator == m_impl->metadataImplementations.end())
    {
        return Metadata{};
    }
    return implementationIterator->second();
}

TensorDesc EntityView::getData(const std::string& operationName, const TensorDesc& indices, const TensorDesc& output) const
{
    if (!m_impl)
    {
        throw std::out_of_range("EntityView::getData: no impls registered for '" + operationName + "'");
    }
    const auto implementationIterator = m_impl->getImplementations.find(operationName);
    if (implementationIterator == m_impl->getImplementations.end())
    {
        throw std::out_of_range("EntityView::getData: get-impl '" + operationName + "' not registered");
    }
    if (!implementationIterator->second.specification.supports)
    {
        throw std::runtime_error("EntityView::getData: impl '" + operationName +
                                 "' is registered but reports supports=false (operation not implemented for this engine)");
    }
    if (!indices.isEmpty() && !implementationIterator->second.specification.supportsIndexedRead)
    {
        throw std::invalid_argument("EntityView::getData: impl '" + operationName + "' does not support indexed reads");
    }
    return implementationIterator->second.callback(indices, output);
}

std::vector<TensorDesc> EntityView::getDataMulti(const std::string& operationName,
                                                 const TensorDesc& indices,
                                                 const std::vector<TensorDesc>& output) const
{
    if (!m_impl)
    {
        throw std::out_of_range("EntityView::getDataMulti: no impls registered for '" + operationName + "'");
    }
    const auto implementationIterator = m_impl->getMultiImplementations.find(operationName);
    if (implementationIterator == m_impl->getMultiImplementations.end())
    {
        throw std::out_of_range("EntityView::getDataMulti: multi-get-impl '" + operationName + "' not registered");
    }
    if (!implementationIterator->second.specification.supports)
    {
        throw std::runtime_error("EntityView::getDataMulti: impl '" + operationName +
                                 "' is registered but reports supports=false (operation not implemented for this engine)");
    }
    if (!indices.isEmpty() && !implementationIterator->second.specification.supportsIndexedRead)
    {
        throw std::invalid_argument("EntityView::getDataMulti: impl '" + operationName +
                                    "' does not support indexed reads");
    }
    return implementationIterator->second.callback(indices, output);
}

void EntityView::setData(const std::string& operationName, const TensorDesc& data, const TensorDesc& indices) const
{
    if (!m_impl)
    {
        throw std::out_of_range("EntityView::setData: no impls registered for '" + operationName + "'");
    }
    const auto implementationIterator = m_impl->setImplementations.find(operationName);
    if (implementationIterator == m_impl->setImplementations.end())
    {
        throw std::out_of_range("EntityView::setData: set-impl '" + operationName + "' not registered");
    }
    if (!implementationIterator->second.specification.supports)
    {
        throw std::runtime_error("EntityView::setData: impl '" + operationName +
                                 "' is registered but reports supports=false (operation not implemented for this engine)");
    }
    if (!indices.isEmpty() && !implementationIterator->second.specification.supportsIndexedWrite)
    {
        throw std::invalid_argument("EntityView::setData: impl '" + operationName + "' does not support indexed writes");
    }
    implementationIterator->second.callback(data, indices);
}

void EntityView::setDataMulti(const std::string& operationName,
                              const std::vector<TensorDesc>& data,
                              const TensorDesc& indices) const
{
    if (!m_impl)
    {
        throw std::out_of_range("EntityView::setDataMulti: no impls registered for '" + operationName + "'");
    }
    const auto implementationIterator = m_impl->setMultiImplementations.find(operationName);
    if (implementationIterator == m_impl->setMultiImplementations.end())
    {
        throw std::out_of_range("EntityView::setDataMulti: multi-set-impl '" + operationName + "' not registered");
    }
    if (!implementationIterator->second.specification.supports)
    {
        throw std::runtime_error("EntityView::setDataMulti: impl '" + operationName +
                                 "' is registered but reports supports=false (operation not implemented for this engine)");
    }
    if (!indices.isEmpty() && !implementationIterator->second.specification.supportsIndexedWrite)
    {
        throw std::invalid_argument("EntityView::setDataMulti: impl '" + operationName +
                                    "' does not support indexed writes");
    }
    implementationIterator->second.callback(data, indices);
}

} // namespace tensors
} // namespace physics
} // namespace isaacsim
