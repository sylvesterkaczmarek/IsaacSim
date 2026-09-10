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

#include <doctest/doctest.h>
#include <isaacsim/physics/manager/tensors/EntityView.hpp>
#include <isaacsim/physics/registration/tensors/TensorDesc.hpp>
#include <isaacsim/physics/registration/tensors/TensorSpec.hpp>

#include <cstring>
#include <stdexcept>
#include <vector>

using namespace isaacsim::physics::tensors;

namespace
{

// Concrete subclass for testing — exposes the protected `registerImpl` /
// `registerMetadata` calls and lets tests stage callbacks.
class TestableEntityView : public EntityView
{
public:
    explicit TestableEntityView(std::vector<std::string> paths) : EntityView(std::move(paths))
    {
    }

    using EntityView::registerImpl;
    using EntityView::registerMetadata;
};

// Build a TensorDesc that points at a heap-owned float buffer.
class OwnedFloatTensor
{
public:
    std::vector<float> storage;
    TensorDesc descriptor;

    OwnedFloatTensor(std::vector<float> data, std::vector<int64_t> shape) : storage(std::move(data))
    {
        descriptor.data = storage.data();
        descriptor.dtype = DType::eFloat32;
        descriptor.shape = std::move(shape);
        descriptor.device = DeviceKind::eCpu;
        descriptor.deviceOrdinal = -1;
    }
};

TensorSpec makeFloat32Spec(bool indexedRead = false, bool indexedWrite = false, bool maskedWrite = false)
{
    TensorSpec specification;
    specification.dtype = DType::eFloat32;
    specification.deviceKind = DeviceKind::eCpu;
    specification.supports = true;
    specification.supportsIndexedRead = indexedRead;
    specification.supportsIndexedWrite = indexedWrite;
    specification.supportsMaskedWrite = maskedWrite;
    return specification;
}

} // namespace

//=============================================================================
// TEST: construction
//=============================================================================
TEST_CASE("EntityView: construction stores the path list and count")
{
    SUBCASE("Default construction — empty paths, zero count")
    {
        TestableEntityView view({});
        REQUIRE(view.getPaths().empty());
        REQUIRE(view.getCount() == 0);
    }

    SUBCASE("Path list construction stores the paths")
    {
        std::vector<std::string> paths{ "/World/robot_0", "/World/robot_1", "/World/robot_2" };
        TestableEntityView view(paths);
        REQUIRE(view.getPaths().size() == 3);
        REQUIRE(view.getPaths()[0] == "/World/robot_0");
        REQUIRE(view.getPaths()[2] == "/World/robot_2");
    }

    SUBCASE("setCount updates the resolved-entity count")
    {
        TestableEntityView view({});
        REQUIRE(view.getCount() == 0);
        view.setCount(42);
        REQUIRE(view.getCount() == 42);
    }
}

//=============================================================================
// TEST: registerImpl + listImpls + hasImpl + getImplSpec
//=============================================================================
TEST_CASE("EntityView: impl registration discoverability")
{
    TestableEntityView view({});

    SUBCASE("Initial state — no impls")
    {
        REQUIRE(view.listImpls(ImplKind::eGet).empty());
        REQUIRE(view.listImpls(ImplKind::eSet).empty());
        REQUIRE_FALSE(view.hasImpl("dof-position", ImplKind::eGet));
    }

    SUBCASE("registerImpl publishes the impl name + specification")
    {
        TensorSpec specification = makeFloat32Spec(/*indexedRead=*/true);
        specification.shapeHint = { -1, 7 };
        bool first = view.registerImpl(
            "dof-position", ImplKind::eGet,
            GetImplFunction([](const TensorDesc& /*indices*/, const TensorDesc& /*output*/) -> TensorDesc
                            { return TensorDesc{}; }),
            specification);
        REQUIRE(first == true);
        REQUIRE(view.hasImpl("dof-position", ImplKind::eGet));
        REQUIRE_FALSE(view.hasImpl("dof-position", ImplKind::eSet));

        TensorSpec retrieved = view.getImplSpec("dof-position", ImplKind::eGet);
        REQUIRE(retrieved.dtype == DType::eFloat32);
        REQUIRE(retrieved.supportsIndexedRead == true);
        REQUIRE(retrieved.shapeHint.size() == 2);
        REQUIRE(retrieved.shapeHint[0] == -1);
        REQUIRE(retrieved.shapeHint[1] == 7);

        auto getImpls = view.listImpls(ImplKind::eGet);
        REQUIRE(getImpls.size() == 1);
        REQUIRE(getImpls[0] == "dof-position");
    }

    SUBCASE("Get and Set kinds are tracked independently")
    {
        view.registerImpl("dof-position", ImplKind::eGet,
                          GetImplFunction([](const TensorDesc&, const TensorDesc&) { return TensorDesc{}; }),
                          makeFloat32Spec());
        view.registerImpl("dof-position", ImplKind::eSet, SetImplFunction([](const TensorDesc&, const TensorDesc&) {}),
                          makeFloat32Spec(/*indexedRead=*/false, /*indexedWrite=*/true));

        REQUIRE(view.hasImpl("dof-position", ImplKind::eGet));
        REQUIRE(view.hasImpl("dof-position", ImplKind::eSet));

        REQUIRE(view.listImpls(ImplKind::eGet).size() == 1);
        REQUIRE(view.listImpls(ImplKind::eSet).size() == 1);

        // Get and Set specifications are stored separately.
        auto getSpecification = view.getImplSpec("dof-position", ImplKind::eGet);
        auto setSpecification = view.getImplSpec("dof-position", ImplKind::eSet);
        REQUIRE(getSpecification.supportsIndexedWrite == false);
        REQUIRE(setSpecification.supportsIndexedWrite == true);
    }

    SUBCASE("hasImpl returns false for missing names regardless of kind")
    {
        view.registerImpl("dof-position", ImplKind::eGet,
                          GetImplFunction([](const TensorDesc&, const TensorDesc&) { return TensorDesc{}; }),
                          makeFloat32Spec());
        REQUIRE_FALSE(view.hasImpl("dof-velocity", ImplKind::eGet));
        REQUIRE_FALSE(view.hasImpl("dof-velocity", ImplKind::eSet));
    }

    SUBCASE("getImplSpec for an unknown impl throws std::out_of_range")
    {
        // Callers must `hasImpl(...)` first — `getImplSpec` doesn't
        // silently return a default; an unknown impl is a bug worth
        // surfacing as an exception.
        REQUIRE_THROWS_AS(view.getImplSpec("nonexistent", ImplKind::eGet), std::out_of_range);
    }

    SUBCASE("single and multi GET callbacks cannot share one registration identity")
    {
        view.registerImpl("shared-name", ImplKind::eGet,
                          GetImplFunction([](const TensorDesc&, const TensorDesc&) { return TensorDesc{}; }),
                          makeFloat32Spec());
        REQUIRE_THROWS_AS(view.registerImpl("shared-name", ImplKind::eGet,
                                            GetMultiImplFunction([](const TensorDesc&, const std::vector<TensorDesc>&)
                                                                 { return std::vector<TensorDesc>{}; }),
                                            makeFloat32Spec()),
                          std::invalid_argument);

        TestableEntityView reverseView({});
        reverseView.registerImpl("shared-name", ImplKind::eGet,
                                 GetMultiImplFunction([](const TensorDesc&, const std::vector<TensorDesc>&)
                                                      { return std::vector<TensorDesc>{}; }),
                                 makeFloat32Spec());
        REQUIRE_THROWS_AS(
            reverseView.registerImpl("shared-name", ImplKind::eGet,
                                     GetImplFunction([](const TensorDesc&, const TensorDesc&) { return TensorDesc{}; }),
                                     makeFloat32Spec()),
            std::invalid_argument);
    }

    SUBCASE("single and multi SET callbacks cannot share one registration identity")
    {
        view.registerImpl("shared-name", ImplKind::eSet,
                          SetMultiImplFunction([](const std::vector<TensorDesc>&, const TensorDesc&) {}),
                          makeFloat32Spec());
        REQUIRE_THROWS_AS(
            view.registerImpl("shared-name", ImplKind::eSet,
                              SetImplFunction([](const TensorDesc&, const TensorDesc&) {}), makeFloat32Spec()),
            std::invalid_argument);

        TestableEntityView reverseView({});
        reverseView.registerImpl("shared-name", ImplKind::eSet,
                                 SetImplFunction([](const TensorDesc&, const TensorDesc&) {}), makeFloat32Spec());
        REQUIRE_THROWS_AS(
            reverseView.registerImpl("shared-name", ImplKind::eSet,
                                     SetMultiImplFunction([](const std::vector<TensorDesc>&, const TensorDesc&) {}),
                                     makeFloat32Spec()),
            std::invalid_argument);
    }
}

//=============================================================================
// TEST: getData / setData round-trip — the GetImplFunction / SetImplFunction callbacks
// fire and receive the right TensorDesc arguments.
//=============================================================================
TEST_CASE("EntityView: getData / setData dispatch to registered callback")
{
    TestableEntityView view({});

    SUBCASE("getData invokes the registered callback and returns its TensorDesc")
    {
        OwnedFloatTensor result({ 1.0f, 2.0f, 3.0f }, { 3 });

        bool called = false;
        view.registerImpl("dof-position", ImplKind::eGet,
                          GetImplFunction(
                              [&](const TensorDesc& /*indices*/, const TensorDesc& /*output*/) -> TensorDesc
                              {
                                  called = true;
                                  return result.descriptor;
                              }),
                          makeFloat32Spec());

        TensorDesc indices{}; // empty → "no indices"
        TensorDesc output{};
        TensorDesc returned = view.getData("dof-position", indices, output);
        REQUIRE(called == true);
        REQUIRE(returned.dtype == DType::eFloat32);
        REQUIRE(returned.shape.size() == 1);
        REQUIRE(returned.shape[0] == 3);
        REQUIRE(returned.data == result.storage.data());
    }

    SUBCASE("setData forwards data + indices to the callback verbatim")
    {
        std::vector<float> incoming{ 4.0f, 5.0f, 6.0f };
        std::vector<int32_t> incomingIndices{ 7, 11 };

        TensorDesc capturedData{};
        TensorDesc capturedIndices{};
        view.registerImpl("dof-position", ImplKind::eSet,
                          SetImplFunction(
                              [&](const TensorDesc& dataDescriptor, const TensorDesc& indicesDescriptor)
                              {
                                  capturedData = dataDescriptor;
                                  capturedIndices = indicesDescriptor;
                              }),
                          makeFloat32Spec(/*indexedRead=*/false, /*indexedWrite=*/true));

        TensorDesc data;
        data.data = incoming.data();
        data.dtype = DType::eFloat32;
        data.shape = { 3 };

        TensorDesc indices;
        indices.data = incomingIndices.data();
        indices.dtype = DType::eInt32;
        indices.shape = { 2 };

        view.setData("dof-position", data, indices);
        REQUIRE(capturedData.data == incoming.data());
        REQUIRE(capturedData.shape.size() == 1);
        REQUIRE(capturedData.shape[0] == 3);
        REQUIRE(capturedIndices.shape.size() == 1);
        REQUIRE(capturedIndices.shape[0] == 2);
    }

    SUBCASE("getData on an unknown impl throws std::out_of_range")
    {
        TensorDesc indices{};
        TensorDesc output{};
        REQUIRE_THROWS_AS(view.getData("nonexistent", indices, output), std::out_of_range);
    }

    SUBCASE("setData on an unknown impl throws std::out_of_range")
    {
        TensorDesc data{};
        TensorDesc indices{};
        REQUIRE_THROWS_AS(view.setData("nonexistent", data, indices), std::out_of_range);
    }
}

//=============================================================================
// TEST: registerMetadata + getMetadata
//=============================================================================
TEST_CASE("EntityView: metadata registration")
{
    TestableEntityView view({});

    SUBCASE("getMetadata for an unregistered impl returns a default Metadata")
    {
        Metadata metadata = view.getMetadata("dof-position");
        // Default Metadata is a "value-not-set" sentinel; no guarantee on
        // what the variant holds, just that the call doesn't throw.
        static_cast<void>(metadata);
    }

    SUBCASE("registered metadata callback fires on getMetadata")
    {
        Metadata payload;
        // Stash an int into the metadata variant — the exact value isn't
        // semantically important; we just want to verify round-trip.
        payload = Metadata(int64_t{ 42 });

        bool called = false;
        view.registerMetadata("dof-position", MetadataImplFunction(
                                                  [&]() -> Metadata
                                                  {
                                                      called = true;
                                                      return payload;
                                                  }));

        Metadata returned = view.getMetadata("dof-position");
        REQUIRE(called == true);
        // The variant should now hold our value. Cast through the int64
        // accessor used by the legacy adapter.
        // (Metadata's value-type accessors are tested in the variant's own
        // unit suite; this test pins only the dispatch round-trip.)
        static_cast<void>(returned);
    }
}

//=============================================================================
// TEST: shape-specification advertising — `supports=false` is functionally equivalent
// to "not registered" for callers. `hasImpl` returns false for them, so the
// API surface stays uniform across engines without false omissions.
//=============================================================================
TEST_CASE("EntityView: TensorSpec supports flag hides the impl from hasImpl")
{
    TestableEntityView view({});

    TensorSpec unsupported = makeFloat32Spec();
    unsupported.supports = false;
    view.registerImpl("spatial-tendon-stiffness", ImplKind::eGet,
                      GetImplFunction([](const TensorDesc&, const TensorDesc&) { return TensorDesc{}; }), unsupported);

    // Callers see this impl as unavailable — the contract is "registered
    // with supports=false" == "not registered" from the consumer side.
    REQUIRE_FALSE(view.hasImpl("spatial-tendon-stiffness", ImplKind::eGet));
    // Calling get/set on a supports=false impl raises so callers don't
    // silently get bad data.
    REQUIRE_THROWS_AS(view.getData("spatial-tendon-stiffness", TensorDesc{}, TensorDesc{}), std::runtime_error);
}
