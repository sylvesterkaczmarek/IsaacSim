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

#include "BindingsPhysics.hpp"
#include "TensorNbHelpers.hpp"

#include <isaacsim/physics/manager/tensors/EntityView.hpp>
#include <isaacsim/physics/manager/tensors/SimulationView.hpp>
#include <isaacsim/physics/registration/tensors/IEntityView.hpp>
#include <isaacsim/physics/registration/tensors/ISimulationView.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <nanobind/trampoline.h>

namespace nb = nanobind;
namespace manager_details = isaacsim::physics::manager::details;
using namespace isaacsim::physics::tensors;

namespace
{
// Wrap a Python callable as a C++ GetImplFunction. The wrapper converts the
// incoming TensorDescs into ndarray views, calls into Python, and converts
// the returned ndarray back into a TensorDesc the manager can hand out to
// callers. The returned TensorDesc aliases the Python result's storage, so it
// carries a keepalive pinning that Python object -- `data` stays valid until
// the last owner of the returned descriptor drops.
GetImplFunction wrapPythonGetCallback(nb::callable callback)
{
    return [callback = std::move(callback)](const TensorDesc& indices, const TensorDesc& output) -> TensorDesc
    {
        nb::gil_scoped_acquire globalInterpreterLockGuard;
        nb::object indicesObject = indices.isEmpty() ?
                                       nb::object{ nb::none() } :
                                       nb::cast(manager_details::convertTensorDescriptorToNumpy(indices, nb::none()));
        nb::object outputObject = output.isEmpty() ?
                                      nb::object{ nb::none() } :
                                      nb::cast(manager_details::convertTensorDescriptorToNumpy(output, nb::none()));
        nb::object result = callback(indicesObject, outputObject);
        if (result.is_none())
        {
            return TensorDesc{};
        }
        // The descriptor aliases the Python result's storage; pin the real owner so
        // `data` can't dangle once this lambda drops its reference.
        return manager_details::convertResultToTensorDescriptor(result, &output);
    };
}

SetImplFunction wrapPythonSetCallback(nb::callable callback)
{
    return [callback = std::move(callback)](const TensorDesc& data, const TensorDesc& indices)
    {
        nb::gil_scoped_acquire globalInterpreterLockGuard;
        nb::object dataObject = data.isEmpty() ?
                                    nb::object{ nb::none() } :
                                    nb::cast(manager_details::convertTensorDescriptorToNumpy(data, nb::none()));
        nb::object indicesObject = indices.isEmpty() ?
                                       nb::object{ nb::none() } :
                                       nb::cast(manager_details::convertTensorDescriptorToNumpy(indices, nb::none()));
        callback(dataObject, indicesObject);
    };
}

GetMultiImplFunction wrapPythonMultiGetCallback(nb::callable callback)
{
    return [callback = std::move(callback)](
               const TensorDesc& indices, const std::vector<TensorDesc>& output) -> std::vector<TensorDesc>
    {
        nb::gil_scoped_acquire globalInterpreterLockGuard;
        nb::object indicesObject = indices.isEmpty() ?
                                       nb::object{ nb::none() } :
                                       nb::cast(manager_details::convertTensorDescriptorToNumpy(indices, nb::none()));
        nb::list outputList;
        for (const TensorDesc& descriptor : output)
        {
            outputList.append(descriptor.isEmpty() ?
                                  nb::object{ nb::none() } :
                                  nb::cast(manager_details::convertTensorDescriptorToNumpy(descriptor, nb::none())));
        }
        nb::object result = callback(indicesObject, outputList);
        std::vector<TensorDesc> results;
        size_t outputIndex = 0;
        for (auto resultItem : nb::cast<nb::sequence>(result))
        {
            // Pin each output's real storage owner; each result aligns with the corresponding caller output slot.
            const TensorDesc* outputAlias = outputIndex < output.size() ? &output[outputIndex] : nullptr;
            results.push_back(manager_details::convertResultToTensorDescriptor(nb::borrow(resultItem), outputAlias));
            ++outputIndex;
        }
        return results;
    };
}

SetMultiImplFunction wrapPythonMultiSetCallback(nb::callable callback)
{
    return [callback = std::move(callback)](const std::vector<TensorDesc>& data, const TensorDesc& indices)
    {
        nb::gil_scoped_acquire globalInterpreterLockGuard;
        nb::list dataList;
        for (const TensorDesc& descriptor : data)
        {
            dataList.append(descriptor.isEmpty() ?
                                nb::object{ nb::none() } :
                                nb::cast(manager_details::convertTensorDescriptorToNumpy(descriptor, nb::none())));
        }
        nb::object indicesObject = indices.isEmpty() ?
                                       nb::object{ nb::none() } :
                                       nb::cast(manager_details::convertTensorDescriptorToNumpy(indices, nb::none()));
        callback(dataList, indicesObject);
    };
}

// Trampoline so Python subclasses of SimulationView can override the seven
// view factories and the scene-control hooks.
class PythonSimulationView : public SimulationView
{
public:
    NB_TRAMPOLINE(SimulationView, 22);

    PythonSimulationView() = default;
    PythonSimulationView(std::string engine, std::string frontendName, int64_t stageId)
        : SimulationView(std::move(engine), std::move(frontendName), stageId)
    {
    }

    int getDeviceOrdinal() const override
    {
        NB_OVERRIDE_NAME("get_device_ordinal", getDeviceOrdinal, );
    }
    int getParameterDeviceOrdinal() const override
    {
        NB_OVERRIDE_NAME("get_param_device_ordinal", getParameterDeviceOrdinal, );
    }
    bool isValid() const override
    {
        NB_OVERRIDE_NAME("is_valid", isValid, );
    }
    void setGravity(Float3 gravity) override
    {
        NB_OVERRIDE_NAME("set_gravity", setGravity, gravity);
    }
    Float3 getGravity() const override
    {
        NB_OVERRIDE_NAME("get_gravity", getGravity, );
    }
    void clearForces() override
    {
        NB_OVERRIDE_NAME("clear_forces", clearForces, );
    }
    void step(float timeStep) override
    {
        NB_OVERRIDE_NAME("step", step, timeStep);
    }
    void updateArticulationsKinematic() override
    {
        NB_OVERRIDE_NAME("update_articulations_kinematic", updateArticulationsKinematic, );
    }
    void initializeKinematicBodies() override
    {
        NB_OVERRIDE_NAME("initialize_kinematic_bodies", initializeKinematicBodies, );
    }
    ObjectType getObjectType(const std::string& primPath) const override
    {
        NB_OVERRIDE_NAME("get_object_type", getObjectType, primPath);
    }

    std::shared_ptr<EntityView> createArticulationView(const std::string& pattern) override
    {
        NB_OVERRIDE_NAME("create_articulation_view", createArticulationView, pattern);
    }
    std::shared_ptr<EntityView> createArticulationView(const std::vector<std::string>& patterns) override
    {
        NB_OVERRIDE_NAME("create_articulation_view", createArticulationView, patterns);
    }
    std::shared_ptr<EntityView> createRigidBodyView(const std::string& pattern) override
    {
        NB_OVERRIDE_NAME("create_rigid_body_view", createRigidBodyView, pattern);
    }
    std::shared_ptr<EntityView> createRigidBodyView(const std::vector<std::string>& patterns) override
    {
        NB_OVERRIDE_NAME("create_rigid_body_view", createRigidBodyView, patterns);
    }
    std::shared_ptr<EntityView> createVolumeDeformableBodyView(const std::string& pattern) override
    {
        NB_OVERRIDE_NAME("create_volume_deformable_body_view", createVolumeDeformableBodyView, pattern);
    }
    std::shared_ptr<EntityView> createVolumeDeformableBodyView(const std::vector<std::string>& patterns) override
    {
        NB_OVERRIDE_NAME("create_volume_deformable_body_view", createVolumeDeformableBodyView, patterns);
    }
    std::shared_ptr<EntityView> createSurfaceDeformableBodyView(const std::string& pattern) override
    {
        NB_OVERRIDE_NAME("create_surface_deformable_body_view", createSurfaceDeformableBodyView, pattern);
    }
    std::shared_ptr<EntityView> createSurfaceDeformableBodyView(const std::vector<std::string>& patterns) override
    {
        NB_OVERRIDE_NAME("create_surface_deformable_body_view", createSurfaceDeformableBodyView, patterns);
    }
    std::shared_ptr<EntityView> createDeformableMaterialView(const std::string& pattern) override
    {
        NB_OVERRIDE_NAME("create_deformable_material_view", createDeformableMaterialView, pattern);
    }
    std::shared_ptr<EntityView> createDeformableMaterialView(const std::vector<std::string>& patterns) override
    {
        NB_OVERRIDE_NAME("create_deformable_material_view", createDeformableMaterialView, patterns);
    }
    std::shared_ptr<EntityView> createRigidContactView(const std::string& pattern,
                                                       const std::vector<std::string>& filterPatterns,
                                                       int maximumContactDataCount) override
    {
        NB_OVERRIDE_NAME(
            "create_rigid_contact_view", createRigidContactView, pattern, filterPatterns, maximumContactDataCount);
    }
    std::shared_ptr<EntityView> createSdfShapeView(const std::string& pattern, int numberOfPoints) override
    {
        NB_OVERRIDE_NAME("create_sdf_shape_view", createSdfShapeView, pattern, numberOfPoints);
    }
};
} // namespace

void isaacsim::physics::manager::details::bindTensorViews(nb::module_& module)
{
    nb::class_<EntityView, IEntityView>(module, "EntityView", nb::dynamic_attr(),
                                        "Expose tensor operations for a resolved collection of physics entities.")
        .def(nb::init<>())
        .def(nb::init<std::vector<std::string>>(), nb::arg("paths"))
        .def_prop_ro("paths", &EntityView::getPaths, "Patterns used to create the view.")
        .def_prop_ro("resolved_prim_paths", &EntityView::getResolvedPrimPaths, "USD prim paths resolved by the engine.")
        .def_prop_ro("usd_stage_id", &EntityView::getUsdStageId, "Identifier of the view's USD stage.")
        .def_prop_rw("count", &EntityView::getCount, &EntityView::setCount, "Number of entities in the view.")
        .def(
            "_register_impl",
            [](EntityView& self, const std::string& operationName, ImplKind kind, nb::callable callback,
               TensorSpec specification)
            {
                // Forward to whichever overload of registerImpl matches. The
                // single-tensor variants are the common case; multi-tensor
                // variants are exposed via separate methods below.
                if (kind == ImplKind::eGet)
                {
                    return self.registerImpl(
                        operationName, kind, wrapPythonGetCallback(std::move(callback)), std::move(specification));
                }
                return self.registerImpl(
                    operationName, kind, wrapPythonSetCallback(std::move(callback)), std::move(specification));
            },
            nb::arg("impl"), nb::arg("kind"), nb::arg("fn"), nb::arg("spec") = TensorSpec{},
            "Register a single-buffer tensor operation.")
        .def(
            "_register_multi_impl",
            [](EntityView& self, const std::string& operationName, ImplKind kind, nb::callable callback,
               TensorSpec specification)
            {
                if (kind == ImplKind::eGet)
                {
                    return self.registerImpl(
                        operationName, kind, wrapPythonMultiGetCallback(std::move(callback)), std::move(specification));
                }
                return self.registerImpl(
                    operationName, kind, wrapPythonMultiSetCallback(std::move(callback)), std::move(specification));
            },
            nb::arg("impl"), nb::arg("kind"), nb::arg("fn"), nb::arg("spec") = TensorSpec{},
            "Register a multi-buffer tensor operation.")
        .def(
            "_register_metadata",
            [](EntityView& self, const std::string& operationName, nb::callable callback)
            {
                return self.registerMetadata(operationName,
                                             [callback = std::move(callback)]() -> Metadata
                                             {
                                                 nb::gil_scoped_acquire globalInterpreterLockGuard;
                                                 nb::object metadataObject = callback();
                                                 if (metadataObject.is_none())
                                                 {
                                                     return Metadata{};
                                                 }
                                                 try
                                                 {
                                                     return Metadata{ nb::cast<bool>(metadataObject) };
                                                 }
                                                 catch (const nb::cast_error&)
                                                 {
                                                 }
                                                 try
                                                 {
                                                     return Metadata{ nb::cast<int64_t>(metadataObject) };
                                                 }
                                                 catch (const nb::cast_error&)
                                                 {
                                                 }
                                                 try
                                                 {
                                                     return Metadata{ nb::cast<double>(metadataObject) };
                                                 }
                                                 catch (const nb::cast_error&)
                                                 {
                                                 }
                                                 try
                                                 {
                                                     return Metadata{ nb::cast<std::string>(metadataObject) };
                                                 }
                                                 catch (const nb::cast_error&)
                                                 {
                                                 }
                                                 try
                                                 {
                                                     return Metadata{ nb::cast<std::vector<std::string>>(metadataObject) };
                                                 }
                                                 catch (const nb::cast_error&)
                                                 {
                                                 }
                                                 try
                                                 {
                                                     return Metadata{ nb::cast<std::vector<int64_t>>(metadataObject) };
                                                 }
                                                 catch (const nb::cast_error&)
                                                 {
                                                 }
                                                 return Metadata{};
                                             });
            },
            "Register a metadata provider.")
        .def("list_impls", &EntityView::listImpls, nb::arg("kind"),
             "Return the registered operation names for an operation kind.")
        .def("has_impl", &EntityView::hasImpl, nb::arg("impl"), nb::arg("kind"),
             "Return whether an operation is registered.")
        .def("get_impl_spec", &EntityView::getImplSpec, nb::arg("impl"), nb::arg("kind"),
             "Return the tensor specification for a single-buffer operation.")
        .def("get_impl_spec_multi", &EntityView::getImplSpecMulti, nb::arg("impl"), nb::arg("kind"),
             "Return the tensor specifications for a multi-buffer operation.")
        .def("get_metadata", &EntityView::getMetadata, nb::arg("impl"), "Return metadata supplied by the named provider.")
        .def(
            "get_data",
            [](nb::object self, const std::string& operationName, std::optional<nb::ndarray<>> indices,
               nb::object outputObject) -> nb::object
            {
                EntityView& view = nb::cast<EntityView&>(self);
                TensorDesc indicesDescriptor =
                    indices ? manager_details::convertArrayToTensorDescriptor(*indices) : TensorDesc{};
                manager_details::validateInt32IndexDataType(indicesDescriptor, "get_data");
                const bool hasOutput = !outputObject.is_none();
                TensorDesc outputDescriptor =
                    hasOutput ? manager_details::convertArrayToTensorDescriptor(nb::cast<nb::ndarray<>>(outputObject)) :
                                TensorDesc{};
                TensorDesc resultDescriptor = view.getData(operationName, indicesDescriptor, outputDescriptor);
                if (resultDescriptor.isEmpty())
                {
                    return nb::none();
                }
                // Hand back a generic DLPack-typed ndarray; the Python wrapper at
                // the public surface (`bindings/__init__.py`) dispatches through
                // `frontend.wrap_tensor(...)` to the caller's native tensor type.
                //
                // Retain the owning Python object so the returned view can't dangle:
                // the caller's `out` when the result aliases it (explicit-output
                // path), otherwise the EntityView itself -- an internal result
                // aliases a buffer owned by the view's stored impl closure, freed
                // with the view.
                nb::object owner;
                if (resultDescriptor.keepalive)
                {
                    // The result owns its buffer (per-call storage); pin it via a capsule
                    // so the array frees the buffer when dropped, independent of the view.
                    std::shared_ptr<void>* keepaliveHolder = new std::shared_ptr<void>(resultDescriptor.keepalive);
                    owner = nb::capsule(keepaliveHolder, [](void* pointer) noexcept
                                        { delete static_cast<std::shared_ptr<void>*>(pointer); });
                }
                else
                {
                    owner = (hasOutput && resultDescriptor.data == outputDescriptor.data) ? outputObject : self;
                }
                return nb::cast(manager_details::convertTensorDescriptorToArray(resultDescriptor, owner));
            },
            nb::arg("impl"), nb::arg("indices") = nb::none(), nb::arg("out") = nb::none(),
            "Read a tensor operation, optionally for selected entity indices.")
        .def(
            "set_data",
            [](EntityView& self, const std::string& operationName, nb::ndarray<> data, std::optional<nb::ndarray<>> indices)
            {
                TensorDesc dataDescriptor = manager_details::convertArrayToTensorDescriptor(data);
                TensorDesc indicesDescriptor =
                    indices ? manager_details::convertArrayToTensorDescriptor(*indices) : TensorDesc{};
                manager_details::validateInt32IndexDataType(indicesDescriptor, "set_data");
                self.setData(operationName, dataDescriptor, indicesDescriptor);
            },
            nb::arg("impl"), nb::arg("data"), nb::arg("indices") = nb::none(),
            "Write a tensor operation, optionally for selected entity indices.")
        .def(
            "get_data_multi",
            [](nb::object self, const std::string& operationName, std::optional<nb::ndarray<>> indices,
               std::vector<nb::object> outputObjects)
            {
                EntityView& view = nb::cast<EntityView&>(self);
                TensorDesc indicesDescriptor =
                    indices ? manager_details::convertArrayToTensorDescriptor(*indices) : TensorDesc{};
                manager_details::validateInt32IndexDataType(indicesDescriptor, "get_data_multi");
                std::vector<TensorDesc> outputDescriptors;
                outputDescriptors.reserve(outputObjects.size());
                for (const nb::object& output : outputObjects)
                {
                    outputDescriptors.push_back(output.is_none() ? TensorDesc{} :
                                                                   manager_details::convertArrayToTensorDescriptor(
                                                                       nb::cast<nb::ndarray<>>(output)));
                }
                std::vector<TensorDesc> resultDescriptors =
                    view.getDataMulti(operationName, indicesDescriptor, outputDescriptors);
                nb::list results;
                for (const TensorDesc& descriptor : resultDescriptors)
                {
                    if (descriptor.isEmpty())
                    {
                        results.append(nb::none());
                        continue;
                    }
                    // Retain the backing so a returned array can't outlive its
                    // storage: a per-call owner (from wrapPythonMultiGetCallback) pinned via a
                    // capsule, else the matching supplied `out` when the result
                    // aliases it, else the EntityView (an internal result aliases a
                    // buffer owned by the view's impl closure).
                    nb::object owner;
                    if (descriptor.keepalive)
                    {
                        std::shared_ptr<void>* keepaliveHolder = new std::shared_ptr<void>(descriptor.keepalive);
                        owner = nb::capsule(keepaliveHolder, [](void* pointer) noexcept
                                            { delete static_cast<std::shared_ptr<void>*>(pointer); });
                    }
                    else
                    {
                        owner = self;
                        for (size_t outputIndex = 0; outputIndex < outputObjects.size(); ++outputIndex)
                        {
                            if (!outputObjects[outputIndex].is_none() &&
                                outputDescriptors[outputIndex].data == descriptor.data)
                            {
                                owner = outputObjects[outputIndex];
                                break;
                            }
                        }
                    }
                    results.append(nb::cast(manager_details::convertTensorDescriptorToArray(descriptor, owner)));
                }
                return results;
            },
            nb::arg("impl"), nb::arg("indices") = nb::none(), nb::arg("out") = std::vector<nb::object>{},
            "Read a multi-buffer tensor operation.")
        .def(
            "set_data_multi",
            [](EntityView& self, const std::string& operationName, std::vector<nb::ndarray<>> data,
               std::optional<nb::ndarray<>> indices)
            {
                std::vector<TensorDesc> dataDescriptors;
                dataDescriptors.reserve(data.size());
                for (const nb::ndarray<>& array : data)
                {
                    dataDescriptors.push_back(manager_details::convertArrayToTensorDescriptor(array));
                }
                TensorDesc indicesDescriptor =
                    indices ? manager_details::convertArrayToTensorDescriptor(*indices) : TensorDesc{};
                manager_details::validateInt32IndexDataType(indicesDescriptor, "set_data_multi");
                self.setDataMulti(operationName, dataDescriptors, indicesDescriptor);
            },
            nb::arg("impl"), nb::arg("data"), nb::arg("indices") = nb::none(), "Write a multi-buffer tensor operation.");

    // ----- SimulationView ----------------------------------------------

    nb::class_<SimulationView, ISimulationView, PythonSimulationView>(
        module, "SimulationView", nb::dynamic_attr(), "Scene-level handle to an engine's tensor data plane.")
        .def(nb::init<>())
        .def(nb::init<std::string, std::string, int64_t>(), nb::arg("engine"), nb::arg("frontend_name"),
             nb::arg("stage_id"))
        .def_prop_ro("engine", &SimulationView::getEngine, "Name of the physics engine.")
        .def_prop_ro("frontend_name", &SimulationView::getFrontendName, "Name of the tensor frontend.")
        .def_prop_ro("stage_id", &SimulationView::getStageId, "Identifier of the USD stage.")
        .def("get_device_ordinal", &SimulationView::getDeviceOrdinal, "Return the ordinal of the simulation device.")
        .def("get_param_device_ordinal", &SimulationView::getParameterDeviceOrdinal,
             "Return the ordinal of the parameter device.")
        .def(
            "is_valid", [](const SimulationView& self) { return self.SimulationView::isValid(); },
            "Whether the simulation view is valid.")
        .def(
            "invalidate", [](SimulationView& self) { self.SimulationView::invalidate(); },
            "Invalidate the simulation view.")
        .def("set_device_ordinal", &SimulationView::setDeviceOrdinal, nb::arg("ord"), "Set the simulation device ordinal.")
        .def("set_param_device_ordinal", &SimulationView::setParameterDeviceOrdinal, nb::arg("ord"),
             "Set the parameter device ordinal.")
        .def("set_valid", &SimulationView::setValid, nb::arg("valid"), "Set whether the simulation view is valid.")
        // Accept either a Float3 or any 3-element sequence (list / tuple /
        // numpy array) so legacy tests calling `sim.set_gravity([x, y, z])`
        // work unchanged alongside callers that pass `Float3(x, y, z)`.
        // The base-class implementation is invoked directly to bypass the
        // PythonSimulationView trampoline (which would try to look up a Python
        // override that this very lambda is the only candidate for, looping).
        .def(
            "set_gravity",
            [](SimulationView& self, nb::object gravity)
            {
                Float3 gravityVector;
                try
                {
                    gravityVector = nb::cast<Float3>(gravity);
                }
                catch (const nb::cast_error&)
                {
                    auto sequence = nb::cast<nb::sequence>(gravity);
                    if (nb::len(sequence) != 3)
                    {
                        throw std::invalid_argument("set_gravity: expected Float3 or 3-element sequence");
                    }
                    gravityVector = Float3{
                        nb::cast<float>(sequence[0]),
                        nb::cast<float>(sequence[1]),
                        nb::cast<float>(sequence[2]),
                    };
                }
                self.SimulationView::setGravity(gravityVector);
            },
            nb::arg("gravity"), "Set the world-space gravity vector.")
        .def(
            "get_gravity", [](const SimulationView& self) { return self.SimulationView::getGravity(); },
            "Return the world-space gravity vector.")
        .def("clear_forces", &SimulationView::clearForces, "Clear externally applied forces.")
        .def("step", &SimulationView::step, nb::arg("dt"), "Advance this simulation view by one time step.")
        .def("update_articulations_kinematic", &SimulationView::updateArticulationsKinematic,
             "Update articulation kinematic state.")
        .def("initialize_kinematic_bodies", &SimulationView::initializeKinematicBodies,
             "Initialize kinematic rigid bodies.")
        .def("get_object_type", &SimulationView::getObjectType, nb::arg("prim_path"),
             R"doc(Classify a physics object.

Args:
    prim_path: Absolute path of the prim to classify.

Returns:
    The object type, or ``ObjectType.Invalid`` when the engine cannot classify
    the path.
)doc")
        .def("create_articulation_view", nb::overload_cast<const std::string&>(&SimulationView::createArticulationView),
             nb::arg("pattern"), "Create an articulation view from one path pattern.")
        .def("create_articulation_view",
             nb::overload_cast<const std::vector<std::string>&>(&SimulationView::createArticulationView),
             nb::arg("patterns"), "Create an articulation view from path patterns.")
        .def("create_rigid_body_view", nb::overload_cast<const std::string&>(&SimulationView::createRigidBodyView),
             nb::arg("pattern"), "Create a rigid-body view from one path pattern.")
        .def("create_rigid_body_view",
             nb::overload_cast<const std::vector<std::string>&>(&SimulationView::createRigidBodyView),
             nb::arg("patterns"), "Create a rigid-body view from path patterns.")
        .def("create_volume_deformable_body_view",
             nb::overload_cast<const std::string&>(&SimulationView::createVolumeDeformableBodyView), nb::arg("pattern"),
             "Create a volume-deformable-body view from one path pattern.")
        .def("create_volume_deformable_body_view",
             nb::overload_cast<const std::vector<std::string>&>(&SimulationView::createVolumeDeformableBodyView),
             nb::arg("patterns"), "Create a volume-deformable-body view from path patterns.")
        .def("create_surface_deformable_body_view",
             nb::overload_cast<const std::string&>(&SimulationView::createSurfaceDeformableBodyView),
             nb::arg("pattern"), "Create a surface-deformable-body view from one path pattern.")
        .def("create_surface_deformable_body_view",
             nb::overload_cast<const std::vector<std::string>&>(&SimulationView::createSurfaceDeformableBodyView),
             nb::arg("patterns"), "Create a surface-deformable-body view from path patterns.")
        .def("create_deformable_material_view",
             nb::overload_cast<const std::string&>(&SimulationView::createDeformableMaterialView), nb::arg("pattern"),
             "Create a deformable-material view from one path pattern.")
        .def("create_deformable_material_view",
             nb::overload_cast<const std::vector<std::string>&>(&SimulationView::createDeformableMaterialView),
             nb::arg("patterns"), "Create a deformable-material view from path patterns.")
        .def("create_rigid_contact_view", &SimulationView::createRigidContactView, nb::arg("pattern"),
             nb::arg("filter_patterns") = std::vector<std::string>{}, nb::arg("max_contact_data_count") = 0,
             "Create a rigid-contact view.")
        .def("create_sdf_shape_view", &SimulationView::createSdfShapeView, nb::arg("pattern"), nb::arg("num_points"),
             "Create a signed-distance-field shape view.");
}
