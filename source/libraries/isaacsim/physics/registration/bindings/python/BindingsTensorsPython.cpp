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

#include "TensorNbHelpers.hpp"

#include <isaacsim/physics/registration/tensors/IEntityView.hpp>
#include <isaacsim/physics/registration/tensors/ISimulationView.hpp>
#include <isaacsim/physics/registration/tensors/Metadata.hpp>
#include <isaacsim/physics/registration/tensors/PhysicsEnums.hpp>
#include <isaacsim/physics/registration/tensors/TensorDesc.hpp>
#include <isaacsim/physics/registration/tensors/TensorRegistry.hpp>
#include <isaacsim/physics/registration/tensors/TensorSpec.hpp>
#include <isaacsim/physics/registration/tensors/TensorTypes.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

namespace isaacsim
{
namespace physics
{
namespace registration
{
namespace details
{

using tensors::DeviceKind;
using tensors::DofDriveType;
using tensors::DofMotion;
using tensors::DofType;
using tensors::DType;
using tensors::EntityFactory;
using tensors::IEntityView;
using tensors::ImplKind;
using tensors::ISimulationView;
using tensors::JointType;
using tensors::ObjectType;
using tensors::SimulationViewFactory;
using tensors::TensorDesc;
using tensors::TensorRegistry;
using tensors::TensorSpec;

void bindTensors(nb::module_& module)
{
    nb::enum_<DType>(module, "DType", "Element types supported by the tensor API.")
        .value("UNKNOWN", DType::eUnknown)
        .value("FLOAT32", DType::eFloat32)
        .value("FLOAT64", DType::eFloat64)
        .value("INT8", DType::eInt8)
        .value("INT16", DType::eInt16)
        .value("INT32", DType::eInt32)
        .value("INT64", DType::eInt64)
        .value("UINT8", DType::eUInt8)
        .value("UINT16", DType::eUInt16)
        .value("UINT32", DType::eUInt32)
        .value("UINT64", DType::eUInt64)
        .value("BOOL", DType::eBool)
        .export_values();

    nb::enum_<DeviceKind>(module, "DeviceKind", "Device-placement requirements for a tensor implementation.")
        .value("CPU", DeviceKind::eCpu)
        .value("GPU", DeviceKind::eGpu)
        .value("ENGINE_DEFAULT", DeviceKind::eEngineDefault)
        .export_values();

    nb::enum_<ImplKind>(module, "ImplKind", "Operation kinds implemented by an entity view.")
        .value("Get", ImplKind::eGet)
        .value("Set", ImplKind::eSet)
        .export_values();

    // Physics-domain enums shared by tensor backends.

    nb::enum_<ObjectType>(module, "ObjectType", "Object types in the physics scene.")
        .value("Invalid", ObjectType::eInvalid)
        .value("RigidBody", ObjectType::eRigidBody)
        .value("Articulation", ObjectType::eArticulation)
        .value("ArticulationLink", ObjectType::eArticulationLink)
        .value("ArticulationRootLink", ObjectType::eArticulationRootLink)
        .value("ArticulationJoint", ObjectType::eArticulationJoint)
        .export_values();

    nb::enum_<JointType>(module, "JointType", "Joint types.")
        .value("Invalid", JointType::eInvalid)
        .value("Fixed", JointType::eFixed)
        .value("Revolute", JointType::eRevolute)
        .value("Prismatic", JointType::ePrismatic)
        .value("Spherical", JointType::eSpherical)
        .export_values();

    nb::enum_<DofType>(module, "DofType", "DOF types.")
        .value("Invalid", DofType::eInvalid)
        .value("Rotation", DofType::eRotation)
        .value("Translation", DofType::eTranslation)
        .export_values();

    nb::enum_<DofMotion>(module, "DofMotion", "DOF motion types.")
        .value("Invalid", DofMotion::eInvalid)
        .value("Free", DofMotion::eFree)
        .value("Limited", DofMotion::eLimited)
        .value("Locked", DofMotion::eLocked)
        .export_values();

    nb::enum_<DofDriveType>(module, "DofDriveType", "DOF drive types.")
        .value("None_", DofDriveType::eNone)
        .value("Force", DofDriveType::eForce)
        .value("Acceleration", DofDriveType::eAcceleration)
        .export_values();

    // Float3 is registered once by the physics bindings (manager::Float3); the
    // tensor data plane's `Float3` aliases it, so it is not re-registered here.

    // ----- TensorSpec ---------------------------------------------------

    nb::class_<TensorSpec>(module, "TensorSpec", "Describe a tensor operation's shape, type, placement, and capabilities.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](TensorSpec* self, DType dtype, std::vector<int64_t> shapeHint, DeviceKind deviceKind, bool supports,
               bool supportsIndexedRead, bool supportsIndexedWrite, bool supportsMaskedWrite)
            {
                new (self) TensorSpec{};
                self->dtype = dtype;
                self->shapeHint = std::move(shapeHint);
                self->deviceKind = deviceKind;
                self->supports = supports;
                self->supportsIndexedRead = supportsIndexedRead;
                self->supportsIndexedWrite = supportsIndexedWrite;
                self->supportsMaskedWrite = supportsMaskedWrite;
            },
            nb::arg("dtype") = DType::eFloat32, nb::arg("shape_hint") = std::vector<int64_t>{},
            nb::arg("device_kind") = DeviceKind::eEngineDefault, nb::arg("supports") = true,
            nb::arg("supports_indexed_read") = false, nb::arg("supports_indexed_write") = false,
            nb::arg("supports_masked_write") = false)
        .def_rw("dtype", &TensorSpec::dtype)
        .def_rw("shape_hint", &TensorSpec::shapeHint)
        .def_rw("device_kind", &TensorSpec::deviceKind)
        .def_rw("supports", &TensorSpec::supports)
        .def_rw("supports_indexed_read", &TensorSpec::supportsIndexedRead)
        .def_rw("supports_indexed_write", &TensorSpec::supportsIndexedWrite)
        .def_rw("supports_masked_write", &TensorSpec::supportsMaskedWrite)
        .def_rw("requires_host_data", &TensorSpec::requiresHostData);

    // ----- TensorDesc ---------------------------------------------------
    //
    // Constructible from a Python ndarray so engines that need to forward an
    // ndarray as a TensorDesc (e.g. for indexed-read fixtures) can do so.

    nb::class_<TensorDesc>(module, "TensorDesc", "Describe a tensor buffer and retain its backing storage.")
        .def(nb::init<>())
        .def(
            "__init__",
            [](TensorDesc* self, nb::ndarray<> array)
            {
                // Retain the wrapped array so a TensorDesc built from a temporary
                // (e.g. `TensorDesc(np.array(...))` returned by a get implementation)
                // keeps its storage alive, not just the raw pointer.
                TensorDesc descriptor = arrayToTensorDescriptor(array);
                if (!descriptor.isEmpty())
                {
                    descriptor.keepalive = createArrayKeepAlive(std::move(array));
                }
                new (self) TensorDesc(std::move(descriptor));
            },
            nb::arg("array"))
        .def_prop_ro("dtype", [](const TensorDesc& descriptor) { return descriptor.dtype; })
        .def_prop_ro("shape", [](const TensorDesc& descriptor) { return descriptor.shape; })
        .def_prop_ro("strides", [](const TensorDesc& descriptor) { return descriptor.strides; })
        .def_prop_ro("device", [](const TensorDesc& descriptor) { return descriptor.device; })
        .def_prop_ro("device_ordinal", [](const TensorDesc& descriptor) { return descriptor.deviceOrdinal; })
        .def_prop_ro("num_elements", [](const TensorDesc& descriptor) { return descriptor.computeElementCount(); })
        .def_prop_ro("is_empty", [](const TensorDesc& descriptor) { return descriptor.isEmpty(); })
        .def(
            "to_array",
            [](const TensorDesc& descriptor)
            {
                if (descriptor.isEmpty())
                {
                    return nb::object(nb::none());
                }
                // Retain the descriptor's backing so the returned array can't dangle:
                // a TensorDesc built from a temporary (`TensorDesc(np.array(...))`)
                // holds its source in `keepalive`; pin it via a capsule so the array
                // owns the storage instead of aliasing a buffer that dies with the
                // temporary descriptor. Falls back to no owner when the descriptor carries
                // no keepalive (engine-owned storage), matching prior behavior.
                nb::object owner = nb::none();
                if (descriptor.keepalive)
                {
                    auto* retainedOwner = new std::shared_ptr<void>(descriptor.keepalive);
                    owner = nb::capsule(retainedOwner, [](void* pointer) noexcept
                                        { delete static_cast<std::shared_ptr<void>*>(pointer); });
                }
                return nb::cast(tensorDescriptorToNumpy(descriptor, owner));
            },
            "Return an array view of the tensor buffer, or None when empty.");

    // ----- EntityView ---------------------------------------------------


    // Register-side base classes engines implement; the concrete views deriving
    // these are bound in the manager module.
    nb::class_<ISimulationView>(module, "ISimulationView", "Base interface implemented by engine simulation views.");
    nb::class_<IEntityView>(module, "IEntityView", "Base interface implemented by engine entity views.");

    nb::class_<TensorRegistry>(
        module, "TensorRegistry", "Process-wide registry that maps (engine, entity-name) to view factories.")
        .def_static("instance", &TensorRegistry::getInstance, nb::rv_policy::reference,
                    "Return the process-wide tensor registry.")
        .def(
            "register_entity",
            [](TensorRegistry& self, const std::string& engine, const std::string& entityName, nb::callable factory)
            {
                EntityFactory entityFactory = [factory = std::move(factory)](
                                                  const std::vector<std::string>& paths) -> std::shared_ptr<IEntityView>
                {
                    nb::gil_scoped_acquire globalInterpreterLockAcquire;
                    nb::object object = factory(paths);
                    if (object.is_none())
                    {
                        return nullptr;
                    }
                    return nb::cast<std::shared_ptr<IEntityView>>(object);
                };
                return self.registerEntity(engine, entityName, std::move(entityFactory));
            },
            nb::arg("engine"), nb::arg("entity_name"), nb::arg("factory"),
            "Register an entity-view factory for an engine.")
        .def(
            "register_simulation_view",
            [](TensorRegistry& self, const std::string& engine, nb::callable factory)
            {
                SimulationViewFactory simulationViewFactory = [factory = std::move(factory)](
                                                                  const std::string& frontendName,
                                                                  int64_t stageId) -> std::shared_ptr<ISimulationView>
                {
                    nb::gil_scoped_acquire globalInterpreterLockAcquire;
                    nb::object object = factory(frontendName, stageId);
                    if (object.is_none())
                    {
                        return nullptr;
                    }
                    return nb::cast<std::shared_ptr<ISimulationView>>(object);
                };
                return self.registerSimulationView(engine, std::move(simulationViewFactory));
            },
            nb::arg("engine"), nb::arg("factory"), "Register a simulation-view factory for an engine.")
        .def("register_engine", &TensorRegistry::registerEngine, nb::arg("engine"),
             "Declare that a physics engine is available. The declaration lasts for the life of the process; it "
             "describes what can be simulated, not what is being simulated.")
        .def("list_simulations", &TensorRegistry::listSimulations,
             "Return the simulation names that have registered factories, which is what create_entity and "
             "create_simulation_view accept.")
        .def("unregister_entity", &TensorRegistry::unregisterEntity, nb::arg("engine"), nb::arg("entity_name"),
             "Unregister an entity-view factory.")
        .def("unregister_simulation_view", &TensorRegistry::unregisterSimulationView, nb::arg("engine"),
             "Unregister a simulation-view factory.")
        .def("has_entity", &TensorRegistry::hasEntity, nb::arg("engine"), nb::arg("entity_name"),
             "Return whether an engine has an entity-view factory.")
        .def("list_engines", &TensorRegistry::listEngines, "Return the names of the declared physics engines.")
        .def("list_entities", &TensorRegistry::listEntities, nb::arg("engine"),
             "Return the entity-view types registered for an engine.")
        .def("clear_for_testing", &TensorRegistry::clearForTesting, "Remove all registered tensor factories.");

    module.def(
        "get_registry", []() -> TensorRegistry& { return TensorRegistry::getInstance(); }, nb::rv_policy::reference,
        "Return the process-wide tensor registry.");
    module.def(
        "register_entity",
        [](const std::string& engine, const std::string& entityName, nb::callable factory)
        {
            EntityFactory entityFactory =
                [factory = std::move(factory)](const std::vector<std::string>& paths) -> std::shared_ptr<IEntityView>
            {
                nb::gil_scoped_acquire globalInterpreterLockAcquire;
                nb::object object = factory(paths);
                if (object.is_none())
                {
                    return nullptr;
                }
                return nb::cast<std::shared_ptr<IEntityView>>(object);
            };
            return TensorRegistry::getInstance().registerEntity(engine, entityName, std::move(entityFactory));
        },
        nb::arg("engine"), nb::arg("entity_name"), nb::arg("factory"), "Register an entity-view factory for an engine.");
    module.def(
        "register_simulation_view",
        [](const std::string& engine, nb::callable factory)
        {
            SimulationViewFactory simulationViewFactory = [factory = std::move(factory)](
                                                              const std::string& frontendName,
                                                              int64_t stageId) -> std::shared_ptr<ISimulationView>
            {
                nb::gil_scoped_acquire globalInterpreterLockAcquire;
                nb::object object = factory(frontendName, stageId);
                if (object.is_none())
                {
                    return nullptr;
                }
                return nb::cast<std::shared_ptr<ISimulationView>>(object);
            };
            return TensorRegistry::getInstance().registerSimulationView(engine, std::move(simulationViewFactory));
        },
        nb::arg("engine"), nb::arg("factory"), "Register a simulation-view factory for an engine.");
    // create_entity / create_simulation_view (the use side) live in the manager
    // module's bindings (BindingsTensorCreate.cpp) per the register/use split.
}

} // namespace details
} // namespace registration
} // namespace physics
} // namespace isaacsim
