// SPDX-FileCopyrightText: Copyright (c) 2018-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "NanobindHelpers.hpp"
#include "isaacsim/physics/manager/PhysicsEvent.hpp"
#include "isaacsim/physics/manager/PhysicsManager.hpp"

#include <isaacsim/physics/registration/Physics.hpp>
#include <isaacsim/physics/registration/simulator/ContactEvent.hpp>
#include <isaacsim/physics/registration/simulator/Simulation.hpp>
#include <isaacsim/physics/registration/simulator/Simulator.hpp>

namespace nb = nanobind;
using namespace isaacsim::physics::manager;

NB_MODULE(_bindings, module)
{
    module.doc() = "Manage physics simulations, scene queries, interactions, profiling, and tensor views.";

    // PhysicsEvent.hpp
    nb::enum_<PhysicsEvent>(module, "PhysicsEvent")
        .value("PHYSICS_PRE_STEP", PhysicsEvent::ePhysicsPreStep)
        .value("PHYSICS_POST_STEP", PhysicsEvent::ePhysicsPostStep)
        .value("PHYSICS_SETUP", PhysicsEvent::ePhysicsSetup)
        .value("PHYSICS_INITIALIZED", PhysicsEvent::ePhysicsInitialized)
        .value("PHYSICS_INVALIDATED", PhysicsEvent::ePhysicsInvalidated);

    // PhysicsManager.hpp
    nb::class_<PhysicsManager>(module, "PhysicsManager")
        .def_static("get_instance", &PhysicsManager::getInstance, nb::rv_policy::reference)
        .def("is_initialized", &PhysicsManager::isInitialized)
        .def("setup", &PhysicsManager::setup, nb::arg("dt") = 1.0f / 60.0f)
        .def(
            "initialize",
            [](PhysicsManager& self, uintptr_t ovstage, int64_t usdStageId)
            { return self.initialize(reinterpret_cast<void*>(ovstage), usdStageId); },
            nb::arg("ovstage_instance"), nb::arg("usd_stage_id"))
        .def("invalidate", &PhysicsManager::invalidate)
        .def("step", &PhysicsManager::step, nb::kw_only(), nb::arg("steps") = 1, nb::arg("callback") = nb::none())
        .def("get_simulated_time", &PhysicsManager::getSimulatedTime)
        .def("get_simulated_physics_steps", &PhysicsManager::getSimulatedPhysicsSteps)
        .def("register_callback", &PhysicsManager::registerCallback, nb::arg("callback"), nb::arg("event"),
             nb::kw_only(), nb::arg("order") = 0)
        .def("deregister_callback", &PhysicsManager::deregisterCallback, nb::arg("uid"))
        .def("deregister_all_callbacks", &PhysicsManager::deregisterAllCallbacks)
        .def("get_registered_physics_engines", &PhysicsManager::getRegisteredPhysicsEngines)
        .def("switch_physics_engine", &PhysicsManager::switchPhysicsEngine, nb::arg("engine"))
        .def("create_entity", &PhysicsManager::createEntity, nb::kw_only(), nb::arg("engine"), nb::arg("entity"),
             nb::arg("paths"));

    // Register the shared vocabulary + engine register API first so the C++
    // types the use interfaces reference (Float3, Subscription, SimulationView,
    // ...) are registered by the time this module binds against them.
    nb::module_::import_("isaacsim.physics.registration.bindings._bindings");

    bindPythonSubscription(module);
    details::bindPhysicsSceneQuery(module);
    details::bindPhysicsInteraction(module);
    details::bindPhysicsBenchmarks(module);
    details::bindPhysicsSimulation(module);
    details::bindTensorViews(module);
    details::bindTensorCreate(module);
}
