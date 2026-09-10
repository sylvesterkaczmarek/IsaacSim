// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "isaacsim/common/array/nanobind/ArrayCaster.hpp"
#include "isaacsim/foundation/ovsim/control/authoring/Authoring.hpp"
#include "isaacsim/foundation/ovsim/control/simulation/Simulation.hpp"
#include "isaacsim/foundation/ovsim/data/Data.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace isaacsim::foundation::ovsim;

NB_MODULE(_bindings, m)
{
    nb::module_ m_data = m.def_submodule("data");
    nb::module_ m_control = m.def_submodule("control");
    nb::module_ m_authoring = m_control.def_submodule("authoring");
    nb::module_ m_simulation = m_control.def_submodule("simulation");

    // control.authoring
    // - Stage operations
    m_authoring.def("create_stage", &control::authoring::createStage);
    m_authoring.def("open_stage", &control::authoring::openStage, nb::arg("usd_path"));
    m_authoring.def("save_stage", &control::authoring::saveStage, nb::arg("usd_path"));
    m_authoring.def("import_stage_from_string", &control::authoring::importStageFromString, nb::arg("usd_string"));
    m_authoring.def("export_stage_to_string", &control::authoring::exportStageToString);
    m_authoring.def("close_stage", &control::authoring::closeStage);
    m_authoring.def("add_reference_to_stage", &control::authoring::addReferenceToStage, nb::arg("usd_path"),
                    nb::arg("path"), nb::arg("type_name") = "Xform");
    // - Prim operations
    m_authoring.def("define_prim", &control::authoring::definePrim, nb::arg("path"), nb::arg("type_name") = "Xform");
    m_authoring.def("move_prim", &control::authoring::movePrim, nb::arg("target_path"), nb::arg("destination_path"));
    m_authoring.def("remove_prim", &control::authoring::removePrim, nb::arg("path"));
    // - Attribute operations
    m_authoring.def("create_prim_attribute", &control::authoring::createPrimAttribute, nb::arg("path"),
                    nb::arg("attribute_name"), nb::arg("type_name"));
    m_authoring.def(
        "remove_prim_attribute", &control::authoring::removePrimAttribute, nb::arg("path"), nb::arg("attribute_name"));
    // - Parameters
    m_authoring.def("set_parameter", &control::authoring::setParameter, nb::arg("provider"), nb::arg("parameter_name"),
                    nb::arg("value"));
    m_authoring.def("get_parameter", &control::authoring::getParameter, nb::arg("provider"), nb::arg("parameter_name"));

    // control.simulation
    // - Lifecycle operations
    m_simulation.def("play", &control::simulation::play);
    m_simulation.def("pause", &control::simulation::pause);
    m_simulation.def("stop", &control::simulation::stop);
    m_simulation.def("initialize", &control::simulation::initialize);
    m_simulation.def("invalidate", &control::simulation::invalidate);
    m_simulation.def("step", &control::simulation::step);
    // - Parameters
    m_simulation.def("set_parameter", &control::simulation::setParameter, nb::arg("provider"),
                     nb::arg("parameter_name"), nb::arg("value"));
    m_simulation.def("get_parameter", &control::simulation::getParameter, nb::arg("provider"), nb::arg("parameter_name"));

    // data
    m_data.def("read", &data::read, nb::arg("paths"), nb::arg("attribute_name"), nb::arg("timestamp") = nb::none());
    m_data.def("write", &data::write, nb::arg("paths"), nb::arg("attribute_name"), nb::arg("values"),
               nb::arg("timestamp") = nb::none());
}
