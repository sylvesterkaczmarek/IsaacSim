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
#include "isaacsim/ovsim/api/Factory.hpp"
#include "isaacsim/ovsim/api/Types.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace isaacsim::ovsim::api;

NB_MODULE(_bindings, m)
{
    nb::class_<types::Authoring>(m, "Authoring")
        // - Stage operations
        .def("create_stage", [](const types::Authoring& self) { return self.createStage(); })
        .def(
            "open_stage",
            [](const types::Authoring& self, const std::string& usdPath) { return self.openStage(usdPath); },
            nb::arg("usd_path"))
        .def(
            "save_stage",
            [](const types::Authoring& self, const std::string& usdPath) { return self.saveStage(usdPath); },
            nb::arg("usd_path"))
        .def(
            "import_stage_from_string",
            [](const types::Authoring& self, const std::string& usdString)
            { return self.importStageFromString(usdString); },
            nb::arg("usd_string"))
        .def("export_stage_to_string", [](const types::Authoring& self) { return self.exportStageToString(); })
        .def("close_stage", [](const types::Authoring& self) { return self.closeStage(); })
        .def(
            "add_reference_to_stage",
            [](const types::Authoring& self, const std::string& usdPath, const std::string& path,
               const std::string& typeName) { return self.addReferenceToStage(usdPath, path, typeName); },
            nb::arg("usd_path"), nb::arg("path"), nb::arg("type_name") = "Xform")
        // - Prim operations
        .def(
            "define_prim",
            [](const types::Authoring& self, const std::string& path, const std::string& typeName)
            { return self.definePrim(path, typeName); },
            nb::arg("path"), nb::arg("type_name") = "Xform")
        .def(
            "move_prim",
            [](const types::Authoring& self, const std::string& targetPath, const std::string& destinationPath)
            { return self.movePrim(targetPath, destinationPath); },
            nb::arg("target_path"), nb::arg("destination_path"))
        .def(
            "remove_prim", [](const types::Authoring& self, const std::string& path) { return self.removePrim(path); },
            nb::arg("path"))
        // - Attribute operations
        .def(
            "create_prim_attribute",
            [](const types::Authoring& self, const std::string& path, const std::string& attributeName,
               const std::string& typeName) { return self.createPrimAttribute(path, attributeName, typeName); },
            nb::arg("path"), nb::arg("attribute_name"), nb::arg("type_name"))
        .def(
            "remove_prim_attribute",
            [](const types::Authoring& self, const std::string& path, const std::string& attributeName)
            { return self.removePrimAttribute(path, attributeName); },
            nb::arg("path"), nb::arg("attribute_name"))
        // - Parameters
        .def(
            "set_parameter",
            [](const types::Authoring& self, const std::string& provider, const std::string& parameterName,
               const ::ovsim::interfaces::control::authoring::InputParameterType& value)
            { return self.setParameter(provider, parameterName, value); },
            nb::arg("provider"), nb::arg("parameter_name"), nb::arg("value"))
        .def(
            "get_parameter",
            [](const types::Authoring& self, const std::string& provider, const std::string& parameterName)
            { return self.getParameter(provider, parameterName); },
            nb::arg("provider"), nb::arg("parameter_name"));

    nb::class_<types::Simulation>(m, "Simulation")
        // - Lifecycle operations
        .def("play", [](const types::Simulation& self) { return self.play(); })
        .def("pause", [](const types::Simulation& self) { return self.pause(); })
        .def("stop", [](const types::Simulation& self) { return self.stop(); })
        .def("initialize", [](const types::Simulation& self) { return self.initialize(); })
        .def("invalidate", [](const types::Simulation& self) { return self.invalidate(); })
        .def("step", [](const types::Simulation& self) { return self.step(); })
        // - Parameters
        .def(
            "set_parameter",
            [](const types::Simulation& self, const std::string& provider, const std::string& parameterName,
               const ::ovsim::interfaces::control::simulation::InputParameterType& value)
            { return self.setParameter(provider, parameterName, value); },
            nb::arg("provider"), nb::arg("parameter_name"), nb::arg("value"))
        .def(
            "get_parameter",
            [](const types::Simulation& self, const std::string& provider, const std::string& parameterName)
            { return self.getParameter(provider, parameterName); },
            nb::arg("provider"), nb::arg("parameter_name"));

    nb::class_<types::Control>(m, "Control")
        .def_ro("authoring", &types::Control::authoring)
        .def_ro("simulation", &types::Control::simulation);

    nb::class_<types::Data>(m, "Data")
        .def(
            "read",
            [](const types::Data& self, const ::ovsim::interfaces::data::PathType& paths,
               const std::string& attributeName, std::optional<double> timeStamp)
            { return self.read(paths, attributeName, timeStamp); },
            nb::arg("paths"), nb::arg("attribute_name"), nb::arg("timestamp") = nb::none())
        .def(
            "write",
            [](const types::Data& self, const ::ovsim::interfaces::data::PathType& paths,
               const std::string& attributeName, const ::ovsim::interfaces::data::InputValueType& values,
               std::optional<double> timeStamp) { return self.write(paths, attributeName, values, timeStamp); },
            nb::arg("paths"), nb::arg("attribute_name"), nb::arg("values"), nb::arg("timestamp") = nb::none());

    nb::class_<types::Implementation>(m, "Implementation")
        .def_ro("control", &types::Implementation::control)
        .def_ro("data", &types::Implementation::data);

    m.def("make_client", &makeClient, nb::arg("name"), nb::arg("configuration") = nb::none());
}
