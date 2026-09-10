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

#include <carb/BindingsPythonUtils.h>

#include <isaacsim/ros2/control/IRos2Control.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

CARB_BINDINGS("isaacsim.ros2.control.python")

namespace py = pybind11;
using isaacsim::ros2::control::Ros2Control;

namespace
{

int py_setup_cm(const std::string& articulation_path,
                const std::string& urdf_xml,
                const std::string& controller_yaml_path,
                const std::string& ns_name,
                bool publish_robot_description,
                bool use_sim_time)
{
    auto* iface = carb::getCachedInterface<Ros2Control>();
    if (!iface || !iface->isReady())
    {
        return -1;
    }
    return iface->setupCm(articulation_path.c_str(), urdf_xml.c_str(), controller_yaml_path.c_str(), ns_name.c_str(),
                          publish_robot_description, use_sim_time);
}

void py_teardown_cm(const std::string& articulation_path)
{
    auto* iface = carb::getCachedInterface<Ros2Control>();
    if (iface)
        iface->teardownCm(articulation_path.c_str());
}

void py_teardown_all()
{
    auto* iface = carb::getCachedInterface<Ros2Control>();
    if (iface)
        iface->teardownAllCms();
}

bool py_is_ready()
{
    auto* iface = carb::getCachedInterface<Ros2Control>();
    return iface && iface->isReady();
}

void py_set_profiling_enabled(bool enabled)
{
    auto* iface = carb::getCachedInterface<Ros2Control>();
    if (iface && iface->setProfilingEnabled)
        iface->setProfilingEnabled(enabled);
}

void py_reset_profiling()
{
    auto* iface = carb::getCachedInterface<Ros2Control>();
    if (iface && iface->resetProfiling)
        iface->resetProfiling();
}

std::string py_get_profiling_json()
{
    auto* iface = carb::getCachedInterface<Ros2Control>();
    if (!iface || !iface->getProfilingJson)
        return "{}";
    const char* json = iface->getProfilingJson();
    return json ? std::string(json) : std::string("{}");
}

} // namespace

PYBIND11_MODULE(_isaacsim_ros2_control, m)
{
    m.doc() = "Bindings for the isaacsim.ros2.control Carbonite plugin (Ros2Control).";

    m.def("setup_cm", &py_setup_cm, py::arg("articulation_path"), py::arg("urdf_xml"), py::arg("controller_yaml_path"),
          py::arg("ns_name") = "", py::arg("publish_robot_description") = true, py::kw_only(),
          py::arg("use_sim_time") = true,
          "Configure a ControllerManager for the given articulation. "
          "When use_sim_time is true, provide /clock with a ROS clock source. "
          "Returns 0 on success, -1 if the plugin is not ready, "
          "-2 if a CM is already registered for this articulation, "
          "-3 if CM init failed.");
    m.def("teardown_cm", &py_teardown_cm, py::arg("articulation_path"),
          "Tear down the CM associated with the given articulation. "
          "No-op if not registered.");
    m.def("teardown_all", &py_teardown_all, "Tear down all registered CMs.");
    m.def("is_ready", &py_is_ready,
          "Returns True iff the Carbonite plugin has loaded the per-distro "
          "backend and the shared rcl_context is valid.");
    m.def("set_profiling_enabled", &py_set_profiling_enabled, py::arg("enabled"),
          "Enable or disable low-level ros2_control timing accumulation.");
    m.def("reset_profiling", &py_reset_profiling, "Clear accumulated low-level ros2_control timing samples.");
    m.def("get_profiling_json", &py_get_profiling_json,
          "Return accumulated low-level ros2_control timing samples as JSON.");
}
