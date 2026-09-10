-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0
--
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--
-- http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.

local ext = get_current_extension_info()

project_ext(ext)

local ros_distributions = { "jazzy", "humble" }

for _, ros_distro in ipairs(ros_distributions) do
    local ros_root = "%{root}/_build/target-deps/nv_ros2_" .. ros_distro
    local ext_src = path.join(root, "source/extensions/isaacsim.ros2.control")
    local distro_prefix = ext.target_dir .. "/" .. ros_distro
    local share_dir = distro_prefix .. "/share"
    local ament_pkg_dir = share_dir .. "/ament_index/resource_index/packages"
    local ament_hwif_dir = share_dir .. "/ament_index/resource_index/hardware_interface__pluginlib__plugin"
    local distro_lib_dir = distro_prefix .. "/lib"

    project_with_location("isaacsim.ros2.control." .. ros_distro)
    targetdir(distro_lib_dir)
    kind("SharedLib")
    language("C++")
    pic("On")
    staticruntime("Off")
    defines { "ROS2_BACKEND_" .. ros_distro:upper() }

    add_files("impl", "library/backend")
    add_files("iface", "include")

    -- Linux nv_ros2 packages flatten ROS headers to include/<pkg>/header.hpp.
    -- Do not add package-specific include directories on Linux: headers such as
    -- rcl/time.h and rosidl_runtime_c/string.h would shadow libc headers.
    local include_list = {
        "%{root}/source/extensions/isaacsim.core.includes/include",
        "%{root}/source/extensions/isaacsim.ros2.core/include",
        "%{root}/source/extensions/isaacsim.ros2.control/include",
        "%{root}/_build/target-deps/usd/%{cfg.buildcfg}/include",
        "%{root}/_build/target-deps/usd/%{cfg.buildcfg}/include/boost",
        ros_root .. "/include",
        ros_root .. "/include/rosidl_runtime_cpp",
        ros_root .. "/include/console_bridge_vendor",
        ros_root .. "/opt/console_bridge_vendor/include",
        ros_root .. "/opt/spdlog_vendor/include",
        ros_root .. "/opt/spdlog_vendor/include/spdlog",
        "%{root}/_build/target-deps/tinyxml2/include",
        "%{root}/_build/target-deps/nlohmann_json/include",
        -- TensorApi (DOF read/write) + omni::usd::UsdContext (stage id lookup).
        "%{root}/_build/target-deps/omni_physics/%{config}/include",
        "%{kit_sdk_bin_dir}/dev/include",
        "%{kit_sdk_bin_dir}/dev/fabric/include",
        -- IImuSensor (header-only, getCachedInterface).
        "%{root}/source/extensions/isaacsim.sensors.experimental.physics/include",
    }
    filter { "system:windows" }
        -- Windows nv_ros2 packages keep headers under include/<pkg>/<pkg>/*.h.
        includedirs {
            ros_root .. "/include/action_msgs",
            ros_root .. "/include/ament_index_cpp",
            ros_root .. "/include/builtin_interfaces",
            ros_root .. "/include/class_loader",
            ros_root .. "/include/control_msgs",
            ros_root .. "/include/controller_interface",
            ros_root .. "/include/controller_manager",
            ros_root .. "/include/controller_manager_msgs",
            ros_root .. "/include/diagnostic_msgs",
            ros_root .. "/include/diagnostic_updater",
            ros_root .. "/include/hardware_interface",
            ros_root .. "/include/joint_limits",
            ros_root .. "/include/libstatistics_collector",
            ros_root .. "/include/lifecycle_msgs",
            ros_root .. "/include/pluginlib",
            ros_root .. "/include/rcl",
            ros_root .. "/include/rcl_action",
            ros_root .. "/include/rcl_interfaces",
            ros_root .. "/include/rcl_lifecycle",
            ros_root .. "/include/rcl_yaml_param_parser",
            ros_root .. "/include/rclcpp",
            ros_root .. "/include/rclcpp_action",
            ros_root .. "/include/rclcpp_lifecycle",
            ros_root .. "/include/rcpputils",
            ros_root .. "/include/rcutils",
            ros_root .. "/include/realtime_tools",
            ros_root .. "/include/rmw",
            ros_root .. "/include/rosgraph_msgs",
            ros_root .. "/include/rosidl_dynamic_typesupport",
            ros_root .. "/include/rosidl_runtime_c",
            ros_root .. "/include/rosidl_typesupport_introspection_c",
            ros_root .. "/include/rosidl_typesupport_introspection_cpp",
            ros_root .. "/include/rosidl_typesupport_interface",
            ros_root .. "/include/service_msgs",
            ros_root .. "/include/statistics_msgs",
            ros_root .. "/include/std_msgs",
            ros_root .. "/include/tracetools",
            ros_root .. "/include/trajectory_msgs",
            ros_root .. "/include/type_description_interfaces",
            ros_root .. "/include/unique_identifier_msgs",
        }
        if ros_distro == "jazzy" then
            includedirs {
                ros_root .. "/include/pal_statistics",
                ros_root .. "/include/pal_statistics_msgs",
            }
        end
    filter {}
    includedirs(include_list)

    libdirs {
        "%{root}/_build/target-deps/nv_ros2_" .. ros_distro .. "/lib",
        extsbuild_dir .. "/omni.usd.core/bin", -- omni::usd::UsdContext
    }

    links {
        "rclcpp", "rclcpp_lifecycle", "rclcpp_action",
        "hardware_interface", "controller_manager", "controller_interface",
        "class_loader",
        "console_bridge",
        "rosidl_typesupport_cpp",
        "rosgraph_msgs__rosidl_typesupport_cpp",
        "std_msgs__rosidl_typesupport_cpp",
        "rcl", "rcutils", "rmw",
        "rcl_yaml_param_parser",
        "carb",
        "omni.usd", -- UsdContext::getContext() in IsaacSimSystem::on_configure
        -- TensorApi is header-only via getCachedInterface; no link needed.
    }
    if ros_distro == "jazzy" then
        links { "pal_statistics" }
    end
    -- Note: pluginlib is header-only; do NOT add it to links{}.

    extra_usd_libs = { "usdPhysics", "usdUtils" }
    add_usd(extra_usd_libs)

    -- Stage pluginlib descriptor + ament_index resources under the active ROS
    -- distro prefix so the in-process controller_manager (running pluginlib)
    -- can discover the matching IsaacSimSystem hardware-interface plugin via
    -- its standard search rules.
    -- AMENT_PREFIX_PATH is augmented at runtime in PluginInterface.cpp to
    -- include this per-distro directory.
    --
    -- The two ament_index marker files are generated inline (vs. checked into
    -- source/data/share/) because they're trivial: one's empty, the other
    -- holds a single-line relative path to the pluginlib XML descriptor. The
    -- real meaningful XML (isaac_sim_system_plugin.xml, multi-line, declares
    -- the class) stays as a file under library/backend/ and is copied here.

    -- Generate the per-distro pluginlib descriptor and the ament marker at premake time
    -- (cross-platform). The marker content is the
    -- relative path to the descriptor and is identical for every distro. Generated files
    -- live under library/backend/_generated (git-ignored) and are staged by {COPYFILE}.
    local gen_dir = ext_src .. "/library/backend/_generated"
    os.mkdir(gen_dir)
    local gen_xml = gen_dir .. "/isaac_sim_system_plugin." .. ros_distro .. ".xml"
    local template = io.readfile(ext_src .. "/library/backend/isaac_sim_system_plugin.xml")
    if template then
        io.writefile(gen_xml, (template:gsub("@ROS_DISTRO@", ros_distro)))
    end
    local gen_marker = gen_dir .. "/ament_hwif_marker"
    io.writefile(gen_marker, "share/isaacsim_ros2_control/isaac_sim_system_plugin.xml\n")

    postbuildcommands {
        "{MKDIR} " .. share_dir .. "/isaacsim_ros2_control",
        "{COPYFILE} " .. gen_xml .. " " .. share_dir .. "/isaacsim_ros2_control/isaac_sim_system_plugin.xml",
        "{COPYFILE} " .. ext_src .. "/library/backend/package.xml " ..
            share_dir .. "/isaacsim_ros2_control/package.xml",
        -- ament_index resource_index markers (pluginlib discovery).
        "{MKDIR} " .. ament_pkg_dir,
        "{MKDIR} " .. ament_hwif_dir,
        "{TOUCH} " .. ament_pkg_dir .. "/isaacsim_ros2_control",
        "{COPYFILE} " .. gen_marker .. " " .. ament_hwif_dir .. "/isaacsim_ros2_control",
        "{MKDIR} " .. distro_lib_dir,
    }

    -- Build the backend directly in the per-distro lib/ so Isaac Sim's
    -- LibraryLoader and ROS pluginlib open the same physical shared object.
    filter { "system:linux" }
        buildoptions {
            "-fvisibility=default",
            "-isystem " .. ros_root .. "/include",
        }
        linkoptions { "-Wl,--export-dynamic" }
    filter {}
end

-- Distro-agnostic Carbonite plugin
project_ext_plugin(ext, "isaacsim.ros2.control.plugin")

add_files("impl", "plugins")
add_files("iface", "include")

includedirs {
    "%{root}/source/extensions/isaacsim.core.includes/include",
    "%{root}/source/extensions/isaacsim.ros2.core/include",
    "%{root}/source/extensions/isaacsim.ros2.control/include",
    "%{root}/_build/target-deps/nlohmann_json/include",
    -- Engine-agnostic physics step events live in omni_physics.
    "%{root}/_build/target-deps/omni_physics/%{config}/include",
    "%{root}/_build/target-deps/usd/%{cfg.buildcfg}/include",
    "%{root}/_build/target-deps/usd/%{cfg.buildcfg}/include/boost",
    "%{kit_sdk_bin_dir}/dev/fabric/include/",
    "%{kit_sdk_bin_dir}/dev/include", -- omni::timeline::ITimeline
}

filter { "system:linux" }
    links { "dl" }
filter {}

-- Pybind11 bindings: mirrors isaacsim.ros2.core's bindings layout via
-- repo_build's project_ext_bindings helper. The helper renames the .so to
-- `_isaacsim_ros2_control.cpython-<ver>-<arch>.so` and lands it under
-- isaacsim/ros2/control/bindings/, paralleling the core extension.
project_ext_bindings {
    ext = ext,
    project_name = "isaacsim.ros2.control.python",
    module = "_isaacsim_ros2_control",
    src = "bindings",
    target_subdir = "isaacsim/ros2/control/bindings",
}
includedirs {
    "%{root}/source/extensions/isaacsim.ros2.control/include",
}

-- Stage the python package (urdf_synth, ros2_control_manager, __init__.py,
-- tests/) into the build output. We copy .py files individually so the
-- bindings/ subdir created by project_ext_bindings can coexist as a sibling;
-- symlinking the parent dir would shadow it.
repo_build.prebuild_link {
    { "docs", ext.target_dir .. "/docs" },
    { "data", ext.target_dir .. "/data" },
    { "isaacsim/ros2/control/tests", ext.target_dir .. "/isaacsim/ros2/control/tests" },
}

repo_build.prebuild_copy {
    { "isaacsim/__init__.py", ext.target_dir .. "/isaacsim" },
    { "isaacsim/ros2/__init__.py", ext.target_dir .. "/isaacsim/ros2" },
    { "isaacsim/ros2/control/*.py", ext.target_dir .. "/isaacsim/ros2/control" },
}
