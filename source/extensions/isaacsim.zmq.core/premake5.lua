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

if os.target() == "linux" then
    local ext = get_current_extension_info()
    local ogn = get_ogn_project_information(ext, "isaacsim/zmq/core")
    project_ext(ext)

    -- Build the ZeroMQ socket backend library. This is pure transport: it moves raw
    -- [topic, payload] byte frames and has no knowledge of protobuf. The message schemas
    -- live in the separate isaacsim.zmq.protos extension.
    project_with_location("isaacsim.zmq.core")
    targetdir(ext.bin_dir)
    kind("SharedLib")
    language("C++")
    cppdialect("C++17")

    pic("On")
    staticruntime("Off")

    add_files("impl", "library/backend")
    add_files("iface", "include")

    includedirs {
        "%{root}/source/extensions/isaacsim.zmq.core/include",
        "%{root}/source/extensions/isaacsim.core.includes/include",
        "%{root}/_build/target-deps/zmq/include",
        "%{root}/_build/target-deps/cppzmq/include",
    }

    libdirs {
        "%{root}/_build/target-deps/zmq/lib",
    }

    -- Dynamic link to zmq.
    links { "zmq" }

    filter { "system:linux" }
    disablewarnings { "error=pragmas" }
    buildoptions { "-fvisibility=default" }
    linkoptions { "-Wl,--export-dynamic", "-Wl,-rpath,'$$ORIGIN/lib'" }
    filter {}
    filter { "system:windows" }
    buildoptions("-D_CRT_SECURE_NO_WARNINGS")
    filter {}

    filter { "configurations:debug" }
    defines { "_DEBUG" }
    filter { "configurations:release" }
    defines { "NDEBUG" }
    filter {}

    -- Bundle libzmq.so alongside the extension binary.
    repo_build.prebuild_copy {
        { "%{root}/_build/target-deps/zmq/lib/libzmq.so*", ext.bin_dir .. "/lib/" },
    }

    repo_build.prebuild_link {
        { "docs",          ext.target_dir .. "/docs" },
        { "include",       ext.target_dir .. "/include" },
        { "python/impl",   ext.target_dir .. "/isaacsim/zmq/core/impl" },
        { "python/tests",  ext.target_dir .. "/isaacsim/zmq/core/tests" },
    }

    repo_build.prebuild_copy {
        { "python/__init__.py", ext.target_dir .. "/isaacsim/zmq/core/" },
    }

    -- Python bindings for the core library
    project_ext_bindings {
        ext = ext,
        project_name = ogn.python_project,
        module = "_isaacsim_zmq_core",
        src = "bindings/isaacsim.zmq.core",
        target_subdir = "isaacsim/zmq/core/bindings",
    }
    add_files("bindings", "bindings/isaacsim.zmq.core/*.*")

    includedirs {
        "%{root}/source/extensions/isaacsim.zmq.core/include",
        "%{root}/_build/target-deps/zmq/include",
        "%{root}/_build/target-deps/cppzmq/include",
    }

    libdirs {
        ext.bin_dir,
        "%{root}/_build/target-deps/zmq/lib",
    }

    links { "isaacsim.zmq.core", "zmq" }

else
    print("SKIPPING BUILD - Only supported on linux")
end
