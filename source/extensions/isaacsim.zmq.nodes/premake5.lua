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
    local ogn = get_ogn_project_information(ext, "isaacsim/zmq/nodes")

    -- The ZMQ sockets come from isaacsim.zmq.core; the generated protobuf headers + message
    -- library come from isaacsim.zmq.protos. Both live in sibling extension install dirs.
    -- ext.bin_dir = _build/<platform>/<config>/exts/isaacsim.zmq.nodes/bin
    local zmq_core_bin_dir = ext.bin_dir:gsub("/isaacsim%.zmq%.nodes/bin$", "/isaacsim.zmq.core/bin")
    local zmq_protos_bin_dir = ext.bin_dir:gsub("/isaacsim%.zmq%.nodes/bin$", "/isaacsim.zmq.protos/bin")
    local proto_out_dir = ext.bin_dir:gsub("/isaacsim%.zmq%.nodes/bin$", "/isaacsim.zmq.protos/generated/proto")

    project_ext(ext)

    -- C++ plugin
    project_ext_plugin(ext, "isaacsim.zmq.nodes.plugin")

    add_files("impl", "plugins")
    add_files("ogn", ogn.nodes_path)

    add_ogn_dependencies(ogn, { "python/nodes" })
    add_ogn_dependencies(ogn, { "nodes" })

    add_cuda_dependencies()
    include_physx()

    includedirs {
        "%{root}/source/extensions/isaacsim.core.includes/include",
        "%{root}/source/extensions/isaacsim.core.nodes/include",
        "%{root}/source/extensions/isaacsim.zmq.core/include",
        "%{root}/source/extensions/isaacsim.zmq.nodes/include",
        "%{root}/_build/target-deps/zmq/include",
        "%{root}/_build/target-deps/cppzmq/include",
        "%{root}/_build/target-deps/protobuf/include",
        "%{root}/_build/target-deps/abseil/include",
        proto_out_dir,
        "%{root}/_build/target-deps/usd/%{cfg.buildcfg}/include",
        "%{root}/_build/target-deps/usd_ext_physics/%{cfg.buildcfg}/include",
        extsbuild_dir .. "/omni.syntheticdata/include",
        "%{kit_sdk_bin_dir}/dev/fabric/include/",
    }

    libdirs {
        "%{root}/_build/target-deps/usd/%{cfg.buildcfg}/lib",
        "%{root}/_build/target-deps/usd_ext_physics/%{cfg.buildcfg}/lib",
        extsbuild_dir .. "/omni.usd.core/bin",
        "%{root}/_build/target-deps/zmq/lib",
        "%{root}/_build/target-deps/protobuf/lib",
        "%{root}/_build/target-deps/abseil/lib",
        zmq_core_bin_dir,
        zmq_protos_bin_dir,
    }

    links {
        "physxSchema",
        "omni.usd",
        "isaacsim.zmq.core",
        "isaacsim.zmq.protos",
        "zmq",
    }

    linkoptions {
        "-Wl,--allow-multiple-definition",
        "-Wl,--whole-archive",
        "%{root}/_build/target-deps/abseil/lib/libabsl_*.a",
        "%{root}/_build/target-deps/protobuf/lib/lib*.a",
        "-Wl,--no-whole-archive",
    }

    extra_usd_libs = { "usdGeom", "usdPhysics", "ts" }
    add_usd(extra_usd_libs)

    filter { "system:linux" }
    disablewarnings { "error=pragmas", "error=narrowing", "error=unused-but-set-variable", "error=unused-variable" }
    buildoptions("-fvisibility=default")
    linkoptions { "-Wl,--export-dynamic" }
    filter { "system:windows" }
    buildoptions("-D_CRT_SECURE_NO_WARNINGS")
    filter {}

    filter { "configurations:debug" }
    defines { "_DEBUG" }
    filter { "configurations:release" }
    defines { "NDEBUG" }
    filter {}

    -- OGN project
    project_ext_ogn(ext, ogn)

    -- Python bindings
    project_ext_bindings {
        ext = ext,
        project_name = ogn.python_project,
        module = "_zmq_nodes",
        src = ogn.bindings_path,
        target_subdir = ogn.bindings_target_path,
    }
    add_files("bindings", "bindings/*.*")
    add_files("python", "python/*.py")
    add_files("python/nodes", "python/nodes/*.py")
    add_files("python/tests", "python/tests/*.py")

    includedirs {
        "%{root}/source/extensions/isaacsim.zmq.nodes/include",
    }

    repo_build.prebuild_copy {
        { "python/__init__.py", ogn.python_target_path },
        { "python/extension.py", ogn.python_target_path },
    }

    repo_build.prebuild_link {
        { "python/tests", ogn.python_target_path .. "/tests" },
        { "python/nodes", ogn.python_target_path .. "/nodes" },
        { "include", ext.target_dir .. "/include" },
        { "docs", ext.target_dir .. "/docs" },
    }

else
    print("SKIPPING BUILD - ZMQ nodes extension only supported on linux-x86_64")
end
