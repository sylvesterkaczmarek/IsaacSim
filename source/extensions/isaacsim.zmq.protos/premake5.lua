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
    project_ext(ext)

    -- Protobuf codegen: run protoc on all .proto files.
    -- C++ generated files land in _build/.../isaacsim.zmq.protos/generated/proto/ and are
    -- compiled into libisaacsim.zmq.protos.so (which isaacsim.zmq.nodes links against).
    -- Python generated files land alongside the extension's Python package.
    local proto_out_dir = ext.bin_dir .. "/../generated/proto"
    local proto_python_out_dir = ext.target_dir .. "/isaacsim/zmq/protos"
    local proto_src_dir = "%{root}/source/extensions/isaacsim.zmq.protos/proto"
    local protoc_bin = "%{root}/_build/target-deps/protobuf/bin/protoc"

    -- Build the protobuf message library (generated *.pb.cc only; no hand-written sources).
    project_with_location("isaacsim.zmq.protos")
    targetdir(ext.bin_dir)
    kind("SharedLib")
    language("C++")
    cppdialect("C++17")

    pic("On")
    staticruntime("Off")

    includedirs {
        "%{root}/_build/target-deps/protobuf/include",
        "%{root}/_build/target-deps/abseil/include",
        proto_out_dir,
    }

    libdirs {
        "%{root}/_build/target-deps/protobuf/lib",
        "%{root}/_build/target-deps/abseil/lib",
    }

    linkoptions {
        "-Wl,--allow-multiple-definition",
        "-Wl,--whole-archive",
        "%{root}/_build/target-deps/abseil/lib/libabsl_*.a",
        "%{root}/_build/target-deps/protobuf/lib/lib*.a",
        "-Wl,--no-whole-archive",
        -- Hide static-lib symbols (abseil, protobuf) so they don't conflict
        -- with Kit's own abseil when both are loaded in the same process.
        "-Wl,--exclude-libs,ALL",
        -- Force libz.so into the NEEDED section (protobuf uses zlib for compression;
        -- --as-needed would otherwise strip the dependency since the symbol may appear
        -- resolved from a previously linked archive).
        "-Wl,--no-as-needed,-lz,--as-needed",
    }

    -- Feed the .proto files to the custom build rule below. The generated .pb.cc
    -- are NOT listed here: premake already compiles them via the rule's buildoutputs.
    -- Listing them as well made premake compile each proto twice (duplicate object).
    files {
        proto_src_dir .. "/clock.proto",
        proto_src_dir .. "/image.proto",
        proto_src_dir .. "/pose.proto",
        proto_src_dir .. "/update_prim_attribute.proto",
        proto_src_dir .. "/joint_states.proto",
        proto_src_dir .. "/joint_command.proto",
    }

    -- Custom build rule: .proto → .pb.cc + .pb.h
    -- gmake generates a make target for each .proto so the .pb.cc deps resolve correctly.
    filter "files:**.proto"
        buildmessage "Generating protobuf: %{file.name}"
        buildcommands {
            "mkdir -p " .. proto_out_dir,
            "mkdir -p " .. proto_python_out_dir,
            protoc_bin .. " --proto_path=" .. proto_src_dir ..
                " --cpp_out=" .. proto_out_dir ..
                " --python_out=" .. proto_python_out_dir ..
                " %{file.abspath}",
        }
        buildoutputs {
            proto_out_dir .. "/%{file.basename}.pb.cc",
            proto_out_dir .. "/%{file.basename}.pb.h",
            proto_python_out_dir .. "/%{file.basename}_pb2.py",
        }
    filter {}

    filter { "system:linux" }
    disablewarnings { "error=pragmas" }
    -- -Wno-undef silences the toolchain's global -Wundef on the generated protobuf
    -- sources, which reference macros (e.g. PROTOBUF_ENABLE_DEBUG_LOGGING_MAY_LEAK_PII)
    -- that protobuf leaves undefined in non-debug builds.
    buildoptions { "-fvisibility=default", "-Wno-undef" }
    linkoptions { "-Wl,--export-dynamic", "-Wl,-rpath,'$$ORIGIN/lib'" }
    filter {}

    filter { "configurations:debug" }
    defines { "_DEBUG" }
    filter { "configurations:release" }
    defines { "NDEBUG" }
    filter {}

    -- libz is needed at runtime because protobuf (statically linked) uses zlib for compression.
    repo_build.prebuild_copy {
        { "%{root}/_build/target-deps/usd/release/lib/libz.so*", ext.bin_dir .. "/lib/" },
    }

    repo_build.prebuild_link {
        { "docs",          ext.target_dir .. "/docs" },
        { "python/impl",   ext.target_dir .. "/isaacsim/zmq/protos/impl" },
        { "python/tests",  ext.target_dir .. "/isaacsim/zmq/protos/tests" },
        -- Bundle the protobuf (and pyzmq) Python runtime so the generated *_pb2 schemas shipped
        -- in this package are importable on their own (also used by the standalone zmq_bridge tool).
        { "%{root}/_build/target-deps/pip_zmq_prebundle", ext.target_dir .. "/pip_prebundle" },
    }

    repo_build.prebuild_copy {
        { "python/__init__.py", ext.target_dir .. "/isaacsim/zmq/protos/" },
    }
else
    print("SKIPPING BUILD - ZMQ protos extension only supported on linux")
end
