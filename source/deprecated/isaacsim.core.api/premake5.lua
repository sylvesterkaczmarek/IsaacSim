-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
ext.target_dir = deprecated_exts_path .. "/" .. ext.id
ext.bin_dir = ext.target_dir .. "/bin"
project_ext(ext)

repo_build.prebuild_link {
    { "docs", ext.target_dir .. "/docs" },
    { "data", ext.target_dir .. "/data" },
}

repo_build.prebuild_link {
    { "python/tests", ext.target_dir .. "/isaacsim/core/api/tests" },
    { "python/impl/world", ext.target_dir .. "/isaacsim/core/api/world" },
    { "python/impl/simulation_context", ext.target_dir .. "/isaacsim/core/api/simulation_context" },
    { "python/impl/scenes", ext.target_dir .. "/isaacsim/core/api/scenes" },
    { "python/impl/sensors", ext.target_dir .. "/isaacsim/core/api/sensors" },
    { "python/impl/objects", ext.target_dir .. "/isaacsim/core/api/objects" },
    { "python/impl/physics_context", ext.target_dir .. "/isaacsim/core/api/physics_context" },
    { "python/impl/articulations", ext.target_dir .. "/isaacsim/core/api/articulations" },
    { "python/impl/controllers", ext.target_dir .. "/isaacsim/core/api/controllers" },
    { "python/impl/loggers", ext.target_dir .. "/isaacsim/core/api/loggers" },
    { "python/impl/materials", ext.target_dir .. "/isaacsim/core/api/materials" },
    { "python/impl/robots", ext.target_dir .. "/isaacsim/core/api/robots" },
    { "python/impl/tasks", ext.target_dir .. "/isaacsim/core/api/tasks" },
}

repo_build.prebuild_copy {
    { "python/impl/*.py", ext.target_dir .. "/isaacsim/core/api" },
}
