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

#include "isaacsim/common/logging/Logging.hpp"

#include <pxr/base/plug/registry.h>

#ifdef _WIN32
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

#include <filesystem>
#include <string>

namespace
{

namespace logging = isaacsim::common::logging;

#ifndef _WIN32
std::filesystem::path getSharedLibraryDirectory()
{
    Dl_info info;
    if (dladdr(reinterpret_cast<void*>(&getSharedLibraryDirectory), &info) && info.dli_fname)
    {
        return std::filesystem::path(info.dli_fname).parent_path();
    }
    return {};
}
#else
std::filesystem::path getSharedLibraryDirectory()
{
    HMODULE hModule = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&getSharedLibraryDirectory), &hModule))
    {
        wchar_t path[MAX_PATH];
        if (GetModuleFileNameW(hModule, path, MAX_PATH))
        {
            return std::filesystem::path(path).parent_path();
        }
    }
    return {};
}
#endif

struct UsdSchemaRegistration
{
    UsdSchemaRegistration()
    {
        const logging::Logger logger("isaacsim.foundation.usd.openusd");
        const std::filesystem::path searchPath = getSharedLibraryDirectory() / "third-party-schemas";
        size_t registeredSchemas = 0;
        std::error_code errorCode;
        for (const auto& entry : std::filesystem::recursive_directory_iterator(searchPath, errorCode))
        {
            if (!entry.is_directory())
            {
                continue;
            }
            if (!std::filesystem::exists(entry.path() / "plugInfo.json"))
            {
                continue;
            }
            const std::string schemaPath = entry.path().string();
            ISAACSIM_REPORT(logger, "Registering third-party schema: {}", schemaPath);
            PXR_NS::PlugRegistry::GetInstance().RegisterPlugins({ schemaPath });
            registeredSchemas++;
        }
        if (!registeredSchemas)
        {
            ISAACSIM_LOG_WARNING(logger, "No third-party schemas found in {}", searchPath.string());
        }
    }
};

static UsdSchemaRegistration s_usdSchemaRegistration;

} // namespace
