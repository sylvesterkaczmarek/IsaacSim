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

#include "isaacsim/common/array/details/CudaKernel.hpp"

#include <isaacsim/common/exceptions/Exceptions.hpp>

#include <filesystem>
#include <string>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace
{

#if defined(_WIN32)
constexpr const char* kKernelLibraryFileName = "isaacsim-common-array-cuda-kernels.dll";

// Resolves the sibling kernel library next to the DLL this code itself lives in, so the lookup
// does not depend on the process' PATH or working directory.
std::filesystem::path siblingLibraryPath()
{
    HMODULE ownModule = nullptr;
    if (!::GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              reinterpret_cast<LPCSTR>(&siblingLibraryPath), &ownModule))
    {
        return kKernelLibraryFileName;
    }
    char path[MAX_PATH]{};
    if (::GetModuleFileNameA(ownModule, path, MAX_PATH) == 0)
    {
        return kKernelLibraryFileName;
    }
    return std::filesystem::path(path).parent_path() / kKernelLibraryFileName;
}

void* loadLibrary(const char* name)
{
    return reinterpret_cast<void*>(::LoadLibraryA(name));
}

std::string lastLoadError()
{
    return "Windows error " + std::to_string(::GetLastError());
}

void* loadSymbol(void* library, const char* name)
{
    return reinterpret_cast<void*>(::GetProcAddress(reinterpret_cast<HMODULE>(library), name));
}

void unloadLibrary(void* library)
{
    if (library)
    {
        ::FreeLibrary(reinterpret_cast<HMODULE>(library));
    }
}
#else
#    if defined(__APPLE__)
constexpr const char* kKernelLibraryFileName = "libisaacsim-common-array-cuda-kernels.dylib";
#    else
constexpr const char* kKernelLibraryFileName = "libisaacsim-common-array-cuda-kernels.so";
#    endif

// Resolves the sibling kernel library next to the shared object this code itself lives in, so
// the lookup does not depend on RPATH/RUNPATH propagation for bare-name dlopen() calls.
std::filesystem::path siblingLibraryPath()
{
    Dl_info info{};
    if (::dladdr(reinterpret_cast<const void*>(&siblingLibraryPath), &info) == 0 || !info.dli_fname)
    {
        return kKernelLibraryFileName;
    }
    return std::filesystem::path(info.dli_fname).parent_path() / kKernelLibraryFileName;
}

void* loadLibrary(const char* name)
{
    return ::dlopen(name, RTLD_LAZY | RTLD_LOCAL);
}

std::string lastLoadError()
{
    const char* error = ::dlerror();
    return error ? error : "unknown error";
}

void* loadSymbol(void* library, const char* name)
{
    return ::dlsym(library, name);
}

void unloadLibrary(void* library)
{
    if (library)
    {
        ::dlclose(library);
    }
}
#endif

} // namespace

namespace isaacsim
{
namespace common
{
namespace array
{
namespace details
{

CudaKernel& CudaKernel::getInstance(bool throwIfInvalid)
{
    static CudaKernel instance;
    if (!instance.isLoaded() && throwIfInvalid)
    {
        throw isaacsim::common::exceptions::CudaRuntimeError(
            "CudaKernel::getInstance", "the CUDA kernel library was not loaded: " + instance.m_loadError);
    }
    return instance;
}

bool CudaKernel::isLoaded() const
{
    return m_handle != nullptr;
}

bool CudaKernel::cast(const void* source,
                      void* destination,
                      size_t count,
                      DType::Kind sourceKind,
                      DType::Kind destinationKind,
                      cudaStream_t stream,
                      bool throwIfInvalid) const
{
    if (!m_cast)
    {
        if (throwIfInvalid)
        {
            throw isaacsim::common::exceptions::CudaRuntimeError(
                "CudaKernel::cast", "the CUDA kernel library was not loaded: " + m_loadError);
        }
        return false;
    }
    cudaError_t result = m_cast(
        source, destination, count, static_cast<int32_t>(sourceKind), static_cast<int32_t>(destinationKind), stream);
    if (result != cudaSuccess)
    {
        if (throwIfInvalid)
        {
            throw isaacsim::common::exceptions::CudaRuntimeError(
                "CudaKernel::cast", "kernel launch failed", static_cast<int>(result));
        }
        return false;
    }
    return true;
}

bool CudaKernel::broadcastTo(const void* source,
                             void* destination,
                             const std::vector<int64_t>& sourceStrides,
                             const std::vector<int64_t>& destinationShape,
                             size_t elementSize,
                             size_t totalElements,
                             cudaStream_t stream,
                             bool throwIfInvalid) const
{
    if (!m_broadcastTo)
    {
        if (throwIfInvalid)
        {
            throw isaacsim::common::exceptions::CudaRuntimeError(
                "CudaKernel::broadcastTo", "the CUDA kernel library was not loaded: " + m_loadError);
        }
        return false;
    }
    if (sourceStrides.size() != destinationShape.size())
    {
        if (throwIfInvalid)
        {
            throw isaacsim::common::exceptions::CudaRuntimeError("CudaKernel::broadcastTo",
                                                                 "strides and shape have different lengths",
                                                                 static_cast<int>(cudaErrorInvalidValue));
        }
        return false;
    }
    cudaError_t result = m_broadcastTo(source, destination, sourceStrides.data(), destinationShape.data(),
                                       destinationShape.size(), elementSize, totalElements, stream);
    if (result != cudaSuccess)
    {
        if (throwIfInvalid)
        {
            throw isaacsim::common::exceptions::CudaRuntimeError(
                "CudaKernel::broadcastTo", "kernel launch failed", static_cast<int>(result));
        }
        return false;
    }
    return true;
}

CudaKernel::CudaKernel()
{
    const std::string libraryPath = siblingLibraryPath().string();
    m_handle = loadLibrary(libraryPath.c_str());
    if (!m_handle)
    {
        m_loadError = "could not load '" + libraryPath + "': " + lastLoadError();
        return;
    }
    // A library that loads but exports nothing usually means a stale copy, or entry points built
    // without the export macro, so collect every name that failed rather than just the first.
    std::string missingSymbols;
    auto resolveSymbol = [&](const char* name)
    {
        void* symbol = loadSymbol(m_handle, name);
        if (!symbol)
        {
            if (!missingSymbols.empty())
            {
                missingSymbols += ", ";
            }
            missingSymbols += name;
        }
        return symbol;
    };

    m_cast = reinterpret_cast<CastFn>(resolveSymbol("arrayCast"));
    m_broadcastTo = reinterpret_cast<BroadcastToFn>(resolveSymbol("arrayBroadcastTo"));

    if (!missingSymbols.empty())
    {
        m_loadError = "loaded '" + libraryPath + "' but could not resolve: " + missingSymbols;
        unloadLibrary(m_handle);
        m_handle = nullptr;
        m_cast = nullptr;
        m_broadcastTo = nullptr;
    }
}

CudaKernel::~CudaKernel()
{
    unloadLibrary(m_handle);
}

} // namespace details
} // namespace array
} // namespace common
} // namespace isaacsim
