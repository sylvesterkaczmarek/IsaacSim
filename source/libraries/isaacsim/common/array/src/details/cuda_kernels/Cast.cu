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
//
// Compiled by nvcc into the standalone `isaacsim-common-array-cuda-kernels` shared library.
// Never linked into `isaacsim-common-array` itself -- see CudaKernel.hpp.

#include "KernelExport.hpp"

#include <isaacsim/common/array/Dtype.hpp>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>

using isaacsim::common::array::DType;

namespace
{

template <typename SrcT, typename DstT>
__global__ void castKernel(const SrcT* source, DstT* destination, size_t count)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count)
    {
        destination[i] = static_cast<DstT>(source[i]);
    }
}

template <typename SrcT, typename DstT>
cudaError_t launchKernel(const void* source, void* destination, size_t count, cudaStream_t stream)
{
    if (count == 0)
    {
        return cudaSuccess;
    }
    constexpr int threadsPerBlock = 256;
    const int blocks = static_cast<int>((count + threadsPerBlock - 1) / threadsPerBlock);
    castKernel<SrcT, DstT><<<blocks, threadsPerBlock, 0, stream>>>(
        static_cast<const SrcT*>(source), static_cast<DstT*>(destination), count);
    return cudaGetLastError();
}

// Maps a DType::Kind runtime value to its C++ scalar type and invokes `fn<T>()` for it.
template <typename Fn>
cudaError_t dispatchByKind(int32_t kind, Fn&& fn)
{
    switch (static_cast<DType::Kind>(kind))
    {
    case DType::Kind::eBool:
        return fn(bool{});
    case DType::Kind::eInt8:
        return fn(int8_t{});
    case DType::Kind::eInt16:
        return fn(int16_t{});
    case DType::Kind::eInt32:
        return fn(int32_t{});
    case DType::Kind::eInt64:
        return fn(int64_t{});
    case DType::Kind::eUInt8:
        return fn(uint8_t{});
    case DType::Kind::eUInt16:
        return fn(uint16_t{});
    case DType::Kind::eUInt32:
        return fn(uint32_t{});
    case DType::Kind::eUInt64:
        return fn(uint64_t{});
    case DType::Kind::eFloat32:
        return fn(float{});
    case DType::Kind::eFloat64:
        return fn(double{});
    }
    return cudaErrorInvalidValue;
}

} // namespace


ISAACSIM_ARRAY_KERNEL_API cudaError_t arrayCast(const void* source,
                                                void* destination,
                                                size_t count,
                                                int32_t sourceKind,
                                                int32_t destinationKind,
                                                cudaStream_t stream)
{
    return dispatchByKind(sourceKind,
                          [&](auto sourceSample) -> cudaError_t
                          {
                              using SrcT = decltype(sourceSample);
                              return dispatchByKind(destinationKind,
                                                    [&](auto destinationSample) -> cudaError_t
                                                    {
                                                        using DstT = decltype(destinationSample);
                                                        return launchKernel<SrcT, DstT>(source, destination, count, stream);
                                                    });
                          });
}
