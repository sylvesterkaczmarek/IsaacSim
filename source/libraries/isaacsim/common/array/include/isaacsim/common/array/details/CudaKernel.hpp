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

#pragma once

#include "isaacsim/common/array/Dtype.hpp"
#include "isaacsim/common/array/Export.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>

namespace isaacsim
{
namespace common
{
namespace array
{
namespace details
{

/**
 * @brief Lazily loads the optional, separately-compiled CUDA kernel library and exposes its kernels.
 * @details
 * The kernels are compiled by nvcc, at configure time, into a sibling shared library
 * (`isaacsim-common-array-cuda-kernels`) that is statically linked against the CUDA runtime and
 * never linked into `isaacsim-common-array` itself. This class `dlopen`s that sibling library
 * next to `isaacsim-common-array`, on first use, mirroring the lazy-loading pattern already used
 * in @ref CudaRuntime. If the sibling library was not built (no vendored `nvcc` at configure time)
 * or fails to load, @ref isLoaded() returns `false` and every kernel call reports failure either by
 * throwing @ref CudaRuntimeError or by returning `false`, depending on its `throwIfInvalid` argument
 * -- `isaacsim-common-array` itself never fails to load or crashes as a result, on a machine with no
 * CUDA toolkit or driver installed. Call `getInstance(false)` to probe availability via
 * @ref isLoaded() without throwing.
 */
class ISAACSIM_COMMON_ARRAY_API CudaKernel
{
public:
    /** @brief Returns the process-wide, lazily-loaded kernel library. */
    static CudaKernel& getInstance(bool throwIfInvalid = true);

    /** @brief Returns whether the sibling kernel library was found and successfully loaded. */
    bool isLoaded() const;

    /**
     * @brief Casts a device-resident buffer from one supported dtype to another.
     * @details
     * Supports every @ref DType::Kind pair (all 11 scalar kinds), dispatched at runtime to a
     * `static_cast`-equivalent element-wise kernel templated on the concrete source/destination
     * C++ types.
     * @param[in] source Device pointer to `count` elements of @p sourceKind.
     * @param[out] destination Device pointer to `count` elements of @p destinationKind.
     * @param[in] count Number of elements to cast.
     * @param[in] sourceKind Element kind of @p source.
     * @param[in] destinationKind Element kind of @p destination.
     * @param[in] stream CUDA stream to launch on, or `nullptr` for the default stream.
     * @param[in] throwIfInvalid If `true`, throws @ref CudaRuntimeError on failure; otherwise returns `false`.
     * @return `true` on success, `false` on failure when @p throwIfInvalid is `false`.
     */
    bool cast(const void* source,
              void* destination,
              size_t count,
              DType::Kind sourceKind,
              DType::Kind destinationKind,
              cudaStream_t stream = nullptr,
              bool throwIfInvalid = true) const;

    /**
     * @brief Gathers a device-resident buffer into a broadcast destination shape.
     * @details
     * Element-wise gather: for every element of the destination (in row-major order), reads from
     * `source` at the offset given by the dot product of the destination multi-index with
     * @p sourceStrides. A stride of 0 marks a broadcast axis (source size 1, or an axis absent
     * from the source that was implicitly prepended), so every element along that axis reads the
     * same source element -- mirroring the CPU gather in `Array::broadcastTo()`.
     * @param[in] source Device pointer to the source buffer.
     * @param[out] destination Device pointer to `totalElements` elements of @p elementSize bytes.
     * @param[in] sourceStrides Per-axis source stride, in elements, aligned to the destination
     * axes (length equal to the destination's number of dimensions).
     * @param[in] destinationShape Destination shape, in elements per axis (same length as
     * @p sourceStrides).
     * @param[in] elementSize Size, in bytes, of one element.
     * @param[in] totalElements Total number of elements in the destination.
     * @param[in] stream CUDA stream to launch on, or `nullptr` for the default stream.
     * @param[in] throwIfInvalid If `true`, throws @ref CudaRuntimeError on failure; otherwise returns `false`.
     * @return `true` on success, `false` on failure when @p throwIfInvalid is `false`.
     */
    bool broadcastTo(const void* source,
                     void* destination,
                     const std::vector<int64_t>& sourceStrides,
                     const std::vector<int64_t>& destinationShape,
                     size_t elementSize,
                     size_t totalElements,
                     cudaStream_t stream = nullptr,
                     bool throwIfInvalid = true) const;

    CudaKernel(const CudaKernel&) = delete;
    CudaKernel& operator=(const CudaKernel&) = delete;

private:
    using CastFn = cudaError_t (*)(const void*, void*, size_t, int32_t, int32_t, cudaStream_t);
    using BroadcastToFn =
        cudaError_t (*)(const void*, void*, const int64_t*, const int64_t*, size_t, size_t, size_t, cudaStream_t);

    CudaKernel();
    ~CudaKernel();

    std::string m_loadError;
    void* m_handle{ nullptr };
    CastFn m_cast{ nullptr };
    BroadcastToFn m_broadcastTo{ nullptr };
};

} // namespace details
} // namespace array
} // namespace common
} // namespace isaacsim
