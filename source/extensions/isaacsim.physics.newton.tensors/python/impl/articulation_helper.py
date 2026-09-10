# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Helper function for articulation tensor computation that wrangles the warp data

import types

import warp as wp


@wp.kernel
def wrangle_matrix_kernel(
    m_warp: wp.array3d[float],
    arti_indices: wp.array[int],
    n_arti: int,
    n_rows: int,
    n_cols: int,
    m_tensor: wp.array3d[float],
):
    """Kernel for compacting newton's matrix into what tensor API wants."""
    art_idx = wp.tid()
    if art_idx >= n_arti:
        return
    orig_id = arti_indices[art_idx]
    for i in range(n_rows):
        for j in range(n_cols):
            m_tensor[art_idx, i, j] = m_warp[orig_id, i, j]


def wrangle_matrix(
    m_warp: wp.array, m_tensor: wp.array | None, arti_indices: wp.array, n_rows: int, n_cols: int, device: str
) -> wp.array:
    """compact newton's matrix into what tensor API wants. Newton's matrix dimension is containing all articulations in the scene.

    Args:
        m_warp: matrix from newton
        m_tensor: matrix in the tensor format
        arti_indices: indices for selecting the articulation from the newton matrix
        n_rows: number of rows of the tensor matrix
        n_cols: number of cols of the tensor matrix
        device: device to allocate the buffer
    Returns:
        The matrix in the tensor format.
    """
    n_arti = arti_indices.shape[0]
    if not m_tensor or m_tensor.shape != (n_arti, n_rows, n_cols):
        m_tensor = wp.array(shape=(n_arti, n_rows, n_cols), dtype=wp.float32, device=device)

    wp.launch(
        kernel=wrangle_matrix_kernel,
        dim=n_arti,
        inputs=[
            m_warp,
            arti_indices,
            n_arti,
            n_rows,
            n_cols,
        ],
        outputs=[m_tensor],
        device=device,
    )
    return m_tensor


def allocate_buffer(m_buffer: wp.array | None, buffer_size: tuple | int, dtype: type, device: str) -> wp.array:
    """Allocate buffer for newton's calculation

    Args:
        m_buffer: buffer used for calculation.
        buffer_size: size for the buffer
        device: device to allocate the buffer
    Returns:
        The buffer.
    """
    if isinstance(buffer_size, int):
        buffer_size = (buffer_size,)
    # We only check size, not the device
    if not m_buffer or m_buffer.shape != buffer_size:
        m_buffer = wp.array(shape=buffer_size, dtype=dtype, device=device)
    return m_buffer
