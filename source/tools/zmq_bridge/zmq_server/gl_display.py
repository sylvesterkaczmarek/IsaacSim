# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Warp GPU kernels for the zmq_server viewer: colorize depth and pad RGB->RGBA8 on the
GPU (no torch), for handoff to an ``omni.ui.ByteImageProvider``."""

from __future__ import annotations

import math

import warp as wp


@wp.kernel
def _colorize_depth_kernel(
    depth: wp.array(dtype=wp.float32, ndim=2),
    out: wp.array(dtype=wp.uint8, ndim=3),
    near: wp.float32,
    log_near: wp.float32,
    inv_span: wp.float32,
    far: wp.float32,
):
    i, j = wp.tid()
    d = wp.clamp(depth[i, j], near, far)
    t = (wp.log(d) - log_near) * inv_span  # 0 (near) .. 1 (far)
    v = wp.uint8((1.0 - t) * 255.0)  # closer == brighter
    out[i, j, 0] = v
    out[i, j, 1] = v
    out[i, j, 2] = v
    out[i, j, 3] = wp.uint8(255)


@wp.kernel
def _rgb_to_rgba_kernel(rgb: wp.array(dtype=wp.uint8, ndim=3), out: wp.array(dtype=wp.uint8, ndim=3)):
    i, j = wp.tid()
    out[i, j, 0] = rgb[i, j, 0]
    out[i, j, 1] = rgb[i, j, 1]
    out[i, j, 2] = rgb[i, j, 2]
    out[i, j, 3] = wp.uint8(255)


def colorize_depth_gpu(depth: wp.array, *, near: float = 1.0, far: float = 100.0) -> wp.array:
    """Map a cuda float32 depth (H, W) to RGBA8 (H, W, 4): grayscale, log-scaled, inverted (closer == brighter). On-GPU."""
    if depth.device.is_cpu:
        raise ValueError(f"depth must be on cuda, got device={depth.device}")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth.reshape((depth.shape[0], depth.shape[1]))
    if depth.ndim != 2:
        raise ValueError(f"expected (H, W) or (H, W, 1) depth, got shape {tuple(depth.shape)}")
    h, w = depth.shape[0], depth.shape[1]
    out = wp.empty((h, w, 4), dtype=wp.uint8, device=depth.device)
    log_near = math.log(near)
    inv_span = 1.0 / (math.log(far) - log_near)
    wp.launch(
        _colorize_depth_kernel,
        dim=(h, w),
        inputs=[depth, out, wp.float32(near), wp.float32(log_near), wp.float32(inv_span), wp.float32(far)],
        device=depth.device,
    )
    return out


def rgb_to_rgba(rgb: wp.array) -> wp.array:
    """Pad a cuda (H, W, 3) uint8 array to (H, W, 4) with opaque alpha (stays on GPU)."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3) rgb, got shape {tuple(rgb.shape)}")
    h, w = rgb.shape[0], rgb.shape[1]
    out = wp.empty((h, w, 4), dtype=wp.uint8, device=rgb.device)
    wp.launch(_rgb_to_rgba_kernel, dim=(h, w), inputs=[rgb, out], device=rgb.device)
    return out
