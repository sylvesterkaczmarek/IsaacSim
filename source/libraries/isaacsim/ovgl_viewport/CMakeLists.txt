# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

isaacsim_add_group(
    NAME isaacsim_ovgl_viewport
    MODULES
        isaacsim.ovgl_viewport.debug
)

add_subdirectory(debug)
