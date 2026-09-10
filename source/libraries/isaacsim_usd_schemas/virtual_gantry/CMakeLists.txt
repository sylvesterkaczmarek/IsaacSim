# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

isaacsim_add_group(
    NAME isaacsim_virtual_gantry_schema
    MODULE_PREFIX isaacsim.robot_setup
    MODULES
        isaacsim.robot_setup.virtual_gantry_schema
)

isaacsim_add_usd_schema_module(
    NAME isaacsim.robot_setup.virtual_gantry_schema
    PYTHON_IMPORTS
        isaacsim.robot_setup.virtual_gantry_schema
    SCHEMAS
        isaacsim/robot_setup/virtual_gantry_schema/virtual_gantry_schema/VirtualGantrySchema.usda
)

isaacsim_add_python_tests(
    MODULE isaacsim.robot_setup.virtual_gantry_schema
    LABELS
        unit
        schema
)
