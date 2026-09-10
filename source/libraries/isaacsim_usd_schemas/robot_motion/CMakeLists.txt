# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

isaacsim_add_group(
    NAME isaacsim_robot_motion_schema
    MODULE_PREFIX isaacsim.robot_motion
    MODULES
        isaacsim.robot_motion.schema
)

isaacsim_add_usd_schema_module(
    NAME isaacsim.robot_motion.schema
    PYTHON_IMPORTS
        isaacsim.robot_motion.schema
    SCHEMAS
        isaacsim/robot_motion/schema/robot_motion_schema/RobotMotionSchema.usda
)

isaacsim_add_python_tests(
    MODULE isaacsim.robot_motion.schema
    LABELS
        unit
        schema
)
