# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

isaacsim_add_group(
    NAME isaacsim_robot_schema
    MODULE_PREFIX usd.schema
    MODULES
        usd.schema.isaac
)

isaacsim_add_usd_schema_module(
    NAME usd.schema.isaac
    PYTHON_IMPORTS
        usd.schema.isaac
        omni.isaac.IsaacSensorSchema
        omni.isaac.RangeSensorSchema
    SCHEMAS
        usd/schema/isaac/robot_schema/RobotSchema.usda
        usd/schema/isaac/sensor_schema/SensorSchema.usda
        usd/schema/isaac/range_sensor_schema/RangeSensorSchema.usda
    PUBLIC_CPP_HEADERS
        isaacsim/robot/schema/robot_schema.hpp
        isaacsim/robot/schema/sensor_tokens.hpp
        isaacsim/robot/schema/utils.hpp
)

isaacsim_add_python_tests(
    MODULE usd.schema.isaac
    LABELS
        unit
        schema
)
