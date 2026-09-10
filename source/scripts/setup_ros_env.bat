:: SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
:: SPDX-License-Identifier: Apache-2.0
::
:: Licensed under the Apache License, Version 2.0 (the "License");
:: you may not use this file except in compliance with the License.
:: You may obtain a copy of the License at
::
:: http://www.apache.org/licenses/LICENSE-2.0
::
:: Unless required by applicable law or agreed to in writing, software
:: distributed under the License is distributed on an "AS IS" BASIS,
:: WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
:: See the License for the specific language governing permissions and
:: limitations under the License.

@echo off

REM setup_ros_env.bat - Configure ROS2 environment for Isaac Sim on Windows

REM Get script directory and set Isaac Sim root path
set SCRIPT_DIR=%~dp0
REM Remove trailing backslash from SCRIPT_DIR
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set ISAAC_SIM_ROOT=%SCRIPT_DIR%

set DEFAULT_ROS_DISTRO=jazzy
set DEFAULT_RMW_IMPLEMENTATION=rmw_zenoh_cpp

set BRIDGE_EXT_PATH=%ISAAC_SIM_ROOT%\exts\isaacsim.ros2.core

REM Set ROS_DISTRO if not already set
set USE_BUNDLED_ROS=false
if "%ROS_DISTRO%"=="" (
    set ROS_DISTRO=%DEFAULT_ROS_DISTRO%
    set USE_BUNDLED_ROS=true
)

set BUNDLED_ROS_PREFIX=%BRIDGE_EXT_PATH%\%ROS_DISTRO%

if "%USE_BUNDLED_ROS%"=="true" (
    REM Prefer the bundled ROS2 libraries over unrelated DLLs already on PATH
    set "PATH=%BUNDLED_ROS_PREFIX%\lib;%PATH%"

    REM Keep custom workspace overlays first and add the bundled ROS2 package index
    if defined AMENT_PREFIX_PATH (
        if "%AMENT_PREFIX_PATH:~-1%"==";" (
            set "AMENT_PREFIX_PATH=%AMENT_PREFIX_PATH%%BUNDLED_ROS_PREFIX%"
        ) else (
            set "AMENT_PREFIX_PATH=%AMENT_PREFIX_PATH%;%BUNDLED_ROS_PREFIX%"
        )
    ) else (
        set "AMENT_PREFIX_PATH=%BUNDLED_ROS_PREFIX%"
    )
)

REM Set RMW implementation to Zenoh if not already set
if "%RMW_IMPLEMENTATION%"=="" (
    set RMW_IMPLEMENTATION=%DEFAULT_RMW_IMPLEMENTATION%
)
