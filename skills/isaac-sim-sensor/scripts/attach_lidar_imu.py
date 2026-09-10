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

"""Attach an RTX Ouster LiDAR and a physics IMU to a robot chassis.

Creates mount xforms under the robot prim, references the Ouster OS1 sensor
asset with a chosen variant, and places an IMU at the chassis center.
Run with $ISAAC_SIM_DIR/python.sh.
"""

from pxr import Gf, UsdGeom


def attach_ouster_lidar(
    stage,
    robot_path: str = "/World/Robot",
    mount_offset: tuple = (0.0, 0.0, 0.35),
    config: str = "OS1",
    variant: str = "OS1_REV6_32ch20hz512res",
    tick_rate: float = 20.0,
):
    """Mount an Ouster OS1 RTX lidar on the robot and return sensor + writer.

    Args:
        stage: Active USD stage.
        robot_path: Prim path of the robot root.
        mount_offset: (x, y, z) translation from robot origin to lidar mount.
        config: Lidar config key from SUPPORTED_LIDAR_CONFIGS.
        variant: Variant selection for the sensor asset.
        tick_rate: Sensor output rate in Hz.

    Returns:
        Tuple of (Lidar prim wrapper, LidarSensor).
    """
    import omni.replicator.core as rep
    from isaacsim.sensors.experimental.rtx import (
        Lidar,
        LidarSensor,
        parse_generic_model_output_data,
    )
    from omni.replicator.core import Writer

    mount_path = f"{robot_path}/lidar_mount"
    mount = UsdGeom.Xform.Define(stage, mount_path)
    UsdGeom.Xformable(mount.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*mount_offset))

    lidar_path = f"{mount_path}/ouster_lidar"
    lidar = Lidar.create(
        path=lidar_path,
        config=config,
        variant=variant,
        aux_output_level="BASIC",
        tick_rate=tick_rate,
        accumulate_outputs=True,
    )

    sensor = LidarSensor(lidar, annotators=[])

    class LidarGmoWriter(Writer):
        def __init__(self):
            self.data_structure = "renderProduct"
            self.annotators = [rep.annotators.get("GenericModelOutput")]

        def write(self, data):
            if "renderProducts" not in data:
                return
            for _rp, rp_data in data["renderProducts"].items():
                raw = rp_data.get("GenericModelOutput")
                if isinstance(raw, dict):
                    raw = raw.get("data")
                gmo = parse_generic_model_output_data(raw)
                if gmo.numElements > 0:
                    print(f"[LiDAR] {gmo.numElements} points received")

    rep.WriterRegistry.register(LidarGmoWriter)
    sensor.attach_writer("LidarGmoWriter")

    return lidar, sensor


def attach_imu(
    stage,
    robot_path: str = "/World/Robot",
    mount_offset: tuple = (0.0, 0.0, 0.1),
    tick_rate: float = 200.0,
):
    """Mount a physics IMU on the robot chassis.

    Args:
        stage: Active USD stage.
        robot_path: Prim path of the robot root.
        mount_offset: (x, y, z) translation from robot origin to IMU.
        tick_rate: Sensor output rate in Hz.

    Returns:
        Tuple of (IMU prim wrapper, IMUSensor).
    """
    from isaacsim.sensors.experimental.physics import IMU, IMUSensor

    mount_path = f"{robot_path}/imu_mount"
    mount = UsdGeom.Xform.Define(stage, mount_path)
    UsdGeom.Xformable(mount.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*mount_offset))

    imu_path = f"{mount_path}/imu"
    imu = IMU.create(path=imu_path, tick_rate=tick_rate)

    sensor = IMUSensor(
        imu,
        annotators=["linear_acceleration", "angular_velocity", "orientation"],
    )

    return imu, sensor


def attach_sensors(
    stage,
    robot_path: str = "/World/Robot",
    simulation_app=None,
):
    """Attach both an Ouster LiDAR and an IMU to a robot, then run the sim loop.

    Args:
        stage: Active USD stage.
        robot_path: Prim path of the robot root.
        simulation_app: Running SimulationApp instance for the update loop.
    """
    import isaacsim.core.experimental.utils.app as app_utils

    lidar, lidar_sensor = attach_ouster_lidar(stage, robot_path)
    imu, imu_sensor = attach_imu(stage, robot_path)

    app_utils.play(commit=True)

    if simulation_app is not None:
        while simulation_app.is_running():
            simulation_app.update()
            imu_data = imu_sensor.get_data()
            if imu_data is not None:
                print(f"[IMU] accel={imu_data.linear_acceleration}, " f"gyro={imu_data.angular_velocity}")
