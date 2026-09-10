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

"""Verify the Surface Gripper tutorial by authoring its UR10 USD and stacking two cubes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path


def _author_usd(output_path: Path) -> None:
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    exit_code = 0
    try:
        import omni.kit.app

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "isaacsim.robot.surface_gripper", True
        )

        import isaacsim.core.experimental.utils.prim as prim_utils
        import usd.schema.isaac.robot_schema as robot_schema
        from isaacsim.storage.native import get_assets_root_path
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateNew(str(output_path))
        robot = stage.DefinePrim("/ur10", "Xform")
        robot.GetReferences().AddReference(
            get_assets_root_path() + "/Isaac/Robots_Multiphysics/UniversalRobots/ur10/ur10.usda"
        )
        stage.SetDefaultPrim(robot)

        for name, position, scale in (
            ("base", 0.05, (0.075, 0.075, 0.1)),
            ("tube", 0.125, (0.025, 0.025, 0.05)),
            ("suction_cup", 0.15, (0.075, 0.075, 0.015)),
        ):
            cylinder = UsdGeom.Cylinder.Define(stage, f"/ur10/ee_link/{name}")
            cylinder.CreateHeightAttr(1.0)
            cylinder.CreateRadiusAttr(0.5)
            cylinder.AddTranslateOp().Set(Gf.Vec3d(position, 0.0, 0.0))
            cylinder.AddOrientOp().Set(Gf.Quatf(0.70710678, 0.0, 0.70710678, 0.0))
            cylinder.AddScaleOp().Set(Gf.Vec3d(*scale))

        joint_path = Sdf.Path("/ur10/ee_link/surface_gripper/suction_joint")
        joint = UsdPhysics.Joint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/ur10/ee_link")])
        joint.CreateExcludeFromArticulationAttr(True)
        joint.CreateLocalPos0Attr(Gf.Vec3f(0.17, 0.0, 0.0))
        joint.CreateLocalRot0Attr(Gf.Quatf(0.70710678, 0.0, 0.70710678, 0.0))
        for axis in ("transX", "transY"):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(1.0)
            limit.CreateHighAttr(-1.0)
        for axis in ("rotX", "rotY", "rotZ"):
            limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
            limit.CreateLowAttr(-5.0)
            limit.CreateHighAttr(5.0)
        z_limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), "transZ")
        z_limit.CreateLowAttr(0.0)
        z_limit.CreateHighAttr(0.01)
        z_drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "transZ")
        z_drive.CreateStiffnessAttr(1000.0)
        z_drive.CreateDampingAttr(100.0)

        robot_schema.ApplyAttachmentPointAPI(joint.GetPrim())
        prim_utils.create_prim_attribute(
            joint.GetPrim(),
            name=robot_schema.Attributes.FORWARD_AXIS.name,
            type_name=robot_schema.Attributes.FORWARD_AXIS.type,
        ).Set("Z")
        gripper = robot_schema.CreateSurfaceGripper(stage, "/ur10/ee_link/SurfaceGripper")
        gripper.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name).SetTargets([joint_path])
        gripper.GetAttribute(robot_schema.Attributes.MAX_GRIP_DISTANCE.name).Set(0.02)
        if not stage.GetRootLayer().Export(str(output_path)):
            raise RuntimeError(f"Failed to write tutorial USD: {output_path}")
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    finally:
        app.close(exit_code=exit_code)
    if exit_code:
        raise SystemExit(exit_code)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-usd", type=Path)
    args, _ = parser.parse_known_args()
    if args.author_usd:
        _author_usd(args.author_usd)
        return

    stacking = Path(__file__).resolve().parents[2] / "api/isaacsim.robot_motion.examples/manipulation/stacking.py"
    with tempfile.TemporaryDirectory() as directory:
        usd_path = Path(directory) / "tutorial_ur10.usda"
        subprocess.run([sys.executable, __file__, "--author-usd", str(usd_path)], check=True)
        result = subprocess.run(
            [
                sys.executable,
                str(stacking),
                "--robot",
                "ur10",
                "--usd-path",
                str(usd_path),
                "--headless",
                "--test",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode or "PASS: ur10 stacked 2 cubes." not in result.stdout:
            message = f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
            raise RuntimeError(message)
        print("PASS: Surface Gripper tutorial stacked 2 cubes.")


if __name__ == "__main__":
    main()
