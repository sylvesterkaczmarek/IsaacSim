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

"""Measure a USD asset: bounding box, shader type, and prim count.

Uses Kit Python (no renderer needed) for fast batch measurement of USD assets.
"""

import glob
import json
import os
from argparse import ArgumentParser


def measure_asset(path: str) -> dict:
    """Measure a USD asset: bbox dimensions, shader compatibility, prim count.

    Args:
        path: Absolute path to the USD file (.usd, .usda, or .usdc).

    Returns:
        Dict with keys: width, depth, height, center, mpu, prims, dual_shader,
        shader_tag.  Returns {"error": <reason>} on failure.
    """
    if not os.path.exists(path):
        return None

    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(path)
    dp = stage.GetDefaultPrim()
    if not dp:
        children = list(stage.GetPseudoRoot().GetChildren())
        dp = children[0] if children else None
    if not dp:
        return {"error": "no default prim"}

    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    r = bc.ComputeWorldBound(dp).ComputeAlignedRange()
    mn, mx = r.GetMin(), r.GetMax()

    if mn[0] > 1e30:
        return {"error": "invalid bbox"}

    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    w = (mx[0] - mn[0]) * mpu
    d = (mx[1] - mn[1]) * mpu
    h = (mx[2] - mn[2]) * mpu
    cx = (mn[0] + mx[0]) / 2
    cy = (mn[1] + mx[1]) / 2
    cz = mn[2]  # base of asset

    # Check for UsdPreviewSurface (headless-compatible)
    has_preview_surface = False
    for p in stage.Traverse():
        if p.IsA(UsdShade.Shader):
            sid = p.GetAttribute("info:id")
            if sid and sid.Get() and "Preview" in str(sid.Get()):
                has_preview_surface = True
                break

    prims = sum(1 for _ in stage.Traverse())

    return {
        "width": w,
        "depth": d,
        "height": h,
        "center": (cx, cy, cz),
        "mpu": mpu,
        "prims": prims,
        "dual_shader": has_preview_surface,
        "shader_tag": "dual" if has_preview_surface else "MDL-only",
    }


def catalog_assets(root_dir: str, extensions: tuple = (".usd", ".usda", ".usdc")) -> dict:
    """Recursively find and measure all USD assets in a directory tree.

    Args:
        root_dir: Root directory to search.
        extensions: File extensions to include.

    Returns:
        Dict mapping asset name to measurement dict (from measure_asset).
    """
    results = {}
    for ext in extensions:
        for path in glob.glob(f"{root_dir}/**/*{ext}", recursive=True):
            info = measure_asset(path)
            if info and "error" not in info:
                name = os.path.basename(path)
                for e in extensions:
                    name = name.replace(e, "")
                results[name] = {**info, "path": path}
    return results


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--asset", help="Path to a single USD asset to measure")
    group.add_argument("--root-dir", help="Directory tree to recursively catalog")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".usd", ".usda", ".usdc"],
        help="USD file extensions to include when cataloging",
    )
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.asset:
        result = measure_asset(args.asset)
    else:
        result = catalog_assets(args.root_dir, tuple(args.extensions))

    indent = 2 if args.pretty else None
    print(json.dumps(result, indent=indent, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
