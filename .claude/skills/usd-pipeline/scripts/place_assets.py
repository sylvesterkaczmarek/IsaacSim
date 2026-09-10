"""Place USD assets at block positions with bbox-center offset correction.

Implements the placeholder-to-asset swap pipeline: collect visible cube blocks
from the stage, measure asset bboxes, then place assets at block positions
corrected for the asset's bbox center offset.
"""

import os
from collections import defaultdict


def get_asset_bbox(stage, asset_path: str, app) -> dict:
    """Reference an asset temporarily to compute its accurate bounding box.

    Args:
        stage: Active USD stage (inside a SimulationApp).
        asset_path: Path to the asset USD file.
        app: Running SimulationApp instance.

    Returns:
        Dict with keys cx, cy, cz (bbox center X, Y, and base Z).
        Returns None if bbox is invalid.
    """
    from pxr import Usd, UsdGeom

    bc = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    name = os.path.basename(asset_path).replace(".", "_")
    test = stage.DefinePrim(f"/BBoxTest_{name}", "Xform")
    test.GetReferences().AddReference(asset_path)
    for _ in range(5):
        app.update()
    r = bc.ComputeWorldBound(test).ComputeAlignedRange()
    mn, mx = r.GetMin(), r.GetMax()
    stage.RemovePrim(f"/BBoxTest_{name}")
    if mn[0] > 1e30:
        return None
    return {"cx": (mn[0] + mx[0]) / 2, "cy": (mn[1] + mx[1]) / 2, "cz": mn[2]}


def collect_blocks(stage) -> dict:
    """Collect visible UsdGeom.Cube prims grouped by name prefix.

    Args:
        stage: Active USD stage.

    Returns:
        Dict mapping prefix string to list of block dicts {name, path, x, y, z}.
    """
    from pxr import Usd, UsdGeom

    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    blocks = defaultdict(list)
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Cube):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility(Usd.TimeCode.Default()) == "invisible":
            continue
        name = prim.GetName()
        prefix = ""
        for c in name:
            if c.isdigit():
                break
            prefix += c
        mtx = xf_cache.GetLocalToWorldTransform(prim)
        pos = mtx.ExtractTranslation()
        blocks[prefix].append(
            {
                "name": name,
                "path": str(prim.GetPath()),
                "x": pos[0],
                "y": pos[1],
                "z": pos[2],
            }
        )
    return blocks


def place_assets(stage, blocks: dict, asset_map: dict, asset_bboxes: dict, module_name: str) -> int:
    """Place real assets at cube-block positions with bbox-center offset correction.

    Args:
        stage: Active USD stage.
        blocks: Output from collect_blocks().
        asset_map: Dict mapping block prefix to asset USD path.
        asset_bboxes: Dict mapping asset path to bbox dict from get_asset_bbox().
        module_name: Name for the root Xform that holds placed assets.

    Returns:
        Number of assets placed.
    """
    from pxr import Gf, UsdGeom

    stage.DefinePrim(f"/World/{module_name}", "Xform")
    placed = 0

    for prefix, asset_path in asset_map.items():
        block_list = blocks.get(prefix, [])
        if not block_list:
            continue
        bb = asset_bboxes.get(asset_path)
        if not bb:
            continue

        stage.DefinePrim(f"/World/{module_name}/{prefix}", "Xform")

        for b in block_list:
            prim = stage.DefinePrim(f"/World/{module_name}/{prefix}/{b['name']}", "Xform")
            xf = UsdGeom.Xformable(prim)
            # OFFSET CORRECTION: translate so asset bbox center lands at block position
            tx = b["x"] - bb["cx"]
            ty = b["y"] - bb["cy"]
            tz = -bb["cz"]  # ground the asset
            xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(tx, ty, tz))
            prim.GetReferences().AddReference(asset_path)
            placed += 1

        # Hide original cubes
        for b in block_list:
            orig = stage.GetPrimAtPath(b["path"])
            if orig:
                UsdGeom.Imageable(orig).MakeInvisible()

    return placed
