#!/usr/bin/env python3
"""Generate a self-contained IRO bin-packing description file offline.

Emits a valid ``isaacsim.replicator.object`` YAML whose ``bin_pack`` harmonizer packs SimReady
cardboard boxes streamed from the Omniverse content server (no local assets, no ``PATH_TO_BOXES``
placeholder), plus a camera, a dome light, and the common output switches so it renders as-is.

The packed layout is gravity-stable: the ``bin_pack`` harmonizer places boxes bottom-up and only
where a box rests on the bin floor or is supported from below, so nothing floats. See the
``object-bin-packing`` SKILL for the headless run command that turns this config into rendered
frames.

The config ``version`` must match the INSTALLED ``isaacsim.replicator.object.core`` version. Prefer
``--from-ext <path>`` to derive it from the build so the config stays in sync; ``--version`` is an
explicit override.

Usage:
    python3 bin_pack_config.py (--from-ext <build|ext dir|extension.toml> | --version <X.Y.Z>)
        --output-path /tmp/iro_bin_pack [--bin-size 400 200 200] [--fill-ratio 0.6]
        [--scale-counts 1.0] [--frames 3] [--seed 21]
    python3 bin_pack_config.py --from-ext "$ISAAC_SIM_DIR" --output-path /tmp/x --validate  # offline gate

Prints YAML to stdout (unless ``--validate``). Exit 0 on success, 2 on bad arguments, 1 on a
failed ``--validate``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

OBJECT_EXT = "isaacsim.replicator.object.core"

# Content-server macro chain (defined inline so the config is self-contained for a headless run
# and does not depend on the extension's global.yaml being merged).
AWS_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
CARDBOARD_ROOT = "$[/server_root]/Assets/DigitalTwin/Assets/Warehouse/Shipping/Cardboard_Boxes"

# Curated SimReady cardboard boxes spanning ~15 cm flats to ~77 cm cubes, in a brown/white mix.
# White_A / Flat_A assets attach materials via a payload; IRO's GMesh auto-disables scene_instance
# caching for them (otherwise cached clones render magenta), so no per-group is_instance flag is
# needed here. This mirrors the shipped demo_bin_pack.yaml that CI renders.
#   (group name, base count, relative usd path under cardboard_boxes_root)
BOX_GROUPS = [
    ("box_a", 4, "Cube_A/CubeBox_A09_77cm_PR_NVD_01.usd"),
    ("box_b", 4, "White_A/WhiteCorrugatedBox_A13_61x61x61cm_PR_NVD_01.usd"),
    ("box_c", 8, "Multi-Depth_A/MultiDepthBox_A12_36x51x36xcm_PR_NVD_01.usd"),
    ("box_d", 8, "Multi-Depth_A/MultiDepthBox_A09_41x41x42cm_PR_NVD_01.usd"),
    ("box_e", 12, "White_A/WhiteCorrugatedBox_A08_30x30x30cm_PR_NVD_01.usd"),
    ("box_f", 12, "Printer_A/PrintersBox_A05_23x28x25cm_PR_NVD_01.usd"),
    ("box_g", 16, "White_A/WhiteCorrugatedBox_A24_20x26x15cm_PR_NVD_01.usd"),
]


def read_ext_version(path: str | Path, ext_name: str) -> str:
    """Read `version = "X"` from an extension.toml, an extension dir, or a build/extscache root.

    Offline (just reads a TOML file) — no Isaac Sim launch. Raises SystemExit if not found.
    """
    p = Path(path).expanduser()
    if p.is_file():
        tomls = [p]
    elif p.is_dir():
        direct = p / "config" / "extension.toml"
        if direct.is_file():
            tomls = [direct]
        else:
            # Search a build/extscache root: match ONLY the exact extension dir or an extscache
            # "<ext_name>-<version>" dir (never a same-prefix different extension), depth 1 first
            # then recursively.
            pats = [f"{ext_name}/config/extension.toml", f"{ext_name}-*/config/extension.toml"]
            tomls = [t for pat in pats for t in sorted(p.glob(pat))]
            if not tomls:
                tomls = [t for pat in pats for t in sorted(p.glob("**/" + pat))]
    else:
        raise SystemExit(f"--from-ext path not found: {path}")
    versions = []
    for toml in tomls:
        m = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"', toml.read_text())
        if m and m.group(1) not in versions:
            versions.append(m.group(1))
    if not versions:
        raise SystemExit(f"could not find a version for {ext_name} under: {path}")
    if len(versions) > 1:
        raise SystemExit(
            f"multiple {ext_name} versions found under {path}: {', '.join(sorted(versions))}; "
            "point --from-ext at the specific extension dir or its extension.toml"
        )
    return versions[0]


def _randomized_range(start, end) -> dict:
    return {"distribution_type": "range", "start": start, "end": end}


def _box_group(count: int, rel_usd: str) -> dict:
    return {
        "count": count,
        "type": "geometry",
        "subtype": "mesh",
        "tracked": True,  # required to appear in labels / segmentation / 3D boxes
        "transform_operators": [
            {
                "transform": {
                    "distribution_type": "harmonized",
                    "harmonizer_name": "bin_pack_H",
                    "pitch": "local_aabb",
                }
            },
            {"scale": _randomized_range([1.2, 1.2, 1.2], [1.25, 1.25, 1.25])},
        ],
        "usd_path": f"$[/cardboard_boxes_root]/{rel_usd}",
    }


def build_config(
    version: str,
    output_path: str,
    bin_size: list[int],
    fill_ratio: float | None,
    scale_counts: float,
    frames: int,
    seed: int,
) -> dict:
    harmonizer = {"harmonizer_type": "bin_pack", "bin_size": list(bin_size)}
    if fill_ratio is not None:
        harmonizer["fill_ratio"] = fill_ratio  # cap packed volume at this fraction of the bin

    body = {
        # required settings
        "version": version,
        "num_frames": frames,
        "seed": seed,
        "output_path": output_path,
        "screen_width": 3840,
        "screen_height": 2160,
        # only tracked geometry is captured; enable the common output switches
        "output_switches": {"images": True, "labels": True, "segmentation": True, "3d_labels": True},
        # content-server macro chain (self-contained: no global.yaml merge, no PATH_TO_BOXES)
        "aws_root": AWS_ROOT,
        "server_root": "$[/aws_root]",
        "cardboard_boxes_root": CARDBOARD_ROOT,
        # camera
        "focal_length": 14.228393962367306,
        "horizontal_aperture": 20.955,
        "camera_parameters": {
            "screen_width": "$[/screen_width]",
            "screen_height": "$[/screen_height]",
            "focal_length": "$[/focal_length]",
            "horizontal_aperture": "$[/horizontal_aperture]",
            "near_clip": 0.001,
            "far_clip": 100000,
        },
        "default_camera": {
            "type": "camera",
            "camera_parameters": "$[/camera_parameters]",
            "transform_operators": [
                {"rotateX": -30},
                {"rotateY": 45},
                {"translate": [0, 0, 1000]},
            ],
        },
        # light
        "dome_light": {
            "type": "light",
            "subtype": "dome",
            "intensity": 1500,
            "transform_operators": [{"rotateX": 270}],
        },
        # the bin-pack harmonizer that correlates all box placements
        "bin_pack_H": harmonizer,
    }

    # box groups (packed by bin_pack_H)
    for name, base_count, rel_usd in BOX_GROUPS:
        count = max(1, round(base_count * scale_counts))
        body[name] = _box_group(count, rel_usd)

    return {"isaacsim.replicator.object": body}


def validate_config(config: dict) -> list[str]:
    """Offline structural gate: return a list of problems (empty = OK to run on IRO).

    Checks the invariants that make an IRO bin-pack config actually runnable without local assets:
    a single root key, a version, exactly one bin_pack harmonizer, at least one box group wired to
    it, and no unresolved PATH_TO_* placeholders. Does not launch Isaac Sim.
    """
    problems: list[str] = []
    if list(config.keys()) != ["isaacsim.replicator.object"]:
        problems.append("root key must be exactly 'isaacsim.replicator.object'")
        return problems
    body = config["isaacsim.replicator.object"]
    if not body.get("version"):
        problems.append("missing 'version' (must equal the installed isaacsim.replicator.object.core)")

    harmonizers = [k for k, v in body.items() if isinstance(v, dict) and v.get("harmonizer_type") == "bin_pack"]
    if len(harmonizers) != 1:
        problems.append(f"expected exactly one bin_pack harmonizer, found {len(harmonizers)}")
    hname = harmonizers[0] if harmonizers else None

    def uses_harmonizer(mut: dict) -> bool:
        for op in mut.get("transform_operators", []):
            t = op.get("transform")
            if isinstance(t, dict) and t.get("harmonizer_name") == hname:
                return True
        return False

    box_groups = [
        k for k, v in body.items() if isinstance(v, dict) and v.get("type") == "geometry" and uses_harmonizer(v)
    ]
    if not box_groups:
        problems.append("no geometry group references the bin_pack harmonizer")
    for k in box_groups:
        if not body[k].get("tracked"):
            problems.append(f"group '{k}' is not tracked:true (won't appear in labels/segmentation)")

    if "PATH_TO_" in _dump_str(config):
        problems.append("config still contains a PATH_TO_* placeholder (won't run without editing)")
    return problems


def _dump_str(config: dict) -> str:
    import yaml

    return yaml.safe_dump(config, sort_keys=False, default_flow_style=False)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    vg = p.add_mutually_exclusive_group(required=True)
    vg.add_argument(
        "--from-ext",
        help="Derive the config version from the installed isaacsim.replicator.object.core: "
        "an extension.toml, an extension dir, or a build/extscache root to search.",
    )
    vg.add_argument("--version", help="Explicit config version = installed isaacsim.replicator.object.core version.")
    p.add_argument("--output-path", required=True, help="Settings output_path (a writable folder).")
    p.add_argument(
        "--bin-size",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=[400, 200, 200],
        help="bin_pack cuboid dimensions (default: 400 200 200).",
    )
    p.add_argument(
        "--fill-ratio",
        type=float,
        default=None,
        help="Optional (0, 1]: cap packed box volume at this fraction of the bin (partially loaded bins).",
    )
    p.add_argument(
        "--scale-counts",
        type=float,
        default=1.0,
        help="Multiply every box group's count by this factor (default 1.0; total is ~64 at 1.0).",
    )
    p.add_argument("--frames", type=int, default=3, help="num_frames.")
    p.add_argument("--seed", type=int, default=21)
    p.add_argument(
        "--validate",
        action="store_true",
        help="Offline structural gate: print OK / problems and exit (does not emit YAML or launch Isaac Sim).",
    )
    args = p.parse_args(argv)

    if args.frames < 1:
        p.error("--frames must be >= 1")
    if args.scale_counts <= 0:
        p.error("--scale-counts must be > 0")
    if args.fill_ratio is not None and not 0.0 < args.fill_ratio <= 1.0:
        p.error("--fill-ratio must be in (0, 1]")
    if any(d <= 0 for d in args.bin_size):
        p.error("--bin-size dimensions must be > 0")

    try:
        import yaml  # lazy import so the module compiles without PyYAML installed  # noqa: F401
    except ImportError:
        raise SystemExit("PyYAML is required to emit YAML (pip install pyyaml)")

    version = args.version or read_ext_version(args.from_ext, OBJECT_EXT)
    config = build_config(
        version, args.output_path, args.bin_size, args.fill_ratio, args.scale_counts, args.frames, args.seed
    )

    if args.validate:
        problems = validate_config(config)
        if problems:
            print("INVALID:")
            for pr in problems:
                print(f"  - {pr}")
            return 1
        print("OK: config is a runnable IRO bin-pack description (self-contained, no local assets).")
        return 0

    yaml.safe_dump(config, sys.stdout, sort_keys=False, default_flow_style=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
