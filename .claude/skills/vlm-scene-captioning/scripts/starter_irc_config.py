#!/usr/bin/env python3
"""Generate a starter standalone IRC (VLM Scene Captioning) config file offline.

Emits a minimal-but-valid ``isaacsim.replicator.caption.core`` YAML with global keys and a
``caption_configs`` block. It does NOT launch Isaac Sim, so it runs anywhere.

IRC requires the config `version` to be an EXACT match to `settings.VERSION` in the installed
isaacsim.replicator.caption.core -- which is NOT the extension's own `[package] version`. Prefer
`--from-ext <path>` to derive it; `--version` is an explicit override (rejected by IRC if wrong).

Usage:
    python3 starter_irc_config.py (--from-ext <build|ext dir|extension.toml> | --version <X.Y.Z>)
        --camera-prim-path /World/Cameras/Camera --output-path /tmp/irc_out
        [--scene-path scene.usd] [--captions brief,global]

Prints YAML to stdout. Exit 0 on success, 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

CAPTION_EXT = "isaacsim.replicator.caption.core"
CAPTION_TYPES = ["brief", "global", "qa"]


def read_config_version(path: str | Path, ext_name: str) -> str:
    """Read the config-schema version IRC validates against, from an installed extension.

    IRC compares the config's `version` against ``settings.VERSION`` in
    ``isaacsim/replicator/caption/core/settings.py`` -- NOT against the extension's
    ``[package] version`` in extension.toml. The two diverged: ``settings.VERSION`` has been
    frozen since the release that introduced config-file support, while the extension itself
    has moved on. Reading extension.toml therefore produces a config IRC always rejects with
    "Config file version X does not match the current version Y".

    Accepts an extension.toml, an extension dir, or a build/extscache root. Offline (just
    reads files) -- no Isaac Sim launch. Raises SystemExit if not found.
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
    if not tomls:
        raise SystemExit(f"could not find {ext_name} under: {path}")

    # extension.toml lives at <ext_root>/config/extension.toml
    settings_rel = Path(*ext_name.split(".")) / "settings.py"
    versions = []
    for toml in tomls:
        settings_py = toml.parent.parent / settings_rel
        if not settings_py.is_file():
            continue
        m = re.search(r'(?m)^\s*VERSION\s*=\s*"([^"]+)"', settings_py.read_text())
        if m and m.group(1) not in versions:
            versions.append(m.group(1))

    if not versions:
        raise SystemExit(
            f"could not read VERSION from {settings_rel} for {ext_name} under: {path}. "
            f"Point --from-ext at an installed extension (not a source checkout), or pass "
            f"--version with the value of settings.VERSION."
        )
    if len(versions) > 1:
        raise SystemExit(
            f"multiple {ext_name} config versions found under {path}: {', '.join(sorted(versions))}; "
            "point --from-ext at the specific extension dir or its extension.toml"
        )
    return versions[0]


def build_config(
    version: str,
    camera_prim_path: str,
    output_path: str,
    scene_path: str | None,
    captions: list[str],
) -> dict:
    caption_configs = {
        "save_full_scene_graph": True,
        "save_pruned_scene_graph": True,
        "attach_label_to_usd": False,
        "use_ai_label": False,
        "visualize_caption": True,
        "max_object_capacity": 100,
        "export_edges": True,
        # which caption types to generate (needs NVIDIA_API_KEY at runtime)
        "global_caption": "global" in captions,
        "qa_caption": "qa" in captions,
        "brief_caption": "brief" in captions,
        "pruning_ratio": 1.0,
        "verbose": True,
        "random_seed": 0,
        "caption_only": False,
        "export_world": True,
    }
    body = {
        "version": version,
        "camera_prim_path": camera_prim_path,
        "output_path": output_path,
        "caption_configs": caption_configs,
    }
    if scene_path:
        # place scene_path before caption_configs for readability
        body = {
            "version": version,
            "camera_prim_path": camera_prim_path,
            "scene_path": scene_path,
            "output_path": output_path,
            "caption_configs": caption_configs,
        }
    return {"isaacsim.replicator.caption.core": body}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    vg = p.add_mutually_exclusive_group(required=True)
    vg.add_argument(
        "--from-ext",
        help="Derive the config version from the installed isaacsim.replicator.caption.core: "
        "an extension.toml, an extension dir, or a build/extscache root to search.",
    )
    vg.add_argument(
        "--version", help="Explicit config version = settings.VERSION of the installed extension (exact match)."
    )
    p.add_argument("--camera-prim-path", required=True, help="USD path to an existing camera prim to caption from.")
    p.add_argument("--output-path", required=True, help="Output directory for captions/graphs.")
    p.add_argument("--scene-path", default=None, help="USD scene to load; omit to caption the already-loaded stage.")
    p.add_argument(
        "--captions", default="brief,global", help="Comma-separated caption types to enable: brief,global,qa."
    )
    args = p.parse_args(argv)

    captions = [c.strip() for c in args.captions.split(",") if c.strip()]
    bad = [c for c in captions if c not in CAPTION_TYPES]
    if bad:
        p.error(f"unknown caption type(s) {bad}; choose from {CAPTION_TYPES}")

    try:
        import yaml  # lazy import so the module compiles without PyYAML installed
    except ImportError:
        raise SystemExit("PyYAML is required to emit YAML (pip install pyyaml)")

    version = args.version or read_config_version(args.from_ext, CAPTION_EXT)
    config = build_config(version, args.camera_prim_path, args.output_path, args.scene_path, captions)
    yaml.safe_dump(config, sys.stdout, sort_keys=False, default_flow_style=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
