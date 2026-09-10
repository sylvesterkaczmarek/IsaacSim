#!/usr/bin/env python3
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

"""Generate deterministic USD lighting override sublayers for Actor SDG."""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlparse

LIGHT_TYPES = {
    "CylinderLight",
    "DiskLight",
    "DistantLight",
    "DomeLight",
    "RectLight",
    "SphereLight",
}

DEFAULT_INTENSITY = {
    "CylinderLight": 1000.0,
    "DiskLight": 1000.0,
    "DistantLight": 3000.0,
    "DomeLight": 1000.0,
    "RectLight": 1000.0,
    "SphereLight": 30000.0,
}


class LightInfo(NamedTuple):
    """Existing USD light and its base intensity."""

    prim_path: str
    light_type: str
    base_intensity: float


def _read_attribute(prim: Any, name: str) -> Any | None:
    """Read an authored USD attribute value."""
    attribute = prim.GetAttribute(name)
    if attribute and attribute.HasValue():
        return attribute.Get()
    return None


def discover_lights(stage: Any) -> list[LightInfo]:
    """Return supported lights discovered in a USD stage.

    Args:
        stage: Open USD stage.

    Returns:
        Supported lights in stage traversal order.
    """
    lights: list[LightInfo] = []
    for prim in stage.Traverse():
        light_type = prim.GetTypeName()
        if light_type not in LIGHT_TYPES:
            continue
        intensity = _read_attribute(prim, "inputs:intensity")
        if intensity is None:
            intensity = DEFAULT_INTENSITY[light_type]
        lights.append(
            LightInfo(
                prim_path=str(prim.GetPath()),
                light_type=light_type,
                base_intensity=float(intensity),
            )
        )
    return lights


def format_light_summary(lights: list[LightInfo]) -> str:
    """Format discovered lights for a dry-run report.

    Args:
        lights: Discovered USD lights.

    Returns:
        Multi-line summary with prim path, type, and intensity.
    """
    lines = [f"Discovered {len(lights)} supported light(s):"]
    for light in lights:
        lines.append(f"  {light.prim_path} [{light.light_type}] intensity={light.base_intensity:g}")
    return "\n".join(lines)


def _sample_log_uniform(generator: random.Random, low: float, high: float) -> float:
    """Sample a value uniformly in logarithmic space."""
    return math.exp(generator.uniform(math.log(low), math.log(high)))


def _sample_color(generator: random.Random, max_deviation: float) -> tuple[float, float, float]:
    """Sample a color near white without changing peak brightness."""
    minimum = max(0.0, 1.0 - max_deviation)
    channels = [generator.uniform(minimum, 1.0) for _ in range(3)]
    brightest = max(channels)
    if brightest == 0.0:
        return (1.0, 1.0, 1.0)
    return tuple(channel / brightest for channel in channels)


def _build_overrides(
    lights: list[LightInfo],
    generator: random.Random,
    intensity_range: tuple[float, float],
    temperature_range: tuple[float, float],
    exposure_range: tuple[float, float],
    color_tint: float,
) -> dict[str, dict[str, Any]]:
    """Build one coherent set of lighting overrides."""
    groups: dict[str, list[LightInfo]] = {}
    for light in lights:
        groups.setdefault(light.light_type, []).append(light)

    overrides: dict[str, dict[str, Any]] = {}
    for light_type in sorted(groups):
        intensity_multiplier = _sample_log_uniform(generator, *intensity_range)
        temperature = generator.uniform(*temperature_range)
        exposure = generator.uniform(*exposure_range)
        color = _sample_color(generator, color_tint) if color_tint > 0.0 else None

        for light in groups[light_type]:
            multiplier = intensity_multiplier * generator.uniform(0.95, 1.05)
            values: dict[str, Any] = {
                "color_temperature": temperature,
                "exposure": exposure,
                "intensity": max(0.0, light.base_intensity * multiplier),
                "intensity_multiplier": multiplier,
            }
            if color is not None:
                values["color"] = color
            overrides[light.prim_path] = values
    return overrides


def _author_override_layer(overrides: dict[str, dict[str, Any]], output_path: Path) -> None:
    """Write lighting opinions to a USD sublayer."""
    from pxr import Gf, Sdf

    layer = Sdf.Layer.CreateAnonymous()
    for prim_path_text, values in overrides.items():
        prim_path = Sdf.Path(prim_path_text)
        for prefix in prim_path.GetPrefixes():
            if prefix == Sdf.Path.absoluteRootPath:
                continue
            prim_spec = Sdf.CreatePrimInLayer(layer, prefix)
            prim_spec.specifier = Sdf.SpecifierOver

        prim_spec = layer.GetPrimAtPath(prim_path)
        intensity = Sdf.AttributeSpec(prim_spec, "inputs:intensity", Sdf.ValueTypeNames.Float)
        intensity.default = values["intensity"]

        enable_temperature = Sdf.AttributeSpec(
            prim_spec,
            "inputs:enableColorTemperature",
            Sdf.ValueTypeNames.Bool,
        )
        enable_temperature.default = True

        temperature = Sdf.AttributeSpec(
            prim_spec,
            "inputs:colorTemperature",
            Sdf.ValueTypeNames.Float,
        )
        temperature.default = values["color_temperature"]

        exposure = Sdf.AttributeSpec(prim_spec, "inputs:exposure", Sdf.ValueTypeNames.Float)
        exposure.default = values["exposure"]

        if "color" in values:
            color = Sdf.AttributeSpec(prim_spec, "inputs:color", Sdf.ValueTypeNames.Color3f)
            color.default = Gf.Vec3f(*values["color"])

    if not layer.Export(str(output_path)):
        raise RuntimeError(f"Could not export USD layer: {output_path}")


def _remove_previous_outputs(output_dir: Path, prefix: str) -> None:
    """Remove only files managed by this generator."""
    for path in output_dir.glob(f"{prefix}_*.usda"):
        if path.is_file():
            path.unlink()
    manifest = output_dir / "manifest.csv"
    if manifest.is_file():
        manifest.unlink()


def _find_previous_outputs(output_dir: Path, prefix: str) -> list[Path]:
    """Return files that a new run would replace."""
    existing = sorted(path for path in output_dir.glob(f"{prefix}_*.usda") if path.is_file())
    manifest = output_dir / "manifest.csv"
    if manifest.is_file():
        existing.append(manifest)
    return existing


def generate_variations(
    lights: list[LightInfo],
    output_dir: Path,
    *,
    count: int,
    seed: int,
    intensity_range: tuple[float, float],
    temperature_range: tuple[float, float],
    exposure_range: tuple[float, float],
    color_tint: float,
    prefix: str,
) -> list[Path]:
    """Generate lighting sublayers and a CSV manifest.

    Args:
        lights: Discovered USD lights.
        output_dir: Directory for sublayers and the manifest.
        count: Number of sublayers to generate.
        seed: Base random seed.
        intensity_range: Low and high intensity multipliers.
        temperature_range: Low and high color temperatures in Kelvin.
        exposure_range: Low and high exposure values.
        color_tint: Maximum per-channel deviation from white.
        prefix: Generated filename prefix.

    Returns:
        Absolute paths to generated sublayers.

    Raises:
        ValueError: If ``lights`` is empty or ``count`` is not positive. Either one leaves the
            manifest with no rows, so it would have no columns to write. With no lights there is
            nothing to override and every variation would be an empty sublayer; with no
            variations nothing is generated at all. ``main()`` rejects both at the call site;
            these guards cover direct callers of the public API.
    """
    if not lights:
        raise ValueError("lights is empty; nothing to vary. Discover lights with discover_lights(stage) first.")
    if count <= 0:
        raise ValueError(f"count must be positive; got {count}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []

    for variation_index in range(count):
        variation_seed = (seed * 2654435761 + variation_index) & 0xFFFFFFFF
        generator = random.Random(variation_seed)
        overrides = _build_overrides(
            lights,
            generator,
            intensity_range,
            temperature_range,
            exposure_range,
            color_tint,
        )
        output_path = (output_dir / f"{prefix}_{variation_index:04d}.usda").resolve()
        _author_override_layer(overrides, output_path)
        generated.append(output_path)

        light_by_path = {light.prim_path: light for light in lights}
        for prim_path, values in overrides.items():
            color = values.get("color")
            manifest_rows.append(
                {
                    "variation_index": variation_index,
                    "file_name": output_path.name,
                    "seed": variation_seed,
                    "prim_path": prim_path,
                    "light_type": light_by_path[prim_path].light_type,
                    "intensity_multiplier": f"{values['intensity_multiplier']:.6f}",
                    "new_intensity": f"{values['intensity']:.6f}",
                    "new_temperature": f"{values['color_temperature']:.6f}",
                    "new_exposure": f"{values['exposure']:.6f}",
                    "new_color_r": f"{color[0]:.6f}" if color else "",
                    "new_color_g": f"{color[1]:.6f}" if color else "",
                    "new_color_b": f"{color[2]:.6f}" if color else "",
                }
            )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return generated


def _resolve_stage_path(value: str, parser: argparse.ArgumentParser) -> str:
    """Validate a local USD path or an Omniverse URL."""
    parsed = urlparse(value)
    if parsed.scheme == "omniverse":
        return value
    if parsed.scheme:
        parser.error("base_stage must be a local path or an omniverse:// URL")

    path = Path(value).expanduser().resolve()
    if not path.is_file():
        parser.error(f"base stage not found: {path}")
    return str(path)


def _start_omniverse_runtime(stage_path: str, parser: argparse.ArgumentParser) -> Any | None:
    """Start a headless Kit runtime when an Omniverse URL requires one."""
    if urlparse(stage_path).scheme != "omniverse":
        return None

    try:
        import omni.kit.app

        if omni.kit.app.get_app() is not None:
            return None
    except (ImportError, RuntimeError):
        pass

    try:
        from isaacsim import SimulationApp
    except ImportError:
        parser.error(
            "omniverse:// inputs require an initialized Isaac Sim runtime; "
            "run this script with the Isaac Sim Python launcher"
        )
    return SimulationApp({"headless": True})


def _validate_range(
    parser: argparse.ArgumentParser,
    option: str,
    values: list[float],
    *,
    positive: bool = False,
) -> tuple[float, float]:
    """Validate an ordered two-value numeric range."""
    low, high = values
    if low >= high:
        parser.error(f"{option} requires LOW < HIGH")
    if positive and low <= 0.0:
        parser.error(f"{option} values must be positive")
    return (low, high)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_stage", help="Local USD stage path or omniverse:// URL.")
    parser.add_argument("--count", type=int, default=10, help="Number of variations. Default: 10.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed. Default: 42.")
    parser.add_argument(
        "--output-dir",
        default="./lighting_variations",
        help="Output directory. Default: ./lighting_variations.",
    )
    parser.add_argument(
        "--intensity-range",
        type=float,
        nargs=2,
        default=[0.3, 3.0],
        metavar=("LOW", "HIGH"),
        help="Intensity multiplier range. Default: 0.3 3.0.",
    )
    parser.add_argument(
        "--temperature-range",
        type=float,
        nargs=2,
        default=[2700.0, 7500.0],
        metavar=("LOW", "HIGH"),
        help="Color temperature range in Kelvin. Default: 2700 7500.",
    )
    parser.add_argument(
        "--exposure-range",
        type=float,
        nargs=2,
        default=[-2.0, 2.0],
        metavar=("LOW", "HIGH"),
        help="Exposure range. Default: -2.0 2.0.",
    )
    parser.add_argument(
        "--color-tint",
        type=float,
        default=0.0,
        metavar="MAX_DEVIATION",
        help="Maximum per-channel color deviation from white, from 0 through 1. Default: 0.",
    )
    parser.add_argument("--prefix", default="lighting_var", help="Generated filename prefix.")
    parser.add_argument("--dry-run", action="store_true", help="Scan lights without writing files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files from a previous run with the same prefix.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the lighting variation command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.count <= 0:
        parser.error("--count must be positive")
    if not 0.0 <= args.color_tint <= 1.0:
        parser.error("--color-tint must be from 0 through 1")
    if not args.prefix or Path(args.prefix).name != args.prefix:
        parser.error("--prefix must be a filename prefix without path separators")

    intensity_range = _validate_range(
        parser,
        "--intensity-range",
        args.intensity_range,
        positive=True,
    )
    temperature_range = _validate_range(parser, "--temperature-range", args.temperature_range)
    exposure_range = _validate_range(parser, "--exposure-range", args.exposure_range)
    stage_path = _resolve_stage_path(args.base_stage, parser)
    simulation_app = _start_omniverse_runtime(stage_path, parser)

    try:
        try:
            from pxr import Usd
        except ImportError:
            parser.error("pxr is unavailable; run this script with the Isaac Sim Python launcher")

        print(f"Opening base stage: {stage_path}")
        stage = Usd.Stage.Open(stage_path, Usd.Stage.LoadAll)
        if stage is None:
            parser.error(f"could not open base stage: {stage_path}")

        lights = discover_lights(stage)
        if not lights:
            parser.error(f"no supported light prims found; expected one of: {', '.join(sorted(LIGHT_TYPES))}")
        print(format_light_summary(lights))
        if args.dry_run:
            print("Dry-run complete. No files were written.")
            return 0

        output_dir = Path(args.output_dir).expanduser().resolve()
        existing = _find_previous_outputs(output_dir, args.prefix) if output_dir.is_dir() else []
        if existing and not args.overwrite:
            parser.error(
                f"{len(existing)} generated output file(s) already exist in {output_dir}; "
                "inspect them and pass --overwrite to overwrite them"
            )
        if existing:
            _remove_previous_outputs(output_dir, args.prefix)

        generated = generate_variations(
            lights,
            output_dir,
            count=args.count,
            seed=args.seed,
            intensity_range=intensity_range,
            temperature_range=temperature_range,
            exposure_range=exposure_range,
            color_tint=args.color_tint,
            prefix=args.prefix,
        )
        print(f"Generated {len(generated)} lighting variation(s) in {output_dir}")
        print(f"Manifest: {output_dir / 'manifest.csv'}")
        print(f"Seed: {args.seed}")
        return 0
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
