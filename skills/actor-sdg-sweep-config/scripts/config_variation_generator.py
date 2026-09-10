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

"""Generate deterministic Isaac Sim Replicator Agent configuration variations."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT_KEY = "isaacsim.replicator.agent"
VARIANT_DIRECTORY_PATTERN = re.compile(r"^variant_\d{4}$")


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping from disk.

    Args:
        path: YAML file to load.

    Returns:
        Parsed YAML mapping.

    Raises:
        RuntimeError: If PyYAML is unavailable.
        ValueError: If the YAML root is not a mapping.
    """
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install it with 'python3 -m pip install pyyaml'") from exc

    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    """Write a YAML mapping to disk."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install it with 'python3 -m pip install pyyaml'") from exc

    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, default_flow_style=False, sort_keys=False, allow_unicode=True)


def parse_path(path: str) -> list[str]:
    """Split an IRA dotted path while preserving its dotted root key.

    Args:
        path: Dotted configuration path.

    Returns:
        Configuration keys in traversal order.
    """
    if path == ROOT_KEY:
        return [ROOT_KEY]
    prefix = f"{ROOT_KEY}."
    if path.startswith(prefix):
        return [ROOT_KEY, *path[len(prefix) :].split(".")]
    return path.split(".")


def get_nested(mapping: dict[str, Any], keys: list[str]) -> Any:
    """Read a value from a nested mapping."""
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            raise KeyError(".".join(keys))
        current = current[key]
    return current


def set_nested(mapping: dict[str, Any], keys: list[str], value: Any) -> None:
    """Set a value below an existing parent mapping."""
    parent = get_nested(mapping, keys[:-1]) if len(keys) > 1 else mapping
    if not isinstance(parent, dict):
        raise KeyError(".".join(keys))
    parent[keys[-1]] = value


def _is_number(value: Any) -> bool:
    """Return whether a value is numeric but not Boolean."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_range(parameter: dict[str, Any], *, integers: bool) -> None:
    """Validate a two-value ordered sampling range."""
    values = parameter.get("range")
    valid_values = (
        isinstance(values, list)
        and len(values) == 2
        and all(_is_number(value) for value in values)
        and values[0] <= values[1]
    )
    if not valid_values:
        raise ValueError(f"{parameter['type']} parameter '{parameter['path']}' needs an ordered two-number range")
    if integers and not all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        raise ValueError(f"random_int parameter '{parameter['path']}' needs integer range values")


def validate_inputs(base: dict[str, Any], sweep: dict[str, Any]) -> None:
    """Validate the base IRA config and sweep specification.

    Args:
        base: Parsed base IRA configuration.
        sweep: Parsed sweep specification.

    Raises:
        ValueError: If either input violates the supported structure.
    """
    if not isinstance(base.get(ROOT_KEY), dict):
        raise ValueError(f"base config must contain the '{ROOT_KEY}' mapping")

    count = sweep.get("count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError("sweep 'count' must be a positive integer")

    parameters = sweep.get("parameters", [])
    if not isinstance(parameters, list):
        raise ValueError("sweep 'parameters' must be a list")

    for index, parameter in enumerate(parameters):
        if not isinstance(parameter, dict):
            raise ValueError(f"parameters[{index}] must be a mapping")
        path = parameter.get("path")
        sample_type = parameter.get("type")
        if not isinstance(path, str) or not path.startswith(f"{ROOT_KEY}."):
            raise ValueError(f"parameters[{index}].path must start with '{ROOT_KEY}.'")
        try:
            get_nested(base, parse_path(path))
        except KeyError as exc:
            if path != f"{ROOT_KEY}.seed":
                raise ValueError(f"sweep path does not exist in the base config: {path}") from exc

        if sample_type == "choice":
            values = parameter.get("values")
            if not isinstance(values, list) or not values:
                raise ValueError(f"choice parameter '{path}' needs a non-empty values list")
        elif sample_type in {"uniform", "linspace"}:
            _validate_range(parameter, integers=False)
            if sample_type == "linspace" and not isinstance(parameter.get("round_to_int", False), bool):
                raise ValueError(f"linspace parameter '{path}' has a non-Boolean round_to_int value")
        elif sample_type == "random_int":
            _validate_range(parameter, integers=True)
        elif sample_type == "fixed":
            if "value" not in parameter:
                raise ValueError(f"fixed parameter '{path}' needs a value")
        else:
            raise ValueError(
                f"parameter '{path}' has unsupported type {sample_type!r}; "
                "use choice, uniform, random_int, linspace, or fixed"
            )


def sample_value(
    parameter: dict[str, Any],
    index: int,
    count: int,
    generator: random.Random,
) -> Any:
    """Sample one parameter value from a validated sweep entry."""
    sample_type = parameter["type"]
    if sample_type == "choice":
        return generator.choice(parameter["values"])
    if sample_type == "uniform":
        return round(generator.uniform(*parameter["range"]), 6)
    if sample_type == "random_int":
        return generator.randint(*parameter["range"])
    if sample_type == "linspace":
        low, high = parameter["range"]
        value = low if count == 1 else low + (high - low) * index / (count - 1)
        return int(round(value)) if parameter.get("round_to_int", False) else round(value, 6)
    return parameter["value"]


def _patch_writer_output_directories(config: dict[str, Any], output_dir: Path) -> None:
    """Set a distinct output directory for each configured writer."""
    replicator = config[ROOT_KEY].get("replicator")
    if not isinstance(replicator, dict):
        return
    writers = replicator.get("writers")
    if not isinstance(writers, dict):
        return
    for writer_name, writer_config in writers.items():
        if isinstance(writer_config, dict):
            writer_config["output_dir"] = str((output_dir / writer_name).resolve())


def build_variants(
    base: dict[str, Any],
    sweep: dict[str, Any],
    output_dir: Path,
    seed: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Build variant configs and manifest rows without writing files."""
    count = sweep["count"]
    parameters = sweep.get("parameters", [])
    generator = random.Random(seed)
    seed_path = f"{ROOT_KEY}.seed"
    seed_is_swept = any(parameter["path"] == seed_path for parameter in parameters)
    variants: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for index in range(count):
        variant = copy.deepcopy(base)
        row: dict[str, Any] = {"variant_id": index}
        for parameter in parameters:
            value = sample_value(parameter, index, count, generator)
            set_nested(variant, parse_path(parameter["path"]), value)
            row[parameter["path"]] = value

        if not seed_is_swept:
            variant_seed = generator.randint(0, 2**32 - 1)
            variant[ROOT_KEY]["seed"] = variant_seed
            row[seed_path] = variant_seed

        variant_dir = output_dir / f"variant_{index:04d}"
        _patch_writer_output_directories(variant, variant_dir / "output")
        row["config_path"] = str(Path(variant_dir.name) / "config.yaml")
        variants.append((variant, row))
    return variants


def _load_schema_document(uri: str) -> dict[str, Any]:
    """Load one file URI for JSON Schema reference resolution."""
    parsed = urlparse(uri)
    path = Path(unquote(parsed.path))
    with path.open(encoding="utf-8") as stream:
        document = json.load(stream)
    document["$id"] = path.resolve().as_uri()
    return document


def find_schema_directory(explicit: Path | None = None) -> Path | None:
    """Find the installed IRA configuration schema directory."""
    if explicit is not None:
        candidate = explicit.expanduser().resolve()
        return candidate if (candidate / "config_schema.json").is_file() else None

    roots = [Path.cwd()]
    isaac_sim_dir = os.environ.get("ISAAC_SIM_DIR")
    if isaac_sim_dir:
        roots.append(Path(isaac_sim_dir).expanduser())

    direct_patterns = (
        "source/extensions/isaacsim.replicator.agent.core/data/config_schema",
        "extensions/isaacsim.replicator.agent.core/data/config_schema",
        "data/config_schema",
    )
    extension_patterns = (
        "extscache/isaacsim.replicator.agent.core-*/data/config_schema",
        "_build/*/release/extscache/isaacsim.replicator.agent.core-*/data/config_schema",
    )

    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root / pattern for pattern in direct_patterns)
        for pattern in extension_patterns:
            candidates.extend(sorted(root.glob(pattern)))

    for candidate in candidates:
        if (candidate / "config_schema.json").is_file():
            return candidate.resolve()
    return None


def build_schema_validator(schema_dir: Path | None) -> tuple[Any | None, str]:
    """Build the IRA JSON Schema validator and return its status."""
    if schema_dir is None:
        return None, "unavailable: IRA config_schema directory was not found"
    try:
        from jsonschema import Draft202012Validator
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012
    except ImportError:
        return None, "unavailable: jsonschema is not installed"

    schema_path = schema_dir / "config_schema.json"
    with schema_path.open(encoding="utf-8") as stream:
        schema = json.load(stream)
    schema_uri = schema_path.resolve().as_uri()
    schema["$id"] = schema_uri

    def retrieve(uri: str) -> Any:
        return Resource.from_contents(
            _load_schema_document(uri),
            default_specification=DRAFT202012,
        )

    registry = Registry(retrieve=retrieve).with_resource(
        schema_uri,
        Resource.from_contents(schema, default_specification=DRAFT202012),
    )
    return Draft202012Validator(schema, registry=registry), f"enabled: {schema_path}"


def validate_variants(
    variants: list[tuple[dict[str, Any], dict[str, Any]]],
    validator: Any,
) -> list[tuple[int, list[str]]]:
    """Return schema errors grouped by variant identifier."""
    failures: list[tuple[int, list[str]]] = []
    for config, row in variants:
        errors: list[str] = []
        for error in sorted(
            validator.iter_errors(config),
            key=lambda item: tuple(str(part) for part in item.absolute_path),
        ):
            path = ".".join(str(part) for part in error.absolute_path) or "(root)"
            errors.append(f"{path}: {error.message}")
        if errors:
            failures.append((row["variant_id"], errors))
    return failures


def _find_managed_outputs(output_dir: Path) -> list[Path]:
    """Find generated paths managed by this script."""
    if not output_dir.is_dir():
        return []
    managed = [
        path for path in output_dir.iterdir() if path.is_dir() and VARIANT_DIRECTORY_PATTERN.fullmatch(path.name)
    ]
    manifest = output_dir / "manifest.csv"
    if manifest.is_file():
        managed.append(manifest)
    return sorted(managed)


def _remove_managed_outputs(paths: list[Path]) -> None:
    """Remove only generated variant directories and their manifest."""
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def write_variants(
    variants: list[tuple[dict[str, Any], dict[str, Any]]],
    output_dir: Path,
) -> Path:
    """Write generated configs and return the manifest path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for config, row in variants:
        variant_dir = output_dir / f"variant_{row['variant_id']:04d}"
        variant_dir.mkdir()
        _write_yaml(variant_dir / "config.yaml", config)
        rows.append(row)

    manifest_path = output_dir / "manifest.csv"
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def print_plan(
    base_path: Path,
    sweep_path: Path,
    output_dir: Path,
    sweep: dict[str, Any],
    seed: int,
    validation_status: str,
) -> None:
    """Print a dry-run summary."""
    print(f"Base config: {base_path}")
    print(f"Sweep spec: {sweep_path}")
    print(f"Output directory: {output_dir}")
    print(f"Variant count: {sweep['count']}")
    print(f"Seed: {seed}")
    print(f"Schema validation: {validation_status}")
    parameters = sweep.get("parameters", [])
    if not parameters:
        print("Parameters: none; only seeds and writer outputs will differ")
        return
    print("Parameters:")
    for parameter in parameters:
        if parameter["type"] == "choice":
            detail = f"values={parameter['values']}"
        elif parameter["type"] in {"uniform", "random_int", "linspace"}:
            detail = f"range={parameter['range']}"
        else:
            detail = f"value={parameter['value']!r}"
        print(f"  {parameter['path']}: {parameter['type']} {detail}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Base IRA YAML configuration.")
    parser.add_argument("--sweep", required=True, type=Path, help="YAML sweep specification.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for variants.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed. Default: 42.")
    parser.add_argument("--schema-dir", type=Path, help="IRA config_schema directory.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and preview without writing files.")
    parser.add_argument(
        "--omit-schema-check",
        action="store_true",
        help=(
            "Write variants without JSON Schema verification. Escape hatch only when the IRA schema "
            "is unavailable; do not use unless the caller accepts unchecked YAML."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing variant_NNNN directories and manifest.csv.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the config variation command."""
    parser = build_parser()
    args = parser.parse_args(argv)
    base_path = args.base.expanduser().resolve()
    sweep_path = args.sweep.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not base_path.is_file():
        parser.error(f"base config not found: {base_path}")
    if not sweep_path.is_file():
        parser.error(f"sweep spec not found: {sweep_path}")
    if args.schema_dir is not None and find_schema_directory(args.schema_dir) is None:
        parser.error(f"config_schema.json not found under --schema-dir: {args.schema_dir}")

    try:
        base = _load_yaml(base_path)
        sweep = _load_yaml(sweep_path)
        validate_inputs(base, sweep)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    variants = build_variants(base, sweep, output_dir, args.seed)
    if args.omit_schema_check:
        validator = None
        validation_status = "skipped by --omit-schema-check"
    else:
        schema_dir = find_schema_directory(args.schema_dir)
        try:
            validator, validation_status = build_schema_validator(schema_dir)
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"could not load the IRA schema: {exc}")

    if validator is not None:
        failures = validate_variants(variants, validator)
        if failures:
            print(f"Schema validation failed for {len(failures)} variant(s):", file=sys.stderr)
            for variant_id, errors in failures:
                print(f"  variant_{variant_id:04d}:", file=sys.stderr)
                for error in errors:
                    print(f"    - {error}", file=sys.stderr)
            return 1
        validation_status = f"passed for {len(variants)} variant(s); {validation_status}"

    print_plan(base_path, sweep_path, output_dir, sweep, args.seed, validation_status)
    if args.dry_run:
        print("Dry-run complete. No files were written.")
        return 0

    existing = _find_managed_outputs(output_dir)
    if existing and not args.overwrite:
        parser.error(
            f"{len(existing)} generated output path(s) already exist in {output_dir}; "
            "inspect them and pass --overwrite to replace them"
        )
    if existing:
        _remove_managed_outputs(existing)

    try:
        manifest_path = write_variants(variants, output_dir)
    except (OSError, RuntimeError) as exc:
        print(f"Error: could not write variants: {exc}", file=sys.stderr)
        return 1

    print(f"Generated {len(variants)} variant(s) in {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Schema validation: {validation_status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
