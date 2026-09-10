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

"""Generate or validate an IRI (isaacsim.replicator.incident) event config file, offline.

Emits the YAML that the Event Config File panel loads, and that
``IncidentManager.setup_incidents_from_config_file()`` consumes. Does NOT launch
Isaac Sim, so it runs anywhere.

Two modes:

  generate (default)  build a config from --event specs and print it
  --validate FILE     structurally check an existing config and report problems

Validation matters because the loader is lenient in a way that hides mistakes:
``EventPropertyBase.set_value()`` copies only the keys it already knows about, so a
misspelled or stale key inside an item group is dropped with no error and the default
value is used instead. This script reports those keys as warnings.

Validation runs in one of two modes, reported as ``mode=`` in the result line:

  ``mode=schema``   validated against ``data/config_schema/iri_config.schema.json``
                    when the build ships it. Authoritative -- it is the same
                    document the extension itself validates with. Found
                    automatically under ``--from-ext``; override with ``--schema``.
  ``mode=builtin``  the checks written into this script, used when the build ships no
                    schema or the ``jsonschema`` package is unavailable.

The fallback is automatic and silent-by-default, so this script works against every
build. Pass ``--require-schema`` to fail instead of falling back.

Event spec grammar (repeatable ``--event``)::

    <type>[:key=value,key=value,...]

    types: topple, fire, spill
    common keys: name, item, time, carb_event, physical_event
    topple keys: radius            -> topple_item.topple_nearby_radius
    spill keys:  target_size, leak_duration

Usage::

    python3 starter_iri_config.py --from-ext $ISAAC_SIM_DIR \\
        --event "topple:time=3,radius=1.5" \\
        --event "fire:time=6,item=/World/Pallet" \\
        --event "spill:time=9,target_size=1.5,leak_duration=5"

    python3 starter_iri_config.py --validate my_iri_config.yaml --from-ext $ISAAC_SIM_DIR

Exit 0 on success, 1 when validation finds errors or an event spec is rejected,
2 on argparse-level bad arguments.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

INCIDENT_EXT = "isaacsim.replicator.incident.core"
# The config header is the extension family name, NOT the .core module name.
CONFIG_HEADER = "isaacsim.replicator.incident"

# Shipped by newer extension builds, relative to the extension root.
# When present it is authoritative; builds without it fall back to the checks in this file.
SCHEMA_REL_PATH = Path("data") / "config_schema" / "iri_config.schema.json"

RANDOM_LOOSE_ITEM = "$random_loose_item$"
RANDOM_FLAMMABLE_ITEM = "$random_flammable_item$"
RANDOM_LEAKABLE_ITEM = "$random_leakable_item$"

TRIGGER_KEYS = {"time": "time", "carb_event": "event_name", "physical_event": "incident_name"}

# Event type -> (yaml event key, item group name, {item key: default}, default event name).
# The item-group keys mirror config_file_defines.py exactly; anything else is dropped
# silently by the loader.
EVENT_TYPES = {
    "topple": (
        "ToppleEvent",
        "topple_item",
        {"item": RANDOM_LOOSE_ITEM, "topple_nearby_radius": 1.5},
        "topple event",
    ),
    "fire": (
        "FireEvent",
        "flammable_item",
        {"item": RANDOM_FLAMMABLE_ITEM},
        "fire event",
    ),
    "spill": (
        "SpillEvent",
        "leakable_item",
        {"item": RANDOM_LEAKABLE_ITEM, "target_size": 1.0, "leak_duration": 5.0},
        "spill event",
    ),
}

# Friendly spec key -> real item-group key, per event type.
SPEC_ALIASES = {
    "topple": {"radius": "topple_nearby_radius"},
    "fire": {},
    "spill": {},
}

EVENT_KEY_TO_TYPE = {yaml_key: name for name, (yaml_key, _, _, _) in EVENT_TYPES.items()}


def find_ext_tomls(path: str | Path, ext_name: str) -> list[Path]:
    """Locate extension.toml files for ext_name under an extension.toml, ext dir, or build root.

    Offline (filesystem lookup only) -- no Isaac Sim launch.
    """
    p = Path(path).expanduser()
    if p.is_file():
        return [p]
    if p.is_dir():
        direct = p / "config" / "extension.toml"
        if direct.is_file():
            return [direct]
        # Search a build/extscache root: match ONLY the exact extension dir or an
        # extscache "<ext_name>-<version>" dir, depth 1 first then recursively.
        pats = [f"{ext_name}/config/extension.toml", f"{ext_name}-*/config/extension.toml"]
        tomls = [t for pat in pats for t in sorted(p.glob(pat))]
        if not tomls:
            tomls = [t for pat in pats for t in sorted(p.glob("**/" + pat))]
        return tomls
    raise SystemExit(f"--from-ext path not found: {path}")


def find_schema(path: str | Path, ext_name: str = INCIDENT_EXT) -> Path | None:
    """Find the extension's shipped JSON Schema, if this build carries one.

    Newer isaacsim.replicator.incident.core builds ship it. Returns None when the
    build does not, so callers can fall back to the built-in checks.
    """
    for toml in find_ext_tomls(path, ext_name):
        candidate = toml.parent.parent / SCHEMA_REL_PATH
        if candidate.is_file():
            return candidate
    return None


def read_ext_version(path: str | Path, ext_name: str) -> str:
    """Read `version = "X"` from an extension.toml, an extension dir, or a build/extscache root.

    Offline (just reads a TOML file) -- no Isaac Sim launch. Raises SystemExit if not found.
    """
    tomls = find_ext_tomls(path, ext_name)
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


def _coerce(value: str):
    """Turn a spec value into int/float where it clearly is one, else leave it a string."""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_event_spec(spec: str, index: int) -> dict:
    """Parse "topple:time=3,radius=1.5" into the nested YAML event dict."""
    type_name, _, rest = spec.partition(":")
    type_name = type_name.strip().lower()
    if type_name not in EVENT_TYPES:
        raise SystemExit(f"unknown event type {type_name!r}; choose from {sorted(EVENT_TYPES)}")

    yaml_key, group_name, group_defaults, default_name = EVENT_TYPES[type_name]
    aliases = SPEC_ALIASES[type_name]

    pairs = {}
    for chunk in rest.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        key, sep, value = chunk.partition("=")
        if not sep:
            raise SystemExit(f"event spec fragment {chunk!r} must be key=value")
        pairs[key.strip()] = value.strip()

    name = pairs.pop("name", f"{default_name} {index}")

    trigger = None
    for trigger_type, field in TRIGGER_KEYS.items():
        if trigger_type in pairs:
            if trigger is not None:
                raise SystemExit(f"event {name!r} declares more than one trigger; pick exactly one")
            trigger = {"type": trigger_type, field: _coerce(pairs.pop(trigger_type))}
    if trigger is None:
        # A time trigger is the only self-contained default; staggering keeps
        # generated multi-event configs from firing everything at once.
        trigger = {"type": "time", "time": float(index * 3)}

    group = dict(group_defaults)
    for key, value in pairs.items():
        real_key = aliases.get(key, key)
        if real_key not in group_defaults:
            raise SystemExit(
                f"event {name!r} ({type_name}): key {key!r} is not part of {group_name}; "
                f"valid keys are {sorted(set(group_defaults) | set(aliases))}. "
                "The loader would silently drop it."
            )
        group[real_key] = _coerce(value)

    return {yaml_key: {"name": name, group_name: group, "trigger": trigger}}


def build_config(version: str, seed: int, report_dir: str, events: list[dict]) -> dict:
    return {
        CONFIG_HEADER: {
            "version": version,
            "global": {"report_dir": report_dir, "seed": seed},
            "event": {"event_list": events},
        }
    }


def _validate_trigger(trigger, label: str, errors: list[str]) -> None:
    if not isinstance(trigger, dict):
        errors.append(f"{label}: 'trigger' must be a mapping")
        return
    trigger_type = trigger.get("type")
    if trigger_type not in TRIGGER_KEYS:
        errors.append(f"{label}: trigger type {trigger_type!r} is not one of {sorted(TRIGGER_KEYS)}")
        return
    required = TRIGGER_KEYS[trigger_type]
    if required not in trigger:
        errors.append(f"{label}: trigger type '{trigger_type}' requires a '{required}' field")


def _schema_error_message(error) -> str:
    """Render a jsonschema error as a field-level message."""
    location = "/".join(str(part) for part in error.path) or "<root>"
    validator = getattr(error, "validator", None)

    if validator == "additionalProperties":
        schema_node = getattr(error, "schema", {})
        instance = getattr(error, "instance", {})
        allowed = set(schema_node.get("properties", {})) if isinstance(schema_node, dict) else set()
        unexpected = [k for k in instance if k not in allowed] if isinstance(instance, dict) else []
        if unexpected:
            return (
                f"{location}: unexpected key(s) {', '.join(repr(k) for k in unexpected)}; "
                f"valid keys are {sorted(allowed)}. Unknown keys are dropped when the config "
                f"is applied, so the default value would be used instead."
            )
    if validator == "required":
        required = getattr(error, "validator_value", []) or []
        instance = getattr(error, "instance", {})
        missing = [k for k in required if isinstance(instance, dict) and k not in instance]
        if missing:
            return f"{location}: missing required field(s) {', '.join(repr(k) for k in missing)}"
    if validator in ("minProperties", "maxProperties"):
        return (
            f"{location}: each event_list entry must be a single-key mapping keyed by the "
            f"event type (ToppleEvent, FireEvent, or SpillEvent)"
        )
    return f"{location}: {error.message}"


def validate_with_schema(
    data: dict, schema_path: Path, expected_major: str | None, strict: bool
) -> tuple[list[str], list[str]]:
    """Validate against the extension's shipped JSON Schema. Returns (errors, warnings).

    Unknown-key violations are warnings unless strict, mirroring the extension's own
    /exts/isaacsim.replicator.incident/strict_config_validation setting.
    """
    from jsonschema import Draft202012Validator

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    for error in sorted(Draft202012Validator(schema).iter_errors(data), key=lambda e: list(e.path)):
        message = _schema_error_message(error)
        if getattr(error, "validator", None) == "additionalProperties" and not strict:
            warnings.append(message)
        else:
            errors.append(message)

    # The schema deliberately does not encode the version-major rule, because the
    # comparison target is whatever extension is installed rather than a constant.
    if expected_major is not None and isinstance(data, dict):
        body = data.get(CONFIG_HEADER)
        if isinstance(body, dict) and "version" in body:
            actual_major = str(body["version"]).split(".")[0]
            if actual_major != expected_major:
                errors.append(
                    f"<root>: version major {actual_major!r} does not match the installed "
                    f"extension major {expected_major!r}; only the major component is compared"
                )
    return errors, warnings


def validate_config(data: dict, expected_major: str | None) -> tuple[list[str], list[str]]:
    """Structurally validate a parsed IRI config. Returns (errors, warnings)."""
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(data, dict) or CONFIG_HEADER not in data:
        errors.append(f"missing top-level header '{CONFIG_HEADER}:'")
        return errors, warnings
    for key in data:
        if key != CONFIG_HEADER:
            warnings.append(f"top-level key {key!r} sits outside the header and is ignored")

    body = data[CONFIG_HEADER]
    if not isinstance(body, dict):
        errors.append(f"'{CONFIG_HEADER}' must be a mapping")
        return errors, warnings

    version = body.get("version")
    if version is None:
        errors.append("'version' is missing; the loader rejects the file")
    elif expected_major is not None and str(version).split(".")[0] != expected_major:
        errors.append(
            f"version major {str(version).split('.')[0]!r} does not match the installed "
            f"extension major {expected_major!r}; only the major component is compared"
        )

    for key in body:
        if key not in ("version", "global", "event"):
            warnings.append(f"section {key!r} is not registered by the extension and is ignored")

    global_section = body.get("global")
    if global_section is not None:
        if not isinstance(global_section, dict):
            errors.append("'global' must be a mapping")
        else:
            for key in global_section:
                if key not in ("seed", "report_dir"):
                    warnings.append(f"global key {key!r} is unknown and is ignored")
            seed = global_section.get("seed")
            if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or seed < 0):
                errors.append(f"global.seed must be a non-negative integer; got {seed!r}")

    event_section = body.get("event")
    if not isinstance(event_section, dict) or "event_list" not in event_section:
        errors.append("'event.event_list' is missing; no incidents would be set up")
        return errors, warnings

    event_list = event_section["event_list"]
    if not isinstance(event_list, list):
        errors.append("'event.event_list' must be a list")
        return errors, warnings
    if not event_list:
        warnings.append("'event.event_list' is empty; setup would create no incidents")

    for position, entry in enumerate(event_list):
        label = f"event_list[{position}]"
        if not isinstance(entry, dict) or len(entry) != 1:
            errors.append(f"{label}: each entry must be a single-key mapping keyed by event type")
            continue
        yaml_key, event_body = next(iter(entry.items()))
        if yaml_key not in EVENT_KEY_TO_TYPE:
            errors.append(f"{label}: event type {yaml_key!r} is not one of {sorted(EVENT_KEY_TO_TYPE)}")
            continue
        if not isinstance(event_body, dict):
            errors.append(f"{label} ({yaml_key}): body must be a mapping")
            continue

        type_name = EVENT_KEY_TO_TYPE[yaml_key]
        _, group_name, group_defaults, _ = EVENT_TYPES[type_name]
        label = f"{label} ({event_body.get('name', yaml_key)})"

        if "name" not in event_body:
            warnings.append(f"{label}: no 'name'; the report keys off the event name, so a default is used")

        for key in event_body:
            if key not in ("name", group_name, "trigger"):
                warnings.append(f"{label}: key {key!r} is dropped silently by the loader")

        group = event_body.get(group_name)
        if group is None:
            warnings.append(f"{label}: no '{group_name}' block; every value falls back to the default")
        elif not isinstance(group, dict):
            errors.append(f"{label}: '{group_name}' must be a mapping")
        else:
            for key in group:
                if key not in group_defaults:
                    warnings.append(
                        f"{label}: '{group_name}.{key}' is not a recognized key and is dropped "
                        f"silently; valid keys are {sorted(group_defaults)}"
                    )
            item = group.get("item")
            if (
                isinstance(item, str)
                and item.startswith("$")
                and item
                not in (
                    RANDOM_LOOSE_ITEM,
                    RANDOM_FLAMMABLE_ITEM,
                    RANDOM_LEAKABLE_ITEM,
                )
            ):
                errors.append(f"{label}: {item!r} is not a known random-item sentinel")

        if "trigger" not in event_body:
            errors.append(f"{label}: no 'trigger'; the event is created but never fires")
        else:
            _validate_trigger(event_body["trigger"], label, errors)

    return errors, warnings


def resolve_schema(args) -> tuple[Path | None, str, str | None]:
    """Decide whether to validate against the extension's schema or the built-in checks.

    Returns (schema_path_or_None, mode_label, note). Falls back silently to the
    built-in checks whenever the schema or jsonschema is unavailable, unless
    --require-schema was given.
    """
    explicit = bool(args.schema)
    if args.schema:
        schema_path = Path(args.schema).expanduser()
        if not schema_path.is_file():
            raise SystemExit(f"--schema path not found: {args.schema}")
    elif args.from_ext:
        schema_path = find_schema(args.from_ext)
    else:
        schema_path = None

    if schema_path is None:
        if args.require_schema:
            raise SystemExit(
                "--require-schema was given but no schema was found. It ships with "
                f"newer {INCIDENT_EXT} builds at {SCHEMA_REL_PATH}; pass --from-ext pointing at "
                "such a build, or --schema with an explicit path."
            )
        note = None
        if args.from_ext:
            note = f"no {SCHEMA_REL_PATH.name} in this build; using this script's built-in checks"
        return None, "builtin", note

    try:
        # Probe the exact symbol, not the package: jsonschema < 4.0 imports fine but
        # has no Draft 2020-12 validator, and 3.2.0 is a common system-python version.
        from jsonschema import Draft202012Validator  # noqa: F401
    except ImportError:
        detail = "jsonschema with Draft 2020-12 support (>=4.0) is unavailable"
        if args.require_schema:
            raise SystemExit(f"--require-schema was given but {detail}.")
        return None, "builtin", f"{detail}; using this script's built-in checks"

    note = None if explicit else f"validating against the extension schema at {schema_path}"
    return schema_path, "schema", note


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--validate", metavar="FILE", help="Validate an existing config instead of generating one.")
    vg = p.add_mutually_exclusive_group()
    vg.add_argument(
        "--from-ext",
        help=f"Derive the config version from the installed {INCIDENT_EXT}: "
        "an extension.toml, an extension dir, or a build/extscache root to search.",
    )
    vg.add_argument("--version", help="Explicit config version (only the major component is compared).")
    p.add_argument(
        "--event",
        action="append",
        default=[],
        metavar="SPEC",
        help="Repeatable event spec, e.g. 'topple:time=3,radius=1.5'. Defaults to one of each type.",
    )
    p.add_argument("--seed", type=int, default=123456, help="global.seed; must be non-negative (default 123456).")
    p.add_argument(
        "--report-dir",
        default="$WORKSPACE_DIR/EventsResult",
        help="global.report_dir, where incidents_report.json is written.",
    )
    p.add_argument("--output", help="Write YAML to this path instead of stdout.")
    p.add_argument(
        "--schema",
        help="Explicit path to iri_config.schema.json. Normally found automatically under --from-ext.",
    )
    p.add_argument(
        "--require-schema",
        action="store_true",
        help="Fail instead of falling back to the built-in checks when the extension schema is unavailable.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat unknown keys as errors rather than warnings, matching the extension's "
        "strict_config_validation default.",
    )
    args = p.parse_args(argv)

    try:
        import yaml  # lazy import so the module compiles without PyYAML installed
    except ImportError:
        raise SystemExit("PyYAML is required (pip install pyyaml)")

    if args.validate:
        path = Path(args.validate).expanduser()
        if not path.is_file():
            raise SystemExit(f"--validate path not found: {args.validate}")
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:
            print(f"IRI_CONFIG_INVALID: {path} is not parseable YAML\n  {exc}")
            return 1
        expected_major = None
        if args.from_ext:
            expected_major = read_ext_version(args.from_ext, INCIDENT_EXT).split(".")[0]
        elif args.version:
            expected_major = str(args.version).split(".")[0]

        schema_path, mode, note = resolve_schema(args)
        if schema_path is not None:
            errors, warnings = validate_with_schema(data, schema_path, expected_major, args.strict)
        else:
            errors, warnings = validate_config(data, expected_major)
            if args.strict:
                # Keep --strict meaningful in either mode.
                errors, warnings = errors + warnings, []

        if note:
            print(f"  note: {note}")
        for warning in warnings:
            print(f"  warning: {warning}")
        for error in errors:
            print(f"  error: {error}")
        status = "INVALID" if errors else "OK"
        print(f"IRI_CONFIG_{status}: {path} mode={mode} errors={len(errors)} warnings={len(warnings)}")
        return 1 if errors else 0

    if args.seed < 0:
        p.error("--seed must be non-negative")
    if not (args.from_ext or args.version):
        p.error("generating a config requires --from-ext or --version")

    version = args.version or read_ext_version(args.from_ext, INCIDENT_EXT)
    specs = args.event or ["topple", "fire", "spill"]
    events = [parse_event_spec(spec, index + 1) for index, spec in enumerate(specs)]
    config = build_config(version, args.seed, args.report_dir, events)

    text = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
    if args.output:
        Path(args.output).expanduser().write_text(text)
        print(f"IRI_CONFIG_WRITTEN: {args.output} events={len(events)} version={version}")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
