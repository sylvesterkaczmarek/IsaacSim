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

"""Drive IRI (isaacsim.replicator.incident.core) incident events on a live stage.

Wraps the extension's Python API: scene tagging commands, the per-type event
managers reached through ``IncidentManager``, and ``IncidentReport`` recording.
Covers both authoring paths -- ``action=add_event`` builds one event through the
API, ``action=setup`` applies a whole config file produced by
``generate-incident-config``.

Send this file to a running Isaac Sim through the python_server socket (see the
``isaac-sim-remote`` skill). It needs a live stage and cannot run under plain
Python.

Injected args (via isaacsim_send.py --arg):
    action: str - one of preflight (default), tag, add_event, setup,
        start_record, stop_record.

    tag:          prims: str  - comma-separated prim paths to tag.
                  tag: str    - loose | flammable | leakable | spillable_area.
                  tag_type: str - vocabulary value; defaults to the first valid
                      one for the tag (loose: NavMesh, RandomDir, ClosestWaypoint;
                      flammable: Box; leakable: Item; spillable_area: Floor).

    add_event:    event: str  - topple | fire | spill.
                  name: str   - event name; also the incident report key.
                  item: str   - tagged prim path, or omit for a random tagged item.
                  time: float - seconds for a time trigger (default 3.0).
                  carb_event: str - carb event name; alternative to time.
                  physical_event: str - IRI event name to chain from.
                  radius: float - topple_nearby_radius / pyro_nearby_radius.
                  target_size: float - spill only (default 1.0).
                  leak_duration: float - spill only (default 5.0).
                  seed: int   - manager RNG seed (default 123456).

    setup:        config: str - path to an IRI config YAML.
                  bake_navmesh: bool - bake and wait before setup (default true).

    start_record: report_dir: str - directory for the report (required).
                  file_name: str  - default incidents_report.json.

Examples:
    isaacsim_send.py --file run_incident_events.py
    isaacsim_send.py --file run_incident_events.py --arg action=tag \\
        --arg prims=/World/Box01,/World/Box02 --arg tag=loose --arg tag_type=NavMesh
    isaacsim_send.py --file run_incident_events.py --arg action=add_event \\
        --arg event=topple --arg name="shelf collapse" --arg time=3
    isaacsim_send.py --file run_incident_events.py --arg action=setup \\
        --arg config=$WORKSPACE_DIR/iri_config.yaml
"""

import omni.kit.app
import omni.kit.commands
import omni.usd

EXTENSION_NAME = "isaacsim.replicator.incident.core"

# Tag vocabularies, mirrored from the extension's command modules. The first
# entry of each list is the default applied when tag_type is omitted.
TAGS = {
    "loose": ("IsaacSim_Replicator_Incident_Attr:LooseItem", ["NavMesh", "RandomDir", "ClosestWaypoint"]),
    "flammable": ("IsaacSim_Replicator_Incident_Attr:FlammableItem", ["Box"]),
    "leakable": ("IsaacSim_Replicator_Incident_Attr:LeakableItem", ["Item"]),
    "spillable_area": ("IsaacSim_Replicator_Incident_Attr:SpillableArea", ["Floor"]),
}

TAG_COMMANDS = {
    "loose": ("ApplyLooseItemTagCommand", "loose_item_type"),
    "flammable": ("ApplyFlammableItemTagCommand", "flammable_item_type"),
    "leakable": ("ApplyLeakableItemTagCommand", "leakable_item_type"),
    "spillable_area": ("ApplySpillableAreaTagCommand", "spillable_area_type"),
}

RANDOM_SENTINELS = {
    "topple": "$random_loose_item$",
    "fire": "$random_flammable_item$",
    "spill": "$random_leakable_item$",
}

if "action" not in dir():
    action = "preflight"
if "prims" not in dir():
    prims = None
if "tag" not in dir():
    tag = None
if "tag_type" not in dir():
    tag_type = None
if "event" not in dir():
    event = None
if "name" not in dir():
    name = None
if "item" not in dir():
    item = None
if "time" not in dir():
    time = None
if "carb_event" not in dir():
    carb_event = None
if "physical_event" not in dir():
    physical_event = None
if "radius" not in dir():
    radius = 0.0
if "target_size" not in dir():
    target_size = 1.0
if "leak_duration" not in dir():
    leak_duration = 5.0
if "seed" not in dir():
    seed = 123456
if "config" not in dir():
    config = None
if "bake_navmesh" not in dir():
    bake_navmesh = True
if "report_dir" not in dir():
    report_dir = None
if "file_name" not in dir():
    file_name = "incidents_report.json"


def _as_bool(value, default=False):
    """Parse an injected arg that may arrive as a real bool or as a string."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _as_list(value):
    """Parse a comma-separated string or a sequence into a list of strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _enable_extension(ext_name):
    """Enable an extension without raising. Returns (ok, detail)."""
    manager = omni.kit.app.get_app().get_extension_manager()
    if manager.is_extension_enabled(ext_name):
        return True, "already-enabled"
    try:
        manager.set_extension_enabled_immediate(ext_name, True)
        omni.kit.app.get_app().update()
    except Exception as exc:  # extension missing from this app's extscache
        return False, f"{type(exc).__name__}: {exc}"
    if manager.is_extension_enabled(ext_name):
        return True, "enabled"
    return False, "not resolvable in this app"


def _find_tagged(stage):
    """Map each incident tag to the prim paths carrying it with a valid type value."""
    tagged = {name: [] for name in TAGS}
    if stage is None:
        return tagged
    for prim in stage.Traverse():
        for tag_name, (attr_name, valid_types) in TAGS.items():
            attribute = prim.GetAttribute(attr_name)
            if attribute and attribute.Get() in valid_types:
                tagged[tag_name].append(str(prim.GetPath()))
    return tagged


def _count_tags(stage):
    """Count prims carrying each incident tag attribute, with a valid type value."""
    return {name: len(paths) for name, paths in _find_tagged(stage).items()}


def _uncovered_leakable_items(stage, leakable_paths, area_paths):
    """Find leakable items with no tagged spillable area beneath them.

    The spill demon searches for a tagged area *under the leaking item*; a floor
    split into many prims (as in the sample warehouse) means tagging one tile
    covers almost nothing. Returns the uncovered item paths, or None when the
    check cannot run.
    """
    if not leakable_paths or not area_paths:
        return None
    try:
        from pxr import Usd, UsdGeom

        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

        areas = []
        for path in area_paths:
            prim = stage.GetPrimAtPath(path)
            if prim and prim.IsValid():
                box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
                areas.append((box.GetMin(), box.GetMax()))
        if not areas:
            return None

        uncovered = []
        for path in leakable_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
            low, high = box.GetMin(), box.GetMax()
            center_x = (low[0] + high[0]) / 2.0
            center_y = (low[1] + high[1]) / 2.0
            # Area must span the item in XY and sit at or below it in Z.
            covered = any(
                a_low[0] <= center_x <= a_high[0] and a_low[1] <= center_y <= a_high[1] and a_low[2] <= high[2]
                for a_low, a_high in areas
            )
            if not covered:
                uncovered.append(path)
        return uncovered
    except Exception:
        return None


def _navmesh_interface():
    """Return the navmesh interface, or None when the extension is unavailable."""
    try:
        import omni.anim.navigation.core as nav

        return nav.acquire_interface()
    except Exception:
        return None


def _navmesh_available():
    """Report whether a baked navmesh is reachable, without baking one."""
    interface = _navmesh_interface()
    if interface is None:
        return False
    try:
        return interface.get_navmesh() is not None
    except Exception:
        return False


def _get_incident_ext():
    """Get the IRI extension singleton.

    Always go through ``get_instance()``. On some builds
    ``GlobalValues.incident_manager`` is overwritten with the ConfigFileLoader
    during startup, so reading it yields the wrong object.
    """
    from isaacsim.replicator.incident.core import get_instance

    instance = get_instance()
    if instance is None:
        raise RuntimeError(
            f"{EXTENSION_NAME} is enabled but its singleton is None; the extension "
            "did not finish on_startup. Check the Isaac Sim console."
        )
    return instance


def _build_trigger():
    """Build a trigger from the injected args. Exactly one trigger kind may be set."""
    from omni.metropolis.pipeline.triggers import TriggersManager

    # `time is not None`, not truthiness: time=0 is a valid "fire immediately" trigger, and
    # --arg injection eval()s values, so `--arg time=0` really does arrive as int 0.
    chosen = [
        k for k, v in (("time", time is not None), ("carb_event", carb_event), ("physical_event", physical_event)) if v
    ]
    if len(chosen) > 1:
        raise ValueError(f"Set at most one of time, carb_event, physical_event; got {chosen}.")
    if carb_event:
        spec = {"type": "carb_event", "event_name": str(carb_event)}
    elif physical_event:
        spec = {"type": "physical_event", "incident_name": str(physical_event)}
    else:
        spec = {"type": "time", "time": float(time) if time is not None else 3.0}
    trigger = TriggersManager.get_instance().create_trigger_by_dict({"trigger": spec})
    if trigger is None:
        raise RuntimeError(f"TriggersManager could not build a trigger from {spec}.")
    return trigger, spec


ext_ok, ext_detail = _enable_extension(EXTENSION_NAME)
stage = omni.usd.get_context().get_stage()

if action == "preflight":
    # Tags are plain USD attributes, so they are countable even when the
    # extension itself did not resolve.
    tagged = _find_tagged(stage)
    counts = {name: len(paths) for name, paths in tagged.items()}
    uncovered = _uncovered_leakable_items(stage, tagged["leakable"], tagged["spillable_area"])
    if uncovered is None:
        spill_covered = "n/a"
    else:
        spill_covered = f"{counts['leakable'] - len(uncovered)}/{counts['leakable']}"
    recording = "n/a"
    if ext_ok and stage is not None:
        try:
            recording = (
                "yes" if _get_incident_ext().get_incident_manager().get_incident_report().is_recording() else "no"
            )
        except Exception as exc:
            recording = f"error({type(exc).__name__})"

    print(
        f"IRI_PREFLIGHT: ext={'enabled' if ext_ok else 'unavailable'} "
        f"stage={'ok' if stage is not None else 'none'} "
        f"loose={counts['loose']} flammable={counts['flammable']} "
        f"leakable={counts['leakable']} spillable_area={counts['spillable_area']} "
        f"spill_covered={spill_covered} "
        f"navmesh={'ok' if _navmesh_available() else 'none'} recording={recording}"
    )
    print(f"  extension: {EXTENSION_NAME} ({ext_detail})")
    if not ext_ok:
        print("  remedy: launch an app that pins the extension, e.g.")
        print("          isaacsim.exp.action_and_event_data_generation.base.sh")
    if stage is None:
        print("  remedy: open a stage before setting up incidents")
    if ext_ok and stage is not None:
        if not any(counts.values()):
            print("  remedy: nothing is tagged; run action=tag, or open a pre-tagged stage such as")
            print("          Isaac/Samples/Replicator/Incidents/full_warehouse_with_incident_tags.usd")
        if counts["leakable"] and not counts["spillable_area"]:
            print("  note: leakable items exist but no spillable area; liquid will spawn at z=0.0")
        if uncovered:
            print(f"  note: {len(uncovered)} leakable item(s) have no tagged spillable area beneath them;")
            print("        their liquid will spawn at z=0.0 instead of on the floor. A floor split")
            print("        into many prims (as in the sample warehouse) needs every tile under a")
            print("        leaking item tagged, not just one. Tag them with:")
            for path in uncovered[:5]:
                print(f"          uncovered: {path}")
            if len(uncovered) > 5:
                print(f"          ... and {len(uncovered) - 5} more")

elif action == "tag":
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}); cannot tag prims.")
    if stage is None:
        raise RuntimeError("No USD stage is open; cannot tag prims.")
    if tag not in TAGS:
        raise ValueError(f"tag must be one of {sorted(TAGS)}; got {tag!r}.")

    prim_paths = _as_list(prims)
    if not prim_paths:
        raise ValueError("Provide prims as a comma-separated list of prim paths.")
    missing = [p for p in prim_paths if not stage.GetPrimAtPath(p).IsValid()]
    if missing:
        raise ValueError(f"These prim paths do not resolve on this stage: {missing}")

    attr_name, valid_types = TAGS[tag]
    resolved_type = str(tag_type) if tag_type else valid_types[0]
    if resolved_type not in valid_types:
        raise ValueError(f"tag_type {resolved_type!r} is not valid for {tag!r}; choose from {valid_types}.")

    command_name, type_kwarg = TAG_COMMANDS[tag]
    omni.kit.commands.execute(command_name, prims=prim_paths, **{type_kwarg: resolved_type})

    # The command may retarget a tag onto an ancestor or descendant that already
    # carries a RigidBodyAPI, so report what actually got tagged rather than the input.
    tagged = []
    for prim in stage.Traverse():
        attribute = prim.GetAttribute(attr_name)
        if attribute and attribute.Get() == resolved_type:
            tagged.append(str(prim.GetPath()))
    print(f"IRI_TAG_RESULT: tag={tag} type={resolved_type} requested={len(prim_paths)} tagged_on_stage={len(tagged)}")
    for path in tagged[:20]:
        print(f"  tagged {path}")
    if len(tagged) > 20:
        print(f"  ... and {len(tagged) - 20} more")
    if len(tagged) < len(prim_paths):
        print("  note: fewer prims carry the tag than were requested. The apply command skips a")
        print("        prim when an ancestor or descendant is already tagged, and retargets to the")
        print("        nearest prim holding a RigidBodyAPI. Save the stage to persist tags.")

elif action == "add_event":
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}); cannot add events.")
    if stage is None:
        raise RuntimeError("No USD stage is open; cannot add events.")
    if event not in RANDOM_SENTINELS:
        raise ValueError(f"event must be one of {sorted(RANDOM_SENTINELS)}; got {event!r}.")

    from isaacsim.replicator.incident.core.extension import IncidentExt

    incident_ext = _get_incident_ext()
    manager = incident_ext.get_incident_manager()
    report = manager.get_incident_report()
    event_name = str(name) if name else f"{event} event"
    selected = str(item) if item else RANDOM_SENTINELS[event]
    trigger, trigger_spec = _build_trigger()

    counts = _count_tags(stage)
    required_tag = {"topple": "loose", "fire": "flammable", "spill": "leakable"}[event]
    if counts[required_tag] == 0:
        raise RuntimeError(
            f"No prim carries the '{required_tag}' tag, so a {event} event cannot resolve an item. "
            f"Run action=tag first, or open a pre-tagged stage."
        )

    if event == "topple":
        event_manager = manager.create_topple_event_manager(seed=int(seed), report=report)
        event_manager.generate_topple_event(
            name=event_name,
            selected_loose_item=selected,
            topple_nearby_radius=float(radius),
            trigger=trigger,
        )
        registered = event_name in event_manager.topple_events
    elif event == "fire":
        event_manager = manager.create_pyro_event_manager(
            data_path=IncidentExt.data_path, seed=int(seed), report=report
        )
        event_manager.generate_pyro_event(
            name=event_name,
            selected_flammable_item_prim_path=selected,
            pyro_nearby_radius=float(radius),
            trigger=trigger,
        )
        registered = event_name in event_manager.pyro_events
    else:
        event_manager = manager.create_spill_event_manager(seed=int(seed), report=report)
        event_manager.generate_spill_event(
            name=event_name,
            selected_spillable_item=selected,
            target_size=float(target_size),
            leak_duration=float(leak_duration),
            trigger=trigger,
        )
        registered = event_name in event_manager.spill_events

    # Every generate_*_event path returns None and logs through carb instead of
    # raising, so confirm the event actually landed in the manager's registry.
    if not registered:
        raise RuntimeError(
            f"{event} event {event_name!r} was not registered. The manager logs the reason "
            f"through carb -- check the Isaac Sim console. The usual cause is that "
            f"{selected!r} is not a tagged prim, or that every tagged item of this type "
            f"was already consumed by an earlier event."
        )

    print(f"IRI_EVENT_RESULT: event={event} name={event_name!r} item={selected} trigger={trigger_spec}")
    print("  note: events fire on the timeline. Start recording, then play, to capture a report.")

elif action == "setup":
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}); cannot set up incidents.")
    if stage is None:
        raise RuntimeError("No USD stage is open; cannot set up incidents.")
    if not config:
        raise ValueError("Provide config=<path to an IRI config YAML>.")

    from isaacsim.replicator.incident.core.config_file_defines import IncidentEventSection, IncidentGlobalSection

    incident_ext = _get_incident_ext()
    loader = incident_ext.get_config_file_loader()
    if not loader.load_config_file(str(config)):
        raise RuntimeError(
            f"Failed to load {config}. The loader logs the reason through carb -- common causes "
            f"are a missing 'isaacsim.replicator.incident:' header, a missing 'version', or a "
            f"version whose MAJOR component differs from the installed extension."
        )

    config_file = loader.get_config_file()
    seed_prop = config_file.get_property(IncidentGlobalSection.name, "seed")
    event_section = config_file.get_section(IncidentEventSection.name)
    if event_section is None:
        raise RuntimeError(f"{config} has no 'event' section; there is nothing to set up.")

    if _as_bool(bake_navmesh, True):
        # Topple items tagged NavMesh resolve their direction against a baked
        # navmesh. Without this the direction lookup silently falls back.
        interface = _navmesh_interface()
        if interface is not None:
            interface.start_navmesh_baking_and_wait()
        else:
            print("  note: omni.anim.navigation.core unavailable; skipped navmesh bake")

    resolved_seed = seed_prop.get_resolved_value() if seed_prop else int(seed)
    manager = incident_ext.get_incident_manager()
    manager.setup_incidents_from_config_file(resolved_seed, event_section)

    managers = {
        "topple": manager.get_topple_event_manager(),
        "fire": manager.get_pyro_event_manager(),
        "spill": manager.get_spill_event_manager(),
    }
    registries = {
        "topple": len(getattr(managers["topple"], "topple_events", {}) or {}),
        "fire": len(getattr(managers["fire"], "pyro_events", {}) or {}),
        "spill": len(getattr(managers["spill"], "spill_events", {}) or {}),
    }
    requested = len(config_file.get_section(IncidentEventSection.name).get_property_group("event_list").data_group)
    total = sum(registries.values())

    print(
        f"IRI_SETUP_RESULT: config={config} seed={resolved_seed} requested={requested} "
        f"registered={total} topple={registries['topple']} fire={registries['fire']} "
        f"spill={registries['spill']}"
    )
    if total < requested:
        print(f"  note: {requested - total} event(s) did not register. setup_incidents_from_config_file")
        print("        logs each failure through carb rather than raising -- check the console.")
        print("        The usual cause is an event whose item is untagged or already consumed.")

elif action == "start_record":
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}); cannot record.")
    if not report_dir:
        raise ValueError("Provide report_dir=<directory for the incident report>.")

    report = _get_incident_ext().get_incident_manager().get_incident_report()
    if report.is_recording():
        print("  note: a recording was already in progress; it is discarded and restarted.")
    report.start_recording(str(report_dir), file_name=str(file_name))
    print(f"IRI_RECORD_RESULT: state=recording dir={report_dir} file={file_name}")
    print("  note: recording only accumulates while the timeline plays. Start the timeline next,")
    print("        then run action=stop_record to write the JSON.")

elif action == "stop_record":
    if not ext_ok:
        raise RuntimeError(f"{EXTENSION_NAME} is not available in this app ({ext_detail}); cannot record.")

    report = _get_incident_ext().get_incident_manager().get_incident_report()
    if not report.is_recording():
        raise RuntimeError("No recording is in progress; nothing to write. Run action=start_record first.")
    report.end_recording()
    print("IRI_RECORD_RESULT: state=stopped")
    print("  note: end_recording logs through carb on write failure. Confirm the JSON exists on disk.")

else:
    raise ValueError(
        f"Unknown action {action!r}; expected one of preflight, tag, add_event, setup, " f"start_record, stop_record."
    )
