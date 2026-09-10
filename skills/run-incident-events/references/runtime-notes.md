# IRI Runtime Notes

Behavior at run time that the API signatures do not reveal. All read from the
`isaacsim.replicator.incident.core` source as resolved by the Action and Event
Data Generation app.

## Nothing raises

Every content-level failure in IRI logs through `carb` and returns. There is no
exception path for "your event did not happen":

| Situation | Observable result |
|---|---|
| Item is not tagged | `carb.log_error("'<path>' is not a tagged prim...")`, event skipped |
| Random pool exhausted | `carb.log_error("No available topple items...")`, event skipped |
| Config property in error | `carb.log_error("Value has error...")`, event skipped |
| Unrecognized event type in a config | `carb.log_warn`, event skipped |
| Report write fails | `carb.log_error("Generate report file fails.")`, no file |

A run in which two of three events silently vanished looks exactly like a clean
one from the caller's side. Two defences:

1. Count registrations after generating (`topple_events`, `pyro_events`,
   `spill_events` — see `api-reference.md`).
2. Watch the Isaac Sim console. Headless runs should capture stdout.

`run_incident_events.py` does the first for both `add_event` and `setup`, which
is why `IRI_SETUP_RESULT` prints `requested` alongside `registered`.

## Ordering constraints

**Tag before you generate.** Each event manager snapshots the tagged items in the
stage when it is constructed. Tagging a prim after the manager exists does not
add it to the pool; you need `reset_incident_managers()` (or a fresh `setup`).

**Bake the navmesh before setup.** Topple items tagged `NavMesh` resolve their
direction against the baked navmesh. The Event Config File panel calls
`start_navmesh_baking_and_wait()` immediately before
`setup_incidents_from_config_file`, and scripted callers must do the same:

```python
import omni.anim.navigation.core as nav
nav.acquire_interface().start_navmesh_baking_and_wait()
```

Skipping it does not error — the items just topple in an unhelpful direction.

**Generate before playing.** Triggers are evaluated against the timeline. An
event created after its trigger time has already passed never fires.

**Start recording before playing.** `IncidentReport.add_*` are no-ops while not
recording.

## Manager lifecycle gotchas

- `create_*_event_manager` is get-or-create and **silently ignores `seed` and
  `report`** when a manager already exists. A second call with a different seed
  changes nothing.
- `setup_incidents_from_config_file` calls `reset_incident_managers()` first, so
  applying a config **destroys events added earlier** through `generate_*_event`.
  Mixing the two paths in one session usually means the API events disappear.
- `reset_incident_managers()` destroys the managers but not the `IncidentReport`,
  which lives on `IncidentManager` and outlives them.
- Event registries are keyed by name. Two events with the same name means the
  second overwrites the first, and the report has one entry for both.

## Item pool consumption

Each manager removes items from its pool as events claim them:

- Topple: `untoppled_items.difference_update(topple_event.selected_loose_items)`.
  Because `topple_nearby_radius` sweeps in neighbours, **one event can consume
  several items** — a generous radius early in the list can starve later events.
- Spill: `unspilled_items.remove(selected_spillable_item)` — one item per event.
- Fire: **may not consume at all.** Where the manager calls
  `unflammed_items.difference_update(pyro_event.selected_flammable_item_prim_path)`
  on a single prim-path *string*, `difference_update` iterates its argument, so
  Python walks the path character by character and removes nothing — the pool
  never shrinks. Builds that use `discard` instead consume correctly. Check the
  installed source if two fire events collide.

Plan tagging so there is at least one item per event that uses a random sentinel,
with headroom for topple.

The fire defect cuts the other way from exhaustion: two fire events using
`$random_flammable_item$` can select the *same* prim, stacking two
`FlowEmitterBox` emitters on one object. The `not in self.unflammed_items` guard
does not catch it, because the item was never removed. On an affected build,
pin `item:` to distinct prim paths rather than relying on random selection when
a scene has more than one fire event.

## Incident report JSON

Written by `IncidentReport._generate_report_file()` to
`<report_dir>/<file_name>`, default `incidents_report.json`.

The file is a JSON object whose **top-level keys are event names**. Each entry
may hold up to three sections:

| Section | Contents |
|---|---|
| `event_data` | How the event was set up — from the config or the generate call |
| `trigger_data` | A nested `trigger` object with `type`, `priority`, and the type's field. Time triggers report `time` in **seconds** |
| `simulation_data` | Timeline metadata in **simulation frame indices**, not seconds |

Per event type:

| Type | `simulation_data` keys |
|---|---|
| Topple | `start_time`, `end_time` (end is when the observer considers the topple settled, e.g. items sleeping — not the trigger instant) |
| Spill | `start_time`, `end_time` (spans the trigger frame through `trigger_time + leak_duration`) |
| Fire | `start_time` and `fire_prim` (the `FlowEmitterBox` emitter path). **No `end_time`** |

Two parsing rules:

1. **Never equate `simulation_data` numbers with `trigger_data` times.** One is
   frames, the other seconds. Convert with the timeline FPS.
2. **Treat every section and key as optional.** Fire has no `end_time`; older
   code paths may omit `trigger_data` entirely. Fire's `event_data` may also
   carry a `flame_emitter` field pointing at the *flammable item* prim, which is
   a different prim from `simulation_data.fire_prim`.

The report is JSON. The event *config* is YAML. They are unrelated files with
different lifecycles — do not conflate them.

## Spill placement

Liquid instantiates on the prim tagged `spillable_area` that sits **beneath the
leaking item**. The search is positional, not global: a tagged area somewhere
else in the scene does not help.

When no area is found under the item, the demon logs

```
[SpillFloorSearch] No floor found for spill event, assuming the floor is at height 0
```

and spawns the liquid at `z = 0.0`. That fallback is invisible in a scene whose
floor already sits at zero, and plainly wrong in a multi-level, mezzanine, or
raised-platform scene, where the puddle detaches from the item and lands at world
zero.

### Split floors are the common trap

A warehouse floor is usually **many prims, not one**. In
`Isaac/Environments/Simple_Warehouse/full_warehouse.usd` it is ~80 separate
`SM_floor*` tiles, each about 6 m x 6 m. Tagging one tile covers only the items
above that tile — every other leaking item still takes the `z = 0.0` fallback.

Verified on that stage: tagging `/Root/SM_floor80` while the leaking item sat
above `/Root/SM_floor59` produced the `SpillFloorSearch` error; tagging
`SM_floor59` cleared it.

So tag by position, not by name:

1. Find the leaking item's world-XY centre.
2. Tag whichever floor prim spans that XY and sits below it.
3. Or tag every floor tile, if the leak location is random (`$random_leakable_item$`
   makes this the safer default — you cannot know in advance which item is chosen).

`action=preflight` does this check for you and reports
`spill_covered=<covered>/<leakable>`, listing any item with no area beneath it.
`spill_covered=n/a` means the check could not run (no leakable items, no tagged
areas, or bbox computation failed).

## Fire and volume annotators

The fire effect is a volume prim (`FlowEmitterBox`). Rasterization-based
annotators (`semantic_segmentation`, `bounding_box_2d_tight`) omit volumes by
default, so a fire that is plainly visible in RGB can be absent from
segmentation. Set `/syntheticdata/sensors/captureVolumes` to `true` to include
it. The fire's bounding box carries the semantic class `FireBoundingBox`,
separate from the `incident_flaming_item` label on the burning object.

## Toppled items as navigation obstacles

`IncidentManager.get_dynamic_obstacle_prim_paths()` returns the prim paths of
items a topple event has moved. Feed these to navigation so robots or characters
react to the new obstacle — see `isaac-sim-robot-navigation` and
`navigation-primitives`. It returns `[]` when no topple manager exists.
