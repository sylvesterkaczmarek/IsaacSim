# IRI Python API Reference

Signatures and constants for `isaacsim.replicator.incident.core`, read from the
extension source as resolved by the Action and Event Data Generation app.

## Entry point

```python
from isaacsim.replicator.incident.core import get_instance

instance = get_instance()          # IncidentExt | None (None until on_startup runs)
manager  = instance.get_incident_manager()      # IncidentManager
loader   = instance.get_config_file_loader()    # ConfigFileLoader
```

`IncidentExt.data_path` is a class attribute set during startup to
`<ext_path>/data`. Fire events need it.

**Do not use `GlobalValues`.** `settings.py` declares
`incident_manager`, `config_file_loader`, and `config_file_format` slots, but
`on_startup` does:

```python
GlobalValues.incident_manager = self._incident_manager
...
GlobalValues.incident_manager = self._config_file_loader   # overwrites it
```

so `GlobalValues.incident_manager` holds a `ConfigFileLoader` and
`GlobalValues.config_file_loader` stays `None`. `GlobalValues.config_file_format`
is correct and is what the extension's own tests use. `get_instance()` works on
every version — use it unconditionally.

## Scene tagging

Applied through `omni.kit.commands`. Each command accepts `prims` as a path,
a `Sdf.Path`, a `Usd.Prim`, or a list of any of those, plus an optional `layer`.

| Command | Type kwarg | Valid values |
|---|---|---|
| `ApplyLooseItemTagCommand` | `loose_item_type` | `NavMesh`, `RandomDir`, `ClosestWaypoint` |
| `ApplyFlammableItemTagCommand` | `flammable_item_type` | `Box` |
| `ApplyLeakableItemTagCommand` | `leakable_item_type` | `Item` |
| `ApplySpillableAreaTagCommand` | `spillable_area_type` | `Floor` |
| `CreateIncidentToppleDestinationCommand` | — | Creates a waypoint prim (`Sphere` or `Cube`) for `ClosestWaypoint` items |

Matching `Remove*TagCommand` variants exist for each.

Tags are plain custom USD string attributes:

| Attribute | Applied by |
|---|---|
| `IsaacSim_Replicator_Incident_Attr:LooseItem` | `ApplyLooseItemTagCommand` |
| `IsaacSim_Replicator_Incident_Attr:FlammableItem` | `ApplyFlammableItemTagCommand` |
| `IsaacSim_Replicator_Incident_Attr:LeakableItem` | `ApplyLeakableItemTagCommand` |
| `IsaacSim_Replicator_Incident_Attr:SpillableArea` | `ApplySpillableAreaTagCommand` |
| `IsaacSim_Replicator_Incident_Attr:ToppleDestination` | waypoint prims |

A tag counts only when its value is one of the vocabulary strings above, which is
how you enumerate tagged prims yourself:

```python
for prim in stage.Traverse():
    attr = prim.GetAttribute("IsaacSim_Replicator_Incident_Attr:LooseItem")
    if attr and attr.Get() in ("NavMesh", "RandomDir", "ClosestWaypoint"):
        ...
```

Because they are USD attributes, tags live in the current edit target and persist
only when the stage is saved. Untagging by hand means deleting the raw USD
property.

### Retargeting

`ApplyLooseItemTagCommand` does not necessarily tag the prim you named. It walks
ancestors then descendants looking for an existing tag (in which case it skips
the prim entirely) or a `UsdPhysics.RigidBodyAPI` (in which case it tags *that*
prim instead). So the set of tagged prims after the call can be smaller than, or
different from, the input list. Enumerate the stage afterwards rather than
assuming.

## Loose item direction types

| Type | Force direction | Needs |
|---|---|---|
| `NavMesh` | Toward the nearest navmesh edge | A baked navmesh; suits items on shelves or tables |
| `RandomDir` | Random | Nothing |
| `ClosestWaypoint` | Toward the nearest point on the nearest waypoint prim | At least one topple-destination prim |

## IncidentManager

```text
manager.create_topple_event_manager(seed: int, report: IncidentReport) -> ToppleEventManager
manager.create_pyro_event_manager(data_path: str, seed: int, report: IncidentReport) -> PyroEventManager
manager.create_spill_event_manager(seed: int, report: IncidentReport) -> SpillEventManager

manager.get_topple_event_manager()   # -> manager | None (no creation)
manager.get_pyro_event_manager()
manager.get_spill_event_manager()

manager.get_incident_report()        # -> the single shared IncidentReport
manager.get_dynamic_obstacle_prim_paths()  # -> list[str] of toppled items, for navigation
manager.setup_incidents_from_config_file(random_seed: int, incident_event_section: IncidentEventSection)
manager.reset_incident_managers()    # destroys all three; the next create_* rebuilds
```

The `create_*` methods are get-or-create: they return the existing manager and
**ignore the seed and report arguments** if one already exists. To change the
seed you must `reset_incident_managers()` first.

`setup_incidents_from_config_file` calls `reset_incident_managers()` itself, so
applying a config discards anything built earlier — including events added
through `generate_*_event`.

## Event generation

```text
ToppleEventManager.generate_topple_event(
    name: str,
    selected_loose_item: str,          # prim path or IncidentSettings.RANDOM_LOOSE_ITEM
    topple_nearby_radius: float = 0.0,
    trigger: TriggerBase = None,
)

PyroEventManager.generate_pyro_event(
    name: str,
    selected_flammable_item_prim_path: str,   # or RANDOM_FLAMMABLE_ITEM
    pyro_nearby_radius: float = 0.0,
    trigger: TriggerBase = None,
)

SpillEventManager.generate_spill_event(
    name: str,
    selected_spillable_item: str,      # or RANDOM_LEAKABLE_ITEM
    target_size: float = 1.0,
    leak_duration: float = 1.0,
    trigger: TriggerBase = None,
)
```

All three return `None` — on success *and* on failure. Confirm registration
through the manager's dict:

| Manager | Registry attribute |
|---|---|
| `ToppleEventManager` | `topple_events` |
| `PyroEventManager` | `pyro_events` |
| `SpillEventManager` | `spill_events` |

Each is keyed by event name, so reusing a name overwrites the earlier event.

`pyro_nearby_radius` works through this API but is unreachable from a config
file — `generate_pyro_event_from_property` hardcodes `1.0`.

The `*_from_property` variants (`generate_topple_event_from_property` and
siblings) take a config `Property` object and are what
`setup_incidents_from_config_file` calls internally.

## Random item sentinels

```python
from isaacsim.replicator.incident.core.settings import IncidentSettings

IncidentSettings.RANDOM_LOOSE_ITEM      # "$random_loose_item$"
IncidentSettings.RANDOM_FLAMMABLE_ITEM  # "$random_flammable_item$"
IncidentSettings.RANDOM_LEAKABLE_ITEM   # "$random_leakable_item$"
```

Each manager draws from a set of tagged items it has not consumed
(`untoppled_items`, `unflammed_items`, `unspilled_items`) using its seeded RNG,
and removes what it uses. An exhausted pool logs a carb error and skips the event.

Exception: on builds where the pyro manager's removal is a no-op,
`unflammed_items` never shrinks and two random fire events can land on the same
prim. See `runtime-notes.md`.

## Triggers

```python
from omni.metropolis.pipeline.triggers import TriggersManager

trigger = TriggersManager.get_instance().create_trigger_by_dict(
    {"trigger": {"type": "time", "time": 3.0}}
)
trigger.add_callback(lambda t: ...)     # optional; the event manager adds its own
```

Note the dict is **wrapped in a `"trigger"` key**. Types: `time` (`time`, seconds),
`carb_event` (`event_name`), `physical_event` (`incident_name`). The extension
also registers its own `IncidentTrigger` type at startup.

## IncidentReport

```text
report = manager.get_incident_report()

report.start_recording(dir_path: str, file_name: str = "incidents_report.json")
report.end_recording()                  # writes the JSON
report.is_recording() -> bool
report.clear()

report.add_event_data(incident_name: str, data: dict)
report.add_simulation_data(incident_name: str, data: dict)
report.add_trigger_data(incident_name: str, data: dict)
```

`add_*` are no-ops while not recording, so events that fire before
`start_recording()` leave no trace. Calling `start_recording()` twice discards
the first recording with a warning.

## Semantic labels

Applied to prims as the simulation runs, for Replicator writers to pick up:

| Label | On |
|---|---|
| `incident_toppled_item` | Items knocked over by a topple event |
| `incident_flaming_item` | Items on fire |
| `incident_leaking_item` | Items leaking liquid |
| `incident_liquid_spill` | The liquid surface itself |
| `FireBoundingBox` | Semantic class of the fire volume's bounding box |

The fire volume is skipped by `semantic_segmentation` and
`bounding_box_2d_tight` unless the carb setting
`/syntheticdata/sensors/captureVolumes` is `true`.
