# IRI Config Schema Reference

Exact key names, types, and defaults for the `isaacsim.replicator.incident`
config YAML, taken from `config_file_defines.py` and the generic loader in
`omni.metropolis.utils.config_file`.

Newer `isaacsim.replicator.incident.core` builds also ship a
machine-readable version of these rules at
`data/config_schema/iri_config.schema.json`. Where the two disagree, that file
wins — it is what the extension validates against. This document stays useful for
the defaults and the behavioral notes, which a JSON Schema cannot express.

## Document shape

```yaml
isaacsim.replicator.incident:   # required header
  version: <str>                # required
  global: {...}                 # registered section
  event: {...}                  # registered section
```

The header is registered in `extension.py` as
`ConfigFileFormat(required_header="isaacsim.replicator.incident", ...)`. It is
the extension *family* name — using the module name `isaacsim.replicator.incident.core`
makes the load fail.

Only `global` and `event` are registered sections. Any other key under the header
is dropped without comment.

## `version`

Compared by `ConfigFileUtil.check_version`:

```python
major = str(required_version).split(".")[0]
yaml_major = str(yaml_data["version"]).split(".")[0]
if major != yaml_major: ...
```

`required_version` is `ExtInfos.ext_ver`, which `on_startup` derives as
`str(ext_id).split("-")[-1]` — the extension version with its Kit build suffix.

**Only the major component is compared.** Against any `0.x` build, `0.1.0`,
`0.42.7`, and `0.999` all pass; `1.0.0` fails. A missing `version` is a hard
error. This is deliberately looser than the exact-match rule enforced by
`isaacsim.replicator.caption.core` (IRC) — do not carry that assumption over.

## `global` section

`IncidentGlobalSection`, `is_required = True`.

| Key | Type | Default | Verified by |
|---|---|---|---|
| `seed` | int | `123456` | `verify_int_non_negative` — negative values log a carb error |
| `report_dir` | str | `~/EventsResult` | none — any string is accepted, including a path that cannot be written |

`seed` drives random item selection in each event manager (`self.random.choice`
over the untouched tagged items). Reusing a seed with an unchanged stage
reproduces the same picks.

`report_dir` is where `IncidentReport` writes `incidents_report.json`. Because it
is unverified, a bad path only surfaces at `end_recording()` time.

## `event` section

`IncidentEventSection`, `is_required = True`. Holds one `ListPropertyGroup` named
`event_list` whose reference group is `[ToppleEventProperty, PyroEventProperty,
SpillEventProperty]`.

Each list entry is a **single-key mapping** whose key is the event type name:

```yaml
event:
  event_list:
  - ToppleEvent:
      name: ...
      topple_item: {...}
      trigger: {...}
```

### Common event keys

| Key | Type | Notes |
|---|---|---|
| `name` | str | Becomes the top-level key of this event's entry in the incident report. Defaults per type (`default topple event`, `default fire event`, `default spill event`). |
| `trigger` | mapping | See below. An event without one is created but never fires. |

### `ToppleEvent` → `topple_item`

| Key | Type | Default |
|---|---|---|
| `item` | str | `$random_loose_item$` |
| `topple_nearby_radius` | float | `1.5` |

Other loose items within `topple_nearby_radius` of the selected item topple with
it. Toppled prims receive the semantic label `incident_toppled_item`.

### `FireEvent` → `flammable_item`

| Key | Type | Default |
|---|---|---|
| `item` | str | `$random_flammable_item$` |

`PyroEventProperty.default_items` defines **only** `item`. There is no radius key
in the schema, and `generate_pyro_event_from_property` sets
`nearby_radius = 1.0` with the config read commented out — so the API's
`pyro_nearby_radius` parameter is unreachable from a config file. The shipped
`default_config.yaml` still lists `flammable_nearby_radius`; it does nothing.

Burning prims receive `incident_flaming_item`. The fire volume also emits a
bounding box under the semantic class `FireBoundingBox`; because it is a volume
prim, the rasterization annotators (`semantic_segmentation`,
`bounding_box_2d_tight`) skip it unless `/syntheticdata/sensors/captureVolumes`
is `true`.

### `SpillEvent` → `leakable_item`

| Key | Type | Default |
|---|---|---|
| `item` | str | `$random_leakable_item$` |
| `target_size` | float | `1.0` |
| `leak_duration` | float | `5.0` |

`leak_duration` is in seconds and sets the spill's simulated end time. Leaking
prims receive `incident_leaking_item`; the liquid surface receives
`incident_liquid_spill`.

Liquid instantiates on a prim tagged as a spillable area beneath the item. With
no such prim it spawns at height `0.0`, which reads as liquid floating on the
world plane in a multi-level scene.

## Random item sentinels

| Sentinel | Constant | Used by |
|---|---|---|
| `$random_loose_item$` | `IncidentSettings.RANDOM_LOOSE_ITEM` | `ToppleEvent` |
| `$random_flammable_item$` | `IncidentSettings.RANDOM_FLAMMABLE_ITEM` | `FireEvent` |
| `$random_leakable_item$` | `IncidentSettings.RANDOM_LEAKABLE_ITEM` | `SpillEvent` |

Each manager keeps a set of untouched tagged items and removes what it consumes,
so N events with the same sentinel need N tagged items. When the pool empties,
`select_random_*` logs a carb error and the event is skipped.

Anything else starting with `$` is treated as a literal prim path and will not
match a tagged item.

## Trigger schemas

Built by `TriggersManager.create_trigger_by_dict()`.

```yaml
trigger:
  type: time
  time: 1.5              # seconds
```

```yaml
trigger:
  type: carb_event
  event_name: my_extension_custom_event
```

```yaml
trigger:
  type: physical_event
  incident_name: MyFireEvent    # fires at the START of that IRI event
```

`carb_event` is the integration seam with other extensions: dispatch the named
carb event and the incident starts. `physical_event` chains incidents by
referencing another event's `name`.

## Complete example

```yaml
isaacsim.replicator.incident:
  version: EXT_VERSION      # from the installed extension; major component must match
  global:
    report_dir: /workspace/EventsResult
    seed: 654321
  event:
    event_list:
    - ToppleEvent:
        name: shelf collapse
        topple_item:
          item: $random_loose_item$
          topple_nearby_radius: 1.5
        trigger:
          type: time
          time: 3
    - FireEvent:
        name: pallet fire
        flammable_item:
          item: /World/Pallet
        trigger:
          type: physical_event
          incident_name: shelf collapse
    - SpillEvent:
        name: drum leak
        leakable_item:
          item: $random_leakable_item$
          target_size: 1.5
          leak_duration: 5.0
        trigger:
          type: carb_event
          event_name: my_extension_custom_event
```
