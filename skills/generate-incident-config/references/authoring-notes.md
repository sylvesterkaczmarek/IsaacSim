# IRI Config Authoring Notes

Why a config can load cleanly and still be wrong. All behavior below was read
from the `isaacsim.replicator.incident.core` and `omni.metropolis.utils` sources
as resolved by the Action and Event Data Generation app.

## The loader drops unknown keys in silence

This is the single most important thing to know about the format.
`EventPropertyBase.set_value()` does not merge your mapping into the defaults —
it starts from the defaults and copies across only the keys it already knows:

```python
self.value = self.default_value.copy()
if "name" in new_val:
    self.value["name"] = new_val["name"]
for default_item_name, default_item_value in self.default_items.items():
    if default_item_name in new_val:
        item = new_val[default_item_name]
        for key in list(item.keys()):
            if key in list(default_item_value.keys()):   # <-- the filter
                self.value[default_item_name][key] = item[key]
```

Consequences:

- `topple_radius: 3.0` (instead of `topple_nearby_radius`) → the default `1.5`
  is used. No warning, no log line.
- A key that was valid in an older build and has since been removed behaves the
  same way — the config keeps loading, quietly using defaults.
- An entire misspelled item group (`topple_items:` with an `s`) → every value in
  that group falls back to its default.

Builds without the shipped schema have no strict mode at all.
`starter_iri_config.py --validate` exists because the extension will not tell you.

Builds that do ship `data/config_schema/iri_config.schema.json` validate
against it before the permissive loader
runs, so unknown keys are reported at load time and (by default) refuse the load.
The `/exts/isaacsim.replicator.incident/strict_config_validation` setting
downgrades those to warnings for configs written against an older schema.

The script detects that schema and validates against it when present
(`mode=schema`), falling back to its own equivalent checks otherwise
(`mode=builtin`). So the same command is correct on every build; only the source
of the rules changes.

Contrast with `caption_configs` in IRC, which *rejects* unknown keys. The two
extensions are inconsistent here; do not carry an assumption from one to the other.

## Version checking is major-only

`check_version` compares `str(version).split(".")[0]` on both sides. Against a
`0.x` build, every `0.*` config passes. Practical effects:

- A config written for one `0.x` version still loads against any other — even if
  the schema changed underneath it, in which case the removed keys are dropped per
  the rule above and you get defaults.
- The version check is therefore **not** a schema guarantee. Treat it as a
  coarse guard and validate structurally regardless.

The shipped `<ext>/config/default_config.yaml` declares a long-stale `version`
and still loads without complaint, which is why the inert
`flammable_nearby_radius` key in it has survived.

## Failure paths log, they do not raise

Nothing in the config → setup path throws on a content error:

| Situation | What happens |
|---|---|
| Missing header | `load_config_file` returns `None`, carb error logged |
| Missing / mismatched `version` | returns `None`, carb error logged |
| Unregistered section name | ignored, no message |
| Unknown key in an event | dropped, no message |
| `seed` negative | carb error from `verify_int_non_negative`, property marked in error |
| Event item not tagged on the stage | carb error, that event is skipped, the others still set up |
| Random pool exhausted | carb error from `select_random_*`, that event is skipped |

So a partially-applied config looks identical to a fully-applied one unless you
count what registered. `run-incident-events action=setup` compares requested
against registered for exactly this reason.

## Random selection and pool exhaustion

Each manager holds a set of tagged items it has not yet used, and removes items
as events consume them:

- topple: `untoppled_items`, and `topple_nearby_radius` consumes *every* nearby
  loose item, not only the selected one.
- fire: `unflammed_items`.
- spill: `unspilled_items`.

Two consequences when authoring:

1. N events on the same sentinel need at least N tagged items of that type —
   more for topple, since each event may swallow several.
2. A large `topple_nearby_radius` in an early event can starve a later one. Order
   matters; events are set up in list order.

## `report_dir` is unverified

`IncidentGlobalSection` gives `report_dir` an empty `verify_funcs` list, so any
string is accepted at load time. A non-writable or nonexistent path only fails
at `end_recording()`, after the run — and that failure is a carb error, not an
exception. Prefer a path under `$WORKSPACE_DIR` and confirm the JSON exists
afterwards.

## What the config does not cover

- **Scene tagging.** Which prims are loose / flammable / leakable / spillable
  area lives in USD attributes on the stage, not in this file. A config against
  an untagged stage sets up nothing. See `run-incident-events`.
- **Navmesh.** Topple items tagged `NavMesh` need a baked navmesh at setup time.
- **Recording.** `report_dir` names the destination, but recording is started and
  stopped explicitly; loading a config does not begin one.
- **Persistence.** The config is not part of the USD stage. Saving the scene does
  not save it, and opening the scene does not restore it.

## Hand-editing checklist

1. Header is exactly `isaacsim.replicator.incident:`.
2. `version` present, major matches the installed build.
3. Every `event_list` entry is a single-key mapping keyed `ToppleEvent`,
   `FireEvent`, or `SpillEvent`.
4. Every event has a `trigger` with the field its type requires
   (`time` / `event_name` / `incident_name`).
5. Item-group keys match the schema exactly — this is what `--validate` is for.
6. Enough tagged items exist for every event that uses a random sentinel.
