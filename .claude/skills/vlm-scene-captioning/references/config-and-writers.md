# IRC — config schema, caption_configs, and writers

IRC (`isaacsim.replicator.caption.core`) runs three ways: the UI panel, the `CaptionAPI`, or as a
per-frame writer inside Actor SDG (IRA) or Object SDG (IRO). The `caption_configs` keys are shared
across all of them.

## Standalone IRC config

Rooted at `isaacsim.replicator.caption.core:`:

```yaml
isaacsim.replicator.caption.core:
  version: SETTINGS_VERSION            # settings.VERSION of the installed extension, NOT its [package] version
  camera_prim_path: /World/Cameras/Camera
  scene_path: USD_FILE                 # omit to caption the already-loaded stage
  output_path: OUTPUT_PATH
  caption_configs:
    save_full_scene_graph: true
    save_pruned_scene_graph: true
    attach_label_to_usd: false
    use_ai_label: false
    visualize_caption: true
    max_object_capacity: 100
    export_edges: true
    global_caption: true
    qa_caption: false
    brief_caption: true
    pruning_ratio: 1.0
    verbose: true
    random_seed: 0
    caption_only: false
    export_world: true
```

### Global keys

- `version` — must equal `VERSION` in the installed extension's `settings.py`, which is **not** its
  `[package] version`. The two diverged when config-file support landed, so a config carrying the
  `[package] version` is rejected with
  `Config file version <package> does not match the current version <settings>`.
  `scripts/starter_irc_config.py --from-ext` reads the right one.
- `camera_prim_path` — camera prim to caption from; must exist in the scene (falls back to
  `default_config.yaml` if omitted).
- `scene_path` — USD scene to load; if omitted, IRC uses whatever stage is already loaded.
- `output_path` — output directory for captions, scene graphs, and metadata.

### `caption_configs` keys

- `save_full_scene_graph` / `save_pruned_scene_graph` — write the full / pruned graph
  (`<output>/<Camera>/Captions/full_scene_graph.json`, `pruned_scene_graph.json`). The pruned
  graph is a Minimum Spanning Tree (MST) of the Support Tree.
- `pruning_ratio` — fraction of MST edges kept (1.0 = no further pruning); `random_seed` controls
  which edges are dropped when < 1.0.
- `attach_label_to_usd` — auto-attach semantic labels (from prim-path basename) to unlabeled
  prims so annotators capture them.
- `use_ai_label` — use AI-generated labels stored in the database (combine with
  `attach_label_to_usd` for prims lacking pre-stored labels).
- `visualize_caption` — render the graph onto the image
  (`<output>/<Camera>/Captions/vis_camera_scene_graph.jpg`).
- `max_object_capacity` — cap objects in the graph (largest 2D bbox first).
- `export_edges` — write spatial-relationship edges into the graph files.
- `export_world` — also export 3D world locations of prims (default: camera-space only).
- `global_caption` / `qa_caption` / `brief_caption` — which caption types to generate; all land in
  `<output>/<Camera>/Captions/scene_graph_caption.json`.
- `caption_only` — include only prims whose USD files have a preprocessed object caption in the DB.
- `verbose` — print scene-graph details (Support Tree, node/edge counts).

## As a writer inside Actor SDG (IRA)

Model **url**/**name**/**key**/**max_tokens**/**timeout**/**enable_thinking** are not
`SceneGraphWriter` params — there's no per-writer override. `SceneGraphWriter` (and IRO's writers
below) call the model with no override arguments, so they always read these values from the
persisted `isaacsim.replicator.caption.core` extension settings. `max_tokens`/`timeout`/
`enable_thinking` live at that carb-settings layer specifically so every IRC entry point — the UI
panel, the `CaptionAPI`, and any extension embedding these writers — shares one place to tune
them; an embedding extension can edit the settings directly instead of routing through the panel,
for example to work around a model timeout at the source. Set them once beforehand via:

- the Model Settings UI panel — its fields are bound directly to these settings, so **Accept** is
  not required for the writer to pick up a value, only to verify the model can be invoked, or
- the underlying carb settings directly:
  ```
  /persistent/exts/isaacsim.replicator.caption.core/model_url
  /persistent/exts/isaacsim.replicator.caption.core/model_name
  /persistent/exts/isaacsim.replicator.caption.core/api_key
  /persistent/exts/isaacsim.replicator.caption.core/model_max_tokens      # default 1024
  /persistent/exts/isaacsim.replicator.caption.core/model_timeout         # default 30.0
  /persistent/exts/isaacsim.replicator.caption.core/model_enable_thinking # default false
  ```
  e.g. `carb.settings.get_settings().set("/persistent/exts/isaacsim.replicator.caption.core/model_timeout", 60.0)`,
  or `--/persistent/exts/isaacsim.replicator.caption.core/model_timeout=60.0` on the Isaac Sim command line.

`CaptionAPI.set_model_params()` does **not** reach either writer — it only stores values for the
standalone `CaptionAPI.get_captions()` call path (`StageInfoManager._model_params()`), which the
writers never call into. Use the settings above for a value to take effect inside an IRA/IRO run.

Add IRC's `SceneGraphWriter` under `replicator.writers` in the actor config:

```yaml
isaacsim.replicator.agent:
  version: MAJOR.MINOR.0        # from the installed isaacsim.replicator.agent
  # environment / sensor / character as usual ...
  replicator:
    writers:
      SceneGraphWriter:
        rgb: true
        camera_params: true
        object_info_bounding_box_2d_tight: true
        object_info_bounding_box_3d: true
        global_caption: true
        brief_caption: true
        qa_caption: false
        save_full_scene_graph: true
        save_pruned_scene_graph: true
        scene_graph_interval: 10        # graph every N frames (default 1)
        caption_interval: 10            # caption every N frames (default 1000)
```

Extra `SceneGraphWriter` params: `output_dir`, `skip_frames` (default 0), `writer_interval`
(default 1), `export_point_cloud` (default False), `export_depth` (default False). Outputs:

- pruned graph: `<output_dir>/<Camera>/caption_pruned_json/scene_graph_pruned_<frame>.json`
- full graph: `<output_dir>/<Camera>/caption_full_json/scene_graph_full_<frame>.json`
- captions: `<output_dir>/<Camera>/caption/scene_graph_caption_<frame>.json`

## As a writer inside Object SDG (IRO)

Same model-settings caveat as IRA above — `caption_configs` has no `max_tokens`/`timeout`/
`enable_thinking` keys; set them via the UI panel or the carb settings directly first (not
`CaptionAPI.set_model_params()`, which does not reach this writer either).

Add a `caption_configs` block (same keys as above) plus a `caption_writer`, and enable the caption
output switch:

```yaml
isaacsim.replicator.object:
  version: EXT_VERSION          # from the installed isaacsim.replicator.object.core
  # camera_parameters ... etc.
  caption_configs:
    global_caption: true
    brief_caption: true
    qa_caption: true
    caption_writer: CombinedIROSceneGraphWriter   # or IROSceneGraphWriter
  output_switches:
    caption: True
```

- `CombinedIROSceneGraphWriter` — writes IRO outputs **and** captions.
- `IROSceneGraphWriter` — writes captions only, suppressing IRO `labels`; still emits `images`,
  `distance_to_image_plane`, `pointcloud`.

Outputs: `<output>/caption/caption_pruned_json/<seed>_<camera>.json`,
`<output>/caption/caption_full_json/<seed>_<camera>.json`,
`<output>/caption_rgb/<seed>_<camera>.jpg` (visualized), and
`<output>/<Camera>/caption_dict/<seed>_<camera>.json` (captions).
