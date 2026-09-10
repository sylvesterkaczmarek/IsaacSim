# Behavior Tree Generation — API and inputs reference

Public API facade: `omni.ai.behavior_tree_gen.core.api` re-exports exactly three functions from
`...core.runtime.planner_runtime`: `setup_workspace`, `prepare_runtime`, `generate_behavior_tree`.
Everything else under `...core.runtime` is internal. Call the three in order — each returns state
the next one consumes.

## 1. `setup_workspace(...)` → `PlannerSession` (synchronous)

Builds the reusable workspace and loads input data. Every argument is technically optional, but
real runs provide:

| Parameter | Purpose |
|---|---|
| `cache_dir` | Folder for reusable planner cache artifacts. |
| `output_dir` | Folder where generated behavior-tree files are written. |
| `context_data_paths` | Actor + object context JSON files. |
| `node_catalog_paths` | Behavior-tree node-catalog JSON files. |

Common optional inputs: `vectorstore_dir` (RAG/vectorstore; derived from `cache_dir` if omitted),
`actor_schema_path`, `object_schema_path`, `blackboard_data_paths`, `apply_blackboard_cache`,
`refresh_cache` (rebuild caches before loading).

Returns a `PlannerSession` with fields `actors_loaded`, `objects_loaded`, `nodes_loaded`,
`workspace_ready`. The session can be reused across runs when tracked inputs don't change.

## 2. `await prepare_runtime(session, api_key=..., ...)` → `PlannerRuntimeResult` (coroutine)

Configures model access, retrievers, and Action IR.

Required: `session`, `api_key` (NVIDIA API key). Common optional: `model_selection_config_path`,
`embedding_model_configs_path`, `named_model_configs_path`, `node_to_model_map_path`,
`actor_types` (warm only these into Action IR), `prefer_cached_action_ir`, and direct overrides
`llm_model` / `embedding_model` / `rag_top_k` / `rag_similarity_threshold`.

Returns `PlannerRuntimeResult`: `success`, `message`, `chat_model_name`, `retriever_ready`,
`action_ir_ready`, `warmed_actor_types`. Check `success` before continuing.

## 3. `await generate_behavior_tree(session, scenario, ...)` → `PipelineResult` (coroutine)

Required: `session` (with a ready workspace + prepared runtime), `scenario` (natural-language
text). Optional: `skip_phase3` (skip tree-construction stage when debugging earlier stages).

Returns `PipelineResult`: `success`, `duration_seconds`, `behavior_tree_folder_path`,
`error_message`.

## Async execution in Script Editor

`setup_workspace` is sync; the other two are coroutines. Wrap all three in one `async def` and
schedule it:

```python
import asyncio
from omni.ai.behavior_tree_gen.core import api as core_api

async def run():
    session = core_api.setup_workspace(cache_dir=CACHE_DIR, output_dir=OUTPUT_DIR)
    rt = await core_api.prepare_runtime(session, api_key=API_KEY)
    if not rt.success:
        raise RuntimeError(rt.message)
    res = await core_api.generate_behavior_tree(session, SCENARIO)
    print(res.success, res.behavior_tree_folder_path)

asyncio.ensure_future(run())
```

The `omni.ai.behavior_tree_gen.bridge` UI performs these same three calls automatically. The
bridge also provides `omni.ai.behavior_tree_gen.bridge.utils` helpers
(`get_example_scene_context_files`, `load_example_scene_config`, `get_extension_path`,
`resolve_example_scene_file_path`) that resolve the bundled `simple` / `warehouse` example files.

## Context files and metadata schemas

Context = the runtime knowledge base of actors/objects; each entry is a JSON object following one
of two base models from `...core.pydantic.models.context_models`:

- `ActorInfo` — `entity_type: "actor"`
- `InteractableObjectInfo` — `entity_type: "object"`

Both share the top-level shape:

```json
{
  "id": "UniqueName",
  "semantic_description": "Natural-language description of this entity.",
  "metadata": { },
  "supported_interactions": [],
  "entity_type": "object"
}
```

- `id` — stable identifier.
- `semantic_description` — used in retrieval/grounding.
- `metadata` — domain-specific fields defined by the actor/object JSON Schema.
- `supported_interactions` — optional passive-interaction labels (normalized to lowercase);
  defined on the base model, **not** in the metadata schema.
- `entity_type` — `"actor"` or `"object"`.

Metadata schema = a standard JSON Schema document defining the `metadata` dict. The core builds a
typed model from it (`build_metadata_model_from_json_schema`, `allow_extra=True`) and attaches it
to `ActorInfo` / `InteractableObjectInfo`. Shipped example required fields:

- Actors: `metadata.prim_path`, `metadata.actor_type`.
- Objects: `metadata.prim_path`, `metadata.interactable_type`.

Schema-defined fields expand into planner-visible term paths used for grounding/retrieval, e.g.
`actors.metadata.role`, `actors.metadata.location.x`, `objects.metadata.move_to_targets.default`,
`objects.metadata.placement_targets.left_end`. Add a custom field by editing the schema, adding
the same field under each matching context entry's `metadata`, then reloading the workspace.
