#!/usr/bin/env python3
"""Generate a starter Behavior Tree Generation context entry or metadata schema offline.

Emits a starter actor/object context entry (following ``ActorInfo`` /
``InteractableObjectInfo``) or a matching JSON-Schema skeleton for the ``metadata`` block. It
does NOT launch Isaac Sim, so it runs anywhere. Feed the results to the Behavior Tree Gen
pipeline via ``context_data_paths`` / ``actor_schema_path`` / ``object_schema_path``.

Usage:
    python3 starter_context.py --entity object --id Table [--prim-path /World/Foo]
    python3 starter_context.py --entity actor  --id Anna
    python3 starter_context.py --emit-schema object   # JSON Schema for object metadata
    python3 starter_context.py --emit-schema actor    # JSON Schema for actor metadata

Prints JSON to stdout. Exit 0 on success, 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import json
import sys

# Required metadata fields the shipped example schemas enforce.
REQUIRED = {
    "actor": ["prim_path", "actor_type"],
    "object": ["prim_path", "interactable_type"],
}


def context_entry(entity: str, entry_id: str, prim_path: str | None) -> dict:
    prim_path = prim_path or f"/World/{'Actors' if entity == 'actor' else 'TestEnv'}/{entry_id}"
    if entity == "actor":
        metadata = {
            "actor_type": "human",
            "role": "test_actor",
            "location": {"x": 0.0, "y": 0.0, "z": 0.0},
            "prim_path": prim_path,
            "semantic_label": "human",
        }
        interactions = []
    else:
        metadata = {
            "interactable_type": "table",
            "prim_path": prim_path,
            "semantic_label": "office table",
            "move_to_targets": {"default": f"{prim_path}/MoveToTarget"},
            "placement_targets": {"middle": f"{prim_path}/PlaceObject_Middle"},
        }
        interactions = ["move to", "place on"]
    return {
        "id": entry_id,
        "semantic_description": f"A starter {entity} named {entry_id}; edit this description.",
        "metadata": metadata,
        "supported_interactions": interactions,
        "entity_type": entity,
    }


def metadata_schema(entity: str) -> dict:
    if entity == "actor":
        props = {
            "prim_path": {"type": "string", "description": "USD path for the actor."},
            "actor_type": {"type": "string", "description": "Category of the actor, e.g. human."},
            "role": {"type": "string", "description": "Planner-facing role label."},
            "semantic_label": {"type": "string", "description": "Semantic label for annotation."},
            "location": {
                "type": "object",
                "properties": {
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "z": {"type": "number"},
                },
                "description": "World-space spawn location.",
            },
        }
    else:
        props = {
            "prim_path": {"type": "string", "description": "USD path for the object."},
            "interactable_type": {"type": "string", "description": "Category of the object."},
            "semantic_label": {"type": "string", "description": "Semantic label for annotation."},
            "move_to_targets": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Named USD paths an actor can move to (e.g. 'default').",
            },
            "placement_targets": {
                "type": "object",
                "additionalProperties": {"type": "string"},
                "description": "Named USD paths where objects can be placed (e.g. 'middle').",
            },
        }
    return {"type": "object", "properties": props, "required": REQUIRED[entity]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--entity", choices=["actor", "object"], help="Generate a starter context entry.")
    p.add_argument("--id", dest="entry_id", help="Entity id (required with --entity).")
    p.add_argument("--prim-path", help="Override the metadata.prim_path USD path.")
    p.add_argument(
        "--emit-schema",
        choices=["actor", "object"],
        help="Emit a JSON Schema for the entity's metadata instead of a context entry.",
    )
    args = p.parse_args(argv)

    if args.emit_schema:
        json.dump(metadata_schema(args.emit_schema), sys.stdout, indent=2)
    elif args.entity:
        if not args.entry_id:
            p.error("--id is required with --entity")
        json.dump(context_entry(args.entity, args.entry_id, args.prim_path), sys.stdout, indent=2)
    else:
        p.error("provide --entity <actor|object> --id <name>, or --emit-schema <actor|object>")
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
