# References — generate-incident-config

Loaded on demand. Read `SKILL.md` first.

| File | Read it when |
|---|---|
| [`config-schema.md`](config-schema.md) | You need exact key names, types, defaults, and trigger schemas for the IRI config YAML. |
| [`authoring-notes.md`](authoring-notes.md) | A config loads without error but behaves wrong, or you need to know what the loader does with an unexpected key. |

Everything here was read from the `isaacsim.replicator.incident.core` and
`omni.metropolis.utils` sources as resolved by the Action and Event Data
Generation app. Both are consumed from `extscache`; they are authored in the
`metrosim` repo, and the app's `.kit` file is where their exact versions live.
