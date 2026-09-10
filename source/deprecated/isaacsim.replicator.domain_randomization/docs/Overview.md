# Overview

```{deprecated} 6.0.0
This extension is deprecated in favor of `isaacsim.replicator.experimental.domain_randomization`.
```

`**isaacsim.replicator.domain_randomization**` provides domain randomization scripts and OmniGraph nodes for randomizing simulation environments. It is intended for synthetic data workflows where scene, object, or simulation properties need to vary across runs or frames. The module exposes grouped APIs for defining when randomization happens, how it is gated, and how physics-related randomization is accessed.

## Concepts

Domain randomization changes simulation parameters to increase variation in generated data. In this extension, randomization behavior is organized around a few public namespaces rather than a large flat API.

The main concepts are:

- Triggers define when randomization logic should run.
- Gates control whether a randomization path is active.
- Physics views provide access to physics-related randomization workflows.
- Utilities provide shared helpers used by domain randomization scripts and nodes.

## Functionality

The extension provides two main kinds of functionality:

- Python scripts for building domain randomization behavior.
- OmniGraph nodes for composing randomization workflows in graph-based pipelines.

This allows randomization to be used from Python code or as part of an OmniGraph-based synthetic data pipeline. The extension is focused on simulation environment variation, especially in the context of Replicator workflows.

## Key Components

### `trigger`

`trigger` contains APIs related to when randomization should occur. Use this namespace when defining randomization points such as frame-based, event-based, or other trigger-driven behavior exposed by the implementation.

```python
import isaacsim.replicator.domain_randomization as dr

# Access trigger-related APIs
dr.trigger
```

### `gate`

`gate` contains APIs related to controlling whether randomization logic is allowed to execute. This is useful when a randomization workflow needs conditional behavior, for example only applying a randomization branch under specific simulation or graph conditions.

```python
import isaacsim.replicator.domain_randomization as dr

# Access gate-related APIs
dr.gate
```

### `physics_view`

`physics_view` contains APIs for physics-related domain randomization workflows. Use this namespace when randomization needs to interact with simulation-facing physics data rather than only visual or scene-level properties.

```python
import isaacsim.replicator.domain_randomization as dr

# Access physics view APIs
dr.physics_view
```

### `utils`

`utils` contains shared helper APIs used by the domain randomization implementation. These helpers are intended to support randomization scripts and graph-node workflows without requiring users to work directly with lower-level implementation details.

```python
import isaacsim.replicator.domain_randomization as dr

# Access utility APIs
dr.utils
```

## Relationships

`**isaacsim.replicator.domain_randomization**` is built for Replicator-based domain randomization workflows. The extension depends on `**omni.replicator.core**`, which provides the Replicator foundation used by synthetic data generation workflows.

The extension also provides OmniGraph nodes and depends on `**omni.graph**` and `**omni.graph.core**` for graph-based randomization composition. This lets randomization behavior be represented as graph nodes in addition to Python scripts.
