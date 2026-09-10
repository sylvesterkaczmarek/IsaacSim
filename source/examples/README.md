<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Isaac Sim examples

These examples are small consumers of public, installed Isaac Sim library APIs. They do not import library
implementation packages or add library source directories to build and import paths.

Windows examples require Windows 11. Windows 10 is not supported.

## Structure

```text
source/examples/
├── README.md
├── examples.py              # Public runner included in installed workspaces.
├── example.sh               # Linux source-checkout launcher.
├── example.bat              # Windows source-checkout launcher.
├── libraries/                # Focused examples of public library APIs.
│   └── isaacsim_common/
│       └── logging/
│           └── hello_world/
│               ├── c/
│               ├── cpp/
│               ├── cpp_python/
│               └── python/
└── series/                   # Co-located, ordered example progressions.
```

Examples have exactly one of two forms: a focused library example under `libraries/`, or a step in an ordered example
series under `series/`. Only categories with content are present in Git. Each runnable example has an `example.toml`.
The runner recursively discovers manifests with `published = true` (the default). A manifest can set
`published = false` to exclude an incomplete or private example from the runner and installer. The manifest supplies a
stable ID, ownership, requirement names, structured build and run adapters, and optional test configurations. Stable
IDs do not depend on directory paths.

A series keeps all of its runnable steps beneath one `series/<name>/` directory. Its `series.toml` defines the
authoritative order and assigns each independently runnable step a `beginner`, `intermediate`, or `expert` level.
Series steps can use and declare any number of libraries or concepts.

## `example.toml` reference

Every runnable example has one `example.toml`. The parser rejects unknown entries, missing required entries, duplicate
values where uniqueness is part of the field's meaning, and combinations that do not agree with the selected
adapters. Manifests have no independent version entry because the examples, runner, installer, and libraries ship as
one release. The generated documentation catalog records the release from `source/libraries/VERSION` through the
library catalog.

### Top-level entries

| Entry | Type | Required | Description |
| --- | --- | --- | --- |
| `id` | String | Yes | Stable command-line selector. It is independent of the directory path and consists of lowercase dot-separated segments matching `[a-z][a-z0-9_]*`. |
| `title` | String | Yes | Non-empty user-facing name. |
| `summary` | String | Yes | Non-empty short description of the example's purpose. |
| `owners` | Array of strings | Yes | Non-empty list of logical public API owners, such as `isaacsim.common.logging`. Each value follows the same syntax as `id`. |
| `published` | Boolean | No | Whether the runner and installer discover this manifest. Defaults to `true`. Set to `false` to exclude it. |
| `categories` | Array of strings | No | Discovery categories. Values match `[a-z][a-z0-9_]*`; an empty array is allowed. |
| `topics` | Array of strings | No | Cross-cutting discovery topics, such as `logging` or `python`. Values follow the same rules as `categories`. |
| `build` | Table | Yes | Structured build adapter configuration. |
| `run` | Table | Yes | The example's single normal run entry point. |
| `requirements` | Table | Yes | Direct library distributions, installed surfaces, and system capabilities consumed by the example. |
| `test` | Array of tables | No | Named automated-test overlays for the normal run entry point. |

### `[build]` entries

| Entry | Type | Required | Description |
| --- | --- | --- | --- |
| `adapter` | String | Yes | `none` for an example with no build step, or `cmake` for a CMake project in the example root. |
| `targets` | Array of strings | For `cmake` | Non-empty CMake target list. A target starts with an alphanumeric character or underscore and then uses only alphanumerics, `_`, `.`, `+`, or `-`. |

The `none` adapter accepts no other entries. The `cmake` adapter requires `CMakeLists.txt` in the example root.

### `[run]` entries

| Entry | Type | Required | Description |
| --- | --- | --- | --- |
| `adapter` | String | Yes | `executable` for a CMake-built program, or `python` for a Python script. |
| `target` | String | For `executable` | CMake target to execute. It must also appear in `build.targets`. |
| `path` | String | For `python` | Existing script path relative to the example root. Absolute paths and paths that escape the example are rejected. |
| `arguments` | Array of strings | No | Default arguments passed to the normal entry point exactly in order. The array and individual strings may be empty, and duplicate values are allowed. |

An `executable` run adapter requires the `cmake` build adapter. A Python-only example normally uses
`build.adapter = "none"` with `run.adapter = "python"`.

### `[requirements]` entries

| Entry | Type | Required | Description |
| --- | --- | --- | --- |
| `system` | Array of strings | No | System capability names used by the example. Supported values are `cmake`, `c`, `cpp`, `python`, and `python_development`. Versions come from `source/libraries/system_requirements.toml`. |
| `modules` | Array of tables | Yes | Non-empty list of direct Isaac Sim library distributions. Author each item with `[[requirements.modules]]`. |

Each `[[requirements.modules]]` item accepts:

| Entry | Type | Required | Description |
| --- | --- | --- | --- |
| `name` | String | Yes | Direct distribution name, such as `isaacsim_common`. Names are unique within the manifest and match `[a-z][a-z0-9_]*`. |
| `surfaces` | Array of strings | Yes | Non-empty installed surfaces consumed from that distribution. `native_sdk` provides native runtime, headers, libraries, and CMake packages; `python` provides its Python distribution. |

Adapter and requirement declarations must be coherent:

- `build.adapter = "cmake"` requires a `native_sdk` module surface plus the `cmake` capability and at least one of
  `c` or `cpp`.
- `run.adapter = "python"` requires the `python` capability.
- A CMake-built Python example also requires `python_development`.
- `python_development` cannot be declared without `python`.

### `[[test]]` entries

Each `[[test]]` table defines a named configuration of the normal run entry point. Tests do not define another build
or run adapter.

| Entry | Type | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `name` | String | Yes | — | Unique test selector within the example. It matches `[a-z][a-z0-9_]*`. |
| `arguments` | Array of strings | No | The normal `run.arguments` | Replacement arguments passed exactly in order. An explicitly empty array runs with no arguments; empty strings and duplicate values are allowed. |
| `timeout_seconds` | Positive integer | No | `60` | Maximum run time before the runner terminates the process tree and fails the test. |
| `expected_exit_code` | Integer | No | `0` | Required process exit code. |
| `stdout_contains` | Array of strings | No | Empty | Substrings that must each occur in captured standard output. |
| `stderr_contains` | Array of strings | No | Empty | Substrings that must each occur in captured standard error. |
| `environment` | Table of string values | No | Empty | Environment variables overlaid on the inherited environment for this test. |

Write the environment overlay as `[test.environment]` immediately after its `[[test]]` table. Variable names must
match `[A-Za-z_][A-Za-z0-9_]*`. Values must be strings and must not contain secrets.
`PYTHONDONTWRITEBYTECODE` cannot be overridden.

An example can omit `[[test]]`; the runner's `test` command then reports it as skipped. Multiple `[[test]]` tables can
exercise different argument, environment, exit-code, timeout, and output configurations.

## Requirements

- Treat `source/libraries/system_requirements.toml` as the authoritative CMake, C, C++, and Python version policy for
  this release. The library build and examples runner both consume it.
- Use the Python version declared in that file to invoke `examples.py`. In a source checkout, the launchers use the
  repository's Packman Python only to bootstrap the runner.
- Install every `native_sdk` and `python` surface declared by the published examples being used. The complete current set
  consumes the native SDKs for `isaacsim_common`, `isaacsim_foundation`, and `isaacsim_ovgl_viewport`. It consumes the
  Python distributions for those modules plus `isaacsim_physics` and `isaacsim_physics_engines`.
- Install the declared Python version and its development headers for the mixed native/Python example.
- Install the declared minimum CMake version and compilers supporting the declared C and C++ standards for native
  builds.

When both are installed, the native SDK and Python distribution must have the same version. Libraries, the installer,
and examples use the release from `source/libraries/VERSION`.

In a source checkout, the runner uses the matching configured `source/libraries` build by default. A complete library
build creates one configuration-specific developer environment containing every package's runtime, development, and
Python surfaces. The runner selects the build's pinned Python and CMake, validates the completed-build state against the
Pixi lock, Packman/source manifests, and library source fingerprint, and uses that environment without installing or
replacing dependencies for individual examples. Multiple examples can share it concurrently; a library rebuild waits
before replacing an environment that is in use. Build the libraries once before running examples:

```bash
cd source/libraries
./build.sh -r
```

Repository examples share the root `pixi.lock`; source examples intentionally do not carry duplicate per-example
locks. `example.toml` remains behavior and capability metadata, while `pixi.toml` remains environment metadata. An
independently released standalone example may receive a generated release manifest and approved lock as an export
artifact, but that lock is not edited in the source example directory.

An example manifest lists only the system capabilities it uses:

```toml
[requirements]
system = ["cmake", "cpp"]
```

The manifest never repeats tool or language versions.

## Build, run, and test

### Source checkout

Run the commands from the repository root:

```bash
# Build every example.
source/examples/example.sh build

# Build one example.
source/examples/example.sh build hello_world.cpp

# Incrementally build and run one example by stable ID.
source/examples/example.sh run hello_world.python

# Run an example by its root or declared Python script path, relative to `source/examples`.
source/examples/example.sh run series/falling_cube/simulate_cube
source/examples/example.sh run series/falling_cube/simulate_cube/main.py

# Pass additional arguments to the normal entry point.
source/examples/example.sh run <example-id-or-path> -- <example-arguments>

# Incrementally build and test every configured example.
source/examples/example.sh test

# Test examples that were built earlier without invoking the compiler.
source/examples/example.sh test --no-build

# Run every test for one example or one named configuration.
source/examples/example.sh test hello_world.cpp
source/examples/example.sh test hello_world.cpp:default
```

The source launchers always select developer mode, so no environment activation is required. They start `examples.py`
with the repository's Packman Python; the runner then switches to the configured library build's Python and prepares
the selected libraries. Do not use a source launcher with `--installed`.

### Installed examples workspace

The installer materializes `README.md`, `examples.py`, and the selected examples, but it does not materialize the
source-checkout launchers or repository tools. Activate the Python environment containing the required distributions,
set `CMAKE_PREFIX_PATH` to the installed SDK prefix when building native examples, and run the public runner directly:

```bash
python examples.py build
python examples.py run hello_world.python
python examples.py test
```

Installed and installer-materialized workspaces default to installed mode. Pass `--installed` to require it explicitly:

```bash
python examples.py --installed run hello_world.python
```

The runner prints the selected environment before operating on an example.

Compiled examples use isolated directories under `_build/examples/<example-id>/`. `build` and `test` select every
published example when no IDs are supplied. Pass `test --no-build` only after building the same selection with the same
build root and configuration. `run` accepts one stable ID, published example root, or declared Python script path
relative to `source/examples`. It does not execute arbitrary paths. By default, `run` and `test` build their selected
examples incrementally before execution. A custom `--build-root` must remain outside `source/examples`.

For Python-only examples, you can also run the declared script directly after activating the required Python
environment. From `source/examples`:

```bash
python series/falling_cube/simulate_cube/main.py
```

Run native and mixed native/Python examples through the runner, which performs their incremental build and selects
their executable entry point.

An example without a `[[test]]` table is reported as skipped. Test arguments replace the normal run arguments when
present. Tests inherit the current environment and can overlay non-secret values using portable environment variable
names. Test overlays cannot enable Python bytecode output in an example root. The default timeout is 60 seconds, and
the default expected exit code is zero.

In a source checkout, use `source/examples/example.sh --help` on Linux or `source\examples\example.bat --help` on
Windows for build configuration options. In an installed workspace, use `python examples.py --help`.

## Assets

Examples may author scenes in code or load caller-supplied assets. The OVGL viewport example expects the input USD and
all of its dependencies to be available locally. The runner does not download, materialize, mount, cache, or resolve
assets; storage and resolver configuration remain external to the example manifests.

## Installer behavior

The installer recursively discovers published `example.toml` files, then reads the selected manifest to resolve its
library dependencies. For native examples, it retrieves the SDK and reports the prefix to use as
`CMAKE_PREFIX_PATH`. For Python examples, it installs the required distribution into the installer's Python
environment.

The installer creates a minimal user-owned workspace containing the selected payload, this runner and README, and the
release-wide system requirements. A selected series includes all of its ordered steps. Canonical Git source contains no
generated files.
