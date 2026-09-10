# Overview

```{deprecated} 6.0.0
This extension is deprecated. No replacement is provided.
```

`**omni.isaac.ml_archive**` provides archived pip packages needed by Isaac extensions that use machine learning related Python dependencies. It is a package distribution extension rather than a user-facing tool, so its purpose is to make bundled Python packages available to other extensions during application startup.

The extension is platform-specific because the pip prebundle content can vary by operating system and Python environment.

## Functionality

`**omni.isaac.ml_archive**` contributes a `pip_prebundle` Python module path that contains the packaged Python dependencies. Other extensions can rely on those bundled packages being available without downloading them at runtime.

The extension is loaded early so dependent extensions can import their required Python packages before their own functionality runs. This is important for extensions that expect machine learning dependencies to already be present in the Python environment.

## Key Components

### Pip Prebundle

The `pip_prebundle` path is the main packaged content of the extension. It provides the archived pip packages that are distributed with the application build.

### Archive Module

The `**omni.isaac.ml_archive**` Python module is present as an extension module entry. It is used for extension discovery and test discovery rather than exposing runtime functionality.

## Dependencies

`**omni.isaac.ml_archive**` depends on:

- `**omni.isaac.core_archive**`, which pulls in the main Isaac pip archive.
- `**omni.kit.pip_archive**`, which provides the base Kit Python archive support.

Together, these archive extensions provide the Python package bundle used by Isaac extensions.
