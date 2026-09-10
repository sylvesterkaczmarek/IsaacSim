# Isaac Sim ROS Workspaces Build Matrix

This matrix contains supported workspace build paths only. It intentionally omits deprecated
WSL2 and unsupported distro/platform combinations.

## Build methods

| Method | Selection | ROS source | Build command | Output |
|---|---|---|---|---|
| Native | Matching ROS install exists, or `--method native` | Existing host installation | `rosdep install`, then `colcon build` | `<distro>_ws/install` |
| Docker | ROS absent, or `--method docker` | Prebuilt ROS container image | Containerized `rosdep install` and `colcon build` | `<distro>_ws/install` |
| Custom | Custom interfaces, Python 3.12, or `build_ros.sh` | Repository Dockerfiles | `./build_ros.sh -d <distro> -v 22.04` | `build_ws/<distro>/...` |
| Pixi | Windows default or explicit Linux Pixi | Pixi/RoboStack environment | `pixi install`, then `pixi run build` | `<distro>_ws/install` |

The Docker and Custom methods both require Docker, but are not interchangeable. Docker mounts
one workspace into a prebuilt ROS image. Custom builds the repository's Python 3.12 ROS image
and workspaces from its Dockerfiles.

## Platform support

| Host | Distro | Native | Docker | Custom | Pixi |
|---|---|---:|---:|---:|---:|
| Ubuntu 24.04 x86_64 | Jazzy | Yes | Yes | No | Explicit only |
| Ubuntu 24.04 aarch64 | Jazzy | Yes | Yes | No | Explicit only |
| Ubuntu 24.04 x86_64 | Humble | No | Yes | No | Explicit only |
| Ubuntu 24.04 aarch64 | Humble | No | Yes | No | Explicit only |
| Ubuntu 22.04 x86_64 | Humble | Yes | Yes | Yes | Explicit only |
| Ubuntu 22.04 aarch64 | Humble | Yes | Yes | Yes | Explicit only |
| Ubuntu 22.04 x86_64 | Jazzy | No | Yes | Yes | Explicit only |
| Ubuntu 22.04 aarch64 | Jazzy | No | Yes | Yes | Explicit only |
| Windows 11 x64 | Jazzy | No | No | No | Yes (default) |

Cross-distro native builds are rejected. Use Docker for Humble workspaces on Ubuntu 24.04 and
Jazzy workspaces on Ubuntu 22.04. Windows Humble and WSL2 are not supported by this skill.

## Docker images

| Distro | x86_64 | aarch64 |
|---|---|---|
| Jazzy | `osrf/ros:jazzy-desktop` | `arm64v8/ros:jazzy` |
| Humble | `osrf/ros:humble-desktop` | `arm64v8/ros:humble` |

The setup script detects architecture and reuses a local image when present; it pulls only when
the selected image is missing.

## Prerequisites by method

### Native

- Matching ROS setup file exists.
- The script checks existing workspace build tools before requesting sudo:
  - `python3-rosdep`
  - `python3-colcon-common-extensions`
  - `build-essential`
- If all tools and rosdep sources are present, apt and sudo are skipped.
- If anything is missing, an interactive terminal or pre-authenticated/passwordless sudo is
  required. A non-interactive session exits with exact remediation commands instead of hanging.
- The script initializes/updates rosdep and resolves dependencies declared in `src`.
- Native rosdep/apt runs with `DEBIAN_FRONTEND=noninteractive`, `NEEDRESTART_MODE=a`, and
  `PIP_BREAK_SYSTEM_PACKAGES=1` so agent sessions do not hit whiptail dialogs or PEP 668 pip
  blocks.

### Docker

- Docker CE is already installed.
- Docker daemon is running.
- `docker info` succeeds for the current user.
- Repository submodules are initialized on the host before the container starts.
- The container stays root for apt/rosdep/colcon, then restores the bind-mounted workspace
  ownership to the calling user so `build`/`install`/`log` do not remain root-owned.
- Build containers use Docker's default networking. Runtime DDS host-network mode and
  `ROS_DOMAIN_ID` belong in the container used to source and run the workspace, not in the
  build step.

The skill does not install Docker. A failed Docker gate must stop before pulling an image.

### Custom interfaces

- Ubuntu 22.04.
- Docker passes the same gate as the Docker method.
- The cloned repository contains executable `build_ros.sh`.
- The requested distro is Humble or Jazzy.

### Windows Pixi

- Windows 11 x64, build 22000 or newer.
- `winget` available.
- MSVC Build Tools 2022 with the C++ workload already installed.
- Git and Pixi are installed by the script through `winget` when absent.
- Jazzy workspace contains `pixi.toml`.

### Linux Pixi

- Explicit user request.
- Ubuntu 22.04 selects Humble; Ubuntu 24.04 selects Jazzy.
- Official Pixi binary is downloaded from GitHub releases and verified with the published
  `.sha256` sidecar when Pixi is absent.
- Selected workspace contains `pixi.toml`.
