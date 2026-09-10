#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail

METHOD="auto"
DISTRO=""
REPO_URL="https://github.com/isaac-sim/IsaacSim-ros_workspaces.git"
REPO_PATH="${HOME}/IsaacSim-ros_workspaces"
BRANCH="main"
ROS_SETUP=""
SKIP_BUILD=false
CLEAN_RECLONE=false
YES_I_KNOW=false

usage() {
    cat <<'EOF'
Usage: setup_ros_workspace.sh [options]

Options:
  --method auto|native|docker|custom|pixi
  --distro humble|jazzy
  --repo-url URL
  --repo-path PATH
  --branch NAME
  --ros-setup PATH       Existing ROS setup.bash (native method only)
  --clean-reclone        Delete an existing Git checkout and clone it again
  --yes-i-know           Required with --clean-reclone when stdin is not a TTY
  --skip-build           Clone and prepare only
  -h, --help

Examples:
  setup_ros_workspace.sh
  setup_ros_workspace.sh --method docker --distro jazzy
  setup_ros_workspace.sh --method custom --distro humble
  setup_ros_workspace.sh --method pixi --repo-url https://git.example/repo.git
  setup_ros_workspace.sh --clean-reclone --yes-i-know
EOF
}

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

step() {
    printf '\n==> %s\n' "$*"
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || fail "$1 requires a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            need_value "$@"
            METHOD="$2"
            shift 2
            ;;
        --distro)
            need_value "$@"
            DISTRO="$2"
            shift 2
            ;;
        --repo-url)
            need_value "$@"
            REPO_URL="$2"
            shift 2
            ;;
        --repo-path)
            need_value "$@"
            REPO_PATH="$2"
            shift 2
            ;;
        --branch)
            need_value "$@"
            BRANCH="$2"
            shift 2
            ;;
        --ros-setup)
            need_value "$@"
            ROS_SETUP="$2"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        --clean-reclone)
            CLEAN_RECLONE=true
            shift
            ;;
        --yes-i-know)
            YES_I_KNOW=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            fail "unknown argument: $1 (use --help)"
            ;;
    esac
done

# --yes-i-know only authorizes a clean reclone; refuse a stray flag.
if [[ "$YES_I_KNOW" == true && "$CLEAN_RECLONE" != true ]]; then
    fail "--yes-i-know is only valid together with --clean-reclone"
fi

case "$METHOD" in
    auto|native|docker|custom|pixi) ;;
    *) fail "--method must be auto, native, docker, custom, or pixi" ;;
esac

[[ "$(uname -s)" == "Linux" ]] || fail "this script supports Linux only; use setup_pixi_workspace.ps1 on Windows"
[[ -r /etc/os-release ]] || fail "cannot detect Ubuntu version: /etc/os-release is missing"

# shellcheck source=/dev/null
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || fail "supported Linux hosts are Ubuntu 22.04 and Ubuntu 24.04"

UBUNTU_VERSION="${VERSION_ID:-}"
case "$UBUNTU_VERSION" in
    22.04) DEFAULT_DISTRO="humble" ;;
    24.04) DEFAULT_DISTRO="jazzy" ;;
    *) fail "unsupported Ubuntu version: ${UBUNTU_VERSION:-unknown}" ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
    x86_64|aarch64) ;;
    *) fail "unsupported architecture: $ARCH (expected x86_64 or aarch64)" ;;
esac

DISTRO="${DISTRO:-$DEFAULT_DISTRO}"
case "$DISTRO" in
    humble|jazzy) ;;
    *) fail "--distro must be humble or jazzy" ;;
esac

if [[ -z "$ROS_SETUP" ]]; then
    ROS_SETUP="/opt/ros/${DISTRO}/setup.bash"
fi

if [[ "$METHOD" == "auto" ]]; then
    if [[ -f "$ROS_SETUP" && "$DISTRO" == "$DEFAULT_DISTRO" ]]; then
        METHOD="native"
    else
        METHOD="docker"
    fi
    step "Auto-selected method: $METHOD"
fi

# Cross-distro hosts (Humble on Ubuntu 24.04, Jazzy on Ubuntu 22.04) use Docker
# (or Custom/Pixi when those methods already allow the pair). Native requires the
# matching host-default distro.
if [[ "$METHOD" == "native" && "$DISTRO" != "$DEFAULT_DISTRO" ]]; then
    fail "native $DISTRO on Ubuntu $UBUNTU_VERSION is not a supported workspace path; use --method docker"
fi

if [[ "$METHOD" == "custom" && "$UBUNTU_VERSION" != "22.04" ]]; then
    fail "custom-interface Python 3.12 builds support Ubuntu 22.04 only"
fi

command -v git >/dev/null 2>&1 || fail "Git is required; install it before running this skill"

check_docker() {
    command -v docker >/dev/null 2>&1 || fail \
        "Docker CE is required for the '$METHOD' method. Install Docker CE, add your user to the docker group, re-login or run 'newgrp docker', then retry."

    if ! docker info >/dev/null 2>&1; then
        fail "Docker is installed but unavailable. Start the daemon and verify current-user access with 'docker info'. If permission is denied, add your user to the docker group, then re-login or run 'newgrp docker'."
    fi
}

MISSING_NATIVE_PACKAGES=()
NEED_ROSDEP_INIT=false

check_native_build_dependencies() {
    if ! dpkg-query -W -f='${Status}' build-essential 2>/dev/null | grep -q "install ok installed"; then
        MISSING_NATIVE_PACKAGES+=("build-essential")
    fi
    if ! command -v colcon >/dev/null 2>&1; then
        MISSING_NATIVE_PACKAGES+=("python3-colcon-common-extensions")
    fi
    if ! command -v rosdep >/dev/null 2>&1; then
        MISSING_NATIVE_PACKAGES+=("python3-rosdep")
    fi
    if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
        NEED_ROSDEP_INIT=true
    fi
}

ensure_sudo_for_native_dependencies() {
    if [[ ${#MISSING_NATIVE_PACKAGES[@]} -eq 0 && "$NEED_ROSDEP_INIT" == false ]]; then
        step "Native workspace build dependencies are already available; root elevation is not required"
        return
    fi

    command -v sudo >/dev/null 2>&1 || fail \
        "Native build setup requires root privileges, but sudo is not installed."

    # Reuse an existing credential cache or passwordless sudo without prompting.
    if sudo -n true 2>/dev/null; then
        return
    fi

    # A real terminal can safely display sudo's password prompt.
    if [[ -t 0 && -t 1 ]]; then
        sudo -v || fail "sudo authentication failed; install the missing prerequisites and retry"
        return
    fi

    {
        echo "ERROR: Native build prerequisites require sudo, but this session cannot prompt for a password."
        if [[ ${#MISSING_NATIVE_PACKAGES[@]} -gt 0 ]]; then
            echo "Missing packages: ${MISSING_NATIVE_PACKAGES[*]}"
            echo "Run in an interactive terminal:"
            echo "  sudo apt-get update"
            echo "  sudo apt-get install -y ${MISSING_NATIVE_PACKAGES[*]}"
        fi
        if [[ "$NEED_ROSDEP_INIT" == true ]]; then
            echo "rosdep also needs one-time initialization:"
            echo "  sudo rosdep init"
        fi
        echo "Then rerun this script. Alternatively, run 'sudo -v' in an interactive terminal before retrying."
    } >&2
    exit 1
}

install_pixi_if_needed() {
    if command -v pixi >/dev/null 2>&1; then
        return
    fi
    if [[ -x "${HOME}/.pixi/bin/pixi" ]]; then
        export PATH="${HOME}/.pixi/bin:${PATH}"
        command -v pixi >/dev/null 2>&1 && return
    fi

    command -v curl >/dev/null 2>&1 || fail "curl is required to install Pixi"
    command -v tar >/dev/null 2>&1 || fail "tar is required to install Pixi"
    command -v sha256sum >/dev/null 2>&1 || fail "sha256sum is required to verify the Pixi download"

    local arch target archive_url checksum_url workdir https_protocol
    arch="$(uname -m)"
    case "$arch" in
        x86_64 | amd64) target="x86_64-unknown-linux-musl" ;;
        aarch64 | arm64) target="aarch64-unknown-linux-musl" ;;
        *) fail "unsupported architecture for automated Pixi install: $arch (install Pixi from https://pixi.sh and retry)" ;;
    esac

    # Official GitHub release binary + sidecar checksum (not the install.sh bootstrap).
    archive_url="https://github.com/prefix-dev/pixi/releases/latest/download/pixi-${target}.tar.gz"
    checksum_url="${archive_url}.sha256"

    step "Downloading Pixi (${target}) from the official GitHub release"
    workdir="$(mktemp -d)"

    # Restrict redirects to HTTPS (SonarQube S5332) while keeping the
    # checksum-verified GitHub release path from this branch.
    https_protocol='=https'
    if ! curl --proto "$https_protocol" --proto-redir "$https_protocol" -fsSL -o "${workdir}/pixi.tar.gz" "$archive_url"; then
        rm -rf "$workdir"
        fail "failed to download Pixi archive from $archive_url"
    fi
    if ! curl --proto "$https_protocol" --proto-redir "$https_protocol" -fsSL -o "${workdir}/pixi.tar.gz.sha256" "$checksum_url"; then
        rm -rf "$workdir"
        fail "failed to download Pixi checksum from $checksum_url"
    fi

    # Checksum file names the release asset; verify against the local download name.
    if ! (
        cd "$workdir"
        awk '{print $1 "  pixi.tar.gz"}' pixi.tar.gz.sha256 > pixi.tar.gz.sha256.local
        sha256sum -c pixi.tar.gz.sha256.local
    ); then
        rm -rf "$workdir"
        fail "Pixi archive checksum verification failed; refusing to install"
    fi

    mkdir -p "${HOME}/.pixi/bin"
    if ! tar -xzf "${workdir}/pixi.tar.gz" -C "${HOME}/.pixi/bin" pixi; then
        rm -rf "$workdir"
        fail "failed to extract Pixi binary"
    fi
    rm -rf "$workdir"
    chmod +x "${HOME}/.pixi/bin/pixi"
    export PATH="${HOME}/.pixi/bin:${PATH}"
    command -v pixi >/dev/null 2>&1 ||
        fail "Pixi extraction completed but pixi is not on PATH; open a new shell and retry"
    step "Pixi installed to ${HOME}/.pixi/bin/pixi ($(pixi --version 2>/dev/null || echo ok))"
}

case "$METHOD" in
    native)
        [[ -f "$ROS_SETUP" ]] || fail \
            "ROS 2 $DISTRO was not found at $ROS_SETUP. This skill does not install ROS 2; use --method docker."
        if [[ "$SKIP_BUILD" == false ]]; then
            check_native_build_dependencies
            ensure_sudo_for_native_dependencies
        fi
        ;;
    docker|custom)
        check_docker
        ;;
    pixi)
        install_pixi_if_needed
        ;;
    *)
        fail "unsupported workspace method after validation: $METHOD"
        ;;
esac

confirm_clean_reclone() {
    local path="$1"
    printf 'WARNING: About to permanently delete the Git checkout at:\n  %s\n' "$path" >&2
    printf 'Local and untracked changes will be lost.\n' >&2
    if [[ -t 0 ]]; then
        local reply=""
        printf 'Type y to confirm deletion: ' >&2
        read -r reply
        [[ "$reply" == "y" || "$reply" == "Y" ]] ||
            fail "clean reclone cancelled (did not receive y)"
    else
        [[ "$YES_I_KNOW" == true ]] ||
            fail "non-interactive --clean-reclone requires --yes-i-know (path: $path)"
        step "Non-interactive confirmation accepted via --yes-i-know"
    fi
}

clone_or_validate_repo() {
    if [[ ! -e "$REPO_PATH" ]]; then
        step "Cloning branch '$BRANCH' from $REPO_URL"
        git clone --branch "$BRANCH" "$REPO_URL" "$REPO_PATH"
        return
    fi

    [[ -d "$REPO_PATH" ]] || fail "repository path exists but is not a directory: $REPO_PATH"
    git -C "$REPO_PATH" rev-parse --is-inside-work-tree >/dev/null 2>&1 ||
        fail "existing repository path is not a Git checkout: $REPO_PATH"

    if [[ "$CLEAN_RECLONE" == true ]]; then
        [[ ! -L "$REPO_PATH" ]] ||
            fail "refusing to clean a symbolic-link repository path: $REPO_PATH"

        local resolved_path resolved_home repo_root
        resolved_path="$(realpath "$REPO_PATH")"
        resolved_home="$(realpath "$HOME")"
        repo_root="$(realpath "$(git -C "$REPO_PATH" rev-parse --show-toplevel)")"
        [[ "$resolved_path" == "$repo_root" ]] ||
            fail "refusing to clean a path inside another Git checkout: $resolved_path (root: $repo_root)"
        [[ "$resolved_path" != "/" && "$resolved_path" != "$resolved_home" ]] ||
            fail "refusing to clean unsafe repository path: $resolved_path"

        confirm_clean_reclone "$resolved_path"
        step "Removing existing checkout at $resolved_path (--clean-reclone)"
        rm -rf -- "$resolved_path"
        step "Cloning clean branch '$BRANCH' from $REPO_URL"
        git clone --branch "$BRANCH" "$REPO_URL" "$resolved_path"
        REPO_PATH="$resolved_path"
        return
    fi

    local current_branch
    current_branch="$(git -C "$REPO_PATH" branch --show-current)"
    if [[ -n "$current_branch" && "$current_branch" != "$BRANCH" ]]; then
        fail "existing checkout is on branch '$current_branch', but '$BRANCH' was requested; reuse it with --branch '$current_branch', choose another --repo-path, or use --clean-reclone only after the user confirms local changes may be deleted"
    fi

    if [[ -n "$(git -C "$REPO_PATH" status --porcelain --untracked-files=normal)" ]]; then
        printf 'WARNING: Reusing an existing checkout with local or untracked changes: %s\n' "$REPO_PATH" >&2
    fi
    step "Using existing repository at $REPO_PATH"
}

clone_or_validate_repo

WS_DIR="${REPO_PATH}/${DISTRO}_ws"
[[ -d "$WS_DIR" ]] || fail "workspace directory not found: $WS_DIR"

step "Initializing repository submodules"
git -C "$REPO_PATH" submodule update --init --recursive

if [[ "$SKIP_BUILD" == true ]]; then
    printf '\nWorkspace prepared at %s (build skipped).\n' "$WS_DIR"
    exit 0
fi

case "$METHOD" in
    native)
        # Keep apt/rosdep/pip non-interactive for agent sessions and PEP 668 hosts.
        export DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
        export NEEDRESTART_MODE="${NEEDRESTART_MODE:-a}"
        export PIP_BREAK_SYSTEM_PACKAGES="${PIP_BREAK_SYSTEM_PACKAGES:-1}"

        if [[ ${#MISSING_NATIVE_PACKAGES[@]} -gt 0 ]]; then
            step "Installing missing native workspace build dependencies (not ROS 2)"
            sudo apt-get update
            sudo --preserve-env=DEBIAN_FRONTEND,NEEDRESTART_MODE \
                env DEBIAN_FRONTEND="$DEBIAN_FRONTEND" NEEDRESTART_MODE="$NEEDRESTART_MODE" \
                apt-get install -y "${MISSING_NATIVE_PACKAGES[@]}"
        else
            step "Skipping apt: native workspace build dependencies are already available"
        fi

        if [[ "$NEED_ROSDEP_INIT" == true ]]; then
            step "Initializing rosdep"
            sudo rosdep init
        fi
        rosdep update --rosdistro "$DISTRO"

        # ROS setup.bash reads optional unbound vars; enable nounset only after sourcing.
        set +u
        # shellcheck source=/dev/null
        source "$ROS_SETUP"
        set -u
        step "Resolving package dependencies for $DISTRO"
        env \
            DEBIAN_FRONTEND="$DEBIAN_FRONTEND" \
            NEEDRESTART_MODE="$NEEDRESTART_MODE" \
            PIP_BREAK_SYSTEM_PACKAGES="$PIP_BREAK_SYSTEM_PACKAGES" \
            rosdep install -i --from-paths "$WS_DIR/src" --rosdistro "$DISTRO" -y

        step "Building $WS_DIR with colcon"
        (
            cd "$WS_DIR"
            colcon build
        )

        printf '\nBuild complete. Source:\n  source %q\n  source %q\n' \
            "$ROS_SETUP" "$WS_DIR/install/local_setup.bash"
        ;;

    docker)
        if [[ "$ARCH" == "aarch64" ]]; then
            DOCKER_IMAGE="arm64v8/ros:${DISTRO}"
        else
            DOCKER_IMAGE="osrf/ros:${DISTRO}-desktop"
        fi

        if docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
            step "Using local image: $DOCKER_IMAGE"
        else
            step "Pulling architecture-appropriate image: $DOCKER_IMAGE"
            docker pull "$DOCKER_IMAGE"
        fi

        step "Building mounted workspace in the prebuilt ROS image"
        # Keep the container as root for apt/rosdep, then restore host ownership on the
        # bind-mounted workspace so build/install/log are not left root-owned.
        # Build-time networking stays on Docker defaults; DDS/runtime flags belong in the
        # container used to source and run the built workspace, not in colcon build.
        docker run --rm \
            --env="PIP_BREAK_SYSTEM_PACKAGES=1" \
            --env="HOST_UID=$(id -u)" \
            --env="HOST_GID=$(id -g)" \
            --volume="${WS_DIR}:/${DISTRO}_ws" \
            "$DOCKER_IMAGE" \
            /bin/bash -c "
                set -eo pipefail
                export DEBIAN_FRONTEND=noninteractive
                export NEEDRESTART_MODE=a
                export PIP_BREAK_SYSTEM_PACKAGES=1
                source /opt/ros/${DISTRO}/setup.bash
                set -u
                cd /${DISTRO}_ws
                trap 'chown -R \"\$HOST_UID:\$HOST_GID\" /${DISTRO}_ws' EXIT
                apt-get update
                rosdep update --rosdistro ${DISTRO}
                rosdep install --from-paths src --ignore-src --rosdistro ${DISTRO} -y
                colcon build
            "

        printf '\nContainer workspace build complete: %s\n' "$WS_DIR/install"
        printf 'Artifacts under %s are owned by the calling user.\n' "$WS_DIR"
        printf 'Source and run these artifacts inside a compatible %s ROS container.\n' "$DISTRO"
        printf 'For runtime DDS discovery, enable Docker host-network mode and set ROS_DOMAIN_ID on the runtime container—not during this build.\n'
        ;;

    custom)
        BUILD_ROS="${REPO_PATH}/build_ros.sh"
        [[ -f "$BUILD_ROS" ]] || fail "custom build entrypoint not found: $BUILD_ROS"

        step "Building Python 3.12 custom-interface workspaces with repository Dockerfiles"
        (
            cd "$REPO_PATH"
            bash ./build_ros.sh -d "$DISTRO" -v 22.04
        )

        printf '\nCustom-interface build complete. Source:\n'
        printf '  source %q\n' "$REPO_PATH/build_ws/$DISTRO/${DISTRO}_ws/install/local_setup.bash"
        printf '  source %q\n' "$REPO_PATH/build_ws/$DISTRO/isaac_sim_ros_ws/install/local_setup.bash"
        ;;

    pixi)
        [[ -f "$WS_DIR/pixi.toml" ]] || fail \
            "Linux Pixi was requested, but $WS_DIR/pixi.toml does not exist in this repository revision"

        step "Installing Pixi dependencies in $WS_DIR"
        (
            cd "$WS_DIR"
            pixi install
            pixi run build
        )
        printf '\nPixi build complete. Use commands through pixi run or pixi shell in %s.\n' "$WS_DIR"
        ;;
    *)
        fail "unsupported workspace method after validation: $METHOD"
        ;;
esac
