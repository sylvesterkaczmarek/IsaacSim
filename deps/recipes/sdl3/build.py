# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build and install the pinned shared SDL library into a Pixi prefix."""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _activate_windows_msvc_environment(repository_root: Path) -> Path:
    """Reuse the standalone library toolchain activation before configuring SDL."""
    build_tool_path = repository_root / "source" / "libraries" / "tools" / "build.py"
    specification = importlib.util.spec_from_file_location("isaacsim_source_library_build", build_tool_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load the source-library build tool from {build_tool_path}")
    build_tool = importlib.util.module_from_spec(specification)
    previous_module = sys.modules.get(specification.name)
    sys.modules[specification.name] = build_tool
    try:
        specification.loader.exec_module(build_tool)
        compiler = build_tool.ensure_windows_msvc_environment()
    finally:
        if previous_module is None:
            sys.modules.pop(specification.name, None)
        else:
            sys.modules[specification.name] = previous_module
    if compiler is None:
        raise RuntimeError("The SDL recipe requires an x64 MSVC compiler on Windows")
    return compiler


def _build_environment(platform_name: str, repository_root: Path) -> dict[str, str]:
    """Create the compiler environment used to configure and build SDL."""
    if platform_name == "nt":
        compiler = _activate_windows_msvc_environment(repository_root)
        print(f"Using Windows MSVC compiler: {compiler}")
    environment = os.environ.copy()
    if platform_name == "nt":
        environment["CC"] = str(compiler)
        environment["CXX"] = str(compiler)
    return environment


def main() -> None:
    """Configure, build, and install the SDL SDK without optional runtime dependencies."""
    source_directory = Path(os.environ["SRC_DIR"])
    prefix = Path(os.environ["PREFIX"])
    repository_root = Path(__file__).resolve().parents[3]
    install_prefix = prefix / "Library" if os.name == "nt" else prefix
    build_directory = Path(os.environ.get("BUILD_DIR", source_directory / "build")) / "sdl3-build"
    cmake = shutil.which("cmake")
    if cmake is None:
        raise RuntimeError("The SDL recipe requires CMake")
    if build_directory.exists():
        shutil.rmtree(build_directory)

    configure_command = [
        cmake,
        "-S",
        str(source_directory),
        "-B",
        str(build_directory),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DCMAKE_INSTALL_PREFIX={install_prefix}",
        f"-DCMAKE_INCLUDE_PATH={prefix / 'include'}",
        f"-DCMAKE_LIBRARY_PATH={prefix / 'lib'}",
        "-DSDL_SHARED=ON",
        "-DSDL_STATIC=OFF",
        "-DSDL_INSTALL=ON",
        "-DSDL_TEST_LIBRARY=OFF",
        "-DSDL_TESTS=OFF",
        "-DSDL_EXAMPLES=OFF",
        "-DSDL_INSTALL_DOCS=OFF",
        "-DSDL_INSTALL_CPACK=OFF",
        "-DSDL_UNINSTALL=OFF",
        "-DSDL_DEPS_SHARED=ON",
        "-DSDL_RPATH=OFF",
        "-DSDL_AUDIO=OFF",
        "-DSDL_HIDAPI=OFF",
    ]
    if os.name != "nt":
        configure_command.extend(
            [
                "-DSDL_X11=ON",
                "-DSDL_X11_SHARED=ON",
                "-DSDL_WAYLAND=OFF",
                "-DSDL_KMSDRM=OFF",
            ]
        )

    environment = _build_environment(os.name, repository_root)
    subprocess.run(configure_command, env=environment, check=True)
    subprocess.run(
        [cmake, "--build", str(build_directory), "--config", "Release", "--parallel"],
        env=environment,
        check=True,
    )
    subprocess.run([cmake, "--install", str(build_directory), "--config", "Release"], env=environment, check=True)


if __name__ == "__main__":
    main()
