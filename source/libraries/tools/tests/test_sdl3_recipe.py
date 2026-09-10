# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test the local SDL3 source recipe."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

RECIPE_PATH = Path(__file__).resolve().parents[4] / "deps" / "recipes" / "sdl3" / "build.py"
RECIPE_SPEC = importlib.util.spec_from_file_location("isaacsim_sdl3_recipe", RECIPE_PATH)
if RECIPE_SPEC is None or RECIPE_SPEC.loader is None:
    raise RuntimeError(f"Cannot load SDL3 recipe from {RECIPE_PATH}")
sdl3_recipe = importlib.util.module_from_spec(RECIPE_SPEC)
sys.modules[RECIPE_SPEC.name] = sdl3_recipe
previous_bytecode_setting = sys.dont_write_bytecode
try:
    sys.dont_write_bytecode = True
    RECIPE_SPEC.loader.exec_module(sdl3_recipe)
finally:
    sys.dont_write_bytecode = previous_bytecode_setting


class SDL3RecipeTests(unittest.TestCase):
    """Test platform-specific SDL3 build setup."""

    def test_windows_activates_msvc_before_copying_environment(self) -> None:
        """Preserve the compiler and SDK values added by Packman toolchain activation."""
        repository_root = Path("repository")
        compiler = Path("toolchain") / "bin" / "cl.exe"

        def activate_toolchain(root: Path) -> Path:
            self.assertEqual(root, repository_root)
            sdl3_recipe.os.environ["INCLUDE"] = "msvc-include"
            sdl3_recipe.os.environ["LIB"] = "msvc-lib"
            return compiler

        with (
            mock.patch.dict(sdl3_recipe.os.environ, {}, clear=True),
            mock.patch.object(
                sdl3_recipe,
                "_activate_windows_msvc_environment",
                side_effect=activate_toolchain,
            ) as activate,
        ):
            environment = sdl3_recipe._build_environment("nt", repository_root)

        activate.assert_called_once_with(repository_root)
        self.assertEqual(environment["CC"], str(compiler))
        self.assertEqual(environment["CXX"], str(compiler))
        self.assertEqual(environment["INCLUDE"], "msvc-include")
        self.assertEqual(environment["LIB"], "msvc-lib")

    def test_linux_keeps_ninja_and_compiler_environment(self) -> None:
        """Preserve the existing Linux generator and compiler activation."""
        with (
            mock.patch.dict(sdl3_recipe.os.environ, {"CC": "gcc", "CXX": "g++"}, clear=False),
            mock.patch.object(sdl3_recipe.shutil, "which") as find_compiler,
        ):
            environment = sdl3_recipe._build_environment("posix", Path("repository"))
        find_compiler.assert_not_called()
        self.assertEqual(environment["CC"], "gcc")
        self.assertEqual(environment["CXX"], "g++")

    def test_main_configures_with_ninja(self) -> None:
        """Keep SDL on the generator supported by both packaged and local MSVC environments."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            environment = {
                "SRC_DIR": str(temporary_root / "source"),
                "PREFIX": str(temporary_root / "prefix"),
                "BUILD_DIR": str(temporary_root / "build"),
            }
            with (
                mock.patch.dict(sdl3_recipe.os.environ, environment, clear=True),
                mock.patch.object(sdl3_recipe.shutil, "which", return_value="cmake"),
                mock.patch.object(sdl3_recipe, "_build_environment", return_value=environment),
                mock.patch.object(sdl3_recipe.subprocess, "run") as run,
            ):
                sdl3_recipe.main()

        configure_command = run.call_args_list[0].args[0]
        generator_index = configure_command.index("-G")
        self.assertEqual(configure_command[generator_index + 1], "Ninja")
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", configure_command)
        self.assertNotIn("Visual Studio 17 2022", configure_command)


if __name__ == "__main__":
    unittest.main()
