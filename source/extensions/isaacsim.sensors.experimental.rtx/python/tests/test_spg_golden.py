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

"""Golden-image and correctness tests for the SPG walkthrough variants.

Renders the vendored Cornell box scene through an SPG authored by ``author_spg`` and
verifies the custom output AOVs two ways:

1. **Derived correctness** (GPU-robust): the grayscale AOV equals the BT.601 luminance
   of the source ``LdrColor`` AOV, and the inverted AOV equals ``255 - grayscale``.
   Both AOVs are read from the same frame, so this holds regardless of the renderer.
2. **Golden image** (regression): the output AOV is compared against a committed
   reference PNG within tolerances.

Custom SPG AOVs are read as NumPy arrays via
``omni.replicator.core.AnnotatorRegistry.register_annotator_from_aov`` (no viewport
required). Set ``ISAAC_SPG_UPDATE_GOLDEN=1`` to (re)generate the golden PNGs.
"""

import os

import carb
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
import numpy as np
import omni.kit.test
import omni.replicator.core as rep
from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera, SPGNode
from isaacsim.test.utils.image_comparison import compare_arrays_within_tolerances
from PIL import Image

# Render resolution (width, height) for the golden captures.
_RESOLUTION = (640, 360)
# Frames to settle the renderer before reading AOVs.
_SETTLE_FRAMES = 30
# Golden comparison tolerances. Loose because raytraced output varies by GPU/driver;
# the derived-correctness checks are the strict, environment-independent gate.
_GOLDEN_MEAN_TOL = 8.0
_GOLDEN_PERCENTILE_TOL = (99.0, 60.0)


def _spg_data_dir() -> str:
    """Return the absolute path to the vendored ``data/spg`` directory."""
    return os.path.join(app_utils.get_extension_path("isaacsim.sensors.experimental.rtx"), "data", "spg")


class _SPGGoldenBase(omni.kit.test.AsyncTestCase):
    """Shared setup and helpers for SPG golden/correctness tests."""

    async def setUp(self) -> None:
        """Enable the SPG runtime and open the vendored Cornell box scene."""
        super().setUp()
        app_utils.enable_extension("omni.rtx.spg")
        carb.settings.get_settings().set("/rtx/spg/enabled", True)

        self._data_dir = _spg_data_dir()
        self._kernels = os.path.join(self._data_dir, "kernels")
        self._golden_dir = os.path.join(self._data_dir, "golden")

        stage_utils.open_stage(usd_path=os.path.join(self._data_dir, "scene", "cornell_box.usda"))
        while stage_utils.is_stage_loading():
            await app_utils.update_app_async()

        # The vendored scene's camera already has the OmniSensorAPI schema.
        self._cam = RtxCamera("/World/Camera")

    async def tearDown(self) -> None:
        """Stop the orchestrator and close the stage so state does not leak to other tests."""
        await rep.orchestrator.stop_async()
        stage_utils.close_stage()
        super().tearDown()

    async def _author_and_capture(self, nodes, connections, aov_names) -> dict:
        """Author an SPG onto a ``CameraSensor`` render product and read back the AOVs.

        Args:
            nodes: SPG nodes to author.
            connections: ``(src, dst)`` connection endpoints.
            aov_names: AOV names to capture as ``(height, width, 4)`` uint8 arrays.

        Returns:
            Mapping from AOV name to its captured NumPy array.
        """
        # A CameraSensor creates and drives the render product. Its resolution is
        # (height, width), the transpose of the (width, height) golden resolution.
        sensor = CameraSensor(self._cam, resolution=(_RESOLUTION[1], _RESOLUTION[0]))
        render_product_path = str(sensor.render_product.GetPath())
        annotators = {}
        try:
            self._cam.author_spg(nodes, connections=connections, render_product=render_product_path)

            registered = set(rep.AnnotatorRegistry.get_registered_annotators())
            for aov in aov_names:
                reg_name = f"spgtest_{aov}"
                # Register once per Kit process; re-registration would churn global registry state.
                if reg_name not in registered:
                    rep.AnnotatorRegistry.register_annotator_from_aov(
                        aov=aov, output_data_type=np.uint8, output_channels=4, is_gpu_enabled=True, name=reg_name
                    )
                annot = rep.AnnotatorRegistry.get_annotator(reg_name, device="cpu", do_array_copy=True)
                annot.attach([render_product_path])
                annotators[aov] = annot

            for _ in range(_SETTLE_FRAMES):
                await rep.orchestrator.step_async()

            data = {aov: np.asarray(annot.get_data()) for aov, annot in annotators.items()}
        finally:
            for annot in annotators.values():
                annot.detach([render_product_path])
            del sensor
            await app_utils.update_app_async(steps=3)

        for aov, array in data.items():
            self.assertEqual(array.shape, (_RESOLUTION[1], _RESOLUTION[0], 4), f"unexpected shape for AOV '{aov}'")
            self.assertGreater(int(array[..., :3].max()), 0, f"AOV '{aov}' is all black")
        return data

    def _check_golden(self, name: str, array: np.ndarray) -> None:
        """Compare an RGB AOV against its committed golden PNG (or regenerate it).

        Args:
            name: Golden basename (without extension).
            array: Captured AOV array of shape ``(H, W, 4)``.
        """
        rgb = array[..., :3]
        golden_path = os.path.join(self._golden_dir, f"{name}.png")

        if os.environ.get("ISAAC_SPG_UPDATE_GOLDEN") == "1":
            os.makedirs(self._golden_dir, exist_ok=True)
            Image.fromarray(rgb).save(golden_path)
            carb.log_warn(f"SPG golden regenerated: {golden_path}")
            return

        self.assertTrue(
            os.path.exists(golden_path),
            f"Golden image missing: {golden_path}. Regenerate with ISAAC_SPG_UPDATE_GOLDEN=1.",
        )
        golden = np.asarray(Image.open(golden_path).convert("RGB"))
        result = compare_arrays_within_tolerances(
            golden_array=golden,
            test_array=rgb,
            allclose_rtol=None,
            allclose_atol=None,
            mean_tolerance=_GOLDEN_MEAN_TOL,
            percentile_tolerance=_GOLDEN_PERCENTILE_TOL,
            print_all_stats=True,
        )
        self.assertTrue(result["passed"], f"Golden comparison failed for '{name}': {result['criteria']}")


class TestSPGGrayscale(_SPGGoldenBase):
    """Grayscale walkthrough: one shader converting LdrColor to a grayscale AOV."""

    async def test_grayscale(self) -> None:
        """Render the grayscale SPG and verify the AOV against luminance and the golden."""
        data = await self._author_and_capture(
            SPGNode(
                "GrayscaleKernel",
                os.path.join(self._kernels, "GrayscaleKernel.cu"),
                sub_identifier="grayscale",
                inputs=["LdrColor"],
                outputs=["LdrGrayscale"],
            ),
            connections=[
                ("LdrColor", "GrayscaleKernel.inputs:LdrColor"),
                ("GrayscaleKernel.outputs:LdrGrayscale", "LdrGrayscale"),
            ],
            aov_names=["LdrColor", "LdrGrayscale"],
        )
        color = data["LdrColor"].astype(np.float32)
        gray = data["LdrGrayscale"]

        # All pixels are neutral gray (R == G == B).
        self.assertTrue(np.all(gray[..., 0] == gray[..., 1]) and np.all(gray[..., 1] == gray[..., 2]))
        # The gray value matches BT.601 luminance of the source color within rounding.
        expected = 0.299 * color[..., 0] + 0.587 * color[..., 1] + 0.114 * color[..., 2]
        max_abs = float(np.max(np.abs(expected - gray[..., 0].astype(np.float32))))
        self.assertLessEqual(max_abs, 2.0, f"grayscale luminance mismatch: max|diff|={max_abs}")

        self._check_golden("grayscale", gray)


class TestSPGGrayscaleInvert(_SPGGoldenBase):
    """Grayscale + invert walkthrough: chained shaders producing an inverted AOV."""

    async def test_grayscale_invert(self) -> None:
        """Render the chained SPG and verify the inverted AOV against 255-gray and the golden."""
        data = await self._author_and_capture(
            [
                SPGNode(
                    "GrayscaleKernel",
                    os.path.join(self._kernels, "GrayscaleKernel.cu"),
                    sub_identifier="grayscale",
                    inputs=["LdrColor"],
                    outputs=["LdrGrayscale"],
                ),
                SPGNode(
                    "InvertKernel",
                    os.path.join(self._kernels, "InvertKernel.cu"),
                    sub_identifier="invert",
                    inputs=["Image"],
                    outputs=["Inverted"],
                    params={"strength": 1.0},
                ),
            ],
            connections=[
                ("LdrColor", "GrayscaleKernel.inputs:LdrColor"),
                ("GrayscaleKernel.outputs:LdrGrayscale", "InvertKernel.inputs:Image"),
                ("InvertKernel.outputs:Inverted", "LdrInverted"),
            ],
            aov_names=["LdrColor", "LdrInverted"],
        )
        color = data["LdrColor"].astype(np.float32)
        inverted = data["LdrInverted"]

        # The inverted AOV is neutral gray (the invert of a grayscale image).
        self.assertTrue(np.all(inverted[..., 0] == inverted[..., 1]) and np.all(inverted[..., 1] == inverted[..., 2]))
        # Composing both kernels at strength 1.0 gives 255 - BT.601 luminance(LdrColor).
        expected = 255.0 - (0.299 * color[..., 0] + 0.587 * color[..., 1] + 0.114 * color[..., 2])
        max_abs = float(np.max(np.abs(expected - inverted[..., 0].astype(np.float32))))
        self.assertLessEqual(max_abs, 2.0, f"invert composition mismatch: max|diff|={max_abs}")

        self._check_golden("grayscale_invert", inverted)
