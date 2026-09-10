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

"""Public facade allow-list for the policy examples extension."""

import hashlib
import importlib
from unittest.mock import Mock, patch

import isaacsim.robot.policy.examples as policy_examples
import omni.kit.test
from isaacsim.robot.policy.examples import model as policy_model
from isaacsim.robot.policy.examples.spec import PolicyArtifact


class TestPublicAPI(omni.kit.test.AsyncTestCase):
    """TestPublicAPI."""

    def test_root_exports_deployment_facade_and_migration_shims(self) -> None:
        """Verify root exports deployment facade and migration shims."""
        self.assertEqual(
            policy_examples.__all__,
            [
                "AnymalFlatTerrainPolicy",
                "FrankaOpenDrawerPolicy",
                "Go2FlatTerrainPolicy",
                "H1FlatTerrainPolicy",
                "IsaacLabPolicyController",
                "PolicyArtifact",
                "PolicyEnvConfig",
                "RobotPolicyRunner",
                "PolicySpec",
                "SpotFlatTerrainPolicy",
                "get_anymal_spec",
                "get_cartpole_spec",
                "get_franka_spec",
                "get_go2_spec",
                "get_h1_spec",
                "get_spot_spec",
                "make_franka_task_state_provider",
            ],
        )

    def test_torchscript_artifact_requires_trusted_sha256(self) -> None:
        """Verify TorchScript artifacts fail closed without a full SHA-256 digest."""
        with self.assertRaisesRegex(ValueError, "requires model_sha256 from a trusted source"):
            PolicyArtifact.from_files("policy.pt", "env.yaml")

        with self.assertRaisesRegex(ValueError, "full 64-character hexadecimal"):
            PolicyArtifact.from_files("policy.pt", "env.yaml", model_sha256="not-a-digest")

        artifact = PolicyArtifact.from_files("policy.pt", "env.yaml", model_sha256="A" * 64)
        self.assertEqual(artifact.model_sha256, "a" * 64)

    def test_torchscript_model_verifies_bytes_before_deserialization(self) -> None:
        """Verify modified model bytes never reach ``torch.jit.load``."""
        trusted_bytes = b"trusted TorchScript archive"
        trusted_sha256 = hashlib.sha256(trusted_bytes).hexdigest()
        import_torch = Mock()

        with (
            patch.object(policy_model, "read_artifact_bytes", return_value=b"tampered archive"),
            patch.object(policy_model, "_import_torch", import_torch),
            self.assertRaisesRegex(RuntimeError, "failed SHA-256 verification"),
        ):
            policy_model.TorchScriptPolicyModel("policy.pt", trusted_sha256)

        import_torch.assert_not_called()

    def test_torchscript_model_loads_verified_bytes(self) -> None:
        """Verify matching model bytes are passed to ``torch.jit.load``."""
        trusted_bytes = b"trusted TorchScript archive"
        trusted_sha256 = hashlib.sha256(trusted_bytes).hexdigest()
        loaded_policy = Mock()
        loaded_policy.to.return_value = loaded_policy
        torch = Mock()
        torch.device.return_value = "cpu"

        def load_model(stream: object) -> Mock:
            self.assertEqual(stream.read(), trusted_bytes)
            return loaded_policy

        torch.jit.load.side_effect = load_model
        with (
            patch.object(policy_model, "read_artifact_bytes", return_value=trusted_bytes),
            patch.object(policy_model, "_import_torch", return_value=torch),
        ):
            loaded = policy_model.TorchScriptPolicyModel("policy.pt", trusted_sha256)

        self.assertIs(loaded._policy, loaded_policy)
        torch.jit.load.assert_called_once()

    def test_removed_policy_classes_explain_the_migration(self) -> None:
        """Verify removed policy classes explain the migration."""
        robot_modules = {
            "AnymalFlatTerrainPolicy": "anymal",
            "CartpolePolicy": "cartpole",
            "FrankaOpenDrawerPolicy": "franka",
            "Go2FlatTerrainPolicy": "go2",
            "H1FlatTerrainPolicy": "h1",
            "SpotFlatTerrainPolicy": "spot",
        }
        expected_replacements = {
            "PolicyController": ("RobotPolicyRunner with PolicySpec and PolicyArtifact", "PolicyArtifact.from_files"),
            "AnymalFlatTerrainPolicy": ("RobotPolicyRunner with get_anymal_spec()", "runner.step(dt, command)"),
            "CartpolePolicy": ("RobotPolicyRunner with get_cartpole_spec()", "runner.step(dt) without a command"),
            "FrankaOpenDrawerPolicy": (
                "RobotPolicyRunner with get_franka_spec()",
                "make_franka_task_state_provider(cabinet, env_config)",
                "runner.step(dt) without a command",
            ),
            "Go2FlatTerrainPolicy": ("RobotPolicyRunner with get_go2_spec()", "runner.step(dt, command)"),
            "H1FlatTerrainPolicy": ("RobotPolicyRunner with get_h1_spec()", "runner.step(dt, command)"),
            "SpotFlatTerrainPolicy": ("RobotPolicyRunner with get_spot_spec()", "runner.step(dt, command)"),
        }
        legacy_imports = [
            ("isaacsim.robot.policy.examples.controllers", "PolicyController"),
            ("isaacsim.robot.policy.examples.controllers.policy_controller", "PolicyController"),
        ]
        self.assertEqual(list(policy_examples.robots.__all__), list(robot_modules))
        self.assertEqual(
            importlib.import_module("isaacsim.robot.policy.examples.controllers").__all__, ["PolicyController"]
        )
        for class_name, module_name in robot_modules.items():
            legacy_imports.extend(
                [
                    ("isaacsim.robot.policy.examples.robots", class_name),
                    (f"isaacsim.robot.policy.examples.robots.{module_name}", class_name),
                ]
            )
            if class_name != "CartpolePolicy":
                legacy_imports.append(("isaacsim.robot.policy.examples", class_name))

        for module_name, class_name in legacy_imports:
            legacy_class = getattr(importlib.import_module(module_name), class_name)
            with self.subTest(import_path=f"{module_name}.{class_name}"):
                with self.assertRaises(RuntimeError) as context:
                    legacy_class()
                message = str(context.exception)
                self.assertIn(
                    f"{class_name} was removed in isaacsim.robot.policy.examples 7.0.0",
                    message,
                )
                self.assertIn("runner.spawn()", message)
                self.assertIn("runner.initialize()", message)
                self.assertIn("runner.articulation", message)
                self.assertIn("Full migration guide:", message)
                for guidance in expected_replacements[class_name]:
                    self.assertIn(guidance, message)

        policy_controller = importlib.import_module("isaacsim.robot.policy.examples.controllers").PolicyController
        for class_name in robot_modules:
            legacy_class = getattr(policy_examples.robots, class_name)
            self.assertTrue(
                issubclass(legacy_class, policy_controller),
                f"{class_name} no longer preserves its develop-era PolicyController base class",
            )
