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

"""The policy authoring surface: artifact file groupings and per-robot deployment specs."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import yaml

__all__ = ["PolicyArtifact", "PolicySpec", "read_artifact_bytes"]


# Every path may be local or an ``omni.client`` URL; these helpers bridge the two worlds.


def _is_url(path: str) -> bool:
    return "://" in path


def read_artifact_bytes(path: str) -> bytes:
    """Read a policy artifact file from a local path or an ``omni.client`` URL.

    Args:
        path: Artifact path, local or an Omniverse URL.

    Returns:
        The resulting bytes.
    """
    if _is_url(path):
        import omni.client

        result, _, content = omni.client.read_file(path)
        if result != omni.client.Result.OK:
            raise FileNotFoundError(f"Failed to read policy artifact {path!r}: {result}.")
        return memoryview(content).tobytes()
    return Path(path).read_bytes()


def _entry_exists(path: str) -> bool:
    if _is_url(path):
        import omni.client

        result, _ = omni.client.stat(path)
        return result == omni.client.Result.OK
    return Path(path).is_file()


def _join(directory: str, relative_path: str) -> str:
    if _is_url(directory):
        return f"{directory.rstrip('/')}/{relative_path}"
    return str(Path(directory) / relative_path)


#: Model file name per supported backend, and the suffix each backend is inferred from.
_MODEL_FILE_BY_BACKEND = {"onnx": "policy.onnx", "torchscript": "policy.pt"}
_BACKEND_BY_SUFFIX = {".onnx": "onnx", ".pt": "torchscript"}


@dataclass(frozen=True)
class PolicyArtifact:
    """Frozen grouping of the files that define one deployable policy.

    Nothing is read at construction; the runner loads the files when it binds.

    Args:
        model_path: Policy model checkpoint (``.pt`` TorchScript or ``.onnx``).
        env_config_path: Exported Isaac Lab env config (deployment settings: timing, spawn,
            gains, actuators).
        backend: Inference runtime, ``torchscript`` or ``onnx``.
        descriptor_path: Exported ``IO_descriptors.yaml``, the policy-interface source. None
            only for policies bound through an explicit spec ``binding`` hook (franka).
        model_sha256: Full SHA-256 digest obtained from a trusted source. TorchScript artifacts
            require this digest because their deserializer can execute code.

    Raises:
        ValueError: If a TorchScript artifact has no valid SHA-256 digest.
    """

    model_path: str
    env_config_path: str
    backend: str
    descriptor_path: str | None = None
    model_sha256: str | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the model digest used at deserialization time."""
        if self.model_sha256 is None:
            if self.backend == "torchscript":
                raise ValueError(f"TorchScript policy {self.model_path!r} requires model_sha256 from a trusted source.")
            return

        digest = self.model_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("model_sha256 must be a full 64-character hexadecimal SHA-256 digest.")
        object.__setattr__(self, "model_sha256", digest)

    def load_descriptor(self) -> Mapping:
        """Load and parse the artifact's exported IO descriptor.

        Returns:
            The resulting mapping.
        """
        if self.descriptor_path is None:
            raise ValueError(
                f"PolicyArtifact for {self.model_path!r} declares no descriptor_path; the IO descriptor "
                "(IO_descriptors.yaml) is required to derive the policy interface."
            )
        return yaml.safe_load(io.BytesIO(read_artifact_bytes(self.descriptor_path)))

    @classmethod
    def from_files(
        cls,
        model_path: str,
        env_config_path: str,
        descriptor_path: str | None = None,
        *,
        backend: str | None = None,
        model_sha256: str | None = None,
    ) -> PolicyArtifact:
        """Create an artifact from explicit paths.

        The backend is inferred from the model suffix (``.pt``/``.onnx``) unless given.

        Args:
            model_path: Path to the policy model file.
            env_config_path: Path to the artifact's ``env.yaml``.
            descriptor_path: Path to the artifact's IO descriptor, or None.
            backend: Model backend name, or None to infer from the suffix.
            model_sha256: Full model digest obtained from a trusted source. Required for TorchScript.

        Returns:
            The resulting :class:`PolicyArtifact`.
        """
        if backend is None:
            suffix = Path(model_path.split("://", 1)[-1]).suffix.lower()
            backend = _BACKEND_BY_SUFFIX.get(suffix)
            if backend is None:
                raise ValueError(
                    f"PolicyArtifact.from_files: cannot infer a backend from model path {model_path!r}; "
                    f"expected one of {sorted(_BACKEND_BY_SUFFIX)} or an explicit backend."
                )
        elif backend not in _MODEL_FILE_BY_BACKEND:
            raise ValueError(
                f"PolicyArtifact.from_files: unsupported backend {backend!r}; "
                f"expected one of {sorted(_MODEL_FILE_BY_BACKEND)}."
            )
        return cls(model_path, env_config_path, backend, descriptor_path, model_sha256)

    @classmethod
    def from_bundle(
        cls,
        bundle_dir: str,
        *,
        backend: str | None = None,
        model_sha256: str | None = None,
    ) -> PolicyArtifact:
        """Create an artifact from a flat bundle directory.

        The directory holds ``policy.onnx``/``policy.pt`` + ``env.yaml`` + ``IO_descriptors.yaml``.

        ``backend`` selects which model file to load; by default whichever exists is used,
        preferring ONNX when the bundle ships both.

        Args:
            bundle_dir: Flat bundle directory.
            backend: Model backend name, or None to infer from the suffix.
            model_sha256: Full model digest obtained from a trusted source. Required when loading TorchScript.

        Returns:
            The resulting :class:`PolicyArtifact`.
        """
        if backend is not None and backend not in _MODEL_FILE_BY_BACKEND:
            raise ValueError(
                f"PolicyArtifact.from_bundle: unsupported backend {backend!r}; "
                f"expected one of {sorted(_MODEL_FILE_BY_BACKEND)}."
            )
        model_files = _MODEL_FILE_BY_BACKEND
        model_path = None
        for candidate in [backend] if backend is not None else list(model_files):
            path = _join(bundle_dir, model_files[candidate])
            if _entry_exists(path):
                model_path = path
                break

        if model_path is None:
            wanted = [model_files[c] for c in ([backend] if backend is not None else list(model_files))]
            raise FileNotFoundError(f"Bundle {bundle_dir!r} contains no model file named one of {wanted}.")

        return cls.from_files(
            model_path,
            _require(bundle_dir, "env.yaml", "the env config"),
            _require(bundle_dir, "IO_descriptors.yaml", "the IO descriptor"),
            model_sha256=model_sha256,
        )

    @classmethod
    def from_training_run(cls, run_dir: str, *, model_sha256: str) -> PolicyArtifact:
        """Create an artifact from an Isaac Lab training-run directory.

        Resolves ``exported/policy.pt`` (the deployable TorchScript export — raw ``model_*.pt``
        training checkpoints are not loadable and are never selected), ``params/env.yaml``, and
        ``io_descriptors/IO_descriptors.yaml``; each missing file raises with the path and,
        for the descriptor, the ``export_io_descriptors`` re-export instruction.

        Args:
            run_dir: Isaac Lab training-run directory.
            model_sha256: Full model digest obtained from a trusted source.

        Returns:
            The resulting :class:`PolicyArtifact`.
        """
        model_path = _join(run_dir, "exported/policy.pt")
        if not _entry_exists(model_path):
            raise FileNotFoundError(
                f"Training run {run_dir!r} has no deployable model at 'exported/policy.pt'; export the "
                "trained policy first (raw model_<iter>.pt checkpoints are not loadable)."
            )

        descriptor_path = _join(run_dir, "io_descriptors/IO_descriptors.yaml")
        if not _entry_exists(descriptor_path):
            raise FileNotFoundError(
                f"Training run {run_dir!r} is missing the IO descriptor at {descriptor_path!r}; re-export the "
                "policy with export_io_descriptors enabled (the descriptor is the policy interface source)."
            )

        return cls.from_files(
            model_path,
            _require(run_dir, "params/env.yaml", "the env config"),
            descriptor_path,
            backend="torchscript",
            model_sha256=model_sha256,
        )


def _require(directory: str, relative_path: str, label: str) -> str:
    """Join and existence-check one required artifact file.

    Args:
        directory: Directory the artifact is expected in.
        relative_path: Path relative to the package root.
        label: Artifact description used in error messages.

    Returns:
        The resolved string.
    """
    path = _join(directory, relative_path)
    if not _entry_exists(path):
        raise FileNotFoundError(f"{directory!r} is missing {label} at {path!r}.")
    return path


@dataclass(frozen=True)
class PolicySpec:
    """Frozen description of one deployable policy: which artifacts to load and how to spawn.

    Authored once per robot/task and handed to :class:`~.runtime.RobotPolicyRunner`. The policy
    interface itself (observation/action terms, command width) is not part of the spec — it is
    derived from the selected artifact's exported IO descriptor at deploy time. Bundled getters
    accept any of these fields as keyword overrides; calls without overrides return process-cached
    default instances.

    Args:
        engines: Engine name (``physx``/``newton``) to the :class:`PolicyArtifact` deployed on it.
        usd_path: Robot USD to spawn: one path for all engines, an engine-keyed mapping covering
            every declared engine, or None to spawn the env config's ``spawn.usd_path``.
        name: Optional diagnostic label used in runner error messages.
        default_spawn_position: Spawn translation used when the caller passes none, or None to
            fall back to the env config's initial root state.
        default_spawn_orientation: Same for the WXYZ spawn orientation. The bundled specs pin
            identity here because the hosted env configs mix quaternion conventions.
        zero_targets_on_initialize: Reset to the default state and zero all targets when the
            runtime is first built (cartpole's effort-mode start semantics). Replay callers own
            any later articulation reset/teleport before ``RobotPolicyRunner.initialize()``.
        binding: Escape hatch for policies whose interface is not derivable from the descriptor
            (multi-articulation or unexported terms, e.g. franka); called as
            ``binding(env_config)`` and must return a :class:`PolicyBinding`.
    """

    engines: Mapping[str, PolicyArtifact]
    usd_path: str | Mapping[str, str] | None = None
    name: str | None = None
    default_spawn_position: tuple[float, float, float] | None = None
    default_spawn_orientation: tuple[float, float, float, float] | None = None
    zero_targets_on_initialize: bool = False
    binding: Callable | None = None

    def __post_init__(self) -> None:
        """Defensively copy and freeze mappings held by process-cached bundled specs."""
        object.__setattr__(self, "engines", MappingProxyType(dict(self.engines)))
        if isinstance(self.usd_path, Mapping):
            object.__setattr__(self, "usd_path", MappingProxyType(dict(self.usd_path)))
