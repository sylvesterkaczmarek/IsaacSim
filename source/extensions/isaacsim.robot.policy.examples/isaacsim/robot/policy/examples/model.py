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

"""Policy model runtimes exposing one flat float32 inference contract per backend."""

from __future__ import annotations

import ctypes
import hashlib
import hmac
import io
import sys
import tempfile
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import carb
import numpy as np

from .spec import read_artifact_bytes

if TYPE_CHECKING:
    from .binding import BoundPolicy
    from .spec import PolicyArtifact

__all__ = ["OnnxPolicyModel", "PolicyModel", "TorchScriptPolicyModel"]

_LINUX_CUDA_LIBRARY_PATHS = (
    "cublas/lib/libcublasLt.so.12",
    "cublas/lib/libcublas.so.12",
    "cuda_nvrtc/lib/libnvrtc.so.12",
    "curand/lib/libcurand.so.10",
    "cufft/lib/libcufft.so.11",
    "cuda_runtime/lib/libcudart.so.12",
    "cudnn/lib/libcudnn.so.9",
)
_CUDA_LIBRARY_HANDLES: list[Any] = []


class PolicyModel(Protocol):
    """Runtime interface for one deployable policy model.

    One flat float32 observation vector in, one flat float32 action vector out per call.
    Multi-input, tuple/dict-output, and dynamic/recurrent inference APIs are not part of this
    contract.
    """

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Run one inference on a flat float32 observation vector.

        Args:
            observation: Flat float32 observation vector.

        Returns:
            The flat float32 action vector.
        """

    def reset(self) -> None:
        """Reset backend recurrent state; a no-op when the backend has none."""

    def close(self) -> None:
        """Release backend resources; safe to call more than once."""


# Module-level import seams; tests patch these to inject fake runtimes without torch/ONNX.


def _import_torch() -> Any:
    from isaacsim.core.deprecation_manager import import_module

    return import_module("torch")


def _import_onnxruntime() -> Any:
    from isaacsim.core.deprecation_manager import import_module

    return import_module("onnxruntime")


def _find_nvidia_library(relative_path: str) -> Path | None:
    """Find one CUDA wheel library from the shared NVIDIA package.

    Args:
        relative_path: Path relative to the package root.

    Returns:
        The resulting :class:`Path | None`.
    """
    try:
        import nvidia
    except ImportError:
        return None

    for nvidia_root in nvidia.__path__:
        library_path = Path(nvidia_root) / relative_path
        if library_path.is_file():
            return library_path
    return None


def _preload_onnxruntime_cuda_libraries(ort: Any) -> None:
    """Preload CUDA wheel libraries so ONNX Runtime can activate its CUDA provider.

    Args:
        ort: Imported ``onnxruntime`` module.
    """
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return

    preload_dlls = getattr(ort, "preload_dlls", None)
    if preload_dlls is None:
        carb.log_warn("Unable to preload ONNXRuntime CUDA libraries: ONNX Runtime does not provide preload_dlls()")
        return

    try:
        if sys.platform.startswith("linux") and not _CUDA_LIBRARY_HANDLES:
            library_handles = []
            for relative_path in _LINUX_CUDA_LIBRARY_PATHS:
                library_path = _find_nvidia_library(relative_path)
                if library_path is None:
                    raise RuntimeError(f"NVIDIA CUDA library not found: nvidia/{relative_path}")
                library_handles.append(ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL))
            _CUDA_LIBRARY_HANDLES.extend(library_handles)
        preload_dlls()
    except (OSError, RuntimeError) as error:
        carb.log_warn(f"Unable to preload ONNXRuntime CUDA libraries: {error}")


def _as_flat_float32(values: object) -> np.ndarray:
    """Return values as a contiguous flat float32 vector.

    Args:
        values: Value sequence to coerce.

    Returns:
        The resulting array.
    """
    return np.ascontiguousarray(values, dtype=np.float32).reshape(-1)


def _validate_onnx_port(ports: list, side: str, requested: int | None, model_path: str) -> tuple[str, int, int]:
    """Validate one side of the ONNX graph and resolve its feature width.

    The runtime feeds exactly one flat float32 vector per call, so the graph must declare a
    single float32 port of rank 1 or 2 (batch dimension 1 when fixed). The static graph
    dimension is authoritative for the width: an explicit ``requested`` width may only agree
    with it — that agreement check is what surfaces truncated exports — and a dynamic
    dimension requires the explicit width.

    Returns:
        ``(port name, rank, feature width)``.

    Args:
        ports: Model input or output ports.
        side: Which port list is being validated.
        requested: Expected width, or None to accept the model's.
        model_path: Path to the policy model file.
    """
    if len(ports) != 1:
        names = [port.name for port in ports]
        raise ValueError(
            f"ONNX policy {model_path!r} declares {len(ports)} {side}s {names}; expected one flat " f"float {side}."
        )

    port = ports[0]
    prefix = f"ONNX policy {model_path!r} {side} {port.name!r}"
    shape = list(port.shape)
    rank = len(shape)

    if port.type != "tensor(float)":
        raise ValueError(f"{prefix} has dtype {port.type!r}; expected 'tensor(float)'.")
    if rank not in (1, 2):
        raise ValueError(f"{prefix} has shape {shape} of rank {rank}; expected rank 1 or 2.")
    if rank == 2 and isinstance(shape[0], int) and shape[0] != 1:
        raise ValueError(f"{prefix} has a fixed batch dimension of {shape[0]}; expected 1.")

    feature = shape[-1]
    if isinstance(feature, int) and feature > 0:
        # Static graph width: authoritative, the requested width may only agree.
        if requested is not None and requested != feature:
            raise ValueError(f"{prefix} has a static width of {feature} but {side}_width {requested} was " "requested.")
        return port.name, rank, feature

    # Dynamic graph width: the requested width is the only source.
    if requested is None:
        raise ValueError(f"{prefix} has a dynamic feature dimension; pass {side}_width explicitly.")
    return port.name, rank, requested


class OnnxPolicyModel:
    """ONNX policy model with I/O metadata introspected and validated at construction.

    The graph must declare exactly one float32 input and one float32 output of rank 1 or 2
    (batch dimension 1 if fixed). A static feature dim conflicting with an explicit
    ``input_width``/``output_width`` argument raises; a dynamic feature dim requires the
    matching explicit width.

    Args:
        model_path: Path to the policy model file.
        device: Compute device the model runs on.
        input_width: Expected observation width, or None to accept the model's.
        output_width: Expected action width, or None to accept the model's.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        input_width: int | None = None,
        output_width: int | None = None,
    ) -> None:
        self._session: Any | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.device = str(device)
        try:
            local_model_path = self._materialize_model(model_path)
            session = self._create_session(_import_onnxruntime(), local_model_path)
            self._session = session
            self._input_name, self._input_rank, self.input_width = _validate_onnx_port(
                list(session.get_inputs()), "input", input_width, model_path
            )
            self._output_name, _, self.output_width = _validate_onnx_port(
                list(session.get_outputs()), "output", output_width, model_path
            )
        except Exception:
            self.close()
            raise

    def __del__(self) -> None:
        """Release the ONNX session and temporary model files before destruction."""
        self.close()

    @property
    def providers(self) -> tuple[str, ...]:
        """Get active ONNXRuntime execution providers in priority order."""
        session = self._session
        if session is None:
            raise RuntimeError("ONNX policy model is closed.")
        return tuple(session.get_providers())

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Run one inference on a flat float32 observation vector.

        Args:
            observation: Flat float32 observation vector.

        Returns:
            The resulting array.
        """
        vector = _as_flat_float32(observation)
        if self._input_rank == 2:
            vector = vector[None, :]
        session = self._session
        if session is None:
            raise RuntimeError("ONNX policy model is closed.")
        outputs = session.run([self._output_name], {self._input_name: vector})
        return _as_flat_float32(outputs[0])

    def reset(self) -> None:
        """Reset recurrent state; ONNX sessions carry none."""

    def close(self) -> None:
        """Release the ONNX session and temporary model files; safe to call more than once."""
        self._session = None
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def _materialize_model(self, model_path: str) -> str:
        """Return a local model path, materializing remote model files when needed.

        Args:
            model_path: Path to the policy model file.

        Returns:
            The resolved string.
        """
        local_path = Path(model_path)
        if local_path.is_file():
            return str(local_path)

        self._temp_dir = tempfile.TemporaryDirectory(prefix="isaac_policy_onnx_")
        materialized_path = Path(self._temp_dir.name) / local_path.name
        materialized_path.write_bytes(read_artifact_bytes(model_path))

        try:
            sidecar_content = read_artifact_bytes(f"{model_path}.data")
        except FileNotFoundError:
            pass
        else:
            Path(f"{materialized_path}.data").write_bytes(sidecar_content)
        return str(materialized_path)

    def _create_session(self, ort: Any, model_path: str) -> Any:
        """Create an ONNXRuntime session using the requested device when available.

        Args:
            ort: Imported ``onnxruntime`` module.
            model_path: Path to the policy model file.

        Returns:
            The resulting :class:`Any`.
        """
        available_providers = list(ort.get_available_providers())
        providers = []
        cuda_device_id = None
        if self.device == "cuda":
            cuda_device_id = 0
        elif self.device.startswith("cuda:"):
            device_index = self.device.removeprefix("cuda:")
            if not device_index.isdecimal():
                raise ValueError(f"Invalid CUDA device {self.device!r}; expected 'cuda' or 'cuda:<index>'.")
            cuda_device_id = int(device_index)

        if cuda_device_id is not None:
            _preload_onnxruntime_cuda_libraries(ort)

            if "CUDAExecutionProvider" in available_providers:
                providers.append(("CUDAExecutionProvider", {"device_id": cuda_device_id}))
            else:
                carb.log_warn("ONNXRuntime CUDAExecutionProvider is unavailable; using CPUExecutionProvider.")
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")
        if not providers:
            providers = available_providers

        session = ort.InferenceSession(model_path, providers=providers)
        if cuda_device_id is not None and "CUDAExecutionProvider" not in session.get_providers():
            carb.log_warn("ONNXRuntime CUDAExecutionProvider did not activate; policy inference is using CPU.")
        return session


class TorchScriptPolicyModel:
    """TorchScript policy model loaded through ``omni.client`` onto a torch device.

    Each call runs under ``torch.no_grad`` and returns the detached, flattened output on the CPU.
    The controller validates the resulting action width against the bound policy interface.

    Args:
        model_path: Path to the policy model file.
        model_sha256: Full SHA-256 digest obtained from a trusted source.
        device: Compute device the model runs on.

    Raises:
        RuntimeError: If the model bytes do not match ``model_sha256``.
    """

    def __init__(self, model_path: str, model_sha256: str, *, device: str = "cpu") -> None:
        self.device = str(device)

        model_bytes = read_artifact_bytes(model_path)
        actual_sha256 = hashlib.sha256(model_bytes).hexdigest()
        if not hmac.compare_digest(actual_sha256, model_sha256):
            raise RuntimeError(
                f"TorchScript policy {model_path!r} failed SHA-256 verification: "
                f"expected {model_sha256}, got {actual_sha256}."
            )

        self._torch = _import_torch()
        file = io.BytesIO(model_bytes)
        self._policy: Any = self._torch.jit.load(file).to(self._torch.device(self.device))

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        """Run one inference on a flat float32 observation vector.

        Args:
            observation: Flat float32 observation vector.

        Returns:
            The resulting array.
        """
        torch = self._torch
        vector = _as_flat_float32(observation)
        # Recurrent TorchScript policies keep their exported weight layout; scope suppression
        # to this inference so importing a policy never changes process-global warning policy.
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="RNN module weights are not part of single contiguous chunk of memory.*",
                category=UserWarning,
            )
            action = self._policy(torch.as_tensor(vector, device=torch.device(self.device)))
            return action.detach().view(-1).cpu().numpy().copy()

    def reset(self) -> None:
        """Reset the loaded policy's recurrent state when it exposes a reset method."""
        reset = getattr(self._policy, "reset", None)
        if callable(reset):
            with self._torch.no_grad():
                reset()

    def close(self) -> None:
        """Release the loaded policy module; safe to call more than once."""
        self._policy = None


def load_policy_model(artifact: PolicyArtifact, interface: BoundPolicy, device: str) -> PolicyModel:
    """Load the artifact's model for its backend, armed with the bound interface widths.

    Args:
        artifact: Artifact naming the model file and its backend.
        interface: Policy interface bound to the robot's joint space.
        device: Compute device the model runs on.

    Returns:
        The resulting :class:`PolicyModel`.
    """
    input_width = interface.observation_width
    output_width = interface.action_width

    if artifact.backend == "torchscript":
        model_sha256 = artifact.model_sha256
        if model_sha256 is None:
            raise ValueError(f"TorchScript policy {artifact.model_path!r} has no trusted SHA-256 digest.")
        return TorchScriptPolicyModel(artifact.model_path, model_sha256, device=device)
    if artifact.backend == "onnx":
        # Passing the bound widths arms the constructor's static-vs-requested conflict check:
        # the Isaac Lab exporter silently skips undecorated observation terms, and a truncated
        # export (real franka case: 28-wide model vs the trained 31-term program) refuses to
        # load instead of binding misaligned observation data.
        return OnnxPolicyModel(artifact.model_path, device=device, input_width=input_width, output_width=output_width)

    raise ValueError(f"Cannot load policy backend {artifact.backend!r}.")
