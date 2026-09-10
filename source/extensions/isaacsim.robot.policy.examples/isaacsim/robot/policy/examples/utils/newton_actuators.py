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

"""Adapt exported Isaac Lab actuator configs to Newton actuator components."""

from __future__ import annotations

import os
import tempfile
import warnings

import warp as wp
from isaacsim.core.experimental.actuators import ActuatorConfig
from newton.actuators import (
    ClampingDCMotor,
    ClampingMaxEffort,
    ClampingPositionBased,
    ControllerNeuralLSTM,
    ControllerNeuralMLP,
    ControllerPD,
    Delay,
)

from ..env_config import SUPPORTED_NEWTON_ACTUATOR_CLASSES, NewtonActuatorSpec


def _materialize_checkpoint(data: bytes) -> str:
    """Write an in-memory checkpoint buffer to a temporary ``.pt`` file.

    Args:
        data: Serialized actuator checkpoint.

    Returns:
        Path to the temporary checkpoint.
    """
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp.write(data)
        return tmp.name


def build_newton_actuator_configs(
    spec: NewtonActuatorSpec,
    network_data: bytes | None = None,
    *,
    num_robots: int = 1,
    device: str | None = None,
) -> list[tuple[ActuatorConfig, str]]:
    """Build Newton ``(ActuatorConfig, dof_name)`` pairs for one exported group.

    Neural actuator checkpoints are accepted as bytes because deployment policies may come
    from Nucleus or other ``omni.client`` paths.

    Args:
        spec: Exported actuator parameters resolved to simulator DOF order.
        network_data: Serialized neural actuator checkpoint, when required.
        num_robots: Number of articulation instances sharing the actuator topology.
        device: Warp device used for actuator component arrays.

    Returns:
        Newton actuator configs paired with their target DOF names.

    Raises:
        ValueError: If an actuator parameter or required checkpoint is invalid.
        NotImplementedError: If the exported actuator class has no Newton adapter.
    """
    if num_robots < 1:
        raise ValueError(f"num_robots must be positive, got {num_robots}.")

    class_name = spec.class_name
    if class_name not in SUPPORTED_NEWTON_ACTUATOR_CLASSES:
        raise NotImplementedError(f"Actuator class {class_name!r} is not supported by the Newton actuator adapter.")

    if class_name.startswith("ActuatorNet") and network_data is None:
        raise ValueError(f"{class_name} requires a neural actuator checkpoint buffer.")

    network_path = _materialize_checkpoint(network_data) if class_name.startswith("ActuatorNet") else None
    try:

        def scalar(value: float) -> object:
            return wp.array([float(value)] * num_robots, dtype=wp.float32, device=device)

        pairs: list[tuple[ActuatorConfig, str]] = []
        for i, dof_name in enumerate(spec.joints):
            kp = spec.stiffness[i]
            kd = spec.damping[i]
            effort_limit = spec.effort_limit[i]
            velocity_limit = spec.velocity_limit[i]
            saturation_effort = spec.saturation_effort[i]

            if class_name == "ActuatorNetLSTM":
                # The LSTM network keeps its exported weight layout and is stepped inside the
                # engine's pre-physics callback, where no caller scope can wrap it; cuDNN's
                # non-contiguous-chunk warning would otherwise print once per substep. The
                # filter is message-narrow but process-global by necessity.
                warnings.filterwarnings(
                    "ignore",
                    message="RNN module weights are not part of single contiguous chunk of memory.*",
                    category=UserWarning,
                )
                controller = ControllerNeuralLSTM(model_path=network_path)
            elif class_name == "ActuatorNetMLP":
                controller = ControllerNeuralMLP(model_path=network_path)
            else:
                controller = ControllerPD(kp=scalar(kp), kd=scalar(kd))

            if class_name == "RemotizedPDActuator":
                if not spec.joint_parameter_lookup:
                    raise ValueError("RemotizedPDActuator requires joint_parameter_lookup.")
                clamping = [
                    ClampingPositionBased(
                        lookup_positions=tuple(row[0] for row in spec.joint_parameter_lookup),
                        lookup_efforts=tuple(row[2] for row in spec.joint_parameter_lookup),
                    )
                ]
            elif class_name in {"IdealPDActuator", "DelayedPDActuator"}:
                clamping = [ClampingMaxEffort(scalar(effort_limit))]
            else:
                clamping = [ClampingDCMotor(scalar(saturation_effort), scalar(velocity_limit), scalar(effort_limit))]

            delay = None
            # Match Isaac Lab's Newton-native export, which authors max_delay as a fixed deployment delay.
            # See https://github.com/isaac-sim/IsaacLab/blob/develop/source/isaaclab/isaaclab/sim/schemas/schemas_actuators.py#L252-L256
            # Note that old IsaacLab Actuators default is to choose a random delay from min_delay to max_delay.
            # https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab/isaaclab/actuators/actuator_pd.py#L310
            if class_name in {"DelayedDCMotor", "DelayedPDActuator", "RemotizedPDActuator"} and spec.max_delay > 0:
                delay_steps = wp.full(num_robots, spec.max_delay, dtype=wp.int32, device=device)
                delay = Delay(delay_steps=delay_steps, max_delay=spec.max_delay)

            pairs.append((ActuatorConfig(controller=controller, clamping=clamping, delay=delay), dof_name))
        return pairs
    finally:
        if network_path is not None and os.path.exists(network_path):
            os.unlink(network_path)
