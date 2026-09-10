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

"""Isaac Lab policy inference controller on the motion-generation control contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import carb
import isaacsim.robot_motion.experimental.motion_generation as mg
import numpy as np

from .utils.observation_history import ObservationHistory

if TYPE_CHECKING:
    from .binding import BoundPolicy
    from .model import PolicyModel

__all__ = ["IsaacLabPolicyController"]


class IsaacLabPolicyController(mg.BaseController):
    """Motion-generation controller that runs one Isaac Lab policy inference per forward call.

    The controller is externally paced and articulation-free: each ``forward`` builds one
    observation, runs one inference, updates its feedback buffer, and decodes one action. Cadence,
    state reads, and target application belong to the runner. ``forward`` holds the single output
    safety check -- a flat, finite vector of the bound action width -- and raises ValueError
    otherwise.

    The feedback buffer is per-controller and holds the raw pre-affine model output of the
    previous control tick, which is what ``last_action`` observation terms read. Exported
    observation history is also maintained per-controller and advances once per inference.

    Args:
        model: Loaded policy model runtime called once per forward call.
        interface: Policy interface bound to the deployed robot's joint space.
    """

    def __init__(self, model: PolicyModel, interface: BoundPolicy) -> None:
        self._model = model
        self._binding = interface
        self._raw_action_width = int(interface.action_width)
        # Raw model output of the preceding control tick, read by last_action terms.
        # Keep it per-controller so controllers may safely share a bound policy.
        self._last_action = np.zeros(self._raw_action_width, dtype=np.float32)
        self._observation_history = ObservationHistory(
            tuple((term.history_length, term.width) for term in interface.binding.observation_terms)
        )

    def reset(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> bool:
        """Reset to a safe initial state: zero the feedback buffer, reset the observation history, and the model backend.

        Backend reset failures are logged and reported through the return value so the caller
        can fail closed.

        Args:
            estimated_state: Current estimated robot state (unused; externally paced).
            setpoint_state: Optional setpoint/command state (unused; externally paced).
            t: Reset clock time; the runner always resets at ``0.0`` (unused).
            **kwargs: Accepted for the BaseController signature (unused).

        Returns:
            True when the feedback buffer and model backend both reset; False (after one carb
            error log) when either raised.

        """
        try:
            self._last_action[:] = 0.0
            self._observation_history.reset()
            self._model.reset()
        except Exception as error:
            carb.log_error(f"IsaacLabPolicyController.reset failed: {error}")
            return False
        return True

    def forward(
        self, estimated_state: mg.RobotState, setpoint_state: mg.RobotState | None, t: float, **kwargs: object
    ) -> mg.RobotState:
        """Produce the desired robot state for the next time step from one policy inference.

        Builds the flat observation vector, runs one inference, validates the raw output,
        records it in the feedback buffer, and decodes it into a partial named desired state.
        The validation is the single output-safety check between the model and the DOF writes.

        Args:
            estimated_state: Current estimated robot state read from the articulation.
            setpoint_state: Optional setpoint state carrying the command terms.
            t: Absolute simulation time since reset (``tick * physics_dt``); unused — the
                runner's decimation gate paces inference, not this clock.
            **kwargs: ``context`` carries values for task-specific observation terms.

        Returns:
            Partial named desired state decoded from the raw action.

        Raises:
            ValueError: The current-frame observation sample had the wrong width, or the raw
                model output was not a flat, fully finite vector of the bound action width.
        """
        observation = self._build_observation(
            estimated_state,
            setpoint_state,
            kwargs.get("context"),
        )
        raw_action = self._model(observation)

        action = np.asarray(raw_action)
        if action.shape != (self._raw_action_width,):
            raise ValueError(
                f"IsaacLabPolicyController: policy model {type(self._model).__name__} produced a raw "
                f"action of shape {action.shape}; expected a flat vector of {self._raw_action_width} values."
            )
        if not np.isfinite(action).all():
            raise ValueError(
                f"IsaacLabPolicyController: policy model {type(self._model).__name__} produced a raw "
                f"action of the expected shape {action.shape} containing non-finite values at indices "
                f"{np.flatnonzero(~np.isfinite(action)).tolist()}; the model diverged or was fed "
                f"non-finite state."
            )

        self._last_action[:] = action
        return self._binding.action(raw_action)

    def _build_observation(
        self,
        estimated_state: mg.RobotState,
        setpoint_state: mg.RobotState | None,
        context: Mapping[str, object] | None,
    ) -> np.ndarray:
        """Build one history-expanded model input and advance term histories.

        Args:
            estimated_state: Current estimated robot state read from the articulation.
            setpoint_state: Optional setpoint state carrying the command terms.
            context: Values for task-specific observation terms, or None.

        Returns:
            Flat history-expanded observation expected by the policy model.

        Raises:
            ValueError: The binding returned a current-frame sample of the wrong width.
        """
        sample = self._binding.observe(
            estimated_state,
            setpoint_state,
            self._last_action,
            context,
        )
        try:
            return self._observation_history.append(sample)
        except ValueError as error:
            raise ValueError(
                f"IsaacLabPolicyController: bound policy produced an invalid observation. {error}"
            ) from error

    def close(self) -> None:
        """Release the model."""
        self._model.close()
