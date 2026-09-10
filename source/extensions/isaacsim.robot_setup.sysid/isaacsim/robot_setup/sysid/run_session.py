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

"""Async optimization session: owns the run loop, USD writes, provenance, and bridge cleanup."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto

from .optimizer_base import OptimizationIterationStatus, OptimizerConfig
from .provenance import ProvenanceWriter, build_provenance_record
from .rollout_plot_data import SysIdRolloutPlotData
from .run_controller import SysIdPreparedRun, write_optimized_parameters_to_usd
from .trajectory_segments import (
    ChunkValidationMetric,
    TelemetryChunkRunSpec,
    TrajectoryChunk,
)
from .usd_write_layer import resolve_multiphysics_write_targets

IterationCallback = Callable[[OptimizationIterationStatus], None]
ProgressCallback = Callable[[str, float], None]
_LOGGER = logging.getLogger(__name__)


class RunState(Enum):  # noqa: D101
    IDLE = auto()
    RUNNING = auto()
    CANCELLED = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class SysIdRunResult:
    """Everything produced by one completed optimization session."""

    final_status: OptimizationIterationStatus
    selected_status: OptimizationIterationStatus
    selected_status_source: str
    last_accepted_status: OptimizationIterationStatus | None = None
    validation_metrics: list[ChunkValidationMetric] = field(default_factory=list)
    validation_rollouts: list[SysIdRolloutPlotData] = field(default_factory=list)
    validation_artifact_source: str = "configured_chunks"
    parameters_written: bool = False
    provenance_path: str = ""
    validation_render_path: str = ""
    validation_render_skipped_reason: str = ""
    validation_animation_enabled: bool = False
    validation_animation_path: str = ""
    validation_animation_fps: int = 24
    validation_animation_max_frames: int = 240
    #: `ParameterConfidenceReport.to_dict()` payload (empty when estimation is off/skipped).
    parameter_confidence: dict = field(default_factory=dict)


class SysIdRunSession:
    """Executes a prepared SysID run asynchronously.

    The session owns:
    - the optimizer run loop
    - USD parameter write-back
    - validation chunk evaluation
    - provenance authoring
    - bridge cleanup

    The caller supplies lightweight callbacks to push progress to the UI.

    Args:
        prepared_run: Constructor value for ``prepared_run``.
        apply_params_to_usd: Constructor value for ``apply_params_to_usd``.
        write_usd_provenance: Constructor value for ``write_usd_provenance``.
        export_provenance_sidecar: Constructor value for ``export_provenance_sidecar``.
        provenance_sidecar_path: Constructor value for ``provenance_sidecar_path``.
        remove_clones_after_run: Constructor value for ``remove_clones_after_run``.
        robot_path: Constructor value for ``robot_path``.
        validation_animation_enabled: Constructor value for ``validation_animation_enabled``.
        validation_animation_path: Constructor value for ``validation_animation_path``.
        validation_animation_fps: Constructor value for ``validation_animation_fps``.
        validation_animation_max_frames: Constructor value for ``validation_animation_max_frames``.
        compute_parameter_confidence: Constructor value for ``compute_parameter_confidence``.
        parameter_confidence_rollout_budget: Constructor value for ``parameter_confidence_rollout_budget``.
    """

    def __init__(
        self,
        prepared_run: SysIdPreparedRun,
        *,
        apply_params_to_usd: bool = True,
        write_usd_provenance: bool = True,
        export_provenance_sidecar: bool = False,
        provenance_sidecar_path: str = "",
        remove_clones_after_run: bool = True,
        robot_path: str = "",
        validation_animation_enabled: bool = False,
        validation_animation_path: str = "",
        validation_animation_fps: int = 24,
        validation_animation_max_frames: int = 240,
        compute_parameter_confidence: bool = True,
        parameter_confidence_rollout_budget: int = 2_000_000,
    ) -> None:
        self._prepared = prepared_run
        self._apply_params_to_usd = apply_params_to_usd
        self._write_usd_provenance = write_usd_provenance
        self._export_provenance_sidecar = export_provenance_sidecar
        self._provenance_sidecar_path = provenance_sidecar_path
        self._remove_clones = remove_clones_after_run
        self._robot_path = robot_path
        self._validation_animation_enabled = bool(validation_animation_enabled)
        self._validation_animation_path = validation_animation_path
        self._validation_animation_fps = int(validation_animation_fps)
        self._validation_animation_max_frames = int(validation_animation_max_frames)
        self._compute_parameter_confidence = bool(compute_parameter_confidence)
        self._parameter_confidence_budget = int(parameter_confidence_rollout_budget)
        self._state = RunState.IDLE

    @property
    def state(self) -> RunState:  # noqa: D102
        return self._state

    def cancel(self) -> None:  # noqa: D102
        self._prepared.optimizer.request_cancel()

    async def run_async(
        self,
        on_iteration: IterationCallback | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> SysIdRunResult:
        """Run optimization and all post-opt steps. Returns when complete.

        Raises ``asyncio.CancelledError`` on cancellation, any other exception
        on failure. Bridge cleanup always runs in ``finally``.

        Args:
            on_iteration: Optional iteration callback.
            on_progress: Optional progress callback.

        Returns:
            Completed run result with selected parameters and validation artifacts.
        """
        if self._state != RunState.IDLE:
            raise RuntimeError("SysIdRunSession may only be used once.")

        self._state = RunState.RUNNING
        prepared = self._prepared
        config = prepared.config
        optimizer = prepared.optimizer
        optimizer.reset_cancel()
        last_accepted: OptimizationIterationStatus | None = None

        def track_iteration(status: OptimizationIterationStatus) -> None:
            nonlocal last_accepted
            if status.accepted and int(status.iteration) > 0:
                last_accepted = status
            if on_iteration is not None:
                on_iteration(status)

        try:
            final = await optimizer.run_async(
                config,
                on_iteration=track_iteration,
                on_progress=on_progress,
            )
            selected, selected_source = _select_status_for_outputs(config, final, last_accepted)

            stage = prepared.stage

            params_written = False
            if self._apply_params_to_usd:
                if stage is None:
                    raise RuntimeError("USD parameter writeback was requested, but the prepared run has no stage.")
                bridge = optimizer.get_bridge() if hasattr(optimizer, "get_bridge") else None
                try:
                    neutral_write_target = None
                    solver_write_target = None
                    physics_backend = str(getattr(prepared, "physics_backend", ""))
                    if self._robot_path and physics_backend in ("newton", "physx"):
                        physics_solver = str(getattr(prepared, "physics_solver", ""))
                        solver_layer_name = "physx" if physics_backend == "physx" else ""
                        if physics_backend == "newton" and physics_solver == "mujoco":
                            solver_layer_name = "mujoco"
                        write_targets = resolve_multiphysics_write_targets(
                            stage,
                            self._robot_path,
                            solver_layer_name=solver_layer_name,
                        )
                        if write_targets is not None:
                            neutral_write_target = write_targets.neutral
                            solver_write_target = write_targets.solver
                    params_written = write_optimized_parameters_to_usd(
                        stage,
                        prepared.parameter_space,
                        config,
                        selected.theta,
                        max(1, len(prepared.link_paths)),
                        bridge,
                        physics_backend=str(getattr(prepared, "physics_backend", "")),
                        neutral_write_layer=neutral_write_target,
                        solver_write_layer=solver_write_target,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Requested USD parameter writeback failed: {exc}") from exc
                if not params_written:
                    raise RuntimeError("USD parameter writeback was requested, but no selected parameter was written.")

            # Gradient work is finished: drop the differentiable bridge's captured
            # CUDA graphs, tapes, and per-substep buffers before the forward-only
            # validation replay and confidence batch (which needs a fresh
            # multi-world context and otherwise OOMs on 16 GB cards).
            release_bridge = optimizer.get_bridge() if hasattr(optimizer, "get_bridge") else None
            release_fn = getattr(release_bridge, "release_rollout_memory", None)
            if callable(release_fn):
                try:
                    release_fn()
                except Exception as exc:
                    _LOGGER.warning("SysId: could not release solve rollout memory: %s", exc)

            validation_metrics: list[ChunkValidationMetric] = []
            validation_rollouts: list[SysIdRolloutPlotData] = []
            validation_chunks = list(prepared.validation_chunks)
            validation_source = "configured_chunks"
            if not validation_chunks:
                fallback_chunk = _full_telemetry_validation_chunk(config)
                if fallback_chunk is not None:
                    validation_chunks = [fallback_chunk]
                    validation_source = "full_telemetry_fallback"
            if validation_chunks:
                if on_progress is not None:
                    message = (
                        "Validating holdout chunks..."
                        if validation_source == "configured_chunks"
                        else "Validating full telemetry fallback..."
                    )
                    on_progress(message, 1.0)
                replay_fn = getattr(optimizer, "evaluate_validation_chunks_with_rollouts", None)
                if callable(replay_fn):
                    replays = await replay_fn(
                        config,
                        validation_chunks,
                        selected.theta,
                        on_progress=on_progress,
                    )
                    validation_metrics = [replay.metric for replay in replays]
                    validation_rollouts = [replay.rollout for replay in replays if replay.rollout is not None]
                else:
                    validate_fn = getattr(optimizer, "evaluate_validation_chunks", None)
                    if callable(validate_fn):
                        validation_metrics = await validate_fn(
                            config,
                            validation_chunks,
                            selected.theta,
                            on_progress=on_progress,
                        )
                if validation_source == "full_telemetry_fallback":
                    for metric in validation_metrics:
                        metric.extra["holdout"] = False
                        metric.extra["source"] = validation_source

            # Diagnostic only, after the USD write (never blocks it) and before bridge
            # cleanup (needs live rollouts); failures degrade to an empty payload.
            confidence_payload: dict = {}
            if self._compute_parameter_confidence and config.param_entries:
                confidence_chunks = validation_chunks or list(prepared.train_chunks)
                chunk_source = "validation_chunks" if validation_chunks else "train_chunks_fallback"
                try:
                    from .analysis import build_parameter_confidence_report

                    confidence_report = await build_parameter_confidence_report(
                        optimizer,
                        config,
                        confidence_chunks,
                        [float(value) for value in selected.theta],
                        max_rollout_sample_budget=self._parameter_confidence_budget,
                        on_progress=on_progress,
                    )
                    confidence_report.chunk_source = chunk_source
                    confidence_payload = confidence_report.to_dict()
                    for line in confidence_report.summary_lines():
                        _LOGGER.info("SysId confidence: %s", line)
                except Exception as exc:
                    _LOGGER.warning("SysId: parameter confidence estimation failed: %s", exc)

            provenance_path = ""
            if stage is not None and (self._write_usd_provenance or self._export_provenance_sidecar):
                provenance_path = _write_provenance(
                    stage,
                    prepared,
                    config,
                    selected,
                    backend_label=prepared.backend_label,
                    robot_path=self._robot_path,
                    validation_metrics=validation_metrics,
                    write_usd=self._write_usd_provenance,
                    export_sidecar=self._export_provenance_sidecar,
                    sidecar_path=self._provenance_sidecar_path,
                )

            self._state = RunState.DONE
            return SysIdRunResult(
                final_status=final,
                selected_status=selected,
                selected_status_source=selected_source,
                last_accepted_status=last_accepted,
                validation_metrics=validation_metrics,
                validation_rollouts=validation_rollouts,
                validation_artifact_source=validation_source,
                parameters_written=params_written,
                provenance_path=provenance_path,
                validation_animation_enabled=self._validation_animation_enabled,
                validation_animation_path=self._validation_animation_path,
                validation_animation_fps=self._validation_animation_fps,
                validation_animation_max_frames=self._validation_animation_max_frames,
                parameter_confidence=confidence_payload,
            )

        except asyncio.CancelledError:
            self._state = RunState.CANCELLED
            raise
        except Exception:
            self._state = RunState.ERROR
            raise
        finally:
            cleanup_async = getattr(optimizer, "cleanup_bridge_async", None)
            try:
                if callable(cleanup_async):
                    await cleanup_async(remove_clones=self._remove_clones)
                else:
                    optimizer.cleanup_bridge(remove_clones=self._remove_clones)
            except Exception as exc:
                _LOGGER.warning("SysId: bridge cleanup failed: %s", exc)


def _select_status_for_outputs(
    config: OptimizerConfig,
    final: OptimizationIterationStatus,
    last_accepted: OptimizationIterationStatus | None,
) -> tuple[OptimizationIterationStatus, str]:
    if final.accepted and int(final.iteration) > 0:
        return final, "final_accepted"
    if last_accepted is not None:
        return last_accepted, "last_accepted"
    if final.theta:
        return final, "initial_fallback"
    theta = config.theta_initial.detach().cpu().tolist()
    return (
        OptimizationIterationStatus(
            iteration=0,
            cost=float(final.cost),
            damping=float(final.damping),
            accepted=False,
            theta=[float(value) for value in theta],
            rollout=final.rollout,
        ),
        "initial_fallback",
    )


def _full_telemetry_validation_chunk(config: OptimizerConfig) -> TrajectoryChunk | None:
    trajectory = config.trajectory
    times = getattr(trajectory, "times", None)
    if times is None or len(times) < 2:
        return None
    start = float(times[0])
    end = float(times[-1])
    sample_count = int(len(times))
    return TrajectoryChunk(
        spec=TelemetryChunkRunSpec(
            name="Full telemetry fallback",
            role="validation",
            excitation="full_telemetry_fallback",
            start=start,
            end=end,
            weight=1.0,
        ),
        trajectory=trajectory,
        sample_count=sample_count,
        duration_seconds=float(end - start),
    )


def _write_provenance(
    stage,  # noqa: ANN001
    prepared: SysIdPreparedRun,
    config: OptimizerConfig,
    final: OptimizationIterationStatus,
    *,
    backend_label: str,
    robot_path: str,
    validation_metrics: list[ChunkValidationMetric],
    write_usd: bool,
    export_sidecar: bool,
    sidecar_path: str,
) -> str:
    """Author provenance to USD and/or a JSON sidecar. Returns the sidecar path (or '').

    Args:
        stage: USD stage used by the operation.
        prepared: Prepared SysID run state.
        config: Configuration for the operation.
        final: Selected optimization status to record.
        backend_label: Optimizer backend label.
        robot_path: Robot prim receiving stage metadata.
        validation_metrics: Per-chunk validation measurements.
        write_usd: Whether to author provenance on the USD stage.
        export_sidecar: Whether to export a JSON sidecar.
        sidecar_path: Destination for the optional JSON sidecar.

    Returns:
        Written sidecar path, or an empty string when no sidecar was requested.
    """  # noqa: DOC107
    record = build_provenance_record(
        config=config,
        final_status=final,
        optimizer_backend=backend_label,
        robot_prim_path=robot_path,
        validation_metrics=validation_metrics,
        simulation_engine=str(getattr(prepared, "simulation_engine", "")),
        newton_config=getattr(prepared, "newton_config", None),
    )
    writer = ProvenanceWriter()
    if write_usd:
        writer.write_to_stage(stage, robot_path, record)
    if export_sidecar and sidecar_path:
        return writer.export_json_sidecar(sidecar_path, record)
    return ""
