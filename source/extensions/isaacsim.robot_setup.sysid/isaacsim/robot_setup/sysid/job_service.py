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

"""Serialize remote SysID jobs inside one long-lived Kit process."""

from __future__ import annotations

import asyncio
import json
import tempfile
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

from .headless_job import (
    HeadlessSysIdJobOutcome,
    HeadlessSysIdJobRequest,
    run_headless_sysid_job,
    summarize_headless_sysid_job,
)
from .run_spec_paths import (
    RUN_SPEC_LOCAL_PATH_KEYS,
    is_run_spec_non_local_path,
    resolve_run_spec_local_path,
    resolve_run_spec_local_paths,
)

JobRunner = Callable[..., Awaitable[HeadlessSysIdJobOutcome]]
RuntimeResetter = Callable[[], Awaitable[None]]
SpecLoader = Callable[[Any], Any]


class SysIdJobState(str, Enum):
    """Lifecycle state for a queued SysID job."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SysIdJobCancelled(RuntimeError):
    """Internal signal for a user-requested SysID job cancellation."""


@dataclass(slots=True)
class SysIdServiceJobRequest:
    """Serializable request accepted by the persistent SysID job service.

    Exactly one of `run_spec_path` and `run_spec` must be supplied.

    Args:
        run_spec_path: Path to a SysID run-spec JSON file.
        run_spec: In-memory run-spec dictionary.
        stage_path: Optional USD stage override.
        result_json: Optional path for the compact completed-job summary.
        save_stage: Whether to save the stage after applying selected parameters.
        start_timeline: Whether to start the timeline before preflight.
        timeline_warmup_updates: Number of Kit updates after starting the timeline.
        reset_stage_after_run: Whether to replace the stage with a clean empty stage after the job.

    Example:

    .. code-block:: python

        request = SysIdServiceJobRequest(run_spec_path="C:/runs/sysid_run_spec.json")
        job_id = submit_sysid_job(request)
        print(job_id)
    """

    run_spec_path: str = ""
    run_spec: dict[str, Any] | None = None
    stage_path: str = ""
    result_json: str = ""
    save_stage: bool = False
    start_timeline: bool = True
    timeline_warmup_updates: int = 0
    reset_stage_after_run: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SysIdServiceJobRequest:
        """Create and validate a service request from JSON-safe values.

        Args:
            payload: Request fields accepted by the class constructor.

        Returns:
            Validated service job request.

        Raises:
            TypeError: If the payload or nested run spec has the wrong type.
            ValueError: If the request does not identify exactly one run spec.

        Example:

        .. code-block:: python

            request = SysIdServiceJobRequest.from_dict({"run_spec_path": "C:/runs/spec.json"})
            print(request.run_spec_path)
        """
        if not isinstance(payload, dict):
            raise TypeError("SysID service request must be a dictionary.")
        allowed = {field_info.name for field_info in cls.__dataclass_fields__.values()}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown SysID service request fields: {', '.join(unknown)}")
        run_spec = payload.get("run_spec")
        if run_spec is not None and not isinstance(run_spec, dict):
            raise TypeError("run_spec must be a dictionary when provided.")
        request = cls(**payload)
        request.validate()
        return request

    def validate(self) -> None:
        """Validate mutually exclusive inputs and numeric settings.

        Raises:
            ValueError: If request fields are inconsistent.
        """
        if bool(self.run_spec_path) == bool(self.run_spec is not None):
            raise ValueError("Provide exactly one of run_spec_path or run_spec.")
        if not isinstance(self.timeline_warmup_updates, int) or self.timeline_warmup_updates < 0:
            raise ValueError("timeline_warmup_updates must be a non-negative integer.")

    def load_spec(self, *, allowed_roots: list[str | Path] | tuple[str | Path, ...] = ()) -> Any:
        """Load the request's run specification under the caller's path jail.

        Reading a request's own filesystem paths is the same privileged
        operation the service performs, so it enforces the same jail. Callers
        that only supply an inline ``run_spec`` need no roots; anything that
        touches the filesystem must pass the roots it is willing to expose.

        Args:
            allowed_roots: Filesystem roots this load may read from. An empty
                sequence rejects every path-backed field.

        Returns:
            Parsed `SysIdRunSpec` instance.

        Raises:
            OSError: If the run-spec file cannot be read.
            ValueError: If the payload is invalid or a path escapes the jail.
        """
        self.validate()
        roots = _normalize_allowed_roots(allowed_roots)
        return _load_service_spec(self, allowed_roots=roots)[0]


@dataclass(slots=True)
class _SysIdJobRecord:
    """Mutable internal state for one service job."""

    job_id: str
    request: SysIdServiceJobRequest
    state: SysIdJobState = SysIdJobState.QUEUED
    message: str = "Queued"
    progress_fraction: float = 0.0
    created_at: str = field(default_factory=lambda: _timestamp())
    started_at: str = ""
    finished_at: str = ""
    backend: str = ""
    iteration: int = 0
    cost: float | None = None
    cancel_requested: bool = False
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe job snapshot.

        Returns:
            The current job state and result metadata.
        """
        return {
            "job_id": self.job_id,
            "state": self.state.value,
            "message": self.message,
            "progress_fraction": self.progress_fraction,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "backend": self.backend,
            "iteration": self.iteration,
            "cost": self.cost,
            "cancel_requested": self.cancel_requested,
            "result": self.result,
            "error": self.error,
        }


class SysIdJobService:
    """Single-worker queue for SysID jobs sharing one Kit runtime.

    The service deliberately runs only one job at a time because Kit exposes a
    process-wide USD context, timeline, and active physics engine.

    Args:
        runner: Async public job API used by the worker.
        runtime_resetter: Async cleanup called after jobs that request a clean stage.
        spec_loader: Callable that converts a service request into a run spec.
        max_retained_jobs: Maximum number of terminal job snapshots retained.
        max_pending_jobs: Maximum number of jobs waiting behind the active job.
        allowed_roots: Filesystem roots available to path-backed requests. An
            empty sequence disables all service filesystem access.
    """

    def __init__(
        self,
        *,
        runner: JobRunner = run_headless_sysid_job,
        runtime_resetter: RuntimeResetter | None = None,
        spec_loader: SpecLoader | None = None,
        max_retained_jobs: int = 100,
        max_pending_jobs: int = 100,
        allowed_roots: list[str | Path] | tuple[str | Path, ...] = (),
    ) -> None:
        if max_retained_jobs < 1:
            raise ValueError("max_retained_jobs must be positive.")
        if max_pending_jobs < 1:
            raise ValueError("max_pending_jobs must be positive.")
        self._runner = runner
        self._runtime_resetter = runtime_resetter or _reset_runtime_stage
        self._spec_loader = spec_loader
        self._allowed_roots = _normalize_allowed_roots(allowed_roots)
        self._max_retained_jobs = int(max_retained_jobs)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=int(max_pending_jobs))
        self._jobs: dict[str, _SysIdJobRecord] = {}
        self._worker_task: asyncio.Task[None] | None = None
        self._active_job_id = ""
        self._active_prepared: Any | None = None
        self._shutting_down = False

    def submit(self, request: SysIdServiceJobRequest | dict[str, Any]) -> str:
        """Queue one job and return its stable identifier.

        Args:
            request: Serializable request object or dictionary.

        Returns:
            UUID identifying the queued job.

        Raises:
            RuntimeError: If called outside a running event loop or during shutdown.
            TypeError: If the request has an unsupported type.
            ValueError: If request fields are invalid.
        """
        asyncio.get_running_loop()
        if self._shutting_down:
            raise RuntimeError("The SysID job service is shutting down.")
        if isinstance(request, dict):
            request = SysIdServiceJobRequest.from_dict(request)
        elif isinstance(request, SysIdServiceJobRequest):
            request.validate()
        else:
            raise TypeError("request must be SysIdServiceJobRequest or a dictionary.")
        _validate_service_request_paths(request, self._allowed_roots)
        self._prune_terminal_jobs()
        if self._queue.full():
            raise RuntimeError("The SysID job queue is full; retry after a pending job completes.")
        job_id = str(uuid4())
        self._jobs[job_id] = _SysIdJobRecord(job_id=job_id, request=request)
        self._queue.put_nowait(job_id)
        self._ensure_worker()
        return job_id

    def status(self, job_id: str) -> dict[str, Any]:
        """Return a JSON-safe snapshot for one job.

        Args:
            job_id: Identifier returned by `submit()`.

        Returns:
            Current job state and any terminal result or error.

        Raises:
            KeyError: If the job identifier is unknown.
        """
        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(f"Unknown SysID job: {job_id}")
        return record.to_dict()

    def list_jobs(self) -> list[dict[str, Any]]:
        """Return snapshots for all retained jobs.

        Returns:
            Job snapshots ordered by submission time.
        """
        return [record.to_dict() for record in self._jobs.values()]

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a queued or running job.

        Args:
            job_id: Identifier returned by `submit()`.

        Returns:
            True when cancellation was newly requested, otherwise False.

        Raises:
            KeyError: If the job identifier is unknown.
        """
        record = self._jobs.get(job_id)
        if record is None:
            raise KeyError(f"Unknown SysID job: {job_id}")
        if record.state in _TERMINAL_STATES or record.cancel_requested:
            return False
        record.cancel_requested = True
        if record.state == SysIdJobState.QUEUED:
            self._remove_queued_job(record.job_id)
            self._terminalize_cancelled(record, "Cancelled before execution")
            return True
        record.message = "Cancellation requested"
        optimizer = getattr(self._active_prepared, "optimizer", None)
        request_cancel = getattr(optimizer, "request_cancel", None)
        if callable(request_cancel):
            request_cancel()
        return True

    def shutdown(self) -> None:
        """Request worker shutdown without abandoning its task handle."""
        self._request_shutdown()

    async def shutdown_async(self) -> None:
        """Request shutdown and wait until worker cleanup has completed."""
        task = self._request_shutdown()
        if task is not None:
            with suppress(asyncio.CancelledError):
                await task

    def _request_shutdown(self) -> asyncio.Task[None] | None:
        """Stop accepting work, request active cancellation, and return the worker.

        Returns:
            Worker task being shut down, or None when no worker exists.
        """
        self._shutting_down = True
        self._cancel_pending_jobs("Cancelled because the SysID job service is shutting down")
        if self._active_job_id:
            try:
                self.cancel(self._active_job_id)
            except KeyError:
                pass
        task = self._worker_task
        if task is not None and not task.done():
            task.cancel()
        return task

    def _cancel_pending_jobs(self, message: str) -> None:
        """Drain the queue and publish a terminal cancellation for every pending job.

        Args:
            message: Terminal status message assigned to each pending job.
        """
        while True:
            try:
                job_id = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                record = self._jobs.get(job_id)
                if record is not None and record.state == SysIdJobState.QUEUED:
                    record.cancel_requested = True
                    self._terminalize_cancelled(record, message)
            finally:
                self._queue.task_done()

    def _remove_queued_job(self, job_id: str) -> None:
        """Remove one cancelled identifier immediately so it no longer consumes queue capacity.

        Args:
            job_id: Queued job identifier to remove.
        """
        retained: list[str] = []
        while True:
            try:
                queued_id = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._queue.task_done()
            if queued_id != job_id:
                retained.append(queued_id)
        for queued_id in retained:
            self._queue.put_nowait(queued_id)

    def _terminalize_cancelled(self, record: _SysIdJobRecord, message: str) -> None:
        """Mark one pending record cancelled and publish its requested result artifact.

        Args:
            record: Pending job record to finalize.
            message: Terminal cancellation message.
        """
        record.state = SysIdJobState.CANCELLED
        record.message = message
        record.result = {
            "ok": False,
            "job_state": record.state.value,
            "message": record.message,
            "error": None,
        }
        if record.request.result_json:
            try:
                _write_json(
                    Path(record.request.result_json),
                    record.result,
                    allowed_roots=self._allowed_roots,
                )
            except Exception as exc:
                record.state = SysIdJobState.FAILED
                record.message = "Result JSON write failed"
                record.error = {"type": type(exc).__name__, "message": str(exc)}
                record.result = {
                    **record.result,
                    "job_state": record.state.value,
                    "message": record.message,
                    "error": dict(record.error),
                }
        record.finished_at = _timestamp()

    def _ensure_worker(self) -> None:
        """Create the worker task when no worker is active."""
        if self._worker_task is None or self._worker_task.done():
            task = asyncio.create_task(self._worker(), name="sysid-persistent-job-worker")
            task.add_done_callback(self._worker_done)
            self._worker_task = task

    def _worker_done(self, task: asyncio.Task[None]) -> None:
        """Consume worker completion and clear only its matching task handle.

        Args:
            task: Completed service worker task.
        """
        with suppress(asyncio.CancelledError):
            task.exception()
        if self._worker_task is task:
            self._worker_task = None

    async def _worker(self) -> None:
        """Process queued job identifiers sequentially."""
        while not self._shutting_down:
            job_id = await self._queue.get()
            try:
                record = self._jobs.get(job_id)
                if record is not None and record.state == SysIdJobState.QUEUED:
                    await self._execute(record)
            finally:
                self._queue.task_done()

    async def _execute(self, record: _SysIdJobRecord) -> None:
        """Execute one queued record and capture terminal state.

        Args:
            record: Mutable job record to execute.
        """
        record.state = SysIdJobState.RUNNING
        record.message = "Loading run specification"
        record.started_at = _timestamp()
        self._active_job_id = record.job_id

        def on_prepared(prepared: Any, _preflight: Any, _quality: Any, _schema_issues: Any) -> None:
            self._active_prepared = prepared
            record.backend = str(getattr(prepared, "backend_label", ""))
            record.message = "Running optimization"
            if record.cancel_requested:
                raise SysIdJobCancelled("Cancelled before optimization started")

        def on_progress(message: str, fraction: float) -> None:
            record.message = str(message)
            record.progress_fraction = max(record.progress_fraction, max(0.0, min(1.0, float(fraction))))

        def on_iteration(status: Any) -> None:
            record.iteration = int(status.iteration)
            record.cost = float(status.cost)

        try:
            loaded_spec = (
                self._spec_loader(record.request)
                if self._spec_loader is not None
                else _load_service_spec(record.request, allowed_roots=self._allowed_roots)
            )
            if isinstance(loaded_spec, tuple) and len(loaded_spec) == 2 and isinstance(loaded_spec[1], dict):
                spec, raw_spec_payload = loaded_spec
            else:
                spec = loaded_spec
                raw_spec_payload = None
            stage_path = record.request.stage_path
            if stage_path:
                stage_path = str(
                    _resolve_jailed_path(
                        Path(stage_path),
                        self._allowed_roots,
                        field_name="stage_path",
                    )
                )
            outcome = await self._runner(
                HeadlessSysIdJobRequest(
                    spec=spec,
                    stage_path=stage_path,
                    raw_spec_payload=raw_spec_payload,
                    start_timeline=record.request.start_timeline,
                    timeline_warmup_updates=record.request.timeline_warmup_updates,
                    save_stage=record.request.save_stage,
                ),
                on_iteration=on_iteration,
                on_progress=on_progress,
                on_prepared=on_prepared,
            )
            summary = summarize_headless_sysid_job(outcome)
            record.result = summary
            # Completion wins over a cancellation request arriving after the
            # runner has already returned a successful solve/write outcome.
            record.state = SysIdJobState.SUCCEEDED
            record.message = "Completed"
            record.progress_fraction = 1.0
        except SysIdJobCancelled:
            record.state = SysIdJobState.CANCELLED
            record.message = "Cancelled"
        except asyncio.CancelledError:
            if not (record.cancel_requested or self._shutting_down):
                record.state = SysIdJobState.FAILED
                record.message = "Job worker task was cancelled externally"
                record.error = {
                    "type": "CancelledError",
                    "message": record.message,
                }
                raise
            record.state = SysIdJobState.CANCELLED
            record.message = "Cancelled"
        except Exception as exc:
            record.state = SysIdJobState.CANCELLED if record.cancel_requested else SysIdJobState.FAILED
            record.message = "Cancelled" if record.cancel_requested else str(exc)
            record.error = {"type": type(exc).__name__, "message": str(exc)}
        finally:
            if record.request.reset_stage_after_run:
                try:
                    await self._runtime_resetter()
                except Exception as exc:
                    if record.error is None:
                        record.error = {
                            "type": type(exc).__name__,
                            "message": f"Runtime reset failed: {exc}",
                        }
                    if record.state == SysIdJobState.SUCCEEDED:
                        record.state = SysIdJobState.FAILED
                        record.message = "Runtime reset failed"
            if record.state != SysIdJobState.SUCCEEDED:
                record.result = {
                    **(record.result or {}),
                    "ok": False,
                    "job_state": record.state.value,
                    "message": record.message,
                    "error": dict(record.error) if record.error is not None else None,
                }
            if record.result is not None and record.request.result_json:
                try:
                    _write_json(
                        Path(record.request.result_json),
                        record.result,
                        allowed_roots=self._allowed_roots,
                    )
                except Exception as exc:
                    record.state = SysIdJobState.FAILED
                    record.message = "Result JSON write failed"
                    record.error = {"type": type(exc).__name__, "message": str(exc)}
                    record.result = {
                        **record.result,
                        "ok": False,
                        "job_state": record.state.value,
                        "error": dict(record.error),
                    }
            record.finished_at = _timestamp()
            self._active_job_id = ""
            self._active_prepared = None

    def _prune_terminal_jobs(self) -> None:
        """Evict oldest terminal jobs above the retention limit."""
        terminal_ids = [job_id for job_id, record in self._jobs.items() if record.state in _TERMINAL_STATES]
        for job_id in terminal_ids[: max(0, len(terminal_ids) - self._max_retained_jobs + 1)]:
            del self._jobs[job_id]


_TERMINAL_STATES = {
    SysIdJobState.SUCCEEDED,
    SysIdJobState.FAILED,
    SysIdJobState.CANCELLED,
}
_SERVICE: SysIdJobService | None = None
_RETIRED_SERVICE: SysIdJobService | None = None
_SERVICE_SHUTDOWN_TASK: asyncio.Task[None] | None = None


def get_sysid_job_service(
    *,
    allowed_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
) -> SysIdJobService:
    """Return the process-wide persistent SysID job service.

    Args:
        allowed_roots: Filesystem roots to configure when creating the
            singleton. Omit to create a path-disabled service.

    Returns:
        Singleton service for the current Kit process.
    """
    global _SERVICE, _SERVICE_SHUTDOWN_TASK
    shutdown_task = _SERVICE_SHUTDOWN_TASK
    if shutdown_task is not None:
        if not shutdown_task.done():
            raise RuntimeError("The previous SysID job service is still shutting down.")
        with suppress(asyncio.CancelledError):
            shutdown_task.exception()
        _SERVICE_SHUTDOWN_TASK = None
    if _SERVICE is None:
        _SERVICE = SysIdJobService(allowed_roots=allowed_roots or ())
    elif allowed_roots is not None and _SERVICE._allowed_roots != _normalize_allowed_roots(allowed_roots):
        raise RuntimeError("The existing SysID job service uses different allowed_roots.")
    return _SERVICE


def submit_sysid_job(request: SysIdServiceJobRequest | dict[str, Any]) -> str:
    """Submit one job to the process-wide service.

    Args:
        request: Serializable job request or request object.

    Returns:
        Stable job identifier for status and cancellation calls.
    """
    return get_sysid_job_service().submit(request)


def get_sysid_job_status(job_id: str) -> dict[str, Any]:
    """Get one retained job snapshot.

    Args:
        job_id: Identifier returned by `submit_sysid_job()`.

    Returns:
        JSON-safe job snapshot.
    """
    for service in (_SERVICE, _RETIRED_SERVICE):
        if service is None:
            continue
        try:
            return service.status(job_id)
        except KeyError:
            continue
    raise KeyError(f"Unknown SysID job: {job_id}")


def list_sysid_jobs() -> list[dict[str, Any]]:
    """List all retained jobs.

    Returns:
        JSON-safe job snapshots ordered by submission time.
    """
    snapshots: dict[str, dict[str, Any]] = {}
    for service in (_RETIRED_SERVICE, _SERVICE):
        if service is not None:
            snapshots.update({row["job_id"]: row for row in service.list_jobs()})
    return list(snapshots.values())


def cancel_sysid_job(job_id: str) -> bool:
    """Request cancellation of one job.

    Args:
        job_id: Identifier returned by `submit_sysid_job()`.

    Returns:
        True when cancellation was newly requested, otherwise False.
    """
    for service in (_SERVICE, _RETIRED_SERVICE):
        if service is None:
            continue
        try:
            return service.cancel(job_id)
        except KeyError:
            continue
    raise KeyError(f"Unknown SysID job: {job_id}")


def shutdown_sysid_job_service() -> None:
    """Request process-wide service shutdown and track it until fully drained."""
    global _RETIRED_SERVICE, _SERVICE, _SERVICE_SHUTDOWN_TASK
    if _SERVICE is not None:
        service = _SERVICE
        _SERVICE = None
        _RETIRED_SERVICE = service
        task = service._request_shutdown()
        if task is not None and not task.done():
            _SERVICE_SHUTDOWN_TASK = task


async def shutdown_sysid_job_service_async() -> None:
    """Shut down and drain the process-wide service while retaining terminal job snapshots."""
    global _RETIRED_SERVICE, _SERVICE, _SERVICE_SHUTDOWN_TASK
    if _SERVICE is not None:
        service = _SERVICE
        _SERVICE = None
        _RETIRED_SERVICE = service
        await service.shutdown_async()
    shutdown_task = _SERVICE_SHUTDOWN_TASK
    if shutdown_task is not None:
        with suppress(asyncio.CancelledError):
            await shutdown_task
        if _SERVICE_SHUTDOWN_TASK is shutdown_task:
            _SERVICE_SHUTDOWN_TASK = None


async def _reset_runtime_stage() -> None:
    """Stop simulation and replace the current stage with a clean empty stage."""
    from isaacsim.core.experimental.utils import app as app_utils
    from isaacsim.core.experimental.utils import stage as stage_utils

    if app_utils.is_playing():
        app_utils.stop()
        await app_utils.update_app_async(steps=2)
    await stage_utils.create_new_stage_async(template="empty")
    await app_utils.update_app_async(steps=5)


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    allowed_roots: tuple[Path, ...],
) -> None:
    """Write one JSON payload atomically enough for job result handoff.

    Args:
        path: Destination JSON path.
        payload: JSON-compatible content to write.
        allowed_roots: Explicit filesystem roots available to the service.
    """
    path = _resolve_jailed_path(path, allowed_roots, field_name="result_json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _resolve_jailed_path(path, allowed_roots, field_name="result_json")
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            temporary_path = Path(stream.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_service_spec(
    request: SysIdServiceJobRequest,
    *,
    allowed_roots: tuple[Path, ...],
) -> tuple[Any, dict[str, Any]]:
    """Load one service request's run specification.

    Args:
        request: Service request containing an inline spec or spec path.
        allowed_roots: Filesystem roots available to the request.

    Returns:
        Resolved run specification and unchanged serialized payload.
    """
    from .run_spec import SysIdRunSpec

    request.validate()
    payload = request.run_spec
    base_directory = Path.cwd()
    if payload is None:
        path = _resolve_jailed_path(
            Path(request.run_spec_path),
            allowed_roots,
            field_name="run_spec_path",
        )
        base_directory = path.parent
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    spec = SysIdRunSpec.from_dict(payload)
    resolve_run_spec_local_paths(spec, base_directory)
    _validate_spec_payload_paths(spec.to_dict(), allowed_roots)
    return spec, payload


def _normalize_allowed_roots(
    roots: list[str | Path] | tuple[str | Path, ...],
) -> tuple[Path, ...]:
    """Resolve and deduplicate explicitly configured filesystem roots.

    Args:
        roots: Filesystem roots configured by the service host.

    Returns:
        Canonical directory paths with duplicates removed.
    """
    normalized: list[Path] = []
    for root in roots:
        resolved = Path(root).expanduser().resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"SysID job-service allowed root is not a directory: {resolved}")
        if resolved not in normalized:
            normalized.append(resolved)
    return tuple(normalized)


def _resolve_jailed_path(
    path: Path,
    allowed_roots: tuple[Path, ...],
    *,
    field_name: str,
) -> Path:
    """Resolve one local path and reject access outside the configured jail.

    Args:
        path: Local path supplied by a service request.
        allowed_roots: Canonical roots available to the service.
        field_name: Request field used in validation errors.

    Returns:
        Canonical path confined beneath an allowed root.

    Raises:
        ValueError: If filesystem access is disabled or the path escapes the jail.
    """
    if not allowed_roots:
        raise ValueError(f"{field_name} requires filesystem access, but this SysID job service has no allowed_roots.")
    resolved = path.expanduser().resolve(strict=False)
    if not any(resolved == root or resolved.is_relative_to(root) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(f"{field_name} resolves outside the SysID job-service allowed_roots ({roots}): {resolved}")
    return resolved


def _validate_service_request_paths(
    request: SysIdServiceJobRequest,
    allowed_roots: tuple[Path, ...],
) -> None:
    """Validate every request-level and inline-spec filesystem path before queueing.

    Args:
        request: Request to validate.
        allowed_roots: Canonical roots available to the service.
    """
    for field_name in ("run_spec_path", "stage_path", "result_json"):
        value = str(getattr(request, field_name, "") or "")
        if value:
            _resolve_jailed_path(Path(value), allowed_roots, field_name=field_name)
    if request.run_spec is not None:
        _validate_spec_payload_paths(request.run_spec, allowed_roots)


def _validate_spec_payload_paths(payload: Any, allowed_roots: tuple[Path, ...], prefix: str = "") -> None:
    """Recursively jail known local-path fields in a serialized run specification.

    Args:
        payload: Serialized run-spec value to inspect.
        allowed_roots: Canonical roots available to the service.
        prefix: Nested field name used in validation errors.
    """
    if isinstance(payload, dict):
        for key, value in payload.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in RUN_SPEC_LOCAL_PATH_KEYS and isinstance(value, str) and value:
                if is_run_spec_non_local_path(value):
                    continue
                resolved_value = resolve_run_spec_local_path(value, Path.cwd())
                _resolve_jailed_path(Path(resolved_value), allowed_roots, field_name=path)
            else:
                _validate_spec_payload_paths(value, allowed_roots, path)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _validate_spec_payload_paths(value, allowed_roots, f"{prefix}[{index}]")


def _timestamp() -> str:
    """Return the current UTC timestamp.

    Returns:
        An ISO-8601 timestamp.
    """
    return datetime.now(timezone.utc).isoformat()
