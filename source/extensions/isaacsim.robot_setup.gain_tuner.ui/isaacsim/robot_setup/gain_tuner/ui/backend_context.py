# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Data class describing the active physics backend state.

The physics backend and solver are detected live from the running Isaac Sim app
via :meth:`BackendContext.from_app`.  The solver is load-bearing rather than
cosmetic: it selects the schema-resolver order the advanced joint parameters are
read through, so ``solver_type`` is carried as a raw token beside the display
name, and :attr:`BackendContext.solver_known` reports when the order cannot be
determined.

The DriveAPI save-target selection and the MuJoCo / Newton mirror state are
populated from the resolved robot layers via
:meth:`BackendContext.apply_save_targets`.  The UI reads only from this object so
those swaps stay single-file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import isaacsim.robot_setup.gain_tuner as gain_tuner

# Map the simulation-manager engine names to the labels the UI uses internally.
_ENGINE_TO_BACKEND = {"physx": "PhysX", "newton": "NewtonAPI"}

# Pretty solver names for the Newton solver tokens.  ``"none"`` is the placeholder
# ``NewtonSolverConfig.solver_type`` carries before Newton initializes, and maps to
# the empty string so the UI reports the solver as unknown rather than naming one.
_SOLVER_PRETTY = {
    "featherstone": "Featherstone",
    "mujoco": "MuJoCo",
    "xpbd": "XPBD",
    "vbd": "VBD",
    "semiimplicit": "Semi-Implicit",
    "none": "",
    "": "",
}

UNKNOWN_BACKEND_DISPLAY = "unknown"
"""Header-badge text for an engine the simulation manager could not be asked about.

Short, because it shares a one-line header with the solver badge, which is why
:func:`~isaacsim.robot_setup.gain_tuner.backend_display_label`'s "the active
physics engine" -- correct in a sentence -- is not used here."""


@dataclass
class BackendContext:
    """Describes which physics backend and gain source are currently active.

    ``backend``/``solver``/``solver_type``/``available_backends`` are populated
    from the live app by :meth:`from_app`.  ``gain_source`` and ``save_target``
    remain stubs until the backend gain-source layer is wired.
    """

    backend: str = ""
    """Active physics backend label, as the simulation manager reports it.

    ``"PhysX"`` and ``"NewtonAPI"`` are the two this extension knows the joint
    schema of (:data:`~isaacsim.robot_setup.gain_tuner.SUPPORTED_BACKENDS`), but
    the manager also reports ``"remotesim"``, and :meth:`from_app` returns ``""``
    when the engine cannot be read at all.  Defaulting to ``"PhysX"`` made an
    unread engine indistinguishable from PhysX in every derived label below, so
    the default is the honest empty string and the labels say so."""

    solver: str = ""
    """Active Newton solver display name, e.g. ``"MuJoCo"`` or ``"XPBD"``.

    Empty string when PhysX is active (PhysX has no named solver), and also when
    the Newton solver cannot be determined -- see :attr:`solver_type`."""

    solver_type: str = ""
    """Active Newton solver token (``"mujoco"`` / ``"xpbd"`` / ``"vbd"``).

    This is what selects the joint-schema resolver order, so it is load-bearing
    rather than cosmetic: the MuJoCo solver inserts ``mjc:*`` between ``newton:*``
    and ``physxJoint:*``.  Empty when PhysX is active or the solver is unknown."""

    gain_source: str = "PhysicsDriveAPI"
    """Which gain source is being edited.  One of ``"PhysicsDriveAPI"`` or
    ``"actuator"``."""

    save_target: str = ""
    """Human-readable filename of the resolved payload USDA layer where tuned
    DriveAPI gains will be written, e.g. ``"physics.usda"``.  Empty string when
    the layer cannot be resolved."""

    save_target_identifier: str = ""
    """Layer identifier (real path) of the selected DriveAPI save target, used
    with ``Sdf.Layer.Find`` / save dialogs.  Empty when unresolved."""

    save_target_candidates: list = field(default_factory=list)
    """Selectable DriveAPI save-target layers as ``(identifier, display_name)``
    tuples, neutral ``physics.usda`` first when present."""

    mirror_info: str = ""
    """Informational message describing the additional MuJoCo / Newton targets
    that will be mirrored / written on save.  Empty when there are none."""

    mirror_enabled: bool = True
    """Whether the Newton actuator writeback is enabled.  Default ON so tuned
    actuator gains persist back to the actuator's own layer."""

    mirror_drive_to_mjc: bool = False
    """Whether the opt-in DriveAPI->MuJoCo mirror is enabled.  Default OFF so a
    save never authors ``mjc:*`` actuator prims unless the user opts in."""

    has_mjc_sources: bool = False
    """True when the robot has MuJoCo-native gains that can be mirrored."""

    has_newton_sources: bool = False
    """True when the robot has Newton actuators that can receive writeback."""

    mjc_target_identifier: str = ""
    """Selected layer identifier mjc mirror edits route to (default the
    ``mujoco.usda`` candidate).  Empty when there is no mjc source."""

    newton_target_identifier: str = ""
    """Selected layer identifier Newton writeback routes to (default the actuator
    prim's own defining layer).  Empty when there is no actuator source."""

    is_active: bool = True
    """Whether the currently displayed gain source is the one actively used by
    the simulation controller.  False triggers a read-only/warning state."""

    multiple_sources: bool = False
    """True when both PhysicsDriveAPI and actuator gains are authored on the
    same robot, creating an ambiguous gain source situation."""

    available_backends: list = field(default_factory=lambda: ["PhysX", "NewtonAPI"])
    """Backends available for selection, populated from the engines registered
    with the simulation manager on the running app."""

    # ------------------------------------------------------------------
    # Live detection
    # ------------------------------------------------------------------

    @classmethod
    def active_backend_label(cls) -> str:
        """Return the live active backend label, without touching the stage.

        This is the cheap half of :meth:`from_app`: a single class-attribute read
        inside the simulation manager, with none of the stage traversal the solver
        lookup needs.  It exists because the engine can change with nobody calling
        :meth:`~isaacsim.core.simulation_manager.SimulationManager.switch_physics_engine`
        -- the manager also flips it in response to simulation activate/deactivate
        events -- and there is no event to subscribe to, so callers that must not
        act on a stale backend (anything that authors a schema) re-read it here
        instead of trusting a cached context.

        Returns:
            ``"PhysX"`` or ``"NewtonAPI"`` for the two engines whose joint schema
            this extension knows, the manager's own name for any other engine it
            reports (``"remotesim"``), and ``""`` when the query fails or comes
            back empty.  Neither of the latter two is reported as PhysX: doing so
            turned a transient query failure into a writable PhysX state, and an
            edit authored on ``physxJoint:*`` for an engine that may not read it is
            a silent no-op.  Callers that author a schema must check
            :func:`~isaacsim.robot_setup.gain_tuner.backend_supported` first.

        Example:

        .. code-block:: python

            >>> from isaacsim.robot_setup.gain_tuner.ui.backend_context import BackendContext

            >>> BackendContext.active_backend_label()  # doctest: +NO_CHECK
            'PhysX'
        """
        try:
            from isaacsim.core.simulation_manager import SimulationManager

            active = (SimulationManager.get_active_physics_engine() or "").lower()
        except Exception:
            return ""
        if not active:
            return ""
        return _ENGINE_TO_BACKEND.get(active, active)

    @classmethod
    def from_app(cls, stage: object = None) -> BackendContext:
        """Build a context from the live Isaac Sim physics backend.

        Detection is defensive but not optimistic: when the simulation manager is
        unavailable the backend reads as empty rather than as PhysX, matching
        :meth:`active_backend_label` so the two never disagree and send the panel
        into a rebuild loop.  The rest of the UI still renders; only the per-backend
        joint parameters, which have to know the engine to author anything, go
        read-only.

        Args:
            stage: Stage used to resolve the Newton solver before Newton has
                initialized.  Newton's live configuration only names a solver once
                it has initialized, so without a stage the solver reads as unknown
                until the first play.

        Returns:
            A context whose ``backend``/``solver``/``solver_type``/
            ``available_backends`` reflect the active engine.  Other fields keep
            their stub defaults.
        """
        backend = ""
        solver_type = ""
        available = ["PhysX", "NewtonAPI"]

        try:
            from isaacsim.core.simulation_manager import SimulationManager

            active = (SimulationManager.get_active_physics_engine() or "").lower()
            if active:
                backend = _ENGINE_TO_BACKEND.get(active, active)
            engines = SimulationManager.get_available_physics_engines() or []
            if engines:
                available = [_ENGINE_TO_BACKEND.get(str(name).lower(), name) for name, _active in engines]
        except Exception:
            pass

        if backend == "NewtonAPI":
            try:
                solver_type = gain_tuner.newton_solver_type(stage)
            except Exception:
                solver_type = ""

        return cls(
            backend=backend,
            solver=_SOLVER_PRETTY.get(solver_type, solver_type),
            solver_type=solver_type,
            available_backends=available,
        )

    # ------------------------------------------------------------------
    # Derived helpers (read-only, computed from the fields above)
    # ------------------------------------------------------------------

    @property
    def show_solver(self) -> bool:
        """True when a solver label should be shown (Newton only)."""
        return bool(self.solver) and self.backend != "PhysX"

    @property
    def solver_label(self) -> str:
        """Formatted solver label, e.g. ``"Solver: Featherstone"``."""
        if self.show_solver:
            return f"Solver: {self.solver}"
        return ""

    @property
    def solver_display(self) -> str:
        """Info-tag text naming the running solver.

        ``"PhysX"`` for the PhysX backend; ``"Newton <Solver>"`` for Newton
        (e.g. ``"Newton Featherstone"``, ``"Newton MuJoCo"``), falling back to
        ``"Newton"`` when the solver type cannot be determined; and
        :attr:`backend_display` for any other engine, which has no solver this
        extension can name.
        """
        if self.backend == "NewtonAPI":
            return f"Newton {self.solver}".strip() if self.solver else "Newton"
        if self.backend == "PhysX":
            return "PhysX"
        return self.backend_display

    @property
    def backend_display(self) -> str:
        """Info-tag text naming the active backend.

        ``"Newton"`` / ``"PhysX"`` for the two engines whose joint schema this
        extension knows, the manager's own label (e.g. ``"remotesim"``) for any
        other engine, and :data:`UNKNOWN_BACKEND_DISPLAY` when the engine could not
        be read.  Naming PhysX for either of the last two put this badge in direct
        contradiction with the joint fields below it, which disable themselves
        saying the active engine is not one the panel can author for.
        """
        if gain_tuner.backend_supported(self.backend):
            return gain_tuner.backend_display_label(self.backend)
        return self.backend or UNKNOWN_BACKEND_DISPLAY

    @property
    def solver_short(self) -> str:
        """Solver name only, no backend prefix.

        ``"Featherstone"`` / ``"MuJoCo"`` / ... for Newton (``"-"`` when the solver
        type cannot be determined), ``"PhysX"`` for the PhysX backend, and ``"-"``
        for any other engine, whose solver this extension cannot enumerate.
        """
        if self.backend == "NewtonAPI":
            return self.solver or "-"
        if self.backend == "PhysX":
            return "PhysX"
        return "-"

    @property
    def is_mujoco_solver(self) -> bool:
        """True when the active Newton solver is the MuJoCo solver.

        MuJoCo-native (``mjc:*``) gains become the active, editable source only in
        this state; otherwise they are shown read-only for inspection.
        """
        return self.backend == "NewtonAPI" and self.solver_type == gain_tuner.SOLVER_MUJOCO

    @property
    def solver_known(self) -> bool:
        """True when the joint-schema resolver order can be determined.

        Always True under PhysX, which has a single chain.  Under Newton it
        requires a resolved solver, because the MuJoCo chain differs from the
        XPBD / VBD one; when it is False the UI must state that the effective
        value of an advanced joint parameter is undeterminable.
        """
        if self.backend != "NewtonAPI":
            return True
        return bool(gain_tuner.resolver_chain(self.backend, self.solver_type))

    @property
    def show_ki(self) -> bool:
        """True when a Ki field may be relevant (actuator context only).

        Ki exists only for PID actuators (``NewtonPIDControlAPI``); PD and neural
        controllers have none.  This is a coarse, robot-level hint — the per-joint
        editor additionally gates the Ki field on ``ActuatorGains.is_pid`` (via the
        resolved gains' ``ki`` / ``ki_label``), so a PD-only joint shows no Ki even
        when this is True.
        """
        return self.gain_source == "actuator"

    @property
    def source_label(self) -> str:
        """Short badge label naming the gain source the active engine consumes.

        ``"Newton"`` under Newton and ``"Physics"`` (the ``PhysicsDriveAPI``) under
        PhysX.  Empty for any other engine: which source it reads is exactly what
        is unknown about it, so naming one would be a guess.

        Note this is not what the gain table's Source column renders -- that is
        per-row and comes from
        :attr:`~isaacsim.robot_setup.gain_tuner.ResolvedGains.source_label`.
        """
        if self.backend == "NewtonAPI":
            return "Newton"
        if self.backend == "PhysX":
            return "Physics"
        return ""

    @property
    def save_target_resolved(self) -> bool:
        """True when a save target has been resolved."""
        return bool(self.save_target)

    @property
    def has_mirror_info(self) -> bool:
        """True when there is MuJoCo / Newton mirror information to surface."""
        return bool(self.mirror_info)

    @property
    def has_mirror_sources(self) -> bool:
        """True when the robot has any non-DriveAPI gain source (mjc / Newton).

        Controls whether the mirror toggle and target pickers are shown.
        """
        return self.has_mjc_sources or self.has_newton_sources

    # ------------------------------------------------------------------
    # Save-target / mirror population
    # ------------------------------------------------------------------

    def apply_save_targets(self, targets: object) -> None:
        """Populate the save-target and mirror fields from resolved write targets.

        Args:
            targets: A ``GainWriteTargets`` (from
                ``isaacsim.robot_setup.gain_tuner.resolve_gain_write_targets``)
                describing the DriveAPI save-target options and any MuJoCo /
                Newton mirror sources.
        """
        options = getattr(targets, "options", None)
        default = options.default if options is not None else None
        if default is not None:
            self.save_target = default.display_name
            self.save_target_identifier = default.identifier
            self.save_target_candidates = [(c.identifier, c.display_name) for c in options.candidates]
        else:
            self.save_target = ""
            self.save_target_identifier = ""
            self.save_target_candidates = []
        self.mirror_enabled = bool(getattr(targets, "mirror_enabled", True))
        self.mirror_drive_to_mjc = bool(getattr(targets, "mirror_drive_to_mjc", False))
        self.has_mjc_sources = bool(getattr(targets, "has_mjc_mirror", False))
        self.has_newton_sources = bool(getattr(targets, "has_newton_writeback", False))
        self.mjc_target_identifier = getattr(targets, "mjc_target_identifier", None) or ""
        self.newton_target_identifier = getattr(targets, "newton_target_identifier", None) or ""
        self.mirror_info = self._format_mirror_info(targets, self.is_mujoco_solver)

    @staticmethod
    def _format_mirror_info(targets: object, mujoco_solver_active: bool = False) -> str:
        """Build the human-readable mirror-info string for the save row.

        MuJoCo (``mjc:*``) mirroring is opt-in and OFF by default: when it is off
        (and the MuJoCo solver is not the one editing mjc directly) the row states
        that mjc:* will not be written.  When on, it names the mirror target.  The
        Newton actuator writeback (default on) is reported whenever actuators are
        present.
        """
        has_mjc = bool(getattr(targets, "has_mjc_mirror", False))
        has_newton = bool(getattr(targets, "has_newton_writeback", False))
        if not (has_mjc or has_newton):
            return ""
        parts: list[str] = []
        if has_mjc:
            if mujoco_solver_active:
                parts.append("MuJoCo gains saved directly (active source)")
            elif bool(getattr(targets, "will_mirror_mjc", False)):
                names = ", ".join(getattr(targets, "mjc_layer_names", []) or ["mujoco.usda"])
                parts.append(f"tuned DriveAPI gains mirrored to MuJoCo ({names})")
            else:
                parts.append("MuJoCo mirror off (default): mjc:* gains will not be written")
        if has_newton and bool(getattr(targets, "will_write_newton", True)):
            names = ", ".join(getattr(targets, "newton_layer_names", []) or ["Newton actuators"])
            parts.append(f"Newton actuator gains written to {names}")
        if not parts:
            return ""
        return "On save: " + "; ".join(parts) + "."
