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

"""Per-backend access to the advanced joint parameters the gain tuner tunes.

Armature, joint friction, and the joint velocity limit are authored on more than
one joint API schema, and each physics backend reads its own.  This module reads
the value the *active* backend resolves and writes only the *active* backend's
schema, leaving the other backend's tuning alone.

The two backends are deliberately tuned to different values, though the reason
differs per parameter.  PhysX's ``jointFriction`` is a dimensionless coefficient
whose ceiling scales with joint load, while ``newton:friction`` is a
load-independent effort, so the two are not convertible at all.  Armature shares
units across both schemas, but doubles as a solver-specific numerical stabiliser,
so its useful value is solver-contingent.  The velocity limit is directly
comparable -- both are degrees per second for angular DOFs -- and diverges only by
authoring choice.  PhysX's own resolution precedence is also the reverse of
Newton's, so a doubly-authored joint is meant to resolve differently per backend.

Resolution chain
----------------

Under PhysX the value comes from ``physxJoint:*``.

Under the Newton backend it comes from the first *authored* value in the
schema-resolver order Newton itself uses, which depends on the running solver
(``isaacsim.physics.newton.impl.newton_stage``):

* MuJoCo solver: ``newton:*``, then ``mjc:*``, then ``physxJoint:*``.
* XPBD / VBD solvers: ``newton:*``, then ``physxJoint:*``.

Two asymmetries fall out of Newton's resolver mappings and are modelled here:

* ``SchemaResolverPhysx`` declares no ``friction`` key, so ``newton:friction``
  has **no PhysX fallback**.  A joint whose only friction opinion is
  ``physxJoint:jointFriction`` is simulated by Newton with no friction at all.
* ``SchemaResolverMjc`` declares no ``velocity_limit`` key, so ``mjc:*`` never
  takes part in resolving the velocity limit even under the MuJoCo solver.

Only explicitly authored values take part: Newton reads through
``HasAuthoredValue``, so merely applying an API schema does not pin a value at
its schema fallback.  When nothing in the chain is authored the engine applies
its own default, which this module reports as "unauthored" rather than guessing a
number.

A ``newton:velocityLimit`` of ``inf`` means "unlimited".  Newton's importer maps
the resolved ``inf`` to its own unlimited default *after* resolution, so an
authored ``inf`` wins the chain and does **not** fall back to
``physxJoint:maxJointVelocity``.

The gain tuner never authors ``mjc:*``: MuJoCo gains are authored by hand (see
the Newton robot setup tips), so ``mjc:*`` is read for reporting only.

When the running solver cannot be determined, resolution falls back to the
schemas every possible chain agrees on rather than assuming one of them.
``newton:*`` leads under every solver, so a value authored there is in effect
regardless; only when nothing is authored that far does the answer depend on the
solver, and :attr:`JointParamResolution.determined` is then False so callers
state that instead of naming a number.

Units
-----

Both joint schemas store the velocity limit in degrees per second for angular
DOFs; Newton converts to radians per second on import.  Armature and friction are
stored and consumed unscaled.

``NewtonJointAPI`` is a codeless schema, so it has no ``pxr`` Python module and
is applied and read through the string API name and raw attribute tokens.  Its
registration comes from ``omni.usd.schema.newton``; without it ``ApplyAPI``
fails, so :func:`author_joint_param` returns no attributes and callers must
surface that rather than treat it as success.

This module touches only USD, with no Kit or ``omni.ui`` imports, so it can be
unit-tested against in-memory stages without a running app.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pxr
from pxr import Sdf, Usd

# USD access can fail with these when a prim or attribute is missing, invalid, a
# duck-typed fake (unit tests substitute plain objects), or holds an unexpected
# value type. Caught at the defensive USD boundaries below so that any
# *unexpected* error type still propagates.
_USD_ACCESS_ERRORS = (AttributeError, TypeError, ValueError, pxr.Tf.ErrorException)

NEWTON_JOINT_API = "NewtonJointAPI"
"""Applied-schema name of the codeless Newton joint API."""

PHYSX_JOINT_API = "PhysxJointAPI"
"""Applied-schema name of the PhysX joint API."""

SCHEMA_NEWTON = "newton"
"""Resolver token for the engine-neutral ``newton:*`` joint schema."""

SCHEMA_MJC = "mjc"
"""Resolver token for the MuJoCo ``mjc:*`` joint schema."""

SCHEMA_PHYSX = "physx"
"""Resolver token for the ``physxJoint:*`` joint schema."""

BACKEND_NEWTON = "NewtonAPI"
"""Active-backend label for the Newton backend."""

BACKEND_PHYSX = "PhysX"
"""Active-backend label for the PhysX backend."""

SUPPORTED_BACKENDS: tuple[str, ...] = (BACKEND_PHYSX, BACKEND_NEWTON)
"""The backends whose joint schema this module knows how to read and write.

The simulation manager can report engines beyond these two (``remotesim``), and a
failed engine query reports nothing at all.  Neither has a known joint schema, so
:func:`backend_write_schema` refuses them rather than guessing ``physxJoint:*``:
an edit written to a schema the running engine does not read is exactly the silent
no-op this module exists to prevent."""

BACKEND_DISPLAY_LABELS = {
    BACKEND_NEWTON: "Newton",
    BACKEND_PHYSX: "PhysX",
}
"""User-facing name per backend label.

:data:`BACKEND_NEWTON` is the simulation manager's internal token, not a name to
put in front of a user; ASCII only, since ``omni.ui`` renders non-ASCII as ``?``."""

SOLVER_MUJOCO = "mujoco"
"""Newton MuJoCo solver token."""

SOLVER_XPBD = "xpbd"
"""Newton XPBD solver token."""

SOLVER_VBD = "vbd"
"""Newton VBD solver token."""

NEWTON_SOLVER_TYPES: tuple[str, ...] = (SOLVER_MUJOCO, SOLVER_XPBD, SOLVER_VBD)
"""Newton solver tokens whose resolver order is known."""

SCHEMA_LABELS = {
    SCHEMA_NEWTON: "newton:*",
    SCHEMA_MJC: "mjc:*",
    SCHEMA_PHYSX: "physxJoint:*",
}
"""Short, ASCII display label per resolver token."""

SCHEMA_JOINT_APIS = {
    SCHEMA_NEWTON: NEWTON_JOINT_API,
    SCHEMA_PHYSX: PHYSX_JOINT_API,
}
"""The API schema each writable resolver token is authored through.

``mjc:*`` is absent on purpose: the gain tuner reports MuJoCo opinions but never
writes them."""

# Schema-resolver order per Newton solver, mirroring the resolver lists in
# ``isaacsim.physics.newton.impl.newton_stage``.  A solver absent from this map
# leaves the chain undetermined rather than defaulting to one of the two.
#
# These three are the complete set Isaac Sim's Newton integration can run:
# ``newton_solver_to_solver_object`` and ``newton_solver_to_api_schema`` in
# ``isaacsim.physics.newton.impl.utils`` register exactly ``mujoco`` / ``xpbd`` /
# ``vbd``, and both the live-config and stage solver lookups in
# :mod:`~isaacsim.robot_setup.gain_tuner.gain_sources` are filtered through them.
# Newton itself also defines Featherstone and Semi-Implicit solvers, which take
# ``newton.ModelBuilder.add_usd``'s default ``[SchemaResolverNewton()]`` -- a
# ``newton:*``-only chain -- but Isaac Sim cannot select them, so a stage never
# reports one and no chain is needed for them here.
_NEWTON_SOLVER_CHAINS = {
    SOLVER_MUJOCO: (SCHEMA_NEWTON, SCHEMA_MJC, SCHEMA_PHYSX),
    SOLVER_XPBD: (SCHEMA_NEWTON, SCHEMA_PHYSX),
    SOLVER_VBD: (SCHEMA_NEWTON, SCHEMA_PHYSX),
}

NEWTON_DEFAULT_ARMATURE = 0.1
"""The armature Newton applies to a joint that authors none.

``NewtonConfig.armature`` in ``isaacsim.physics.newton``, copied onto
``newton.ModelBuilder.default_joint_cfg.armature`` in ``NewtonStage`` before the
stage is parsed.  It is not zero, so a joint with no authored armature is still
simulated with rotor inertia under Newton.

Note that the 0.1 is Isaac Sim's, not Newton's: ``JointDofConfig.armature``
defaults to 0.0 upstream, and ``isaacsim.physics.newton`` overrides it.  A copy of
a number this extension does not own is a claim that can silently go stale, so
``TestTheNewtonDefaultIsStillNewtons`` pins it against the config it mirrors."""


@dataclass(frozen=True)
class JointParamSpec:
    """One advanced joint parameter and the schemas that carry it."""

    key: str
    """Stable identifier used as the gain-table column key."""

    label: str
    """User-facing label for the column header and detail-panel field."""

    newton_attr: str
    """``newton:*`` attribute name on ``NewtonJointAPI``."""

    physx_attr: str
    """``physxJoint:*`` attribute name on ``PhysxJointAPI``."""

    mjc_attr: str | None = None
    """``mjc:*`` attribute name, or None when the MuJoCo resolver has no such key.

    None for the velocity limit, which ``SchemaResolverMjc`` does not declare, so
    ``mjc:*`` stays out of the chain even under the MuJoCo solver."""

    newton_reads_physx: bool = True
    """Whether the Newton backend falls back to :attr:`physx_attr`.

    False for joint friction: ``SchemaResolverPhysx`` declares no ``friction``
    key, so ``physxJoint:jointFriction`` is never read under Newton."""

    unauthored_sentinel: float | None = None
    """Resolved value that means "no limit", such as an infinite velocity limit.

    Newton's importer maps this to the engine's own unlimited default *after*
    resolution, so an authored sentinel wins the chain rather than falling
    through to the next schema."""

    newton_default: float = 0.0
    """Value the Newton backend simulates when nothing in its chain is authored.

    Newton resolves through ``HasAuthoredValue``, so the ``newton:*`` schema
    fallback never applies and this is the ``ModelBuilder`` default instead --
    which for armature is :data:`NEWTON_DEFAULT_ARMATURE`, not zero."""

    physx_default: float = 0.0
    """Value PhysX simulates when nothing in its chain is authored.

    The ``PhysxJointAPI`` schema fallback: zero for armature and friction,
    ``inf`` for the velocity limit."""

    def engine_default(self, backend: str) -> float | None:
        """Return the value ``backend`` simulates when nothing is authored.

        This is what an unauthored parameter is worth in the running simulation,
        which is the only honest thing to show for a cell nothing has been
        authored into.  Zero is that value for friction under both backends, but
        not for armature under Newton and not for the velocity limit under
        either.

        Args:
            backend: The active backend label.

        Returns:
            The engine's own default, ``math.inf`` when the engine enforces no
            limit, or None when ``backend`` is not one of
            :data:`SUPPORTED_BACKENDS` and therefore has no known default.

        Example:

        .. code-block:: python

            >>> from isaacsim.robot_setup.gain_tuner import joint_param_spec

            >>> joint_param_spec("armature").engine_default("NewtonAPI")
            0.1
        """
        if newton_backend_selected(backend):
            return self.newton_default
        if str(backend) == BACKEND_PHYSX:
            return self.physx_default
        return None

    def attr_for_schema(self, schema: str) -> str | None:
        """Return the attribute name this parameter uses on ``schema``.

        Args:
            schema: One of :data:`SCHEMA_NEWTON`, :data:`SCHEMA_MJC`,
                :data:`SCHEMA_PHYSX`.

        Returns:
            The attribute name, or None when the parameter does not exist on that
            schema.

        Example:

        .. code-block:: python

            >>> from isaacsim.robot_setup.gain_tuner import SCHEMA_MJC, joint_param_spec

            >>> joint_param_spec("joint_friction").attr_for_schema(SCHEMA_MJC)
            'mjc:frictionloss'
        """
        if schema == SCHEMA_NEWTON:
            return self.newton_attr
        if schema == SCHEMA_PHYSX:
            return self.physx_attr
        if schema == SCHEMA_MJC:
            return self.mjc_attr
        return None

    @property
    def schemas(self) -> tuple[str, ...]:
        """Every schema this parameter exists on, in resolver order."""
        return tuple(s for s in (SCHEMA_NEWTON, SCHEMA_MJC, SCHEMA_PHYSX) if self.attr_for_schema(s) is not None)


JOINT_PARAM_SPECS: tuple[JointParamSpec, ...] = (
    JointParamSpec(
        "armature",
        "Armature",
        "newton:armature",
        "physxJoint:armature",
        mjc_attr="mjc:armature",
        newton_default=NEWTON_DEFAULT_ARMATURE,
    ),
    JointParamSpec(
        "joint_friction",
        "Joint Friction",
        "newton:friction",
        "physxJoint:jointFriction",
        mjc_attr="mjc:frictionloss",
        newton_reads_physx=False,
    ),
    JointParamSpec(
        "max_joint_velocity",
        "Max Joint Velocity",
        "newton:velocityLimit",
        "physxJoint:maxJointVelocity",
        unauthored_sentinel=math.inf,
        newton_default=math.inf,
        physx_default=math.inf,
    ),
)
"""The advanced joint parameters the gain tuner reads and writes per backend."""


def joint_param_spec(key: str) -> JointParamSpec | None:
    """Return the spec registered under ``key``, or None when unknown.

    Args:
        key: The :attr:`JointParamSpec.key` to look up.

    Returns:
        The matching spec, or None.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import joint_param_spec

        >>> joint_param_spec("armature").newton_attr
        'newton:armature'
    """
    for spec in JOINT_PARAM_SPECS:
        if spec.key == key:
            return spec
    return None


def newton_backend_selected(backend: str) -> bool:
    """Return whether ``backend`` names the Newton backend.

    Args:
        backend: A backend label such as ``"PhysX"`` or ``"NewtonAPI"``.

    Returns:
        True for :data:`BACKEND_NEWTON`, False otherwise.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import newton_backend_selected

        >>> newton_backend_selected("NewtonAPI")
        True
    """
    return str(backend) == BACKEND_NEWTON


def backend_supported(backend: str) -> bool:
    """Return whether ``backend`` is a backend whose joint schema is known.

    The simulation manager reports ``remotesim`` alongside ``physx`` and
    ``newton``, and a failed engine query reports nothing at all.  Neither has a
    joint schema this module can read or write, so callers must check this before
    authoring anything and disable the edit when it is False.

    Args:
        backend: The active backend label.

    Returns:
        True for :data:`SUPPORTED_BACKENDS`, False for anything else including
        the empty string.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import backend_supported

        >>> backend_supported("remotesim")
        False
    """
    return str(backend) in SUPPORTED_BACKENDS


def backend_display_label(backend: str) -> str:
    """Return the user-facing name for ``backend``.

    Args:
        backend: The active backend label.

    Returns:
        The name from :data:`BACKEND_DISPLAY_LABELS`, or ``"the active physics
        engine"`` for a backend with no known name, which reads correctly in a
        sentence where a made-up product name would not.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import backend_display_label

        >>> backend_display_label("NewtonAPI")
        'Newton'
    """
    return BACKEND_DISPLAY_LABELS.get(str(backend), "the active physics engine")


def backend_reads_usd_while_playing(backend: str) -> bool:
    """Return whether ``backend`` picks up a joint-param edit made mid-run.

    The two backends differ, and the difference is not documented anywhere, so it
    was measured (see the ``TestLiveUsdWriteReachesTheEngine`` tests): authoring
    ``physxJoint:armature`` or ``physxJoint:maxJointVelocity`` while the timeline
    plays changes what PhysX simulates on the next step, but the equivalent
    ``newton:*`` edit does not reach Newton.  Newton builds its ``Model`` once, in
    ``ModelBuilder.add_usd`` at initialization, and ``isaacsim.physics.newton``
    registers no USD notice handler, so nothing re-reads the stage until the next
    play.

    Callers authoring one of the advanced joint params while playing should say
    that the value will not take effect until a replay when this is False.  It
    describes only the USD-write path: the tensor views stay live under both
    backends.

    Args:
        backend: The active backend label.

    Returns:
        True under PhysX, False under Newton and under any backend whose
        behaviour is unknown, so an unrecognized engine is warned about rather
        than assumed to be live.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import backend_reads_usd_while_playing

        >>> backend_reads_usd_while_playing("NewtonAPI")
        False
    """
    return str(backend) == BACKEND_PHYSX


def backend_write_schema(backend: str) -> str | None:
    """Return the schema an edit under ``backend`` is authored on.

    Args:
        backend: The active backend label.

    Returns:
        :data:`SCHEMA_NEWTON` under the Newton backend, :data:`SCHEMA_PHYSX`
        under PhysX, and None for any other backend -- an unrecognised engine, or
        an engine query that came back empty.  Never :data:`SCHEMA_MJC`.  None is
        a refusal, not a default\\: writing ``physxJoint:*`` for an engine that
        may not read it is a silent no-op, so callers must surface it instead.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import backend_write_schema

        >>> backend_write_schema("PhysX")
        'physx'
        >>> backend_write_schema("remotesim") is None
        True
    """
    if newton_backend_selected(backend):
        return SCHEMA_NEWTON
    if str(backend) == BACKEND_PHYSX:
        return SCHEMA_PHYSX
    return None


def other_backend(backend: str) -> str | None:
    """Return the backend that is not ``backend``.

    Args:
        backend: The active backend label.

    Returns:
        :data:`BACKEND_PHYSX` under Newton, :data:`BACKEND_NEWTON` under PhysX,
        and None for an unsupported backend, which has no opposite to copy to.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import other_backend

        >>> other_backend("NewtonAPI")
        'PhysX'
    """
    if newton_backend_selected(backend):
        return BACKEND_PHYSX
    if str(backend) == BACKEND_PHYSX:
        return BACKEND_NEWTON
    return None


def resolver_chain(backend: str, solver: str = "") -> tuple[str, ...] | None:
    """Return the schema-resolver order the active backend uses.

    Args:
        backend: The active backend label.
        solver: The running Newton solver token (``"mujoco"``, ``"xpbd"``,
            ``"vbd"``).  Ignored under PhysX.

    Returns:
        The resolver tokens in priority order, or None when the chain cannot be
        determined -- either the backend is Newton and the solver is unknown (the
        MuJoCo chain differs from the XPBD / VBD one), or the backend is not one
        of :data:`SUPPORTED_BACKENDS` and reads no schema this module knows.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import resolver_chain

        >>> resolver_chain("NewtonAPI", "mujoco")
        ('newton', 'mjc', 'physx')
        >>> resolver_chain("NewtonAPI", "") is None
        True
    """
    if newton_backend_selected(backend):
        return _NEWTON_SOLVER_CHAINS.get(str(solver).strip().lower())
    if str(backend) == BACKEND_PHYSX:
        return (SCHEMA_PHYSX,)
    return None


def _filter_chain(spec: JointParamSpec, chain: tuple[str, ...], newton_active: bool) -> tuple[str, ...]:
    """Drop the schemas of ``chain`` that do not carry ``spec``."""
    order: list[str] = []
    for schema in chain:
        if spec.attr_for_schema(schema) is None:
            continue
        if newton_active and schema == SCHEMA_PHYSX and not spec.newton_reads_physx:
            continue
        order.append(schema)
    return tuple(order)


def param_resolver_chain(spec: JointParamSpec, backend: str, solver: str = "") -> tuple[str, ...] | None:
    """Return the resolver order for one parameter, dropping schemas that lack it.

    Args:
        spec: The parameter whose chain is wanted.
        backend: The active backend label.
        solver: The running Newton solver token.  Ignored under PhysX.

    Returns:
        The resolver tokens that actually carry this parameter, in priority order,
        or None when the chain cannot be determined.  Joint friction under Newton
        drops :data:`SCHEMA_PHYSX`; the velocity limit drops :data:`SCHEMA_MJC`.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import joint_param_spec, param_resolver_chain

        >>> param_resolver_chain(joint_param_spec("joint_friction"), "NewtonAPI", "xpbd")
        ('newton',)
    """
    chain = resolver_chain(backend, solver)
    if chain is None:
        return None
    return _filter_chain(spec, chain, newton_backend_selected(backend))


def candidate_param_chains(spec: JointParamSpec, backend: str, solver: str = "") -> tuple[tuple[str, ...], ...]:
    """Return every resolver order the parameter could be resolved through.

    One entry when the chain is known, and one per possible Newton solver when it
    is not.  Callers use this to separate what is uncertain (which of several
    chains applies) from what is certain regardless (their common prefix, and the
    schemas no candidate reads at all).

    Args:
        spec: The parameter whose chains are wanted.
        backend: The active backend label.
        solver: The running Newton solver token.

    Returns:
        The candidate resolver orders, deduplicated, in solver-registry order.
        Empty for a backend outside :data:`SUPPORTED_BACKENDS`, which reads no
        candidate chain at all rather than an uncertain one.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import candidate_param_chains, joint_param_spec

        >>> candidate_param_chains(joint_param_spec("armature"), "NewtonAPI", "")
        (('newton', 'mjc', 'physx'), ('newton', 'physx'))
    """
    known = param_resolver_chain(spec, backend, solver)
    if known is not None:
        return (known,)
    newton_active = newton_backend_selected(backend)
    if not newton_active:
        return ()
    candidates: list[tuple[str, ...]] = []
    for solver_token in NEWTON_SOLVER_TYPES:
        chain = _filter_chain(spec, _NEWTON_SOLVER_CHAINS[solver_token], newton_active)
        if chain not in candidates:
            candidates.append(chain)
    return tuple(candidates)


def _common_prefix(chains: tuple[tuple[str, ...], ...]) -> tuple[str, ...]:
    """Return the leading schemas every chain in ``chains`` agrees on."""
    if not chains:
        return ()
    prefix: list[str] = []
    for schemas in zip(*chains):
        if len(set(schemas)) != 1:
            break
        prefix.append(schemas[0])
    return tuple(prefix)


@dataclass(frozen=True)
class JointParamResolution:
    """What one advanced joint parameter resolves to under the active backend.

    Produced by :func:`resolve_joint_param`.  Carries the authored opinion of
    every schema the parameter exists on -- not just the winning one -- so a
    caller can state each backend's value, which one is in effect, and which
    authored values the active backend never reads.
    """

    spec: JointParamSpec
    """The parameter this resolution describes."""

    backend: str
    """The backend label the chain was built for."""

    solver: str
    """The Newton solver token the chain was built for (empty under PhysX)."""

    chain: tuple[str, ...]
    """Resolver order actually consulted.

    The parameter's full resolver order when the solver is known.  When it is not,
    the leading schemas every possible Newton chain agrees on -- ``newton:*`` is
    first under every solver, so an authored ``newton:*`` value wins no matter
    which solver is running, and only what lies past the prefix is uncertain."""

    candidate_chains: tuple
    """Every resolver order the parameter could be resolved through: one entry when
    the solver is known, one per possible solver when it is not."""

    chain_complete: bool
    """True when :attr:`chain` is the parameter's full resolver order, i.e. the
    running solver is known."""

    authored: dict
    """Map of resolver token -> explicitly authored value (None when unauthored),
    for every schema the parameter exists on."""

    effective_schema: str | None
    """Resolver token that supplied the winning authored value, or None when
    nothing in the consulted chain is authored."""

    effective_value: float | None
    """The value the active backend consumes, or None when nothing in the consulted
    chain is authored or the resolved value is the "no limit" sentinel.
    Distinguish those with :attr:`determined` and :attr:`unlimited`."""

    unlimited: bool
    """True when the chain resolved to the "no limit" sentinel, so the engine
    applies its own unlimited default instead of a limit."""

    @property
    def determined(self) -> bool:
        """True when the effective value follows from what is known.

        False only when the running solver is unknown *and* the answer depends on
        it: with nothing authored on the schemas every candidate chain agrees on,
        which of the remaining schemas is read is a function of the solver."""
        return self.chain_complete or self.effective_schema is not None

    @property
    def newton_value(self) -> float | None:
        """The value authored on ``newton:*``, or None when unauthored."""
        return self.authored.get(SCHEMA_NEWTON)

    @property
    def physx_value(self) -> float | None:
        """The value authored on ``physxJoint:*``, or None when unauthored."""
        return self.authored.get(SCHEMA_PHYSX)

    @property
    def mjc_value(self) -> float | None:
        """The value authored on ``mjc:*``, or None when unauthored / absent."""
        return self.authored.get(SCHEMA_MJC)

    @property
    def backend_supported(self) -> bool:
        """True when the backend's joint schema is known, so an edit can land."""
        return backend_supported(self.backend)

    @property
    def engine_default(self) -> float | None:
        """The value the backend simulates with nothing authored.

        ``math.inf`` means the engine enforces no limit; None means the backend is
        unsupported, so its default is unknown."""
        return self.spec.engine_default(self.backend)

    @property
    def backend_label(self) -> str:
        """The active backend's user-facing name, e.g. ``"Newton"``."""
        return backend_display_label(self.backend)

    @property
    def write_schema(self) -> str | None:
        """Resolver token an edit is authored on (never :data:`SCHEMA_MJC`).

        None when the backend is unsupported and no edit can be authored."""
        return backend_write_schema(self.backend)

    @property
    def write_attr(self) -> str | None:
        """Attribute name an edit is authored to, or None when there is none."""
        schema = self.write_schema
        return None if schema is None else self.spec.attr_for_schema(schema)

    @property
    def other_schema(self) -> str | None:
        """Resolver token of the backend that is not active, or None."""
        other = other_backend(self.backend)
        return None if other is None else backend_write_schema(other)

    @property
    def other_value(self) -> float | None:
        """The other backend's authored value, or None when unauthored."""
        schema = self.other_schema
        return None if schema is None else self.authored.get(schema)

    @property
    def copy_value(self) -> float | None:
        """The value a copy to the other backend should author.

        :attr:`effective_value` except for a limit that is unlimited, which has no
        finite value but is not therefore uncopyable: ``inf`` is exactly what the
        other schema wants in order to mean the same thing.  Reporting it as
        uncopyable is what let the copy claim the other backend already agreed.

        None when there is genuinely nothing to copy: nothing authored anywhere, an
        undetermined resolver chain, or an unsupported backend.
        """
        if self.unlimited:
            return math.inf
        return self.effective_value

    @property
    def other_effective_value(self) -> float | None:
        """What the other backend simulates for this parameter.

        Its authored value, or its own engine default when it has authored none --
        an unauthored parameter is not worth nothing, so comparing a copy against
        the raw authored value alone reports a difference that copying cannot
        settle (writing ``inf`` onto a backend that is already unlimited) and
        misses one it can (Newton's armature default is not PhysX's).

        None when the other backend is unknown and has no default to name.
        """
        other = other_backend(self.backend)
        if other is None:
            return None
        authored = self.other_value
        return authored if authored is not None else self.spec.engine_default(other)

    @property
    def other_backend_agrees(self) -> bool:
        """True when the other backend already simulates :attr:`copy_value`.

        NaN never agrees with itself, so it is reported as a difference no more than
        :attr:`backends_diverge` does.
        """
        value = self.copy_value
        other = self.other_effective_value
        if value is None or other is None:
            return False
        if math.isnan(value) or math.isnan(other):
            return False
        return value == other

    @property
    def backends_diverge(self) -> bool:
        """True when the Newton and PhysX halves both hold different values.

        NaN on either side is not a divergence: it compares unequal to itself, so
        it would report a difference that no edit could settle.
        """
        newton_value = self.newton_value
        physx_value = self.physx_value
        if newton_value is None or physx_value is None:
            return False
        if math.isnan(newton_value) or math.isnan(physx_value):
            return False
        return newton_value != physx_value

    @property
    def resolved_from_fallback_schema(self) -> bool:
        """True when the value in effect comes from a schema an edit would not write.

        The active backend authors :attr:`write_schema` but resolves through a
        chain, so a joint that authors only the *other* backend's half is
        simulated from it -- Newton reading ``physxJoint:armature`` because
        ``newton:armature`` is unauthored.  Nothing about the number says so, and
        the first edit moves the value onto :attr:`write_schema`, which then wins
        the chain, so this is the case most worth stating.
        """
        schema = self.write_schema
        if schema is None or self.effective_schema is None:
            return False
        return self.effective_schema != schema

    @property
    def unread_schemas(self) -> tuple[str, ...]:
        """Schemas holding an authored value that no candidate chain ever reads.

        Under Newton this is where a PhysX-only friction value shows up: it is
        authored, it is in no chain under any solver, and nothing simulates it.
        Reported only for schemas every candidate rules out, so an unknown solver
        never turns a merely-uncertain value into an ignored one.
        """
        readable = {schema for chain in self.candidate_chains for schema in chain}
        return tuple(s for s, v in self.authored.items() if v is not None and s not in readable)

    @property
    def shadowed_schemas(self) -> tuple[str, ...]:
        """Schemas in the chain whose authored value a higher-priority one wins over."""
        if self.effective_schema is None or self.effective_schema not in self.chain:
            return ()
        winner = self.chain.index(self.effective_schema)
        return tuple(s for i, s in enumerate(self.chain) if i > winner and self.authored.get(s) is not None)

    @property
    def any_authored(self) -> bool:
        """True when at least one schema carries an authored value."""
        return any(v is not None for v in self.authored.values())


def _attr(joint: object, name: str) -> Usd.Attribute | None:
    """Return a valid attribute of ``joint`` by name, or None."""
    if not name:
        return None
    try:
        attr = joint.GetAttribute(name)
    except _USD_ACCESS_ERRORS:
        return None
    return attr if (attr is not None and attr.IsValid()) else None


def _authored_value(attr: Usd.Attribute | None) -> float | None:
    """Return an attribute's explicitly authored float value, or None."""
    if attr is None:
        return None
    try:
        if not attr.HasAuthoredValue():
            return None
        value = attr.Get()
    except _USD_ACCESS_ERRORS:
        return None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_authored_value(attr: Usd.Attribute | None) -> bool:
    """Return whether ``attr`` exists and carries an explicitly authored value."""
    if attr is None:
        return False
    try:
        return bool(attr.HasAuthoredValue())
    except _USD_ACCESS_ERRORS:
        return False


def joint_param_attrs(joint: object, spec: JointParamSpec) -> tuple[Usd.Attribute | None, Usd.Attribute | None]:
    """Return the ``(newton_attr, physx_attr)`` pair backing a parameter.

    Either entry is None when the joint does not have the corresponding API
    schema applied.  ``mjc:*`` is not returned: the gain tuner never authors it,
    so it has no write target (use :func:`authored_joint_param_values` to read
    it).

    Args:
        joint: The joint prim.
        spec: The parameter to resolve.

    Returns:
        The live Newton and PhysX attributes, each None when unavailable.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import JOINT_PARAM_SPECS, joint_param_attrs

        >>> newton_attr, physx_attr = joint_param_attrs(joint, JOINT_PARAM_SPECS[0])  # doctest: +NO_CHECK
    """
    return _attr(joint, spec.newton_attr), _attr(joint, spec.physx_attr)


def authored_joint_param_attrs(
    joint: object, spec: JointParamSpec
) -> tuple[Usd.Attribute | None, Usd.Attribute | None]:
    """Return the ``(newton_attr, physx_attr)`` entries carrying an authored value.

    Applying either API schema makes all of its attributes resolvable at their
    schema fallbacks, so "the attribute exists" says nothing about whether anyone
    set it.  Callers that persist opinions must use this rather than
    :func:`joint_param_attrs`, or they will write fallbacks such as an infinite
    velocity limit into the asset as though the user had chosen them.

    Args:
        joint: The joint prim.
        spec: The parameter to resolve.

    Returns:
        The Newton and PhysX attributes, each None unless it has an explicitly
        authored value.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import JOINT_PARAM_SPECS, authored_joint_param_attrs

        >>> newton_attr, physx_attr = authored_joint_param_attrs(joint, JOINT_PARAM_SPECS[0])  # doctest: +NO_CHECK
    """
    return tuple(  # type: ignore[return-value]
        attr if _has_authored_value(attr) else None for attr in joint_param_attrs(joint, spec)
    )


def authored_joint_param_values(joint: object, spec: JointParamSpec) -> dict:
    """Return every schema's explicitly authored opinion of a parameter.

    Args:
        joint: The joint prim.
        spec: The parameter to read.

    Returns:
        Map of resolver token -> authored value, one entry per schema the
        parameter exists on, with None where that schema carries no explicitly
        authored value.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import authored_joint_param_values, joint_param_spec

        >>> authored_joint_param_values(joint, joint_param_spec("armature"))  # doctest: +NO_CHECK
        {'newton': 0.25, 'mjc': None, 'physx': 0.75}
    """
    return {schema: _authored_value(_attr(joint, spec.attr_for_schema(schema))) for schema in spec.schemas}


def resolve_joint_param(joint: object, spec: JointParamSpec, backend: str, solver: str = "") -> JointParamResolution:
    """Resolve a parameter the way the active backend and solver resolve it.

    Walks :func:`param_resolver_chain` and takes the first *explicitly authored*
    value, matching Newton's ``SchemaResolverManager``.  A resolved
    :attr:`JointParamSpec.unauthored_sentinel` means "no limit" and wins the
    chain, because Newton's importer maps it to the engine's own unlimited
    default rather than continuing down the chain.

    An unknown Newton solver narrows the walk to the schemas every possible chain
    agrees on rather than guessing one: ``newton:*`` leads under every solver, so
    a value authored there is in effect whichever solver is running.  Only when
    nothing is authored that far does the answer depend on the solver, and the
    result is then reported as undetermined.

    Args:
        joint: The joint prim.
        spec: The parameter to resolve.
        backend: The active backend label (``"PhysX"`` / ``"NewtonAPI"``).
        solver: The running Newton solver token.  Ignored under PhysX.

    Returns:
        A :class:`JointParamResolution` carrying every schema's authored opinion
        and the one in effect.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import joint_param_spec, resolve_joint_param

        >>> resolution = resolve_joint_param(joint, joint_param_spec("armature"), "PhysX")  # doctest: +NO_CHECK
        >>> resolution.effective_value  # doctest: +NO_CHECK
        0.75
    """
    authored = authored_joint_param_values(joint, spec)
    candidates = candidate_param_chains(spec, backend, solver)
    chain_complete = len(candidates) == 1
    chain = candidates[0] if chain_complete else _common_prefix(candidates)
    effective_schema: str | None = None
    effective_value: float | None = None
    unlimited = False
    for schema in chain:
        value = authored.get(schema)
        if value is None:
            continue
        effective_schema = schema
        if spec.unauthored_sentinel is not None and value == spec.unauthored_sentinel:
            unlimited = True
        else:
            effective_value = value
        break
    return JointParamResolution(
        spec=spec,
        backend=str(backend),
        solver=str(solver),
        chain=chain,
        candidate_chains=candidates,
        chain_complete=chain_complete,
        authored=authored,
        effective_schema=effective_schema,
        effective_value=effective_value,
        unlimited=unlimited,
    )


def resolve_joint_params(joint: object, backend: str, solver: str = "") -> list[JointParamResolution]:
    """Resolve every advanced joint parameter for a joint.

    Args:
        joint: The joint prim.
        backend: The active backend label.
        solver: The running Newton solver token.

    Returns:
        One :class:`JointParamResolution` per :data:`JOINT_PARAM_SPECS` entry, in
        registry order.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import resolve_joint_params

        >>> [r.spec.key for r in resolve_joint_params(joint, "PhysX")]  # doctest: +NO_CHECK
        ['armature', 'joint_friction', 'max_joint_velocity']
    """
    return [resolve_joint_param(joint, spec, backend, solver) for spec in JOINT_PARAM_SPECS]


def diverging_joint_params(joint: object, backend: str, solver: str = "") -> list[JointParamResolution]:
    """Return the parameters whose Newton and PhysX halves hold different values.

    Divergence is a legitimate authoring choice -- the two solvers need different
    tuning -- so this reports it for display, not as a fault to repair.

    Args:
        joint: The joint prim.
        backend: The active backend label.
        solver: The running Newton solver token.

    Returns:
        One :class:`JointParamResolution` per diverging parameter, in
        :data:`JOINT_PARAM_SPECS` order.  Empty when the two halves agree or
        either is unauthored.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import diverging_joint_params

        >>> [r.spec.key for r in diverging_joint_params(joint, "PhysX")]  # doctest: +NO_CHECK
        ['joint_friction']
    """
    return [r for r in resolve_joint_params(joint, backend, solver) if r.backends_diverge]


def _apply_api(joint: object, schema_name: str) -> bool:
    """Apply an API schema to ``joint`` when absent; return True when present after."""
    try:
        if joint.HasAPI(schema_name):
            return True
        return bool(joint.ApplyAPI(schema_name))
    except _USD_ACCESS_ERRORS:
        return False


def _create_attr(joint: object, name: str) -> Usd.Attribute | None:
    """Return ``name`` on ``joint``, creating it as a float when missing."""
    attr = _attr(joint, name)
    if attr is not None:
        return attr
    try:
        created = joint.CreateAttribute(name, Sdf.ValueTypeNames.Float)
    except _USD_ACCESS_ERRORS:
        return None
    return created if (created is not None and created.IsValid()) else None


def author_joint_param(joint: object, spec: JointParamSpec, value: float, backend: str) -> Usd.Attribute | None:
    """Write a parameter to the active backend's schema only.

    The other backend's value is left exactly as it was, because the two solvers
    are tuned independently.  ``mjc:*`` is never written.

    Args:
        joint: The joint prim.
        spec: The parameter to write.
        value: The new value, in the schema's storage units.
        backend: The active backend label, which selects the schema written.

    Returns:
        The attribute that was written, or None when nothing was written -- the
        backend is not one of :data:`SUPPORTED_BACKENDS`, so no schema is known
        to be the right target, or the schema could not be applied (most often
        ``NewtonJointAPI`` with ``omni.usd.schema.newton`` absent).  Either way
        callers must surface it rather than treat it as success.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import author_joint_param, joint_param_spec

        >>> author_joint_param(joint, joint_param_spec("armature"), 0.05, "PhysX")  # doctest: +NO_CHECK
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    schema = backend_write_schema(backend)
    if schema is None:
        return None
    attr_name = spec.attr_for_schema(schema)
    if attr_name is None:
        return None
    if not _apply_api(joint, SCHEMA_JOINT_APIS[schema]):
        return None
    attr = _create_attr(joint, attr_name)
    if attr is None:
        return None
    try:
        attr.Set(value)
    except _USD_ACCESS_ERRORS:
        return None
    return attr


def copy_joint_param_to_backend(
    joint: object, spec: JointParamSpec, value: float, backend: str
) -> Usd.Attribute | None:
    """Write a parameter to the schema of the backend that is *not* active.

    This is the explicit, opt-in "copy this value to the other backend" action.
    It changes what the other backend simulates, so callers must say so before
    invoking it.

    Args:
        joint: The joint prim.
        spec: The parameter to write.
        value: The value to copy, in the schema's storage units.
        backend: The *active* backend label.  The write lands on the other one.

    Returns:
        The attribute that was written, or None when it could not be -- including
        when ``backend`` is unsupported and so has no other backend to copy to.

    Example:

    .. code-block:: python

        >>> from isaacsim.robot_setup.gain_tuner import copy_joint_param_to_backend, joint_param_spec

        >>> copy_joint_param_to_backend(joint, joint_param_spec("armature"), 0.05, "PhysX")  # doctest: +NO_CHECK
    """
    target = other_backend(backend)
    if target is None:
        return None
    return author_joint_param(joint, spec, value, target)
