# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the per-backend advanced joint parameter helpers."""

from __future__ import annotations

import math
import pathlib
import re

import omni.kit.test
from isaacsim.robot_setup import gain_tuner
from pxr import Sdf, Usd, UsdPhysics

NEWTON_JOINT_API = gain_tuner.NEWTON_JOINT_API
PHYSX_JOINT_API = gain_tuner.PHYSX_JOINT_API

NEWTON = gain_tuner.BACKEND_NEWTON
PHYSX = gain_tuner.BACKEND_PHYSX

MUJOCO = gain_tuner.SOLVER_MUJOCO
XPBD = gain_tuner.SOLVER_XPBD
VBD = gain_tuner.SOLVER_VBD


class _JointCase(omni.kit.test.AsyncTestCase):
    """Base case holding the in-memory stage the joint prim lives on."""

    def _joint(self) -> Usd.Prim:
        """Return a revolute joint on a stage kept alive for the test's duration."""
        self._stage = Usd.Stage.CreateInMemory()
        return UsdPhysics.RevoluteJoint.Define(self._stage, "/Robot/joint").GetPrim()

    def _both_schemas(self) -> Usd.Prim:
        """Return a joint with both the Newton and PhysX joint APIs applied."""
        joint = self._joint()
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.ApplyAPI(PHYSX_JOINT_API)
        return joint

    @staticmethod
    def _set_mjc(joint: Usd.Prim, spec, value: float) -> None:
        """Author the parameter's ``mjc:*`` half.

        The MuJoCo joint schema is not applied through an API name here: the gain
        tuner only ever *reads* ``mjc:*``, by attribute token, so a raw attribute
        is exactly what it sees on an asset whose MuJoCo gains were authored by
        hand.
        """
        joint.CreateAttribute(spec.mjc_attr, Sdf.ValueTypeNames.Float).Set(value)


class TestJointParamSpecs(omni.kit.test.AsyncTestCase):
    """The registry of advanced joint parameters and the schemas carrying them."""

    async def test_specs_cover_the_three_multi_schema_params(self) -> None:
        """Armature, joint friction, and the velocity limit are registered."""
        self.assertEqual(
            {spec.key for spec in gain_tuner.JOINT_PARAM_SPECS},
            {"armature", "joint_friction", "max_joint_velocity"},
        )

    async def test_spec_lookup_by_key(self) -> None:
        """Specs are addressable by key and carry each schema's attribute name."""
        spec = gain_tuner.joint_param_spec("joint_friction")
        self.assertEqual(spec.newton_attr, "newton:friction")
        self.assertEqual(spec.physx_attr, "physxJoint:jointFriction")
        self.assertEqual(spec.mjc_attr, "mjc:frictionloss")
        self.assertIsNone(gain_tuner.joint_param_spec("not_a_param"))

    async def test_velocity_limit_has_no_mjc_attribute(self) -> None:
        """SchemaResolverMjc declares no velocity_limit key, so mjc:* has none either."""
        self.assertIsNone(gain_tuner.joint_param_spec("max_joint_velocity").mjc_attr)
        self.assertEqual(gain_tuner.joint_param_spec("armature").mjc_attr, "mjc:armature")

    async def test_only_friction_lacks_a_physx_fallback_under_newton(self) -> None:
        """SchemaResolverPhysx declares no friction key, so Newton never reads it."""
        self.assertFalse(gain_tuner.joint_param_spec("joint_friction").newton_reads_physx)
        self.assertTrue(gain_tuner.joint_param_spec("armature").newton_reads_physx)
        self.assertTrue(gain_tuner.joint_param_spec("max_joint_velocity").newton_reads_physx)

    async def test_only_velocity_limit_has_an_unlimited_sentinel(self) -> None:
        """An infinite velocity limit is the one value that means "no limit"."""
        self.assertEqual(gain_tuner.joint_param_spec("max_joint_velocity").unauthored_sentinel, math.inf)
        self.assertIsNone(gain_tuner.joint_param_spec("armature").unauthored_sentinel)
        self.assertIsNone(gain_tuner.joint_param_spec("joint_friction").unauthored_sentinel)

    async def test_attr_for_schema_maps_every_token(self) -> None:
        """Each resolver token maps to its own attribute name, or None when absent."""
        spec = gain_tuner.joint_param_spec("armature")
        self.assertEqual(spec.attr_for_schema(gain_tuner.SCHEMA_NEWTON), "newton:armature")
        self.assertEqual(spec.attr_for_schema(gain_tuner.SCHEMA_MJC), "mjc:armature")
        self.assertEqual(spec.attr_for_schema(gain_tuner.SCHEMA_PHYSX), "physxJoint:armature")
        self.assertIsNone(spec.attr_for_schema("not_a_schema"))


class TestTheNewtonDefaultIsStillNewtons(omni.kit.test.AsyncTestCase):
    """`NEWTON_DEFAULT_ARMATURE` is a copy, and copies go stale.

    The panel states "0.1 (default)" over an unauthored Newton armature and names
    the engine in the tooltip, so the number has to be the one Newton is actually
    given.  It is set in ``isaacsim.physics.newton``'s ``NewtonConfig`` -- a
    user-editable config field, not a frozen constant -- and copied onto
    ``ModelBuilder.default_joint_cfg.armature`` when the stage is built.  Nothing
    connects the two, so an Isaac Sim or Newton change moves one and leaves the
    panel asserting the other.

    This extension does not depend on ``isaacsim.physics.newton``: the joint schema
    is USD, and reading it needs no engine.  So the upstream is read the way a test
    can read it without adding a runtime dependency -- imported when the extension
    happens to be enabled, and otherwise parsed out of its own source.

    Newton's ``ModelBuilder`` default is *not* what is pinned here, despite being
    what the constant's name suggests: ``JointDofConfig.armature`` is 0.0 upstream,
    and the 0.1 is Isaac Sim overriding it.  Pinning the wrong one of the two would
    have failed immediately, which is some evidence the pin is real.
    """

    _CONFIG_MODULE = "isaacsim.physics.newton"
    _FIELD = "armature"

    @staticmethod
    def _imported_default() -> float | None:
        """Return ``NewtonConfig().armature``, or None when the extension is absent."""
        try:
            from isaacsim.physics.newton import NewtonConfig
        except Exception:
            return None
        try:
            return float(NewtonConfig().armature)
        except (AttributeError, TypeError, ValueError):
            return None

    @classmethod
    def _source_default(cls) -> float | None:
        """Return the dataclass default parsed from ``newton_config.py``.

        The fallback for a disabled extension, which is the normal case here: this
        extension declares no dependency on ``isaacsim.physics.newton``, so its
        module is not importable from the gain tuner's own test app.

        The extension is found as a sibling of this one, which holds in both
        layouts -- ``source/extensions/`` in the repo and ``exts/`` once built.
        Parsing reads the same declaration the import would have evaluated, so it
        fails on the same edit; what it cannot see is a default computed at runtime,
        which would leave the field unparseable and the test saying so rather than
        passing.
        """
        pattern = re.compile(rf"^\s*{cls._FIELD}\s*:\s*float\s*=\s*([0-9eE.+-]+)\s*$", re.MULTILINE)
        for parent in pathlib.Path(__file__).resolve().parents:
            root = parent / cls._CONFIG_MODULE
            if not root.is_dir():
                continue
            # Globbed rather than spelled out: the module sits under ``python/impl``
            # in the source tree and ``isaacsim/physics/newton/impl`` once built.
            for source in root.rglob("newton_config.py"):
                match = pattern.search(source.read_text())
                if match:
                    return float(match.group(1))
        return None

    async def test_the_panels_newton_default_matches_the_config_it_mirrors(self) -> None:
        """The number the panel calls Newton's default is the one Newton is handed."""
        upstream = self._imported_default()
        source = "NewtonConfig().armature"
        if upstream is None:
            upstream, source = self._source_default(), "NewtonConfig.armature in newton_config.py"
        self.assertIsNotNone(
            upstream,
            "could not read isaacsim.physics.newton's default armature by import or from source; "
            "NEWTON_DEFAULT_ARMATURE is unverifiable, which is the state this test exists to prevent",
        )
        self.assertAlmostEqual(
            gain_tuner.NEWTON_DEFAULT_ARMATURE,
            upstream,
            places=6,
            msg=(
                f"NEWTON_DEFAULT_ARMATURE is {gain_tuner.NEWTON_DEFAULT_ARMATURE} but {source} is {upstream}. "
                "The panel labels an unauthored Newton armature with its own copy, so update the constant "
                "(and the sentences quoting it) to whatever Newton is now given."
            ),
        )


class TestResolverChain(omni.kit.test.AsyncTestCase):
    """The schema-resolver order, which depends on the backend and the solver."""

    async def test_physx_backend_reads_only_its_own_schema(self) -> None:
        """PhysX has no chain: it reads physxJoint:* and nothing else."""
        self.assertEqual(gain_tuner.resolver_chain(PHYSX), (gain_tuner.SCHEMA_PHYSX,))
        self.assertEqual(gain_tuner.resolver_chain(PHYSX, MUJOCO), (gain_tuner.SCHEMA_PHYSX,))

    async def test_mujoco_solver_puts_mjc_between_newton_and_physx(self) -> None:
        """Mirrors the [Newton, Mjc, Physx] resolver list in newton_stage."""
        self.assertEqual(
            gain_tuner.resolver_chain(NEWTON, MUJOCO),
            (gain_tuner.SCHEMA_NEWTON, gain_tuner.SCHEMA_MJC, gain_tuner.SCHEMA_PHYSX),
        )

    async def test_xpbd_and_vbd_solvers_skip_mjc(self) -> None:
        """Mirrors the [Newton, Physx] resolver list in newton_stage."""
        for solver in (XPBD, VBD):
            self.assertEqual(
                gain_tuner.resolver_chain(NEWTON, solver),
                (gain_tuner.SCHEMA_NEWTON, gain_tuner.SCHEMA_PHYSX),
                msg=f"{solver} chain",
            )

    async def test_solver_token_is_matched_case_insensitively(self) -> None:
        """The stage and the config disagree on case; both must resolve."""
        self.assertEqual(gain_tuner.resolver_chain(NEWTON, " MuJoCo "), gain_tuner.resolver_chain(NEWTON, MUJOCO))

    async def test_every_solver_isaac_sim_can_select_has_a_chain(self) -> None:
        """The chain table must cover the solver registry, not a subset of it.

        Newton also defines Featherstone and Semi-Implicit, which take
        ``add_usd``'s default ``newton:*``-only chain -- but Isaac Sim registers no
        scene API for them, so a stage cannot report one.  If that changes, this
        fails rather than silently reporting the solver as undetermined.
        """
        try:
            from isaacsim.physics.newton.impl.utils import newton_solver_to_api_schema
        except ImportError:
            self.skipTest("isaacsim.physics.newton is not enabled")
        self.assertEqual(set(newton_solver_to_api_schema), set(gain_tuner.NEWTON_SOLVER_TYPES))
        for solver in newton_solver_to_api_schema:
            self.assertIsNotNone(gain_tuner.resolver_chain(NEWTON, solver), msg=solver)

    async def test_unknown_solver_leaves_the_newton_chain_undetermined(self) -> None:
        """Guessing here would confidently report the wrong number on a MuJoCo run."""
        self.assertIsNone(gain_tuner.resolver_chain(NEWTON, ""))
        self.assertIsNone(gain_tuner.resolver_chain(NEWTON, "featherstone"))

    async def test_friction_drops_physx_from_the_newton_chain(self) -> None:
        """physxJoint:jointFriction is not in any Newton chain at all."""
        spec = gain_tuner.joint_param_spec("joint_friction")
        self.assertEqual(gain_tuner.param_resolver_chain(spec, NEWTON, XPBD), (gain_tuner.SCHEMA_NEWTON,))
        self.assertEqual(
            gain_tuner.param_resolver_chain(spec, NEWTON, MUJOCO),
            (gain_tuner.SCHEMA_NEWTON, gain_tuner.SCHEMA_MJC),
        )

    async def test_friction_is_still_read_under_physx(self) -> None:
        """The no-fallback rule is Newton's, not a claim that PhysX ignores it."""
        spec = gain_tuner.joint_param_spec("joint_friction")
        self.assertEqual(gain_tuner.param_resolver_chain(spec, PHYSX), (gain_tuner.SCHEMA_PHYSX,))

    async def test_velocity_limit_drops_mjc_even_under_the_mujoco_solver(self) -> None:
        """SchemaResolverMjc has no velocity_limit key, so mjc:* never takes part."""
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        self.assertEqual(
            gain_tuner.param_resolver_chain(spec, NEWTON, MUJOCO),
            (gain_tuner.SCHEMA_NEWTON, gain_tuner.SCHEMA_PHYSX),
        )

    async def test_undetermined_chain_propagates_per_parameter(self) -> None:
        """An unknown solver is unknown for every parameter, not just some."""
        for spec in gain_tuner.JOINT_PARAM_SPECS:
            self.assertIsNone(gain_tuner.param_resolver_chain(spec, NEWTON, ""), msg=spec.key)

    async def test_backend_write_schema_and_other_backend(self) -> None:
        """An edit lands on the active backend; the copy action targets the other."""
        self.assertEqual(gain_tuner.backend_write_schema(NEWTON), gain_tuner.SCHEMA_NEWTON)
        self.assertEqual(gain_tuner.backend_write_schema(PHYSX), gain_tuner.SCHEMA_PHYSX)
        self.assertEqual(gain_tuner.other_backend(NEWTON), PHYSX)
        self.assertEqual(gain_tuner.other_backend(PHYSX), NEWTON)

    async def test_an_unsupported_backend_has_no_write_schema(self) -> None:
        """Fail closed: no schema, no opposite, no chain, no default.

        ``SimulationManager.get_active_physics_engine`` is typed to include
        ``remotesim``, and a failed query reports nothing at all.  Returning
        ``physxJoint:*`` for either sent an edit to a schema with no evidence the
        running engine reads it, which is a silent no-op rather than a default.
        """
        for backend in ("remotesim", "", "wobble", None):
            self.assertFalse(gain_tuner.backend_supported(backend), msg=repr(backend))
            self.assertIsNone(gain_tuner.backend_write_schema(backend), msg=repr(backend))
            self.assertIsNone(gain_tuner.other_backend(backend), msg=repr(backend))
            self.assertIsNone(gain_tuner.resolver_chain(backend), msg=repr(backend))
            for spec in gain_tuner.JOINT_PARAM_SPECS:
                self.assertIsNone(spec.engine_default(backend), msg=f"{backend!r}/{spec.key}")
                self.assertIsNone(gain_tuner.param_resolver_chain(spec, backend), msg=f"{backend!r}/{spec.key}")
                self.assertEqual(gain_tuner.candidate_param_chains(spec, backend), (), msg=f"{backend!r}/{spec.key}")

    async def test_the_supported_backends_are_exactly_the_two(self) -> None:
        """Adding an engine means adding its schema, not falling through to PhysX."""
        self.assertEqual(set(gain_tuner.SUPPORTED_BACKENDS), {PHYSX, NEWTON})
        for backend in (PHYSX, NEWTON):
            self.assertTrue(gain_tuner.backend_supported(backend), msg=backend)

    async def test_backend_display_labels_are_user_facing_and_ascii(self) -> None:
        """ "NewtonAPI" is an internal token, not a name to show a user."""
        self.assertEqual(gain_tuner.backend_display_label(NEWTON), "Newton")
        self.assertEqual(gain_tuner.backend_display_label(PHYSX), "PhysX")
        # An unnameable engine reads as a phrase, not as an invented product name.
        label = gain_tuner.backend_display_label("remotesim")
        self.assertNotIn("remotesim", label)
        for backend in (NEWTON, PHYSX, "remotesim", ""):
            self.assertTrue(gain_tuner.backend_display_label(backend).isascii(), msg=repr(backend))

    async def test_only_physx_re_reads_usd_while_the_timeline_plays(self) -> None:
        """Measured, not documented -- see ``TestLiveUsdWriteReachesTheEngine``.

        PhysX applies a mid-run ``physxJoint:armature`` write to the running
        articulation; Newton builds its model once in ``ModelBuilder.add_usd`` and
        has no USD notice handler, so the same edit is unread until the next play.
        An engine whose behaviour was never measured is treated as Newton-like,
        so the user is told to replay rather than assured the edit took.
        """
        self.assertTrue(gain_tuner.backend_reads_usd_while_playing(PHYSX))
        self.assertFalse(gain_tuner.backend_reads_usd_while_playing(NEWTON))
        for backend in ("remotesim", "", "wobble"):
            self.assertFalse(gain_tuner.backend_reads_usd_while_playing(backend), msg=repr(backend))

    async def test_engine_defaults_are_what_each_engine_actually_simulates(self) -> None:
        """The values an unauthored parameter is worth, per backend.

        Zero is right only for friction.  Newton copies its config armature onto the
        ``ModelBuilder`` default, and neither backend clamps an unauthored velocity
        limit -- so showing zero for either states a value nothing uses.
        """
        armature = gain_tuner.joint_param_spec("armature")
        friction = gain_tuner.joint_param_spec("joint_friction")
        velocity = gain_tuner.joint_param_spec("max_joint_velocity")

        self.assertAlmostEqual(armature.engine_default(NEWTON), gain_tuner.NEWTON_DEFAULT_ARMATURE, places=6)
        self.assertEqual(armature.engine_default(PHYSX), 0.0)
        self.assertEqual(friction.engine_default(NEWTON), 0.0)
        self.assertEqual(friction.engine_default(PHYSX), 0.0)
        self.assertEqual(velocity.engine_default(NEWTON), math.inf)
        self.assertEqual(velocity.engine_default(PHYSX), math.inf)
        self.assertNotEqual(gain_tuner.NEWTON_DEFAULT_ARMATURE, 0.0)

    async def test_mjc_is_never_a_write_target(self) -> None:
        """MuJoCo gains are authored by hand; the tuner only reports them."""
        self.assertNotIn(gain_tuner.SCHEMA_MJC, gain_tuner.SCHEMA_JOINT_APIS)

    async def test_a_known_chain_has_exactly_one_candidate(self) -> None:
        """Nothing is uncertain once the solver is known."""
        spec = gain_tuner.joint_param_spec("armature")
        self.assertEqual(
            gain_tuner.candidate_param_chains(spec, NEWTON, MUJOCO),
            (gain_tuner.param_resolver_chain(spec, NEWTON, MUJOCO),),
        )

    async def test_an_unknown_solver_yields_every_possible_chain(self) -> None:
        """Both Newton chains are in play, which is what makes mjc:* uncertain."""
        self.assertEqual(
            gain_tuner.candidate_param_chains(gain_tuner.joint_param_spec("armature"), NEWTON, ""),
            (
                (gain_tuner.SCHEMA_NEWTON, gain_tuner.SCHEMA_MJC, gain_tuner.SCHEMA_PHYSX),
                (gain_tuner.SCHEMA_NEWTON, gain_tuner.SCHEMA_PHYSX),
            ),
        )

    async def test_a_solver_independent_parameter_has_one_candidate_regardless(self) -> None:
        """The velocity limit resolves the same way under every solver."""
        self.assertEqual(
            gain_tuner.candidate_param_chains(gain_tuner.joint_param_spec("max_joint_velocity"), NEWTON, ""),
            ((gain_tuner.SCHEMA_NEWTON, gain_tuner.SCHEMA_PHYSX),),
        )


class TestResolveJointParam(_JointCase):
    """Per-backend, per-solver resolution: first authored value in chain order wins."""

    async def test_physx_reads_its_own_schema_and_ignores_newton(self) -> None:
        """Under PhysX the newton:* half is not part of the chain."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.25)
        joint.GetAttribute(spec.physx_attr).Set(0.75)

        resolution = gain_tuner.resolve_joint_param(joint, spec, PHYSX)
        self.assertAlmostEqual(resolution.effective_value, 0.75, places=5)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_PHYSX)

    async def test_newton_prefers_its_own_schema_over_physx(self) -> None:
        """newton:* is first in every Newton chain."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.25)
        joint.GetAttribute(spec.physx_attr).Set(0.75)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertAlmostEqual(resolution.effective_value, 0.25, places=5)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_NEWTON)

    async def test_newton_armature_falls_back_to_physx(self) -> None:
        """An unauthored newton:armature resolves to physxJoint:armature."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.physx_attr).Set(0.5)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertAlmostEqual(resolution.effective_value, 0.5, places=5)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_PHYSX)

    async def test_newton_velocity_limit_falls_back_to_physx(self) -> None:
        """An unauthored newton:velocityLimit resolves to physxJoint:maxJointVelocity."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        joint.GetAttribute(spec.physx_attr).Set(107.0)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertAlmostEqual(resolution.effective_value, 107.0, places=5)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_PHYSX)

    async def test_physx_only_friction_is_ignored_by_newton(self) -> None:
        """The asymmetry that matters: Newton simulates this joint frictionless.

        ``SchemaResolverPhysx`` declares no ``friction`` key, so the authored
        ``physxJoint:jointFriction`` is not merely outranked -- it is never read.
        """
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("joint_friction")
        joint.GetAttribute(spec.physx_attr).Set(0.6)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertIsNone(resolution.effective_value)
        self.assertIsNone(resolution.effective_schema)
        self.assertEqual(resolution.unread_schemas, (gain_tuner.SCHEMA_PHYSX,))
        # The same value is what PhysX actually simulates.
        self.assertAlmostEqual(gain_tuner.resolve_joint_param(joint, spec, PHYSX).effective_value, 0.6, places=5)

    async def test_mjc_armature_sits_between_newton_and_physx(self) -> None:
        """Under the MuJoCo solver an mjc:* value outranks the PhysX half."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        self._set_mjc(joint, spec, 0.4)
        joint.GetAttribute(spec.physx_attr).Set(0.9)

        mujoco = gain_tuner.resolve_joint_param(joint, spec, NEWTON, MUJOCO)
        self.assertAlmostEqual(mujoco.effective_value, 0.4, places=5)
        self.assertEqual(mujoco.effective_schema, gain_tuner.SCHEMA_MJC)

        # The same stage under XPBD skips mjc:* entirely.
        xpbd = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertAlmostEqual(xpbd.effective_value, 0.9, places=5)
        self.assertEqual(xpbd.effective_schema, gain_tuner.SCHEMA_PHYSX)

    async def test_mjc_frictionloss_is_read_under_the_mujoco_solver(self) -> None:
        """Friction has no PhysX fallback but does have an mjc:* one."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("joint_friction")
        self._set_mjc(joint, spec, 0.15)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, MUJOCO)
        self.assertAlmostEqual(resolution.effective_value, 0.15, places=5)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_MJC)

    async def test_mjc_velocity_limit_is_never_read(self) -> None:
        """Even authored, mjc:* has no velocity_limit key to be read through."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        joint.CreateAttribute("mjc:velocityLimit", Sdf.ValueTypeNames.Float).Set(12.0)
        joint.GetAttribute(spec.physx_attr).Set(107.0)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, MUJOCO)
        self.assertAlmostEqual(resolution.effective_value, 107.0, places=5)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_PHYSX)

    async def test_authored_infinite_newton_limit_wins_the_chain(self) -> None:
        """`inf` means "no limit" and is resolved, not skipped.

        Newton's importer maps the *resolved* ``inf`` to its own unlimited default,
        so an authored ``inf`` does not fall through to
        ``physxJoint:maxJointVelocity``.  This is the documented neutral-layer
        pattern: ``newton:velocityLimit = inf`` beside a deliberately different
        ``physxJoint:maxJointVelocity``.
        """
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        joint.GetAttribute(spec.newton_attr).Set(math.inf)
        joint.GetAttribute(spec.physx_attr).Set(107.0)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_NEWTON)
        self.assertTrue(resolution.unlimited)
        self.assertIsNone(resolution.effective_value)
        # PhysX still enforces its own, different limit.
        self.assertAlmostEqual(gain_tuner.resolve_joint_param(joint, spec, PHYSX).effective_value, 107.0, places=5)

    async def test_applied_but_unauthored_schema_pins_nothing(self) -> None:
        """Newton reads through HasAuthoredValue, so a fallback is not an opinion."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertIsNone(resolution.effective_schema)
        self.assertIsNone(resolution.effective_value)
        self.assertFalse(resolution.unlimited)
        self.assertFalse(resolution.any_authored)

    async def test_unknown_solver_still_resolves_a_newton_authored_value(self) -> None:
        """newton:* leads under every solver, so this answer does not need one."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.25)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, "")
        self.assertFalse(resolution.chain_complete)
        self.assertTrue(resolution.determined)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_NEWTON)
        self.assertAlmostEqual(resolution.effective_value, 0.25, places=5)

    async def test_unknown_solver_leaves_a_solver_dependent_value_undetermined(self) -> None:
        """Whether mjc:* outranks physxJoint:* is exactly what the solver decides."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        self._set_mjc(joint, spec, 0.4)
        joint.GetAttribute(spec.physx_attr).Set(0.9)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, "")
        self.assertFalse(resolution.determined)
        self.assertIsNone(resolution.effective_schema)
        self.assertIsNone(resolution.effective_value)
        # The authored opinions are still reported; only the winner is unknown.
        self.assertAlmostEqual(resolution.mjc_value, 0.4, places=5)
        self.assertAlmostEqual(resolution.physx_value, 0.9, places=5)
        # Neither is ruled out, so neither may be called ignored.
        self.assertEqual(resolution.unread_schemas, ())

    async def test_velocity_limit_needs_no_solver_at_all(self) -> None:
        """Its chain is the same under every solver, so it is never undetermined."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        joint.GetAttribute(spec.physx_attr).Set(107.0)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, "")
        self.assertTrue(resolution.chain_complete)
        self.assertAlmostEqual(resolution.effective_value, 107.0, places=5)

    async def test_friction_is_ignored_under_newton_even_without_a_solver(self) -> None:
        """No Newton solver reads physxJoint:jointFriction, so the finding holds."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("joint_friction")
        joint.GetAttribute(spec.physx_attr).Set(0.6)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, "")
        self.assertEqual(resolution.unread_schemas, (gain_tuner.SCHEMA_PHYSX,))

    async def test_resolution_carries_every_schema_opinion(self) -> None:
        """Callers state each backend's value, so all of them must be present."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.25)
        self._set_mjc(joint, spec, 0.4)
        joint.GetAttribute(spec.physx_attr).Set(0.75)

        resolution = gain_tuner.resolve_joint_param(joint, spec, PHYSX)
        self.assertAlmostEqual(resolution.newton_value, 0.25, places=5)
        self.assertAlmostEqual(resolution.mjc_value, 0.4, places=5)
        self.assertAlmostEqual(resolution.physx_value, 0.75, places=5)
        self.assertAlmostEqual(resolution.other_value, 0.25, places=5)
        self.assertEqual(resolution.write_attr, spec.physx_attr)

    async def test_shadowed_schemas_are_the_outranked_authored_ones(self) -> None:
        """A lower-priority authored value is shadowed, not unread."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.25)
        joint.GetAttribute(spec.physx_attr).Set(0.75)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertEqual(resolution.shadowed_schemas, (gain_tuner.SCHEMA_PHYSX,))
        self.assertEqual(resolution.unread_schemas, ())

    async def test_resolve_all_params_in_registry_order(self) -> None:
        """The per-joint sweep the UI uses covers every parameter, once each."""
        resolutions = gain_tuner.resolve_joint_params(self._both_schemas(), PHYSX)
        self.assertEqual(
            [resolution.spec.key for resolution in resolutions],
            ["armature", "joint_friction", "max_joint_velocity"],
        )


class TestDivergingJointParams(_JointCase):
    """Detection of the two backends holding different values, for reporting only."""

    async def test_differing_authored_values_diverge(self) -> None:
        """Two authored halves with different values is the reportable case."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("joint_friction")
        joint.GetAttribute(spec.newton_attr).Set(0.25)
        joint.GetAttribute(spec.physx_attr).Set(0.75)

        self.assertTrue(gain_tuner.resolve_joint_param(joint, spec, PHYSX).backends_diverge)

    async def test_equal_values_do_not_diverge(self) -> None:
        """Two halves holding the same number have nothing to report."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("joint_friction")
        joint.GetAttribute(spec.newton_attr).Set(0.4)
        joint.GetAttribute(spec.physx_attr).Set(0.4)

        self.assertFalse(gain_tuner.resolve_joint_param(joint, spec, PHYSX).backends_diverge)

    async def test_one_side_unauthored_does_not_diverge(self) -> None:
        """A single opinion is not a difference of opinion."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.25)

        self.assertFalse(gain_tuner.resolve_joint_param(joint, spec, PHYSX).backends_diverge)

    async def test_nan_is_not_reported_as_a_divergence(self) -> None:
        """NaN compares unequal to itself, so it would never settle."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(math.nan)
        joint.GetAttribute(spec.physx_attr).Set(0.5)

        self.assertFalse(gain_tuner.resolve_joint_param(joint, spec, PHYSX).backends_diverge)

    async def test_mjc_value_alone_is_not_a_backend_divergence(self) -> None:
        """mjc:* is neither backend's write target, so it is not one half of a pair."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        self._set_mjc(joint, spec, 0.4)
        joint.GetAttribute(spec.physx_attr).Set(0.9)

        self.assertFalse(gain_tuner.resolve_joint_param(joint, spec, NEWTON, MUJOCO).backends_diverge)

    async def test_scan_lists_every_diverging_param_in_registry_order(self) -> None:
        """The per-joint scan reports each difference once, in registry order."""
        joint = self._both_schemas()
        for key, newton_value, physx_value in (
            ("armature", 0.1, 0.2),
            ("joint_friction", 0.3, 0.3),
            ("max_joint_velocity", 90.0, 180.0),
        ):
            spec = gain_tuner.joint_param_spec(key)
            joint.GetAttribute(spec.newton_attr).Set(newton_value)
            joint.GetAttribute(spec.physx_attr).Set(physx_value)

        self.assertEqual(
            [resolution.spec.key for resolution in gain_tuner.diverging_joint_params(joint, PHYSX)],
            ["armature", "max_joint_velocity"],
        )

    async def test_a_single_backend_edit_leaves_the_other_diverging(self) -> None:
        """This is the intended outcome now, not a state to be repaired."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.physx_attr).Set(0.75)
        gain_tuner.author_joint_param(joint, spec, 0.25, NEWTON)

        self.assertTrue(gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD).backends_diverge)
        self.assertAlmostEqual(joint.GetAttribute(spec.physx_attr).Get(), 0.75, places=5)


class TestResolvedFromFallbackSchema(_JointCase):
    """The value in effect coming from a schema an edit would not write.

    Newton resolves ``newton:* -> mjc:* -> physxJoint:*``, so a joint that authors
    only the PhysX half is still simulated from it.  Nothing about the number says
    so, and the first edit moves the value onto ``newton:*``, which then wins the
    chain -- which is why the UI has to be able to ask.
    """

    async def test_newton_reading_a_physx_only_value_is_a_fallback(self) -> None:
        """The case worth reporting: authored nowhere Newton would write."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.physx_attr).Set(0.6)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertTrue(resolution.resolved_from_fallback_schema)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_PHYSX)
        self.assertEqual(resolution.write_schema, gain_tuner.SCHEMA_NEWTON)

    async def test_an_mjc_value_read_under_mujoco_is_also_a_fallback(self) -> None:
        """``mjc:*`` is never an edit target, so reading it is the same surprise."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        self._set_mjc(joint, spec, 0.4)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, MUJOCO)
        self.assertTrue(resolution.resolved_from_fallback_schema)
        self.assertEqual(resolution.effective_schema, gain_tuner.SCHEMA_MJC)

    async def test_a_backends_own_authored_value_is_not_a_fallback(self) -> None:
        """The common case: the chain's first entry answers, under either backend."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.25)
        joint.GetAttribute(spec.physx_attr).Set(0.75)

        for backend, solver in ((NEWTON, XPBD), (PHYSX, "")):
            resolution = gain_tuner.resolve_joint_param(joint, spec, backend, solver)
            self.assertFalse(resolution.resolved_from_fallback_schema, msg=backend)

    async def test_physx_has_no_fallback_to_fall_into(self) -> None:
        """PhysX reads ``physxJoint:*`` alone, so a Newton value is unread, not read."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.newton_attr).Set(0.55)

        resolution = gain_tuner.resolve_joint_param(joint, spec, PHYSX)
        self.assertFalse(resolution.resolved_from_fallback_schema)
        self.assertIsNone(resolution.effective_schema)

    async def test_nothing_authored_is_not_a_fallback(self) -> None:
        """With no value in effect there is no source to attribute it to."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")

        for backend, solver in ((NEWTON, XPBD), (PHYSX, "")):
            resolution = gain_tuner.resolve_joint_param(joint, spec, backend, solver)
            self.assertFalse(resolution.resolved_from_fallback_schema, msg=backend)

    async def test_an_unsupported_backend_claims_no_fallback(self) -> None:
        """With no schema to write, "the schema an edit would not write" is meaningless."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.physx_attr).Set(0.6)

        self.assertFalse(gain_tuner.resolve_joint_param(joint, spec, "remotesim").resolved_from_fallback_schema)

    async def test_the_first_edit_ends_the_fallback(self) -> None:
        """Which is the half the annotation has to state: the value moves."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        joint.GetAttribute(spec.physx_attr).Set(0.6)
        gain_tuner.author_joint_param(joint, spec, 0.2, NEWTON)

        resolution = gain_tuner.resolve_joint_param(joint, spec, NEWTON, XPBD)
        self.assertFalse(resolution.resolved_from_fallback_schema)
        self.assertAlmostEqual(resolution.effective_value, 0.2, places=5)


class TestAuthorJointParam(_JointCase):
    """A write lands on the active backend's schema and nowhere else."""

    async def test_physx_edit_writes_only_the_physx_half(self) -> None:
        """The Newton half is left exactly as it was."""
        joint = self._joint()
        spec = gain_tuner.joint_param_spec("armature")
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.GetAttribute(spec.newton_attr).Set(0.25)

        written = gain_tuner.author_joint_param(joint, spec, 0.125, PHYSX)

        self.assertEqual(written.GetName(), spec.physx_attr)
        self.assertAlmostEqual(joint.GetAttribute(spec.physx_attr).Get(), 0.125, places=5)
        self.assertAlmostEqual(joint.GetAttribute(spec.newton_attr).Get(), 0.25, places=5)

    async def test_newton_edit_writes_only_the_newton_half(self) -> None:
        """The PhysX half keeps its own, independently tuned value."""
        joint = self._joint()
        spec = gain_tuner.joint_param_spec("joint_friction")
        joint.ApplyAPI(PHYSX_JOINT_API)
        joint.GetAttribute(spec.physx_attr).Set(0.75)

        written = gain_tuner.author_joint_param(joint, spec, 0.2, NEWTON)

        self.assertEqual(written.GetName(), spec.newton_attr)
        self.assertAlmostEqual(joint.GetAttribute(spec.newton_attr).Get(), 0.2, places=5)
        self.assertAlmostEqual(joint.GetAttribute(spec.physx_attr).Get(), 0.75, places=5)

    async def test_writing_applies_only_the_active_backend_api(self) -> None:
        """The inactive backend's schema is not applied as a side effect."""
        joint = self._joint()
        spec = gain_tuner.joint_param_spec("armature")

        gain_tuner.author_joint_param(joint, spec, 0.05, PHYSX)

        self.assertTrue(joint.HasAPI(PHYSX_JOINT_API))
        self.assertFalse(joint.HasAPI(NEWTON_JOINT_API))

    async def test_an_edit_never_authors_mjc(self) -> None:
        """MuJoCo gains are authored by hand; the tuner must not start writing them."""
        joint = self._joint()
        spec = gain_tuner.joint_param_spec("armature")
        gain_tuner.author_joint_param(joint, spec, 0.05, NEWTON)
        self.assertFalse(joint.GetAttribute(spec.mjc_attr).IsValid())

    async def test_authored_value_round_trips_through_resolution(self) -> None:
        """A written velocity limit resolves back rather than reading as unlimited."""
        joint = self._joint()
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        gain_tuner.author_joint_param(joint, spec, 45.0, PHYSX)
        resolution = gain_tuner.resolve_joint_param(joint, spec, PHYSX)
        self.assertAlmostEqual(resolution.effective_value, 45.0, places=5)

    async def test_writing_one_param_authors_only_that_param(self) -> None:
        """Applying a schema must not turn its other params into authored opinions.

        Both schemas resolve every one of their attributes once applied, so a
        caller that persists whatever resolves would save this joint's untouched
        velocity limit as an explicit `inf`.
        """
        joint = self._joint()
        gain_tuner.author_joint_param(joint, gain_tuner.joint_param_spec("armature"), 0.05, PHYSX)

        for key in ("joint_friction", "max_joint_velocity"):
            spec = gain_tuner.joint_param_spec(key)
            self.assertEqual(
                gain_tuner.authored_joint_param_attrs(joint, spec),
                (None, None),
                msg=f"{key} became authored as a side effect of writing armature",
            )

    async def test_non_numeric_value_writes_nothing(self) -> None:
        """A value that is not a number leaves the joint untouched."""
        joint = self._joint()
        spec = gain_tuner.joint_param_spec("armature")
        self.assertIsNone(gain_tuner.author_joint_param(joint, spec, "not-a-number", PHYSX))
        self.assertFalse(joint.HasAPI(PHYSX_JOINT_API))

    async def test_copy_writes_the_inactive_backend_only(self) -> None:
        """The opt-in copy action is the mirror image of an edit."""
        joint = self._joint()
        spec = gain_tuner.joint_param_spec("armature")
        joint.ApplyAPI(NEWTON_JOINT_API)
        joint.GetAttribute(spec.newton_attr).Set(0.25)

        written = gain_tuner.copy_joint_param_to_backend(joint, spec, 0.25, NEWTON)

        self.assertEqual(written.GetName(), spec.physx_attr)
        self.assertAlmostEqual(joint.GetAttribute(spec.physx_attr).Get(), 0.25, places=5)
        self.assertAlmostEqual(joint.GetAttribute(spec.newton_attr).Get(), 0.25, places=5)


class TestJointParamAttrs(_JointCase):
    """Attribute-pair resolution used by the save plan and the detail panel."""

    async def test_pair_reports_only_the_applied_schemas(self) -> None:
        """Each half is None until its API schema is applied."""
        joint = self._joint()
        spec = gain_tuner.joint_param_spec("armature")

        self.assertEqual(gain_tuner.joint_param_attrs(joint, spec), (None, None))

        joint.ApplyAPI(PHYSX_JOINT_API)
        newton_attr, physx_attr = gain_tuner.joint_param_attrs(joint, spec)
        self.assertIsNone(newton_attr)
        self.assertEqual(physx_attr.GetName(), spec.physx_attr)

        joint.ApplyAPI(NEWTON_JOINT_API)
        newton_attr, physx_attr = gain_tuner.joint_param_attrs(joint, spec)
        self.assertEqual(newton_attr.GetName(), spec.newton_attr)
        self.assertEqual(physx_attr.GetName(), spec.physx_attr)

    async def test_authored_pair_ignores_schema_fallbacks(self) -> None:
        """Only explicitly authored halves are reported for persistence."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")

        # Both attributes resolve, but neither carries an opinion worth saving.
        self.assertNotEqual(gain_tuner.joint_param_attrs(joint, spec), (None, None))
        self.assertEqual(gain_tuner.authored_joint_param_attrs(joint, spec), (None, None))

        joint.GetAttribute(spec.physx_attr).Set(0.3)
        newton_attr, physx_attr = gain_tuner.authored_joint_param_attrs(joint, spec)
        self.assertIsNone(newton_attr)
        self.assertEqual(physx_attr.GetName(), spec.physx_attr)


class TestAuthoredJointParamValues(_JointCase):
    """Every schema's authored opinion, keyed by resolver token."""

    async def test_unapplied_schemas_have_no_opinions(self) -> None:
        """A joint with no joint API reports each of its schemas as unauthored."""
        spec = gain_tuner.joint_param_spec("armature")
        self.assertEqual(
            gain_tuner.authored_joint_param_values(self._joint(), spec),
            {gain_tuner.SCHEMA_NEWTON: None, gain_tuner.SCHEMA_MJC: None, gain_tuner.SCHEMA_PHYSX: None},
        )

    async def test_velocity_limit_has_no_mjc_entry(self) -> None:
        """The map covers exactly the schemas the parameter exists on."""
        spec = gain_tuner.joint_param_spec("max_joint_velocity")
        self.assertEqual(
            set(gain_tuner.authored_joint_param_values(self._joint(), spec)),
            {gain_tuner.SCHEMA_NEWTON, gain_tuner.SCHEMA_PHYSX},
        )

    async def test_only_authored_values_are_reported(self) -> None:
        """Schema fallbacks are not opinions; an authored half is."""
        joint = self._both_schemas()
        spec = gain_tuner.joint_param_spec("armature")
        self.assertFalse(any(gain_tuner.authored_joint_param_values(joint, spec).values()))

        joint.GetAttribute(spec.newton_attr).Set(0.25)
        values = gain_tuner.authored_joint_param_values(joint, spec)
        self.assertAlmostEqual(values[gain_tuner.SCHEMA_NEWTON], 0.25, places=5)
        self.assertIsNone(values[gain_tuner.SCHEMA_PHYSX])
