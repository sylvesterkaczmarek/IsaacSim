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

"""Exercise core ``SimulationView`` behavior across tensor backends.

Each test class is a mixin that
``backends/test_<engine>_tensors.py`` subclasses with the engine selected
via ``backend`` / ``frontend`` class attributes.

This module covers registry round-trips, entity registration, the supported
view matrix, and compatibility method delegation. Detailed operation contracts
are exercised by the scenario modules under ``common``.

"""

from __future__ import annotations

import isaacsim.physics.manager.impl.tensors as t
import pytest


class SimulationViewBasicsMixin:
    """Backend-agnostic tests for SimulationView creation and queries."""

    backend: str = "<override>"
    # Entity types that should report supports=False for this backend.
    # Override in subclasses as each engine adds real implementations.
    # Default unsupported list for Newton; ovphysx overrides to empty.
    unsupported_entity_types: tuple[str, ...] = (
        "sdf-shape",
        "volume-deformable-body",
        "surface-deformable-body",
        "deformable-material",
    )

    def test_engine_registered(self) -> None:
        """Verify that registry enumeration contains the selected backend."""
        engines = set(t.get_registry().list_engines())
        assert self.backend in engines

    def test_all_seven_entity_types_registered(self) -> None:
        """Verify that the backend registers all seven tensor entity types."""
        entities = set(t.get_registry().list_entities(self.backend))
        assert entities == {
            "articulation",
            "deformable-material",
            "rigid-body",
            "rigid-contact",
            "sdf-shape",
            "surface-deformable-body",
            "volume-deformable-body",
        }

    def test_unsupported_views_report_supports_false(self) -> None:
        """Verify that unsupported entity operations explicitly report no support."""
        for entity in self.unsupported_entity_types:
            view = t.create_entity(self.backend, entity, ["/World/X"])
            impls = view.list_impls(t.ImplKind.Get)
            assert len(impls) > 0, f"{self.backend}/{entity} did not register a get implementation"
            for impl in impls:
                assert not view.has_impl(
                    impl, t.ImplKind.Get
                ), f"{self.backend}/{entity} unexpectedly reports support for {impl!r}"

    def test_create_entity_matches_simulation_view_construction(self) -> None:
        """Verify the two construction paths agree, per entity type.

        ``create_entity`` resolves the active simulation and calls the same internal factory
        ``SimulationView.create_*_view`` uses, so the two cannot disagree unless someone reintroduces a
        second implementation. This guards that seam: the placeholder bug this replaced went unnoticed
        precisely because nothing compared the two paths.
        """
        builders = self._simulation_view_builders()
        if not builders:
            pytest.skip(f"{self.backend}: no active simulation, so the two construction paths cannot be compared")
        paths = ["/World/X"]
        for entity, create_view in builders.items():
            entity_view = t.create_entity(self.backend, entity, paths)
            simulation_built = create_view(paths)
            assert set(entity_view.list_impls(t.ImplKind.Get)) == set(
                simulation_built.list_impls(t.ImplKind.Get)
            ), f"{self.backend}/{entity}: create_entity and create_*_view registered different get impls"
            for impl in entity_view.list_impls(t.ImplKind.Get):
                assert entity_view.has_impl(impl, t.ImplKind.Get) == simulation_built.has_impl(
                    impl, t.ImplKind.Get
                ), f"{self.backend}/{entity}: the two paths disagree on support for {impl!r}"

    def _simulation_view_builders(self) -> dict[str, object]:
        """Map each entity type to the simulation-view call that builds it.

        Returns:
            Entity type to a callable taking prim paths and returning the view, for the entity types this
            backend can construct without a live simulation.
        """
        stage_id = self._stage_id()
        if stage_id < 0:
            return {}
        simulation_view = t.create_simulation_view(engine=self.backend, stage_id=stage_id)
        if simulation_view is None:
            return {}
        return {
            "articulation": simulation_view.create_articulation_view,
            "rigid-body": simulation_view.create_rigid_body_view,
            "volume-deformable-body": simulation_view.create_volume_deformable_body_view,
            "surface-deformable-body": simulation_view.create_surface_deformable_body_view,
            "deformable-material": simulation_view.create_deformable_material_view,
        }

    def _stage_id(self) -> int:
        """Resolve the identifier the simulation-view factory expects for this backend.

        Returns:
            The active simulation's integer identifier, or ``-1`` when none is active.
        """
        from isaacsim.physics.registration import get_active_simulation_id, k_invalid_simulation_id

        simulation_id = get_active_simulation_id(self.backend)
        return -1 if simulation_id == k_invalid_simulation_id else int(simulation_id.id)


class LegacyMethodDelegationMixin:
    """Verify compatibility-method delegation on ``_LegacyAdapter`` subclasses.

    The wrapped adapter exposes explicit compatibility method names
    (`view.get_dof_positions()`, `view.set_dof_positions(data, indices)`,
    and similar operations). Its ``__getattr__`` implementation maps those
    calls to the string-keyed tensor API, such as
    ``view.get_data("dof-positions")``.
    """

    backend: str = "<override>"
    # Engine module that exposes a `_LegacyAdapter` for inspection.
    adapter_module: str = "<override>"

    def test_adapter_has_getattr_delegation(self) -> None:
        """Verify that the backend adapter defines compatibility-method delegation."""
        import importlib

        mod = importlib.import_module(self.adapter_module)
        adapter_cls = getattr(mod, "_LegacyAdapter")
        assert "__getattr__" in adapter_cls.__dict__, (
            f"{adapter_cls.__name__} must define __getattr__ to delegate "
            f"unknown attributes to the wrapped compatibility view"
        )
