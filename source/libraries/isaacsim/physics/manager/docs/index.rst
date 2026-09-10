..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _isaacsim-physics-manager:
.. _isaacsim-physics-manager-overview:

isaacsim.physics.manager
========================

The standalone physics modules expose two complementary APIs:

* :mod:`isaacsim.physics.registration` is the engine-facing registration API.
  Backends publish their simulation callbacks and tensor factories there.
* :mod:`isaacsim.physics.manager` is the application-facing API. Applications
  attach stages, step simulation, issue scene queries, subscribe to events, and
  create tensor views there.

Both C++ APIs use declarations from public headers directly. There are no
interface objects to acquire or ``get_*_interface`` functions.

Registering a backend
---------------------

A backend fills a :class:`isaacsim.physics.registration.Simulation` with the
callback groups it supports, then registers that value under a unique name.
Unsupported callbacks may remain unset; manager operations skip backends that do
not implement the requested operation.

.. code-block:: python

    import isaacsim.physics.registration as registration

    simulation = registration.Simulation()
    simulation.simulation_fns.initialize = lambda ovstage, usd_identifier: True
    simulation.simulation_fns.close = lambda: True
    simulation.simulation_fns.get_attached_stage = lambda: 0
    simulation.simulation_fns.has_attached_stage = lambda: False
    simulation.simulation_fns.simulate = lambda elapsed_time, current_time: None
    simulation.simulation_fns.simulate_async = lambda elapsed_time, current_time: None
    simulation.simulation_fns.fetch_results = lambda: None
    simulation.simulation_fns.check_results = lambda: True

    simulation_id = registration.register_simulation(simulation, "MyEngine")
    try:
        print(registration.get_simulation_name(simulation_id))
    finally:
        registration.unregister_simulation(simulation_id)

The callback groups are ``simulation_fns``, ``scene_query_fns``,
``interaction_fns``, and ``benchmark_fns``.

Driving simulation
------------------

Lifecycle and stepping operations are broadcast to every active registered
backend. ``initialize`` takes a caller-built native stage handle and the
identifier format expected by the selected backend. For example, Newton resolves
a numeric ``pxr.UsdUtils.StageCache`` identifier encoded as a string, while
OvPhysX consumes the supplied ovstage handle. A filesystem path is not a portable
stage identifier.

Pass the owning Python stage object through ``owner`` when the native handle is
caller-owned. The binding keeps it alive until every backend confirms detach.

.. code-block:: python

    import isaacsim.physics.manager as manager

    # ovstage_handle, usd_identifier, and ovstage_owner are created by the caller.
    if not manager.initialize(ovstage_handle, usd_identifier, owner=ovstage_owner):
        raise RuntimeError("physics initialization failed")

    try:
        manager.simulate(1.0 / 60.0, 0.0)
        manager.simulate_async(1.0 / 60.0, 1.0 / 60.0)
        manager.fetch_results()
    finally:
        if not manager.close():
            raise RuntimeError("a backend did not detach its stage")

Subscriptions
-------------

Registry, step, contact, and profile-stat subscriptions return a
``Subscription`` object. Retain it for as long as events are needed and call
``unsubscribe()`` for deterministic release.

.. code-block:: python

    def on_step(elapsed_time, context):
        print(elapsed_time, int(context.simulation_id))


    subscription = manager.subscribe_physics_on_step_events(
        pre_step=False,
        order=0,
        on_update=on_step,
    )
    try:
        manager.simulate(1.0 / 60.0, 0.0)
    finally:
        subscription.unsubscribe()

Scene queries
-------------

Scene queries use value types from the registration module and free functions
from the manager module.

.. code-block:: python

    import isaacsim.physics.manager as manager
    import isaacsim.physics.registration as registration

    origin = registration.Float3(0.0, 0.0, 10.0)
    direction = registration.Float3(0.0, 0.0, -1.0)
    hit_found, hit = manager.raycast_closest(
        origin=origin,
        unit_dir=direction,
        distance=100.0,
        both_sides=False,
    )
    if hit_found:
        print(hit.distance, hit.position)

The manager provides closest, any, and reporting-callback variants for raycasts,
sphere and box sweeps, authored-shape sweeps, and overlaps.

Tensor views
------------

Engine modules register tensor factories with
:mod:`isaacsim.physics.registration`. Applications request the high-level Warp
view through the manager:

.. code-block:: python

    simulation_view = manager.create_simulation_view(
        engine="ovphysx",
        stage_id=simulation_id.id,
        frontend_name="warp",
    )
    rigid_bodies = simulation_view.create_rigid_body_view("/World/envs/*/Cube")
    transforms = rigid_bodies.get_data("transforms")

The ``stage_id`` factory argument is the registered simulation identifier used by
the backend adapter; it is distinct from the ``usd_identifier`` passed to
``initialize``.

C++ usage
---------

Include only the operation headers needed by the caller and link
``isaacsim::physics-manager`` and ``isaacsim::physics-registration``.

.. code-block:: cpp

    #include <isaacsim/physics/manager/PhysicsSimulation.hpp>
    #include <isaacsim/physics/registration/Physics.hpp>

    isaacsim::physics::registration::Simulation simulation;
    // Populate the callback groups supported by this backend.
    const auto id =
        isaacsim::physics::registration::registerSimulation(simulation, "MyEngine");

    const auto result =
        isaacsim::physics::manager::initialize(ovstageHandle, usdIdentifier);
    if (result == isaacsim::physics::manager::InitializeResult::eOk)
    {
        isaacsim::physics::manager::simulate(1.0f / 60.0f, 0.0f);
        isaacsim::physics::manager::close();
    }

    isaacsim::physics::registration::unregisterSimulation(id);

.. toctree::
    :maxdepth: 2

    api_cpp
    api_python
