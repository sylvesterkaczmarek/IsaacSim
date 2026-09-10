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

.. _isaacsim-foundation-objects-api-cpp:

=========
C++ guide
=========

.. isaacsim-libraries-api-guide-start

All declarations live in the ``isaacsim::foundation::objects`` namespace.

.. code-block:: cpp

    #include <isaacsim/foundation/objects/Stage.hpp>
    #include <isaacsim/foundation/objects/Prim.hpp>
    // shapes, lights, camera follow the same pattern

    using namespace isaacsim::foundation::objects;

.. _isaacsim-foundation-objects-api-cpp-stage:

Stage
=====

``Stage`` wraps a USD-compatible stage identified by a numeric ID. The constructor
requires a ``backend`` string — either ``"openusd"`` or ``"ovstage"`` — and
an optional numeric stage ID to attach to an existing stage. Call
``openStage`` or ``createStage`` to associate it with a file, then
``saveStage`` / ``closeStage`` to complete the lifecycle:

.. code-block:: cpp

    Stage stage("openusd");
    stage.createStage();                   // blank stage
    stage.definePrim("/World", "Xform");
    stage.addReference("/path/to/robot.usd", "/World/Robot");
    stage.saveStage("/tmp/scene.usda");
    stage.closeStage();

Use ``isValid()`` to guard against an uninitialized or closed stage:

.. code-block:: cpp

    if (!stage.isValid())
        stage.openStage("/assets/scene.usda");

``exportStageToString`` / ``importStageFromString`` round-trip the root layer
as a USDA string without touching the filesystem:

.. code-block:: cpp

    std::string usda = stage.exportStageToString();
    Stage copy("openusd");
    copy.importStageFromString(usda);

``generateStringRepresentation`` returns a human-readable summary of the
stage hierarchy (``"tree"`` or ``"list"`` mode):

.. code-block:: cpp

    std::cout << stage.generateStringRepresentation("tree");

Stage metadata (up-axis, units, time codes) can be read and written at any
time:

.. code-block:: cpp

    stage.setUpAxis("Z");
    stage.setUnits(/*metersPerUnit=*/0.01f, /*kilogramsPerUnit=*/1.0f);
    auto [metersPerUnit, kilogramsPerUnit] = stage.getUnits();
    auto [start, end, fps] = stage.getTimeCode();

.. _isaacsim-foundation-objects-api-cpp-prim:

Prim
====

``Prim`` wraps one or more USD prim paths and provides batch operations for
attributes, schemas, and variant selections. Pass a single path string or a
``std::vector<std::string>``; regular expressions are expanded against the
active stage unless ``resolvePaths`` is ``false``:

.. code-block:: cpp

    Prim cube("/World/Cube");
    Prim robots({ "/World/Robot0", "/World/Robot1" });

    // Query type and hierarchy
    auto names   = cube.getName();      // { "Cube" }
    auto parents = robots.getParent();  // { "/World", "/World" }

    // Read and write a custom attribute across all wrapped prims
    robots.createAttribute("mass", "float");
    robots.setAttributeValues("mass", array::Array{ 10.0f, 12.5f });

    // Apply a USD API schema
    cube.applyApi("PhysicsRigidBodyAPI");

Variant selections can be queried and changed in one call:

.. code-block:: cpp

    auto sel = robots.getVariantSelection();  // { {set -> variant}, ... }
    robots.setVariantSelection({ {"lod", "high"} });

.. _isaacsim-foundation-objects-api-cpp-xform:

Xform
=========

``Xform`` extends ``Prim`` with pose (position/orientation) and scale
management in both world and local frames, visibility control, and a
default-state mechanism for resetting prims to a previously recorded pose.
Transform operations on non-root articulation links are not supported.

Include: ``<isaacsim/foundation/objects/Xform.hpp>``

.. _isaacsim-foundation-objects-api-cpp-camera:

Camera
======

``Camera`` extends ``Xform`` and wraps one or more ``UsdGeomCamera`` prims.
It exposes batch get/set methods for focal length, focus distance, aperture,
f-stop, projection type, clipping range, shutter timing, and stereo role.

Include: ``<isaacsim/foundation/objects/Camera.hpp>``

.. _isaacsim-foundation-objects-api-cpp-mesh:

Mesh
====

``Mesh`` extends ``Xform`` and wraps one or more ``UsdGeomMesh`` prims. It
exposes batch get/set methods for points, normals, display colors, and face,
crease, corner, and subdivision specifications. Mesh attributes are ragged, so
these methods take and return one array per prim rather than a single batched
array. When creating prims, it can generate the geometry of a set of built-in
primitives (``Cone``, ``Cube``, ``Cylinder``, ``Disk``, ``Plane``, ``Sphere``,
``Torus``).

Include: ``<isaacsim/foundation/objects/Mesh.hpp>``

.. _isaacsim-foundation-objects-api-cpp-shapes:

Shapes
======

``Shape`` is the abstract base class that extends ``Xform`` with display
color control and a ``updateExtents`` interface. Concrete subclasses
(``Capsule``, ``Cone``, ``Cube``, ``Cylinder``, ``Plane``, ``Sphere``) author
the corresponding ``UsdGeom`` prim and keep USD extents synchronized with their
geometry attributes.

Include: ``<isaacsim/foundation/objects/shapes/<ClassName>.hpp>``

.. _isaacsim-foundation-objects-api-cpp-lights:

Lights
======

``Light`` is the abstract base class that extends ``Xform`` with intensity,
color, exposure, and shadow controls. Concrete subclasses (``CylinderLight``,
``DiskLight``, ``DistantLight``, ``DomeLight``, ``RectLight``, ``SphereLight``)
wrap the corresponding ``UsdLux`` prim type.

Include: ``<isaacsim/foundation/objects/lights/<ClassName>.hpp>``
