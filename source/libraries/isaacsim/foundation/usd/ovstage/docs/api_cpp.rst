..
   Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.

.. _isaacsim-foundation-usd-ovstage-api-cpp:

=========
C++ guide
=========

.. isaacsim-libraries-api-guide-start

Include ``isaacsim/foundation/usd/ovstage/Usd.hpp`` and link
``isaacsim::foundation-usd-ovstage``. All declarations live in the
``isaacsim::foundation::usd::ovstage`` namespace.

.. code-block:: cpp

    #include <isaacsim/foundation/usd/ovstage/Usd.hpp>

    using namespace isaacsim::foundation::usd::ovstage;

Stages are identified by an opaque ``int64_t`` handle returned by
:ref:`createStage <isaacsim-foundation-usd-ovstage-api-cpp-stage>` or
:ref:`openStage <isaacsim-foundation-usd-ovstage-api-cpp-stage>`. Pass this
handle to every subsequent call. A value of ``-1`` indicates failure or an
invalid stage.

.. _isaacsim-foundation-usd-ovstage-api-cpp-stage:

Stage
=====

``createStage`` and ``openStage`` return an opaque stage identifier. Every
other function in this section takes that identifier as its first argument.
All prim paths are absolute ``SdfPath`` strings (e.g. ``"/World/Prim"``).

.. code-block:: cpp

    // Create a new, empty in-memory stage.
    int64_t stageId = createStage();

    // Open an existing USD file.
    int64_t stageId = openStage("/path/to/scene.usd");

    // Validate and release.
    bool valid = isStageValid(stageId);
    bool closed = closeStage(stageId);

Authoring prims
---------------

.. code-block:: cpp

    // Define a prim, creating ancestor Xform prims as needed.
    definePrim(stageId, "/World/Prim", "Xform");

    // Move a prim to a new path.
    auto [ok, error] = movePrim(stageId, "/World/OldName", "/World/NewName");

    // Remove a prim and all its descendants.
    bool removed = removePrim(stageId, "/World/Prim");

Traversal
---------

``traversePrim`` visits every prim in the subtree rooted at ``path``. Return
``false`` from the callback to stop early:

.. code-block:: cpp

    traversePrim(stageId, "/World",
        [](const std::string& primPath) -> bool
        {
            // process primPath ...
            return true; // continue
        });

    // Find prims whose path string matches a pattern.
    std::vector<std::string> matches =
        findMatchingPrimPaths(stageId, "*Prim*", /*traverse=*/true);

Stage representation
--------------------

``generateStageRepresentation`` produces a human-readable summary of the
stage hierarchy. Supported ``mode`` values are ``"tree"`` (indented
parent-child tree) and ``"list"`` (flat list of paths):

.. code-block:: cpp

    std::string tree = generateStageRepresentation(stageId, "tree");
    std::string list = generateStageRepresentation(stageId, "list");

.. _isaacsim-foundation-usd-ovstage-api-cpp-sdfpath:

SdfPath
=======

.. code-block:: cpp

    bool valid = isValidPathString("/World/Prim");   // true
    bool invalid = isValidPathString("not/absolute"); // false

.. _isaacsim-foundation-usd-ovstage-api-cpp-prim:

Prim
====

Validity and hierarchy
----------------------

.. code-block:: cpp

    bool exists   = isPrimValid(stageId, "/World/Prim");
    std::string name     = getName(stageId, "/World/Prim");   // "Prim"
    std::string typeName = getTypeName(stageId, "/World/Prim"); // e.g. "Xform"
    std::string parent   = getParent(stageId, "/World/Prim");   // "/World"
    std::vector<std::string> children = getChildren(stageId, "/World");

Schema type and API schemas
---------------------------

``isA`` tests whether a prim's type is or derives from a given schema type.
``hasApi``, ``applyApi``, and ``removeApi`` operate on applied API schemas.
Pass a non-empty ``instanceName`` for multiple-apply schemas:

.. code-block:: cpp

    bool isMesh = isA(stageId, "/World/Mesh", "Mesh");

    bool hasPhysics = hasApi(stageId, "/World/Prim", "PhysicsRigidBodyAPI");

    // Single-apply API schema.
    applyApi(stageId, "/World/Prim", "PhysicsRigidBodyAPI");

    // Multiple-apply API schema (e.g. material binding).
    applyApi(stageId, "/World/Prim", "MaterialBindingAPI", "preview");
    removeApi(stageId, "/World/Prim", "MaterialBindingAPI", "preview");

    std::vector<std::string> schemas = getAppliedSchemas(stageId, "/World/Prim");

.. _isaacsim-foundation-usd-ovstage-api-cpp-attribute:

Prim Attribute
==============

Creating and removing attributes
---------------------------------

.. code-block:: cpp

    // Create a "float" attribute named "mass" on the prim.
    bool created = createPrimAttribute(stageId, "/World/Prim", "mass", "float");

    // Remove it.
    bool removed = removePrimAttribute(stageId, "/World/Prim", "mass");

Inspecting attributes
----------------------

.. code-block:: cpp

    std::string typeName =
        getPrimAttributeTypeName(stageId, "/World/Prim", "mass"); // "float"

    std::vector<std::string> attrNames =
        getPrimAttributeNames(stageId, "/World/Prim");

Reading and writing values
---------------------------

``getPrimAttributeValues`` and ``setPrimAttributeValues`` operate on multiple
prims in a single call to minimize round-trips:

.. code-block:: cpp

    std::vector<std::string> paths = { "/World/Cube1", "/World/Cube2" };

    // Read "size" from both prims.
    auto values = getPrimAttributeValues(stageId, paths, "size");

    if (auto* arr = std::get_if<array::Array>(&values))
    {
        // Numeric attribute - process arr ...
    }

    // Write new values to both prims.
    array::Array newSizes(std::vector<float>{ 1.0f, 2.0f });
    std::vector<bool> success = setPrimAttributeValues(stageId, paths, "size", newSizes);

.. _isaacsim-foundation-usd-ovstage-api-cpp-xform:

Xform
=====

All Xform functions operate on **Xformable** prims (``Xform``, ``Mesh``, etc.).
Translations and positions use ``[x, y, z]`` order. Orientations are quaternions
in ``[w, ix, iy, iz]`` order.  Scale vectors use ``[sx, sy, sz]`` order.

Scales
------

.. code-block:: cpp

    std::vector<std::string> paths = { "/World/A", "/World/B" };

    // Read local scales - returns an Array of shape (N, 3).
    array::Array scales = getXformLocalScales(stageId, paths);

    // Write local scales - preserves the existing local translation and rotation.
    array::Array newScales(std::vector<std::vector<double>>{ { 2.0, 3.0, 4.0 },
                                                             { 5.0, 6.0, 7.0 } });
    setXformLocalScales(stageId, paths, newScales);

Local poses
-----------

``getXformLocalPoses`` returns a tuple of two arrays:
the translations (shape ``(N, 3)``) and the orientations (shape ``(N, 4)``).

``setXformLocalPoses`` accepts optional translations and orientations; pass
``std::nullopt`` to leave a component unchanged. The existing local scale is
always preserved.

.. code-block:: cpp

    // Read local poses.
    auto [translations, orientations] = getXformLocalPoses(stageId, paths);
    auto t = translations.get<std::vector<std::vector<double>>>();  // [x, y, z]
    auto q = orientations.get<std::vector<std::vector<double>>>(); // [w, ix, iy, iz]

    // Write both translation and orientation.
    array::Array newTranslations(std::vector<std::vector<double>>{ { 1.0, 2.0, 3.0 },
                                                                   { 4.0, 5.0, 6.0 } });
    array::Array newOrientations(std::vector<std::vector<double>>{ { 1.0, 0.0, 0.0, 0.0 },
                                                                   { 1.0, 0.0, 0.0, 0.0 } });
    setXformLocalPoses(stageId, paths, newTranslations, newOrientations);

World poses
-----------

``getXformWorldPoses`` returns the same tuple layout as ``getXformLocalPoses``
but in world space.  ``setXformWorldPoses`` back-computes the required local
transform so that the prim ends up at the requested world pose; the existing
local scale is preserved.

.. code-block:: cpp

    // Read world poses.
    auto [positions, orientations] = getXformWorldPoses(stageId, paths);

    // Move both prims to new world positions, keeping their current orientations.
    array::Array newPositions(std::vector<std::vector<double>>{ { 10.0, 0.0, 0.0 },
                                                                {  0.0, 5.0, 0.0 } });
    setXformWorldPoses(stageId, paths, newPositions, std::nullopt);

    // Set world orientation only.
    const double s = std::sqrt(2.0) / 2.0;          // 90° around Z
    array::Array newOrientations(std::vector<std::vector<double>>{ { s, 0.0, 0.0, s } });
    setXformWorldPoses(stageId, { "/World/A" }, std::nullopt, newOrientations);
