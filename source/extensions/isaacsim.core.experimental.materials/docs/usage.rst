.. |_nbsp| unicode:: 0xA0 0xA0 0xA0 0xA0
    :trim:

Material specification
======================

Non-Visual Materials
--------------------

Non-visual materials determine how non-visual sensors interact with surfaces.
These materials enable the computation of reflection and transmission coefficients
based on surfaces' physical and spectral properties.

Attributes are authored following the `SimReady non-visual materials specification
<https://nvidia.github.io/simready-foundation/latest/capabilities/nonvisual_sensors/nonvisual_materials/capability-nonvisual_materials.html>`_,
using the following USD attributes:

- ``omni:simready:nonvisual:base`` (``token``) — the base material.
- ``omni:simready:nonvisual:coating`` (``token``) — optional surface coating.
- ``omni:simready:nonvisual:attributes`` (``token[]``) — optional surface attributes (one or more).

The complete set of valid values is defined in the `SimReady non-visual sensor attributes table
<https://nvidia.github.io/simready-foundation/latest/capabilities/nonvisual_sensors/nonvisual_materials/nonvisual_attributes_table.html>`_
and reproduced below.

These fields are encoded into a single unsigned 16-bit integer (``uint16``) material ID:
the lower byte holds the base material index, the lower 3 bits of the upper byte encode the coating,
and the upper 5 bits encode the attributes as a bitfield (so multiple attributes can be combined).

.. code-block:: text

    attributes  coatings    base material
    xxxxx       xxx         xxxxxxxx

Supported values
^^^^^^^^^^^^^^^^

The following tables details all the supported base materials, coatings, and attributes.

.. note::

    When parametrizing non-visual materials, the base material field is required,
    while the coatings and attributes fields are optional.

.. csv-table:: Base
    :file: ../data/specifications/base.csv
    :header-rows: 1

.. csv-table:: Coating
    :file: ../data/specifications/coating.csv
    :header-rows: 1

.. csv-table:: Attribute
    :file: ../data/specifications/attribute.csv
    :header-rows: 1
