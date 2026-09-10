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

.. _isaacsim-common-exceptions-api-python:

==============================
Python guide and API reference
==============================

The Python API exposes the same exception hierarchy as the C++ API through nanobind. Every exception is a subclass of
``IsaacSimException``, which is itself a subclass of the built-in ``Exception``.

.. _isaacsim-common-exceptions-python-summary:

Summary
========

* ``IsaacSimException``
* ``PrimPathError``
* ``PrimPathStringError``
* ``AttributeNameError``
* ``ValueTypeError``

.. _isaacsim-common-exceptions-python-base:

IsaacSimException
==================

Catch ``IsaacSimException`` to handle any exception raised by Isaac Sim code without depending on a specific
exception subtype:

.. code-block:: python

    from isaacsim.common.exceptions import IsaacSimException

    try:
        ...
    except IsaacSimException as error:
        print(str(error))

.. _isaacsim-common-exceptions-python-invalid-prim-path:

PrimPathError
================

Raised when a USD prim path does not refer to a usable prim. Exposes the offending path through ``prim_path``:

.. code-block:: python

    from isaacsim.common.exceptions import PrimPathError

    try:
        ...
    except PrimPathError as error:
        print(error.prim_path)

.. _isaacsim-common-exceptions-python-invalid-prim-path-string:

PrimPathStringError
=======================

Raised when a string does not parse as a valid USD prim path. Exposes the offending string through
``prim_path``:

.. code-block:: python

    from isaacsim.common.exceptions import PrimPathStringError

    try:
        ...
    except PrimPathStringError as error:
        print(error.prim_path)

.. _isaacsim-common-exceptions-python-invalid-attribute-name:

AttributeNameError
======================

Raised when a caller references an attribute name that does not exist or is not accepted. Exposes the offending name
through ``attribute_name``.

.. _isaacsim-common-exceptions-python-invalid-value-type:

ValueTypeError
==================

Raised when a value's type does not match the type expected for an attribute. Exposes ``attribute_name``,
``expected_type``, and ``actual_type``:

.. code-block:: python

    from isaacsim.common.exceptions import ValueTypeError

    try:
        ...
    except ValueTypeError as error:
        print(f"{error.attribute_name}: expected {error.expected_type}, got {error.actual_type}")
