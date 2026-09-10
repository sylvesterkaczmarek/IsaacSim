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

.. _isaacsim-common-string-api-cpp:

=========
C++ guide
=========

.. isaacsim-libraries-api-guide-start

Include ``isaacsim/common/string/String.hpp`` and link ``isaacsim::common-string``.

All examples on this page assume the following namespace alias:

.. code-block:: cpp

    #include <isaacsim/common/string/String.hpp>

    namespace string = isaacsim::common::string;

.. _isaacsim-common-string-api-cpp-case:

Case conversion
================

``toLower()`` and ``toUpper()`` return a copy of the input string with every
character converted to lowercase or uppercase, respectively:

.. code-block:: cpp

    string::toLower("Isaac Sim");  // "isaac sim"
    string::toUpper("Isaac Sim");  // "ISAAC SIM"

.. _isaacsim-common-string-api-cpp-color:

Color resolution
=================

``resolveColor()`` parses a matplotlib-style color specification and returns its
RGB components as three normalized floats in the 0-1 range. It mirrors
matplotlib's ``to_rgb`` and accepts:

- ``#rgb`` / ``#rrggbb`` short and full hex (case-insensitive; ``#rgba`` /
  ``#rrggbbaa`` is accepted and the alpha channel is discarded).
- Grayscale strings such as ``"0.5"`` (a float in the 0-1 range).
- Single-letter base colors such as ``"k"``.
- X11/CSS4 named colors such as ``"AquaMarine"`` (case-insensitive).
- Tableau colors such as ``"tab:Green"`` (case-insensitive).
- The ``CN`` cycle specification, such as ``"C2"``.
- ``"none"`` (case-insensitive), the fully transparent color, resolved to
  black ``{0, 0, 0}``.

.. code-block:: cpp

    string::resolveColor("#ff8800");   // {1.0f, 0.533f, 0.0f}
    string::resolveColor("tab:green"); // Tableau green
    string::resolveColor("C2");        // third color in the default cycle
    string::resolveColor("0.5");       // {0.5f, 0.5f, 0.5f}

``resolveColor()`` throws ``std::invalid_argument`` when the input is not a
recognized color specification.
