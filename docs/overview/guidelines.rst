..
   Copyright (c) 2022-2026, NVIDIA CORPORATION. All rights reserved.
   NVIDIA CORPORATION and its licensors retain all intellectual property
   and proprietary rights in and to this software, related documentation
   and any modifications thereto. Any use, reproduction, disclosure or
   distribution of this software and related documentation without an express
   license agreement from NVIDIA CORPORATION is strictly prohibited.

.. _coding_style_guidelines:

=======================
Coding style guidelines
=======================

Use these conventions when you contribute C, C++, or Python code to Isaac Sim. C and C++ use the same casing for
types, functions, fields, parameters, and local variables, while Python follows PEP 8. Enumeration values retain
language-specific conventions because C enumerators are unscoped and C++ ``enum class`` values are scoped. The
repository formatters control mechanical layout; the rules on this page control naming, API design, ownership, error
handling, and documentation.

.. _coding_style_common:

Common conventions
==================

Use US English and descriptive, complete words for identifiers. Common technical abbreviations such as ``id``,
``url``, and ``GPU`` are acceptable, but apply the casing of the surrounding identifier. For example, use
``gpuBuffer``, ``HtmlPage``, and ``getUserId``.

Avoid redundant names and begin function names with verbs. Use ``get`` and ``set`` for inexpensive attribute access,
and use verbs such as ``compute``, ``read``, and ``write`` when an operation performs significant work. Give Boolean
variables stateful names, such as ``enabled`` or ``isInitialized``.

Do not use identifiers reserved by C or C++. In particular, avoid names that contain a double underscore, begin with
an underscore followed by an uppercase letter, or begin with an underscore at global scope.

Source files use the following extensions and names:

.. list-table:: Language and file naming
   :header-rows: 1

   * - Language
     - Headers or modules
     - Sources
   * - C
     - ``PascalCase.h``
     - ``PascalCase.c``
   * - C++
     - ``PascalCase.hpp``
     - ``PascalCase.cpp``
   * - Python
     - ``snake_case.py`` modules and ``snake_case`` packages
     - Not applicable.

End text files with a newline. Native source files include their own header first after the license banner. Public
headers include everything required to compile independently.

Formatting
----------

C and C++ use the repository ``.clang-format`` configuration: four-space indentation, a 120-character line limit,
and Allman braces. Put opening braces for namespaces, types, functions, and control statements on the following line.
Braced initializer expressions remain formatter-controlled.

Python uses four-space indentation, a 120-character line limit, Black, and isort with the Black profile. Use
parentheses instead of backslashes for line continuation. Do not manually override layout controlled by a formatter.

Run ``format_code.sh`` on Linux or ``format_code.bat`` on Windows before submitting a change.

.. _coding_style_c:

C
=

Language and API boundaries
---------------------------

Write portable C11 and isolate any required compiler extension behind a documented platform boundary. Use ``.c`` for
implementations and ``.h`` for C-compatible headers. Reserve ``.hpp`` for C++ headers.

A public C header must compile as both C11 and C++17. Include C standard-library headers such as ``<stddef.h>`` and
``<stdint.h>`` in the public interface, use ``#pragma once``, and wrap declarations in an ``extern "C"`` guard:

.. code-block:: c

   #pragma once

   #include <stddef.h>

   #ifdef __cplusplus
   extern "C"
   {
   #endif

   /* Public C declarations. */

   #ifdef __cplusplus
   }
   #endif

Do not expose C++ types, exceptions, overloads, references, templates, namespaces, or standard-library containers
through a C API. Never allow a C++ exception to cross a C boundary.

Naming
------

C uses the same identifier casing as C++ for equivalent constructs:

.. list-table:: C naming
   :header-rows: 1

   * - Construct
     - Convention
   * - Headers and sources
     - ``PascalCase.h`` and ``PascalCase.c``
   * - Structures, unions, enumerations, and typedefs
     - ``PascalCase``
   * - Exported and internal functions
     - ``camelCase``
   * - Structure fields, parameters, and local variables
     - ``camelCase``
   * - Macros and unscoped enumeration constants
     - Module-prefixed ``SCREAMING_SNAKE_CASE``

C enumerators enter the surrounding scope, so module-prefixed names prevent collisions. Keep these names when a C
header is included from C++. A C++ wrapper may map them to a scoped ``enum class`` with ``eCamelCase`` values.

Do not add a ``_t`` suffix to public typedefs. Prefix every exported function and type with the owning library or API
name because C has no namespaces. Flatten a dotted API name without separators. For example,
``isaacsim.common.logging`` uses ``isaacsimCommonLogging`` for function prefixes and
``IsaacSimCommonLogging`` for type prefixes.

.. code-block:: c

   typedef enum IsaacSimExampleResult
   {
       ISAACSIM_EXAMPLE_RESULT_SUCCESS = 0,
       ISAACSIM_EXAMPLE_RESULT_INVALID_ARGUMENT = 1,
   } IsaacSimExampleResult;

   typedef struct IsaacSimExampleCopyOptions
   {
       size_t structSize;
       size_t byteCount;
   } IsaacSimExampleCopyOptions;

   IsaacSimExampleResult isaacsimExampleCopy(
       const void* source, void* destination, const IsaacSimExampleCopyOptions* options);

API and ABI design
------------------

Use fixed-width integers when width is part of the ABI contract and ``size_t`` for object and buffer sizes. Put a
buffer pointer before its size. Document whether strings are null terminated and whether pointers may be ``NULL``.

State ownership and lifetime for every pointer that crosses the API. Use ``const`` for borrowed, read-only data, and
do not retain caller memory unless the contract says that you do.

Return explicit result values for recoverable failures. Prefer useful zero values so zero-initialized structures are
valid. Give extensible public structures a ``structSize`` or equivalent version field, and document how the callee
handles older and newer structure sizes.

Keep implementation symbols hidden and export only the declared C ABI. Avoid global mutable state. If global state is
unavoidable, document initialization, shutdown, synchronization, thread safety, and reentrancy.

Documentation
-------------

Use Doxygen comments for every exported function, typedef, structure, field, enumeration, enumeration value,
callback, constant, and public macro. Begin with ``@brief`` and document every parameter with ``@param[in]``,
``@param[out]``, or ``@param[in,out]``. Use ``@return`` to explain result categories.

Document pointer nullability, ownership, lifetime, buffer capacity, string termination, numeric units, valid ranges,
thread safety, callback behavior, and ABI initialization. Describe the caller-visible contract, not implementation
history. Do not use C++-specific concepts such as exceptions, namespaces, ``@class``, ``@tparam``, or ``@throws``.

.. _coding_style_cpp:

C++
===

Language and structure
----------------------

Write C++17. Do not use C++20 or newer features. Use ``.hpp`` for C++ headers and ``.cpp`` for implementations;
reserve ``.h`` for headers that expose a C-compatible interface.

Use ``#pragma once`` in headers. Derive namespaces from the owning API name, such as
``isaacsim.foo.bar`` to ``isaacsim::foo::bar``. Namespace names are lowercase, code inside namespaces is not
indented, and each namespace has a separate declaration rather than C++17 nested-namespace syntax. Use a ``details``
namespace for public-header implementation details and an anonymous namespace for translation-unit internals.

Naming
------

.. list-table:: C++ naming
   :header-rows: 1

   * - Construct
     - Convention
   * - Headers and sources
     - ``PascalCase.hpp`` and ``PascalCase.cpp``
   * - Classes, structures, enumerations, and typedefs
     - ``PascalCase``
   * - Constants
     - ``kCamelCase``
   * - Scoped enumeration values
     - ``eCamelCase``
   * - Public functions and members
     - ``camelCase``
   * - Private or protected functions
     - ``_camelCase``
   * - Private or protected members
     - ``m_camelCase``
   * - Private or protected static members
     - ``s_camelCase``
   * - File-scope static or constant variables
     - ``g_camelCase``
   * - Local variables
     - ``camelCase``
   * - Macros
     - ``SCREAMING_SNAKE_CASE``

Use ``eCamelCase`` only for scoped C++ enumeration values. Preserve module-prefixed ``SCREAMING_SNAKE_CASE``
enumerators from C-compatible headers rather than giving the same C declaration a second spelling in C++.

Class and type design
---------------------

List class access sections once each in ``public``, ``protected``, and ``private`` order. Put public data members
first and private or protected data members last. Put constructors and destructors before other functions in their
access section. Use ``override`` for overridden virtual functions.

Use structures for data-only types and do not add member functions to them. Prefer ``enum class`` for strongly typed
enumerations. Represent combinable bit flags with appropriately typed ``constexpr`` constants rather than an
enumeration that implies mutually exclusive values.

Use RAII for resource management. Prefer ``std::unique_ptr`` and ``std::shared_ptr`` for ownership; raw pointers and
references may express non-owning access when their lifetime is clear. Prefer ``std::string_view`` for borrowed text,
``std::optional`` for optional values, and standard containers over raw arrays.

Use ``constexpr`` or a function instead of a macro when possible. Avoid C-style casts, maintain const-correctness, and
use west-const style, such as ``const int``. Validate inputs at API boundaries and avoid unnecessary allocation.

Error handling and documentation
--------------------------------

Follow the owning API's documented error model. Use exceptions only when the public contract permits them, and use
explicit results for no-throw or C-compatible boundaries. Do not introduce a logging dependency for a single message.

Use Doxygen for public C++ types, functions, members, enumeration values, and template parameters. Describe the
current caller-visible contract, including ownership, lifetime, units, side effects, thread behavior, return values,
and exceptions that can actually be thrown. Use ``@brief``, directional ``@param`` tags, ``@tparam``, ``@return``, and
``@throws`` as applicable. Keep implementation details and change history out of API comments.

.. _coding_style_python:

Python
======

Language and naming
-------------------

Write idiomatic Python 3.10 or newer and follow PEP 8. A component may require a newer Python version. Use the
following naming conventions:

.. list-table:: Python naming
   :header-rows: 1

   * - Construct
     - Convention
   * - Modules and packages
     - ``snake_case``
   * - Classes and type variables
     - ``PascalCase``
   * - Exceptions
     - ``PascalCase`` ending in ``Error`` or ``Exception``
   * - Constants and enumeration members
     - ``SCREAMING_SNAKE_CASE``
   * - Public functions, methods, attributes, parameters, and local variables
     - ``snake_case``
   * - Private functions, methods, and attributes
     - ``_snake_case``

Use descriptive verbs for functions. Avoid trivial getter and setter methods; use direct access or a property when
appropriate. Use stateful Boolean names such as ``enabled`` and ``is_initialized``.

Type annotations
----------------

Annotate every function signature, return type, and class attribute. Include ``-> None`` when a function returns
nothing. Use modern built-in and PEP 604 syntax:

.. code-block:: python

   from __future__ import annotations

   def find_prim(paths: list[str], index: int | None = None) -> Usd.Prim | None:
       """Return the selected prim when it exists."""

Do not quote type names. Import ``annotations`` from ``__future__`` when postponed evaluation is required. Use
``X | None`` only when ``None`` is a valid value. Prefer ``list[T]``, ``dict[K, V]``, ``set[T]``, and
``tuple[T, ...]`` over their legacy ``typing`` equivalents.

Organization and error handling
-------------------------------

Put the module docstring after the license header, followed by standard-library, third-party, and project imports in
separate groups. Sort imports alphabetically within each group, prefer absolute imports, keep ``__init__.py`` files
minimal, and declare public module APIs with ``__all__``.

Prefer dataclasses or named tuples for data-only classes, context managers for resource ownership, ``pathlib.Path``
for paths, and comprehensions when they improve readability. Use ``None`` instead of a mutable default value.

Raise specific exception types, validate inputs at function boundaries, and use ``raise ... from ...`` to preserve
exception context. Never use a bare ``except`` clause. Use assertions only for debugging, not runtime validation.
Use the logging facility selected by the component; reserve ``print`` for intentional command-line or result output.

Documentation
-------------

Use Google-style docstrings for every public module, class, function, and method. Write the summary in imperative
mood, end sentences with periods, and describe the public contract rather than implementation history. Put types in
annotations, not docstrings, and do not repeat default values in parameter descriptions.

Document constructor arguments and raised exceptions in the class docstring, not in ``__init__``. Document class
attributes immediately below each attribute with Sphinx ``#:`` comments rather than an ``Attributes`` section.

Use ``Args``, ``Returns``, and ``Raises`` sections as applicable. Explain when an optional return is ``None``. Avoid an
unescaped colon in a ``Returns`` description because Sphinx can interpret the preceding text as a type. Include an
``Example`` section for public functions, methods, and properties unless the API is deprecated.

.. code-block:: python

   def get_depth_range(self) -> tuple[float, float]:
       """Get the sensor depth range.

       Returns:
           Minimum and maximum depth values in meters.

       Example:

           .. code-block:: python

               minimum_depth, maximum_depth = sensor.get_depth_range()
       """

Native bindings
---------------

Python-facing documentation strings in native binding sources follow the Python rules, including Python names and
generated signatures. The surrounding binding implementation follows the C++ style and Doxygen rules. Document
ownership, lifetime, mutability, thread behavior, and Global Interpreter Lock behavior when Python callers can
observe them.

.. _coding_style_testing:

Testing conventions
===================

Use the test framework selected by the owning component. Native unit tests use doctest where the component provides
it; group related tests with ``TEST_SUITE`` and name individual cases with ``TEST_CASE``. Python tests use descriptive
names such as ``test_<function_name>_<scenario>`` and remain isolated from external state.

Test API boundaries, invalid inputs, ownership behavior, and critical code paths. A public C header must have coverage
that compiles it with both a C11 compiler and a C++17 compiler.

For additional Carbonite-specific C++ guidance, see the `Carbonite coding style guide
<https://docs.omniverse.nvidia.com/kit/docs/carbonite/latest/CODING.html>`_. Isaac Sim conventions on this page take
precedence when the guidance differs.
