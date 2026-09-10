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

.. _isaacsim-common-array-api-cpp:

=========
C++ guide
=========

.. isaacsim-libraries-api-guide-start

Include ``isaacsim/common/array/Array.hpp`` and link ``isaacsim::common-array``.
The header transitively includes ``Device.hpp``, ``Dtype.hpp``, and ``Shape.hpp``.

All examples on this page assume the following namespace alias:

.. code-block:: cpp

    #include <isaacsim/common/array/Array.hpp>

    namespace array = isaacsim::common::array;

.. _isaacsim-common-array-api-cpp-array:

Array
=====

``array::Array`` is a multi-dimensional array backed by a reference-counted buffer
on either the CPU or a CUDA device.

Construction
------------

Construct an ``array::Array`` from a scalar, a ``std::vector``, or a
``std::vector<std::vector>``. The element type is inferred from the value unless
you supply an explicit ``array::DType``:

.. code-block:: cpp

    // Scalar — produces a 0-D float32 array.
    array::Array scalar(3.14f);

    // 1-D — produces a float32 array of shape (3,).
    array::Array vector(std::vector<float>{1.0f, 2.0f, 3.0f});

    // 2-D with explicit dtype — produces an int32 array of shape (2, 3).
    array::Array matrix(std::vector<std::vector<int32_t>>{{1, 2, 3}, {4, 5, 6}},
                        array::DType::Int32());

    // On a CUDA device.
    array::Array cuda_array(std::vector<float>{1.0f, 2.0f, 3.0f}, std::nullopt, array::Device::Cuda());

Inspection
----------

.. code-block:: cpp

    array::Array a(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    a = a.reshape(array::Shape({2, 3}));

    a.shape();   // array::Shape({2, 3})
    a.ndim();    // 2
    a.size();    // 6
    a.nbytes();  // 24
    a.dtype();   // array::DType::Float32()
    a.device();  // array::Device::Cpu()

Use ``data()`` to obtain a raw pointer to element 0, or ``buffer()`` to retrieve
the underlying shared allocation (which may start before element 0 for sliced views):

.. code-block:: cpp

    const float* ptr = static_cast<const float*>(a.data());

Views and reshaping
-------------------

``reshape()`` and ``at()`` return views over the same buffer with no data duplication;
``copy()`` produces an independent, compacted deep copy:

.. code-block:: cpp

    array::Array flat(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f});

    // Infer the second dimension with -1.
    array::Array matrix = flat.reshape(array::Shape({2, -1}));  // shape (2, 2)

    // Slice along axis 0 — returns a (2,) view.
    array::Array row = matrix.at(0);

    // Negative indices count from the end.
    array::Array last_row = matrix.at(-1);

    // Deep copy — independent, compacted buffer (offset reset to 0).
    array::Array compact = matrix.copy();

    // Deep copy — preserves full buffer layout including any pre-element offset.
    array::Array independent = matrix.clone();

Device and dtype conversion
---------------------------

``toDevice()`` and ``toDtype()`` return the same array if no conversion is needed
(optionally forcing a copy with the second argument), or a new array otherwise:

.. code-block:: cpp

    array::Array cpu_array(std::vector<float>{1.0f, 2.0f});

    // Transfer to GPU — allocates a new CUDA buffer.
    array::Array gpu_array = cpu_array.toDevice(array::Device::Cuda());

    // Cast to double — allocates a new buffer.
    array::Array f64 = cpu_array.toDtype(array::DType::Float64());

    // Always return an independent copy, even when no conversion is needed.
    array::Array copy = cpu_array.toDevice(array::Device::Cpu(), /*copy=*/true);

Broadcasting
------------

``broadcastTo()`` replicates elements along broadcast axes following NumPy rules.
It returns the same array if no broadcasting is needed
(optionally forcing a copy with the second argument), or a new array otherwise:

.. code-block:: cpp

    array::Array row(std::vector<float>{1.0f, 2.0f, 3.0f});   // shape (3,)
    array::Array tiled = row.broadcastTo(array::Shape({4, 3})); // shape (4, 3)

Reading and writing elements
----------------------------

Use ``get<T>()`` to extract the array contents as a C++ scalar or nested vector.
The template argument selects the output rank:

.. code-block:: cpp

    array::Array a(std::vector<std::vector<float>>{{1.0f, 2.0f}, {3.0f, 4.0f}});

    // 2-D output.
    auto matrix = a.get<std::vector<std::vector<float>>>();

    // Scalar — array must contain exactly one element.
    array::Array scalar(42.0f);
    float value = scalar.get<float>();

    // item() is stricter: requires exactly one element regardless of shape.
    float value = scalar.item<float>();

Use ``set()`` to overwrite the array's contents. The argument is broadcast and cast
to match the array's shape and dtype:

.. code-block:: cpp

    array::Array a(std::vector<float>{1.0f, 2.0f, 3.0f});

    // From a scalar or vector.
    a.set(std::vector<float>{4.0f, 5.0f, 6.0f});

    // From another array — broadcast and cast automatically.
    array::Array b(7.0f);
    a.set(b);

String representation
---------------------

``toString()`` returns a human-readable summary:

.. code-block:: cpp

    array::Array a(std::vector<float>{1.0f, 2.0f});
    a.toString();
    // Array([1, 2], shape=(2,), dtype='float32', device='cpu')

.. _isaacsim-common-array-api-cpp-dtype:

DType
=====

``array::DType`` identifies the scalar element type of an array. Supported element
types are ``bool``, the fixed-width integer types from ``<cstdint>``, ``float``, and ``double``.

Construction
------------

Construct an ``array::DType`` from a ``Kind`` enumerator, a string name, or a named
factory:

.. code-block:: cpp

    array::DType a = array::DType::Float32();
    array::DType b = array::DType(array::DType::Kind::eFloat32);
    array::DType c = array::DType("float32");       // equivalent to both above
    array::DType d = array::DType::fromString("float32");

    // Compile-time mapping from a C++ type.
    array::DType e = array::DType::fromType<float>();  // array::DType::Float32()

Named factories
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 15

   * - Factory
     - Kind
     - Size (bytes)
   * - ``DType::Bool()``
     - ``eBool``
     - 1
   * - ``DType::Int8()``
     - ``eInt8``
     - 1
   * - ``DType::Int16()``
     - ``eInt16``
     - 2
   * - ``DType::Int32()``
     - ``eInt32``
     - 4
   * - ``DType::Int64()``
     - ``eInt64``
     - 8
   * - ``DType::UInt8()``
     - ``eUInt8``
     - 1
   * - ``DType::UInt16()``
     - ``eUInt16``
     - 2
   * - ``DType::UInt32()``
     - ``eUInt32``
     - 4
   * - ``DType::UInt64()``
     - ``eUInt64``
     - 8
   * - ``DType::Float32()``
     - ``eFloat32``
     - 4
   * - ``DType::Float64()``
     - ``eFloat64``
     - 8

Queries
-------

.. code-block:: cpp

    array::DType t = array::DType::Float32();

    t.kind();        // array::DType::Kind::eFloat32
    t.size();        // 4
    t.isFloating();  // true
    t.isIntegral();  // false
    t.isSigned();    // true
    t.isUnsigned();  // false
    t.toString();    // "float32"

    array::DType::Float32() == array::DType::Float32();  // true
    array::DType::Float32() != array::DType::Int32();    // true

.. _isaacsim-common-array-api-cpp-shape:

Shape
=====

``array::Shape`` holds an ordered sequence of dimension sizes as signed 64-bit
integers. An empty ``Shape`` is zero-dimensional (scalar).

Construction
------------

.. code-block:: cpp

    array::Shape s1({3, 4, 5});                           // 3-D shape from initializer list
    array::Shape s2(std::vector<int32_t>{3, 4, 5});       // from a SupportedShapeSpec vector
    array::Shape s3(12);                                   // 1-D shape with dimension 12

Queries
-------

.. code-block:: cpp

    array::Shape s({3, 4, 5});

    s.ndim();        // 3
    s.size();        // 60
    s.shape();       // std::vector<int64_t>{3, 4, 5}
    s[0];            // 3
    s[-1];           // 5  (negative indices count from the end)
    s.toString();    // "(3, 4, 5)"

Broadcasting and reshape
------------------------

``canBroadcastTo()`` checks NumPy broadcasting compatibility. ``resolve()`` expands
a -1 placeholder in a target shape to the correct dimension size:

.. code-block:: cpp

    array::Shape src({1, 4});
    src.canBroadcastTo(array::Shape({3, 4}));  // true
    src.canBroadcastTo(array::Shape({3, 5}));  // false

    array::Shape base({2, 6});
    base.resolve(array::Shape({3, -1}));  // array::Shape({3, 4})

.. _isaacsim-common-array-api-cpp-device:

Device
======

``array::Device`` identifies a compute device: the CPU (ordinal -1) or a CUDA GPU
(non-negative ordinal).

Construction
------------

.. code-block:: cpp

    array::Device cpu     = array::Device::Cpu();
    array::Device gpu0    = array::Device::Cuda();      // ordinal 0
    array::Device gpu1    = array::Device::Cuda(1);     // ordinal 1
    array::Device from_str = array::Device::fromString("cuda:0");

    // From a SupportedDeviceSpec.
    array::Device d1 = array::Device(int32_t(0));       // CUDA device 0
    array::Device d2 = array::Device(std::string("cpu"));

Queries
-------

.. code-block:: cpp

    array::Device d = array::Device::Cuda(1);

    d.ordinal();      // 1
    d.isCpu();        // false
    d.isCuda();       // true
    d.isAvailable();  // true if CUDA device 1 is present
    d.toString();     // "cuda:1"

    array::Device::Cpu() == array::Device::Cpu();   // true
    array::Device::Cpu() != array::Device::Cuda();  // true

.. _isaacsim-common-array-api-cpp-device-guard:

DeviceGuard
===========

``array::DeviceGuard`` is an RAII guard that sets the active CUDA device on
construction and restores the previous device on destruction. It has no effect when
targeting the CPU.

``DeviceGuard`` is non-copyable and non-movable.

.. code-block:: cpp

    {
        array::DeviceGuard guard(array::Device::Cuda(1));
        // CUDA calls here run on device 1.
    }
    // Previous device is restored.

    // Construct directly from an ordinal.
    {
        array::DeviceGuard guard(1);
        // ...
    }
