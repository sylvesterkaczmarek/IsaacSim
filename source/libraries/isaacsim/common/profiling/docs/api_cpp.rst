..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

.. _isaacsim-common-profiling-api-cpp:

C++ guide
---------

.. isaacsim-libraries-api-guide-start

Include ``isaacsim/common/profiling/Profiling.hpp`` and link ``isaacsim::common-profiling``. Use ``Zone`` directly or
the convenience macros in ``ProfileMacros.hpp`` to emit events only while the selected profiling mask is enabled.
