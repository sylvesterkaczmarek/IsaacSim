..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: Apache-2.0

.. _isaacsim-common-profiling-api-c:

C guide
-------

.. isaacsim-libraries-api-guide-start

Include ``isaacsim/common/profiling/Profiling.h`` and link ``isaacsim::common-profiling``. Use
``ProfilingHost.h`` only when the application owns the Carbonite profiler that backs the profiling facade.
