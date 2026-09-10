# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include_guard(GLOBAL)

# Keep developer-tool caches and Python bytecode out of staged and installed
# packages. Expand this list unquoted in file(COPY) and install(DIRECTORY).
set(ISAACSIM_PACKAGING_EXCLUDE_ARGS
    PATTERN "__pycache__" EXCLUDE
    PATTERN ".pytest_cache" EXCLUDE
    PATTERN ".mypy_cache" EXCLUDE
    PATTERN ".ruff_cache" EXCLUDE
    PATTERN "*.pyc" EXCLUDE
    PATTERN "*.pyo" EXCLUDE
)

# Apply the same policy to file inventories used as custom-command dependencies.
set(ISAACSIM_PACKAGING_EXCLUDE_REGEX
    "/(__pycache__|\\.pytest_cache|\\.mypy_cache|\\.ruff_cache)/|\\.(pyc|pyo)$"
)
