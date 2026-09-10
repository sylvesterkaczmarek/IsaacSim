# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test backend behavior."""

import isaacsim.foundation.utils.backend as backend_utils
import pytest


def test_backend_lifecycle() -> None:
    # no backend active outside a guard
    """Test backend lifecycle."""
    assert not backend_utils.is_backend_set()
    # backend active inside a guard
    with backend_utils.use_backend("backend_1"):
        assert backend_utils.is_backend_set()
        assert backend_utils.get_current_backend(["backend_1", "backend_2"]) == "backend_1"
    # no backend active outside a guard
    assert not backend_utils.is_backend_set()
    # restored after exception
    try:
        with backend_utils.use_backend("backend_1"):
            assert backend_utils.get_current_backend(["backend_2"]) == "backend_2"  # fallback to 1st supported backend
            raise RuntimeError("forced exception")
    except RuntimeError:
        pass
    assert not backend_utils.is_backend_set()


def test_nested_backends() -> None:
    """Test nested backends."""
    assert not backend_utils.is_backend_set()
    with backend_utils.use_backend("backend_1"):
        assert backend_utils.is_backend_set()
        assert backend_utils.get_current_backend(["backend_1", "backend_2"]) == "backend_1"
        with backend_utils.use_backend("backend_2"):
            assert backend_utils.is_backend_set()
            assert backend_utils.get_current_backend(["backend_1", "backend_2"]) == "backend_2"
        assert backend_utils.is_backend_set()
        assert backend_utils.get_current_backend(["backend_1", "backend_2"]) == "backend_1"
    assert not backend_utils.is_backend_set()


def test_backend_raise_on_unsupported() -> None:
    # via the context flag
    """Test backend raise on unsupported."""
    assert not backend_utils.should_raise_on_unsupported()
    with backend_utils.use_backend("unknown", raise_on_unsupported=True):
        assert backend_utils.should_raise_on_unsupported()
        with pytest.raises(RuntimeError):
            backend_utils.get_current_backend(["backend"])
    # via the call-site parameter
    assert not backend_utils.should_raise_on_unsupported()
    with backend_utils.use_backend("unknown"):
        assert not backend_utils.should_raise_on_unsupported()
        with pytest.raises(RuntimeError):
            backend_utils.get_current_backend(["backend"], raise_on_unsupported=True)
    # explicit False at the call site overrides the context flag
    assert not backend_utils.should_raise_on_unsupported()
    with backend_utils.use_backend("unknown", raise_on_unsupported=True):
        assert backend_utils.should_raise_on_unsupported()
        assert backend_utils.get_current_backend(["backend"], raise_on_unsupported=False) == "backend"


def test_backend_raise_on_fallback() -> None:
    """Test backend raise on fallback."""
    assert not backend_utils.should_raise_on_fallback()
    with backend_utils.use_backend("backend", raise_on_fallback=True):
        assert backend_utils.should_raise_on_fallback()
