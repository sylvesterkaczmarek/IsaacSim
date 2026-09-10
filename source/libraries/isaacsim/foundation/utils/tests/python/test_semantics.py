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

"""Test semantics behavior."""

from typing import Any

import isaacsim.foundation.utils.semantics as semantics_utils

from .fixtures import stage  # noqa: F401 - imported so pytest can discover the fixture


def test_add_labels(capsys: Any, stage: Any) -> None:
    """Test add labels.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
    """
    stage.define_prim("/World/A", "Cube")
    assert semantics_utils.get_labels("/World/A") == {}
    # add labels to default and custom taxonomy
    semantics_utils.add_labels("/World/A", labels=["label_0", "label_1"])
    semantics_utils.add_labels("/World/A", labels=["label_1", "label_2", "label_3"], taxonomy="test")
    assert semantics_utils.get_labels("/World/A") == {
        "class": ["label_0", "label_1"],
        "test": ["label_1", "label_2", "label_3"],
    }
    # add a new label to existing ones
    semantics_utils.add_labels("/World/A", labels="label_4")
    semantics_utils.add_labels("/World/A", labels="label_4")  # add same label again (no-op)
    assert semantics_utils.get_labels("/World/A") == {
        "class": ["label_0", "label_1", "label_4"],
        "test": ["label_1", "label_2", "label_3"],
    }


def test_get_labels(capsys: Any, stage: Any) -> None:
    """Test get labels.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
    """
    stage.define_prim("/World/A", "Cube")
    stage.define_prim("/World/B", "Xform")
    assert semantics_utils.get_labels("/World/A") == {}
    assert semantics_utils.get_labels("/World/B") == {}
    # add labels
    semantics_utils.add_labels("/World/A", labels=["label_0", "label_1"])
    semantics_utils.add_labels("/World/B", labels=["label_1", "label_2", "label_3"], taxonomy="test")
    # get labels from specific prims
    assert semantics_utils.get_labels("/World/A") == {"class": ["label_0", "label_1"]}
    assert semantics_utils.get_labels("/World/B") == {"test": ["label_1", "label_2", "label_3"]}
    # prim without semantics applied
    assert semantics_utils.get_labels("/World") == {}
    # prim without semantics applied but with descendants
    assert semantics_utils.get_labels("/World", include_descendants=True) == {
        "class": ["label_0", "label_1"],
        "test": ["label_1", "label_2", "label_3"],
    }


def test_remove_labels(capsys: Any, stage: Any) -> None:
    """Test remove labels.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
    """
    stage.define_prim("/World/A", "Cube")
    assert semantics_utils.get_labels("/World/A") == {}
    # add labels
    semantics_utils.add_labels("/World/A", labels=["label_0", "label_1", "label_4"])
    semantics_utils.add_labels("/World/A", labels=["label_1", "label_2", "label_3"], taxonomy="test")
    assert semantics_utils.get_labels("/World/A") == {
        "class": ["label_0", "label_1", "label_4"],
        "test": ["label_1", "label_2", "label_3"],
    }
    # remove label from a specific taxonomy
    semantics_utils.remove_labels("/World/A", labels="label_2", taxonomy="test")
    assert semantics_utils.get_labels("/World/A") == {
        "class": ["label_0", "label_1", "label_4"],
        "test": ["label_1", "label_3"],
    }
    # remove label from all taxonomies
    semantics_utils.remove_labels("/World/A", labels="label_1")
    assert semantics_utils.get_labels("/World/A") == {"class": ["label_0", "label_4"], "test": ["label_3"]}
    # remove label from descendants
    semantics_utils.remove_labels("/World", labels="label_4", include_descendants=True)
    assert semantics_utils.get_labels("/World/A") == {"class": ["label_0"], "test": ["label_3"]}
    # remove a label that does not exist is a no-op
    semantics_utils.remove_labels("/World/A", labels="label_5")
    assert semantics_utils.get_labels("/World/A") == {"class": ["label_0"], "test": ["label_3"]}


def test_remove_all_labels(capsys: Any, stage: Any) -> None:
    """Test remove all labels.

    Args:
        capsys: Pytest output-capture fixture.
        stage: Stage used by the test.
    """
    stage.define_prim("/World/A", "Cube")
    assert semantics_utils.get_labels("/World/A") == {}
    # add labels
    semantics_utils.add_labels("/World/A", labels=["label_0", "label_1", "label_4"])
    semantics_utils.add_labels("/World/A", labels=["label_1", "label_2", "label_3"], taxonomy="test")
    assert semantics_utils.get_labels("/World/A") == {
        "class": ["label_0", "label_1", "label_4"],
        "test": ["label_1", "label_2", "label_3"],
    }
    # remove all labels but keep taxonomies
    semantics_utils.remove_all_labels("/World/A")
    assert semantics_utils.get_labels("/World/A") == {"class": [], "test": []}
    # remove taxonomies on the parent only (child /World/A is unaffected)
    semantics_utils.remove_all_labels("/World", remove_taxonomies=True)
    assert semantics_utils.get_labels("/World/A") == {"class": [], "test": []}
    # remove taxonomies on parent and all descendants
    semantics_utils.remove_all_labels("/World", remove_taxonomies=True, include_descendants=True)
    assert semantics_utils.get_labels("/World/A") == {}
