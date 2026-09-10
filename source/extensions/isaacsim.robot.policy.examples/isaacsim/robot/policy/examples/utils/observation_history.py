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

"""Maintain independent circular histories for flat observation terms."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = ["ObservationHistory"]


class ObservationHistory:
    """Circular history buffers for the terms in one flat observation sample.

    The first sample backfills every frame of each term. Later samples replace the oldest
    frame, and :meth:`append` returns all terms flattened from oldest to newest.

    Args:
        term_shapes: Ordered ``(history_length, sample_width)`` pairs for observation terms.

    Raises:
        ValueError: If a history length or sample width is less than one.
    """

    def __init__(self, term_shapes: Sequence[tuple[int, int]]) -> None:
        shapes = tuple(term_shapes)
        if any(history_length < 1 or sample_width < 1 for history_length, sample_width in shapes):
            raise ValueError(f"Observation history shapes must be positive, got {shapes}.")

        self._sample_width = sum(sample_width for _, sample_width in shapes)
        self._sample_widths = tuple(sample_width for _, sample_width in shapes)
        self._buffers = tuple(
            np.empty((history_length, sample_width), dtype=np.float32) for history_length, sample_width in shapes
        )
        self._next_indices = [0] * len(shapes)
        self._initialized = [False] * len(shapes)

    def reset(self) -> None:
        """Discard all buffered samples."""
        self._next_indices[:] = [0] * len(self._next_indices)
        self._initialized[:] = [False] * len(self._initialized)

    def append(self, sample: np.ndarray) -> np.ndarray:
        """Append one flat sample and return its history-expanded representation.

        Args:
            sample: Current-frame values in term order.

        Returns:
            Flat histories in term order, with each term ordered from oldest to newest.

        Raises:
            ValueError: If the sample is not a flat vector of the configured width.
        """
        sample = np.asarray(sample)
        expected_shape = (self._sample_width,)
        if sample.shape != expected_shape:
            raise ValueError(f"Observation sample has shape {sample.shape}; expected {expected_shape}.")

        histories = []
        start = 0
        for index, (sample_width, buffer) in enumerate(zip(self._sample_widths, self._buffers)):
            stop = start + sample_width
            current = sample[start:stop]
            start = stop

            if not self._initialized[index]:
                buffer[:] = current
                self._initialized[index] = True
            else:
                buffer[self._next_indices[index]] = current
                self._next_indices[index] = (self._next_indices[index] + 1) % len(buffer)

            next_index = self._next_indices[index]
            histories.append(np.concatenate((buffer[next_index:], buffer[:next_index])).reshape(-1))

        return np.concatenate(histories)
