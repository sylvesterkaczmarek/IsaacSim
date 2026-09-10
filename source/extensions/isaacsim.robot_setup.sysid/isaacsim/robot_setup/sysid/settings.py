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

"""Kit settings adapter for otherwise portable optimizer configuration."""

from __future__ import annotations

import json

from .errors import SysIdRuntimeUnavailableError
from .optimizer_config import OptimizerBackendConfig

SETTINGS_BACKEND_KEY = "/persistent/isaacsim/robot_setup/sysid/optimizer_backend"
SETTINGS_CONFIG_KEY = "/persistent/isaacsim/robot_setup/sysid/optimizer_backend_config"


class CarbOptimizerSettingsStore:
    """Persist optimizer configuration through the Carbonite settings service."""

    def __init__(self) -> None:
        try:
            import carb.settings
        except ModuleNotFoundError as exc:
            if str(exc.name or "") not in {"carb", "carb.settings"}:
                raise
            raise SysIdRuntimeUnavailableError("Carb settings require a bootstrapped Kit runtime.") from exc
        try:
            self._settings = carb.settings.get_settings()
        except (AttributeError, RuntimeError) as exc:
            raise SysIdRuntimeUnavailableError("Carb settings require a bootstrapped Kit runtime.") from exc

    def load(self) -> OptimizerBackendConfig:
        """Load optimizer settings from Carbonite.

        Returns:
            Persisted optimizer configuration, or defaults when no values exist.
        """
        payload: dict = {}
        backend_raw = self._settings.get(SETTINGS_BACKEND_KEY)
        config_raw = self._settings.get(SETTINGS_CONFIG_KEY)
        if backend_raw:
            payload["backend"] = backend_raw
        if isinstance(config_raw, dict):
            payload.update(config_raw)
        elif isinstance(config_raw, str) and config_raw:
            payload.update(json.loads(config_raw))
        return OptimizerBackendConfig.from_dict(payload) if payload else OptimizerBackendConfig()

    def save(self, config: OptimizerBackendConfig) -> None:
        """Save optimizer settings to Carbonite.

        Args:
            config: Optimizer configuration to persist.
        """
        self._settings.set(SETTINGS_BACKEND_KEY, config.backend.value)
        payload = config.to_dict()
        payload.pop("backend", None)
        self._settings.set(SETTINGS_CONFIG_KEY, payload)
