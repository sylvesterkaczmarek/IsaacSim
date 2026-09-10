# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Extension for Omniverse client authentication in Isaac Sim storage."""

import os

import carb.settings
import omni.client
import omni.ext

ASSET_ROOT_ENV_VAR = "ISAACSIM_ASSET_ROOT"
ASSET_REGION_PROFILE_ENV_VAR = "ISAACSIM_ASSET_REGION_PROFILE"

ASSET_ROOT_SETTING = "/persistent/isaac/asset_root/default"
ASSET_REGION_PROFILES_SETTING = "/exts/isaacsim.storage.native/asset_region_profiles"
USD_SEARCH_ENDPOINT_SETTING = "/exts/omni.simready.content.browser/usd_search_endpoint"


class Extension(omni.ext.IExt):
    """Isaac Sim storage native extension for Omniverse client authentication and asset root configuration.

    This extension handles three concerns:
    1. Applies the asset region profile named by the ISAACSIM_ASSET_REGION_PROFILE environment variable.
    2. Overrides the default asset root path from the ISAACSIM_ASSET_ROOT environment variable.
    3. Registers an authentication callback for the Omniverse client when ETM_ACTIVE is set.
    """

    def on_startup(self, ext_id: str) -> None:
        """Initialize the extension.

        Applies the selected asset region profile (if any), then the ISAACSIM_ASSET_ROOT env var
        override (if set), then registers an authentication callback if ETM_ACTIVE is set.

        Args:
            ext_id: The extension ID.
        """
        self._auth_cb = None
        self._profile_headers: list[str] = []

        self._apply_asset_region_profile(os.getenv(ASSET_REGION_PROFILE_ENV_VAR))

        asset_root = os.getenv(ASSET_ROOT_ENV_VAR)
        if asset_root:
            carb.settings.get_settings().set_string(ASSET_ROOT_SETTING, asset_root.rstrip("/"))
            carb.log_info(f"Overriding asset root from {ASSET_ROOT_ENV_VAR}: {asset_root}")

        if os.getenv("ETM_ACTIVE"):
            self._auth_cb = omni.client.register_authentication_callback(self._authenticate)

    def _apply_asset_region_profile(self, profile_name: str | None) -> None:
        """Apply the routing and asset root described by a named asset region profile.

        Profiles are declared under the ``asset_region_profiles`` setting and may define an S3
        endpoint, an optional CDN to serve reads, HTTP headers required for provider
        compatibility, a default asset root, and a USD Search endpoint.

        The S3 configuration is applied in memory, so the user's ``omniverse.toml`` is
        never modified.

        Args:
            profile_name: Name of the profile to apply. No-op when empty or unknown.
        """
        if not profile_name:
            return

        settings = carb.settings.get_settings()
        profile = settings.get(f"{ASSET_REGION_PROFILES_SETTING}/{profile_name}")
        if not profile:
            carb.log_warn(f"Ignoring {ASSET_REGION_PROFILE_ENV_VAR}: no asset region profile named '{profile_name}'")
            return

        endpoint = profile.get("endpoint")
        if endpoint:
            result = omni.client.set_s3_configuration(
                url=endpoint,
                bucket=profile.get("bucket"),
                region=profile.get("region"),
                cloudfrontUrl=profile.get("cdn_url") or None,
                cloudfrontForList=bool(profile.get("cdn_for_list", False)),
                writeConfig=False,
            )
            if result != omni.client.Result.OK:
                carb.log_error(f"Asset region profile '{profile_name}' failed to configure {endpoint}: {result}")
                return

        for key, value in (profile.get("http_headers") or {}).items():
            omni.client.set_http_header(key, value)
            self._profile_headers.append(key)

        asset_root = profile.get("asset_root")
        if asset_root:
            settings.set_string(ASSET_ROOT_SETTING, asset_root.rstrip("/"))

        usd_search_endpoint = profile.get("usd_search_endpoint")
        if usd_search_endpoint:
            settings.set_string(USD_SEARCH_ENDPOINT_SETTING, usd_search_endpoint)

        carb.log_info(f"Applied asset region profile '{profile_name}'")

    def _authenticate(self, prefix: str) -> tuple[str, str] | None:
        """Authentication callback for Omniverse client.

        Retrieves credentials from ISAACSIM_OMNI_USER and ISAACSIM_OMNI_PASS
        environment variables.

        Args:
            prefix: URL prefix for authentication.

        Returns:
            Tuple of (username, password) if credentials are available, None otherwise.
        """
        omniuser = os.getenv("ISAACSIM_OMNI_USER")
        omnipass = os.getenv("ISAACSIM_OMNI_PASS")

        if omniuser and omnipass:
            return (omniuser, omnipass)
        return None

    def on_shutdown(self) -> None:
        """Clean up resources when the extension is shut down."""
        for key in self._profile_headers:
            omni.client.set_http_header(key, None)
        self._profile_headers = []
        self._auth_cb = None
