# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OAuth client presets and on-disk credential storage.

Two layers: ``oauth`` wraps the underlying MCP OAuth helpers with a
client object and provider presets (OpenAI, Anthropic, GitHub PAT,
GitHub Copilot); ``storage`` persists issued tokens under
``$XERXES_HOME/credentials`` with ``0600`` permissions.
"""

from .oauth import OAuthClient, OAuthConfig, OAuthToken
from .storage import CredentialStorage, list_providers, load, remove, save

__all__ = [
    "CredentialStorage",
    "OAuthClient",
    "OAuthConfig",
    "OAuthToken",
    "list_providers",
    "load",
    "remove",
    "save",
]
