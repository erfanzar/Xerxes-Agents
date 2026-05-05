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
"""Public exports for the Xerxes channels package.

This module exposes the core channel abstractions, registry, identity
resolution, OAuth helpers, and the adapters sub-package.
"""

from . import adapters
from ._helpers import WebhookChannel
from .base import Channel
from .identity import IdentityRecord, IdentityResolver
from .oauth import OAuthClient, OAuthProvider, OAuthToken
from .registry import ChannelRegistry, InboundHandler, gather_inbound
from .types import ChannelMessage, MessageDirection
from .webhooks import WebhookDispatcher, WebhookHandler, WebhookResponse

__all__ = [
    "Channel",
    "ChannelMessage",
    "ChannelRegistry",
    "IdentityRecord",
    "IdentityResolver",
    "InboundHandler",
    "MessageDirection",
    "OAuthClient",
    "OAuthProvider",
    "OAuthToken",
    "WebhookChannel",
    "WebhookDispatcher",
    "WebhookHandler",
    "WebhookResponse",
    "adapters",
    "gather_inbound",
]
