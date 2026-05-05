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
"""Concrete channel adapters for external messaging platforms.

Each submodule implements a ``WebhookChannel`` subclass for a specific
service (e.g. Slack, Telegram, Discord, etc.).
"""

from .bluebubbles import BlueBubblesChannel
from .dingtalk import DingTalkChannel
from .discord import DiscordChannel
from .email_imap import EmailChannel
from .feishu import FeishuChannel
from .home_assistant import HomeAssistantChannel
from .matrix import MatrixChannel
from .mattermost import MattermostChannel
from .signal import SignalChannel
from .slack import SlackChannel
from .sms import TwilioSMSChannel
from .telegram import TelegramChannel
from .wecom import WeComChannel
from .whatsapp import WhatsAppChannel

__all__ = [
    "BlueBubblesChannel",
    "DingTalkChannel",
    "DiscordChannel",
    "EmailChannel",
    "FeishuChannel",
    "HomeAssistantChannel",
    "MatrixChannel",
    "MattermostChannel",
    "SignalChannel",
    "SlackChannel",
    "TelegramChannel",
    "TwilioSMSChannel",
    "WeComChannel",
    "WhatsAppChannel",
]
