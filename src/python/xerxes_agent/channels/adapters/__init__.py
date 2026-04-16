# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License")
"""Channel adapter implementations for every platform Hermes supports.

Each adapter is webhook-shaped and accepts an injected ``http_client``
callable for testability — production deployments use the default
``httpx`` path, while tests pass a fake.
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
