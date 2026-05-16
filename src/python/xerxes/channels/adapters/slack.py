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
"""Slack channel adapter.

Speaks Slack's Events API for inbound traffic and ``chat.postMessage``
for outbound. Accepts either a static bot token or an ``OAuthClient`` so
multi-workspace installs can keep per-team credentials. The signing
secret, when configured, gates every inbound payload via
``_verify_slack_signature`` — anything without a valid HMAC is dropped
silently rather than allowed to drive the agent.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import time
import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection

logger = logging.getLogger(__name__)


def _verify_slack_signature(
    signing_secret: str,
    headers: dict[str, str],
    body: bytes,
    *,
    max_skew_seconds: int = 5 * 60,
    now: tp.Callable[[], float] = time.time,
) -> bool:
    """Verify Slack's HMAC-SHA256 webhook signature in constant time.

    Slack signs requests with two headers:

    * ``X-Slack-Request-Timestamp`` — Unix timestamp.
    * ``X-Slack-Signature`` — ``"v0=<hex hmac>"`` where the HMAC is computed
      over ``"v0:<ts>:<raw body>"`` with the workspace signing secret.

    Timestamps further than ``max_skew_seconds`` from now are rejected to
    defeat replay attacks. An empty ``signing_secret`` returns ``False``
    (fail-closed) so misconfigured deployments cannot be tricked into
    treating any payload as valid.

    Args:
        signing_secret: The workspace's Slack signing secret.
        headers: HTTP headers from the webhook request; matched
            case-insensitively.
        body: Raw request body bytes (signed verbatim).
        max_skew_seconds: Allowed clock skew. Defaults to 5 minutes per
            Slack's recommendation.
        now: Injectable clock for testing.

    Returns:
        ``True`` when the signature is valid and the timestamp is fresh.
    """
    if not signing_secret:
        return False
    # Headers from FastAPI / Starlette are case-sensitive but channel callers
    # pass them as lowercase. Accept either capitalisation.
    lowered = {k.lower(): v for k, v in headers.items()}
    ts = lowered.get("x-slack-request-timestamp", "")
    sig = lowered.get("x-slack-signature", "")
    if not ts or not sig or not ts.isdigit():
        return False
    if abs(now() - int(ts)) > max_skew_seconds:
        return False
    basestring = b"v0:" + ts.encode("ascii") + b":" + body
    digest = hmac.new(signing_secret.encode("utf-8"), basestring, hashlib.sha256).hexdigest()
    expected = f"v0={digest}"
    return hmac.compare_digest(expected.encode("ascii"), sig.encode("ascii"))


class SlackChannel(WebhookChannel):
    """Slack adapter speaking Events API in and ``chat.postMessage`` out."""

    name = "slack"

    def __init__(
        self,
        bot_token: str = "",
        *,
        signing_secret: str = "",
        oauth_client: tp.Any = None,
        install_id: str = "default",
        http_client: tp.Any = None,
    ) -> None:
        """Build the channel.

        Args:
            bot_token: Static Slack bot token; takes precedence over the
                OAuth lookup when set.
            signing_secret: Workspace signing secret used to verify inbound
                webhooks. Empty means the channel refuses every inbound
                payload — operators must set this to receive real events.
            oauth_client: Optional ``OAuthClient``-shaped object exposing
                ``get_valid_token(install_id)``. Used when ``bot_token`` is
                empty, e.g. for multi-workspace installs.
            install_id: Default installation id used for OAuth lookups when
                the inbound payload does not carry a verified ``team_id``.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.bot_token = bot_token
        self.signing_secret = signing_secret
        self.oauth_client = oauth_client
        self.install_id = install_id
        self._http = http_client

    def _resolve_token(self, install_id: str | None = None) -> str:
        """Pick the bot token to authenticate one outbound Slack request.

        Outbound calls triggered by an inbound message pass the
        ``verified_install_id`` (the verified ``team_id``) so replies use
        the right workspace's token; otherwise the constructor default
        applies.

        Args:
            install_id: Per-call override; falls back to ``self.install_id``.

        Returns:
            The static ``bot_token`` if set, otherwise the access token
            obtained from the OAuth client.

        Raises:
            RuntimeError: No token can be resolved from either source.
        """
        if self.bot_token:
            return self.bot_token
        if self.oauth_client is not None:
            tok = self.oauth_client.get_valid_token(install_id or self.install_id)
            if tok and tok.access_token:
                return tok.access_token
        raise RuntimeError("Slack bot token unavailable")

    def _parse_inbound(self, headers, body):
        """Translate a Slack Events API payload into ``ChannelMessage`` list.

        Drops URL-verification handshakes, bot-loopback events, and message
        subtypes that should not drive the agent. Signature verification
        runs first — when ``signing_secret`` is set an invalid signature
        returns an empty list and logs the rejection. ``team_id`` is only
        trusted (and stamped into ``metadata['verified_install_id']``)
        after the signature passes.

        Args:
            headers: HTTP headers; ``X-Slack-Signature`` and
                ``X-Slack-Request-Timestamp`` must be valid when a signing
                secret is configured.
            body: Raw JSON webhook body.

        Returns:
            Parsed inbound messages, or an empty list for handshakes,
            loopback events, and signature mismatches.
        """
        # Refuse unsigned events when a signing secret is configured. Without
        # this check anyone who learns the events URL spoofs Slack payloads
        # and drives the agent.
        if self.signing_secret and not _verify_slack_signature(self.signing_secret, headers, body):
            logger.warning("rejected Slack webhook with invalid signature")
            return []
        data = parse_json_body(body)
        if data.get("type") == "url_verification":
            return []
        ev = data.get("event") or {}
        if ev.get("type") not in ("message", "app_mention"):
            return []
        # ``bot_id`` skips first-party bot loopback; ``subtype`` covers other
        # automated events (message_changed, bot_message, etc.) that we never
        # want to treat as user input.
        if ev.get("bot_id") or ev.get("subtype"):
            return []
        # team_id comes from the verified outer payload — only trust it once
        # signature verification has passed. Used downstream as the OAuth
        # install_id so outbound replies use the right workspace's token.
        team_id = str(data.get("team_id", "")) if (self.signing_secret or self.bot_token) else ""
        return [
            ChannelMessage(
                text=ev.get("text", ""),
                channel=self.name,
                channel_user_id=str(ev.get("user", "")),
                room_id=str(ev.get("channel", "")),
                platform_message_id=str(ev.get("ts", "")),
                direction=MessageDirection.INBOUND,
                metadata={
                    "team_id": team_id,
                    "thread_ts": ev.get("thread_ts", ""),
                    "verified_install_id": team_id,
                },
            )
        ]

    async def _send_outbound(self, message):
        """Deliver one message via ``chat.postMessage``.

        ``room_id`` selects the target Slack channel, ``text`` carries the
        body, and ``reply_to`` (when set) becomes ``thread_ts`` so replies
        land in the right thread. ``metadata['verified_install_id']`` —
        populated by ``_parse_inbound`` only after signature verification
        — selects which workspace's OAuth token is used, so a forged
        ``install_id`` cannot drive another workspace's token.

        Args:
            message: Outbound message.
        """
        body = {"channel": message.room_id, "text": message.text}
        if message.reply_to:
            body["thread_ts"] = message.reply_to
        install_id = (message.metadata or {}).get("verified_install_id") or None
        http_post(
            "https://slack.com/api/chat.postMessage",
            json_body=body,
            headers={"Authorization": f"Bearer {self._resolve_token(install_id)}"},
            http_client=self._http,
        )
