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
"""Discord channel adapter."""

from __future__ import annotations

import asyncio
import logging
import re
import typing as tp

from .._helpers import WebhookChannel, http_post, parse_json_body
from ..types import ChannelMessage, MessageDirection

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
DISCORD_MESSAGE_LIMIT = 2000
_FALSE_VALUES = {"0", "false", "no", "off", ""}


class DiscordChannel(WebhookChannel):
    """Discord bot adapter.

    ``transport="gateway"`` uses ``discord.py`` for live bot ingestion. The
    webhook parser remains available for tests and deployments that front
    Discord events through their own verifier.
    """

    name = "discord"

    def __init__(
        self,
        bot_token: str = "",
        *,
        token: str = "",
        transport: str = "webhook",
        require_mention: bool | str = False,
        always_reply_in_channels: bool | str = False,
        allowed_channel_ids: str | list[str] | tuple[str, ...] | set[str] | None = None,
        allowed_channel_names: str | list[str] | tuple[str, ...] | set[str] | None = None,
        allowed_guild_ids: str | list[str] | tuple[str, ...] | set[str] | None = None,
        address_names: str | list[str] | tuple[str, ...] | set[str] | None = None,
        instance_name: str = "",
        device_name: str = "",
        register_commands: bool | str = True,
        ignore_bots: bool | str = True,
        message_content_intent: bool | str = True,
        suppress_mentions: bool | str = True,
        max_message_chars: int | str = DISCORD_MESSAGE_LIMIT,
        bot_user_id: str = "",
        http_client: tp.Any = None,
    ) -> None:
        """Build the channel.

        Args:
            bot_token: Discord bot authentication token. ``token`` is accepted
                as a config alias.
            token: Alias for ``bot_token``.
            transport: ``"gateway"`` for live Discord Gateway ingestion or
                ``"webhook"`` for externally delivered payloads.
            require_mention: In guild channels, require a bot mention unless
                a channel is explicitly allowed or always-reply is enabled.
            always_reply_in_channels: Let the bot respond to every guild
                channel message that passes allowlists.
            allowed_channel_ids: Optional comma-separated/list allowlist.
            allowed_channel_names: Optional comma-separated/list allowlist for
                Discord channel or thread names.
            allowed_guild_ids: Optional comma-separated/list allowlist.
            address_names: Optional comma-separated/list names. When set, the
                adapter only accepts messages starting with one of these names
                (for example ``m2-max: status``).
            instance_name: Label prepended to outbound replies so a shared
                Discord room shows which Xerxes instance answered.
            device_name: Alias for ``instance_name``.
            register_commands: Register Discord application commands for
                ``/ask``, ``/skills``, ``/skill``, and ``/status``.
            ignore_bots: Ignore bot-authored messages.
            message_content_intent: Request Discord's message content intent
                when starting the Gateway client.
            suppress_mentions: Disable mention parsing on outbound replies.
            max_message_chars: Outbound content chunk size.
            bot_user_id: Known bot user id. Learned automatically after
                Gateway READY when omitted.
            http_client: Optional HTTP client override forwarded to
                ``http_post``.
        """
        super().__init__()
        self.bot_token = str(bot_token or token or "")
        if not self.bot_token:
            raise ValueError("Discord bot token is required; set bot_token, token, or token_env")
        self.transport = str(transport or "webhook").lower()
        if self.transport not in {"webhook", "gateway"}:
            raise ValueError("Discord transport must be 'webhook' or 'gateway'")
        self.require_mention = _as_bool(require_mention)
        self.always_reply_in_channels = _as_bool(always_reply_in_channels)
        self.allowed_channel_ids = _as_id_set(allowed_channel_ids)
        self.allowed_channel_names = _as_name_set(allowed_channel_names)
        self.allowed_guild_ids = _as_id_set(allowed_guild_ids)
        self.address_names = _as_name_set(address_names)
        self.instance_name = str(instance_name or device_name or "").strip()
        self.register_commands = _as_bool(register_commands)
        self.ignore_bots = _as_bool(ignore_bots)
        self.message_content_intent = _as_bool(message_content_intent)
        self.suppress_mentions = _as_bool(suppress_mentions)
        self.max_message_chars = max(1, min(int(max_message_chars or DISCORD_MESSAGE_LIMIT), DISCORD_MESSAGE_LIMIT))
        self._bot_user_id = str(bot_user_id or "")
        self._http = http_client
        self._client: tp.Any | None = None
        self._discord: tp.Any | None = None
        self._gateway_task: asyncio.Task[tp.Any] | None = None
        self.gateway_error = ""

    async def start(self, on_inbound) -> None:
        """Register the inbound handler and start the Gateway client when configured."""
        await super().start(on_inbound)
        if self.transport != "gateway":
            return
        if self._gateway_task is not None and not self._gateway_task.done():
            return
        self._client = self._build_discord_client()
        self.gateway_error = ""
        self._gateway_task = asyncio.create_task(self._client.start(self.bot_token))
        self._gateway_task.add_done_callback(self._record_gateway_result)
        await asyncio.sleep(0)

    async def stop(self) -> None:
        """Stop the Gateway client and drop the inbound handler."""
        client = self._client
        if client is not None:
            await client.close()
        if self._gateway_task is not None and not self._gateway_task.done():
            self._gateway_task.cancel()
            await asyncio.gather(self._gateway_task, return_exceptions=True)
        self._gateway_task = None
        self._client = None
        await super().stop()

    def _parse_inbound(self, headers, body):
        """Translate a Discord webhook payload into ``ChannelMessage`` instances.

        Tolerates two shapes — payloads with a nested ``message`` object
        and payloads where the message fields sit at the top level.

        Args:
            headers: HTTP headers (unused).
            body: Raw JSON webhook body.

        Returns:
            One parsed inbound message, or an empty list if the payload
            decoded to an empty dict.
        """
        data = parse_json_body(body)
        if not data:
            return []
        msg = _extract_message_payload(data)
        channel_message = self._message_from_mapping(msg)
        return [channel_message] if channel_message is not None else []

    async def _send_outbound(self, message):
        """Send one message via ``POST /channels/{channel_id}/messages``.

        Args:
            message: Outbound message. ``room_id`` is the channel id,
                ``text`` the content, and ``reply_to`` (when set) becomes a
                ``message_reference`` so the reply quotes the original.
        """
        if not message.room_id:
            raise ValueError("Discord outbound messages require room_id=channel_id")
        text = _label_outbound(message.text or "(no response)", self.instance_name)
        for index, chunk in enumerate(_chunk_text(text, self.max_message_chars)):
            sent = await self._send_via_discord_py(message, chunk, include_reference=index == 0)
            if sent:
                continue
            await self._send_via_rest(message, chunk, include_reference=index == 0)

    async def send_typing(self, room_id: str | None) -> None:
        """Send Discord's typing indicator for the target channel when possible."""
        if not room_id:
            return
        await asyncio.to_thread(
            http_post,
            f"{DISCORD_API_BASE}/channels/{room_id}/typing",
            json_body={},
            headers={"Authorization": f"Bot {self.bot_token}"},
            http_client=self._http,
        )

    def _build_discord_client(self) -> tp.Any:
        """Create a ``discord.py`` client wired to this adapter."""
        try:
            import discord
            from discord import app_commands
        except ImportError as exc:
            raise RuntimeError("Discord gateway mode requires discord.py; reinstall or update xerxes-agent.") from exc

        self._discord = discord
        intents = discord.Intents.default()
        intents.guilds = True
        intents.messages = True
        if hasattr(intents, "message_content"):
            intents.message_content = self.message_content_intent
        outer = self

        class XerxesDiscordClient(discord.Client):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.tree = app_commands.CommandTree(self)
                self._xerxes_commands_synced = False
                outer._install_app_commands(self, app_commands)

            async def on_ready(self) -> None:
                if self.user is not None:
                    outer._bot_user_id = str(self.user.id)
                await outer._sync_app_commands(self, discord)

            async def on_message(self, message) -> None:
                await outer._handle_discord_py_message(message)

        return XerxesDiscordClient(intents=intents)

    def _install_app_commands(self, client: tp.Any, app_commands: tp.Any) -> None:
        """Install Discord application commands on the live client."""
        if not self.register_commands:
            return

        @client.tree.command(name="ask", description="Ask Xerxes to do something in this channel.")
        @app_commands.describe(prompt="Prompt to send to Xerxes.")
        async def ask(interaction, prompt: str) -> None:
            await self._handle_app_command(interaction, prompt)

        @client.tree.command(name="skills", description="List available Xerxes skills.")
        async def skills(interaction) -> None:
            await self._handle_app_command(interaction, "/skills")

        @client.tree.command(name="skill", description="Run a Xerxes skill.")
        @app_commands.describe(name="Skill name, optionally with :subcommand.", prompt="Optional task for the skill.")
        async def skill(interaction, name: str, prompt: str = "") -> None:
            text = f"/skill {name.strip()}"
            if prompt.strip():
                text = f"{text} {prompt.strip()}"
            await self._handle_app_command(interaction, text)

        @client.tree.command(name="status", description="Show Xerxes runtime status.")
        async def status(interaction) -> None:
            await self._handle_app_command(interaction, "/status")

    async def _sync_app_commands(self, client: tp.Any, discord: tp.Any) -> None:
        """Sync application commands to the visible guilds for fast availability."""
        if not self.register_commands or getattr(client, "_xerxes_commands_synced", False):
            return
        try:
            guild_ids = self.allowed_guild_ids or {str(getattr(guild, "id", "")) for guild in client.guilds}
            guild_ids = {item for item in guild_ids if item}
            if guild_ids:
                for guild_id in guild_ids:
                    guild = discord.Object(id=int(guild_id))
                    client.tree.copy_global_to(guild=guild)
                    await client.tree.sync(guild=guild)
            else:
                await client.tree.sync()
            client._xerxes_commands_synced = True
        except Exception:
            logger.warning("discord application command sync failed", exc_info=True)

    async def _handle_app_command(self, interaction: tp.Any, text: str) -> None:
        """Acknowledge one Discord app command and pass it to the daemon handler."""
        channel = getattr(interaction, "channel", None)
        guild = getattr(interaction, "guild", None)
        user = getattr(interaction, "user", None)
        channel_id = str(getattr(interaction, "channel_id", "") or getattr(channel, "id", "") or "")
        guild_id = str(getattr(interaction, "guild_id", "") or getattr(guild, "id", "") or "")
        if not self._routing_allows(
            channel_id=channel_id,
            channel_names=_channel_names_from_discord_py(channel),
            guild_id=guild_id,
            mentioned=True,
        ):
            try:
                if not interaction.response.is_done():
                    await interaction.response.send_message(
                        "This Xerxes instance is not configured for this channel.",
                        ephemeral=True,
                    )
            except Exception:
                logger.warning("discord interaction rejection failed", exc_info=True)
            return
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message("Queued.", ephemeral=True)
        except Exception:
            logger.warning("discord interaction acknowledgement failed", exc_info=True)
        if self._handler is None:
            return
        await self._handler(
            ChannelMessage(
                text=text,
                channel=self.name,
                channel_user_id=str(getattr(user, "id", "") or ""),
                room_id=channel_id,
                platform_message_id="",
                direction=MessageDirection.INBOUND,
                metadata={
                    "guild_id": guild_id,
                    "guild_name": str(getattr(guild, "name", "") or ""),
                    "thread_id": channel_id if getattr(channel, "parent_id", None) is not None else "",
                    "parent_channel_id": str(getattr(channel, "parent_id", "") or ""),
                    "channel_name": str(getattr(channel, "name", "") or ""),
                    "chat_type": "group" if guild_id else "private",
                    "discord_interaction": True,
                    "instance_name": self.instance_name,
                },
            )
        )

    def _record_gateway_result(self, task: asyncio.Task[tp.Any]) -> None:
        """Capture Gateway task failure for channel status and logs."""
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.gateway_error = str(exc)
            logger.warning("discord gateway stopped", exc_info=True)

    async def _handle_discord_py_message(self, message: tp.Any) -> None:
        """Convert one ``discord.py`` message event and pass it to the daemon."""
        if self._handler is None:
            return
        channel_message = self._message_from_discord_py(message)
        if channel_message is None:
            return
        await self._handler(channel_message)

    def _message_from_mapping(self, msg: dict[str, tp.Any]) -> ChannelMessage | None:
        """Build a neutral message from a Discord JSON message object."""
        author = msg.get("author") if isinstance(msg.get("author"), dict) else {}
        author_id = str(author.get("id", ""))
        channel_id = str(msg.get("channel_id", ""))
        channel_names = _channel_names_from_mapping(msg)
        guild_id = str(msg.get("guild_id", "") or "")
        content = str(msg.get("content", "") or "")
        attachments = _attachments_from_mapping(msg.get("attachments", []))
        if self.ignore_bots and bool(author.get("bot", False)):
            return None
        mentioned = self._bot_mentioned(msg.get("mentions", []), content)
        if not self._routing_allows(
            channel_id=channel_id,
            channel_names=channel_names,
            guild_id=guild_id,
            mentioned=mentioned,
        ):
            return None
        text = self._strip_bot_mention(content).strip()
        text = _strip_address(text, self.address_names)
        if text is None:
            return None
        if not text and attachments:
            text = _attachment_text(attachments)
        if not text:
            return None
        return ChannelMessage(
            text=text,
            channel=self.name,
            channel_user_id=author_id,
            room_id=channel_id,
            platform_message_id=str(msg.get("id", "")),
            attachments=attachments,
            direction=MessageDirection.INBOUND,
            metadata={
                "guild_id": guild_id,
                "thread_id": str(msg.get("thread_id", "") or ""),
                "channel_name": sorted(channel_names)[0] if channel_names else "",
                "instance_name": self.instance_name,
                "chat_type": "group" if guild_id else "private",
                "author_username": str(author.get("username", "")),
            },
        )

    def _message_from_discord_py(self, message: tp.Any) -> ChannelMessage | None:
        """Build a neutral message from a ``discord.py`` Message object."""
        author = getattr(message, "author", None)
        author_id = str(getattr(author, "id", "") or "")
        if self.ignore_bots and bool(getattr(author, "bot", False)):
            return None
        guild = getattr(message, "guild", None)
        guild_id = str(getattr(guild, "id", "") or "")
        channel = getattr(message, "channel", None)
        channel_id = str(getattr(channel, "id", "") or "")
        channel_names = _channel_names_from_discord_py(channel)
        mentioned = self._bot_mentioned(getattr(message, "mentions", []), getattr(message, "content", "") or "")
        if not self._routing_allows(
            channel_id=channel_id,
            channel_names=channel_names,
            guild_id=guild_id,
            mentioned=mentioned,
        ):
            return None
        attachments = _attachments_from_discord_py(getattr(message, "attachments", []))
        text = self._strip_bot_mention(str(getattr(message, "content", "") or "")).strip()
        text = _strip_address(text, self.address_names)
        if text is None:
            return None
        if not text and attachments:
            text = _attachment_text(attachments)
        if not text:
            return None
        thread_id = channel_id if getattr(channel, "parent_id", None) is not None else ""
        return ChannelMessage(
            text=text,
            channel=self.name,
            channel_user_id=author_id,
            room_id=channel_id,
            platform_message_id=str(getattr(message, "id", "") or ""),
            attachments=attachments,
            direction=MessageDirection.INBOUND,
            metadata={
                "guild_id": guild_id,
                "guild_name": str(getattr(guild, "name", "") or ""),
                "thread_id": thread_id,
                "parent_channel_id": str(getattr(channel, "parent_id", "") or ""),
                "channel_name": str(getattr(channel, "name", "") or ""),
                "instance_name": self.instance_name,
                "chat_type": "group" if guild_id else "private",
                "author_display_name": str(getattr(author, "display_name", "") or ""),
            },
        )

    def _routing_allows(
        self,
        *,
        channel_id: str,
        channel_names: set[str],
        guild_id: str,
        mentioned: bool,
    ) -> bool:
        """Apply guild/channel filters and mention policy."""
        if self.allowed_guild_ids and guild_id and guild_id not in self.allowed_guild_ids:
            return False
        if self.allowed_channel_ids and channel_id not in self.allowed_channel_ids:
            return False
        name_matches = bool(self.allowed_channel_names and self.allowed_channel_names.intersection(channel_names))
        if self.allowed_channel_names and not name_matches:
            return False
        if not guild_id:
            return True
        if self.always_reply_in_channels:
            return True
        if self.allowed_channel_ids and channel_id in self.allowed_channel_ids:
            return True
        if name_matches:
            return True
        return mentioned if self.require_mention else True

    def _bot_mentioned(self, mentions: tp.Any, content: str) -> bool:
        """Return true when the configured or learned bot id appears in mentions."""
        if not self._bot_user_id:
            return False
        for item in mentions if isinstance(mentions, list | tuple | set) else []:
            if isinstance(item, dict) and str(item.get("id", "")) == self._bot_user_id:
                return True
            if str(getattr(item, "id", "") or "") == self._bot_user_id:
                return True
        return f"<@{self._bot_user_id}>" in content or f"<@!{self._bot_user_id}>" in content

    def _strip_bot_mention(self, content: str) -> str:
        """Remove the leading bot mention that wakes the agent in guild channels."""
        if not self._bot_user_id:
            return content
        return re.sub(rf"<@!?{re.escape(self._bot_user_id)}>\s*", "", content).strip()

    async def _send_via_discord_py(self, message: ChannelMessage, chunk: str, *, include_reference: bool) -> bool:
        """Try sending through the live ``discord.py`` client."""
        client = self._client
        discord = self._discord
        if client is None or discord is None or self._http is not None:
            return False
        try:
            channel_id = int(str(message.room_id))
            channel = client.get_channel(channel_id)
            if channel is None:
                channel = await client.fetch_channel(channel_id)
            kwargs: dict[str, tp.Any] = {"content": chunk}
            if self.suppress_mentions:
                kwargs["allowed_mentions"] = discord.AllowedMentions.none()
            if include_reference and message.reply_to:
                kwargs["reference"] = discord.MessageReference(
                    message_id=int(str(message.reply_to)),
                    channel_id=channel_id,
                    fail_if_not_exists=False,
                )
            await channel.send(**kwargs)
            return True
        except Exception:
            logger.warning("discord.py send failed; falling back to REST", exc_info=True)
            return False

    async def _send_via_rest(self, message: ChannelMessage, chunk: str, *, include_reference: bool) -> None:
        """Send one chunk through Discord REST."""
        body: dict[str, tp.Any] = {"content": chunk}
        if self.suppress_mentions:
            body["allowed_mentions"] = {"parse": [], "replied_user": False}
        if include_reference and message.reply_to:
            body["message_reference"] = {"message_id": message.reply_to, "fail_if_not_exists": False}
        await asyncio.to_thread(
            http_post,
            f"{DISCORD_API_BASE}/channels/{message.room_id}/messages",
            json_body=body,
            headers={"Authorization": f"Bot {self.bot_token}"},
            http_client=self._http,
        )


def _as_bool(value: bool | str) -> bool:
    """Coerce config/env booleans."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in _FALSE_VALUES


def _as_id_set(value: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    """Normalize comma-separated or list-like id config into a string set."""
    if value is None:
        return set()
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part).strip() for part in value if str(part).strip()}


def _as_name_set(value: str | list[str] | tuple[str, ...] | set[str] | None) -> set[str]:
    """Normalize channel-name config into a case-insensitive set."""
    return {_normalize_channel_name(part) for part in _as_id_set(value) if _normalize_channel_name(part)}


def _normalize_channel_name(name: str) -> str:
    """Normalize a Discord channel name for routing comparisons."""
    return str(name).strip().lstrip("#").lower()


def _extract_message_payload(data: dict[str, tp.Any]) -> dict[str, tp.Any]:
    """Accept top-level, nested, or Gateway event-shaped message payloads."""
    if data.get("t") == "MESSAGE_CREATE" and isinstance(data.get("d"), dict):
        return tp.cast(dict[str, tp.Any], data["d"])
    nested = data.get("message")
    if isinstance(nested, dict):
        return nested
    return data


def _channel_names_from_mapping(msg: dict[str, tp.Any]) -> set[str]:
    """Extract channel/thread names from a JSON-shaped Discord message."""
    names = {
        _normalize_channel_name(str(msg.get("channel_name", ""))),
        _normalize_channel_name(str(msg.get("thread_name", ""))),
    }
    for key in ("channel", "thread", "parent_channel"):
        value = msg.get(key)
        if isinstance(value, dict):
            names.add(_normalize_channel_name(str(value.get("name", ""))))
    return {name for name in names if name}


def _channel_names_from_discord_py(channel: tp.Any) -> set[str]:
    """Extract the active and parent channel names from ``discord.py`` objects."""
    names = {_normalize_channel_name(str(getattr(channel, "name", "") or ""))}
    parent = getattr(channel, "parent", None)
    if parent is not None:
        names.add(_normalize_channel_name(str(getattr(parent, "name", "") or "")))
    return {name for name in names if name}


def _attachments_from_mapping(raw: tp.Any) -> list[dict[str, tp.Any]]:
    """Normalize Discord JSON attachments."""
    attachments: list[dict[str, tp.Any]] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        attachments.append(
            {
                "id": str(item.get("id", "")),
                "filename": str(item.get("filename", "")),
                "url": str(item.get("url", "")),
                "content_type": str(item.get("content_type", "")),
                "size": item.get("size", 0),
                "width": item.get("width"),
                "height": item.get("height"),
            }
        )
    return attachments


def _attachments_from_discord_py(raw: tp.Any) -> list[dict[str, tp.Any]]:
    """Normalize ``discord.py`` Attachment objects."""
    attachments: list[dict[str, tp.Any]] = []
    for item in raw:
        attachments.append(
            {
                "id": str(getattr(item, "id", "") or ""),
                "filename": str(getattr(item, "filename", "") or ""),
                "url": str(getattr(item, "url", "") or ""),
                "content_type": str(getattr(item, "content_type", "") or ""),
                "size": getattr(item, "size", 0),
                "width": getattr(item, "width", None),
                "height": getattr(item, "height", None),
            }
        )
    return attachments


def _attachment_text(attachments: list[dict[str, tp.Any]]) -> str:
    """Produce a usable prompt body for attachment-only messages."""
    urls = [str(item.get("url", "")) for item in attachments if item.get("url")]
    return "Attachments:\n" + "\n".join(urls) if urls else "Attachments received."


def _label_outbound(text: str, instance_name: str) -> str:
    """Prefix outbound text with the configured instance/device label."""
    if not instance_name:
        return text
    return f"[{instance_name}]\n{text}"


def _strip_address(text: str, address_names: set[str]) -> str | None:
    """Strip a leading per-instance address, or reject when no address matches."""
    if not address_names:
        return text
    stripped = text.strip()
    lowered = stripped.lower()
    for name in address_names:
        for prefix in (name, f"@{name}", f"/{name}"):
            if lowered == prefix:
                return ""
            if not lowered.startswith(prefix):
                continue
            next_char = lowered[len(prefix) : len(prefix) + 1]
            if next_char in {" ", ":", ",", "-"}:
                return stripped[len(prefix) :].lstrip(" :,-")
    return None


def _chunk_text(text: str, limit: int) -> list[str]:
    """Split outbound text under Discord's message size limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n", 0, limit)
        if split_at < max(1, limit // 2):
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    return chunks
