---
name: add-channel-adapter
description: Scaffold a new messaging channel adapter in the Xerxes channels/ directory. Covers Channel/WebhookChannel subclassing, start/stop/send/parse_inbound, and registry registration.
version: 1.0.0
tags: [channels, adapter, messaging, xerxes]
required_tools: [ReadFile, WriteFile, FileEditTool, GlobTool]
---

# When to use

Use this skill when adding a new messaging platform adapter to Xerxes. The framework supports 14+ adapters (Telegram, Slack, Discord, Email, Matrix, WhatsApp, etc.) and adding a new one follows a well-defined pattern.

Examples:
- A new chat platform (e.g., Microsoft Teams, Discord forum channels, IRC)
- A new webhook-based gateway (e.g., custom internal notification system)
- A new SMS/notification provider (e.g., Twilio, Vonage, PagerDuty)

Do NOT use this for:
- Adding a new LLM provider (use `add-llm-provider` skill)
- Adding a new tool (use `add-tool-module` skill)
- Adding a new memory backend (use `add-memory-backend` skill)

# How to use

## 1. Inspect the base classes

Read `src/python/xerxes/channels/base.py` to understand the `Channel` ABC:

```python
class Channel(ABC):
    name: str = ""

    @abstractmethod
    def start(self, on_inbound: Callable[[ChannelMessage], None]) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def send(self, message: ChannelMessage) -> None: ...
```

For webhook-based platforms, read `WebhookChannel` in the same file:

```python
class WebhookChannel(Channel, ABC):
    @abstractmethod
    def _parse_inbound(self, headers: dict, body: bytes) -> list[ChannelMessage]: ...

    @abstractmethod
    def _send_outbound(self, message: ChannelMessage) -> None: ...
```

Also read one existing adapter as a reference:
- `src/python/xerxes/channels/adapters/telegram.py` — polling-based
- `src/python/xerxes/channels/adapters/slack.py` — webhook-based
- `src/python/xerxes/channels/adapters/email_imap.py` — IMAP polling

## 2. Create the adapter module

Create a new file under `src/python/xerxes/channels/adapters/` (e.g., `my_platform.py`):

```python
# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
# ... (Apache-2.0 header)

from __future__ import annotations
import logging
from typing import Any

from ..base import Channel, ChannelMessage, WebhookChannel

logger = logging.getLogger(__name__)


class MyPlatformChannel(Channel):
    """Adapter for MyPlatform messaging."""

    name = "my_platform"

    def __init__(self, token: str | None = None, **kwargs: Any) -> None:
        self.token = token
        self._client = None
        self._on_inbound: Callable[[ChannelMessage], None] | None = None

    def start(self, on_inbound: Callable[[ChannelMessage], None]) -> None:
        """Start receiving messages."""
        self._on_inbound = on_inbound
        # For polling: start a background thread or asyncio task
        # For webhook: register URL handler with the daemon
        logger.info("MyPlatform channel started")

    def stop(self) -> None:
        """Stop receiving messages and close connections."""
        if self._client:
            self._client.close()
        logger.info("MyPlatform channel stopped")

    def send(self, message: ChannelMessage) -> None:
        """Send an outbound message."""
        # Implement platform-specific send logic
        pass
```

**Rules:**
- The class MUST set `name` as a class attribute (snake_case, matching the module name convention).
- `start()`, `stop()`, and `send()` are mandatory for all channels.
- For webhook-based channels, subclass `WebhookChannel` and implement `_parse_inbound()` and `_send_outbound()` instead.

## 3. Register the adapter

Open `src/python/xerxes/channels/registry.py` (or the equivalent registration point) and add:

```python
from .adapters.my_platform import MyPlatformChannel

# In the registry setup or factory function:
registry.register("my_platform", MyPlatformChannel)
```

If the framework uses entry-point or auto-discovery registration, add the adapter to the discovery list. Check how existing adapters are registered (grep for `registry.register` or `ChannelRegistry` in `channels/`).

## 4. Add to `__init__.py` re-exports

Open `src/python/xerxes/channels/__init__.py` and add the adapter to `__all__` if the package re-exports adapters:

```python
from .adapters.my_platform import MyPlatformChannel

__all__ = [
    # ... existing adapters ...
    "MyPlatformChannel",
]
```

## 5. Add configuration support (if needed)

If the channel requires configuration (e.g., API token, webhook URL), check how existing channels load config:

- Read `src/python/xerxes/channels/_helpers.py` for config parsing helpers.
- Read `src/python/xerxes/core/config.py` to see if channel config is part of `XerxesConfig`.

Many channels accept config via:
- Environment variables (e.g., `TELEGRAM_BOT_TOKEN`)
- Constructor arguments passed by the daemon or bridge
- YAML config files

## 6. Add tests

Create `tests/channels/adapters/test_my_platform.py`:

```python
import pytest
from xerxes.channels.adapters.my_platform import MyPlatformChannel
from xerxes.channels.base import ChannelMessage


class TestMyPlatformChannel:
    def test_name_is_set(self):
        assert MyPlatformChannel.name == "my_platform"

    def test_start_and_stop(self):
        ch = MyPlatformChannel(token="dummy")
        received = []
        ch.start(lambda msg: received.append(msg))
        ch.stop()
        # Assert no exceptions and proper cleanup

    def test_send_outbound(self):
        ch = MyPlatformChannel(token="dummy")
        msg = ChannelMessage(text="hello", channel="my_platform", sender_id="u1")
        # Mock the HTTP client and assert send() calls it correctly
```

**Rules:**
- Mock all external HTTP/API calls. Never hit real endpoints in tests.
- Test `start()`/`stop()` lifecycle, `send()` outbound, and `_parse_inbound()` for webhooks.
- For webhook channels, test `_parse_inbound()` with sample HTTP request bodies.

## 7. Run lint and type check

```bash
uv run ruff check --fix src/python/xerxes/channels/adapters/my_platform.py
uv run mypy src/python/xerxes/channels/adapters/my_platform.py --ignore-missing-imports
```

## 8. Verify registration

Run:

```bash
uv run python -c "
from xerxes.channels.registry import ChannelRegistry
# or whatever the registration mechanism is
print('my_platform' in registry.list_channels())
"
```

## Common pitfalls

- **Missing `name` attribute:** The channel won't be discoverable by the registry.
- **Not implementing `stop()`:** Resource leaks (open HTTP connections, background threads) will accumulate.
- **Blocking `start()`:** `start()` should return immediately and spin up background work asynchronously. Don't block the caller.
- **Uncaught exceptions in background threads:** Any exception in the polling loop should be logged and the channel should attempt to reconnect, not crash silently.
- **Webhook URL construction without `urllib.parse`:** Always use `urllib.parse` for URL construction to avoid injection bugs (e.g., `urllib.parse.urljoin(base, endpoint)`).
- **Forgetting to handle `ChannelMessage` type:** Ensure `send()` accepts the correct `ChannelMessage` type from `channels/base.py`.
