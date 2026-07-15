# Telegram gateway

The native CLI can start a daemon with Telegram settings:

```sh
export TELEGRAM_BOT_TOKEN="…"
bun run xerxes telegram --project-dir .
```

Pass `--token` to supply a token for the process without storing it in a command profile:

```sh
bun run xerxes telegram --token "$TELEGRAM_BOT_TOKEN" --project-dir .
```

The command loads the native daemon configuration, enables the Telegram channel, and then starts
the configured daemon control surfaces. Use `--host`, `--port`, `--socket`, and `--pid-file` for
the documented daemon launch options. Channel-specific behavior and webhook setup belong in the
explicit daemon configuration; do not assume a transport or remote registration occurred without a
configured adapter.

For a webhook deployment, configure the native adapter explicitly. The channel router keeps one
daemon session per Telegram conversation, journals safe input and final replies into the Markdown
workspace, and loads that workspace as trusted per-turn system context. Telegram replies stream as
one edited preview by default; set `stream_previews` to `false` to send only the final reply.

```json
{
  "workspace": { "root": "~/.xerxes/agents/default" },
  "channels": {
    "telegram": {
      "type": "telegram",
      "enabled": true,
      "settings": {
        "token_env": "TELEGRAM_BOT_TOKEN",
        "transport": "webhook",
        "webhook_url": "https://bot.example/channels/telegram/webhook",
        "webhook_secret_token_env": "XERXES_TELEGRAM_WEBHOOK_SECRET",
        "allowed_user_ids": ["123456789"],
        "bot_username": "xerxes_bot",
        "require_allowed_sender": true,
        "stream_previews": true,
        "preview_interval": 1
      }
    }
  }
}
```

`preview_interval` is in seconds. Long-polling clears an existing Telegram webhook before it
receives updates. The allowlist is fail-closed when `require_allowed_sender` is enabled.

Treat inbound channel content as untrusted. The configured runtime still applies policy,
permissions, prompt scanning, path safety, and the selected tool sandbox before executing a turn.
Use injected transports in tests so no real Telegram credential or network call is required.
