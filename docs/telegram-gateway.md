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

## Approvals and questions in the conversation

When a channel turn raises an approval or question prompt — for example an always-approval
tool such as `send_message`, or the ask-user question tool — the router forwards the request
to the originating Telegram conversation and parks the turn until that conversation answers.
The next inbound message is interpreted as the answer instead of starting a new turn:

- approvals accept `yes` (approve once), `session` (approve for this session), or `no` (deny);
  any other reply re-sends the usage line and keeps the request pending;
- questions accept freeform text, or the 1-based number of a listed option.

A request that expired before the answer arrived (aborted turn, daemon restart) is reported as
no longer pending rather than resolved. Without this routing the prompt would park unanswered,
because channel conversations have no terminal to prompt on.

Treat inbound channel content as untrusted. The configured runtime still applies policy,
permissions, prompt scanning, path safety, and the selected tool sandbox before executing a turn.
Use injected transports in tests so no real Telegram credential or network call is required.
