# Telegram Gateway

Xerxes can run as a Telegram bot through the channel gateway.

Long polling is the default, so no public HTTPS endpoint is required:

```bash
export TELEGRAM_BOT_TOKEN="123456:..."
xerxes telegram
```

Webhook mode is also supported:

```bash
export TELEGRAM_BOT_TOKEN="123456:..."
xerxes telegram \
  --transport webhook \
  --host 0.0.0.0 \
  --port 11997 \
  --webhook-url https://your-domain.example/telegram/webhook
```

If `--transport webhook` is used and `--webhook-url` is omitted, the server
still exposes `POST /telegram/webhook`; you can register the webhook yourself
through Telegram's Bot API.

The gateway uses the active Xerxes provider profile. Configure that first in the
TUI with `/provider`, or pass `--model`, `--base-url`, and `--api-key`.

## Workspace Memory

Telegram sessions load a Markdown workspace before every agent turn. The default
location is:

```text
$XERXES_HOME/agents/default
```

If `XERXES_HOME` is not set, this is `~/.xerxes/agents/default`.

Files:

| File | Purpose |
|---|---|
| `AGENTS.md` | Operating rules for channel sessions |
| `SOUL.md` | Personality, values, tone, and boundaries |
| `IDENTITY.md` | Optional presentation identity |
| `USER.md` | Operator/user preferences and stable context |
| `TOOLS.md` | Environment-specific tool notes |
| `MEMORY.md` | Durable long-term memory |
| `memory/YYYY-MM-DD.md` | Daily notes; today and yesterday are loaded |

The gateway creates safe default files on first run. Each Telegram turn appends
a short inbound/outbound note to today's daily memory file. `SOUL.md` and
`MEMORY.md` are loaded as context, but the gateway does not directly rewrite
them.

## Channel Behavior

- Long polling is the default transport; webhook is optional.
- Telegram receives a single preview message while the agent streams. The
  gateway edits that message as text arrives, then edits it to the final answer.
  Use `--no-stream-previews` to send only final replies.
- Private chats are handled directly.
- Group and supergroup messages are ignored unless they mention `xerxes` or use
  `/xerxes`.
- Sessions are isolated by private user ID, or by group chat plus forum topic
  thread ID when present.
- Channel messages are treated as untrusted input in the default `AGENTS.md`.

## Local Testing

For local webhook testing, expose the port with a tunnel and use that public URL:

```bash
ngrok http 11997
xerxes telegram --webhook-url https://<ngrok-id>.ngrok-free.app/telegram/webhook
```
