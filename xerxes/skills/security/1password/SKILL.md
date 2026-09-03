---
name: 1password
description: Set up the 1Password CLI (op), sign in, and read or inject secrets without plaintext files.
version: 1.0.0
author: arceus77-7 (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [security, secrets, 1password, op, cli]
source: https://github.com/NousResearch/hermes-agent/tree/main/optional-skills/security/1password
---

# 1Password CLI

Use this skill when the user wants secrets managed through 1Password
instead of plaintext env vars or files.

## When to Use

- Install or configure the 1Password CLI (`op`)
- Sign in with `op signin`
- Read secret references like `op://Vault/Item/field`
- Inject secrets into config/templates using `op inject`
- Run commands with secret env vars via `op run`

## Requirements

- 1Password account
- 1Password CLI (`op`) installed — `brew install 1password-cli`
  (macOS), `winget install AgileBits.1Password.CLI` (Windows), or the
  official Linux packages
- One of: desktop app integration, service account token
  (`OP_SERVICE_ACCOUNT_TOKEN`), or Connect server

## Authentication Methods

### Service account (recommended for agents)

Set `OP_SERVICE_ACCOUNT_TOKEN` in the environment (create a service
account at https://my.1password.com → Settings → Service Accounts). No
desktop app needed; the token persists across shell-tool calls.

```bash
op whoami   # verify — should show Type: SERVICE_ACCOUNT
```

### Desktop app integration (interactive)

1. Enable in the 1Password desktop app: Settings → Developer →
   Integrate with 1Password CLI
2. Ensure the app is unlocked
3. Run `op signin` and approve the biometric prompt

Because agent shell calls are non-interactive and can lose auth context
between calls, run sign-in and secret operations inside a dedicated
tmux session when using this flow:

```bash
SOCKET="${TMPDIR:-/tmp}/op-auth.sock"
tmux -S "$SOCKET" new -d -s op-auth
tmux -S "$SOCKET" send-keys -t op-auth -- \
  'eval "$(op signin --account my.1password.com)"' Enter
tmux -S "$SOCKET" send-keys -t op-auth -- 'op whoami' Enter
tmux -S "$SOCKET" capture-pane -p -J -t op-auth -S -50
```

### Connect server (self-hosted)

```bash
export OP_CONNECT_HOST="http://localhost:8080"
export OP_CONNECT_TOKEN="your-connect-token"
```

## Common Operations

```bash
# Read a secret
op read "op://app-prod/db/password"

# Get a one-time password
op read "op://app-prod/npm/one-time password?attribute=otp"

# Inject into a template
echo "db_password: {{ op://app-prod/db/password }}" | op inject

# Run a command with a secret env var (op resolves the reference)
op run -- sh -c 'echo "$DB_PASSWORD"'
```

## Guardrails

- Never print raw secrets back to the user unless they explicitly ask
  for the value.
- Prefer `op run` / `op inject` over writing secrets into files.
- "account is not signed in" → run `op signin` again (same tmux session
  if using desktop-app flow).
- Headless/CI: use the service account token flow; service accounts
  require CLI v2.18.0+.

## Verification

- `op --version` and `op whoami` succeed
- A test `op read` on a known reference returns the secret without
  echoing it into logs or chat more than the user asked for

## References

- https://developer.1password.com/docs/cli/
- https://developer.1password.com/docs/service-accounts/

---

Adapted from the `1password` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright arceus77-7 and Hermes Agent contributors.
