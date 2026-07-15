---
name: github-auth
description: Use GitHub authentication that the user or host has explicitly supplied through gh, an injected host integration, SSH, or the current process environment.
version: 1.2.0
author: Xerxes Agent
license: MIT
metadata:
  xerxes:
    tags: [GitHub, Authentication, Git, gh-cli, SSH, Setup]
    related_skills: [github-pr-workflow, github-code-review, github-issues, github-repo-management]
---

# GitHub Authentication

Use only authentication that the user or host has deliberately made available. Authentication discovery is read-only and must never turn into credential-file discovery.

## Allowed authentication sources

Use the first available source:

1. A host-provided GitHub integration or credential port.
2. An existing `gh` session confirmed by `gh auth status`.
3. An SSH or Git credential helper already configured by the user for ordinary `git` commands.
4. A `GITHUB_TOKEN` already present in the current process environment.

Never:

- read dotfiles, Git credential stores, keychains, shell history, or unrelated environment files to find a token;
- run `gh auth token`, print a token, place one in a remote URL, or copy one into a command transcript;
- create, persist, refresh, or broaden credentials on the user's behalf;
- begin an interactive login unless the user explicitly asks for setup and remains in control of the prompt.

## Safe detection

```bash
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  echo "GitHub authentication: existing gh session"
elif [ -n "${GITHUB_TOKEN:-}" ]; then
  echo "GitHub authentication: host-supplied environment token"
else
  echo "GitHub authentication required: ask the user to configure gh, SSH, or a host integration"
  exit 1
fi
```

This check establishes availability only. It does not reveal or relocate the credential.

## User-controlled setup

If no authentication is available, stop and ask the user to choose and complete one of these paths outside the agent's command flow.

### Existing gh session

The user runs:

```bash
gh auth login
gh auth status
```

Afterward, verify only with `gh auth status`. Prefer `gh` for API requests because it owns token handling.

### SSH for Git operations

The user configures an SSH key and adds its public key to GitHub. Verify without reading private key material:

```bash
ssh -T git@github.com
git remote -v
```

Do not enumerate or open private keys. Normal `git fetch`, `pull`, and `push` operations should use the user's configured SSH agent.

### Host-injected token

A trusted host may inject `GITHUB_TOKEN` into the current process. Check only whether it is present. Do not echo, persist, transform, or recover it from another location.

Prefer a host GitHub adapter. When a direct REST request is unavoidable, keep the token in the environment and keep response/error logging redacted.

## Repository identity

Repository metadata is not secret and may be read from the current checkout:

```bash
REMOTE_URL=$(git remote get-url origin)
OWNER_REPO=$(printf '%s' "$REMOTE_URL" | sed -E 's|.*github\.com[:/]||; s|\.git$||')
printf 'Repository: %s\n' "$OWNER_REPO"
```

Do not rewrite remotes, global Git configuration, credential helpers, or SSH configuration unless the user explicitly requests that mutation.

## Verification

Use the least-privileged read-only operation that exercises the selected path:

```bash
gh auth status
gh api user --jq .login
git ls-remote --heads origin
```

Do not test authentication by creating a repository, pushing a commit, changing a setting, or writing a secret.

## Troubleshooting

| Problem | Safe response |
|---|---|
| `gh` is not authenticated | Ask the user to run `gh auth login`; do not start or scrape the flow automatically |
| Git prompts for credentials | Ask the user to configure SSH or their platform credential helper |
| Permission denied | Report the operation and required scope without requesting a broader token automatically |
| Multiple accounts | Ask the user which existing host/`gh`/SSH identity to use |
| Token environment is absent | Stop with an actionable authentication requirement; do not search the filesystem |
