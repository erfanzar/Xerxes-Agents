# Xerxes in GitHub Actions

Xerxes runs headless in CI exactly like it runs in your terminal: one Bun
process, one prompt, one answer. The same capabilities that power the
interactive TUI — provider routing, tools, skills, hooks — are available to
automation through the one-shot CLI path.

## Quick start: PR review

Copy [`examples/github-action-pr-review.yml`](https://github.com/erfanzar/Xerxes-Agents/blob/main/examples/github-action-pr-review.yml)
to `.github/workflows/xerxes-review.yml` in your repository. On every pull
request it:

1. checks out the PR and collects the diff (capped at 200 KB),
2. runs a headless review turn with `xerxes --output-format json`,
3. posts the findings as a PR comment, updating the same comment on new
   pushes instead of spamming new ones.

Configure it with repository settings — no config files are committed:

| Setting                    | Kind   | Purpose                                        |
| -------------------------- | ------ | ---------------------------------------------- |
| `XERXES_API_KEY`           | secret | Provider API key (required)                    |
| `XERXES_MODEL`             | var    | Model to review with (required)                |
| `XERXES_BASE_URL`          | var    | Optional OpenAI-compatible endpoint override   |
| `XERXES_FALLBACK_MODEL`    | var    | Optional model used when the primary overloads |

## Headless CLI reference

```sh
# Plain text to stdout (default)
echo "summarize this repo" | xerxes

# One buffered JSON result object: { type, response, session_id, model, usage, is_error }
xerxes --output-format json "review staged changes"

# NDJSON event stream, one JSON object per line as the turn runs
xerxes --output-format stream-json "list the entry points"

# Resume a persisted daemon session non-interactively
xerxes --resume <session_id> --output-format json "continue the refactor"
```

A turn that ends in a terminal provider failure exits non-zero and reports
`is_error: true` in JSON mode, so `set -e` and CI status checks do the right
thing without parsing output.

## Scripting with stream-json

Each line is a self-contained event. Text deltas, tool boundaries, and the
final result are distinguishable by `type`:

```sh
xerxes --output-format stream-json "tally the test files" \
  | bun -e '
      for await (const line of console) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);
        if (event.type === "result") console.error("done:", event.usage ?? "");
        if (event.type === "text") process.stdout.write(event.text);
      }
    '
```

## Notes and limits

- One-shot mode runs with `accept-all` permissions and never waits for an
  interactive approval — that is what makes it safe to leave unattended in
  CI. If a workflow needs a human gate, put the gate in the workflow, not in
  the turn.
- The PR template installs from the published `@xsimurgh/xerxes-agents` npm package. To
  track a source checkout instead, replace the install step with
  `bun install --frozen-lockfile && bun run xerxes …` against your clone.
- Hooks from `~/.xerxes/config.yaml` also run in CI (`~` is the runner user),
  so `PreToolUse`/`PostToolUse` policies apply to automated turns the same
  way they apply locally.
