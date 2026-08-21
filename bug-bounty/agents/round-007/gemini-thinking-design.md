# Round 007 — Gemini neutral thinking mapping

## Status

Confirmed missing mapping. Test/design only because `xerxes/src/llms/gemini.ts` has concurrent ownership; no production source was edited.

## Finding

`CompletionRequest.thinking` is the project's provider-neutral contract:

```ts
interface ThinkingRequest {
  readonly budgetTokens?: number
  readonly effort?: string
}
```

The runtime resolves ordinary session thinking to a 10,000-token budget and medium effort, while prompt escalation supplies explicit budgets. Anthropic's adapter establishes the convention for a budget-based provider: consume `budgetTokens`, ignore the effort hint, and use 10,000 when a request is effort-only.

Gemini's native Generate Content adapter currently builds `generationConfig` from output/sampling settings only. Both `complete()` and `stream()` call the same `geminiRequestPayload()` / `requestGenerationConfig()` path, but `request.thinking` is never read. Consequently, resolved thinking directives are silently dropped for Gemini even though response thought parts and thought-token usage are already normalized.

## Exact native mapping

For Gemini's native `generateContent` / `streamGenerateContent` request, map the neutral directive under `generationConfig.thinkingConfig`:

```json
{
  "generationConfig": {
    "thinkingConfig": {
      "thinkingBudget": 4000,
      "includeThoughts": true
    }
  }
}
```

Recommended project-convention mapping:

- `request.thinking === undefined`: omit `thinkingConfig` entirely, preserving model defaults and current behavior.
- `request.thinking.budgetTokens` present: use it as `thinkingBudget`.
- effort-only request: use `thinkingBudget: 10000`, matching `resolveTurnThinking()` session defaults and the Anthropic budget-provider fallback.
- always set `includeThoughts: true` when neutral thinking is requested, because Xerxes exposes provider-neutral `thinking` deltas/completions; enabling a budget without requesting thought output would make that observable contract inconsistent.
- do not translate `effort` independently. Gemini's budget-based wire contract has no equivalent graded-effort field in this adapter's native API convention.

## Proposed production change (not applied)

In `requestGenerationConfig()` in `xerxes/src/llms/gemini.ts`, after copying ordinary settings:

```ts
if (request.thinking !== undefined) {
  generationConfig.thinkingConfig = {
    thinkingBudget: request.thinking.budgetTokens ?? 10_000,
    includeThoughts: true,
  }
}
```

Optional hardening: validate the neutral budget as a non-negative safe integer before serialization. The current runtime emits positive integer ladder values, but `CompletionRequest` is public and direct callers can provide malformed numbers. This validation policy should be coordinated with the other adapters rather than invented only for Gemini.

## Regression tests

Added `xerxes/test/geminiThinking.test.ts`, covering observable fetch payloads for:

1. non-streaming explicit budget + effort maps to native `thinkingConfig`, with effort ignored;
2. streaming effort-only request maps to the shared 10,000-token fallback;
3. an absent neutral request omits `thinkingConfig`.

Pre-fix focused result:

```text
bun test xerxes/test/geminiThinking.test.ts
1 pass
2 fail
3 expect() calls
```

Both failures show the actual payload contains only `contents`; the expected `generationConfig.thinkingConfig` is absent. The omission test passes.

## Files

- Added: `xerxes/test/geminiThinking.test.ts`
- Added: `bug-bounty/agents/round-007/gemini-thinking-design.md`
- Not modified: `xerxes/src/llms/gemini.ts`
