# Terminal design guidelines

Xerxes' terminal client is a React application rendered by OpenTUI from `src/typescript/src/ui/`.
It should feel calm and fast: the prompt is the focus, session context is compact, and tools or
approvals appear only when needed.

## Visual principles

- Use a quiet dark canvas with a readable light theme fallback.
- Keep workspace, model, mode, and session context in a thin header or footer rather than a large
  dashboard.
- Make the existing `❯` prompt visually dominant and keep keyboard hints concise.
- Use the code-native Xerxes/Derafsh Kaviani mark consistently. Do not import or imitate third
  party logos, artwork, or proprietary product text.
- Prefer spacing, divider rules, typography, and color contrast over decorative chrome.

## Interaction principles

- Keyboard operation is complete without a pointer; mouse behavior is additive.
- Preserve transcript virtualization, streaming states, approvals, clarification, and session
  overlays when changing layout.
- Ensure narrow terminals collapse nonessential metadata before hiding input or active turn status.
- Screen-reader/plain-text output must retain the same information without depending on color or
  glyph-only state.
- A visual refactor must not change daemon v35 payloads or gateway behavior.

## Verification

Use component tests for mark rendering, theme tokens, compact/wide decisions, and status labels.
Then run:

```sh
bun run typecheck
bun run test:ui
bun run build:ui
bun run --cwd src/typescript smoke:ui
```
