# Design Guidelines

Xerxes has one user-facing "design surface": the Rust ratatui TUI. This document covers its structure, theming, and interaction conventions.

(For backend Python conventions, see [code-standards.md](code-standards.md).)

## Why Rust for the TUI?

The TUI has to:

- Render streaming LLM tokens as they arrive (no per-token allocation glitches).
- Respond to keystrokes at 60 Hz while the LLM is streaming.
- Display syntax-highlighted code, diffs, markdown, and tool output without screen tearing.
- Fail gracefully when the Python side crashes or hangs.

Rust + ratatui gives us deterministic rendering, zero-GC input handling, and the ability to spawn/manage the Python process as a subprocess with clean IPC. Python → ratatui bindings exist but add a GIL + FFI layer in a latency-sensitive path.

## Workspace layout

```
src/rust/
├── Cargo.toml                   # Workspace root
└── xerxes-cli/
    ├── Cargo.toml
    └── src/
        ├── main.rs              # Clap CLI + terminal setup + event loop
        ├── app.rs               # Application state machine
        ├── bridge.rs            # JSON-RPC with the Python bridge subprocess
        ├── events.rs            # Request / Event protocol structs
        ├── render.rs            # All ratatui rendering (largest file)
        ├── input.rs             # Crossterm keyboard handling
        ├── slash.rs             # Slash command catalogue
        ├── markdown.rs          # Markdown → ratatui Lines
        ├── diff.rs              # Unified diff rendering
        ├── theme.rs             # Color scheme + Style helpers
        └── spinner.rs           # Animated progress spinner
```

**Rules of thumb:**

- Never put render code outside `render.rs`.
- Never put state mutation outside `app.rs`.
- Never put IPC logic outside `bridge.rs`.

This split keeps rebuild times low (edits to `render.rs` don't recompile `bridge.rs`) and makes it obvious where a given kind of change belongs.

## State machine

```rust
pub enum Mode {
    Input,              // accepting user keystrokes
    Streaming,          // LLM response in progress
    Permission,         // waiting for the user to approve a tool call
    ProviderSetup,      // multi-step wizard: profile → base_url → api_key → name → model
    SlashPopup,         // slash-command autocomplete list visible
}
```

Transitions are driven by `Event` messages from the Python bridge and keystrokes from `input.rs`. The invariant: at any time, *exactly one* mode is active; render logic dispatches on mode and draws the appropriate widget tree.

## Cell-based conversation model

The conversation is a scrollable list of `Cell` values:

```rust
pub enum Cell {
    User { text: String },
    Assistant { lines: Vec<Line> },      // streaming text, appended in place
    Thinking { lines: Vec<Line> },
    ToolCall { name, args, status },
    ToolResult { name, output, duration_ms, truncated: bool },
    Error { message: String },
    System { message: String },
    Divider,
}
```

`render.rs` walks the cell list and draws each as a bordered block with cell-specific styling. Long tool outputs are truncated to the first 5 lines in-place; the full output is available via `/last-output` slash command.

## Theme

[theme.rs](../src/rust/xerxes-cli/src/theme.rs) centralizes all colors and text styles:

```rust
pub fn user() -> Style        // human input accent
pub fn assistant() -> Style   // LLM response text
pub fn thinking() -> Style    // reasoning / scratch pad
pub fn tool() -> Style        // tool call headers
pub fn tool_bullet() -> Style // tool output bullets
pub fn error() -> Style       // error cells
pub fn spinner() -> Style     // progress indicator
pub fn heading1() -> Style    // markdown H1
pub fn heading2() -> Style    // markdown H2
pub fn code() -> Style        // inline code + fenced blocks
```

**Never hard-code a `Color::*` in render code.** Always go through `theme::`. This makes it one edit to retheme the whole app (dark→light, high-contrast variants, etc.).

## Slash commands

[slash.rs](../src/rust/xerxes-cli/src/slash.rs) enumerates the 20 supported commands:

| Command | Purpose |
|---------|---------|
| `/help` | Show command list |
| `/provider` | Switch / add / remove provider profile |
| `/model` | Change active model |
| `/sampling` | Tune temperature / top_p / max_tokens interactively |
| `/cost` | Show cumulative token + USD cost |
| `/context` | Inspect current transcript size + remaining budget |
| `/clear` | Reset conversation (warn before) |
| `/compact` | Trigger context compaction manually |
| `/history` | Scroll session history |
| `/tools` | List active tool catalogue |
| `/skills` | List available skills |
| `/skill <name>` | Invoke a specific skill |
| `/skill-create` | Enter skill-authoring mode |
| `/plan` | Show current plan state (if any) |
| `/agents` | List registered agents |
| `/thinking` | Toggle thinking-chunk display |
| `/verbose` | Toggle verbose event display |
| `/debug` | Toggle debug overlay |
| `/permissions` | Set permission mode (auto/accept-all/manual) |
| `/config` | Show effective config |
| `/exit` | Quit cleanly |

Pressing `/` opens a fuzzy-match popup over the command list; Tab completes the first match; Enter executes.

## Input layer

[input.rs](../src/rust/xerxes-cli/src/input.rs) implements emacs-style line editing:

| Binding | Action |
|---------|--------|
| `Ctrl-a` / `Home` | Move to start |
| `Ctrl-e` / `End` | Move to end |
| `Ctrl-u` | Clear line before cursor |
| `Ctrl-k` | Clear line after cursor |
| `Ctrl-w` | Delete previous word |
| `Alt-←` / `Alt-→` | Word motion |
| `Up` / `Down` | Scroll conversation |
| `PageUp` / `PageDown` | Scroll conversation by page |
| `Ctrl-r` | Reverse-search input history |
| `Esc` | Cancel current operation / close popup |
| `Enter` | Submit prompt (send `query` request to bridge) |
| `Shift-Enter` / `Cmd-Enter` | Newline (multi-line input) |
| `Ctrl-c` | Interrupt current streaming response |
| `Ctrl-d` (empty line) | Exit |

**Bracketed paste is enabled** — multi-line pastes don't spray the text into the input as if each line were a submit. Crossterm emits a `Paste` event that's routed to the input buffer.

## Markdown rendering

[markdown.rs](../src/rust/xerxes-cli/src/markdown.rs) converts markdown into ratatui `Line` / `Span` values. Supported syntax:

- Headings (`#`, `##`, `###`) — colored + bolded via `theme::heading1/2`.
- Bold (`**text**`, `__text__`) and italic (`*text*`, `_text_`) — spans with modifier.
- Inline code (`` `text` ``) — styled via `theme::code`.
- Fenced code blocks (``` ```lang\n…\n``` ```) — rendered as paragraph in code style.
- Bullet lists (`-`, `*`, `+`) with nesting.
- Numbered lists.
- Blockquotes (`>`).
- Horizontal rules (`---`).

Not supported (kept simple on purpose): tables, HTML passthrough, LaTeX, footnotes. For rich rendering, let the user copy output to a browser.

## Diff rendering

[diff.rs](../src/rust/xerxes-cli/src/diff.rs) parses unified diff output and colorizes:

- Added lines (`+…`) in green.
- Removed lines (`-…`) in red.
- Context lines (`…`) in default.
- Hunk headers (`@@ … @@`) in cyan.
- File headers (`--- / +++`) in bold.

Used by the `FileEditTool` result cell to show what changed when the agent edits a file.

## Progress indication

While waiting for the LLM or a tool, `spinner.rs` animates a small rotating glyph next to the current turn marker. Frame rate: ~10 Hz — fast enough to feel alive, slow enough not to distract.

## Rendering rules

1. **Never block the event loop.** All I/O happens in separate tasks; the render path reads from a channel.
2. **Clear+redraw, never patch.** Ratatui expects a full terminal buffer per frame; partial updates are the framework's job, not ours.
3. **Wrap long text** at viewport width — use `Line` / `Span` APIs, not hand-wrapped strings.
4. **Preserve scroll position** across redraws. If the user has scrolled back, new tokens don't yank them to the bottom (that's `app.following_tail`).
5. **Graceful degradation for small terminals.** At <20 rows or <40 cols, hide non-essential panels.

## Accessibility

- All color uses respect the terminal's color profile (256-color vs 16-color vs truecolor).
- Text has sufficient contrast in both light and dark terminal themes (tested manually).
- No flashing / blinking elements.
- Every interactive element is reachable via keyboard; no mouse-only actions.

## Testing the TUI

Integration-testing ratatui apps is painful; current approach:

- Unit-test pure helpers (markdown parsing, diff parsing, cell layout) in `src/rust/xerxes-cli/tests/`.
- For end-to-end TUI behavior, run `xerxes` manually against a mock bridge that replays a canned event log. A JSONL replay tool lives at `src/rust/xerxes-cli/src/bridge.rs` (the `--replay` flag).
- For the Python side of the protocol, Python tests (`tests/test_runtime_bridge.py`) exercise the bridge without the TUI.

## Future TUI work

Known gaps (tracked informally):

- Image rendering in the conversation (screenshot cells currently show placeholder text).
- A "browse mode" for stepping through past turns without mutating history.
- Key-rebinding via `~/.xerxes/keybindings.toml`.

None are blockers for the current release.
