# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Lightweight interactive REPL for Calute.

Inspired by the nano-claude-code REPL, this provides a minimal terminal
interface with:

- Readline history and tab completion for slash commands
- Streaming text output with optional Rich markdown re-rendering
- Tool call visualization with diff rendering
- Thinking/reasoning display in verbose mode
- Interactive permission prompts
- Slash commands for model switching, cost tracking, sessions, etc.
- Session save/load
- Background agent notifications

Usage::

    from calute.tui.repl import repl

    repl(config={"model": "gpt-4o", "permission_mode": "auto"})

Or from CLI::

    python -m calute.tui.repl --model gpt-4o
"""

from __future__ import annotations

import json
import os
import readline
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from ..llms.registry import calc_cost, detect_provider
from ..runtime.bridge import build_tool_executor, populate_registry
from ..runtime.cost_tracker import CostTracker
from ..streaming.events import (
    AgentState,
    PermissionRequest,
    TextChunk,
    ThinkingChunk,
    ToolEnd,
    ToolStart,
    TurnDone,
)
from ..streaming.loop import run as run_agent_loop

# ── ANSI color helpers ─────────────────────────────────────────────────────

_STYLES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}
_RESET = "\033[0m"


def clr(text: str, *styles: str) -> str:
    """Apply ANSI styles to text."""
    codes = "".join(_STYLES.get(s, "") for s in styles)
    return f"{codes}{text}{_RESET}" if codes else text


# ── Rich integration (optional) ───────────────────────────────────────────

try:
    from rich.console import Console

    _console = Console()
    _HAS_RICH = True
except ImportError:
    _console = None
    _HAS_RICH = False


# ── Readline setup ─────────────────────────────────────────────────────────

HISTORY_DIR = Path.home() / ".calute"
HISTORY_FILE = HISTORY_DIR / "repl_history"
SESSIONS_DIR = HISTORY_DIR / "sessions"

SLASH_COMMANDS = [
    "/help",
    "/model",
    "/set",
    "/sampling",
    "/reset-sampling",
    "/cost",
    "/history",
    "/context",
    "/save",
    "/load",
    "/clear",
    "/verbose",
    "/debug",
    "/thinking",
    "/config",
    "/permissions",
    "/agents",
    "/memory",
    "/skills",
    "/tools",
    "/registry",
    "/session",
    "/audit",
    "/exit",
    "/quit",
]


def _setup_readline() -> None:
    """Setup readline with history and tab completion."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(str(HISTORY_FILE))
    except (FileNotFoundError, PermissionError, OSError):
        pass

    def completer(text: str, state: int) -> str | None:
        options = [c for c in SLASH_COMMANDS if c.startswith(text)]
        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")


def _save_readline() -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(HISTORY_FILE))
    except (PermissionError, OSError):
        pass


# ── Streaming display ──────────────────────────────────────────────────────

_text_buffer: list[str] = []
_suppressing_tool_xml = False
_suppress_buffer: list[str] = []


def _stream_text(chunk: str) -> None:
    """Print a text chunk inline, suppressing <tool_call> XML blocks."""
    global _suppressing_tool_xml

    _text_buffer.append(chunk)

    if _suppressing_tool_xml:
        # We're inside a <tool_call> block — buffer silently
        _suppress_buffer.append(chunk)
        # Check if closing tag arrived
        joined = "".join(_suppress_buffer)
        if "</tool_call>" in joined:
            # Emit any text AFTER the closing tag
            after = joined.split("</tool_call>", 1)[1]
            _suppressing_tool_xml = False
            _suppress_buffer.clear()
            if after.strip():
                print(after, end="", flush=True)
        return

    # Check if this chunk starts or contains <tool_call>
    if "<tool_call" in chunk:
        # Split at the tag — print before, suppress from tag onwards
        before, _, rest = chunk.partition("<tool_call")
        if before:
            print(before, end="", flush=True)
        _suppressing_tool_xml = True
        _suppress_buffer.clear()
        _suppress_buffer.append("<tool_call" + rest)
        # Check if it also closes in same chunk
        joined = "".join(_suppress_buffer)
        if "</tool_call>" in joined:
            after = joined.split("</tool_call>", 1)[1]
            _suppressing_tool_xml = False
            _suppress_buffer.clear()
            if after.strip():
                print(after, end="", flush=True)
        return

    print(chunk, end="", flush=True)


def _flush_response() -> None:
    """Flush accumulated text buffer."""
    global _suppressing_tool_xml
    _text_buffer.clear()
    _suppress_buffer.clear()
    _suppressing_tool_xml = False
    print()


def _print_tool_start(name: str, inputs: dict, verbose: bool, debug: bool = False) -> None:
    """Display tool invocation start."""
    desc = _tool_desc(name, inputs)
    print(clr(f"\n  ⚙  {desc}", "dim", "cyan"), flush=True)
    if debug:
        print(clr(f"     [DEBUG] tool: {name}", "yellow"))
        print(clr(f"     [DEBUG] inputs: {json.dumps(inputs, ensure_ascii=False, indent=2)}", "yellow"))
    elif verbose:
        preview = json.dumps(inputs, ensure_ascii=False)[:200]
        print(clr(f"     inputs: {preview}", "dim"))


def _print_tool_end(
    name: str, result: str, permitted: bool, duration_ms: float, verbose: bool, debug: bool = False
) -> None:
    """Display tool invocation result."""
    if not permitted:
        print(clr("  ✗ Denied", "dim", "red"), flush=True)
        return

    lines = result.count("\n") + 1
    size = len(result)
    summary = f"→ {lines} lines ({size} chars, {duration_ms:.0f}ms)"

    if result.startswith("Error"):
        print(clr(f"  ✗ {result[:200]}", "dim", "red"), flush=True)
    else:
        print(clr(f"  ✓ {summary}", "dim", "green"), flush=True)
        # Render diffs for Edit/Write
        if name in ("FileEditTool", "Edit", "Write", "WriteFile") and "---" in result and "+++" in result:
            _render_diff(result)

    if debug:
        preview = result[:1000].replace("\n", "\n     ")
        print(clr(f"     [DEBUG] result: {preview}", "yellow"))
    elif verbose and not result.startswith("Denied"):
        preview = result[:500].replace("\n", "\n     ")
        print(clr(f"     {preview}", "dim"))

    # Print prefix for next text
    print(clr("│ ", "dim"), end="", flush=True)


def _render_diff(text: str) -> None:
    """Render a unified diff with colors."""
    for line in text.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            print(clr(f"     {line}", "green"))
        elif line.startswith("-") and not line.startswith("---"):
            print(clr(f"     {line}", "red"))
        elif line.startswith("@@"):
            print(clr(f"     {line}", "cyan"))
        elif line.startswith("---") or line.startswith("+++"):
            print(clr(f"     {line}", "bold"))


def _tool_desc(name: str, inputs: dict) -> str:
    """Short description of a tool call."""
    if name in ("Read", "ReadFile") and "file_path" in inputs:
        return f"Read {inputs['file_path']}"
    if name in ("Write", "WriteFile") and "file_path" in inputs:
        return f"Write {inputs['file_path']}"
    if name in ("Edit", "FileEditTool") and "file_path" in inputs:
        return f"Edit {inputs['file_path']}"
    if name in ("Bash", "ExecuteShell") and "command" in inputs:
        return f"$ {inputs['command'][:60]}"
    if name in ("Glob", "GlobTool") and "pattern" in inputs:
        return f"Glob {inputs['pattern']}"
    if name in ("Grep", "GrepTool") and "pattern" in inputs:
        return f"Grep /{inputs['pattern']}/"
    first_val = next(iter(inputs.values()), "") if inputs else ""
    return f"{name}({str(first_val)[:40]})"


def _ask_permission(description: str) -> bool:
    """Interactive permission prompt."""
    print(clr("\n  🔒 Permission required: ", "yellow", "bold") + description)
    try:
        answer = input(clr("  Allow? [y/N] ", "yellow")).strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ── Slash command handler ──────────────────────────────────────────────────


def _handle_slash(
    line: str,
    state: AgentState,
    config: dict[str, Any],
    cost_tracker: CostTracker,
) -> bool:
    """Handle slash commands. Returns True if handled."""
    if not line.startswith("/"):
        return False
    parts = line[1:].split(None, 1)
    if not parts:
        return False
    cmd = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    if cmd in ("help", "h"):
        print(clr("Commands:", "bold"))
        print("  /help              — Show this help")
        print("  /model NAME        — Switch model")
        print("  /set PARAM VALUE   — Set sampling param (e.g. /set temperature 0.7)")
        print("  /sampling          — Show current sampling params")
        print("  /reset-sampling    — Reset sampling to defaults")
        print("  /cost              — Show cost summary")
        print("  /history           — Show message history")
        print("  /verbose           — Toggle verbose mode")
        print("  /debug             — Toggle debug mode (shows full tool I/O, request info)")
        print("  /thinking          — Toggle thinking display")
        print("  /save [name]       — Save session")
        print("  /load [name]       — Load session")
        print("  /clear             — Clear conversation")
        print("  /tools             — List available tools")
        print("  /registry          — Show execution registry")
        print("  /context           — Show context info")
        print("  /config            — Show config")
        print("  /permissions       — Toggle permission mode")
        print("  /exit              — Exit")
        return True

    if cmd == "model":
        if args:
            config["model"] = args
            print(clr(f"  Model set to: {args}", "green"))
        else:
            print(clr(f"  Current model: {config.get('model', '(none)')}", "cyan"))
        return True

    if cmd == "cost":
        print(cost_tracker.summary())
        return True

    if cmd == "history":
        if not state.messages:
            print(clr("  (empty conversation)", "dim"))
        else:
            for i, m in enumerate(state.messages):
                role = m["role"].upper()
                content = m.get("content", "")
                if isinstance(content, str):
                    print(f"  [{i}] {clr(role, 'bold')}: {content[:150]}")
                elif isinstance(content, list):
                    print(f"  [{i}] {clr(role, 'bold')}: [{len(content)} blocks]")
                thinking = m.get("thinking", "")
                if thinking:
                    print(clr(f"       [thinking: {len(thinking)} chars]", "dim"))
        return True

    if cmd == "verbose":
        config["verbose"] = not config.get("verbose", False)
        print(clr(f"  Verbose: {config['verbose']}", "yellow"))
        return True

    if cmd == "thinking":
        config["thinking"] = not config.get("thinking", False)
        print(clr(f"  Thinking: {config['thinking']}", "yellow"))
        return True

    if cmd == "debug":
        config["debug"] = not config.get("debug", False)
        print(clr(f"  Debug: {config['debug']}", "yellow"))
        if config["debug"]:
            print(clr("  Debug shows: provider, base_url, message count, sampling, tool inputs/outputs", "dim"))
        return True

    if cmd == "set":
        if not args.strip():
            print(clr("  Sampling params:", "bold"))
            for k in (
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
                "repetition_penalty",
            ):
                val = config.get(k, "(default)")
                print(f"    {k}: {val}")
            print(clr("\n  Usage: /set temperature 0.7", "dim"))
            return True
        parts_set = args.strip().split(None, 1)
        if len(parts_set) != 2:
            print(clr("  Usage: /set <param> <value>", "red"))
            return True
        param, val_str = parts_set
        valid_params = {
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "thinking_budget",
        }
        if param not in valid_params:
            print(clr(f"  Unknown param: {param}. Valid: {', '.join(sorted(valid_params))}", "red"))
            return True
        try:
            val: int | float = int(val_str) if param in ("max_tokens", "top_k", "thinking_budget") else float(val_str)
        except ValueError:
            print(clr(f"  Invalid value: {val_str}", "red"))
            return True
        config[param] = val
        print(clr(f"  {param} = {val}", "green"))
        return True

    if cmd == "sampling":
        print(clr("  Sampling params:", "bold"))
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "thinking_budget",
        ):
            val = config.get(k, "(default)")
            print(f"    {k}: {val}")
        return True

    if cmd == "reset-sampling":
        for k in (
            "temperature",
            "top_p",
            "top_k",
            "min_p",
            "max_tokens",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
        ):
            config.pop(k, None)
        print(clr("  Sampling params reset to defaults.", "yellow"))
        return True

    if cmd == "clear":
        state.messages.clear()
        state.thinking_content.clear()
        state.tool_executions.clear()
        state.turn_count = 0
        print(clr("  Conversation cleared.", "yellow"))
        return True

    if cmd in ("save",):
        fname = args.strip() or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = SESSIONS_DIR / fname if "/" not in fname else Path(fname)
        data = {
            "messages": state.messages,
            "turn_count": state.turn_count,
            "total_input_tokens": state.total_input_tokens,
            "total_output_tokens": state.total_output_tokens,
            "thinking_content": state.thinking_content,
            "tool_executions": state.tool_executions,
            "model": config.get("model", ""),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str))
        print(clr(f"  Session saved to {path}", "green"))
        return True

    if cmd in ("load",):
        if not args.strip():
            sessions = sorted(SESSIONS_DIR.glob("*.json"))
            if sessions:
                print(clr("  Saved sessions:", "cyan"))
                for s in sessions:
                    print(f"    {s.name}")
            else:
                print(clr("  No saved sessions.", "dim"))
            return True
        fname = args.strip()
        path = SESSIONS_DIR / fname if "/" not in fname else Path(fname)
        if not path.exists():
            print(clr(f"  File not found: {path}", "red"))
            return True
        data = json.loads(path.read_text())
        state.messages = data.get("messages", [])
        state.turn_count = data.get("turn_count", 0)
        state.total_input_tokens = data.get("total_input_tokens", 0)
        state.total_output_tokens = data.get("total_output_tokens", 0)
        state.thinking_content = data.get("thinking_content", [])
        state.tool_executions = data.get("tool_executions", [])
        if data.get("model"):
            config["model"] = data["model"]
        print(clr(f"  Session loaded: {len(state.messages)} messages", "green"))
        return True

    if cmd == "tools":
        registry = populate_registry()
        for entry in registry.list_tools()[:30]:
            safe = clr(" [safe]", "green") if entry.safe else ""
            print(f"  {entry.name}{safe} — {entry.description[:60]}")
        print(clr(f"  ({registry.tool_count} total)", "dim"))
        return True

    if cmd == "registry":
        registry = populate_registry()
        print(registry.summary()[:2000])
        return True

    if cmd == "context":
        print(f"  CWD: {os.getcwd()}")
        print(f"  Model: {config.get('model', '(none)')}")
        print(f"  Provider: {detect_provider(config.get('model', ''))}")
        print(f"  Turns: {state.turn_count}")
        print(f"  Messages: {len(state.messages)}")
        print(f"  Tokens: {state.total_input_tokens} in / {state.total_output_tokens} out")
        print(f"  Cost: ${calc_cost(config.get('model', ''), state.total_input_tokens, state.total_output_tokens):.4f}")
        print(f"  Thinking entries: {len(state.thinking_content)}")
        print(f"  Tool executions: {len(state.tool_executions)}")
        return True

    if cmd == "config":
        for k, v in sorted(config.items()):
            if not k.startswith("_"):
                print(f"  {k}: {v}")
        return True

    if cmd == "permissions":
        modes = ["auto", "accept-all", "manual"]
        current = config.get("permission_mode", "auto")
        idx = modes.index(current) if current in modes else 0
        new_mode = modes[(idx + 1) % len(modes)]
        config["permission_mode"] = new_mode
        print(clr(f"  Permission mode: {new_mode}", "yellow"))
        return True

    if cmd in ("exit", "quit", "q"):
        _save_readline()
        print(clr("Goodbye!", "green"))
        sys.exit(0)

    print(clr(f"  Unknown command: /{cmd} (type /help)", "red"))
    return True


# ── Main REPL ──────────────────────────────────────────────────────────────


def repl(
    config: dict[str, Any] | None = None,
    initial_prompt: str | None = None,
    system_prompt: str = "",
) -> None:
    """Run the interactive REPL.

    Args:
        config: Configuration dict with ``model``, ``permission_mode``, etc.
        initial_prompt: Optional single prompt to run (non-interactive mode).
        system_prompt: System prompt for the LLM.
    """
    config = config or {}
    config.setdefault("model", "gpt-4o")
    config.setdefault("permission_mode", "auto")
    config.setdefault("verbose", False)
    config.setdefault("thinking", True)
    config.setdefault("debug", False)

    _setup_readline()

    state = AgentState()
    cost_tracker = CostTracker()
    registry = populate_registry()
    tool_executor = build_tool_executor(registry=registry)
    tool_schemas = registry.tool_schemas()

    if not system_prompt:
        from ..runtime.bootstrap import bootstrap

        boot = bootstrap(model=config["model"])
        system_prompt = boot.system_prompt

    verbose = config.get("verbose", False)

    def run_query(user_input: str) -> None:
        nonlocal verbose
        verbose = config.get("verbose", False)
        debug = config.get("debug", False)

        model = config.get("model", "")
        provider = detect_provider(model)
        print(
            clr("\n╭─ Calute ", "dim")
            + clr("●", "green")
            + clr(f" {model} ", "dim", "cyan")
            + clr("─────────────────────────", "dim")
        )

        if debug:
            print(clr(f"  [DEBUG] provider={provider} model={model}", "yellow"))
            print(clr(f"  [DEBUG] base_url={config.get('base_url', '(auto)')}", "yellow"))
            print(clr(f"  [DEBUG] messages={len(state.messages)} turns={state.turn_count}", "yellow"))
            sampling = {
                k: config[k]
                for k in (
                    "temperature",
                    "top_p",
                    "top_k",
                    "min_p",
                    "max_tokens",
                    "frequency_penalty",
                    "presence_penalty",
                )
                if k in config
            }
            if sampling:
                print(clr(f"  [DEBUG] sampling={sampling}", "yellow"))
            print(clr(f"  [DEBUG] tools={len(tool_schemas)} schemas", "yellow"))

        print(clr("│ ", "dim"), end="", flush=True)

        thinking_started = False

        for event in run_agent_loop(
            user_message=user_input,
            state=state,
            config=config,
            system_prompt=system_prompt,
            tool_executor=tool_executor,
            tool_schemas=tool_schemas,
        ):
            if isinstance(event, TextChunk):
                _stream_text(event.text)

            elif isinstance(event, ThinkingChunk):
                if not thinking_started:
                    print(clr("\n  [thinking]", "dim", "italic"))
                    thinking_started = True
                print(clr(event.text, "dim"), end="", flush=True)

            elif isinstance(event, ToolStart):
                _flush_response()
                _print_tool_start(event.name, event.inputs, verbose, debug)

            elif isinstance(event, PermissionRequest):
                event.granted = _ask_permission(event.description)

            elif isinstance(event, ToolEnd):
                _print_tool_end(event.name, event.result, event.permitted, event.duration_ms, verbose, debug)

            elif isinstance(event, TurnDone):
                cost_tracker.record_turn(
                    config.get("model", ""),
                    event.input_tokens,
                    event.output_tokens,
                )
                if verbose:
                    print(
                        clr(
                            f"\n  [tokens: +{event.input_tokens} in / +{event.output_tokens} out]",
                            "dim",
                        )
                    )

        _flush_response()
        print(clr("╰──────────────────────────────────────────────", "dim"))
        print()

    # ── Non-interactive mode ──
    if initial_prompt:
        try:
            run_query(initial_prompt)
        except KeyboardInterrupt:
            print()
        _save_readline()
        return

    # ── Banner ──
    model = config["model"]
    from ..llms.registry import PROVIDERS

    provider = detect_provider(model)
    if provider not in PROVIDERS:
        provider = "custom"
    if config.get("base_url"):
        provider = "custom"
    pmode = config.get("permission_mode", "auto")

    line_model = f"  Model: {model} ({provider})"
    line_perms = f"  Permissions: {pmode}"
    line_help = "  /model to switch · /help for commands"
    width = max(len(line_model), len(line_perms), len(line_help), 40) + 4

    print(clr(f"╭─ Calute REPL {'─' * (width - 15)}╮", "dim"))
    print(
        clr("│", "dim")
        + clr("  Model: ", "dim")
        + clr(model, "cyan", "bold")
        + clr(f" ({provider})", "dim")
        + clr(f"{' ' * (width - len(line_model) - 1)}│", "dim")
    )
    print(
        clr("│", "dim")
        + clr("  Permissions: ", "dim")
        + clr(pmode, "yellow")
        + clr(f"{' ' * (width - len(line_perms) - 1)}│", "dim")
    )
    print(clr(f"│{line_help}{' ' * (width - len(line_help) - 1)}│", "dim"))
    print(clr(f"╰{'─' * (width - 1)}╯", "dim"))
    print()

    # ── Main loop ──
    while True:
        try:
            cwd_short = Path.cwd().name
            prompt = clr(f"\n[{cwd_short}] ", "dim") + clr("> ", "cyan", "bold")
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print(clr("Goodbye!", "green"))
            _save_readline()
            sys.exit(0)

        if not user_input:
            continue

        if _handle_slash(user_input, state, config, cost_tracker):
            continue

        try:
            run_query(user_input)
        except KeyboardInterrupt:
            print(clr("\n  (interrupted)", "yellow"))


# ── CLI entry point ────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for the REPL."""
    import argparse

    parser = argparse.ArgumentParser(description="Calute REPL — lightweight terminal agent")
    parser.add_argument("-m", "--model", default=None, help="Model name (e.g. gpt-4o, claude-sonnet-4-6)")
    parser.add_argument("--base-url", default=None, help="API base URL (e.g. http://localhost:11434/v1)")
    parser.add_argument("--api-key", default=None, help="API key for the provider")
    parser.add_argument("--accept-all", action="store_true", help="Auto-approve all tool calls")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--debug", action="store_true", help="Debug mode — show full tool I/O and request info")
    parser.add_argument("--thinking", action="store_true", help="Enable extended thinking")
    parser.add_argument("-p", "--print", dest="prompt", help="Non-interactive: run a single prompt and exit")
    args = parser.parse_args()

    config: dict[str, Any] = {}
    if args.model:
        config["model"] = args.model
    else:
        # Auto-detect from env
        for env_key, model in [
            ("ANTHROPIC_API_KEY", "claude-sonnet-4-6"),
            ("OPENAI_API_KEY", "gpt-4o"),
            ("GEMINI_API_KEY", "gemini-2.0-flash"),
            ("DEEPSEEK_API_KEY", "deepseek-chat"),
        ]:
            if os.environ.get(env_key):
                config["model"] = model
                break
        config.setdefault("model", "gpt-4o")

    if args.base_url:
        config["base_url"] = args.base_url
    if args.api_key:
        config["api_key"] = args.api_key
    if args.accept_all:
        config["permission_mode"] = "accept-all"
    if args.verbose:
        config["verbose"] = True
    if args.debug:
        config["debug"] = True
    if args.thinking:
        config["thinking"] = True

    repl(config=config, initial_prompt=args.prompt)


if __name__ == "__main__":
    main()
