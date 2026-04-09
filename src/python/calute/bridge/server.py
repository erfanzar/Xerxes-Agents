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

"""JSON-RPC bridge server for the Rust TUI frontend.

Protocol
--------
Communication is over **stdin/stdout**, one JSON object per line (newline-delimited JSON).

**Requests** (Rust -> Python)::

    {"method": "init",  "params": {"model": "gpt-4o", "permission_mode": "auto"}}
    {"method": "query", "params": {"text": "hello"}}
    {"method": "permission_response", "params": {"granted": true}}
    {"method": "slash",  "params": {"command": "/help"}}
    {"method": "cancel"}

**Events** (Python -> Rust)::

    {"event": "ready",              "data": {...}}
    {"event": "text_chunk",         "data": {"text": "Hello..."}}
    {"event": "thinking_chunk",     "data": {"text": "..."}}
    {"event": "tool_start",         "data": {"name": "Read", "inputs": {...}}}
    {"event": "tool_end",           "data": {"name": "Read", "result": "...", ...}}
    {"event": "permission_request", "data": {"tool_name": "Bash", "description": "..."}}
    {"event": "turn_done",          "data": {"input_tokens": 500, "output_tokens": 200}}
    {"event": "query_done",         "data": {}}
    {"event": "slash_result",       "data": {"output": "..."}}
    {"event": "error",              "data": {"message": "..."}}
    {"event": "state",              "data": {...}}

Usage::

    python -m calute.bridge.server [--model MODEL] [--base-url URL]
"""

from __future__ import annotations

import json
import os
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..llms.registry import calc_cost, detect_provider
from ..runtime.bootstrap import bootstrap
from ..runtime.config_context import set_config as set_global_config
from ..runtime.bridge import build_tool_executor, populate_registry
from ..runtime.cost_tracker import CostTracker
from . import profiles
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


class BridgeServer:
    """Bidirectional JSON-RPC bridge between a Rust TUI and the Python agent runtime."""

    SESSIONS_DIR = Path.home() / ".calute" / "sessions"

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.state = AgentState()
        self.cost_tracker = CostTracker()
        self.system_prompt = ""
        self.tool_executor = None
        self.tool_schemas: list[dict[str, Any]] = []
        self._initialized = False
        self._pending_permission: PermissionRequest | None = None
        self._permission_event = threading.Event()
        self._cancel = False
        self._out_lock = threading.Lock()
        self._stdout = sys.stdout
        self._suppressing_tag = False
        self._suppress_buf: list[str] = []
        self._session_id = str(uuid.uuid4())[:8]
        self._session_cwd = os.getcwd()
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


    def _emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Send a JSON event line to stdout (unbuffered)."""
        msg = {"event": event, "data": data or {}}
        line = json.dumps(msg, ensure_ascii=False, default=str)
        with self._out_lock:
            self._stdout.write(line + "\n")
            self._stdout.flush()

    def _emit_error(self, message: str) -> None:
        self._emit("error", {"message": message})

    def _emit_state(self) -> None:
        self._emit(
            "state",
            {
                "turn_count": self.state.turn_count,
                "total_input_tokens": self.state.total_input_tokens,
                "total_output_tokens": self.state.total_output_tokens,
                "message_count": len(self.state.messages),
                "tool_execution_count": len(self.state.tool_executions),
                "cost_usd": calc_cost(
                    self.config.get("model", ""),
                    self.state.total_input_tokens,
                    self.state.total_output_tokens,
                ),
            },
        )


    def _emit_text(self, text: str) -> None:
        """Emit text chunk, suppressing <function=...>...</function> markup."""
        if self._suppressing_tag:
            self._suppress_buf.append(text)
            joined = "".join(self._suppress_buf)
            if "</function>" in joined:
                after = joined.split("</function>", 1)[1]
                self._suppressing_tag = False
                self._suppress_buf.clear()
                if after.strip():
                    self._emit("text_chunk", {"text": after})
            return

        if "<function=" in text:
            before, _, rest = text.partition("<function=")
            if before.strip():
                self._emit("text_chunk", {"text": before})
            self._suppressing_tag = True
            self._suppress_buf.clear()
            self._suppress_buf.append("<function=" + rest)
            joined = "".join(self._suppress_buf)
            if "</function>" in joined:
                after = joined.split("</function>", 1)[1]
                self._suppressing_tag = False
                self._suppress_buf.clear()
                if after.strip():
                    self._emit("text_chunk", {"text": after})
            return

        stripped = text.strip()
        if stripped.startswith('{"name":') and '"arguments"' in stripped:
            return

        self._emit("text_chunk", {"text": text})

    def _save_session(self) -> None:
        """Save current state to a JSON file."""
        data = {
            "session_id": self._session_id,
            "model": self.config.get("model", ""),
            "cwd": self._session_cwd,
            "created_at": getattr(self, "_created_at", datetime.now(UTC).isoformat()),
            "updated_at": datetime.now(UTC).isoformat(),
            "messages": self.state.messages,
            "turn_count": self.state.turn_count,
            "total_input_tokens": self.state.total_input_tokens,
            "total_output_tokens": self.state.total_output_tokens,
            "thinking_content": self.state.thinking_content,
            "tool_executions": self.state.tool_executions,
        }
        path = self.SESSIONS_DIR / f"{self._session_id}.json"
        path.write_text(json.dumps(data, indent=2, default=str, ensure_ascii=False))

    def _load_session(self, session_id: str) -> bool:
        """Load a session from disk. Returns True on success."""
        path = self.SESSIONS_DIR / f"{session_id}.json"
        if not path.exists():
            return False
        try:
            data = json.loads(path.read_text())
            self._session_id = data["session_id"]
            self._created_at = data.get("created_at", "")
            self.state.messages = data.get("messages", [])
            self.state.turn_count = data.get("turn_count", 0)
            self.state.total_input_tokens = data.get("total_input_tokens", 0)
            self.state.total_output_tokens = data.get("total_output_tokens", 0)
            self.state.thinking_content = data.get("thinking_content", [])
            self.state.tool_executions = data.get("tool_executions", [])
            return True
        except (json.JSONDecodeError, KeyError):
            return False

    def _list_sessions(self) -> list[dict[str, Any]]:
        """List all saved sessions."""
        sessions = []
        for path in sorted(self.SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text())
                preview = ""
                for msg in data.get("messages", []):
                    if msg.get("role") == "user":
                        content = msg.get("content", "")
                        if isinstance(content, str):
                            preview = content[:60]
                            break
                sessions.append({
                    "session_id": data.get("session_id", path.stem),
                    "model": data.get("model", ""),
                    "cwd": data.get("cwd", ""),
                    "updated_at": data.get("updated_at", ""),
                    "turns": data.get("turn_count", 0),
                    "preview": preview,
                })
            except (json.JSONDecodeError, KeyError):
                continue
        return sessions


    def handle_init(self, params: dict[str, Any]) -> None:
        self.config = {
            "permission_mode": params.get("permission_mode", "auto"),
            "verbose": params.get("verbose", False),
            "thinking": params.get("thinking", True),
            "debug": params.get("debug", False),
        }

        model = params.get("model", "")
        base_url = params.get("base_url", "")
        api_key = params.get("api_key", "")

        if not model and not base_url:
            profile = profiles.get_active_profile()
            if profile:
                model = profile.get("model", "")
                base_url = profile.get("base_url", "")
                api_key = profile.get("api_key", "")
                for k, v in profile.get("sampling", {}).items():
                    self.config[k] = v

        has_profile = bool(model)
        self.config["model"] = model if model else ""
        if base_url:
            self.config["base_url"] = base_url
        if api_key:
            self.config["api_key"] = api_key

        boot = bootstrap(model=self.config["model"])
        self.system_prompt = boot.system_prompt

        registry = populate_registry()
        self.tool_executor = build_tool_executor(registry=registry)
        self.tool_schemas = registry.tool_schemas()

        self._initialized = True

        set_global_config(self.config)

        provider = detect_provider(model)
        self._emit(
            "ready",
            {
                "model": model,
                "provider": provider,
                "tools": len(self.tool_schemas),
                "permission_mode": self.config["permission_mode"],
                "has_profile": has_profile,
            },
        )


    def handle_query(self, params: dict[str, Any]) -> None:
        if not self._initialized:
            self._emit_error("Not initialized. Send 'init' first.")
            return

        text = params.get("text", "").strip()
        if not text:
            self._emit_error("Empty query.")
            return

        self._cancel = False
        self._suppressing_tag = False
        self._suppress_buf.clear()

        for event in run_agent_loop(
            user_message=text,
            state=self.state,
            config=self.config,
            system_prompt=self.system_prompt,
            tool_executor=self.tool_executor,
            tool_schemas=self.tool_schemas,
            cancel_check=lambda: self._cancel,
        ):
            if self._cancel:
                break

            if isinstance(event, TextChunk):
                self._emit_text(event.text)

            elif isinstance(event, ThinkingChunk):
                self._emit("thinking_chunk", {"text": event.text})

            elif isinstance(event, ToolStart):
                self._suppressing_tag = False
                self._suppress_buf.clear()
                self._emit(
                    "tool_start",
                    {
                        "name": event.name,
                        "inputs": event.inputs,
                        "tool_call_id": event.tool_call_id,
                    },
                )

            elif isinstance(event, ToolEnd):
                self._emit(
                    "tool_end",
                    {
                        "name": event.name,
                        "result": event.result,
                        "permitted": event.permitted,
                        "tool_call_id": event.tool_call_id,
                        "duration_ms": event.duration_ms,
                    },
                )

            elif isinstance(event, PermissionRequest):
                self._emit(
                    "permission_request",
                    {
                        "tool_name": event.tool_name,
                        "description": event.description,
                        "inputs": event.inputs,
                    },
                )
                self._pending_permission = event
                self._permission_event.clear()
                self._permission_event.wait()
                self._pending_permission = None

            elif isinstance(event, TurnDone):
                self.cost_tracker.record_turn(
                    self.config.get("model", ""),
                    event.input_tokens,
                    event.output_tokens,
                )
                self._emit(
                    "turn_done",
                    {
                        "input_tokens": event.input_tokens,
                        "output_tokens": event.output_tokens,
                        "tool_calls_count": event.tool_calls_count,
                        "model": event.model,
                    },
                )

        self._emit("query_done", {})
        self._emit_state()


    def handle_permission_response(self, params: dict[str, Any]) -> None:
        if self._pending_permission is not None:
            self._pending_permission.granted = params.get("granted", False)
            self._permission_event.set()
        else:
            self._emit_error("No pending permission request.")


    def handle_cancel(self) -> None:
        self._cancel = True
        if self._pending_permission is not None:
            self._pending_permission.granted = False
            self._permission_event.set()


    def handle_provider_list(self) -> None:
        """List saved provider profiles."""
        plist = profiles.list_profiles()
        self._emit("provider_list", {"profiles": plist})

    def handle_fetch_models(self, params: dict[str, Any]) -> None:
        """Fetch models from a provider's base URL."""
        base_url = params.get("base_url", "")
        api_key = params.get("api_key", "")
        if not base_url:
            self._emit_error("base_url is required for fetch_models")
            return
        models = profiles.fetch_models(base_url, api_key)
        self._emit("models_list", {"models": models, "base_url": base_url})

    def handle_provider_save(self, params: dict[str, Any]) -> None:
        """Save a provider profile and switch to it."""
        name = params.get("name", "")
        base_url = params.get("base_url", "")
        api_key = params.get("api_key", "")
        model = params.get("model", "")
        if not name or not base_url or not model:
            self._emit_error("name, base_url, and model are required")
            return

        provider = params.get("provider", "")
        profile = profiles.save_profile(
            name=name,
            base_url=base_url,
            api_key=api_key,
            model=model,
            provider=provider,
            set_active=True,
        )

        self.config["model"] = model
        self.config["base_url"] = base_url
        if api_key:
            self.config["api_key"] = api_key
        set_global_config(self.config)

        self._emit("provider_saved", {
            "profile": profile,
            "message": f"Profile '{name}' saved and activated. Model: {model}",
        })

    def handle_provider_select(self, params: dict[str, Any]) -> None:
        """Switch to an existing saved profile."""
        name = params.get("name", "")
        if not name:
            self._emit_error("Profile name is required")
            return
        if not profiles.set_active(name):
            self._emit_error(f"Profile '{name}' not found")
            return
        profile = profiles.get_active_profile()
        if profile:
            self.config["model"] = profile["model"]
            self.config["base_url"] = profile.get("base_url", "")
            if profile.get("api_key"):
                self.config["api_key"] = profile["api_key"]
            for k, v in profile.get("sampling", {}).items():
                self.config[k] = v
            set_global_config(self.config)
            self._emit("provider_saved", {
                "profile": profile,
                "message": f"Switched to profile '{name}'. Model: {profile['model']}",
            })

    def handle_provider_delete(self, params: dict[str, Any]) -> None:
        """Delete a provider profile."""
        name = params.get("name", "")
        if profiles.delete_profile(name):
            self._emit("slash_result", {"output": f"Profile '{name}' deleted."})
        else:
            self._emit_error(f"Profile '{name}' not found")


    def handle_slash(self, params: dict[str, Any]) -> None:
        command = params.get("command", "").strip()
        if not command.startswith("/"):
            self._emit_error(f"Not a slash command: {command}")
            return

        parts = command[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        output = self._run_slash(cmd, args)
        self._emit("slash_result", {"output": output})

    def _handle_sampling(self, args: str) -> str:
        """Handle /sampling — view, set, or reset sampling params.

        Usage:
            /sampling                  — show current params
            /sampling temperature 0.7  — set a param
            /sampling reset            — reset all to defaults
            /sampling save             — save current params to active profile
        """
        valid = profiles.SAMPLING_PARAMS

        if not args.strip():
            lines = ["Sampling parameters (current session):"]
            for k in sorted(valid):
                val = self.config.get(k, None)
                if val is not None:
                    lines.append(f"  {k}: {val}")
                else:
                    lines.append(f"  {k}: (default)")
            lines.append("")
            lines.append("Usage: /sampling <param> <value>")
            lines.append("       /sampling reset")
            lines.append("       /sampling save  (persist to active profile)")
            return "\n".join(lines)

        parts = args.strip().split(None, 1)
        subcmd = parts[0].lower()

        if subcmd == "reset":
            for k in valid:
                self.config.pop(k, None)
            return "Sampling parameters reset to defaults."

        if subcmd == "save":
            profile = profiles.get_active_profile()
            if not profile:
                return "No active profile. Run /provider first."
            sampling = {}
            for k in valid:
                if k in self.config:
                    sampling[k] = self.config[k]
            profiles.update_sampling(profile["name"], sampling)
            return f"Sampling parameters saved to profile '{profile['name']}'."

        if len(parts) != 2:
            return f"Usage: /sampling <param> <value>\nValid params: {', '.join(sorted(valid))}"

        param = subcmd
        val_str = parts[1]

        if param not in valid:
            return f"Unknown param: {param}\nValid: {', '.join(sorted(valid))}"

        try:
            if param in ("max_tokens", "top_k"):
                val: int | float = int(val_str)
            else:
                val = float(val_str)
        except ValueError:
            return f"Invalid value: {val_str}"

        self.config[param] = val
        return f"{param} = {val}"

    def _handle_compact(self) -> str:
        """Compact conversation history using the LLM to summarize."""
        messages = self.state.messages
        if len(messages) < 4:
            return "Nothing to compact (fewer than 4 messages)."

        model = self.config.get("model", "")
        if not model:
            return "No model configured. Run /provider first."

        system_msgs = [m for m in messages if m.get("role") == "system"]
        conv_msgs = [m for m in messages if m.get("role") != "system"]

        if len(conv_msgs) < 3:
            return "Nothing to compact."

        preserve_recent = 2
        older = conv_msgs[:-preserve_recent]
        recent = conv_msgs[-preserve_recent:]

        conv_text = []
        for msg in older:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, dict):
                        parts.append(p.get("text", str(p)))
                    else:
                        parts.append(str(p))
                content = "\n".join(parts)
            if len(content) > 500:
                content = content[:500] + "..."
            conv_text.append(f"[{role}]: {content}")

        conversation = "\n\n".join(conv_text)

        try:
            from openai import OpenAI
            from ..llms.registry import PROVIDERS, get_api_key

            provider_name = detect_provider(model)
            api_key = self.config.get("api_key") or get_api_key(provider_name, self.config)
            prov = PROVIDERS.get(provider_name, PROVIDERS.get("openai"))
            base_url = (
                self.config.get("base_url")
                or self.config.get("custom_base_url")
                or (prov.base_url if prov else None)
                or "https://api.openai.com/v1"
            )
            client = OpenAI(api_key=api_key or "dummy", base_url=base_url)

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a conversation summarizer. Summarize the following conversation "
                            "into a concise summary that preserves all key information: decisions made, "
                            "files discussed, code changes, tool results, and any important context. "
                            "Be factual and specific. Output only the summary, no preamble."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this conversation ({len(older)} messages):\n\n{conversation}",
                    },
                ],
                max_tokens=1024,
                temperature=0.2,
            )

            summary = response.choices[0].message.content or ""
            if not summary.strip():
                return "Compaction failed: LLM returned empty summary."

        except Exception as exc:
            return f"Compaction failed: {exc}"

        original_count = len(self.state.messages)
        self.state.messages = (
            system_msgs
            + [{"role": "user", "content": f"[Previous conversation summary — {len(older)} messages compacted]\n\n{summary}"}]
            + recent
        )
        new_count = len(self.state.messages)

        return (
            f"Compacted {original_count} messages → {new_count} messages.\n"
            f"Summarized {len(older)} older messages, kept {len(recent)} recent + {len(system_msgs)} system."
        )

    def _run_slash(self, cmd: str, args: str) -> str:
        if cmd in ("help", "h"):
            return (
                "Commands:\n"
                "  /provider          Setup or switch provider profile\n"
                "  /sampling          View or set sampling parameters\n"
                "  /compact           Summarize conversation to free context\n"
                "  /model NAME        Switch model\n"
                "  /cost              Show cost summary\n"
                "  /context           Show context info\n"
                "  /clear             Clear conversation\n"
                "  /tools             List available tools\n"
                "  /thinking          Toggle thinking display\n"
                "  /verbose           Toggle verbose mode\n"
                "  /debug             Toggle debug mode\n"
                "  /permissions       Cycle permission mode\n"
                "  /config            Show config\n"
                "  /history           Show message count\n"
                "  /exit              Exit"
            )

        if cmd == "model":
            if args:
                self.config["model"] = args
                return f"Model set to: {args}"
            return f"Current model: {self.config.get('model', '(none)')}"

        if cmd == "cost":
            return self.cost_tracker.summary()

        if cmd == "history":
            return f"{len(self.state.messages)} messages, {self.state.turn_count} turns"

        if cmd == "verbose":
            self.config["verbose"] = not self.config.get("verbose", False)
            return f"Verbose: {self.config['verbose']}"

        if cmd == "thinking":
            self.config["thinking"] = not self.config.get("thinking", False)
            return f"Thinking: {self.config['thinking']}"

        if cmd == "sampling":
            return self._handle_sampling(args)

        if cmd == "compact":
            return self._handle_compact()

        if cmd == "debug":
            self.config["debug"] = not self.config.get("debug", False)
            return f"Debug: {self.config['debug']}"

        if cmd == "clear":
            self.state.messages.clear()
            self.state.thinking_content.clear()
            self.state.tool_executions.clear()
            self.state.turn_count = 0
            return "Conversation cleared."

        if cmd == "context":
            model = self.config.get("model", "")
            provider = detect_provider(model)
            cost = calc_cost(model, self.state.total_input_tokens, self.state.total_output_tokens)
            return (
                f"CWD: {os.getcwd()}\n"
                f"Model: {model}\n"
                f"Provider: {provider}\n"
                f"Turns: {self.state.turn_count}\n"
                f"Messages: {len(self.state.messages)}\n"
                f"Tokens: {self.state.total_input_tokens} in / {self.state.total_output_tokens} out\n"
                f"Cost: ${cost:.4f}"
            )

        if cmd == "config":
            lines = [f"  {k}: {v}" for k, v in sorted(self.config.items()) if not k.startswith("_")]
            return "\n".join(lines) if lines else "(empty config)"

        if cmd == "permissions":
            modes = ["auto", "accept-all", "manual"]
            current = self.config.get("permission_mode", "auto")
            idx = modes.index(current) if current in modes else 0
            new_mode = modes[(idx + 1) % len(modes)]
            self.config["permission_mode"] = new_mode
            return f"Permission mode: {new_mode}"

        if cmd == "tools":
            registry = populate_registry()
            lines = []
            for entry in registry.list_tools():
                safe = " [safe]" if entry.safe else ""
                lines.append(f"  {entry.name}{safe} -- {entry.description[:60]}")
            lines.append(f"  ({registry.tool_count} total)")
            return "\n".join(lines)

        if cmd in ("exit", "quit", "q"):
            self._emit("exit", {})
            sys.exit(0)

        return f"Unknown command: /{cmd} (type /help)"


    def run(self) -> None:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue

            try:
                msg = json.loads(line)
            except json.JSONDecodeError as exc:
                self._emit_error(f"Invalid JSON: {exc}")
                continue

            method = msg.get("method", "")
            params = msg.get("params", {})

            try:
                if method == "init":
                    self.handle_init(params)
                elif method == "query":
                    self.handle_query(params)
                elif method == "permission_response":
                    self.handle_permission_response(params)
                elif method == "cancel":
                    self.handle_cancel()
                elif method == "slash":
                    self.handle_slash(params)
                elif method == "provider_list":
                    self.handle_provider_list()
                elif method == "fetch_models":
                    self.handle_fetch_models(params)
                elif method == "provider_save":
                    self.handle_provider_save(params)
                elif method == "provider_select":
                    self.handle_provider_select(params)
                elif method == "provider_delete":
                    self.handle_provider_delete(params)
                else:
                    self._emit_error(f"Unknown method: {method}")
            except Exception as exc:
                self._emit_error(f"{type(exc).__name__}: {exc}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Calute bridge server (JSON-RPC over stdio)")
    parser.add_argument("-m", "--model", default=None, help="Model name")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    args = parser.parse_args()

    server = BridgeServer()

    if args.model:
        init_params: dict[str, Any] = {"model": args.model}
        if args.base_url:
            init_params["base_url"] = args.base_url
        if args.api_key:
            init_params["api_key"] = args.api_key
        server.handle_init(init_params)

    server.run()


if __name__ == "__main__":
    main()
