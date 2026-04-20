# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""JSON-RPC bridge server for the TypeScript/Ink CLI frontend.

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

    python -m xerxes.bridge.server [--model MODEL] [--base-url URL]
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.paths import xerxes_subdir
from ..extensions.skill_authoring.pipeline import SkillAuthoringPipeline
from ..extensions.skills import SkillRegistry
from ..context.compaction_strategies import CompactionStrategy, get_compaction_strategy
from ..llms.registry import calc_cost, detect_provider, get_context_limit
from ..runtime.bootstrap import bootstrap
from ..runtime.bridge import build_tool_executor, populate_registry
from ..runtime.config_context import set_config as set_global_config
from ..runtime.config_context import set_event_callback
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
from ..tools.agent_meta_tools import set_skill_registry
from ..tools.claude_tools import set_ask_user_question_callback
from . import profiles


class BridgeServer:
    """Bidirectional JSON-RPC bridge between a Rust TUI and the Python agent runtime."""

    SESSIONS_DIR = xerxes_subdir("sessions")

    def __init__(self) -> None:
        self.config: dict[str, Any] = {}
        self.state = AgentState()
        self.cost_tracker = CostTracker()
        self.system_prompt = ""
        self.tool_executor = None
        self.tool_schemas: list[dict[str, Any]] = []
        self._initialized = False
        self._running = True
        self._cancel = False
        self._out_lock = threading.Lock()
        self._stdout = sys.stdout
        self._suppressing_tag = False
        self._suppress_buf: list[str] = []
        self._session_id = str(uuid.uuid4())[:8]
        self._session_cwd = os.getcwd()
        self.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

        self._skill_registry = SkillRegistry()
        self._skills_dir = xerxes_subdir("skills")
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        # Discover bundled (built-in) skills shipped with the package
        import xerxes as _xerxes_pkg

        _bundled_skills_dir = Path(_xerxes_pkg.__file__).parent / "skills"
        discover_dirs = [str(self._skills_dir), str(Path.cwd() / "skills")]
        if _bundled_skills_dir.is_dir():
            discover_dirs.insert(0, str(_bundled_skills_dir))

        self._skill_registry.discover(*discover_dirs)
        set_skill_registry(self._skill_registry)

        self._pending_skill_name: str = ""
        self._query_thread: threading.Thread | None = None
        self._permission_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._question_queue: queue.Queue[dict[str, Any]] = queue.Queue()

        # Wire up AskUserQuestionTool so it can block and wait for CLI input.
        set_ask_user_question_callback(self._ask_question)

        self._authoring_pipeline = SkillAuthoringPipeline(
            skills_dir=self._skills_dir,
            skill_registry=self._skill_registry,
        )
        self._pending_tool_inputs: dict[str, Any] | None = None

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
        model = self.config.get("model", "")
        context_limit = get_context_limit(model)
        total_tokens = self.state.total_input_tokens + self.state.total_output_tokens
        remaining = max(0, context_limit - total_tokens)
        self._emit(
            "state",
            {
                "turn_count": self.state.turn_count,
                "total_input_tokens": self.state.total_input_tokens,
                "total_output_tokens": self.state.total_output_tokens,
                "message_count": len(self.state.messages),
                "tool_execution_count": len(self.state.tool_executions),
                "context_limit": context_limit,
                "remaining_context": remaining,
                "cost_usd": calc_cost(
                    model,
                    self.state.total_input_tokens,
                    self.state.total_output_tokens,
                ),
            },
        )

    def _emit_text(self, text: str) -> None:
        """Emit text chunk, suppressing <function=...>...</function> markup.

        Drops pure-whitespace chunks to avoid visual gaps in the TUI.
        """
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
        if not stripped:
            return
        if stripped.startswith('{"name":') and '"arguments"' in stripped:
            return

        self._emit("text_chunk", {"text": text})

    def _on_agent_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Forward sub-agent and orchestration events to the Rust CLI."""
        self._emit(event_type, data)

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
                sessions.append(
                    {
                        "session_id": data.get("session_id", path.stem),
                        "model": data.get("model", ""),
                        "cwd": data.get("cwd", ""),
                        "updated_at": data.get("updated_at", ""),
                        "turns": data.get("turn_count", 0),
                        "preview": preview,
                    }
                )
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

                # Skip model list validation during init to avoid blocking on
                # slow/unreachable providers (SSL handshake can take 3-10s).
                # Trust the user-configured model; validation happens on query.

        has_profile = bool(model)
        self.config["model"] = model if model else ""
        if base_url:
            self.config["base_url"] = base_url
        if api_key:
            self.config["api_key"] = api_key

        boot = bootstrap(model=self.config["model"])
        self.system_prompt = boot.system_prompt

        # Auto-inject the xerxes-agent skill into the system prompt so the model
        # always knows it can (and should) spawn parallel sub-agents.
        agent_skill = self._skill_registry.get("xerxes-agent")
        if agent_skill is not None:
            self.system_prompt += "\n\n" + agent_skill.to_prompt_section()

        registry = populate_registry()
        self.tool_executor = build_tool_executor(registry=registry)
        self.tool_schemas = registry.tool_schemas()

        self._initialized = True

        set_global_config(self.config)
        set_event_callback(self._on_agent_event)

        provider = detect_provider(model)
        skill_names = sorted(self._skill_registry.skill_names)
        self._emit(
            "ready",
            {
                "model": model,
                "provider": provider,
                "tools": len(self.tool_schemas),
                "permission_mode": self.config["permission_mode"],
                "has_profile": has_profile,
                "skills": skill_names,
            },
        )

    def handle_query(self, params: dict[str, Any], override_tool_schemas: list[dict[str, Any]] | None = None) -> None:
        if not self._initialized:
            self._emit_error("Not initialized. Send 'init' first.")
            return

        text = params.get("text", "").strip()
        if not text:
            self._emit_error("Empty query.")
            return

        if self._pending_skill_name:
            skill_name = self._pending_skill_name
            self._pending_skill_name = ""
            result = self._generate_skill(skill_name, text)
            self._emit("slash_result", {"output": result})
            self._emit("query_done", {})
            return

        self._cancel = False
        self._suppressing_tag = False
        self._suppress_buf.clear()
        self._pending_tool_inputs = None

        # Auto-compact conversation history when near context limit.
        self._maybe_compact_context()

        self._authoring_pipeline.begin_turn(
            agent_id="default",
            user_prompt=text,
        )

        schemas = override_tool_schemas if override_tool_schemas is not None else self.tool_schemas

        for event in run_agent_loop(
            user_message=text,
            state=self.state,
            config=self.config,
            system_prompt=self.system_prompt,
            tool_executor=self.tool_executor,
            tool_schemas=schemas,
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
                self._pending_tool_inputs = event.inputs
                self._emit(
                    "tool_start",
                    {
                        "name": event.name,
                        "inputs": event.inputs,
                        "tool_call_id": event.tool_call_id,
                    },
                )

            elif isinstance(event, ToolEnd):
                self._authoring_pipeline.record_call(
                    tool_name=event.name,
                    arguments=self._pending_tool_inputs or {},
                    status="success" if event.permitted else "blocked",
                    duration_ms=event.duration_ms,
                )
                self._pending_tool_inputs = None
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

                event.granted = self._wait_for_permission()

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

        final_response = ""
        for msg in reversed(self.state.messages):
            if msg.get("role") == "assistant":
                final_response = msg.get("content", "")
                break

        if not self._cancel:
            authoring_result = self._authoring_pipeline.on_turn_end(final_response=final_response)
            if authoring_result.authored and authoring_result.skill_path:
                self._emit(
                    "skill_suggested",
                    {
                        "skill_name": authoring_result.skill_name,
                        "version": authoring_result.version,
                        "source_path": str(authoring_result.skill_path),
                        "tool_count": len(authoring_result.candidate.events),
                        "unique_tools": authoring_result.candidate.unique_tools,
                    },
                )

        self._emit("query_done", {})
        self._emit_state()

    def _maybe_compact_context(self) -> None:
        """Compact conversation history when approaching the context limit.

        Uses a sliding-window strategy to drop oldest messages (except system
        and the most recent 6) when total tokens exceed 75 % of the model's
        context window.
        """
        model = self.config.get("model", "")
        if not model:
            return
        context_limit = get_context_limit(model)
        if context_limit <= 0:
            return
        total_tokens = self.state.total_input_tokens + self.state.total_output_tokens
        threshold = int(context_limit * 0.75)
        if total_tokens < threshold:
            return
        try:
            target = int(context_limit * 0.4)
            strategy = get_compaction_strategy(
                CompactionStrategy.SLIDING_WINDOW,
                target_tokens=target,
                model=model,
            )
            original_count = len(self.state.messages)
            self.state.messages = strategy.compact(self.state.messages)
            compacted_count = len(self.state.messages)
            if compacted_count < original_count:
                self._emit(
                    "slash_result",
                    {
                        "output": (
                            f"[Auto-compact] Context at {total_tokens:,}/{context_limit:,} tokens. "
                            f"Trimmed history from {original_count} → {compacted_count} messages."
                        ),
                    },
                )
        except Exception:
            logger.warning("Auto-compaction failed", exc_info=True)

    def _wait_for_permission(self) -> bool:
        """Wait for a permission_response from the main stdin loop.

        The main loop enqueues permission responses; we poll the queue
        so that cancel messages can still interrupt us.
        """
        while True:
            if self._cancel:
                return False
            try:
                msg = self._permission_queue.get(timeout=0.1)
                return msg.get("params", {}).get("granted", False)
            except queue.Empty:
                continue

    def _ask_question(self, question: str) -> str:
        """Emit a question_request event and block until the CLI responds.

        Called by :class:`AskUserQuestionTool` when it needs interactive
        user input.
        """
        self._emit("question_request", {"question": question})
        return self._wait_for_question_response()

    def _wait_for_question_response(self) -> str:
        """Poll the question queue for a response from the CLI.

        Returns the user's answer text, or a cancellation marker if the
        query is cancelled.
        """
        while True:
            if self._cancel:
                return "[cancelled]"
            try:
                msg = self._question_queue.get(timeout=0.1)
                return msg.get("params", {}).get("answer", "")
            except queue.Empty:
                continue

    def handle_question_response(self, params: dict[str, Any]) -> None:
        """Enqueue a question answer from the CLI."""
        self._question_queue.put({"params": params})

    def handle_cancel(self) -> None:
        self._cancel = True

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
        try:
            models = profiles.fetch_models(base_url, api_key)
        except Exception as exc:
            self._emit_error(f"Failed to fetch models: {exc}")
            return
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

        self._emit(
            "provider_saved",
            {
                "profile": profile,
                "message": f"Profile '{name}' saved and activated. Model: {model}",
            },
        )

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
            self._emit(
                "provider_saved",
                {
                    "profile": profile,
                    "message": f"Switched to profile '{name}'. Model: {profile['model']}",
                },
            )

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

        self._pending_skill_name = ""

        if cmd == "skill":
            self._handle_skill_invoke(args)
            return

        output = self._run_slash(cmd, args)
        if output:
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
                current_val = self.config.get(k, None)
                if current_val is not None:
                    lines.append(f"  {k}: {current_val}")
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
            set_global_config(self.config)
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
        set_global_config(self.config)
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
        self.state.messages = [
            *system_msgs,
            {
                "role": "user",
                "content": f"[Previous conversation summary — {len(older)} messages compacted]\n\n{summary}",
            },
            *recent,
        ]
        new_count = len(self.state.messages)

        return (
            f"Compacted {original_count} messages → {new_count} messages.\n"
            f"Summarized {len(older)} older messages, kept {len(recent)} recent + {len(system_msgs)} system."
        )

    def _handle_plan(self, args: str) -> str:
        """Handle /plan — create and execute a multi-step plan."""
        objective = args.strip()
        if not objective:
            return "Usage: /plan <objective>\n\nExample: /plan refactor the auth module into separate files"

        from ..tools.claude_tools import PlanTool

        return PlanTool.static_call(objective=objective, execute=True)

    def _handle_agents_list(self) -> str:
        """Handle /agents — list available agent types and running agents."""
        from ..agents.definitions import list_agent_definitions
        from ..tools.claude_tools import _get_agent_manager

        defs = list_agent_definitions()
        lines = [f"Agent types ({len(defs)}):"]
        for d in defs:
            source_tag = f" [{d.source}]" if d.source != "built-in" else ""
            lines.append(f"  {d.name}{source_tag} — {d.description}")

        mgr = _get_agent_manager()
        tasks = mgr.list_tasks()
        if tasks:
            lines.append(f"\nRunning agents ({len(tasks)}):")
            for t in tasks:
                agent_type = f" ({t.agent_def_name})" if t.agent_def_name else ""
                lines.append(f"  {t.name}{agent_type} [{t.status}] — {t.prompt[:60]}")
        else:
            lines.append("\nNo running agents.")

        return "\n".join(lines)

    def _handle_skills_list(self) -> str:
        """List all discovered skills."""

        import xerxes as _xerxes_pkg

        _bundled = Path(_xerxes_pkg.__file__).parent / "skills"
        discover_dirs = [str(self._skills_dir), str(Path.cwd() / "skills")]
        if _bundled.is_dir():
            discover_dirs.insert(0, str(_bundled))
        self._skill_registry.discover(*discover_dirs)
        skills = self._skill_registry.get_all()
        if not skills:
            return f"No skills found.\n  Skills directory: {self._skills_dir}\n  Create one with /skill-create"
        lines = [f"Skills ({len(skills)}):"]
        for s in skills:
            tags = f" [{', '.join(s.metadata.tags)}]" if s.metadata.tags else ""
            lines.append(f"  {s.name}{tags} — {s.metadata.description or 'No description'}")
        lines.append("\nUse /skill <name> to invoke a skill")
        return "\n".join(lines)

    def _handle_skill_invoke(self, args: str) -> None:
        """Invoke a skill by name — execute it immediately as a query."""
        name = args.strip()
        if not name:
            self._emit("slash_result", {"output": "Usage: /skill <name>\nUse /skills to list available skills."})
            return

        skill_args = ""
        if ":" in name:
            name, skill_args = name.split(":", 1)
            name = name.strip()
            skill_args = skill_args.strip()

        skill = self._skill_registry.get(name)
        if not skill:
            matches = self._skill_registry.search(name)
            if matches:
                suggestions = ", ".join(s.name for s in matches[:5])
                self._emit("slash_result", {"output": f"Skill '{name}' not found. Did you mean: {suggestions}"})
            else:
                self._emit(
                    "slash_result", {"output": f"Skill '{name}' not found. Use /skills to list available skills."}
                )
            return

        from xerxes.extensions.skills import skill_matches_platform

        if not skill_matches_platform(skill):
            self._emit(
                "slash_result",
                {"output": f"Skill '{name}' is not compatible with this platform ({__import__('sys').platform})."},
            )
            return

        from xerxes.extensions.skills import inject_skill_config

        prompt_section = skill.to_prompt_section()
        config_block = inject_skill_config(skill)
        skill_message = f"[Skill '{name}' activated]{config_block}\n\n{prompt_section}"

        self.state.messages.append(
            {
                "role": "user",
                "content": skill_message,
            }
        )

        self._emit("slash_result", {"output": f"Running skill '{name}'..."})

        trigger = skill_args if skill_args else f"Execute the '{name}' skill now."
        filtered_schemas = [s for s in self.tool_schemas if s.get("name") != "SkillTool"]
        self.handle_query({"text": trigger}, override_tool_schemas=filtered_schemas)

    def _handle_skill_create(self, args: str) -> str:
        """Create a new skill — two-step: name first, then description prompt."""
        name = args.strip()
        if not name:
            return (
                "Usage: /skill-create <name>\n"
                "  Example: /skill-create code-review\n\n"
                "After entering the name, describe what the skill should do\n"
                "and the SKILL.md will be auto-generated."
            )

        if not all(c.isalnum() or c in "-_" for c in name):
            return f"Invalid skill name '{name}'. Use only letters, numbers, hyphens, and underscores."

        skill_dir = self._skills_dir / name
        if skill_dir.exists():
            return f"Skill '{name}' already exists at {skill_dir}"

        self._pending_skill_name = name
        return f"Creating skill '{name}'. Describe what this skill should do:"

    def _generate_skill(self, name: str, description: str) -> str:
        """Generate a SKILL.md using the LLM from the user's description."""
        model = self.config.get("model", "")
        if not model:
            return self._create_skill_template(name, description)

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
                            "You generate SKILL.md files for the Xerxes agent framework. "
                            "A skill is a reusable set of instructions that an agent follows "
                            "when the skill is invoked via `/skill <name>`.\n\n"
                            "Output format (YAML frontmatter + markdown body):\n"
                            "```\n"
                            "---\n"
                            "name: skill-name\n"
                            "description: One-line description\n"
                            'version: "1.0"\n'
                            "tags: [tag1, tag2]\n"
                            "---\n\n"
                            "# Skill Title\n\n"
                            "Detailed step-by-step instructions for the agent...\n"
                            "```\n\n"
                            "Write clear, actionable instructions. Be specific about what "
                            "tools to use, what to check, and what format to output. "
                            "Output ONLY the SKILL.md content, nothing else."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Create a skill named '{name}' that does the following:\n\n{description}",
                    },
                ],
                max_tokens=2048,
                temperature=0.3,
            )

            content = response.choices[0].message.content or ""

            if content.startswith("```"):
                lines = content.split("\n")
                if lines[0].strip().startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                content = "\n".join(lines)

            if not content.strip():
                return self._create_skill_template(name, description)

        except Exception as exc:
            return self._create_skill_template(name, description, error=str(exc))

        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(content, encoding="utf-8")

        self._skill_registry.discover(str(self._skills_dir))
        self._emit("skills_updated", {"skills": sorted(self._skill_registry.skill_names)})

        return f"Skill '{name}' generated and saved to {skill_dir}/SKILL.md\nUse /skill {name} to invoke it."

    def _create_skill_template(self, name: str, description: str, error: str = "") -> str:
        """Fallback: create a SKILL.md template with the description embedded."""
        title = name.replace("-", " ").replace("_", " ").title()
        skill_dir = self._skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text(
            f"---\n"
            f"name: {name}\n"
            f"description: {description[:80]}\n"
            f'version: "1.0"\n'
            f"tags: []\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"{description}\n",
            encoding="utf-8",
        )
        self._skill_registry.discover(str(self._skills_dir))
        self._emit("skills_updated", {"skills": sorted(self._skill_registry.skill_names)})
        err_note = f"\n(LLM generation failed: {error}. Created template instead.)" if error else ""
        return f"Skill '{name}' created at {skill_dir}/SKILL.md{err_note}\nUse /skill {name} to invoke it."

    def _run_slash(self, cmd: str, args: str) -> str:
        if cmd in ("help", "h"):
            return (
                "Commands:\n"
                "  /provider          Setup or switch provider profile\n"
                "  /sampling          View or set sampling parameters\n"
                "  /compact           Summarize conversation to free context\n"
                "  /plan OBJECTIVE    Plan and execute a multi-step task\n"
                "  /agents            List agent types and running agents\n"
                "  /skills            List available skills\n"
                "  /skill NAME        Invoke a skill by name\n"
                "  /skill-create      Create a new skill\n"
                "  /model NAME        Switch model\n"
                "  /cost              Show cost summary\n"
                "  /context           Show context info\n"
                "  /clear             Clear conversation\n"
                "  /tools             List available tools\n"
                "  /thinking          Toggle thinking display\n"
                "  /verbose           Toggle verbose mode\n"
                "  /debug             Toggle debug mode\n"
                "  /permissions       Cycle permission mode\n"
                "  /yolo              Toggle accept-all permission mode\n"
                "  /config            Show config\n"
                "  /history           Show message count\n"
                "  /exit              Exit"
            )

        if cmd == "model":
            if args:
                self.config["model"] = args
                set_global_config(self.config)
                self._emit("model_changed", {"model": args, "provider": detect_provider(args)})
                return f"Model set to: {args}"

            current = self.config.get("model", "(none)")
            base_url = self.config.get("base_url", "")
            api_key = self.config.get("api_key", "")
            lines = [f"Current model: {current}"]
            if base_url:
                try:
                    available = profiles.fetch_models(base_url, api_key)
                except Exception as exc:
                    lines.append(f"Could not fetch models from {base_url}/models: {exc}")
                    return "\n".join(lines)
                if available:
                    lines.append(f"\nAvailable models ({len(available)}):")
                    for m in available:
                        marker = " (active)" if m == current else ""
                        lines.append(f"  {m}{marker}")
                    lines.append("\nUse /model <name> to switch")
                else:
                    lines.append(f"No models returned from {base_url}/models")
            return "\n".join(lines)

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

        if cmd == "skills":
            return self._handle_skills_list()

        if cmd == "skill-create":
            return self._handle_skill_create(args)

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
            set_global_config(self.config)
            return f"Permission mode: {new_mode}"

        if cmd == "yolo":
            current = self.config.get("permission_mode", "auto")
            if current == "accept-all":
                self.config["permission_mode"] = "auto"
            else:
                self.config["permission_mode"] = "accept-all"
            set_global_config(self.config)
            return f"YOLO mode {'OFF (auto)' if current == 'accept-all' else 'ON (accept-all)'}"

        if cmd == "tools":
            registry = populate_registry()
            lines = []
            for entry in registry.list_tools():
                safe = " [safe]" if entry.safe else ""
                lines.append(f"  {entry.name}{safe} -- {entry.description[:60]}")
            lines.append(f"  ({registry.tool_count} total)")
            return "\n".join(lines)

        if cmd == "plan":
            return self._handle_plan(args)

        if cmd == "agents":
            return self._handle_agents_list()

        if cmd in ("exit", "quit", "q"):
            self._emit("exit", {})
            sys.exit(0)

        skill = self._skill_registry.get(cmd)
        if skill:
            self._handle_skill_invoke(cmd)
            return ""
        return f"Unknown command: /{cmd} (type /help)"

    def _parse_json_messages(self, line: str) -> list[dict[str, Any]]:
        """Parse a line that may contain one or more JSON messages.

        Tries to parse the whole line first, then falls back to splitting
        by whitespace to handle multiple JSON objects on one line (which can
        happen with rapid-fire commands from the TypeScript CLI).

        Returns a list of parsed message dicts.
        """

        try:
            return [json.loads(line)]
        except json.JSONDecodeError:
            pass

        messages = []
        for token in line.split():
            if not token.strip():
                continue
            try:
                messages.append(json.loads(token))
            except json.JSONDecodeError:
                self._emit_error(f"Invalid JSON: {token[:100]}")
        return messages

    def run(self) -> None:
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line or not self._running:
                continue

            messages = self._parse_json_messages(line)
            if not messages:
                continue

            for msg in messages:
                method = msg.get("method", "")
                params = msg.get("params", {})

                try:
                    if method == "init":
                        self.handle_init(params)
                    elif method == "query":
                        if self._query_thread is not None and self._query_thread.is_alive():
                            self._emit_error("A query is already running. Wait or send cancel.")
                        else:

                            def _run_query(_params: dict[str, Any]) -> None:
                                try:
                                    self.handle_query(_params)
                                except Exception as exc:
                                    self._emit_error(f"{type(exc).__name__}: {exc}")
                                    self._emit("query_done", {})

                            self._query_thread = threading.Thread(target=_run_query, args=(params,), daemon=True)
                            self._query_thread.start()
                    elif method == "permission_response":
                        self._permission_queue.put(msg)
                    elif method == "question_response":
                        self.handle_question_response(params)
                    elif method == "cancel":
                        self.handle_cancel()
                    elif method == "slash":
                        if self._query_thread is not None and self._query_thread.is_alive():
                            self._emit_error("A query is already running. Wait or send cancel.")
                        else:

                            def _run_slash(_params: dict[str, Any]) -> None:
                                try:
                                    self.handle_slash(_params)
                                except Exception as exc:
                                    self._emit_error(f"{type(exc).__name__}: {exc}")
                                    self._emit("query_done", {})

                            self._query_thread = threading.Thread(target=_run_slash, args=(params,), daemon=True)
                            self._query_thread.start()
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
                    elif method == "shutdown":
                        self._running = False
                    else:
                        self._emit_error(f"Unknown method: {method}")
                except Exception as exc:
                    self._emit_error(f"{type(exc).__name__}: {exc}")


def main() -> None:
    import argparse
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    parser = argparse.ArgumentParser(description="Xerxes bridge server (JSON-RPC over stdio)")
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
