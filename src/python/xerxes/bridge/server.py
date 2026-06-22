# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
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
"""Stdio JSON-RPC bridge hosting the agent runtime in-process.

:class:`BridgeServer` reads newline-delimited JSON-RPC messages from
``stdin``, dispatches them to handlers (``init``, ``query``, ``slash``,
``cancel``, provider/profile CRUD, ...), and emits events back on ``stdout``
under one of two protocols:

* **Legacy** ``{event, data}`` flat objects — what older Xerxes clients expect.
* **Wire** mode — Kimi-Code-compatible JSON-RPC 2.0 NDJSON with
  ``method=event`` / ``method=request`` frames so the same TUI can talk to
  Kimi and Xerxes interchangeably.

The bridge owns the same streaming loop the daemon uses, plus per-session
JSON persistence under ``$XERXES_HOME/sessions``, the skill authoring
pipeline, and inline auto-compaction when the conversation exceeds 75% of
the model context window. :func:`main` is the console-script entry point.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import sys
import threading
import uuid
from pathlib import Path
from typing import Any

from ..context.compaction_provisioner import (
    CompactionProvisioner,
    compaction_summary_agent_from_config,
)
from ..context.window_usage import estimate_context_tokens
from ..core.paths import xerxes_subdir
from ..extensions.skill_authoring.pipeline import SkillAuthoringPipeline
from ..extensions.skills import SkillRegistry, default_skill_discovery_dirs
from ..llms.registry import calc_cost, detect_provider, get_context_limit
from ..runtime.agent_memory import AgentMemory
from ..runtime.bootstrap import bootstrap
from ..runtime.bridge import build_tool_executor, populate_registry
from ..runtime.change_guard import analyze_workspace_changes, format_change_guard_notification
from ..runtime.config_context import set_config as set_global_config
from ..runtime.config_context import set_event_callback
from ..runtime.cost_tracker import CostTracker
from ..runtime.interaction_modes import (
    agent_name_for_mode,
    mode_switch_hint,
    normalize_interaction_mode,
)
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
from ..tools.agent_memory_tool import set_active_memory
from ..tools.agent_meta_tools import set_skill_registry
from ..tools.claude_tools import set_ask_user_question_callback
from . import profiles
from .session_mixin import SessionMixin
from .slash_handler import SlashHandlerMixin
from .wire_events_mixin import WireEventMixin

logger = logging.getLogger(__name__)


def _normalize_interaction_mode(mode: Any, *, plan_mode: bool = False) -> str:
    """Coerce assorted mode labels through the shared interaction-mode vocabulary."""
    return normalize_interaction_mode(mode, plan_mode=plan_mode)


def _agent_name_for_mode(mode: str) -> str:
    """Map an interaction mode to its YAML agent definition."""
    return agent_name_for_mode(mode)


class BridgeServer(WireEventMixin, SlashHandlerMixin, SessionMixin):
    """Stdio JSON-RPC server hosting one in-process agent session.

    Owns the conversation state (:class:`AgentState`), the tool executor and
    schema list, the cost tracker, the active skill registry, the sub-agent
    event aggregator, and the per-session resume / replay machinery. Frame
    emission is dual-mode: the legacy ``{event, data}`` shape and the Kimi
    wire-protocol JSON-RPC frames are toggled by ``wire_mode``.
    """

    SESSIONS_DIR = xerxes_subdir("sessions")

    def __init__(self, wire_mode: bool = False) -> None:
        """Build state, registries, and IO locks.

        Args:
            wire_mode: When true, emit Kimi-compatible JSON-RPC 2.0 NDJSON
                instead of the legacy flat ``{event, data}`` shape.
        """
        self.config: dict[str, Any] = {}
        self.state = AgentState()
        self.cost_tracker = CostTracker()
        self.system_prompt = ""
        self.agent_memory: AgentMemory | None = None
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
        self._wire_mode = wire_mode
        self._current_tool_call_id = ""
        self._step_count = 0

        self._subagent_text_buffers: dict[str, str] = {}
        self._subagent_thinking_buffers: dict[str, str] = {}
        self._subagent_buffer_lock = threading.Lock()
        self._subagent_parent_tool: dict[str, str] = {}
        self._subagent_tool_id_fifo: dict[str, list[str]] = {}
        self._SUBAGENT_FLUSH_THRESHOLD = 400

        self._pending_resume_replays: list[dict[str, Any]] = []

        self._skill_registry = SkillRegistry()
        self._skills_dir = xerxes_subdir("skills")
        self._skills_dir.mkdir(parents=True, exist_ok=True)

        self._skill_registry.discover(*default_skill_discovery_dirs(user_skills_dir=self._skills_dir))
        set_skill_registry(self._skill_registry)

        self._pending_skill_name: str = ""
        self._query_thread: threading.Thread | None = None
        self._permission_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._question_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        # Id of the permission/question request currently being waited on, used
        # to reject stale responses that arrive after a cancel or that belong to
        # a previous turn (an empty string means "no active prompt").
        self._active_permission_id: str = ""
        self._active_question_id: str = ""

        set_ask_user_question_callback(self._ask_question)

        self._authoring_pipeline = SkillAuthoringPipeline(
            skills_dir=self._skills_dir,
            skill_registry=self._skill_registry,
        )
        self._pending_tool_inputs: dict[str, Any] | None = None

    def _emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Emit a legacy ``{event, data}`` line on stdout.

        In wire mode, ``slash_result`` events are reflected as
        ``category="slash"`` notifications instead; other event names are
        dropped (wire mode already emits them through ``_emit_wire_*``).
        """
        if self._wire_mode:
            if event == "slash_result":
                self._emit_wire_notification(
                    notification_id=str(uuid.uuid4()),
                    category="slash",
                    type_="slash_result",
                    severity="info",
                    title="",
                    body=str((data or {}).get("output", "")),
                )
            return
        msg = {"event": event, "data": data or {}}
        line = json.dumps(msg, ensure_ascii=False, default=str)
        with self._out_lock:
            self._stdout.write(line + "\n")
            self._stdout.flush()

    def _emit_error(self, message: str) -> None:
        """Emit ``message`` as an error event (or wire ``severity=error`` notification)."""
        if self._wire_mode:
            self._emit_wire_notification(
                notification_id=str(uuid.uuid4()),
                category="bridge",
                type_="error",
                severity="error",
                title="Bridge error",
                body=message,
            )
            return
        self._emit("error", {"message": message})

    def _emit_state(self) -> None:
        """Emit a legacy ``state`` event summarising tokens, cost, and message count."""
        model = self.config.get("model", "")
        context_limit = get_context_limit(model)
        context_tokens = estimate_context_tokens(self.state.messages, model=model)
        remaining = max(0, context_limit - context_tokens)
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
                "used_context": context_tokens,
                "cost_usd": calc_cost(
                    model,
                    self.state.total_input_tokens,
                    self.state.total_output_tokens,
                ),
                "reasoning_effort": self.config.get("reasoning_effort", "off"),
            },
        )

    def _emit_text(self, text: str) -> None:
        """Emit a legacy ``text_chunk`` event, eliding any ``<function=...>...</function>`` payloads.

        Tool calls some providers stream as inline ``<function>`` blocks would
        otherwise leak into the rendered transcript; this stateful filter
        buffers across chunks until the closing tag arrives.
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
        """Handle a runtime callback event (mode change, sub-agent activity, ...).

        Re-emits the event on both protocols and, for ``agent_*`` events in
        wire mode, calls :meth:`_emit_subagent_summary` so the TUI sees the
        chatty stream folded into compact preview notifications.
        """
        if event_type == "interaction_mode_changed":
            mode = _normalize_interaction_mode(data.get("mode"), plan_mode=bool(data.get("plan_mode", False)))
            self.config["mode"] = mode
            self.config["plan_mode"] = mode == "plan"
            set_global_config(self.config)
            if self._wire_mode:
                self._emit_wire_status()
            self._emit(event_type, {**data, "mode": mode, "plan_mode": mode == "plan"})
            return
        self._emit(event_type, data)
        if self._wire_mode and event_type.startswith("agent_"):
            self._emit_subagent_summary(event_type, data)

    def _agent_definition_for_mode(self, mode: str) -> Any:
        """Look up the YAML agent definition for ``mode`` or return ``None``."""
        try:
            from ..agents.definitions import get_agent_definition

            return get_agent_definition(_agent_name_for_mode(mode))
        except Exception:
            return None

    def _system_prompt_for_mode(self, mode: str) -> str:
        """Return the mode-specific system prompt with the mode-switching hint appended."""
        agent_def = self._agent_definition_for_mode(mode)
        prompt_value = getattr(agent_def, "system_prompt", None) if agent_def is not None else None
        if isinstance(prompt_value, str) and prompt_value:
            prompt = prompt_value.rstrip()
        else:
            prompt = self.system_prompt.rstrip()

        if self.agent_memory is not None:
            try:
                memory_section = self.agent_memory.to_prompt_section()
            except Exception as exc:
                logger.warning("Failed to render agent memory section: %s", exc)
            else:
                if memory_section:
                    prompt += "\n\n" + memory_section

        return prompt + "\n\n" + self._mode_switch_hint(mode) + "\n"

    def _begin_turn_memory_sync(self) -> None:
        """Keep the active two-tier memory bound for this bridge turn."""
        if self.agent_memory is None:
            return
        self.agent_memory.ensure()
        set_active_memory(self.agent_memory)

    @staticmethod
    def _mode_switch_hint(mode: str) -> str:
        """Return the ``[Mode control]`` paragraph appended to system prompts.

        The hint tells the model which ``SetInteractionModeTool`` call moves
        forward from the current mode.
        """
        return mode_switch_hint(mode)

    def handle_init(self, params: dict[str, Any]) -> None:
        """Bootstrap the runtime and emit ``ready``/``init_done``.

        ``params`` may carry ``permission_mode``, ``verbose``, ``thinking``,
        ``debug``, ``mode``, ``model``, ``base_url``, ``api_key``,
        ``resume_session_id``, and ``mcp_servers``. The active provider
        profile fills any gaps, the bootstrap is run, tools and skills are
        loaded, agent definitions are resolved, and MCP servers (if any) are
        connected in background threads.
        """
        self.config = {
            "permission_mode": params.get("permission_mode", "accept-all"),
            "verbose": params.get("verbose", False),
            "thinking": params.get("thinking", True),
            "debug": params.get("debug", False),
            "mode": _normalize_interaction_mode(params.get("mode")),
            "plan_mode": False,
        }

        model = params.get("model", "")
        base_url = params.get("base_url", "")
        api_key = params.get("api_key", "")

        resume_id = params.get("resume_session_id") or ""
        if resume_id:
            self._load_session(resume_id)

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
        project_root = Path(str(params.get("project_dir") or self._session_cwd or os.getcwd())).expanduser()
        try:
            project_root = project_root.resolve()
        except OSError:
            project_root = project_root.absolute()
        self._session_cwd = str(project_root)
        self.config["project_dir"] = str(project_root)

        if base_url and not params.get("model", ""):
            try:
                available = profiles.fetch_models(base_url, api_key)
            except Exception:
                available = []
            self._auto_switch_stale_model(available)

        boot = bootstrap(model=self.config["model"], cwd=project_root)
        self.system_prompt = boot.system_prompt

        agent_skill = self._skill_registry.get("xerxes-agent")
        if agent_skill is not None:
            self.system_prompt += "\n\n" + agent_skill.to_prompt_section()

        self.agent_memory = AgentMemory(project_root=project_root)
        self.agent_memory.ensure()
        set_active_memory(self.agent_memory)

        registry = populate_registry()
        self.tool_executor = build_tool_executor(registry=registry)
        self.tool_schemas = registry.tool_schemas()

        try:
            from ..agents.definitions import (
                get_agent_definition,
                list_agent_definition_load_errors,
                list_agent_definitions,
            )

            agent_defs = list_agent_definitions()
            agent_errors = list_agent_definition_load_errors()
            self.config["_agent_definitions"] = [d.name for d in agent_defs]
            default_agent = get_agent_definition("default")
            if default_agent:
                self.tool_schemas = self._filter_tool_schemas_for_agent(self.tool_schemas, default_agent)
        except Exception as exc:
            agent_defs = []
            agent_errors = [f"Failed to load agent definitions: {exc}"]

        if self._wire_mode:
            for error in agent_errors:
                self._emit_wire_notification(
                    notification_id=str(uuid.uuid4()),
                    category="agents",
                    type_="agent_spec_error",
                    severity="error",
                    title="agent.yaml ignored",
                    body=error,
                )

        mcp_servers = params.get("mcp_servers", [])
        for server_config in mcp_servers:
            server_name = server_config.get("name", "unknown")
            if self._wire_mode:
                self._emit_wire_event(
                    "mcp_loading_begin",
                    {"server_name": server_name},
                )
            try:
                from ..mcp import MCPManager, MCPServerConfig

                manager = MCPManager()
                cfg = MCPServerConfig(
                    name=server_config.get("name", ""),
                    command=server_config.get("command", ""),
                    args=server_config.get("args", []),
                    env=server_config.get("env"),
                    url=server_config.get("url"),
                    transport=server_config.get("transport", "stdio"),
                    enabled=server_config.get("enabled", True),
                )

                def _connect(
                    _manager: MCPManager = manager,
                    _cfg: MCPServerConfig = cfg,
                ) -> None:
                    """Spin up a dedicated event loop and connect the MCP server."""
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(_manager.add_server(_cfg))
                    finally:
                        loop.close()

                import threading

                t = threading.Thread(target=_connect, daemon=True)
                t.start()
                success = True
            except Exception as exc:
                logger.warning("Failed to connect MCP server %s: %s", server_name, exc)
                success = False

            if self._wire_mode:
                self._emit_wire_event(
                    "mcp_loading_end",
                    {"server_name": server_name, "success": success},
                )

        self._initialized = True

        set_global_config(self.config)
        set_event_callback(self._on_agent_event)

        if self._pending_resume_replays:
            self._replay_pending_tool_calls()

        active_model = self.config.get("model", "")
        provider = detect_provider(active_model)
        skill_names = sorted(self._skill_registry.skill_names)

        self._emit_wire_init_done()

        if resume_id and self._wire_mode and self.state.messages:
            self._replay_history_to_wire()
        self._emit(
            "ready",
            {
                "model": active_model,
                "provider": provider,
                "tools": len(self.tool_schemas),
                "permission_mode": self.config["permission_mode"],
                "has_profile": has_profile,
                "skills": skill_names,
                "agents": [d.name for d in agent_defs],
            },
        )

    def handle_query(self, params: dict[str, Any], override_tool_schemas: list[dict[str, Any]] | None = None) -> None:
        """Run one streaming-loop turn against the model.

        ``params["text"]`` carries the user input (string or list of content
        parts). ``params["mode"]``/``params["plan_mode"]`` switch the
        interaction mode for this turn. If a skill scaffold is pending from
        ``/skill-create``, this turn generates the SKILL.md instead of
        running the agent. ``override_tool_schemas`` narrows the toolset
        exposed to the model (used when a skill suppresses ``SkillTool``).
        """
        if not self._initialized:
            self._emit_error("Not initialized. Send 'init' first.")
            return

        raw_text = params.get("text", "")
        if isinstance(raw_text, list):
            parts: list[str] = []
            for item in raw_text:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", item)))
                else:
                    parts.append(str(item))
            raw_text = "\n".join(parts)
        text = str(raw_text).strip()
        if not text:
            self._emit_error("Empty query.")
            return

        if "plan_mode" in params:
            self.config["plan_mode"] = bool(params.get("plan_mode"))
            set_global_config(self.config)
        interaction_mode = _normalize_interaction_mode(
            params.get("mode", self.config.get("mode")),
            plan_mode=bool(self.config.get("plan_mode", False)),
        )
        self.config["mode"] = interaction_mode
        self.config["plan_mode"] = interaction_mode == "plan"
        set_global_config(self.config)

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
        # Discard any permission/question response that was enqueued for a
        # previous (possibly cancelled) turn so it cannot auto-resolve a fresh
        # prompt raised during this turn.
        self._active_permission_id = ""
        self._active_question_id = ""
        self._drain_queue(self._permission_queue)
        self._drain_queue(self._question_queue)

        # Proactively sync memory at the start of each turn
        self._begin_turn_memory_sync()

        if self._wire_mode:
            self._maybe_compact_context_wire()
        else:
            self._maybe_compact_context()

        self._authoring_pipeline.begin_turn(
            agent_id="default",
            user_prompt=text,
        )

        if self._wire_mode:
            self._emit_wire_turn_begin(text)

        schemas = override_tool_schemas if override_tool_schemas is not None else self.tool_schemas
        system_prompt = self._system_prompt_for_mode(interaction_mode)
        mode_agent_def = self._agent_definition_for_mode(interaction_mode)
        if mode_agent_def is not None:
            schemas = self._filter_tool_schemas_for_agent(schemas, mode_agent_def)

        for event in run_agent_loop(
            user_message=text,
            state=self.state,
            config=self.config,
            system_prompt=system_prompt,
            tool_executor=self.tool_executor,
            tool_schemas=schemas,
            cancel_check=lambda: self._cancel,
        ):
            if self._cancel:
                break

            if isinstance(event, TextChunk):
                if self._wire_mode:
                    self._emit_wire_text(event.text)
                self._emit_text(event.text)

            elif isinstance(event, ThinkingChunk):
                if self._wire_mode:
                    self._emit_wire_think(event.text)
                self._emit("thinking_chunk", {"text": event.text})

            elif isinstance(event, ToolStart):
                self._suppressing_tag = False
                self._suppress_buf.clear()
                self._pending_tool_inputs = event.inputs
                if self._wire_mode:
                    self._emit_wire_tool_start(event.tool_call_id, event.name, event.inputs)
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
                if self._wire_mode:
                    self._emit_wire_tool_result(
                        event.tool_call_id,
                        event.result,
                        event.permitted,
                        event.duration_ms,
                    )
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
                # Drain any response left over from a previous (possibly
                # cancelled) prompt before issuing a fresh request id, so a
                # stale verdict can never auto-resolve this request.
                self._drain_queue(self._permission_queue)
                if self._wire_mode:
                    self._active_permission_id = self._emit_wire_permission_request(
                        self._current_tool_call_id, event.tool_name, event.description
                    )

                    event.granted = self._wait_for_permission()
                else:
                    self._active_permission_id = str(uuid.uuid4())
                    self._emit(
                        "permission_request",
                        {
                            "id": self._active_permission_id,
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

        self._emit_change_guard_if_needed()

        try:
            self._save_session()
        except Exception as exc:
            logger.warning("Failed to save session %s: %s", self._session_id, exc)

        if self._wire_mode:
            self._emit_wire_turn_end()
            self._emit_wire_status()

        self._emit("query_done", {})
        self._emit_state()

    def _emit_change_guard_if_needed(self) -> None:
        """Notify once when the current Git diff contains high-risk edits."""
        report = analyze_workspace_changes(
            Path(self._session_cwd or os.getcwd()).expanduser(), self.state.tool_executions
        )
        if not report.should_notify:
            return

        metadata = dict(self.state.metadata or {})
        if metadata.get("last_change_guard_fingerprint") == report.fingerprint:
            return
        metadata["last_change_guard_fingerprint"] = report.fingerprint
        self.state.metadata = metadata

        payload = {
            "fingerprint": report.fingerprint,
            "findings": [finding.__dict__ for finding in report.findings],
            "verification_commands": list(report.verification_commands),
        }
        if self._wire_mode:
            self._emit_wire_notification(
                notification_id=str(uuid.uuid4()),
                category="workspace_guard",
                type_="risky_changes",
                severity=report.severity,
                title="Workspace guard",
                body=format_change_guard_notification(report),
                payload=payload,
            )
            return
        self._emit(
            "notification",
            {
                "id": str(uuid.uuid4()),
                "category": "workspace_guard",
                "type": "risky_changes",
                "severity": report.severity,
                "title": "Workspace guard",
                "body": format_change_guard_notification(report),
                "payload": payload,
            },
        )

    @staticmethod
    def _filter_tool_schemas_for_agent(tool_schemas: list[dict[str, Any]], agent_def: Any) -> list[dict[str, Any]]:
        """Return ``tool_schemas`` restricted to ``agent_def``'s ``tools``/``allowed_tools``/``exclude_tools``."""
        names = {schema.get("name", "") for schema in tool_schemas}
        allowed = set(getattr(agent_def, "tools", None) or names)
        explicit_allowed = getattr(agent_def, "allowed_tools", None)
        if explicit_allowed:
            allowed &= set(explicit_allowed)
        allowed -= set(getattr(agent_def, "exclude_tools", None) or [])
        return [schema for schema in tool_schemas if schema.get("name", "") in allowed]

    def _context_compaction_provisioner(self) -> CompactionProvisioner | None:
        """Return the bridge context compaction provisioner, if a model is configured."""
        model = self.config.get("model", "")
        if not model:
            return None
        context_limit = get_context_limit(model)
        if context_limit <= 0:
            return None
        return CompactionProvisioner(
            model=model,
            max_context_tokens=context_limit,
            threshold_ratio=float(self.config.get("compaction_threshold", 0.75)),
            target_ratio=float(self.config.get("compaction_target", 0.5)),
            summary_agent=compaction_summary_agent_from_config(model, self.config),
        )

    def _run_context_compaction(self, *, force: bool) -> Any:
        """Run the shared provisioner and install compacted messages on success."""
        provisioner = self._context_compaction_provisioner()
        if provisioner is None:
            return None
        result = provisioner.compact(self.state.messages, force=force)
        if result.compacted:
            self.state.messages = result.messages
        return result

    def _maybe_compact_context(self) -> None:
        """Auto-compact older messages via the shared agent-backed provisioner."""
        provisioner = self._context_compaction_provisioner()
        if provisioner is None or not provisioner.should_compact(self.state.messages):
            return

        result = self._run_context_compaction(force=False)
        if result is None or not result.compacted:
            if result is not None and result.error:
                logger.warning("Auto-compaction failed: %s", result.error)
            return

        self._emit(
            "slash_result",
            {
                "output": (
                    f"[Auto-compact] Context at {result.tokens_before:,} tokens. "
                    f"Summarized {result.summarized_count} messages; kept {len(self.state.messages)} messages."
                ),
            },
        )

    def _maybe_compact_context_wire(self) -> None:
        """Wire-protocol twin of :meth:`_maybe_compact_context`."""
        provisioner = self._context_compaction_provisioner()
        if provisioner is None or not provisioner.should_compact(self.state.messages):
            return

        self._emit_wire_compaction_begin()
        result = self._run_context_compaction(force=False)
        self._emit_wire_compaction_end()
        if result is None or not result.compacted:
            return

        self._emit_wire_notification(
            notification_id=str(uuid.uuid4()),
            category="context",
            type_="compacted",
            severity="info",
            title="Context compacted",
            body=f"Summarized {result.summarized_count} messages; kept {len(self.state.messages)} messages.",
        )

    @staticmethod
    def _drain_queue(q: queue.Queue[dict[str, Any]]) -> None:
        """Discard every item currently buffered in ``q`` without blocking.

        Called at the start of each turn so a permission/question response that
        was enqueued after a cancel (or for a previous turn) can never be popped
        as the answer to a *new* prompt.
        """
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def _wait_for_permission(self) -> bool:
        """Block the query thread until the TUI answers; ``True`` means approved (incl. for-session).

        Responses whose ``request_id`` does not match the active prompt are
        discarded so a stale verdict (e.g. one that arrived after a cancel)
        cannot auto-resolve an unrelated permission request.
        """
        while True:
            if self._cancel:
                self._active_permission_id = ""
                return False
            try:
                msg = self._permission_queue.get(timeout=0.1)
                params = msg.get("params", {})
                request_id = params.get("request_id", "")
                if self._active_permission_id and request_id and request_id != self._active_permission_id:
                    # Stale response for a previous prompt — drop it and keep waiting.
                    continue
                self._active_permission_id = ""
                response = params.get("response")
                if response is not None:
                    return response in ("approve", "approve_for_session")
                return params.get("granted", False)
            except queue.Empty:
                continue

    def _ask_question(self, question: str) -> str:
        """Synchronously prompt the user for an answer; returns ``"[cancelled]"`` on cancel."""
        # Drop any answer left over from a previous prompt before arming a new one.
        self._drain_queue(self._question_queue)
        if self._wire_mode:
            self._active_question_id = self._emit_wire_question_request(
                [
                    {
                        "id": str(uuid.uuid4()),
                        "question": question,
                        "options": [],
                        "allow_free_form": True,
                    }
                ]
            )
        else:
            self._active_question_id = str(uuid.uuid4())
            self._emit("question_request", {"id": self._active_question_id, "question": question})
        return self._wait_for_question_response()

    def _ask_free_form(self, question: str) -> str:
        """Wire-protocol free-form prompt; returns the user's answer or ``"[cancelled]"``."""
        self._drain_queue(self._question_queue)
        self._active_question_id = self._emit_wire_question_request(
            [
                {
                    "id": str(uuid.uuid4()),
                    "question": question,
                    "options": [],
                    "allow_free_form": True,
                }
            ]
        )
        return self._wait_for_question_response()

    def _provider_create_interactive(self) -> str:
        """Walk the user through the wire-mode ``/provider`` profile wizard."""
        name = self._ask_free_form("Profile name (e.g. 'kimi', 'openai-prod')").strip()
        if not name or name == "[cancelled]":
            return "Cancelled."
        if any(p.get("name") == name for p in profiles.list_profiles()):
            return f"Profile '{name}' already exists. Cancelled."

        base_url = self._ask_free_form("Base URL (e.g. https://api.openai.com/v1)").strip()
        if not base_url or base_url == "[cancelled]":
            return "Cancelled."

        api_key = self._ask_free_form("API key (leave blank if none)").strip()
        if api_key == "[cancelled]":
            return "Cancelled."

        model = self._ask_free_form("Default model name").strip()
        if not model or model == "[cancelled]":
            return "Cancelled."

        self.handle_provider_save({"name": name, "base_url": base_url, "api_key": api_key, "model": model})
        return f"Created and switched to profile '{name}'  (model: {model})"

    def _wait_for_question_response(self) -> str:
        """Block the query thread until the TUI returns an answer.

        Responses whose ``request_id`` does not match the active prompt are
        discarded so a stale answer (e.g. one that arrived after a cancel)
        cannot be consumed as the answer to an unrelated question.
        """
        while True:
            if self._cancel:
                self._active_question_id = ""
                return "[cancelled]"
            try:
                msg = self._question_queue.get(timeout=0.1)
                params = msg.get("params", {})
                request_id = params.get("request_id", "")
                if self._active_question_id and request_id and request_id != self._active_question_id:
                    # Stale response for a previous prompt — drop it and keep waiting.
                    continue
                self._active_question_id = ""
                answers = params.get("answers")
                if isinstance(answers, dict):
                    return "\n".join(str(v) for v in answers.values())
                return params.get("answer", "")
            except queue.Empty:
                continue

    def handle_question_response(self, params: dict[str, Any]) -> None:
        """Enqueue a ``question_response`` so the blocked query thread can wake."""
        self._question_queue.put({"params": params})

    def handle_cancel(self) -> None:
        """Request cancellation of the in-flight query and cancel running sub-agents."""
        self._cancel = True
        try:
            from ..tools.claude_tools import _get_agent_manager

            _get_agent_manager().cancel_all()
        except Exception as exc:
            logger.warning("Failed to cancel sub-agents: %s", exc)

    def handle_cancel_all(self) -> None:
        """Cancel the active query plus every running sub-agent, surfacing the count."""
        self._cancel = True
        try:
            from ..tools.claude_tools import _get_agent_manager

            mgr = _get_agent_manager()
            n = mgr.cancel_all()
            if n:
                self._emit("slash_result", {"output": f"Cancelled {n} running sub-agent(s)."})
        except Exception as exc:
            logger.warning("Failed to cancel sub-agents: %s", exc)

    def handle_provider_list(self) -> None:
        """Emit ``provider_list`` carrying every stored profile."""
        plist = profiles.list_profiles()
        self._emit("provider_list", {"profiles": plist})

    def handle_fetch_models(self, params: dict[str, Any]) -> None:
        """Emit ``models_list`` for ``params["base_url"]`` using :func:`profiles.fetch_models`."""
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

    def _switch_model(self, model: str, *, persist_active_profile: bool = True) -> None:
        """Update the runtime model, optionally persist it on the active profile, refresh init."""
        self.config["model"] = model
        if persist_active_profile:
            profiles.update_active_model(model)
        set_global_config(self.config)
        self._emit("model_changed", {"model": model, "provider": detect_provider(model)})
        if self._wire_mode and self._initialized:
            self._emit_wire_init_done()
            self._emit_wire_status()

    def _auto_switch_stale_model(self, available: list[str]) -> str:
        """If the provider returns exactly one model and ours isn't in the list, switch to it.

        Returns the newly-active model id, or empty string when no switch occurred.
        """
        current = self.config.get("model", "")
        if len(available) != 1:
            return ""
        model = available[0]
        if current == model:
            return ""
        if current and current in available:
            return ""
        self._switch_model(model)
        return model

    def handle_provider_save(self, params: dict[str, Any]) -> None:
        """Persist a profile from ``params`` and activate it; emits ``provider_saved`` on success."""
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
        self._emit("model_changed", {"model": model, "provider": detect_provider(model)})
        if self._wire_mode and self._initialized:
            self._emit_wire_init_done()
            self._emit_wire_status()

        self._emit(
            "provider_saved",
            {
                "profile": profile,
                "message": f"Profile '{name}' saved and activated. Model: {model}",
            },
        )

    def handle_provider_select(self, params: dict[str, Any]) -> None:
        """Activate the profile named ``params["name"]`` and re-publish runtime config."""
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
        """Delete the named provider profile via :func:`profiles.delete_profile`."""
        name = params.get("name", "")
        if profiles.delete_profile(name):
            self._emit("slash_result", {"output": f"Profile '{name}' deleted."})
        else:
            self._emit_error(f"Profile '{name}' not found")

    def handle_slash(self, params: dict[str, Any]) -> None:
        """Dispatch ``params["command"]`` to the matching slash handler."""
        command = params.get("command", "").strip()
        if not command.startswith("/"):
            self._emit_error(f"Not a slash command: {command}")
            return

        # A slash command (e.g. interactive ``/provider``) may raise its own
        # question prompts; clear any answer left over from a previous turn so
        # it cannot be consumed as the answer to a fresh prompt.
        self._active_permission_id = ""
        self._active_question_id = ""
        self._drain_queue(self._permission_queue)
        self._drain_queue(self._question_queue)

        parts = command[1:].split(None, 1)
        cmd = parts[0].lower() if parts else ""
        args = parts[1] if len(parts) > 1 else ""

        self._pending_skill_name = ""

        if self._wire_mode:
            if cmd in ("btw", "plan", "steer"):
                self._run_wire_slash(cmd, args)
                return

        if cmd == "skill":
            self._handle_skill_invoke(args)
            return

        output = self._run_slash(cmd, args)
        if output:
            self._emit("slash_result", {"output": output})
        # Refresh the status bar for commands that change footer-visible state.
        if self._wire_mode and cmd in {"thinking", "reasoning", "mode", "plan"}:
            self._emit_wire_status()

    def _parse_json_messages(self, line: str) -> list[dict[str, Any]]:
        """Parse one or more JSON objects from one stdin line (fall back to whitespace-split)."""
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
        """Read JSON-RPC requests from stdin until shutdown, dispatching each method."""
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
                    if method in ("init", "initialize"):
                        self.handle_init(params)
                    elif method in ("query", "prompt"):
                        if self._query_thread is not None and self._query_thread.is_alive():
                            self._emit_error("A query is already running. Wait or send cancel.")
                        else:

                            def _run_query(_params: dict[str, Any]) -> None:
                                """Worker-thread body: run :meth:`handle_query` and surface any error."""
                                try:
                                    if "user_input" in _params and "text" not in _params:
                                        _params = {**_params, "text": _params.get("user_input", "")}
                                    self.handle_query(_params)
                                except Exception as exc:
                                    self._emit_error(f"{type(exc).__name__}: {exc}")

                                    if self._wire_mode:
                                        self._emit_wire_turn_end()
                                        self._emit_wire_status()
                                    self._emit("query_done", {})

                            self._query_thread = threading.Thread(target=_run_query, args=(params,), daemon=True)
                            self._query_thread.start()
                    elif method == "permission_response":
                        self._permission_queue.put(msg)
                    elif method == "question_response":
                        self.handle_question_response(params)
                    elif method == "steer":
                        content = params.get("user_input", params.get("content", ""))
                        self._run_wire_slash("steer", str(content))
                    elif method == "set_plan_mode":
                        enabled = bool(params.get("enabled", params.get("plan_mode", False)))
                        self.config["plan_mode"] = enabled
                        self.config["mode"] = _normalize_interaction_mode(
                            params.get("mode", self.config.get("mode")), plan_mode=enabled
                        )
                        set_global_config(self.config)
                        if self._wire_mode:
                            self._emit_wire_status()
                    elif method == "set_mode":
                        mode = _normalize_interaction_mode(params.get("mode"), plan_mode=False)
                        self.config["mode"] = mode
                        self.config["plan_mode"] = mode == "plan"
                        set_global_config(self.config)
                        if self._wire_mode:
                            self._emit_wire_status()
                    elif method == "replay":
                        if self._wire_mode:
                            self._emit_wire_notification(
                                notification_id=str(uuid.uuid4()),
                                category="session",
                                type_="replay_unavailable",
                                severity="warning",
                                title="Replay unavailable",
                                body="This Xerxes bridge does not persist Kimi wire replay records yet.",
                            )
                    elif method == "cancel":
                        self.handle_cancel()
                    elif method == "cancel_all":
                        self.handle_cancel_all()
                    elif method == "slash":
                        if self._query_thread is not None and self._query_thread.is_alive():
                            self._emit_error("A query is already running. Wait or send cancel.")
                        else:

                            def _run_slash(_params: dict[str, Any]) -> None:
                                """Worker-thread body: run :meth:`handle_slash` and surface any error."""
                                try:
                                    self.handle_slash(_params)
                                except Exception as exc:
                                    self._emit_error(f"{type(exc).__name__}: {exc}")
                                    if self._wire_mode:
                                        self._emit_wire_status()
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
    """Console-script entry point: parse flags, build :class:`BridgeServer`, run it on stdin/stdout."""
    import signal

    signal.signal(signal.SIGINT, signal.SIG_IGN)

    parser = argparse.ArgumentParser(description="Xerxes bridge server (JSON-RPC over stdio)")
    parser.add_argument("-m", "--model", default=None, help="Model name")
    parser.add_argument("--base-url", default=None, help="API base URL")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument(
        "--wire",
        action="store_true",
        help=("Use Kimi-Code-compatible wire protocol (JSON-RPC 2.0 NDJSON) instead of the legacy flat-event protocol."),
    )
    args = parser.parse_args()

    server = BridgeServer(wire_mode=args.wire)

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
