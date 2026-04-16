# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
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

"""Operator runtime state and tool construction.

Provides :class:`OperatorState`, the central composition root for the
operator subsystem.  It owns the PTY, browser, plan, user-prompt, and
sub-agent managers and exposes a :meth:`~OperatorState.build_tools`
method that returns all operator tool callables ready for registration
in the Xerxes runtime.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import re
import typing as tp
from datetime import datetime, timedelta

import httpx
from PIL import Image

from ..tools.google_search import GoogleSearch
from ..types.messages import ImageChunk, TextChunk, UserMessage
from .browser import BrowserManager
from .config import HIGH_POWER_OPERATOR_TOOLS, OperatorRuntimeConfig
from .helpers import operator_tool
from .plans import PlanStateManager
from .pty import PTYSessionManager
from .subagents import SpawnedAgentManager
from .types import ImageInspectionResult
from .user_prompt import UserPromptManager


class OperatorState:
    """Own runtime managers and expose operator tool callables.

    Acts as the single entry point for the operator subsystem.  On
    construction it creates all sub-managers (PTY, browser, plan,
    user-prompt) and defers sub-agent manager creation until the
    runtime is attached via :meth:`attach_runtime`.

    Attributes:
        config: The :class:`OperatorRuntimeConfig` governing this
            operator instance.
        pty_manager: Manages persistent PTY shell sessions.
        browser_manager: Manages Playwright browser pages.
        plan_manager: Manages the structured execution plan.
        user_prompt_manager: Manages user clarification questions.
        xerxes: Reference to the parent Xerxes instance, set by
            :meth:`attach_runtime`.
        runtime_state: Reference to the shared runtime state, set by
            :meth:`attach_runtime`.
        subagent_manager: Manages spawned background sub-agents.
            ``None`` until :meth:`attach_runtime` is called.
    """

    def __init__(self, config: OperatorRuntimeConfig) -> None:
        """Initialise the operator state with the given configuration.

        Creates all sub-managers except the sub-agent manager, which
        requires a live Xerxes instance and is deferred to
        :meth:`attach_runtime`.

        Args:
            config: Operator runtime configuration controlling browser
                headless mode, screenshot directory, shell defaults,
                and tool allowlists.
        """
        self.config = config
        self.pty_manager = PTYSessionManager()
        self.browser_manager = BrowserManager(
            headless=config.browser_headless,
            screenshot_dir=config.browser_screenshot_dir,
        )
        self.plan_manager = PlanStateManager()
        self.user_prompt_manager = UserPromptManager()
        self.xerxes: tp.Any = None
        self.runtime_state: tp.Any = None
        self.subagent_manager: SpawnedAgentManager | None = None
        self._tool_cache: list[tp.Callable] | None = None

    def attach_runtime(self, xerxes: tp.Any, runtime_state: tp.Any) -> None:
        """Bind the operator runtime to a concrete Xerxes instance.

        This must be called before any sub-agent tools can be used.
        It stores references to the parent runtime and creates the
        :class:`SpawnedAgentManager`.

        Args:
            xerxes: The parent :class:`Xerxes` instance.
            runtime_state: The shared runtime state object that holds
                policy, configuration, and audit emitter references.
        """
        self.xerxes = xerxes
        self.runtime_state = runtime_state
        self.subagent_manager = SpawnedAgentManager(xerxes, runtime_state)

    def set_power_tools_enabled(self, enabled: bool) -> None:
        """Update the power-tools flag and effective optional-tools policy.

        When enabling, removes high-power tools from the policy
        engine's ``optional_tools`` set so they become available.
        When disabling, adds them back so the policy engine blocks
        them.

        Args:
            enabled: ``True`` to activate high-power tools,
                ``False`` to deactivate them.
        """
        self.config.power_tools_enabled = enabled
        if self.runtime_state is None:
            return
        policy = self.runtime_state.policy_engine.global_policy
        if enabled:
            policy.optional_tools.difference_update(HIGH_POWER_OPERATOR_TOOLS)
        else:
            policy.optional_tools.update(HIGH_POWER_OPERATOR_TOOLS)

    def list_operator_state(self) -> dict[str, tp.Any]:
        """Return summaries for operator-managed runtime state.

        Aggregates status from every sub-manager into a single
        dictionary suitable for TUI display or API responses.

        Returns:
            A dictionary containing:

            - ``power_tools_enabled``: Current power-tools flag.
            - ``pty_sessions``: List of PTY session summaries.
            - ``browser_pages``: List of tracked browser page summaries.
            - ``spawned_agents``: List of sub-agent handle snapshots.
            - ``plan``: Current plan state dictionary.
            - ``pending_user_prompt``: Pending question dictionary, or
              ``None``.
        """
        return {
            "power_tools_enabled": self.config.power_tools_enabled,
            "pty_sessions": self.pty_manager.list_sessions(),
            "browser_pages": self.browser_manager.list_pages(),
            "spawned_agents": self.subagent_manager.list_handles() if self.subagent_manager else [],
            "plan": self.plan_manager.state.to_dict(),
            "pending_user_prompt": self.user_prompt_manager.get_pending(),
        }

    def build_tools(self) -> list[tp.Callable]:
        """Build and cache the operator tool functions.

        On first call, instantiates all operator tool closures and
        caches the list.  Subsequent calls return a shallow copy of
        the cached list.

        Returns:
            A list of callable operator tool functions, each decorated
            with :func:`operator_tool` metadata.
        """
        if self._tool_cache is None:
            self._tool_cache = [
                self._build_exec_command(),
                self._build_write_stdin(),
                self._build_apply_patch(),
                self._build_spawn_agent(),
                self._build_resume_agent(),
                self._build_send_input(),
                self._build_wait_agent(),
                self._build_close_agent(),
                self._build_ask_user(),
                self._build_view_image(),
                self._build_update_plan(),
                self._build_web_search_query(),
                self._build_web_image_query(),
                self._build_web_open(),
                self._build_web_click(),
                self._build_web_find(),
                self._build_web_screenshot(),
                self._build_web_weather(),
                self._build_web_finance(),
                self._build_web_sports(),
                self._build_web_time(),
            ]
        return list(self._tool_cache)

    @staticmethod
    def _validate_patch_text(patch: str) -> None:
        """Reject clearly malformed patch payloads before calling git apply.

        Checks that the patch text contains unified diff headers
        (``---``/``+++`` or ``diff --git``) and at least one ``@@``
        hunk marker.

        Args:
            patch: The raw unified diff text to validate.

        Raises:
            ValueError: If the patch is empty, lacks headers, or has
                no ``@@`` hunk markers.
        """
        text = patch.strip()
        if not text:
            raise ValueError("Patch text must be non-empty")

        has_headers = ("--- " in text and "+++ " in text) or "diff --git " in text
        has_hunks = bool(re.search(r"(?m)^@@ ", text))
        if not has_headers or not has_hunks:
            raise ValueError("Patch must be a unified diff with ---/+++ headers and @@ hunks")

    def create_reinvoke_message(self, result: tp.Any) -> UserMessage | None:
        """Convert special operator tool results into a reinvocation message.

        When the tool result is an :class:`ImageInspectionResult`, a
        multimodal :class:`UserMessage` is constructed containing both
        a text summary and the image data, enabling the LLM to inspect
        the image in the next turn.

        Args:
            result: The raw tool result to inspect.

        Returns:
            A :class:`UserMessage` with text and image chunks when the
            result is an :class:`ImageInspectionResult`, or ``None``
            for all other result types.
        """
        if isinstance(result, ImageInspectionResult):
            image = result.image.copy()
            return UserMessage(
                content=[
                    TextChunk(text=f"[TOOL IMAGE RESULT] {result.summary()}"),
                    ImageChunk(image=image),
                ]
            )
        return None

    def summarize_result(self, result: tp.Any) -> tuple[tp.Any, dict[str, tp.Any]]:
        """Return a tool-message-safe result plus serialisable metadata.

        For :class:`ImageInspectionResult`, the heavy PIL image is
        replaced by a compact text summary in the first element, and
        the metadata dictionary contains all scalar fields.

        Args:
            result: The raw tool result to summarise.

        Returns:
            A two-tuple of ``(safe_result, metadata)`` where
            *safe_result* is suitable for inclusion in a tool-call
            message and *metadata* is a JSON-serialisable dictionary
            (empty for non-image results).
        """
        if isinstance(result, ImageInspectionResult):
            return result.summary(), result.tool_metadata()
        return result, {}

    def _build_exec_command(self) -> tp.Callable:
        """Build the ``exec_command`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            starts a persistent PTY session.
        """

        @operator_tool(
            "exec_command",
            description=(
                "Start a persistent PTY-backed shell session that stays alive across calls. "
                "Use it for interactive commands, REPLs, long-running builds, or anything that "
                "needs follow-up input through write_stdin."
            ),
        )
        async def exec_command(
            cmd: str,
            workdir: str | None = None,
            yield_time_ms: int | None = None,
            max_output_chars: int | None = None,
            login: bool = True,
        ) -> dict[str, tp.Any]:
            """Start an interactive terminal session and return its session metadata.

            Args:
                cmd: Shell command to launch inside the PTY.  This can
                    be a shell, REPL, long-running script, or one-shot
                    command that you may want to continue interacting
                    with later.
                workdir: Working directory for the new session.  If
                    omitted, the operator runtime default working
                    directory is used.
                yield_time_ms: How long to wait before collecting
                    initial output.  Higher values are useful for
                    commands that need time to render a prompt or
                    produce startup output.
                max_output_chars: Maximum number of characters captured
                    from the initial output chunk.
                login: When ``True``, start the shell with login
                    semantics so normal shell initialisation files run.

            Returns:
                A dictionary that includes a stable ``session_id``,
                startup output, exit state if the process finished
                quickly, and runtime metadata needed by
                ``write_stdin``.
            """
            return await asyncio.to_thread(
                self.pty_manager.create_session,
                cmd,
                workdir=workdir or self.config.shell_default_workdir,
                yield_time_ms=yield_time_ms or self.config.shell_default_yield_ms,
                max_output_chars=max_output_chars or self.config.shell_default_max_output_chars,
                login=login,
            )

        return exec_command

    def _build_write_stdin(self) -> tp.Callable:
        """Build the ``write_stdin`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            writes to a live PTY session.
        """

        @operator_tool(
            "write_stdin",
            description=(
                "Send text, EOF, or an interrupt signal to a live PTY session and read back the "
                "next chunk of output. Use this after exec_command to drive interactive programs."
            ),
        )
        async def write_stdin(
            session_id: str,
            chars: str = "",
            yield_time_ms: int | None = None,
            max_output_chars: int | None = None,
            close_stdin: bool = False,
            interrupt: bool = False,
        ) -> dict[str, tp.Any]:
            """Continue interacting with an existing PTY session.

            Args:
                session_id: Identifier returned by ``exec_command`` for
                    the live session you want to continue.
                chars: Text to send to the process stdin.  Include
                    ``\\n`` when the program expects Enter to be
                    pressed.
                yield_time_ms: How long to wait before reading the next
                    output chunk after sending input.
                max_output_chars: Maximum number of output characters
                    to collect from this interaction.
                close_stdin: When ``True``, close the session stdin
                    after sending any provided text.  Useful for
                    programs waiting on EOF.
                interrupt: When ``True``, send an interrupt signal to
                    the running process before reading output.

            Returns:
                The latest session output plus status fields such as
                whether the process is still running or has exited.
            """
            return await asyncio.to_thread(
                self.pty_manager.write,
                session_id,
                chars=chars,
                close_stdin=close_stdin,
                interrupt=interrupt,
                yield_time_ms=yield_time_ms or self.config.shell_default_yield_ms,
                max_output_chars=max_output_chars or self.config.shell_default_max_output_chars,
            )

        return write_stdin

    def _build_apply_patch(self) -> tp.Callable:
        """Build the ``apply_patch`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            applies a unified diff via ``git apply``.
        """

        @operator_tool(
            "apply_patch",
            description=(
                "Apply a unified diff directly to the current working tree through git apply. "
                "Use it for structured code edits when you already know the exact patch to make."
            ),
        )
        def apply_patch(patch: str, check: bool = False, workdir: str | None = None) -> dict[str, tp.Any]:
            """Validate and apply a unified patch to the repository worktree.

            Args:
                patch: Full unified diff text, including file headers
                    and at least one ``@@`` hunk.  Malformed patches
                    are rejected before git is called.
                check: When ``True``, validate the patch with
                    ``git apply --check`` without modifying files.
                workdir: Directory in which ``git apply`` should run.
                    Defaults to the current process working directory.

            Returns:
                Result metadata containing whether the patch was
                applied or only checked, the effective working
                directory, and any stdout emitted by git.

            Raises:
                ValueError: If the patch text is malformed.
                RuntimeError: If ``git apply`` exits with a non-zero
                    return code.
            """
            import subprocess

            self._validate_patch_text(patch)
            resolved_workdir = os.path.abspath(workdir or os.getcwd())
            args = ["git", "apply"]
            if check:
                args.append("--check")
            proc = subprocess.run(
                args,
                input=patch,
                text=True,
                cwd=resolved_workdir,
                capture_output=True,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or "git apply failed")
            return {
                "applied": not check,
                "checked": check,
                "workdir": resolved_workdir,
                "stdout": proc.stdout,
            }

        return apply_patch

    def _build_spawn_agent(self) -> tp.Callable:
        """Build the ``spawn_agent`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            creates a managed background sub-agent.
        """

        @operator_tool(
            "spawn_agent",
            description=(
                "Create a managed background sub-agent handle and optionally give it a task "
                "immediately. Use this when work should continue in parallel with the current agent."
            ),
        )
        async def spawn_agent(
            message: str | None = None,
            task_description: str | None = None,
            agent_id: str | None = None,
            prompt_profile: str | None = None,
            nickname: str | None = None,
        ) -> dict[str, tp.Any]:
            """Spawn a background Xerxes sub-agent.

            Args:
                message: Optional initial task or instruction for the
                    spawned agent to start working on immediately.
                task_description: Backward-compatible alias for
                    ``message`` accepted by older tool callers.
                agent_id: Specific registered agent ID to use.  If
                    omitted, the runtime chooses the current/default
                    agent behaviour.
                prompt_profile: Prompt profile override for the spawned
                    agent.  If omitted, the operator runtime default
                    for sub-agents is used.
                nickname: Optional human-readable label to make later
                    references easier.

            Returns:
                Handle metadata including the spawned agent ID,
                current status, and any task-start information.

            Raises:
                RuntimeError: If the sub-agent manager has not been
                    initialised via :meth:`attach_runtime`.
            """
            if self.subagent_manager is None:
                raise RuntimeError("Sub-agent manager is not available")
            return await self.subagent_manager.spawn(
                message=message,
                task_description=task_description,
                agent_id=agent_id,
                prompt_profile=prompt_profile,
                nickname=nickname,
            )

        return spawn_agent

    def _build_resume_agent(self) -> tp.Callable:
        """Build the ``resume_agent`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            resumes a previously closed sub-agent handle.
        """

        @operator_tool(
            "resume_agent",
            description=(
                "Reopen a previously closed spawned-agent handle so it can receive more input or be waited on again."
            ),
        )
        def resume_agent(id: str) -> dict[str, tp.Any]:  # noqa: A002
            """Resume a previously closed sub-agent handle.

            Args:
                id: Handle identifier returned by ``spawn_agent``.

            Returns:
                Updated handle metadata showing that the handle is
                active again.

            Raises:
                RuntimeError: If the sub-agent manager is not
                    available.
                ValueError: If the handle ID is not found.
            """
            if self.subagent_manager is None:
                raise RuntimeError("Sub-agent manager is not available")
            return self.subagent_manager.resume(id)

        return resume_agent

    def _build_send_input(self) -> tp.Callable:
        """Build the ``send_input`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            delivers instructions to a spawned agent.
        """

        @operator_tool(
            "send_input",
            description=(
                "Send more work to an existing spawned agent, either queued behind current work "
                "or as an immediate interrupt."
            ),
        )
        async def send_input(
            target: str | None = None,
            message: str | None = None,
            interrupt: bool = False,
            id: str | None = None,  # noqa: A002
            handle_id: str | None = None,
            task_description: str | None = None,
        ) -> dict[str, tp.Any]:
            """Deliver a new instruction to a spawned agent.

            Args:
                target: Spawned-agent handle ID that should receive
                    the input. When omitted, the most recently updated
                    non-closed handle is used.
                message: Text instruction or follow-up task for the
                    spawned agent.
                interrupt: When ``True``, stop the agent's current
                    task and handle this message immediately.  When
                    ``False``, queue the message behind current work.
                id: Backward-compatible alias for ``target``.
                handle_id: Backward-compatible alias for ``target``.
                task_description: Backward-compatible alias for
                    ``message`` accepted by older tool callers.

            Returns:
                Delivery metadata including the target handle and
                updated status.

            Raises:
                RuntimeError: If the sub-agent manager is not
                    available.
                ValueError: If the handle is closed or not found.
            """
            if self.subagent_manager is None:
                raise RuntimeError("Sub-agent manager is not available")
            resolved_target = target or id or handle_id
            return await self.subagent_manager.send_input(
                resolved_target,
                message=message,
                task_description=task_description,
                interrupt=interrupt,
            )

        return send_input

    def _build_wait_agent(self) -> tp.Callable:
        """Build the ``wait_agent`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            waits for spawned agents to reach a terminal state.
        """

        @operator_tool(
            "wait_agent",
            description=("Wait for one or more spawned agents to reach a terminal state or until a timeout expires."),
        )
        async def wait_agent(targets: list[str], timeout_ms: int = 30000) -> dict[str, tp.Any]:
            """Wait for spawned agents to finish.

            Args:
                targets: One or more spawned-agent handle IDs to watch.
                timeout_ms: Maximum time to wait in milliseconds before
                    returning a timeout-style result.

            Returns:
                Completion or timeout data for the requested handles,
                including any final message when available.

            Raises:
                RuntimeError: If the sub-agent manager is not
                    available.
                ValueError: If any target handle ID is not found.
            """
            if self.subagent_manager is None:
                raise RuntimeError("Sub-agent manager is not available")
            return await self.subagent_manager.wait(targets, timeout_ms=timeout_ms)

        return wait_agent

    def _build_close_agent(self) -> tp.Callable:
        """Build the ``close_agent`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            closes a spawned-agent handle.
        """

        @operator_tool(
            "close_agent",
            description=(
                "Close a spawned-agent handle and cancel any running task tied to it. "
                "Use this to clean up background agents that are no longer needed."
            ),
        )
        def close_agent(target: str) -> dict[str, tp.Any]:
            """Close a spawned-agent handle.

            Args:
                target: Spawned-agent handle ID to close.

            Returns:
                Final handle status after shutdown/cancellation.

            Raises:
                RuntimeError: If the sub-agent manager is not
                    available.
                ValueError: If the handle ID is not found.
            """
            if self.subagent_manager is None:
                raise RuntimeError("Sub-agent manager is not available")
            return self.subagent_manager.close(target)

        return close_agent

    def _build_ask_user(self) -> tp.Callable:
        """Build the ``ask_user`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            pauses the run to ask the user a question.
        """

        @operator_tool(
            "ask_user",
            description=(
                "Pause the current run and ask the human user a direct clarification question, "
                "optionally with numbered choices. Use it when the next action depends on a decision "
                "the model cannot safely infer."
            ),
        )
        async def ask_user(
            question: str,
            options: list[str] | None = None,
            allow_freeform: bool = True,
            placeholder: str | None = None,
        ) -> dict[str, tp.Any]:
            """Request clarification from the live UI and wait for the answer.

            Args:
                question: The exact question to show to the user.
                options: Optional list of suggested choices.  The UI
                    renders these as numbered options that the user can
                    choose by number or by typing the full label.
                allow_freeform: When ``True``, the user may type a
                    custom answer instead of selecting one of the
                    provided options.
                placeholder: Optional input hint shown in the terminal
                    UI while waiting for the answer.

            Returns:
                The resolved answer payload, including the raw input,
                normalised answer text, and matched option when one
                was selected.
            """
            return await self.user_prompt_manager.request(
                question,
                options=options,
                allow_freeform=allow_freeform,
                placeholder=placeholder,
            )

        return ask_user

    def _build_view_image(self) -> tp.Callable:
        """Build the ``view_image`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            loads and inspects a local image file.
        """

        @operator_tool(
            "view_image",
            description=(
                "Load a local image file and pass it back as an image-capable tool result for "
                "multimodal follow-up. Use it when the model needs to inspect a real image rather "
                "than just discuss a path."
            ),
        )
        def view_image(path: str, detail: str = "auto") -> ImageInspectionResult:
            """Open a local image and prepare it for multimodal inspection.

            Args:
                path: Absolute or relative path to an image file on
                    disk.
                detail: Requested inspection detail level forwarded
                    with the tool result.  ``"auto"`` lets the runtime
                    decide.

            Returns:
                Structured metadata plus an in-memory PIL image that
                can be attached to the reinvocation message.

            Raises:
                FileNotFoundError: If the resolved path does not point
                    to an existing file.
            """
            resolved = pathlib.Path(path).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(f"Image path not found: {resolved}")
            with Image.open(resolved) as img:
                image_format = img.format
                image = img.copy()
            return ImageInspectionResult(
                path=str(resolved),
                format=image_format,
                mode=image.mode,
                width=image.width,
                height=image.height,
                image=image,
                detail=detail,
            )

        return view_image

    def _build_update_plan(self) -> tp.Callable:
        """Build the ``update_plan`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            mutates the shared execution plan.
        """

        @operator_tool(
            "update_plan",
            description=(
                "Update the current structured execution plan for this session. "
                "Use it to record steps, statuses, and a short explanation of the latest plan change."
            ),
        )
        def update_plan(explanation: str | None = None, plan: list[dict[str, str]] | None = None) -> dict[str, tp.Any]:
            """Mutate the shared structured plan state.

            Args:
                explanation: Optional short note explaining why the
                    plan changed or what the current state means.
                plan: Full list of plan steps.  Each item should
                    include ``step`` and ``status`` fields.

            Returns:
                The updated plan payload, including revision number,
                explanation, and normalised steps.
            """
            updated = self.plan_manager.update(explanation, plan or [])
            if self.runtime_state is not None and self.runtime_state.audit_emitter is not None:
                self.runtime_state.audit_emitter.emit_hook_mutation(
                    hook_name="update_plan",
                    tool_name="update_plan",
                    mutated_field="plan_state",
                )
            return updated

        return update_plan

    def _build_web_search_query(self) -> tp.Callable:
        """Build the ``web.search_query`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            performs a Google web search.
        """

        @operator_tool(
            "web.search_query",
            description=(
                "Search the public web through Google and return compact result dictionaries. "
                "Use it for up-to-date information, news, and source discovery before opening pages."
            ),
        )
        def web_search_query(
            q: str,
            search_type: str = "text",
            n_results: int = 5,
            domains: list[str] | None = None,
        ) -> dict[str, tp.Any]:
            """Run a Google search and return normalised results.

            Args:
                q: Search query text.
                search_type: Vertical hint (``text``/``news``);
                    Google search currently treats all values as text.
                n_results: Number of results to request.
                domains: Optional list of preferred domains. The first
                    domain (if any) becomes a ``site:`` restriction.

            Returns:
                ``{"query", "search_type", "results"}`` where ``results``
                is a list of ``{"title", "url", "snippet"}`` dicts.
            """
            site = domains[0] if domains else None
            time_range = "d" if search_type == "news" else None
            payload = GoogleSearch.static_call(
                query=q,
                n_results=n_results,
                site=site,
                time_range=time_range,
            )
            return {
                "query": q,
                "search_type": search_type,
                "results": payload.get("results", []),
                "engine": payload.get("engine", "google"),
            }

        return web_search_query

    def _build_web_image_query(self) -> tp.Callable:
        """Build the ``web.image_query`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            searches for image results via Google.
        """

        @operator_tool(
            "web.image_query",
            description=(
                "Search public image results through Google. Use it when visual references "
                "would help answer the task before opening or analyzing specific pages."
            ),
        )
        def web_image_query(q: str, n_results: int = 5, domains: list[str] | None = None) -> dict[str, tp.Any]:
            """Search for image-bearing pages related to a query.

            Note: Google CSE returns web pages, not raw image URLs —
            the agent must follow up with ``browser_get_images`` (or
            curl) on the result page to fetch the actual image links.

            Args:
                q: Search query text.
                n_results: Number of result pages to request.
                domains: Optional preferred domains; first becomes a
                    ``site:`` restriction.

            Returns:
                Query metadata plus normalised result pages.
            """
            site = domains[0] if domains else None
            payload = GoogleSearch.static_call(
                query=f"{q} images",
                n_results=n_results,
                site=site,
            )
            return {"query": q, "results": payload.get("results", [])}

        return web_image_query

    def _build_web_open(self) -> tp.Callable:
        """Build the ``web.open`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            opens or re-inspects a browser page.
        """

        @operator_tool(
            "web.open",
            description=(
                "Open a URL in the shared browser manager or revisit an existing tracked page by ref_id. "
                "Use it after search results when you need the actual page content."
            ),
        )
        async def web_open(url: str | None = None, ref_id: str | None = None, wait_ms: int = 500) -> dict[str, tp.Any]:
            """Open a new page or inspect an already tracked page.

            Args:
                url: URL to open in the browser.  Provide this for a
                    new page visit.
                ref_id: Existing browser page reference to revisit
                    instead of opening a new URL.
                wait_ms: Additional time to wait after navigation so
                    the page has time to settle before metadata is
                    captured.

            Returns:
                Page reference data, title, URL, and extracted browser
                metadata for follow-up actions like click or find.
            """
            return await self.browser_manager.open(url=url, ref_id=ref_id, wait_ms=wait_ms)

        return web_open

    def _build_web_click(self) -> tp.Callable:
        """Build the ``web.click`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            clicks an element on a tracked browser page.
        """

        @operator_tool(
            "web.click",
            description=(
                "Click a discovered link or DOM selector on a tracked browser page and return the updated page state."
            ),
        )
        async def web_click(
            ref_id: str,
            link_id: int | None = None,
            selector: str | None = None,
            text: str | None = None,
            wait_ms: int = 500,
        ) -> dict[str, tp.Any]:
            """Interact with a tracked browser page.

            Args:
                ref_id: Browser page reference returned by ``web.open``
                    or a previous browser action.
                link_id: Numeric link identifier from the page summary.
                    Use this when the runtime exposed clickable links
                    with IDs.
                selector: CSS selector to click directly.
                text: Fallback text match used when selector and link
                    ID are not available.
                wait_ms: Additional post-click wait time in
                    milliseconds.

            Returns:
                Updated browser page summary after the click.
            """
            return await self.browser_manager.click(
                ref_id,
                link_id=link_id,
                selector=selector,
                text=text,
                wait_ms=wait_ms,
            )

        return web_click

    def _build_web_find(self) -> tp.Callable:
        """Build the ``web.find`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            searches for text on a tracked browser page.
        """

        @operator_tool(
            "web.find",
            description=(
                "Search the visible text of a tracked browser page for a string or pattern and return matches."
            ),
        )
        async def web_find(ref_id: str, pattern: str) -> dict[str, tp.Any]:
            """Find text on a previously opened browser page.

            Args:
                ref_id: Browser page reference to inspect.
                pattern: Text or pattern to search for inside the page
                    content.

            Returns:
                Match information such as hit count and matched
                snippets or locations, depending on the browser
                manager output.
            """
            return await self.browser_manager.find(ref_id, pattern)

        return web_find

    def _build_web_screenshot(self) -> tp.Callable:
        """Build the ``web.screenshot`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            captures a screenshot of a tracked browser page.
        """

        @operator_tool(
            "web.screenshot",
            description=(
                "Capture a screenshot of a tracked browser page and save it to disk. "
                "Use it when a visual snapshot of the current page state is needed."
            ),
        )
        async def web_screenshot(ref_id: str, path: str | None = None, full_page: bool = True) -> dict[str, tp.Any]:
            """Capture and save a page screenshot.

            Args:
                ref_id: Browser page reference to capture.
                path: Optional destination path.  If omitted, the
                    browser manager chooses a default location.
                full_page: When ``True``, capture the entire page
                    instead of only the visible viewport.

            Returns:
                Screenshot metadata including the saved path and page
                reference.
            """
            return await self.browser_manager.screenshot(ref_id, path=path, full_page=full_page)

        return web_screenshot

    def _build_web_weather(self) -> tp.Callable:
        """Build the ``web.weather`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            fetches current weather data via Open-Meteo.
        """

        @operator_tool(
            "web.weather",
            description=(
                "Resolve a place name and fetch current weather data through Open-Meteo. "
                "Use it for practical local weather questions without needing general web search."
            ),
        )
        async def web_weather(location: str) -> dict[str, tp.Any]:
            """Fetch current weather data for a human-readable location string.

            Geocodes the location name via the Open-Meteo geocoding API
            and then retrieves the current forecast for the resolved
            coordinates.

            Args:
                location: City, region, or place name to geocode first.

            Returns:
                Normalised weather payload including resolved location
                metadata and current forecast values such as
                temperature, apparent temperature, and wind speed.

            Raises:
                ValueError: If the location cannot be geocoded.
                httpx.HTTPStatusError: If either upstream API returns
                    a non-2xx response.
            """
            async with httpx.AsyncClient(timeout=20) as client:
                geo = await client.get(
                    "https://geocoding-api.open-meteo.com/v1/search",
                    params={"name": location, "count": 1, "language": "en", "format": "json"},
                )
                geo.raise_for_status()
                results = geo.json().get("results") or []
                if not results:
                    raise ValueError(f"Location not found: {location}")
                place = results[0]
                forecast = await client.get(
                    "https://api.open-meteo.com/v1/forecast",
                    params={
                        "latitude": place["latitude"],
                        "longitude": place["longitude"],
                        "current": "temperature_2m,apparent_temperature,wind_speed_10m,weather_code",
                    },
                )
                forecast.raise_for_status()
                return {
                    "location": place.get("name"),
                    "country": place.get("country"),
                    "current": forecast.json().get("current", {}),
                }

        return web_weather

    def _build_web_finance(self) -> tp.Callable:
        """Build the ``web.finance`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            fetches financial quote data from Yahoo Finance.
        """

        @operator_tool(
            "web.finance",
            description=(
                "Fetch current quote data for a ticker symbol from Yahoo Finance. "
                "Use it for quick price checks and market snapshots."
            ),
        )
        async def web_finance(ticker: str, market: str | None = None, kind: str = "equity") -> dict[str, tp.Any]:
            """Fetch current quote data for a ticker symbol.

            Args:
                ticker: Symbol to look up, such as ``AAPL`` or
                    ``BTC-USD``.
                market: Optional market suffix appended to the symbol
                    for providers that expect ``TICKER.MARKET``
                    notation.
                kind: Asset kind hint such as ``equity``, ``crypto``,
                    or ``fund``.  This value is returned for context
                    but does not change the upstream request shape.

            Returns:
                Normalised quote information including current price,
                currency, price change, percent change, and the raw
                quote payload.

            Raises:
                ValueError: If no finance data is returned for the
                    symbol.
                httpx.HTTPStatusError: If Yahoo Finance returns a
                    non-2xx response.
            """
            symbol = ticker if not market else f"{ticker}.{market}"
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(
                    "https://query1.finance.yahoo.com/v7/finance/quote",
                    params={"symbols": symbol},
                )
                response.raise_for_status()
                quotes = response.json().get("quoteResponse", {}).get("result", [])
                if not quotes:
                    raise ValueError(f"No finance data returned for {symbol}")
                quote = quotes[0]
                return {
                    "ticker": ticker,
                    "kind": kind,
                    "market": market,
                    "price": quote.get("regularMarketPrice"),
                    "currency": quote.get("currency"),
                    "change": quote.get("regularMarketChange"),
                    "change_percent": quote.get("regularMarketChangePercent"),
                    "raw": quote,
                }

        return web_finance

    def _build_web_sports(self) -> tp.Callable:
        """Build the ``web.sports`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            fetches sports data from ESPN.
        """

        @operator_tool(
            "web.sports",
            description=(
                "Fetch sports schedule or standings data from ESPN for a supported league. "
                "Use it for quick scoreboard and standings lookups."
            ),
        )
        async def web_sports(
            league: str,
            fn: str = "schedule",
            team: str | None = None,
            opponent: str | None = None,
        ) -> dict[str, tp.Any]:
            """Fetch schedule or standings data for a supported sports league.

            Args:
                league: Supported league code such as ``nba``,
                    ``wnba``, ``nfl``, ``nhl``, ``mlb``, or ``epl``.
                fn: Either ``"schedule"`` or ``"standings"``.
                team: Optional team filter echoed back in the response
                    for caller-side narrowing.
                opponent: Optional opponent filter echoed back in the
                    response for caller-side narrowing.

            Returns:
                League metadata and the raw ESPN payload for the
                requested function.

            Raises:
                ValueError: If the league code is not supported.
                httpx.HTTPStatusError: If ESPN returns a non-2xx
                    response.
            """
            league_map = {
                "nba": "basketball/nba",
                "wnba": "basketball/wnba",
                "nfl": "football/nfl",
                "nhl": "hockey/nhl",
                "mlb": "baseball/mlb",
                "epl": "soccer/eng.1",
            }
            if league not in league_map:
                raise ValueError(f"Unsupported sports league: {league}")
            base = f"https://site.api.espn.com/apis/site/v2/sports/{league_map[league]}"
            path = "standings" if fn == "standings" else "scoreboard"
            async with httpx.AsyncClient(timeout=20) as client:
                response = await client.get(f"{base}/{path}")
                response.raise_for_status()
                payload = response.json()
            return {
                "league": league,
                "fn": fn,
                "team": team,
                "opponent": opponent,
                "data": payload,
            }

        return web_sports

    def _build_web_time(self) -> tp.Callable:
        """Build the ``web.time`` operator tool closure.

        Returns:
            A callable decorated with operator tool metadata that
            computes the current time for a UTC offset.
        """

        @operator_tool(
            "web.time",
            description=(
                "Return the current local time for a UTC offset without using the network. "
                "Use it for quick timezone calculations when only an offset is needed."
            ),
        )
        def web_time(utc_offset: str) -> dict[str, str]:
            """Compute the current date and time for a UTC offset.

            Parses the offset string, applies the delta to the current
            UTC time, and returns formatted date/time components.

            Args:
                utc_offset: Offset string like ``+03:00`` or
                    ``-05:00``.

            Returns:
                Normalised time fields including ISO timestamp, date,
                time, and the offset that was applied.
            """
            sign = 1 if utc_offset.startswith("+") else -1
            hours_str, minutes_str = utc_offset[1:].split(":", 1)
            delta = timedelta(hours=sign * int(hours_str), minutes=sign * int(minutes_str))
            current = datetime.utcnow() + delta
            return {
                "utc_offset": utc_offset,
                "iso": current.isoformat(timespec="seconds"),
                "time": current.strftime("%H:%M:%S"),
                "date": current.strftime("%Y-%m-%d"),
            }

        return web_time
