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
"""Aggregate operator state attached to a Xerxes session.

:class:`OperatorState` is the integration point that ties the per-session
managers (PTY shells, browser pages, structured plan, user prompts, spawned
subagents) into a single object the streaming runtime can use. It also owns
the factory methods that produce the operator tool callables registered with
the model.
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
    """Container that owns every operator-side manager for one session.

    The session lifecycle is:

    1. The session creates an :class:`OperatorState` from its
       :class:`OperatorRuntimeConfig`.
    2. Once the Xerxes runtime and streaming state are available, the host
       calls :meth:`attach_runtime` to wire in subagent support and audit
       emission.
    3. The session asks for the operator tool list via :meth:`build_tools`
       and merges it into the model tool registry.

    The class owns no I/O of its own — each manager attribute is responsible
    for its own resources (PTY processes, browser contexts, subagent
    handles).
    """

    def __init__(self, config: OperatorRuntimeConfig) -> None:
        """Construct managers and remember the runtime configuration.

        Args:
            config: Session-scoped operator configuration; consumed by the
                browser and PTY managers and retained for tool factories.
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
        """Bind a live Xerxes runtime so subagent spawning can work.

        Called once per session after the streaming runtime has been
        constructed. After this point :meth:`build_tools` can return
        ``spawn_agent`` and the other subagent control tools.
        """

        self.xerxes = xerxes
        self.runtime_state = runtime_state
        self.subagent_manager = SpawnedAgentManager(xerxes, runtime_state)

    def set_power_tools_enabled(self, enabled: bool) -> None:
        """Toggle the high-power operator tool group at runtime.

        When disabled, every name in :data:`HIGH_POWER_OPERATOR_TOOLS` is
        added to the policy engine's optional-tool set so the streaming
        runtime hides them from the model. Re-enabling removes the gating
        again. A no-op until :meth:`attach_runtime` has been called.
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
        """Return a JSON-safe snapshot of every owned manager.

        Used by the bridge to power introspection endpoints and the TUI
        status panels.
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
        """Return the ordered list of operator tools for the streaming loop.

        Results are cached: the first invocation instantiates each tool and
        subsequent calls return a fresh copy of the cached list (so callers
        can mutate the list without disturbing the cache).
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
        """Reject patch text that is not a real unified diff."""

        text = patch.strip()
        if not text:
            raise ValueError("Patch text must be non-empty")

        has_headers = ("--- " in text and "+++ " in text) or "diff --git " in text
        has_hunks = bool(re.search(r"(?m)^@@ ", text))
        if not has_headers or not has_hunks:
            raise ValueError("Patch must be a unified diff with ---/+++ headers and @@ hunks")

    def create_reinvoke_message(self, result: tp.Any) -> UserMessage | None:
        """Build a follow-up multimodal message for tool results that need it.

        Currently fires only for :class:`ImageInspectionResult`: the image
        returned by ``view_image`` is wrapped in a synthetic user message so
        the next model turn can actually see it. Returns ``None`` when no
        re-invocation is needed.
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
        """Reduce a tool result to a wire-safe value plus structured metadata.

        Most results pass through unchanged with an empty metadata dict.
        :class:`ImageInspectionResult` is collapsed to its summary string and
        the JSON-safe metadata payload.

        Returns:
            ``(display_value, metadata)`` — ``display_value`` is what gets
            shown to the model; ``metadata`` is attached to the tool
            envelope for downstream consumers.
        """

        if isinstance(result, ImageInspectionResult):
            return result.summary(), result.tool_metadata()
        return result, {}

    def _build_exec_command(self) -> tp.Callable:
        """Factory for the ``exec_command`` PTY tool."""

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
            """Spawn a new PTY session and stream its first chunk of output."""

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
        """Factory for the ``write_stdin`` PTY interaction tool."""

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
            """Write to an existing PTY session and return any new output."""

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
        """Factory for the ``apply_patch`` git-apply tool."""

        @operator_tool(
            "apply_patch",
            description=(
                "Apply a unified diff directly to the current working tree through git apply. "
                "Use it for structured code edits when you already know the exact patch to make."
            ),
        )
        def apply_patch(patch: str, check: bool = False, workdir: str | None = None) -> dict[str, tp.Any]:
            """Validate and shell out to ``git apply`` (optionally ``--check``)."""

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
        """Factory for the ``spawn_agent`` subagent-creation tool."""

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
            """Delegate to :meth:`SpawnedAgentManager.spawn`."""

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
        """Factory for the ``resume_agent`` reattach tool."""

        @operator_tool(
            "resume_agent",
            description=(
                "Reopen a previously closed spawned-agent handle so it can receive more input or be waited on again."
            ),
        )
        def resume_agent(agent_id: str) -> dict[str, tp.Any]:
            """Delegate to :meth:`SpawnedAgentManager.resume`."""

            if self.subagent_manager is None:
                raise RuntimeError("Sub-agent manager is not available")
            return self.subagent_manager.resume(agent_id)

        return resume_agent

    def _build_send_input(self) -> tp.Callable:
        """Factory for the ``send_input`` subagent-messaging tool."""

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
            agent_id: str | None = None,
            handle_id: str | None = None,
            task_description: str | None = None,
        ) -> dict[str, tp.Any]:
            """Resolve the target alias and delegate to the subagent manager."""

            if self.subagent_manager is None:
                raise RuntimeError("Sub-agent manager is not available")
            resolved_target = target or agent_id or handle_id
            return await self.subagent_manager.send_input(
                resolved_target,
                message=message,
                task_description=task_description,
                interrupt=interrupt,
            )

        return send_input

    def _build_wait_agent(self) -> tp.Callable:
        """Factory for the ``wait_agent`` join tool."""

        @operator_tool(
            "wait_agent",
            description=("Wait for one or more spawned agents to reach a terminal state or until a timeout expires."),
        )
        async def wait_agent(targets: list[str], timeout_ms: int = 30000) -> dict[str, tp.Any]:
            """Delegate to :meth:`SpawnedAgentManager.wait`."""

            if self.subagent_manager is None:
                raise RuntimeError("Sub-agent manager is not available")
            return await self.subagent_manager.wait(targets, timeout_ms=timeout_ms)

        return wait_agent

    def _build_close_agent(self) -> tp.Callable:
        """Factory for the ``close_agent`` teardown tool."""

        @operator_tool(
            "close_agent",
            description=(
                "Close a spawned-agent handle and cancel any running task tied to it. "
                "Use this to clean up background agents that are no longer needed."
            ),
        )
        def close_agent(target: str) -> dict[str, tp.Any]:
            """Delegate to :meth:`SpawnedAgentManager.close`."""

            if self.subagent_manager is None:
                raise RuntimeError("Sub-agent manager is not available")
            return self.subagent_manager.close(target)

        return close_agent

    def _build_ask_user(self) -> tp.Callable:
        """Factory for the ``ask_user`` human-in-the-loop tool."""

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
            """Queue a prompt with :class:`UserPromptManager` and await reply."""

            return await self.user_prompt_manager.request(
                question,
                options=options,
                allow_freeform=allow_freeform,
                placeholder=placeholder,
            )

        return ask_user

    def _build_view_image(self) -> tp.Callable:
        """Factory for the ``view_image`` multimodal helper."""

        @operator_tool(
            "view_image",
            description=(
                "Load a local image file and pass it back as an image-capable tool result for "
                "multimodal follow-up. Use it when the model needs to inspect a real image rather "
                "than just discuss a path."
            ),
        )
        def view_image(path: str, detail: str = "auto") -> ImageInspectionResult:
            """Load the image off disk and return an :class:`ImageInspectionResult`."""

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
        """Factory for the ``update_plan`` structured-plan tool."""

        @operator_tool(
            "update_plan",
            description=(
                "Update the current structured execution plan for this session. "
                "Use it to record steps, statuses, and a short explanation of the latest plan change."
            ),
        )
        def update_plan(explanation: str | None = None, plan: list[dict[str, str]] | None = None) -> dict[str, tp.Any]:
            """Apply the update via :class:`PlanStateManager` and emit an audit hook."""

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
        """Factory for the ``web.search_query`` Google search tool."""

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
            """Run a Google text query, optionally domain- or news-restricted."""

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
        """Factory for the ``web.image_query`` image-search tool."""

        @operator_tool(
            "web.image_query",
            description=(
                "Search public image results through Google. Use it when visual references "
                "would help answer the task before opening or analyzing specific pages."
            ),
        )
        def web_image_query(q: str, n_results: int = 5, domains: list[str] | None = None) -> dict[str, tp.Any]:
            """Run a Google image query for the supplied phrase."""

            site = domains[0] if domains else None
            payload = GoogleSearch.static_call(
                query=f"{q} images",
                n_results=n_results,
                site=site,
            )
            return {"query": q, "results": payload.get("results", [])}

        return web_image_query

    def _build_web_open(self) -> tp.Callable:
        """Factory for the ``web.open`` browser-navigation tool."""

        @operator_tool(
            "web.open",
            description=(
                "Open a URL in the shared browser manager or revisit an existing tracked page by ref_id. "
                "Use it after search results when you need the actual page content."
            ),
        )
        async def web_open(url: str | None = None, ref_id: str | None = None, wait_ms: int = 500) -> dict[str, tp.Any]:
            """Delegate to :meth:`BrowserManager.open`."""

            return await self.browser_manager.open(url=url, ref_id=ref_id, wait_ms=wait_ms)

        return web_open

    def _build_web_click(self) -> tp.Callable:
        """Factory for the ``web.click`` DOM interaction tool."""

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
            """Delegate to :meth:`BrowserManager.click`."""

            return await self.browser_manager.click(
                ref_id,
                link_id=link_id,
                selector=selector,
                text=text,
                wait_ms=wait_ms,
            )

        return web_click

    def _build_web_find(self) -> tp.Callable:
        """Factory for the ``web.find`` in-page text search tool."""

        @operator_tool(
            "web.find",
            description=(
                "Search the visible text of a tracked browser page for a string or pattern and return matches."
            ),
        )
        async def web_find(ref_id: str, pattern: str) -> dict[str, tp.Any]:
            """Delegate to :meth:`BrowserManager.find`."""

            return await self.browser_manager.find(ref_id, pattern)

        return web_find

    def _build_web_screenshot(self) -> tp.Callable:
        """Factory for the ``web.screenshot`` page-capture tool."""

        @operator_tool(
            "web.screenshot",
            description=(
                "Capture a screenshot of a tracked browser page and save it to disk. "
                "Use it when a visual snapshot of the current page state is needed."
            ),
        )
        async def web_screenshot(ref_id: str, path: str | None = None, full_page: bool = True) -> dict[str, tp.Any]:
            """Delegate to :meth:`BrowserManager.screenshot`."""

            return await self.browser_manager.screenshot(ref_id, path=path, full_page=full_page)

        return web_screenshot

    def _build_web_weather(self) -> tp.Callable:
        """Factory for the ``web.weather`` Open-Meteo tool."""

        @operator_tool(
            "web.weather",
            description=(
                "Resolve a place name and fetch current weather data through Open-Meteo. "
                "Use it for practical local weather questions without needing general web search."
            ),
        )
        async def web_weather(location: str) -> dict[str, tp.Any]:
            """Geocode ``location`` and fetch the current forecast block."""

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
        """Factory for the ``web.finance`` Yahoo-quote tool."""

        @operator_tool(
            "web.finance",
            description=(
                "Fetch current quote data for a ticker symbol from Yahoo Finance. "
                "Use it for quick price checks and market snapshots."
            ),
        )
        async def web_finance(ticker: str, market: str | None = None, kind: str = "equity") -> dict[str, tp.Any]:
            """Hit the Yahoo Finance quote endpoint for the resolved symbol."""

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
        """Factory for the ``web.sports`` ESPN scoreboard tool."""

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
            """Hit the ESPN scoreboard or standings endpoint for a known league."""

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
        """Factory for the offline ``web.time`` UTC-offset tool."""

        @operator_tool(
            "web.time",
            description=(
                "Return the current local time for a UTC offset without using the network. "
                "Use it for quick timezone calculations when only an offset is needed."
            ),
        )
        def web_time(utc_offset: str) -> dict[str, str]:
            """Return the wall clock for the given ``+HH:MM`` or ``-HH:MM`` offset."""

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
