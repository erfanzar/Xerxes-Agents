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
from __future__ import annotations

import asyncio
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from xerxes import Agent, AgentRuntimeOverrides
from xerxes.operators import (
    BrowserPageState,
    OperatorRuntimeConfig,
    OperatorState,
    PTYSessionManager,
    SpawnedAgentManager,
    UserPromptManager,
)
from xerxes.types import Completion, ResponseResult


def _operator_tools(state: OperatorState) -> dict[str, callable]:
    return {getattr(func, "__xerxes_schema__", {}).get("name", func.__name__): func for func in state.build_tools()}


def test_pty_session_manager_round_trip():
    manager = PTYSessionManager()
    script = (
        "import sys; print('ready', flush=True); line = sys.stdin.readline().strip(); print(f'echo:{line}', flush=True)"
    )
    cmd = f"{shlex.quote(sys.executable)} -u -c {shlex.quote(script)}"

    started = manager.create_session(cmd, yield_time_ms=300)
    session_id = started["session_id"]
    assert "ready" in started["stdout"]

    output = manager.write(session_id, chars="hello\n", close_stdin=True, yield_time_ms=300)
    assert "echo:hello" in output["stdout"]

    closed = manager.close(session_id)
    assert closed["closed"] is True


def test_apply_patch_tool_applies_unified_diff_and_rejects_malformed(tmp_path: Path):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    target = tmp_path / "demo.txt"
    target.write_text("old\n", encoding="utf-8")

    state = OperatorState(OperatorRuntimeConfig(enabled=True, power_tools_enabled=True))
    apply_patch = _operator_tools(state)["apply_patch"]

    patch = """--- a/demo.txt
+++ b/demo.txt
@@ -1 +1 @@
-old
+new
"""
    result = apply_patch(patch, workdir=str(tmp_path))
    assert result["applied"] is True
    assert target.read_text(encoding="utf-8") == "new\n"

    with pytest.raises(ValueError):
        apply_patch("not a patch", workdir=str(tmp_path))


class _FakeLocator:
    def __init__(self, page, selector: str):
        self.page = page
        self.selector = selector
        self.first = self

    async def inner_text(self):
        return self.page.body_text

    async def evaluate_all(self, _script: str):
        return list(self.page.links)

    async def click(self):
        if self.page.links:
            self.page.url = self.page.links[0]


class _FakePage:
    def __init__(self):
        self.url = "https://example.com"
        self._title = "Example"
        self.body_text = "hello world from xerxes"
        self.links = ["https://example.com/next"]

    async def goto(self, url: str, wait_until: str = "domcontentloaded"):
        self.url = url

    async def wait_for_timeout(self, _wait_ms: int):
        return None

    async def title(self):
        return self._title

    def locator(self, selector: str):
        return _FakeLocator(self, selector)

    def get_by_text(self, text: str):
        return _FakeLocator(self, f"text:{text}")

    async def screenshot(self, path: str, full_page: bool = True):
        Path(path).write_text("fake screenshot", encoding="utf-8")


@pytest.mark.asyncio
async def test_browser_manager_open_click_find_and_screenshot(tmp_path: Path):
    from xerxes.operators.browser import BrowserManager

    manager = BrowserManager()
    manager._browser = object()
    page = _FakePage()
    ref_id = "page_1"
    manager._pages[ref_id] = page
    manager._page_state[ref_id] = BrowserPageState(ref_id=ref_id, url=page.url)

    opened = await manager.open(ref_id=ref_id)
    assert opened["title"] == "Example"
    assert opened["links"][0]["url"] == "https://example.com/next"

    clicked = await manager.click(ref_id, link_id=0)
    assert clicked["url"] == "https://example.com/next"

    found = await manager.find(ref_id, "hello")
    assert found["match_count"] == 1

    shot = await manager.screenshot(ref_id, path=str(tmp_path / "shot.png"))
    assert Path(shot["path"]).exists()


class _FakeOrchestrator:
    def __init__(self, agent: Agent):
        self.agents = {agent.id: agent}
        self._agent = agent

    def get_current_agent(self):
        return self._agent


class _FakeXerxes:
    def __init__(self, agent: Agent):
        self.orchestrator = _FakeOrchestrator(agent)

    async def create_response(self, prompt, agent_id=None, stream=False, apply_functions=True):
        return ResponseResult(
            content=f"done:{prompt}",
            response=None,
            completion=Completion(final_content=f"done:{prompt}", agent_id=getattr(agent_id, "id", "")),
            agent_id=getattr(agent_id, "id", ""),
        )


class _FakeRuntimeState:
    def __init__(self):
        self.config = SimpleNamespace(agent_overrides={})

    def get_agent_overrides(self, _agent_id: str):
        return AgentRuntimeOverrides()


@pytest.mark.asyncio
async def test_spawned_agent_manager_lifecycle():
    source_agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    manager = SpawnedAgentManager(_FakeXerxes(source_agent), _FakeRuntimeState())

    spawned = await manager.spawn(message="hello")
    waited = await manager.wait([spawned["id"]], timeout_ms=1000)
    assert waited["completed"][0]["last_output"] == "done:hello"

    closed = manager.close(spawned["id"])
    assert closed["previous_status"] in {"completed", "running", "idle"}

    resumed = manager.resume(spawned["id"])
    assert resumed["closed"] is False


@pytest.mark.asyncio
async def test_spawn_agent_tool_accepts_task_description_alias():
    source_agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes = _FakeXerxes(source_agent)
    runtime_state = _FakeRuntimeState()
    state = OperatorState(OperatorRuntimeConfig(enabled=True, power_tools_enabled=True))
    state.attach_runtime(xerxes, runtime_state)

    spawn_agent = _operator_tools(state)["spawn_agent"]
    spawned = await spawn_agent(task_description="hello from alias")
    waited = await state.subagent_manager.wait([spawned["id"]], timeout_ms=1000)

    assert waited["completed"][0]["last_input"] == "hello from alias"
    assert waited["completed"][0]["last_output"] == "done:hello from alias"


@pytest.mark.asyncio
async def test_send_input_tool_supports_missing_target_and_id_alias():
    source_agent = Agent(id="assistant", model="fake", instructions="Help", functions=[])
    xerxes = _FakeXerxes(source_agent)
    runtime_state = _FakeRuntimeState()
    state = OperatorState(OperatorRuntimeConfig(enabled=True, power_tools_enabled=True))
    state.attach_runtime(xerxes, runtime_state)

    tools = _operator_tools(state)
    spawn_agent = tools["spawn_agent"]
    send_input = tools["send_input"]

    first = await spawn_agent(nickname="first")
    second = await spawn_agent(nickname="second")

    latest = await send_input(message="latest handle")
    waited_latest = await state.subagent_manager.wait([second["id"]], timeout_ms=1000)
    assert latest["id"] == second["id"]
    assert waited_latest["completed"][0]["last_output"] == "done:latest handle"

    explicit = await send_input(id=first["id"], message="explicit alias")
    waited_explicit = await state.subagent_manager.wait([first["id"]], timeout_ms=1000)
    assert explicit["id"] == first["id"]
    assert waited_explicit["completed"][0]["last_output"] == "done:explicit alias"


@pytest.mark.asyncio
async def test_user_prompt_manager_waits_for_answer():
    manager = UserPromptManager()
    task = asyncio.create_task(
        manager.request(
            "Which mode should I use?",
            options=["scan", "grep", "all"],
            allow_freeform=False,
        )
    )

    await asyncio.sleep(0)
    pending = manager.get_pending()
    assert pending is not None
    assert pending["question"] == "Which mode should I use?"
    assert pending["options"][1]["label"] == "grep"

    answered = manager.answer("2")
    assert answered["answer"] == "grep"

    result = await task
    assert result["selected_option"]["label"] == "grep"


@pytest.mark.asyncio
async def test_web_adapter_tools_use_search_and_http_mocks(monkeypatch):
    search_calls: list[tuple[str, dict]] = []

    def fake_search(query: str, **kwargs):
        search_calls.append((query, kwargs))
        return {
            "engine": "google_api",
            "query": query,
            "count": 1,
            "results": [{"title": "Result", "url": "https://example.com", "snippet": ""}],
        }

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url, params=None):
            if "geocoding-api.open-meteo.com" in url:
                return _FakeResponse(
                    {"results": [{"name": "Istanbul", "country": "Turkey", "latitude": 41.0, "longitude": 29.0}]}
                )
            if "api.open-meteo.com" in url:
                return _FakeResponse({"current": {"temperature_2m": 19}})
            if "finance.yahoo.com" in url:
                return _FakeResponse({"quoteResponse": {"result": [{"regularMarketPrice": 42, "currency": "USD"}]}})
            if "site.api.espn.com" in url:
                return _FakeResponse({"events": [{"id": "1"}]})
            raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr("xerxes.operators.state.GoogleSearch.static_call", fake_search)
    monkeypatch.setattr("xerxes.operators.state.httpx.AsyncClient", lambda timeout=20: _FakeClient())

    state = OperatorState(OperatorRuntimeConfig(enabled=True, power_tools_enabled=True))
    tools = _operator_tools(state)

    search_result = tools["web.search_query"]("xerxes")
    image_result = tools["web.image_query"]("waterfalls")
    weather = await tools["web.weather"]("Istanbul")
    finance = await tools["web.finance"]("AMD")
    sports = await tools["web.sports"]("nba")

    assert search_result["results"][0]["title"] == "Result"
    assert image_result["results"][0]["url"] == "https://example.com"
    assert search_calls[0][0] == "xerxes"
    assert weather["location"] == "Istanbul"
    assert finance["price"] == 42
    assert sports["league"] == "nba"


def test_web_search_query_passes_recency_for_news(monkeypatch):
    """``search_type='news'`` should translate to GoogleSearch ``time_range='d'``."""
    captured: dict = {}

    def fake_search(query: str, **kwargs):
        captured.update(kwargs)
        return {
            "engine": "google_api",
            "query": query,
            "count": 1,
            "results": [{"title": "Recent news", "url": "https://example.com/news", "snippet": ""}],
        }

    monkeypatch.setattr("xerxes.operators.state.GoogleSearch.static_call", fake_search)

    state = OperatorState(OperatorRuntimeConfig(enabled=True, power_tools_enabled=True))
    tools = _operator_tools(state)

    search_result = tools["web.search_query"]("latest OpenAI news", search_type="news", domains=["openai.com"])

    assert search_result["search_type"] == "news"
    assert search_result["results"][0]["title"] == "Recent news"
    assert captured["time_range"] == "d"
    assert captured["site"] == "openai.com"
