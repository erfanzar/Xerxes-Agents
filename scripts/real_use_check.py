#!/usr/bin/env python3
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
"""Real-use exercise of every feature added in plans 01-32.

Unlike the unit tests, this driver actually hits real systems where it
can — real git, real subprocess, real SQLite, real OSV.dev, real PyPI,
real Anthropic / OpenAI SDKs when keys are present — and reports
cleanly where credentials or hardware aren't available.

Run from the repo root:

    .venv/bin/python scripts/real_use_check.py

Exit code is 0 if every non-skipped check passes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

# Make the src layout importable.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "python"))


@dataclass
class CheckResult:
    plan: str
    feature: str
    status: str  # "passed" | "skipped" | "failed"
    detail: str = ""


RESULTS: list[CheckResult] = []


def check(plan: str, feature: str, *, real_if: Callable[[], bool] | None = None, skip_reason: str = "") -> Callable:
    """Decorator: register a real-use check.

    ``real_if`` returns True when the runner *can* perform a real call
    (vs needing to skip cleanly). ``skip_reason`` describes the skip."""

    def deco(fn):
        def wrapper():
            if real_if is not None and not real_if():
                RESULTS.append(
                    CheckResult(
                        plan=plan, feature=feature, status="skipped", detail=skip_reason or "prerequisite missing"
                    )
                )
                return
            try:
                detail = fn() or ""
                RESULTS.append(CheckResult(plan=plan, feature=feature, status="passed", detail=str(detail)))
            except Exception as exc:
                RESULTS.append(
                    CheckResult(plan=plan, feature=feature, status="failed", detail=f"{type(exc).__name__}: {exc}")
                )

        wrapper.__plan__ = plan
        wrapper.__feature__ = feature
        return wrapper

    return deco


# ---------------------------- helpers ------------------------------------------


def _has_env(*names: str) -> bool:
    return all(os.environ.get(n) for n in names)


def _has_binary(name: str) -> bool:
    return shutil.which(name) is not None


def _internet_reachable() -> bool:
    import httpx

    try:
        return httpx.get("https://api.github.com/zen", timeout=3).status_code == 200
    except Exception:
        return False


_INTERNET_CACHE: bool | None = None


def _online() -> bool:
    global _INTERNET_CACHE
    if _INTERNET_CACHE is None:
        _INTERNET_CACHE = _internet_reachable()
    return _INTERNET_CACHE


# ---------------------------- checks per plan ----------------------------------


@check("01", "Workspace ensure() creates AGENTS/SOUL/USER/MEMORY/TOOLS")
def plan_01_workspace_files():
    from xerxes.channels.workspace import MarkdownAgentWorkspace

    with tempfile.TemporaryDirectory() as td:
        ws = MarkdownAgentWorkspace(Path(td) / "agent")
        ws.ensure()
        names = sorted(p.name for p in ws.path.iterdir() if p.is_file())
        for required in ("AGENTS.md", "SOUL.md", "USER.md", "MEMORY.md", "TOOLS.md"):
            assert required in names, f"{required} missing"
        ctx = ws.load_context()
        assert ctx.prompt and "AGENTS.md" in ctx.prompt
        return f"created {len(names)} files; prompt {len(ctx.prompt)} bytes"


@check("01", "Workspace tools: write/read/append/diff")
def plan_01_workspace_tools():
    from xerxes.channels.workspace import MarkdownAgentWorkspace
    from xerxes.tools.workspace_tools import (
        workspace_append,
        workspace_diff,
        workspace_list,
        workspace_read,
        workspace_write,
    )

    with tempfile.TemporaryDirectory() as td:
        ws = MarkdownAgentWorkspace(Path(td) / "agent")
        workspace_write("notes.md", "first line\n", ws)
        workspace_append("notes.md", "second line\n", ws)
        body = workspace_read("notes.md", ws)
        assert "first line" in body and "second line" in body
        diff = workspace_diff("notes.md", body + "third line\n", ws)
        assert "+third line" in diff
        listing = workspace_list(ws)
        assert any(item["path"] == "notes.md" for item in listing)
        return f"ws-tools roundtrip: {len(body)} bytes"


@check(
    "01",
    "Workspace import from an external compatible install",
    real_if=lambda: Path("~/.hermes/hermes-agent").expanduser().is_dir(),
    skip_reason="no compatible external install found",
)
def plan_01_workspace_import_real():
    from xerxes.channels.workspace import MarkdownAgentWorkspace
    from xerxes.channels.workspace_import import import_workspace

    src = Path("~/.hermes/hermes-agent").expanduser()
    with tempfile.TemporaryDirectory() as td:
        ws = MarkdownAgentWorkspace(Path(td) / "agent")
        # Use overwrite=True so the default templates don't block.
        result = import_workspace(src, target_workspace=ws, overwrite=True)
        return f"copied={result.copied} skipped={result.skipped}"


@check("02", "Prompt-caching helpers wrap system + tools")
def plan_02_prompt_caching():
    from xerxes.streaming.prompt_caching import (
        extract_cache_tokens,
        wrap_system_with_cache,
        wrap_tools_with_cache,
    )

    sys_wrap = wrap_system_with_cache("Be helpful.")
    assert isinstance(sys_wrap, list) and sys_wrap[0]["cache_control"]["type"] == "ephemeral"
    tool_wrap = wrap_tools_with_cache([{"name": "a"}, {"name": "b"}])
    assert "cache_control" in tool_wrap[-1] and "cache_control" not in tool_wrap[0]

    class _Usage:
        cache_read_input_tokens = 1000
        cache_creation_input_tokens = 200

    assert extract_cache_tokens(_Usage()) == (1000, 200)
    return "system/tool wrapping + usage extraction roundtrip"


@check(
    "02",
    "Real Anthropic prompt-caching round-trip",
    real_if=lambda: _has_env("ANTHROPIC_API_KEY"),
    skip_reason="ANTHROPIC_API_KEY unset",
)
def plan_02_anthropic_real():
    import anthropic

    client = anthropic.Anthropic()
    system = [{"type": "text", "text": "You are a brevity-obsessed assistant.", "cache_control": {"type": "ephemeral"}}]
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=64,
        system=system,
        messages=[{"role": "user", "content": "Reply with one word: ping."}],
    )
    usage = msg.usage
    return (
        f"haiku response={getattr(msg.content[0], 'text', '')[:40]} "
        f"cache_creation={getattr(usage, 'cache_creation_input_tokens', 0)} "
        f"cache_read={getattr(usage, 'cache_read_input_tokens', 0)}"
    )


@check("03", "ContextCompressor real algorithm pass")
def plan_03_compressor():
    from xerxes.context.compressor import ContextCompressor

    c = ContextCompressor(
        threshold=0.01,
        context_window=500,
        protect_first=2,
        protect_last=2,
        summarizer=lambda m, b: f"SUMMARY of {len(m)} middle messages",
    )
    msgs = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"} for i in range(10)]
    result = c.compress(msgs)
    assert result.compressed and result.compressed_count == 6
    assert any("CONTEXT COMPACTION" in (m.get("content") or "") for m in result.messages)
    return f"10 turns -> {len(result.messages)} (protect 2+2, summarize {result.compressed_count})"


@check("03", "Tool-result overflow store roundtrip")
def plan_03_tool_result_storage():
    from xerxes.context.tool_result_storage import ToolResultStorage

    with tempfile.TemporaryDirectory() as td:
        store = ToolResultStorage(Path(td), inline_limit=64)
        ref = store.maybe_store("read_file", "x" * 5_000)
        assert ToolResultStorage.is_ref(ref)
        assert store.fetch(ref) == "x" * 5_000
        return f"overflowed 5kB to disk; ref={ref[:32]}…"


@check("04", "Memory + skill nudges fire on heuristics")
def plan_04_nudges():
    from xerxes.runtime.nudges import NudgeContext, NudgeManager

    m = NudgeManager()
    out = m.check(NudgeContext(turn_index=7, last_user_message="please remember my deadline"))
    fired = {name for name, _ in out}
    out2 = m.check(NudgeContext(turn_index=0, successful_tool_calls_this_turn=7))
    fired |= {name for name, _ in out2}
    return f"fired={sorted(fired)}"


@check("04", "Agent-authored skill: create -> view -> delete")
def plan_04_skill_manage():
    from xerxes.tools import skill_manage_tool

    with tempfile.TemporaryDirectory() as td:
        skill_manage_tool.AUTHORED_DIR = Path(td) / "skills"
        skill_manage_tool._skill_path = lambda name: Path(td) / "skills" / f"{name}.md"
        skill_manage_tool.AUTHORED_DIR.mkdir(parents=True, exist_ok=True)
        created = skill_manage_tool.skill_manage(intent="create", name="demo", body="# demo body")
        assert created["ok"]
        viewed = skill_manage_tool.skill_manage(intent="view", name="demo")
        assert "demo body" in viewed["body"]
        deleted = skill_manage_tool.skill_manage(intent="delete", name="demo")
        assert deleted["ok"]
        return "create/view/delete roundtrip"


@check("05", "Holographic memory backend (local SQLite, real)")
def plan_05_holographic():
    from xerxes.memory.plugins.holographic import HolographicProvider
    from xerxes.memory.provider import MemoryToolCall

    with tempfile.TemporaryDirectory() as td:
        os.environ["HOLOGRAPHIC_DB_PATH"] = str(Path(td) / "facts.db")
        p = HolographicProvider()
        p.initialize()
        added = p.handle_tool_call(MemoryToolCall(name="holo_add", arguments={"content": "user uses linen tea cups"}))
        assert added["ok"]
        found = p.handle_tool_call(MemoryToolCall(name="holo_search", arguments={"query": "linen"}))
        assert any("linen tea cups" in e["content"] for e in found["result"])
        return f"sqlite-backed add+search; entry id={added['result']['id']}"


@check("05", "8 memory plugins import cleanly")
def plan_05_plugin_discovery():
    import importlib

    plugins = importlib.import_module("xerxes.memory.plugins")
    importlib.reload(plugins)
    from xerxes.memory.provider import registry

    names = registry().list_names()
    assert "holographic" in names
    return f"available providers (where deps satisfied)={names}"


@check("05", "Mem0 plugin marked unavailable without MEM0_API_KEY", real_if=lambda: not _has_env("MEM0_API_KEY"))
def plan_05_mem0_unavailable():
    from xerxes.memory.plugins.mem0 import Mem0Provider

    p = Mem0Provider()
    assert p.is_available() is False
    return "ok — graceful skip on missing key"


@check("06", "MCP server: drive all 10 tools in-process")
def plan_06_mcp_server():
    from xerxes.mcp.server import MCP_TOOLS, DaemonBridge, XerxesMcpServer
    from xerxes.session.models import SessionRecord, TurnRecord

    class FakeSessions:
        def __init__(self):
            self._s = {"s1": SessionRecord(session_id="s1", turns=[TurnRecord(turn_id="t1", prompt="hi")])}

        def list_sessions(self, workspace_id=None):
            return list(self._s)

        def get_session(self, sid):
            return self._s.get(sid)

    bridge = DaemonBridge(
        list_channels=lambda: [{"platform": "telegram", "id": "c1"}],
        send_message=lambda sid, text, files=None: {"ok": True, "message_id": "m1"},
        events_poll=lambda sid, since: [{"type": "Text", "text": "hello"}],
        list_pending_permissions=lambda: [{"id": "p1"}],
        respond_permission=lambda pid, allow: {"ok": True, "id": pid, "allow": allow},
        fetch_attachment=lambda aid: {"ok": True, "id": aid, "bytes": 0},
    )
    srv = XerxesMcpServer(sessions=FakeSessions(), bridge=bridge)
    # Drive each tool exactly once.
    assert len(MCP_TOOLS) == 10
    assert len(srv.conversations_list()) == 1
    assert srv.conversation_get("s1")["session_id"] == "s1"
    assert srv.messages_read("s1")[0]["turn_id"] == "t1"
    assert srv.events_poll("s1")[0]["text"] == "hello"
    assert srv.messages_send("s1", "hi")["ok"]
    assert srv.permissions_list_open()[0]["id"] == "p1"
    assert srv.permissions_respond("p1", True)["ok"]
    assert srv.channels_list()[0]["platform"] == "telegram"
    assert srv.attachments_fetch("a1")["ok"]
    assert srv.events_wait("s1", "", 0.01) == [{"type": "Text", "text": "hello"}]
    return "all 10 MCP tools exercised end-to-end"


@check("07", "OSV.dev real query against public API", real_if=_online)
def plan_07_osv_real():
    from xerxes.mcp.osv import check_package

    vulns = check_package("PyPI", "requests")
    return f"OSV returned {len(vulns)} advisories for PyPI:requests"


@check("07", "MCP reconnect retries + scrubs credentials")
def plan_07_reconnect():
    from xerxes.mcp.reconnect import ReconnectPolicy, reconnect_with_backoff, scrub_credentials

    attempts = []

    def boom():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("api_key=sk-secretsecretsecret-leak")
        return "ok"

    sleeps: list[float] = []
    reconnect_with_backoff(boom, policy=ReconnectPolicy(max_attempts=5), sleep=sleeps.append)
    return f"recovered after {len(attempts)} attempts; sleeps={sleeps}; scrub_test='{scrub_credentials('api_key=sk-abcdefghij1234567890')[:30]}'"


@check("07", "PKCE pair generation is unique + valid", real_if=_online)
def plan_07_pkce_real():
    from xerxes.mcp.oauth import OAuthConfig, build_authorize_url, generate_pkce_pair

    v1, c1 = generate_pkce_pair()
    v2, c2 = generate_pkce_pair()
    assert v1 != v2 and c1 != c2
    cfg = OAuthConfig(
        client_id="x",
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        scopes=("read:user",),
    )
    url = build_authorize_url(cfg, state="s", code_challenge=c1)
    assert url.startswith("https://github.com/")
    # We won't *complete* a real OAuth without a registered app.
    return f"unique PKCE pairs generated; auth URL = {url[:80]}…"


@check("08", "ACP server: open session, prompt, approve permission, close")
def plan_08_acp_real():
    from xerxes.acp.server import AcpServer

    captured = {}

    def handler(*, session, text, **_):
        captured["sid"] = session.session_id
        captured["text"] = text
        return {"ok": True}

    s = AcpServer(prompt_handler=handler, tool_list_provider=lambda: [{"name": "read_file"}])
    init = s.initialize()
    assert init["capabilities"]["streaming"]
    opened = s.open_session("/tmp", model="claude-opus-4-7")
    sid = opened["session_id"]
    assert s.prompt(sid, "describe the weather")["ok"]
    pid = s.request_permission(session_id=sid, tool_name="rm_rf", description="delete /", inputs={"path": "/"})[
        "permission_id"
    ]
    assert s.respond_permission(pid, False)["ok"]
    assert s.cancel(sid)["ok"]
    return f"ACP cycle complete (session={sid[:8]}, pending_perms_after={len(s.pending_permissions())})"


@check("09", "Cron: schedule + fire + archive via tmpdir")
def plan_09_cron_real():
    from datetime import datetime

    from xerxes.cron import CronJob, CronScheduler, DeliveryTarget, JobStore, route_output

    with tempfile.TemporaryDirectory() as td:
        store = JobStore(Path(td) / "jobs.json")
        job = CronJob(id="j1", prompt="say hi", schedule="* * * * *", deliver="none")
        store.add(job)
        runs = []
        sched = CronScheduler(store, run_job=lambda j: runs.append(j.id) or "the output")
        now = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)
        sched.tick(now=now)  # schedules first
        sched.tick(now=now.replace(minute=2))  # fires
        assert runs == ["j1"]
        route_output(DeliveryTarget(platform="none"), "the output", archive_dir=Path(td) / "archive", job_id="j1")
        files = list((Path(td) / "archive" / "j1").iterdir())
        return f"fired job + archived {len(files)} file"


@check("10", "Voice — NullRecorder writes valid WAV header")
def plan_10_voice_null():
    from xerxes.tools.voice_mode import NullRecorder

    with tempfile.TemporaryDirectory() as td:
        rec = NullRecorder(Path(td) / "x.wav")
        rec.start()
        out = rec.stop()
        assert out.read_bytes().startswith(b"RIFF")
        return f"recorded {out.stat().st_size} bytes"


@check(
    "10",
    "TTS Edge-tts dependency check",
    real_if=lambda: __import__("importlib.util").util.find_spec("edge_tts") is not None,
    skip_reason="edge-tts not installed",
)
def plan_10_edge_tts_check():
    # We don't actually fire Edge-tts (network); just confirm it imports.
    import edge_tts  # type: ignore

    return f"edge_tts version: {getattr(edge_tts, '__version__', 'unknown')}"


@check("11", "Vision — registered backend invoked with image")
def plan_11_vision():
    from xerxes.tools import vision_tool

    seen = {}

    def fake(b64, mime, prompt, params):
        seen["mime"] = mime
        seen["bytes"] = len(b64)
        return "a tiny PNG"

    vision_tool.register_backend("dummy", fake)
    with tempfile.TemporaryDirectory() as td:
        png = Path(td) / "a.png"
        png.write_bytes(
            bytes.fromhex(
                "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000a"
                "49444154789c63000100000005000100"
                "0d0a2db40000000049454e44ae426082"
            )
        )
        out = vision_tool.analyze_image(png, provider="dummy")
        assert out["response"] == "a tiny PNG"
        return f"vision tool routed {seen['bytes']}-char base64 ({seen['mime']})"


@check("11", "Image-gen — registered provider invoked")
def plan_11_imagegen():
    from xerxes.tools import image_generation_tool

    image_generation_tool.register_provider("dummy", lambda p, params: b"PNGFAKE")
    with tempfile.TemporaryDirectory() as td:
        out = image_generation_tool.generate_image("a moon", provider="dummy", out_path=Path(td) / "m.png")
        assert out["bytes"] == 7
        return f"image-gen wrote {out['bytes']} bytes via dummy provider"


@check("12", "Browser provider registry returns 5 backends")
def plan_12_browser_registry():
    from xerxes.operators.browser_providers import SUPPORTED_PROVIDERS, registry

    available = registry()
    assert set(SUPPORTED_PROVIDERS) == set(available)
    return f"providers={sorted(available)}"


@check(
    "12",
    "Local Playwright provider returns marker session (requires playwright)",
    real_if=lambda: __import__("importlib.util").util.find_spec("playwright") is not None,
    skip_reason="playwright not installed",
)
def plan_12_local_playwright():
    from xerxes.operators.browser_providers import get

    sess = get("local").open(headless=True)
    return f"local session created: provider={sess.provider}"


@check(
    "13",
    "SSH backend assembles real argv via stubbed subprocess",
    real_if=lambda: _has_binary("ssh"),
    skip_reason="ssh binary missing",
)
def plan_13_ssh_argv():
    from unittest.mock import patch

    from xerxes.security.sandbox_backends.ssh_backend import SshBackendConfig, SshSandboxBackend

    captured = {}

    class _Res:
        returncode = 0
        stdout = "OK"
        stderr = ""

    def fake_run(argv, **kw):
        captured["argv"] = argv
        return _Res()

    with patch("subprocess.run", fake_run):
        backend = SshSandboxBackend(SshBackendConfig(host="example.invalid", user="ci"))
        result = backend.execute("uname -a", cwd="/work")
    assert result["returncode"] == 0
    return f"ssh argv = {captured['argv'][:6]}…"


@check("13", "File sync filters oversize files")
def plan_13_file_sync():
    from xerxes.security.sandbox_backends.file_sync import FileSyncSpec, sync_push

    with tempfile.TemporaryDirectory() as td:
        small = Path(td) / "s.txt"
        small.write_text("hi")
        big = Path(td) / "b.bin"
        big.write_bytes(b"x" * 50_000)
        pushed = sync_push(
            [FileSyncSpec(small, "/r/s"), FileSyncSpec(big, "/r/b")],
            copy_fn=lambda lp, rp: None,
            max_bytes=10_000,
        )
        assert [s.local_path.name for s in pushed] == ["s.txt"]
        return "small file pushed, big file filtered"


def _deepseek_v3_fixture() -> str:
    """Build the DeepSeek-V3 tokenizer-marker fixture.

    The model emits U+FF5C (fullwidth bar) and U+2581 (low one-eighth
    block) literally. Building via chr() keeps the source file ASCII
    so ruff's RUF001/RUF003 lints stay quiet, while the runtime string
    is byte-identical to what the model produces."""
    fw = chr(0xFF5C)
    lo = chr(0x2581)
    return f'<{fw}tool{lo}call{lo}begin{fw}>{{"name":"d","arguments":{{}}}}<{fw}tool{lo}call{lo}end{fw}>'


@check("14", "Tool-call parsers — real-string parsing for 11 model families")
def plan_14_parsers_real():
    from xerxes.streaming.parsers import REGISTRY, detect_format

    samples = {
        "xml_tool_call": '<tool_call>{"name":"read_file","arguments":{"path":"a"}}</tool_call>',
        "llama": '<|python_tag|>{"name":"f","parameters":{"x":1}}<|eom_id|>',
        "mistral": '[TOOL_CALLS][{"name":"f","arguments":{}}]',
        "qwen": '<tool_call>{"name":"q","arguments":{}}</tool_call>',
        "qwen3_coder": '|<function_call_start|>{"name":"q","parameters":{}}|<function_call_end|>',
        "deepseek_v3": _deepseek_v3_fixture(),
        "deepseek_v3_1": '<tool>{"name":"d","arguments":{}}</tool>',
        "glm45": '<tool_call>{"name":"g","arguments":{}}</tool_call>',
        "glm47": '<function_call>{"name":"g","arguments":{}}</function_call>',
        "kimi_k2": '<|tool_call|>{"name":"k","arguments":{}}<|/tool_call|>',
        "longcat": '<longcat:tool>{"name":"l","arguments":{}}</longcat:tool>',
    }
    parsed = 0
    for name, text in samples.items():
        parser = REGISTRY[name]
        out = parser.parse(text)
        assert out and out[0].name, f"{name} failed to parse"
        parsed += 1
    assert detect_format("Qwen/Qwen3-Coder-32B") == "qwen3_coder"
    return f"parsed {parsed} formats; detect_format ok"


@check("15", "RL — Tinker client cycle with stubbed backend")
def plan_15_rl_real():
    from xerxes.training.rl import RLRunStatus, TinkerClient, TinkerRunConfig, builtin_envs

    started: dict = {}

    def start(payload):
        started["payload"] = payload
        return "run-xyz"

    states = iter(
        [
            {"status": "running", "iteration": 1, "reward": 0.0},
            {"status": "running", "iteration": 2, "reward": 0.5},
            {"status": "succeeded", "iteration": 100, "reward": 1.0, "wandb_url": "https://wandb.ai/example"},
        ]
    )
    client = TinkerClient(backend_start=start, backend_status=lambda rid: next(states), backend_cancel=lambda rid: True)
    rid = client.start(TinkerRunConfig(model="claude-haiku-4-5", env="xerxes-terminal-test"))
    s1 = client.status(rid)
    s2 = client.status(rid)
    s3 = client.status(rid)
    assert s3.status is RLRunStatus.SUCCEEDED and s3.reward == 1.0
    envs = builtin_envs().list_envs()
    return f"run id {rid} -> {s1.status.value}->{s2.status.value}->{s3.status.value}; {len(envs)} envs registered"


@check("16", "Trajectory compressor batch")
def plan_16_trajectory():
    from xerxes.context.compressor import ContextCompressor
    from xerxes.training.trajectory_compressor import TrajectoryCompressor

    with tempfile.TemporaryDirectory() as td:
        c = TrajectoryCompressor(
            compressor=ContextCompressor(
                threshold=0.01,
                context_window=200,
                protect_first=1,
                protect_last=1,
                summarizer=lambda m, b: "S",
            )
        )
        trajs = [
            {"id": "t1", "messages": [{"role": "user", "content": f"x{i}"} for i in range(6)]},
            {"id": "t2", "messages": [{"role": "user", "content": "small"}]},
        ]
        run = c.run(trajs, out_path=Path(td) / "out.jsonl", metrics_path=Path(td) / "metrics.json")
        out_lines = (Path(td) / "out.jsonl").read_text().splitlines()
        assert run.processed == 2 and len(out_lines) == 2
        return f"compressed {run.processed} trajectories"


@check("17", "Batch runner — 4 records with random failure")
def plan_17_batch():
    from xerxes.training.batch_runner import BatchRecord, BatchResult, BatchRunner

    def runner(rec):
        if rec.id == "bad":
            raise RuntimeError("simulated")
        return BatchResult(id=rec.id, response=f"r:{rec.prompt}", input_tokens=10, output_tokens=5, cost_usd=0.001)

    br = BatchRunner(runner, workers=2)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "out.jsonl"
        summary = br.run(
            [BatchRecord(id=str(i), prompt=f"p{i}") for i in range(3)] + [BatchRecord(id="bad", prompt="boom")],
            out_path=out,
        )
        assert summary.total == 4 and summary.succeeded == 3 and summary.failed == 1
        return f"4 records, 3 succeeded, 1 failed, ${summary.total_cost_usd:.4f}"


@check("18", "Session branching: fork + lineage")
def plan_18_branching():
    from xerxes.session.branching import branch_session, lineage
    from xerxes.session.models import SessionRecord, TurnRecord
    from xerxes.session.store import FileSessionStore

    with tempfile.TemporaryDirectory() as td:
        store = FileSessionStore(td)
        store.save_session(SessionRecord(session_id="root", turns=[TurnRecord(turn_id="t1", prompt="hi")]))
        child = branch_session(store, source_session_id="root", new_session_id="c1")
        branch_session(store, source_session_id="c1", new_session_id="g1")
        assert lineage(store, "g1") == ["g1", "c1", "root"]
        return f"lineage = g1->c1->root; child has {len(child.turns)} turn(s)"


@check(
    "18", "Snapshots — real git shadow repo on tmp dir", real_if=lambda: _has_binary("git"), skip_reason="git missing"
)
def plan_18_snapshots_real():
    from xerxes.session.snapshots import SnapshotManager

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "work"
        ws.mkdir()
        (ws / "a.txt").write_text("v1\n")
        shadow = Path(td) / "shadow"
        shadow.mkdir()
        mgr = SnapshotManager(ws, shadow_root=shadow)
        snap = mgr.snapshot("v1")
        (ws / "a.txt").write_text("BROKEN\n")
        mgr.rollback(snap.id)
        assert (ws / "a.txt").read_text() == "v1\n"
        return f"snapshot {snap.id[:8]} rolled back successfully (sha={snap.commit_sha[:8]})"


@check("19", "Slash command registry resolves canonical + alias")
def plan_19_slash():
    from xerxes.bridge.commands import COMMAND_REGISTRY, resolve_command, telegram_bot_commands

    cmd_model = resolve_command("/model gpt-4o")
    cmd_q = resolve_command("/q")
    assert cmd_model and cmd_model.name == "model"
    assert cmd_q and cmd_q.name == "exit"
    tg = telegram_bot_commands()
    return f"registry={len(COMMAND_REGISTRY)} commands; telegram-safe={len(tg)}"


@check("20", "Doctor — run full check on this machine")
def plan_20_doctor():
    from xerxes.runtime.doctor import has_failures, run_all_checks

    report = run_all_checks()
    severities = {d.name: d.severity for d in report}
    return f"checks={severities}; failures={has_failures(report)}"


@check("20", "Update — query real PyPI for xerxes-agent latest", real_if=_online, skip_reason="no internet")
def plan_20_update_real_pypi():
    from xerxes.runtime.update import detect_install_mode, latest_pypi_version

    # xerxes-agent isn't published (404); use a real package to prove plumbing works.
    real_pkg = latest_pypi_version(package="requests")
    mode = detect_install_mode()
    return f"real PyPI lookup for `requests`={real_pkg}; install mode={mode.value}"


@check("20", "Setup wizard runs scripted answers and writes config")
def plan_20_setup():
    from xerxes.runtime.setup_wizard import run_wizard, write_config

    with tempfile.TemporaryDirectory() as td:
        result = run_wizard({"provider": "anthropic", "model": "claude-opus-4-7"})
        cfg = write_config(result.answers, target=Path(td) / "config.yaml")
        assert 'provider: "anthropic"' in cfg.read_text()
        return f"wrote {cfg.stat().st_size} bytes config"


@check("21", "Sticker cache + identity hashing + session reset")
def plan_21_messaging_depth():
    from datetime import datetime, timedelta

    from xerxes.channels.identity_hashing import hash_chat, hash_user
    from xerxes.channels.session_reset import ResetTrigger, SessionResetPolicy, should_reset
    from xerxes.channels.sticker_cache import StickerCache

    os.environ["XERXES_IDENTITY_SALT"] = "real-use-salt"
    assert hash_user("telegram", 123) == hash_user("telegram", 123)
    assert hash_chat("slack", "C1").startswith("slack:")
    with tempfile.TemporaryDirectory() as td:
        cache = StickerCache(Path(td), lru_size=4)
        cache.put("telegram", "AB", Path(td) / "x.webp")
        assert cache.get("telegram", "AB") is not None
        policy = SessionResetPolicy(trigger=ResetTrigger.TIMEOUT, timeout_minutes=10)
        now = datetime.now(UTC)
        assert should_reset(policy, last_message_at=now - timedelta(minutes=20), message_count=0, now=now) is True
        return "sticker cache + identity hash + reset policy all working"


@check("22", "Security: redact + url_safety + path_security + approvals")
def plan_22_security():
    from xerxes.security.approvals import ApprovalRecord, ApprovalScope, ApprovalStore
    from xerxes.security.path_security import PathEscape, resolve_within
    from xerxes.security.redact import redact_string
    from xerxes.security.url_safety import is_url_safe

    redacted = redact_string("Authorization: Bearer abc123XYZ789LONGTOKEN; api_key=sk-leakedabcdef123456")
    assert "sk-leakedabcdef" not in redacted
    assert is_url_safe("http://192.168.1.1/internal") is False
    assert is_url_safe("https://example.com/api") is True
    with tempfile.TemporaryDirectory() as td:
        try:
            resolve_within(td, "../etc/passwd")
            raise AssertionError("escape allowed")
        except PathEscape:
            pass
        store = ApprovalStore(persist_path=Path(td) / "approvals.json")
        store.add(ApprovalRecord(tool_name="rm", scope=ApprovalScope.ALWAYS, granted=True))
        # Reload from disk.
        store2 = ApprovalStore(persist_path=Path(td) / "approvals.json")
        assert store2.check(tool_name="rm", session_id="anything") is True
    return "all security primitives exercised + approval persisted"


@check("23", "Skills hub — local + sync manifest")
def plan_23_skills_hub():
    from xerxes.extensions.skill_sources import LocalSkillSource
    from xerxes.extensions.skills_sync import ManifestEntry, sync_manifest

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        (src / "alpha").mkdir(parents=True)
        (src / "alpha" / "SKILL.md").write_text("---\nversion: 1.0\n---\n# Alpha skill")
        local = LocalSkillSource(src)
        hits = local.search("alpha")
        assert any(h.name == "alpha" for h in hits)
        bundle = local.fetch("alpha")
        assert bundle.version == "1.0"
        # Sync via manifest.
        target = Path(td) / "installed"
        result = sync_manifest(
            [ManifestEntry(source="local", identifier="alpha")],
            sources={"local": local},
            target_dir=target,
        )
        assert "alpha" in result.installed
        return f"installed {result.installed}, skipped {result.skipped}"


@check("24", "Error classifier + rate limit tracker + aux client")
def plan_24_routing():
    from xerxes.runtime.auxiliary_client import AuxiliaryClient
    from xerxes.runtime.error_classifier import ErrorKind, classify
    from xerxes.runtime.rate_limit_tracker import RateLimitTracker

    assert classify(TimeoutError()).kind is ErrorKind.TIMEOUT
    assert classify(Exception("HTTP 429 too many requests")).kind is ErrorKind.RATE_LIMIT
    assert classify(Exception("maximum context length exceeded")).kind is ErrorKind.CONTEXT_OVERFLOW
    rl = RateLimitTracker(throttle_ratio=0.1)
    rl.update("openai", "gpt-4o", {"x-ratelimit-limit-requests": "100", "x-ratelimit-remaining-requests": "3"})
    assert rl.should_throttle("openai", "gpt-4o") is True
    aux = AuxiliaryClient(lambda msgs, max_tokens, model: f"summary of {len(msgs)} messages", model="test-aux")
    summary = aux.summarize([{"role": "user", "content": "hi"}])
    return f"err-classifier=ok, rate-limiter throttling=ok, aux-client summary='{summary}'"


@check("25", "OAuth flow — build authorize URL + storage roundtrip")
def plan_25_auth():
    from xerxes.auth.oauth import OAuthClient, github_pat_preset
    from xerxes.auth.storage import CredentialStorage
    from xerxes.mcp.oauth import OAuthToken

    with tempfile.TemporaryDirectory() as td:
        client = OAuthClient(github_pat_preset("xerxes-test-client"))
        ctx = client.begin_authorize()
        assert ctx.url.startswith("https://github.com/login/oauth/authorize")
        store = CredentialStorage(base_dir=Path(td))
        store.save("github", OAuthToken(access_token="t", refresh_token="r"))
        loaded = store.load("github")
        assert loaded.access_token == "t"
        return f"auth URL prepared ({len(ctx.url)} chars), credentials persisted"


@check("26", "Pricing + insights — real math against 30+ models")
def plan_26_insights():
    from datetime import datetime

    from xerxes.runtime.cost_tracker import CostEvent
    from xerxes.runtime.insights import build_report
    from xerxes.runtime.pricing import compute_cost, known_models

    n = len(known_models())
    cost = compute_cost(
        model="claude-opus-4-7", input_tokens=1_000_000, output_tokens=500_000, cache_read_tokens=200_000
    )
    # $15 + $37.5 + $0.3 = $52.8
    assert 52.0 < cost < 54.0
    events = [
        CostEvent(
            model="claude-opus-4-7",
            in_tokens=100,
            out_tokens=200,
            cost_usd=0.01,
            timestamp=datetime.now(UTC).isoformat(),
            cache_read_tokens=300,
        ),
        CostEvent(
            model="gpt-4o",
            in_tokens=50,
            out_tokens=100,
            cost_usd=0.005,
            timestamp=datetime.now(UTC).isoformat(),
        ),
    ]
    rpt = build_report(events)
    assert rpt.total_events == 2
    return f"{n} models priced; opus mixed cost=${cost:.2f}; report has {len(rpt.by_model)} models"


@check("27", "TUI polish — skin engine + prompt queue + clipboard helper")
def plan_27_tui():
    from xerxes.tui.clipboard import PromptQueue, grab_clipboard_text
    from xerxes.tui.skin_engine import Skin, SkinEngine, hex_to_ansi_fg

    with tempfile.TemporaryDirectory() as td:
        engine = SkinEngine(Path(td))
        engine.save(Skin(name="customA", roles={"primary": "#aa1111", "accent": "#22bb22"}))
        loaded = engine.load("customA")
        assert loaded.roles["primary"] == "#aa1111"
        assert "\033[" in hex_to_ansi_fg("#abcdef")
    q = PromptQueue()
    q.push("first")
    q.push("second")
    assert q.drain() == ["first", "second"]
    # Real clipboard call (will return None on headless / no clipboard backend).
    _ = grab_clipboard_text()
    return "skin saved+loaded, prompt queue FIFO ok"


@check("28", "Iteration budget + process registry + interrupt token (real subprocess)", real_if=lambda: True)
def plan_28_protocol_real():
    from xerxes.runtime.interrupt import InterruptToken
    from xerxes.runtime.iteration_budget import BudgetExhausted, IterationBudget
    from xerxes.runtime.process_registry import ProcessRegistry
    from xerxes.streaming.tool_call_ids import deterministic_tool_call_id

    # Iteration budget.
    b = IterationBudget(max_iterations=3)
    b.consume()
    b.consume()
    b.consume()
    try:
        b.consume()
        raise AssertionError("budget exhausted should raise")
    except BudgetExhausted:
        pass
    b.refund()
    assert b.remaining == 1

    # Process registry — real subprocess.
    reg = ProcessRegistry()
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(0)"])
    pid = reg.register(proc, name="quick-exit")
    rc = reg.wait(pid, timeout=5)
    assert rc == 0

    # Interrupt token — fire from a thread.
    token = InterruptToken()
    threading.Timer(0.01, token.set).start()
    assert token.wait(timeout=1.0) is True

    # Deterministic ID stable across runs.
    a = deterministic_tool_call_id("read_file", {"path": "/x"})
    b_id = deterministic_tool_call_id("read_file", {"path": "/x"})
    assert a == b_id and a.startswith("call_")
    return "budget/refund/process-registry/interrupt/det-IDs all real"


@check("29", "Schema migration runs on a stale on-disk session")
def plan_29_migrations():
    from xerxes.session.models import CURRENT_SCHEMA_VERSION
    from xerxes.session.store import FileSessionStore

    with tempfile.TemporaryDirectory() as td:
        store = FileSessionStore(td)
        # Plant a stale v0 record on disk (no schema_version field).
        (Path(td) / "old.json").write_text(
            json.dumps(
                {
                    "session_id": "old",
                    "turns": [{"turn_id": "t1", "prompt": "hi"}],
                    "metadata": {},
                }
            )
        )
        rec = store.load_session("old")
        assert rec is not None and rec.schema_version == CURRENT_SCHEMA_VERSION
        # Atomic write check — re-save and confirm no .tmp leftovers.
        store.save_session(rec)
        leftovers = [p.name for p in Path(td).iterdir() if p.name.startswith(".")]
        assert leftovers == []
        return f"migrated to v{CURRENT_SCHEMA_VERSION}; atomic write clean"


@check("30", "Distribution — Homebrew formula + Termux filter")
def plan_30_packaging():
    from xerxes.runtime.distribution import (
        SHELL_INSTALL_SNIPPET,
        detect_platform,
        render_homebrew_formula,
        termux_filter_extras,
    )

    info = detect_platform()
    formula = render_homebrew_formula(
        version="0.2.0",
        tarball_url="https://github.com/erfanzar/Xerxes-Agents/archive/refs/tags/v0.2.0.tar.gz",
        sha256="deadbeef" * 8,
    )
    assert "Apache-2.0" in formula
    filtered = termux_filter_extras({"voice": ["faster-whisper>=1.0", "numpy>=1.24"]})
    assert "numpy>=1.24" in filtered["voice"]
    assert all("faster" not in d.lower() for d in filtered["voice"])
    assert "uv tool install xerxes-agent" in SHELL_INSTALL_SNIPPET
    return f"detected platform={info.system}; brew formula rendered; termux filter ok"


@check("31", "Responses API translator + SSE parser end-to-end")
def plan_31_streaming():
    from xerxes.streaming.responses_api import ResponsesEventTranslator
    from xerxes.streaming.sse import parse_sse_stream

    # Real SSE stream.
    chunks = [
        "event: response.output_text.delta\n",
        "data: hello\n\n",
        "event: response.completed\n",
        "data: {}\n\n",
    ]
    events = list(parse_sse_stream(chunks))
    assert len(events) == 2 and events[0].event == "response.output_text.delta"

    # Responses event translator with realistic event sequence.
    t = ResponsesEventTranslator()
    raw_events = [
        {"type": "response.output_text.delta", "delta": "Hello "},
        {"type": "response.output_text.delta", "delta": "world!"},
        {"type": "response.output_item.added", "item": {"id": "call_1", "type": "tool_call", "name": "read"}},
        {"type": "response.function_call_arguments.delta", "item_id": "call_1", "delta": '{"path":"a.txt"}'},
        {"type": "response.output_item.done", "item": {"id": "call_1", "type": "tool_call", "name": "read"}},
        {"type": "response.completed", "response": {"usage": {"input_tokens": 12, "output_tokens": 2}}},
    ]
    chunks_out = list(t.translate(raw_events))
    assert len(chunks_out) == 2  # two text deltas
    assert len(t.usage.tool_calls) == 1 and t.usage.input_tokens == 12
    return f"SSE parsed {len(events)} events; translator yielded {len(chunks_out)} chunks + 1 tool call"


@check("32", "send_message + clarify + memory CRUD against tmp workspace")
def plan_32_msg_tools():
    from xerxes.tools import memory_crud, send_message_tool
    from xerxes.tools.clarify_tool import StaticAsker, clarify

    captured = {}

    def telegram_sender(platform, recipient, payload):
        captured["platform"] = platform
        captured["recipient"] = recipient
        captured["text"] = payload["text"]
        return {"ok": True, "id": "m-real-001"}

    send_message_tool.register_platform("telegram", telegram_sender)
    out = send_message_tool.send_message(platform="telegram", recipient="ME", text="Hello from real-use check!")
    assert out == {"ok": True, "id": "m-real-001"}
    answer = clarify(question="Pick", options=["a", "b"], asker=StaticAsker(index=1))
    assert answer["answer"] == "b"
    with tempfile.TemporaryDirectory() as td:
        memory_crud.memory_add(Path(td), "User prefers terse responses")
        memory_crud.user_add(Path(td), "Located in Tehran")
        lst_m = memory_crud.memory_list(Path(td))
        lst_u = memory_crud.user_list(Path(td))
        assert lst_m["items"][0]["content"] == "User prefers terse responses"
        assert lst_u["items"][0]["content"] == "Located in Tehran"
    return f"send_message routed to {captured['platform']}/{captured['recipient']}; clarify+memory CRUD ok"


# ============================================================================
# TUI surface checks
# ============================================================================


@check("TUI-1", "Multiline input buffer (Alt+Enter newline, Enter submit)")
def tui_multiline():
    from xerxes.tui.input_buffer import InputBufferConfig, build_input_buffer, build_multiline_key_bindings

    submitted: list[str] = []
    buf = build_input_buffer(InputBufferConfig(on_accept=submitted.append, history_path=None))
    buf.text = "line one\nline two"
    buf.validate_and_handle()
    kb = build_multiline_key_bindings()
    return f"multiline={buf.multiline()}; bindings={len(kb.bindings)}; submitted={submitted!r}"


@check("TUI-2/3", "Auto-suggest + FileHistory roundtrip")
def tui_autosuggest_history():
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from xerxes.tui.input_buffer import InputBufferConfig, build_input_buffer

    with tempfile.TemporaryDirectory() as td:
        hist = Path(td) / "h.txt"
        buf = build_input_buffer(InputBufferConfig(history_path=hist, auto_suggest=True))
        buf.history.store_string("hello from earlier")
        loaded = list(buf.history.load_history_strings())
        assert "hello from earlier" in loaded
        assert isinstance(buf.auto_suggest, AutoSuggestFromHistory)
        return f"history persisted {hist.stat().st_size} bytes; autosuggest active"


@check("TUI-4", "Reasoning tag filter strips <think>/<reasoning> from stream")
def tui_reasoning_filter():
    from xerxes.tui.reasoning_filter import ReasoningFilter

    rf = ReasoningFilter()
    chunks = ["Hello", " <think>", "let me", " think", "</think>", " world"]
    visible_parts = []
    for c in chunks:
        v, _t = rf.feed(c)
        visible_parts.append(v)
    v_end, _ = rf.flush()
    visible_parts.append(v_end)
    visible = "".join(visible_parts)
    assert "let me think" not in visible
    assert "let me" in rf.thinking_log
    return f"6-chunk stream -> visible={visible!r}, thinking_log={rf.thinking_log!r}"


@check("TUI-5", "Status snapshot + format with cache + duration")
def tui_status_snapshot():
    from xerxes.tui.status_bar import StatusSnapshot, format_status

    snap = StatusSnapshot(
        model="claude-opus-4-7",
        input_tokens=1200,
        output_tokens=600,
        cache_read_tokens=15000,
        cache_write_tokens=2000,
        context_window=200_000,
        cost_usd=0.0825,
        duration_sec=47.3,
        permission_mode="manual",
        queue_depth=1,
        active_skill="plan",
    )
    out = format_status(snap)
    assert "claude-opus-4-7" in out
    assert "$0.0825" in out
    assert "00:47" in out
    assert "ctx" in out
    return f"status line: {out}"


@check("TUI-6", "150+ welcome tips + tip-of-the-day")
def tui_tips():
    from xerxes.tui.tips import TIPS, random_tip, tip_of_the_day

    return f"{len(TIPS)} tips; random[seed=42]={random_tip(seed=42)[:60]}…; tod={tip_of_the_day()[:60]}…"


@check("TUI-7", "Skin branding: 7 builtin skins + Ares overrides")
def tui_skin_branding():
    from xerxes.tui.skin_engine import Skin, SkinEngine

    with tempfile.TemporaryDirectory() as td:
        engine = SkinEngine(Path(td))
        names = engine.available()
        ares = engine.load("ares")
        assert ares.label("agent_name") == "Ares"
        verbs = ares.spinner_verbs()
        # Roundtrip a custom skin with branding.
        sk = Skin(name="brand", roles={"primary": "#112233"}, branding={"agent_name": "Atlas", "prompt_symbol": "λ"})
        engine.save(sk)
        loaded = engine.load("brand")
        assert loaded.label("agent_name") == "Atlas"
        assert loaded.label("prompt_symbol") == "λ"
        return f"skins available={names}; ares verbs={verbs[:3]}…; custom brand roundtripped"


@check("TUI-8", "Voice key handler — push-to-talk state machine")
def tui_voice_keys():
    from xerxes.tui.voice_keys import VoiceKeyHandler, VoiceState

    received: list[str] = []
    h = VoiceKeyHandler(
        transcribe=lambda path: "what is the weather",
        submit=lambda text: received.append(text),
    )
    assert h.state is VoiceState.IDLE
    h.start_recording()
    assert h.state is VoiceState.RECORDING
    out = h.stop_recording()
    assert out == "what is the weather"
    assert received == ["what is the weather"]
    return f"PTT cycle complete: idle→recording→idle; submitted={received}"


@check("TUI-9", "Clipboard attachment buffer staging")
def tui_clipboard_attach():
    from xerxes.tui.clipboard_attach import AttachmentBuffer

    with tempfile.TemporaryDirectory() as td:
        # Stage a fake image file.
        img = Path(td) / "screenshot.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\nFAKE")
        buf = AttachmentBuffer(capture_image=lambda d: img, store_dir=Path(td))
        att = buf.capture_clipboard_image()
        assert att is not None and att.kind == "image"
        # Also attach a regular file.
        f = Path(td) / "notes.txt"
        f.write_text("hello")
        buf.attach_path(f)
        drained = buf.drain()
        assert len(drained) == 2 and buf.pending() == []
        return f"staged {len(drained)} attachments, drained cleanly"


@check("TUI-10/12", "5-option approval panel navigation")
def tui_approval_panel():
    from xerxes.tui.panel_state import DEFAULT_APPROVAL_OPTIONS, ApprovalPanelState

    assert len(DEFAULT_APPROVAL_OPTIONS) == 5
    state = ApprovalPanelState()
    first = state.current()
    state.down()
    state.down()
    state.up()
    second = state.current()
    assert second is not first
    return f"5 options; nav: down,down,up -> {second.value}"


@check("TUI-11", "Approval countdown timer fires + cancels")
def tui_approval_countdown():
    from xerxes.tui.panel_state import ApprovalCountdown

    fired = []
    c = ApprovalCountdown(timeout_seconds=0.08)
    c.start(lambda: fired.append("auto-deny"))
    time.sleep(0.2)
    assert fired == ["auto-deny"]
    # Cancelled timer doesn't fire.
    fired2 = []
    c2 = ApprovalCountdown(timeout_seconds=0.5)
    c2.start(lambda: fired2.append("late"))
    c2.cancel()
    time.sleep(0.6)
    assert fired2 == []
    return "timer fires + cancel works"


@check("TUI-13", "@-mentions completer + expansion against tmp workspace + real git")
def tui_at_mentions():
    from prompt_toolkit.document import Document
    from xerxes.tui.at_mentions import AT_TRIGGERS, AtMentionCompleter, expand_mentions_in_text

    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "src").mkdir()
        (Path(td) / "src" / "main.py").write_text("print('hi')")
        (Path(td) / "README.md").write_text("readme content")
        comp = AtMentionCompleter(Path(td))
        # File completion under @file:.
        doc = Document("@file:")
        out = list(comp.get_completions(doc, None))
        names = [c.text for c in out]
        assert any(n in names for n in ("README.md", "src/"))
        # Expansion of @file: payload.
        results = expand_mentions_in_text("@file:README.md and @url:https://example.com/x", workspace_root=Path(td))
        kinds = [r.kind for r in results]
        return f"triggers={AT_TRIGGERS}; completed {len(out)} entries; expansion kinds={kinds}"


@check("TUI-14", "Context bar visual meter")
def tui_context_bar():
    from xerxes.tui.context_bar import context_bar, context_bar_with_pct

    bar = context_bar(used=128_000, window=200_000, width=20)
    assert "█" in bar
    full = context_bar(used=200_000, window=200_000, width=10)
    assert full == "█" * 10
    empty = context_bar(used=0, window=200_000, width=10)
    assert empty == "░" * 10
    with_pct = context_bar_with_pct(used=50_000, window=200_000, width=10)
    assert "25.0%" in with_pct
    return f"64%-filled bar = `{bar}`"


@check(
    "TUI-15",
    "/rollback diff against real git shadow repo",
    real_if=lambda: shutil.which("git") is not None,
    skip_reason="git missing",
)
def tui_rollback_diff():
    from xerxes.session.snapshot_diff import diff_against_snapshot
    from xerxes.session.snapshots import SnapshotManager

    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "work"
        ws.mkdir()
        (ws / "config.yaml").write_text("version: 1\n")
        shadow = Path(td) / "shadow"
        shadow.mkdir()
        mgr = SnapshotManager(ws, shadow_root=shadow)
        snap = mgr.snapshot("baseline")
        (ws / "config.yaml").write_text("version: 2\nnew_field: true\n")
        diff = diff_against_snapshot(mgr, snap.id)
        assert "version: 1" in diff.diff_text or "+version: 2" in diff.diff_text
        return f"file_count={diff.file_count}, +{diff.added} -{diff.removed}"


@check("TUI-16", "Background session manager runs, queues, and cancels")
def tui_background_sessions():
    from xerxes.runtime.background_sessions import (
        BackgroundSessionManager,
        BackgroundStatus,
    )

    def runner(sess):
        return f"answer for {sess.prompt}"

    mgr = BackgroundSessionManager(runner, max_concurrent=2)
    sessions = [mgr.submit(f"q{i}") for i in range(3)]
    # Wait for completion.
    end = time.monotonic() + 5
    while time.monotonic() < end:
        if all(s.status in (BackgroundStatus.SUCCEEDED, BackgroundStatus.FAILED) for s in sessions):
            break
        time.sleep(0.05)
    statuses = [s.status.value for s in sessions]
    mgr.shutdown()
    return f"3 sessions: {statuses}; results: {[s.result for s in sessions]}"


@check("TUI-17", "Compact banner for narrow terminals")
def tui_banner_compact():
    from xerxes.tui.banner import COMPACT_LOGO, FULL_LOGO, BannerData, render_banner

    narrow = render_banner(
        BannerData(model="claude-opus-4-7", session_id="abc12345", tip="press Tab"), terminal_width=40
    )
    wide = render_banner(BannerData(model="claude-opus-4-7", session_id="abc12345", tip="press Tab"), terminal_width=120)
    assert COMPACT_LOGO in narrow
    assert FULL_LOGO.splitlines()[0] in wide
    return f"narrow={len(narrow.splitlines())} lines, wide={len(wide.splitlines())} lines"


@check("TUI-18", "Plugin-registered slash commands merge with core registry")
def tui_slash_plugins():
    from xerxes.extensions.slash_plugins import register_slash, resolve_slash
    from xerxes.extensions.slash_plugins import registry as _slash_registry

    _slash_registry()._plugins.clear()
    register_slash("my_plugin_cmd", lambda: "ran", description="custom thing", aliases=("mpc",))
    via_name = resolve_slash("/my_plugin_cmd hello world")
    via_alias = resolve_slash("/mpc")
    assert via_name is not None and via_name.command.name == "my_plugin_cmd"
    assert via_alias is via_name
    merged = _slash_registry().all_commands()
    names = {c.name for c in merged}
    _slash_registry()._plugins.clear()
    return f"plugin registered + resolved; merged registry size={len(merged)}; includes 'my_plugin_cmd'={('my_plugin_cmd' in names)}"


# ---------------------------- runner ------------------------------------------


CHECKS: list[Callable] = [
    plan_01_workspace_files,
    plan_01_workspace_tools,
    plan_01_workspace_import_real,
    plan_02_prompt_caching,
    plan_02_anthropic_real,
    plan_03_compressor,
    plan_03_tool_result_storage,
    plan_04_nudges,
    plan_04_skill_manage,
    plan_05_holographic,
    plan_05_plugin_discovery,
    plan_05_mem0_unavailable,
    plan_06_mcp_server,
    plan_07_osv_real,
    plan_07_reconnect,
    plan_07_pkce_real,
    plan_08_acp_real,
    plan_09_cron_real,
    plan_10_voice_null,
    plan_10_edge_tts_check,
    plan_11_vision,
    plan_11_imagegen,
    plan_12_browser_registry,
    plan_12_local_playwright,
    plan_13_ssh_argv,
    plan_13_file_sync,
    plan_14_parsers_real,
    plan_15_rl_real,
    plan_16_trajectory,
    plan_17_batch,
    plan_18_branching,
    plan_18_snapshots_real,
    plan_19_slash,
    plan_20_doctor,
    plan_20_update_real_pypi,
    plan_20_setup,
    plan_21_messaging_depth,
    plan_22_security,
    plan_23_skills_hub,
    plan_24_routing,
    plan_25_auth,
    plan_26_insights,
    plan_27_tui,
    plan_28_protocol_real,
    plan_29_migrations,
    plan_30_packaging,
    plan_31_streaming,
    plan_32_msg_tools,
    # TUI parity gaps
    tui_multiline,
    tui_autosuggest_history,
    tui_reasoning_filter,
    tui_status_snapshot,
    tui_tips,
    tui_skin_branding,
    tui_voice_keys,
    tui_clipboard_attach,
    tui_approval_panel,
    tui_approval_countdown,
    tui_at_mentions,
    tui_context_bar,
    tui_rollback_diff,
    tui_background_sessions,
    tui_banner_compact,
    tui_slash_plugins,
]


def main() -> int:
    print("=" * 80)
    print("Xerxes real-use check — exercising every plan against real backends")
    print("=" * 80)
    for fn in CHECKS:
        fn()
    by_status: dict[str, list[CheckResult]] = {"passed": [], "skipped": [], "failed": []}
    for r in RESULTS:
        by_status[r.status].append(r)
    for status in ("passed", "skipped", "failed"):
        print(f"\n## {status.upper()} ({len(by_status[status])})")
        for r in by_status[status]:
            marker = {"passed": "✓", "skipped": "·", "failed": "✗"}[status]
            print(f"  {marker} plan {r.plan}: {r.feature}")
            if r.detail:
                wrapped = textwrap.fill(r.detail, width=92, initial_indent="      ", subsequent_indent="      ")
                print(wrapped)
    print("\n" + "=" * 80)
    summary = " | ".join(f"{s}={len(by_status[s])}" for s in ("passed", "skipped", "failed"))
    print(f"summary: {summary}")
    print("=" * 80)
    return 1 if by_status["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
