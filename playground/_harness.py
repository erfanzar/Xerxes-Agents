# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Shared driver for the eval playgrounds.

IMPORTANT: importing this module sets ``XERXES_HOME`` to a private, per-process
directory BEFORE any ``xerxes`` module is imported — so each eval run gets its
OWN daemon socket / pid / memory / sessions and CANNOT collide with your normal
`xerxes` TUI or another eval run. The active provider profile is copied in so the
same model/creds are used. Always ``import _harness`` (or `from _harness import …`)
*before* importing anything from ``xerxes`` in the eval scripts.

Exposes: ``Agent`` (async BridgeClient wrapper with per-turn telemetry),
``cleanup`` (kills this run's daemon + removes its home), ``EVAL_HOME``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src" / "python"))


def _real_home() -> Path:
    h = os.environ.get("XERXES_HOME", "").strip()
    return Path(h).expanduser() if h else Path.home() / ".xerxes"


# ---- isolate the home BEFORE importing xerxes (paths freeze at import) -------
_EVAL_BASE = Path(__file__).resolve().parent / ".eval_home"
EVAL_HOME = _EVAL_BASE / f"run-{os.getpid()}"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _sweep_stale_homes() -> None:
    """Remove leftover run-* homes whose daemon is no longer alive."""
    if not _EVAL_BASE.is_dir():
        return
    for d in _EVAL_BASE.glob("run-*"):
        try:
            pid = int((d / "daemon" / "daemon.pid").read_text().strip())
            if _pid_alive(pid):
                continue
        except Exception:
            pass
        try:
            shutil.rmtree(d)
        except OSError:
            pass


def _setup_home() -> None:
    _sweep_stale_homes()
    real = _real_home()
    EVAL_HOME.mkdir(parents=True, exist_ok=True)
    src = real / "profiles.json"
    if src.exists():  # carry over the active provider profile (model + creds)
        shutil.copy2(src, EVAL_HOME / "profiles.json")


_setup_home()
os.environ["XERXES_HOME"] = str(EVAL_HOME)
# The daemon also binds a TCP WebSocket gateway on a fixed port (default 11996);
# two daemons can't share it. The eval only uses the Unix socket, so bind the
# gateway on an ephemeral port (0) — lets the eval daemon coexist with your TUI.
os.environ["XERXES_DAEMON_PORT"] = "0"

# Now safe to import xerxes — DAEMON_DIR / PROFILES_DIR resolve under EVAL_HOME.
from xerxes.streaming.wire_events import (  # noqa: E402
    ApprovalRequest,
    InitDone,
    Notification,
    StatusUpdate,
    TextPart,
    ToolCall,
    TurnEnd,
)
from xerxes.tui.engine import BridgeClient  # noqa: E402


class Agent:
    """Async BridgeClient wrapper with per-turn telemetry, on the isolated daemon."""

    def __init__(self) -> None:
        self.client = BridgeClient()
        self.model = "?"

    async def start(self) -> None:
        self.client.spawn()
        await self.client.initialize(permission_mode="accept-all")
        self.model = await self._drain(2.5)

    async def fresh_session(self) -> None:
        await self.client.initialize(permission_mode="accept-all")
        await self._drain(1.5)

    async def _drain(self, secs: float) -> str:
        model = self.model
        end = time.time() + secs
        while time.time() < end:
            try:
                ev = await asyncio.wait_for(self.client._event_queue.get(), timeout=end - time.time())
            except TimeoutError:
                break
            if isinstance(ev, InitDone) and ev.model:
                model = ev.model
        return model

    async def turn(self, prompt: str, timeout: float = 450.0, retries: int = 2) -> dict:
        """Run a turn, retrying on transient provider/network errors.

        A per-turn *timeout* (model too slow / stuck) is NOT retried — that's a
        genuine failure. Connection/rate-limit/5xx errors ARE retried with
        backoff, so a flaky API doesn't get scored as a model failure."""
        res = await self._one_turn(prompt, timeout)
        attempt = 0
        while attempt < retries and _is_transient(res):
            attempt += 1
            await asyncio.sleep(2.0 * attempt)
            res = await self._one_turn(prompt, timeout)
        if attempt:
            res["retries"] = attempt
        return res

    async def _one_turn(self, prompt: str, timeout: float = 450.0) -> dict:
        t0 = time.time()
        text: list[str] = []
        tools: list[str] = []
        ctx = 0
        err = None
        await self.client.query(prompt)
        while True:
            remaining = t0 + timeout - time.time()
            if remaining <= 0:
                err = "timeout"
                break
            try:
                ev = await asyncio.wait_for(self.client._event_queue.get(), timeout=remaining)
            except TimeoutError:
                err = "timeout"
                break
            if isinstance(ev, TextPart):
                text.append(ev.text)
            elif isinstance(ev, ToolCall):
                tools.append(ev.name)
            elif isinstance(ev, ApprovalRequest):
                await self.client.permission_response(ev.id, "approve")
            elif isinstance(ev, StatusUpdate):
                ctx = getattr(ev, "context_tokens", ctx) or ctx
            elif isinstance(ev, Notification) and getattr(ev, "severity", "") == "error":
                err = (ev.body or ev.title or "error")[:160]
            elif isinstance(ev, TurnEnd):
                break
        return {"text": "".join(text).strip(), "tools": tools, "latency": time.time() - t0, "ctx": ctx, "error": err}

    def close(self) -> None:
        self.client.close()


def _kill_eval_daemon() -> None:
    """Kill ONLY this run's daemon (by its pid file), waiting for it to exit."""
    try:
        pid = int((EVAL_HOME / "daemon" / "daemon.pid").read_text().strip())
    except Exception:
        return
    if not _pid_alive(pid):
        return
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return
    for _ in range(40):  # up to ~4s for a clean shutdown
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    if _pid_alive(pid):
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass


def cleanup(remove_home: bool = True) -> None:
    """Stop this run's daemon and (optionally) delete its isolated home."""
    _kill_eval_daemon()
    if remove_home:
        for _ in range(3):
            try:
                shutil.rmtree(EVAL_HOME)
                return
            except OSError:
                time.sleep(0.3)


import re as _re  # noqa: E402

# Substrings that mark a transient provider/network error worth retrying. Only
# matched inside an actual error field or an "[Error: …]" reply envelope, so a
# normal answer that happens to contain "503" won't trigger a retry.
_TRANSIENT_MARKERS = (
    "connection",
    "reset by peer",
    "econnreset",
    "broken pipe",
    "remote disconnected",
    "rate limit",
    "ratelimit",
    "429",
    "502",
    "503",
    "504",
    "service unavailable",
    "unavailable",
    "overloaded",
    "temporarily",
    "timed out",
    "read timeout",
)


def _is_transient(res: dict) -> bool:
    """True if the turn failed with a retryable provider/network error (not a turn timeout)."""
    err = (res.get("error") or "").lower()
    if err == "timeout":
        return False  # the per-turn cap (model too slow / stuck) — genuine failure
    text = (res.get("text") or "").strip().lower()
    envelope = err or (text if text.startswith("[error:") else "")
    return bool(envelope) and any(m in envelope for m in _TRANSIENT_MARKERS)


_judge_llm = None


def _get_judge():
    """Lazily build a direct provider client (active profile) for failure diagnosis."""
    global _judge_llm
    if _judge_llm is None:
        from xerxes.bridge import profiles
        from xerxes.llms import LLMConfig, create_llm

        p = profiles.get_active_profile() or {}
        _judge_llm = create_llm(
            p.get("provider") or "openai",
            LLMConfig(
                model=p.get("model", ""), base_url=p.get("base_url", ""), api_key=p.get("api_key", ""), timeout=120
            ),
        )
    return _judge_llm


async def diagnose(
    *, task_prompt: str, why_hard: str, reply: str, tools: list[str], grader_detail: str, error: str | None
) -> str:
    """Return a one-sentence explanation of WHY the model failed this task.

    Timeouts/errors are explained deterministically; grader failures are
    diagnosed by a direct LLM call over (task, the agent's answer, tools, and
    the grader's expected-vs-actual). Falls back to the grader detail if the
    diagnosis call is unavailable. Reasoning effort is forced low to keep it cheap."""
    if error == "timeout":
        return "timed out — the model did not finish within the turn limit (too slow, or stuck looping)."
    if error:
        return f"the turn errored before completing: {error}"
    if reply.strip().startswith("[Error:"):
        return f"API/transport error that persisted through retries (not a model mistake): {reply.strip()[:160]}"
    prompt = (
        "An AI agent FAILED an automated eval task. In ONE concise sentence, state the SPECIFIC mistake "
        "(start with the concrete error; no preamble, do not restate the task).\n\n"
        f"TASK:\n{task_prompt}\n\nWHY IT'S TRICKY:\n{why_hard[:500]}\n\n"
        f"THE AGENT'S FINAL ANSWER:\n{reply[:900]}\n\nTOOLS USED: {tools}\n\n"
        f"GRADER (expected vs. produced):\n{grader_detail}\n\nOne sentence:"
    )
    try:
        llm = _get_judge()
        resp = await llm.generate_completion(prompt, max_tokens=1200, temperature=0, reasoning_effort="low")
        why = _re.sub(r"(?s)<think>.*?</think>", "", llm.extract_content(resp)).strip()
        return why[:320] or f"(model gave no diagnosis); grader: {grader_detail[:120]}"
    except Exception as exc:
        return f"(auto-diagnosis unavailable: {type(exc).__name__}); grader: {grader_detail[:140]}"


__all__ = ["EVAL_HOME", "Agent", "cleanup", "diagnose"]
