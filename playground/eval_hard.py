#!/usr/bin/env python3
# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Hard, behaviorally-graded eval battery for the Xerxes agent.

Loads tasks from ``playground/hard_tasks.json`` and runs them through the REAL
agent on an ISOLATED daemon (see _harness — its own XERXES_HOME/socket, so it
never collides with your TUI or another run). Each task gets its own sandbox;
``{dir}`` in a prompt -> the task's working dir; ``fresh`` -> a clean session.
Graders run the agent's actual code (behavioral, anti-cheat) under a watchdog.

Usage:
    .venv/bin/python playground/eval_hard.py            # run all
    .venv/bin/python playground/eval_hard.py -v         # print prompts/replies
    .venv/bin/python playground/eval_hard.py -k memory  # filter by name substring
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import threading
from pathlib import Path

import _harness  # noqa: F401 — MUST be first: isolates XERXES_HOME before xerxes imports
from _harness import Agent, cleanup, diagnose

PLAYGROUND = Path(__file__).resolve().parent
SANDBOX = PLAYGROUND / "workspace_hard"
TASKS_FILE = PLAYGROUND / "hard_tasks.json"


def _run_check(check_source: str, results: list[dict], ws: Path, timeout: float = 90.0) -> tuple[bool, str]:
    """Exec the task's grader and call check(results, ws) under a watchdog."""
    box: dict = {}
    try:
        ns: dict = {"Path": Path}
        exec(compile(check_source, "<check>", "exec"), ns)
        fn = ns.get("check")
        if not callable(fn):
            return False, "grader missing check()"
    except Exception as exc:
        return False, f"grader compile error: {type(exc).__name__}: {exc}"

    def _target():
        try:
            box["out"] = fn(results, ws)
        except Exception as exc:
            box["out"] = (False, f"grader raised: {type(exc).__name__}: {exc}")

    th = threading.Thread(target=_target, daemon=True)
    th.start()
    th.join(timeout)
    if th.is_alive():
        return False, f"grader timed out (> {timeout:.0f}s)"
    out = box.get("out")
    if isinstance(out, tuple) and len(out) == 2:
        return bool(out[0]), str(out[1])[:120]
    return False, f"grader returned non-(bool,str): {out!r}"


def _setup_task_dir(task: dict) -> Path:
    ws = SANDBOX / task["name"]
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    for f in task.get("files", []):
        p = ws / f["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f["content"])
    return ws


async def run(verbose: bool, keyword: str | None) -> int:
    if not TASKS_FILE.exists():
        print(f"missing {TASKS_FILE} — generate it first.")
        return 2
    tasks = json.loads(TASKS_FILE.read_text())
    if keyword:
        tasks = [t for t in tasks if keyword in t["name"]]
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir(parents=True)
    os.chdir(SANDBOX)  # the isolated daemon bootstraps + works inside the sandbox

    agent = Agent()
    await agent.start()
    print(f"\n  Xerxes HARD eval — model: {agent.model}   tasks: {len(tasks)}   sandbox: {SANDBOX}\n")

    rows = []
    try:
        for t in tasks:
            ws = _setup_task_dir(t)
            rel = t["name"]  # task dir relative to the agent's cwd (SANDBOX)
            step_results = []
            last_prompt = ""
            for step in t["steps"]:
                if step.get("fresh"):
                    await agent.fresh_session()
                last_prompt = step["prompt"].replace("{dir}", rel)
                res = await agent.turn(last_prompt)
                step_results.append(res)
                if verbose:
                    print(
                        f"    · [{t['name']}] {last_prompt[:70]!r}\n        -> {res['text'][:140]!r} (tools={res['tools']}, {res['latency']:.0f}s)"
                    )
            last = step_results[-1]
            if last["error"]:
                ok, detail = False, f"ERROR: {last['error']}"
            else:
                ok, detail = _run_check(t["check_source"], step_results, ws)
            tools_used = sorted({tool for s in step_results for tool in s["tools"]})
            rows.append(
                {
                    "name": t["name"],
                    "cat": t["category"],
                    "diff": t.get("difficulty", 0),
                    "ok": ok,
                    "detail": detail,
                    "latency": sum(s["latency"] for s in step_results),
                }
            )
            print(
                f"  [{'PASS' if ok else 'FAIL'}] d{t.get('difficulty', '?')} {t['name']:<28} {t['category']:<12} {rows[-1]['latency']:5.0f}s  {detail}"
            )
            if not ok:
                why = await diagnose(
                    task_prompt=last_prompt,
                    why_hard=t.get("why_hard", ""),
                    reply=last["text"],
                    tools=tools_used,
                    grader_detail=detail,
                    error=last.get("error"),
                )
                print(f"        ↳ WHY: {why}")
                if last["text"]:
                    print(f"          model said: {last['text'][:180]!r}")
                print(f"          tools used: {tools_used or 'none'}")
    finally:
        agent.close()
        cleanup()

    passed = sum(1 for r in rows if r["ok"])
    print("\n  " + "-" * 70)
    print(
        f"  SCORE: {passed}/{len(rows)} passed  ({100 * passed // max(1, len(rows))}%)   total {sum(r['latency'] for r in rows):.0f}s   model={agent.model}"
    )
    cats: dict[str, list[bool]] = {}
    diffs: dict[int, list[bool]] = {}
    for r in rows:
        cats.setdefault(r["cat"], []).append(r["ok"])
        diffs.setdefault(r["diff"], []).append(r["ok"])
    print("  by category:   " + "  ".join(f"{c}={sum(v)}/{len(v)}" for c, v in sorted(cats.items())))
    print("  by difficulty: " + "  ".join(f"d{d}={sum(v)}/{len(v)}" for d, v in sorted(diffs.items())))
    if passed < len(rows):
        print("  FAILED: " + ", ".join(f"{r['name']} ({r['detail'][:44]})" for r in rows if not r["ok"]))
    print()
    return 0 if passed == len(rows) else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Hard scored eval battery for the Xerxes agent.")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-k", "--keyword", default=None)
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.verbose, args.keyword)))


if __name__ == "__main__":
    main()
