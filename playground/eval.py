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
"""Scored evaluation playground for the Xerxes agent (warm-up suite).

Runs the REAL agent on an ISOLATED daemon (see _harness — its own home/socket,
so it never collides with your TUI or another eval run) in a sandbox workspace,
runs a small task battery, auto-grades each, and prints a scorecard. For the
harder battery use eval_hard.py.

Usage:
    .venv/bin/python playground/eval.py            # run the suite
    .venv/bin/python playground/eval.py -v         # also print each reply
    .venv/bin/python playground/eval.py -k memory  # only tasks matching 'memory'
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
from pathlib import Path

import _harness  # noqa: F401 — MUST be first: isolates XERXES_HOME before xerxes imports
from _harness import Agent, cleanup

SANDBOX = Path(__file__).resolve().parent / "workspace"


def build_sandbox() -> None:
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir(parents=True)
    (SANDBOX / "config.txt").write_text("APP_NAME=demo\nAPI_VERSION=4.2.0\nDEBUG=false\n")
    (SANDBOX / "greeting.py").write_text('def greet():\n    return "hello"\n')
    (SANDBOX / "calc.py").write_text("def add(a, b):\n    return a - b  # BUG: should add\n")
    notes = SANDBOX / "notes"
    notes.mkdir()
    (notes / "a.txt").write_text("nothing interesting here\n")
    (notes / "b.txt").write_text("the TREASURE is buried in this file\n")
    (notes / "c.txt").write_text("just some other text\n")


def _safe_call(filename: str, func: str, *args):
    ns: dict = {}
    try:
        exec((SANDBOX / filename).read_text(), ns)
        return ns[func](*args)
    except Exception as exc:
        return f"<{type(exc).__name__}>"


def _behaves(filename: str, func: str, args: tuple, expected) -> bool:
    return _safe_call(filename, func, *args) == expected


def task_suite() -> list[dict]:
    return [
        dict(
            name="reasoning",
            cat="reason",
            steps=[("What is 17 times 23? Reply with only the number.", True)],
            check=lambda r: ("391" in r[-1]["text"], r[-1]["text"][:40]),
        ),
        dict(
            name="file_read",
            cat="tools",
            steps=[
                ("Read the file config.txt in the current directory and reply with ONLY the value of API_VERSION.", True)
            ],
            check=lambda r: ("4.2.0" in r[-1]["text"], r[-1]["text"][:40]),
        ),
        dict(
            name="file_edit",
            cat="coding",
            steps=[("Edit greeting.py so greet() returns 'goodbye' instead of 'hello', then confirm.", True)],
            check=lambda r: (
                _behaves("greeting.py", "greet", (), "goodbye"),
                "greet() -> " + repr(_safe_call("greeting.py", "greet")),
            ),
        ),
        dict(
            name="bug_fix",
            cat="coding",
            steps=[("There is a bug in calc.py: add(a, b) subtracts instead of adds. Fix it so it adds.", True)],
            check=lambda r: (
                _behaves("calc.py", "add", (2, 3), 5),
                "add(2,3) -> " + repr(_safe_call("calc.py", "add", 2, 3)),
            ),
        ),
        dict(
            name="tool_search",
            cat="tools",
            steps=[
                (
                    "In the notes/ directory, find which file contains the word TREASURE and reply with ONLY its filename.",
                    True,
                )
            ],
            check=lambda r: ("b.txt" in r[-1]["text"], f"tools={r[-1]['tools'][:4]} ans={r[-1]['text'][:24]!r}"),
        ),
        dict(
            name="shell",
            cat="tools",
            steps=[("Run the shell command `echo eval-ok-7` and reply with ONLY the command's output.", True)],
            check=lambda r: ("eval-ok-7" in r[-1]["text"], r[-1]["text"][:30]),
        ),
        dict(
            name="multiturn",
            cat="context",
            steps=[
                ("My favorite number is 7. Acknowledge in one word.", True),
                ("Multiply my favorite number by 6. Reply with only the number.", False),
            ],
            check=lambda r: ("42" in r[-1]["text"], r[-1]["text"][:30]),
        ),
        dict(
            name="memory_recall",
            cat="memory",
            steps=[
                ("Note for your memory: the deploy target is fly.io. Acknowledge in one word.", True),
                (
                    "Check your memory/journal: what is the deploy target I noted earlier? Reply with only the value.",
                    True,
                ),
            ],
            check=lambda r: ("fly.io" in r[-1]["text"].lower(), f"fresh session -> {r[-1]['text'][:24]!r}"),
        ),
    ]


async def run(verbose: bool, keyword: str | None) -> int:
    build_sandbox()
    os.chdir(SANDBOX)
    suite = [t for t in task_suite() if not keyword or keyword in t["name"]]

    agent = Agent()
    await agent.start()
    print(f"\n  Xerxes eval playground — model: {agent.model}   sandbox: {SANDBOX}\n")

    rows = []
    try:
        for t in suite:
            step_results = []
            for prompt, fresh in t["steps"]:
                if fresh:
                    await agent.fresh_session()
                res = await agent.turn(prompt)
                step_results.append(res)
                if verbose:
                    print(
                        f"    · [{t['name']}] {prompt[:60]!r}\n        -> {res['text'][:120]!r}  (tools={res['tools']}, {res['latency']:.1f}s)"
                    )
            last = step_results[-1]
            ok, detail = t["check"](step_results)
            if last["error"]:
                ok, detail = False, f"ERROR: {last['error']}"
            rows.append(
                {
                    "name": t["name"],
                    "cat": t["cat"],
                    "ok": ok,
                    "detail": detail,
                    "latency": sum(s["latency"] for s in step_results),
                }
            )
            print(f"  [{'PASS' if ok else 'FAIL'}] {t['name']:<14} {t['cat']:<8} {rows[-1]['latency']:5.1f}s  {detail}")
    finally:
        agent.close()
        cleanup()

    passed = sum(1 for r in rows if r["ok"])
    print("\n  " + "-" * 60)
    print(
        f"  SCORE: {passed}/{len(rows)} passed   ({100 * passed // max(1, len(rows))}%)   total {sum(r['latency'] for r in rows):.1f}s   model={agent.model}"
    )
    if passed < len(rows):
        print("  failed: " + ", ".join(r["name"] for r in rows if not r["ok"]))
    print()
    return 0 if passed == len(rows) else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Scored eval playground for the Xerxes agent.")
    ap.add_argument("-v", "--verbose", action="store_true", help="print each prompt/reply")
    ap.add_argument("-k", "--keyword", default=None, help="only run tasks whose name contains this")
    args = ap.parse_args()
    sys.exit(asyncio.run(run(args.verbose, args.keyword)))


if __name__ == "__main__":
    main()
