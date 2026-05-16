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
"""CLI entry point for the ``xerxes`` command.

Dispatches between three modes:

* ``xerxes telegram ...`` — start the daemon with the Telegram gateway
  enabled (token via flag or ``TELEGRAM_BOT_TOKEN`` env var).
* One-shot — a prompt provided as positional args or piped on stdin;
  streams assistant text to stdout and exits.
* Interactive — no prompt and stdin is a tty; launches ``XerxesTUI``.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys


def _resolve_one_shot_prompt(
    prompt_parts: list[str],
    *,
    stdin_is_tty: bool,
    stdin_text: str | None = None,
) -> tuple[str, bool]:
    """Decide whether to run one-shot and produce the prompt text.

    Positional CLI args take precedence; if absent, non-tty stdin is
    consumed. Returns ``(prompt, True)`` when the CLI should run
    non-interactively, ``("", False)`` to open the TUI.
    """
    parts = list(prompt_parts)
    if parts and parts[0] == "--":
        parts = parts[1:]
    if parts:
        return " ".join(parts).strip(), True
    if not stdin_is_tty:
        text = sys.stdin.read() if stdin_text is None else stdin_text
        return text.strip(), True
    return "", False


async def _run_one_shot(prompt: str, *, resume_session_id: str = "", mode: str = "code") -> None:
    """Run one prompt against a spawned daemon and stream text to stdout.

    Auto-rejects approval requests (no interactive UI) and surfaces
    error notifications on stderr. Returns when ``TurnEnd`` is seen.
    """
    from .streaming.wire_events import ApprovalRequest, Notification, TextPart, TurnEnd
    from .tui.engine import BridgeClient

    client = BridgeClient()
    wrote_text = False
    try:
        client.spawn()
        await client.initialize(
            permission_mode="accept-all",
            resume_session_id=resume_session_id,
        )
        await client.query(prompt, plan_mode=mode == "plan", mode=mode)
        async for event in client.events():
            if isinstance(event, TextPart):
                sys.stdout.write(event.text)
                sys.stdout.flush()
                wrote_text = True
            elif isinstance(event, ApprovalRequest):
                await client.permission_response(event.id, "reject")
                print(
                    f"Rejected permission request for {event.action}: non-interactive CLI has no approval UI.",
                    file=sys.stderr,
                )
            elif isinstance(event, Notification) and event.severity == "error":
                body = event.body or event.title
                if body:
                    print(body, file=sys.stderr)
            elif isinstance(event, TurnEnd):
                break
        if wrote_text:
            sys.stdout.write("\n")
            sys.stdout.flush()
    finally:
        client.close()


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and dispatch to telegram, one-shot, or TUI mode.

    Imports the TUI lazily so that ``xerxes telegram`` and one-shot
    paths avoid the heavy ``prompt_toolkit`` startup. Honours
    ``KeyboardInterrupt`` quietly.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "telegram":
        telegram_parser = argparse.ArgumentParser(
            prog="xerxes telegram", description="Start the daemon with Telegram enabled."
        )
        telegram_parser.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN", ""))
        telegram_parser.add_argument("--project-dir", default="")
        telegram_parser.add_argument("--host", default="")
        telegram_parser.add_argument("--port", type=int, default=0)
        telegram_args = telegram_parser.parse_args(argv[1:])
        if telegram_args.token:
            os.environ["TELEGRAM_BOT_TOKEN"] = telegram_args.token
        os.environ["XERXES_DAEMON_ENABLE_TELEGRAM"] = "1"

        from .daemon.config import load_config
        from .daemon.server import DaemonServer

        config = load_config(project_dir=telegram_args.project_dir)
        if telegram_args.host:
            config.ws_host = telegram_args.host
        if telegram_args.port:
            config.ws_port = telegram_args.port
        asyncio.run(DaemonServer(config).run())
        return

    from .tui import XerxesTUI

    parser = argparse.ArgumentParser(
        prog="xerxes",
        description="Xerxes — interactive AI agent in your terminal.",
    )
    parser.add_argument(
        "-r",
        "--resume",
        metavar="SESSION_ID",
        default="",
        help="Resume a previous session by id (saved under ~/.xerxes/sessions).",
    )
    parser.add_argument(
        "--mode",
        choices=("code", "researcher", "research", "plan"),
        default="code",
        help="Mode for one-shot prompts.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Ignored for one-shot prompts; they always use accept-all permissions.",
    )
    parser.add_argument(
        "prompt",
        nargs=argparse.REMAINDER,
        help="Run a one-shot prompt instead of opening the TUI.",
    )
    args = parser.parse_args(argv)
    prompt, one_shot = _resolve_one_shot_prompt(
        args.prompt,
        stdin_is_tty=sys.stdin.isatty(),
    )

    if one_shot:
        if not prompt:
            parser.error("empty prompt")
        mode = "researcher" if args.mode == "research" else args.mode
        try:
            asyncio.run(_run_one_shot(prompt, resume_session_id=args.resume, mode=mode))
        except KeyboardInterrupt:
            pass
        return

    async def _run() -> None:
        """Open the interactive TUI and await its lifecycle."""
        tui = XerxesTUI(resume_session_id=args.resume)
        async with tui:
            await tui.wait_until_done()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
