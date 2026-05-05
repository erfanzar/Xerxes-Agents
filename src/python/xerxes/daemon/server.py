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
"""Main daemon server orchestrating runtime bootstrap, gateways, and tasks.

``DaemonServer`` ties together the WebSocket gateway, Unix socket channel,
agent runtime, and task execution loop. The ``main`` entry point parses CLI
arguments and starts the event loop.
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..bridge import profiles
from ..runtime.bootstrap import bootstrap
from ..runtime.bridge import build_tool_executor, populate_registry
from ..runtime.config_context import set_config as set_global_config
from .config import DaemonConfig
from .gateway import WebSocketGateway
from .log import DaemonLogger
from .socket_channel import SocketChannel
from .task_runner import Task, create_task, run_task


class DaemonServer:
    """Background agent server with WebSocket and Unix socket interfaces.

    Args:
        config (DaemonConfig): IN: Fully populated configuration. OUT: Used
            to initialize logger, gateway, socket channel, and thread pool.
    """

    def __init__(self, config: DaemonConfig) -> None:
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (DaemonConfig): IN: config. OUT: Consumed during execution."""
        self.config = config
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (DaemonConfig): IN: config. OUT: Consumed during execution."""
        """Initialize the instance.

        Args:
            self: IN: The instance. OUT: Used for attribute access.
            config (DaemonConfig): IN: config. OUT: Consumed during execution."""
        self.logger = DaemonLogger(config.log_dir)
        self.tasks: dict[str, Task] = {}
        self._pool = ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)
        self._shutdown = False

        self._runtime_config: dict[str, Any] = {}
        self._system_prompt = ""
        self._tool_executor: Any = None
        self._tool_schemas: list[dict[str, Any]] = []

        self._gateway = WebSocketGateway(config.ws_host, config.ws_port, auth_token=config.auth_token or None)
        self._socket = SocketChannel(config.socket_path)

    def _bootstrap(self) -> None:
        """Load profile, verify model, and initialise the agent runtime.

        Returns:
            None: OUT: Sets ``_runtime_config``, ``_system_prompt``,
            ``_tool_executor``, and ``_tool_schemas``.

        Raises:
            SystemExit: OUT: Exits with code 1 if no profile or model is
                configured.
        """

        self.logger.info("Bootstrapping agent runtime")

        profile = profiles.get_active_profile()
        if profile:
            base_url = profile.get("base_url", "")
            api_key = profile.get("api_key", "")
            saved_model = profile.get("model", "")

            model = saved_model
            if base_url:
                available = profiles.fetch_models(base_url, api_key)
                if available:
                    if saved_model and saved_model in available:
                        self.logger.info("Model verified", model=saved_model)
                    else:
                        model = available[0]
                        self.logger.info(
                            "Saved model not available, auto-selected",
                            saved=saved_model,
                            selected=model,
                            available=len(available),
                        )

                        profiles.save_profile(
                            name=profile["name"],
                            base_url=base_url,
                            api_key=api_key,
                            model=model,
                            provider=profile.get("provider", ""),
                        )
                else:
                    self.logger.info("Could not fetch models, using saved", model=saved_model)

            self._runtime_config = {
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "permission_mode": "accept-all",
            }
            for k, v in profile.get("sampling", {}).items():
                self._runtime_config[k] = v
            self.logger.info(
                "Profile loaded",
                model=model,
                provider=profile.get("provider", ""),
            )
        elif self.config.model:
            self._runtime_config = {
                "model": self.config.model,
                "base_url": self.config.base_url,
                "api_key": self.config.api_key,
                "permission_mode": "accept-all",
            }
        else:
            self.logger.error("No profile configured. Run `xerxes` and use /provider first.")
            sys.exit(1)

        set_global_config(self._runtime_config)

        boot = bootstrap(model=self._runtime_config.get("model", ""))
        self._system_prompt = boot.system_prompt

        registry = populate_registry()
        self._tool_executor = build_tool_executor(registry=registry)
        self._tool_schemas = registry.tool_schemas()

        self.logger.info(
            "Runtime ready",
            tools=len(self._tool_schemas),
            model=self._runtime_config.get("model", ""),
        )

    async def run(self) -> None:
        """Start gateways and enter the main keep-alive loop.

        Returns:
            None: OUT: Blocks until ``shutdown()`` sets ``_shutdown``.
        """

        self._bootstrap()
        self._write_pid()

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        await self._gateway.start(
            submit_fn=self._submit_ws,
            list_fn=self._list_tasks,
            status_fn=self._status,
            cancel_fn=self._cancel_task,
        )
        await self._socket.start(
            submit_fn=self._submit_socket,
            list_fn=self._list_tasks,
            status_fn=self._status,
        )

        model = self._runtime_config.get("model", "(none)")
        self.logger.info(f"Daemon running — ws://{self.config.ws_host}:{self.config.ws_port} — model: {model}")

        while not self._shutdown:
            await asyncio.sleep(1)

        self.logger.info("Daemon stopped")

    async def shutdown(self) -> None:
        """Cancel running tasks, stop channels, and clean up resources.

        Returns:
            None: OUT: Idempotent; safe to call multiple times.
        """

        if self._shutdown:
            return
        self._shutdown = True
        self.logger.info("Shutting down...")

        for task in self.tasks.values():
            if task.status == "running":
                task.cancel()

        await self._socket.stop()
        await self._gateway.stop()
        self._pool.shutdown(wait=False)
        self._remove_pid()
        self.logger.close()

    async def _submit_ws(
        self,
        prompt: str,
        source: str,
        on_event: Callable[[str, dict[str, Any]], None],
    ) -> str:
        """Submit a task from the WebSocket gateway.

        Args:
            prompt (str): IN: User prompt text. OUT: Passed to ``run_task``.
            source (str): IN: Source label (e.g. ``"ws:{msg_id}"``). OUT:
                Stored on the ``Task``.
            on_event (Callable[[str, dict[str, Any]], None]): IN: Callback for
                streaming events. OUT: Forwarded to ``run_task``.

        Returns:
            str: OUT: Task result text.
        """

        task = create_task(prompt, source)
        self.tasks[task.id] = task
        self.logger.info("Task submitted", task_id=task.id, source=source, prompt=prompt[:100])

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._pool,
            run_task,
            task,
            dict(self._runtime_config),
            self._system_prompt,
            self._tool_executor,
            self._tool_schemas,
            on_event,
        )

        self.logger.info("Task completed", task_id=task.id, status=task.status)
        return result

    async def _submit_socket(self, prompt: str, source: str) -> str:
        """Submit a task from the Unix socket channel.

        Args:
            prompt (str): IN: User prompt text. OUT: Passed to ``run_task``.
            source (str): IN: Source label (e.g. ``"socket"``). OUT: Stored on
                the ``Task``.

        Returns:
            str: OUT: Task result text.
        """

        task = create_task(prompt, source)
        self.tasks[task.id] = task
        self.logger.info("Task submitted", task_id=task.id, source=source, prompt=prompt[:100])

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self._pool,
            run_task,
            task,
            dict(self._runtime_config),
            self._system_prompt,
            self._tool_executor,
            self._tool_schemas,
            None,
        )

        self.logger.info("Task completed", task_id=task.id, status=task.status)
        return result

    def _cancel_task(self, task_id: str) -> bool:
        """Cancel a running task by ID.

        Args:
            task_id (str): IN: Task identifier. OUT: Looked up in
                ``self.tasks``.

        Returns:
            bool: OUT: ``True`` if the task was running and cancelled.
        """
        task = self.tasks.get(task_id)
        if task and task.status == "running":
            task.cancel()
            return True
        return False

    def _list_tasks(self) -> list[dict[str, Any]]:
        """Return a snapshot of all known tasks.

        Returns:
            list[dict[str, Any]]: OUT: Dicts with ``id``, truncated ``prompt``,
            ``source``, ``status``, and ``created_at``.
        """
        return [
            {
                "id": t.id,
                "prompt": t.prompt[:80],
                "source": t.source,
                "status": t.status,
                "created_at": t.created_at,
            }
            for t in self.tasks.values()
        ]

    def _status(self) -> dict[str, Any]:
        """Return daemon health and runtime status.

        Returns:
            dict[str, Any]: OUT: Dict with ``status``, ``pid``, ``model``,
            ``active_tasks``, ``total_tasks``, and ``ws`` URI.
        """
        active = sum(1 for t in self.tasks.values() if t.status == "running")
        return {
            "status": "running",
            "pid": os.getpid(),
            "model": self._runtime_config.get("model", ""),
            "active_tasks": active,
            "total_tasks": len(self.tasks),
            "ws": f"ws://{self.config.ws_host}:{self.config.ws_port}",
        }

    def _write_pid(self) -> None:
        """Persist the current process ID to the configured PID file.

        Returns:
            None: OUT: File created or overwritten with the current PID.
        """
        pid_path = Path(self.config.pid_file).expanduser()
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        """Delete the PID file if it exists.

        Returns:
            None: OUT: PID file is removed; errors are ignored.
        """
        pid_path = Path(self.config.pid_file).expanduser()
        pid_path.unlink(missing_ok=True)


def main() -> None:
    """CLI entry point for the Xerxes daemon.

    Parses ``--project-dir``, ``--host``, and ``--port`` arguments, loads
    configuration, overrides with CLI values, and starts ``DaemonServer``.

    Returns:
        None: OUT: Blocks until the daemon shuts down.
    """

    import argparse

    parser = argparse.ArgumentParser(description="Xerxes daemon — background agent")
    parser.add_argument("--project-dir", default="", help="Working directory")
    parser.add_argument("--host", default="", help="WebSocket host")
    parser.add_argument("--port", type=int, default=0, help="WebSocket port")
    args = parser.parse_args()

    from .config import load_config

    config = load_config(project_dir=args.project_dir)
    if args.host:
        config.ws_host = args.host
    if args.port:
        config.ws_port = args.port

    server = DaemonServer(config)
    asyncio.run(server.run())
