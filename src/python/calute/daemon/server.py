# Copyright 2025 The EasyDeL/Calute Author @erfanzar (Erfan Zare Chavoshi).
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

"""Core daemon server — persistent background agent.

Runs an asyncio event loop that accepts tasks from a WebSocket gateway
and a Unix domain socket, dispatches them to a thread-pool-based
:class:`~calute.agents.subagent_manager.SubAgentManager`, and streams
results back to connected clients.
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
    """Persistent background agent server.

    Runs an asyncio event loop with:
    - WebSocket gateway for external clients
    - Unix socket for local `calute send` commands
    - ThreadPoolExecutor for concurrent agent tasks
    """

    def __init__(self, config: DaemonConfig) -> None:
        self.config = config
        self.logger = DaemonLogger(config.log_dir)
        self.tasks: dict[str, Task] = {}
        self._pool = ThreadPoolExecutor(max_workers=config.max_concurrent_tasks)
        self._shutdown = False

        # Runtime state (initialized in _bootstrap).
        self._runtime_config: dict[str, Any] = {}
        self._system_prompt = ""
        self._tool_executor: Any = None
        self._tool_schemas: list[dict[str, Any]] = []

        # Channels.
        self._gateway = WebSocketGateway(config.ws_host, config.ws_port)
        self._socket = SocketChannel(config.socket_path)

    def _bootstrap(self) -> None:
        """Initialize the agent runtime — same pattern as BridgeServer.handle_init()."""
        self.logger.info("Bootstrapping agent runtime")

        # Load active profile.
        profile = profiles.get_active_profile()
        if profile:
            base_url = profile.get("base_url", "")
            api_key = profile.get("api_key", "")
            saved_model = profile.get("model", "")

            # Verify the saved model actually exists on the provider.
            model = saved_model
            if base_url:
                available = profiles.fetch_models(base_url, api_key)
                if available:
                    if saved_model and saved_model in available:
                        self.logger.info("Model verified", model=saved_model)
                    else:
                        # Saved model not available — pick the first one.
                        model = available[0]
                        self.logger.info(
                            "Saved model not available, auto-selected",
                            saved=saved_model,
                            selected=model,
                            available=len(available),
                        )
                        # Update the profile so it stays correct.
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
            self.logger.error("No profile configured. Run `calute` and use /provider first.")
            sys.exit(1)

        # Set global config so sub-agents inherit provider settings.
        set_global_config(self._runtime_config)

        # Bootstrap system prompt + tools.
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
        """Main daemon loop."""
        self._bootstrap()
        self._write_pid()

        # Setup signal handlers.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))

        # Start channels.
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

        # Keep alive until shutdown.
        while not self._shutdown:
            await asyncio.sleep(1)

        self.logger.info("Daemon stopped")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._shutdown:
            return
        self._shutdown = True
        self.logger.info("Shutting down...")

        # Cancel running tasks.
        for task in self.tasks.values():
            if task.status == "running":
                task.cancel()

        await self._socket.stop()
        await self._gateway.stop()
        self._pool.shutdown(wait=False)
        self._remove_pid()
        self.logger.close()

    # ── Task submission ───────────────────────────────────────────────

    async def _submit_ws(
        self,
        prompt: str,
        source: str,
        on_event: Callable[[str, dict[str, Any]], None],
    ) -> str:
        """Submit a task from the WebSocket gateway."""
        task = create_task(prompt, source)
        self.tasks[task.id] = task
        self.logger.info("Task submitted", task_id=task.id, source=source, prompt=prompt[:100])

        # Run in thread pool (the streaming loop is synchronous).
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
        """Submit a task from the Unix socket (no streaming callback)."""
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
            None,  # No event callback for socket.
        )

        self.logger.info("Task completed", task_id=task.id, status=task.status)
        return result

    def _cancel_task(self, task_id: str) -> bool:
        task = self.tasks.get(task_id)
        if task and task.status == "running":
            task.cancel()
            return True
        return False

    def _list_tasks(self) -> list[dict[str, Any]]:
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
        active = sum(1 for t in self.tasks.values() if t.status == "running")
        return {
            "status": "running",
            "pid": os.getpid(),
            "model": self._runtime_config.get("model", ""),
            "active_tasks": active,
            "total_tasks": len(self.tasks),
            "ws": f"ws://{self.config.ws_host}:{self.config.ws_port}",
        }

    # ── PID file ──────────────────────────────────────────────────────

    def _write_pid(self) -> None:
        pid_path = Path(self.config.pid_file).expanduser()
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))

    def _remove_pid(self) -> None:
        pid_path = Path(self.config.pid_file).expanduser()
        pid_path.unlink(missing_ok=True)


def main() -> None:
    """CLI entry point for the daemon."""
    import argparse

    parser = argparse.ArgumentParser(description="Calute daemon — background agent")
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
