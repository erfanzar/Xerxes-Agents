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
"""Daemon configuration loading and data structures.

Defines ``DaemonConfig`` and the ``load_config`` helper that merges JSON file
settings with environment variable overrides.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from xerxes.core.paths import xerxes_subdir

DAEMON_DIR = xerxes_subdir("daemon")
CONFIG_FILE = DAEMON_DIR / "config.json"


@dataclass
class DaemonConfig:
    """Runtime configuration for the Xerxes background daemon.

    Attributes:
        ws_host (str): Host address for the WebSocket gateway. IN: IP or hostname.
            OUT: Passed to ``asyncio.start_server``.
        ws_port (int): Port number for the WebSocket gateway. IN: 1-65535.
            OUT: Bound by the gateway server.
        socket_path (str): Filesystem path for the Unix domain socket.
            IN: Absolute or relative path string. OUT: Used by ``SocketChannel``.
        pid_file (str): Path where the daemon writes its PID. IN: Path string.
            OUT: Created by ``DaemonServer._write_pid``.
        log_dir (str): Directory for JSONL log files. IN: Path string.
            OUT: Created and used by ``DaemonLogger``.
        max_concurrent_tasks (int): Thread-pool size for task execution.
            IN: Positive integer. OUT: Passed to ``ThreadPoolExecutor``.
        project_dir (str): Working directory for the daemon process.
            IN: Absolute path. OUT: Defaults to ``os.getcwd()``.
        model (str): LLM model identifier. IN: Model name string.
            OUT: Used during runtime bootstrap.
        base_url (str): Base URL for the LLM API. IN: URL string.
            OUT: Used during runtime bootstrap.
        api_key (str): API key for the LLM provider. IN: Secret string.
            OUT: Used during runtime bootstrap.
        auth_token (str): Bearer token for WebSocket authentication.
            IN: Secret string or empty. OUT: Enforced by ``WebSocketGateway``.
    """

    ws_host: str = "127.0.0.1"
    ws_port: int = 11996
    socket_path: str = str(DAEMON_DIR / "xerxes.sock")
    pid_file: str = str(DAEMON_DIR / "daemon.pid")
    log_dir: str = str(DAEMON_DIR / "logs")
    max_concurrent_tasks: int = 5
    project_dir: str = ""

    model: str = ""
    base_url: str = ""
    api_key: str = ""

    auth_token: str = ""


def load_config(project_dir: str = "") -> DaemonConfig:
    """Load daemon configuration from file and environment.

    Reads ``config.json`` inside the daemon data directory, then overrides
    select fields with environment variables.

    Args:
        project_dir (str): IN: Preferred working directory. OUT: Assigned to
            ``DaemonConfig.project_dir`` or defaults to ``os.getcwd()``.

    Returns:
        DaemonConfig: OUT: Fully populated configuration instance.
    """

    cfg = DaemonConfig(project_dir=project_dir or os.getcwd())

    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            for k, v in data.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        except (json.JSONDecodeError, OSError):
            pass

    if v := os.environ.get("XERXES_DAEMON_HOST"):
        cfg.ws_host = v
    if v := os.environ.get("XERXES_DAEMON_PORT"):
        cfg.ws_port = int(v)
    if v := os.environ.get("XERXES_MAX_TASKS"):
        cfg.max_concurrent_tasks = int(v)
    if v := os.environ.get("XERXES_DAEMON_TOKEN"):
        cfg.auth_token = v

    return cfg
