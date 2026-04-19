# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Daemon configuration.

Loads configuration from ``~/.xerxes/daemon/config.json`` with environment
variable overrides. Provides the :class:`DaemonConfig` dataclass holding
all runtime parameters for the daemon process.
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
    """Load daemon config from ~/.xerxes/daemon/config.json + env vars."""
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
