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
"""System service integration for the Xerxes daemon.

Provides ``install``, ``uninstall``, and ``status`` helpers that create
launchd (macOS) or systemd (Linux) service definitions and manage their
lifecycle.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def _python_path() -> str:
    """Return the path of the current Python interpreter.

    Returns:
        str: OUT: ``sys.executable``.
    """
    return sys.executable


def _daemon_command() -> str:
    """Return the CLI command used to start the daemon.

    Returns:
        str: OUT: ``"{python} -m xerxes.daemon"``.
    """
    return f"{_python_path()} -m xerxes.daemon"


PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / "com.xerxes.daemon.plist"

PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.xerxes.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>xerxes.daemon</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{cwd}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>{log_dir}/daemon-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>{log_dir}/daemon-stderr.log</string>
</dict>
</plist>
"""

SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"
SYSTEMD_PATH = SYSTEMD_DIR / "xerxes-daemon.service"

SYSTEMD_TEMPLATE = """\
[Unit]
Description=Xerxes Daemon — Background Agent
After=network.target

[Service]
Type=simple
ExecStart={python} -m xerxes.daemon
WorkingDirectory={cwd}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def install(project_dir: str = "", log_dir: str = "") -> str:
    """Install and start the daemon as a system service.

    On macOS creates a launchd plist; on Linux creates a systemd user unit.

    Args:
        project_dir (str): IN: Working directory for the service. OUT:
            Defaults to ``os.getcwd()`` and inserted into the service file.
        log_dir (str): IN: Directory for stdout/stderr logs. OUT: Defaults to
            the daemon logs subdirectory and created if missing.

    Returns:
        str: OUT: Human-readable result message.
    """

    cwd = project_dir or os.getcwd()
    if not log_dir:
        from xerxes.core.paths import xerxes_subdir

        log_dir = str(xerxes_subdir("daemon", "logs"))
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    system = platform.system()

    if system == "Darwin":
        plist = PLIST_TEMPLATE.format(
            python=_python_path(),
            cwd=cwd,
            log_dir=log_dir,
        )
        PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        PLIST_PATH.write_text(plist)
        subprocess.run(["launchctl", "load", str(PLIST_PATH)], check=True)
        return f"Installed: {PLIST_PATH}\nStarted via launchctl."

    elif system == "Linux":
        unit = SYSTEMD_TEMPLATE.format(
            python=_python_path(),
            cwd=cwd,
        )
        SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
        SYSTEMD_PATH.write_text(unit)
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", "xerxes-daemon"], check=True)
        subprocess.run(["systemctl", "--user", "start", "xerxes-daemon"], check=True)
        return f"Installed: {SYSTEMD_PATH}\nStarted via systemctl --user."

    else:
        return f"Unsupported platform: {system}. Run manually with `python -m xerxes.daemon`."


def uninstall() -> str:
    """Stop and remove the daemon system service.

    Returns:
        str: OUT: Human-readable result message.
    """

    system = platform.system()

    if system == "Darwin":
        if PLIST_PATH.exists():
            subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
            PLIST_PATH.unlink()
            return f"Removed: {PLIST_PATH}"
        return "No launchd service found."

    elif system == "Linux":
        if SYSTEMD_PATH.exists():
            subprocess.run(["systemctl", "--user", "stop", "xerxes-daemon"], check=False)
            subprocess.run(["systemctl", "--user", "disable", "xerxes-daemon"], check=False)
            SYSTEMD_PATH.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            return f"Removed: {SYSTEMD_PATH}"
        return "No systemd service found."

    return f"Unsupported platform: {system}."


def status() -> str:
    """Query whether the daemon service is currently running.

    Returns:
        str: OUT: Human-readable status description.
    """

    system = platform.system()

    if system == "Darwin":
        result = subprocess.run(
            ["launchctl", "list", "com.xerxes.daemon"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return f"Running (launchd)\n{result.stdout.strip()}"
        return "Not running (no launchd service)"

    elif system == "Linux":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "xerxes-daemon"],
            capture_output=True,
            text=True,
        )
        state = result.stdout.strip()
        if state == "active":
            return "Running (systemd)"
        return f"Not running (systemd: {state})"

    from xerxes.core.paths import xerxes_subdir

    pid_file = xerxes_subdir("daemon", "daemon.pid")
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            return f"Running (PID: {pid})"
        except OSError:
            return f"Stale PID file (PID {pid} not running)"
    return "Not running"
