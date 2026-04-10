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

"""System service installer for the Calute daemon.

Generates and manages platform-specific service definitions:

- **macOS**: ``~/Library/LaunchAgents/com.calute.daemon.plist`` via ``launchctl``
- **Linux**: ``~/.config/systemd/user/calute-daemon.service`` via ``systemctl --user``

The installed service starts the daemon at login and restarts it on failure.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def _python_path() -> str:
    return sys.executable


def _daemon_command() -> str:
    return f"{_python_path()} -m calute.daemon"


# ── macOS launchd ─────────────────────────────────────────────────────────

PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / "com.calute.daemon.plist"

PLIST_TEMPLATE = """\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.calute.daemon</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>-m</string>
        <string>calute.daemon</string>
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


# ── Linux systemd ─────────────────────────────────────────────────────────

SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"
SYSTEMD_PATH = SYSTEMD_DIR / "calute-daemon.service"

SYSTEMD_TEMPLATE = """\
[Unit]
Description=Calute Daemon — Background Agent
After=network.target

[Service]
Type=simple
ExecStart={python} -m calute.daemon
WorkingDirectory={cwd}
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
"""


def install(project_dir: str = "", log_dir: str = "") -> str:
    """Install the daemon as a system service."""
    cwd = project_dir or os.getcwd()
    log_dir = log_dir or str(Path.home() / ".calute" / "daemon" / "logs")
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
        subprocess.run(["systemctl", "--user", "enable", "calute-daemon"], check=True)
        subprocess.run(["systemctl", "--user", "start", "calute-daemon"], check=True)
        return f"Installed: {SYSTEMD_PATH}\nStarted via systemctl --user."

    else:
        return f"Unsupported platform: {system}. Run manually with `python -m calute.daemon`."


def uninstall() -> str:
    """Remove the system service."""
    system = platform.system()

    if system == "Darwin":
        if PLIST_PATH.exists():
            subprocess.run(["launchctl", "unload", str(PLIST_PATH)], check=False)
            PLIST_PATH.unlink()
            return f"Removed: {PLIST_PATH}"
        return "No launchd service found."

    elif system == "Linux":
        if SYSTEMD_PATH.exists():
            subprocess.run(["systemctl", "--user", "stop", "calute-daemon"], check=False)
            subprocess.run(["systemctl", "--user", "disable", "calute-daemon"], check=False)
            SYSTEMD_PATH.unlink()
            subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
            return f"Removed: {SYSTEMD_PATH}"
        return "No systemd service found."

    return f"Unsupported platform: {system}."


def status() -> str:
    """Check if the daemon service is running."""
    system = platform.system()

    if system == "Darwin":
        result = subprocess.run(
            ["launchctl", "list", "com.calute.daemon"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return f"Running (launchd)\n{result.stdout.strip()}"
        return "Not running (no launchd service)"

    elif system == "Linux":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "calute-daemon"],
            capture_output=True,
            text=True,
        )
        state = result.stdout.strip()
        if state == "active":
            return "Running (systemd)"
        return f"Not running (systemd: {state})"

    # Fallback: check PID file.
    pid_file = Path.home() / ".calute" / "daemon" / "daemon.pid"
    if pid_file.exists():
        pid = int(pid_file.read_text().strip())
        try:
            os.kill(pid, 0)
            return f"Running (PID: {pid})"
        except OSError:
            return f"Stale PID file (PID {pid} not running)"
    return "Not running"
