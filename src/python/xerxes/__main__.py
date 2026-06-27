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
* Interactive — no prompt and stdin is a tty; launches the TypeScript TUI.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from .core.paths import xerxes_subdir
from .runtime.interaction_modes import normalize_interaction_mode

_DISCORD_SERVICE_STOP_TIMEOUT = 5.0
_DISCORD_SERVICE_START_TIMEOUT = 3.0
_NEW_TUI_ENTRY_ENV = "XERXES_TUI_ENTRY"
_NEW_TUI_PROJECT_DIR_ENV = "XERXES_PROJECT_DIR"
_NEW_TUI_RESUME_ENV = "XERXES_TUI_RESUME"
_NEW_TUI_PYTHON_ENV = "XERXES_PYTHON"
_NEW_TUI_SOURCE_ENTRY = Path("src") / "ui-tui" / "dist" / "entry.js"
_NEW_TUI_PACKAGE_ENTRY = Path("_tui_dist") / "entry.js"
_NODE_ENV = "XERXES_NODE"
_NPM_ENV = "XERXES_NPM"
_NODE_VERSION_ENV = "XERXES_NODE_VERSION"
_NODE_DIST_BASE_URL_ENV = "XERXES_NODE_DIST_BASE_URL"
_MANAGED_NODE_VERSION = "22.17.1"


def _new_tui_entry_candidates() -> list[Path]:
    """Return candidate paths for the TypeScript TUI bundle."""
    candidates: list[Path] = []
    override = os.environ.get(_NEW_TUI_ENTRY_ENV, "").strip()
    if override:
        candidates.append(Path(override).expanduser())

    package_root = Path(__file__).resolve().parent
    candidates.append(package_root / _NEW_TUI_PACKAGE_ENTRY)

    source_root = package_root.parents[2]
    candidates.append(source_root / _NEW_TUI_SOURCE_ENTRY)
    return candidates


def _find_new_tui_entry() -> Path | None:
    """Find the built TypeScript TUI entrypoint, if available."""
    for candidate in _new_tui_entry_candidates():
        if candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_new_tui_project_dir(cwd: Path) -> Path:
    """Resolve the project root the Node TUI should use."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(cwd), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        proc = None
    if proc is not None and proc.returncode == 0:
        root = proc.stdout.strip()
        if root:
            return Path(root).resolve()
    return cwd.resolve()


def _managed_node_bin(name: str) -> Path:
    executable = f"{name}.exe" if os.name == "nt" else name
    return xerxes_subdir("node", "current", "bin", executable)


def _resolve_node_executable() -> str:
    override = os.environ.get(_NODE_ENV, "").strip()
    if override:
        return override
    managed = _managed_node_bin("node")
    if managed.is_file() and os.access(managed, os.X_OK):
        return str(managed)
    return shutil.which("node") or ""


def _resolve_npm_executable() -> str:
    override = os.environ.get(_NPM_ENV, "").strip()
    if override:
        return override
    managed = _managed_node_bin("npm")
    if managed.is_file() and os.access(managed, os.X_OK):
        return str(managed)
    return shutil.which("npm") or ""


def _run_new_tui(*, resume_session_id: str = "") -> None:
    """Launch the Node/Ink TUI that talks to the Xerxes daemon."""
    entry = _find_new_tui_entry()
    if entry is None:
        searched = ", ".join(str(path) for path in _new_tui_entry_candidates())
        raise RuntimeError(
            "New Xerxes TUI bundle was not found. "
            "Build it with `cd src/ui-tui && npm run build`, or run `xerxes --legacy-tui`. "
            f"Searched: {searched}"
        )

    node = _resolve_node_executable()
    if not node:
        raise RuntimeError(
            "Node.js was not found. Run `xerxes install --node` to install the managed Xerxes Node runtime, "
            "install Node.js yourself and set XERXES_NODE, or run `xerxes --legacy-tui`."
        )

    env = os.environ.copy()
    env.setdefault(_NEW_TUI_PYTHON_ENV, sys.executable)
    env["NODE_ENV"] = os.environ.get("XERXES_TUI_NODE_ENV", "production")
    env["DEV"] = "true" if os.environ.get("XERXES_TUI_DEVTOOLS") == "1" else "false"
    project_dir = _resolve_new_tui_project_dir(Path.cwd())
    env.setdefault(_NEW_TUI_PROJECT_DIR_ENV, str(project_dir))
    env.setdefault("XERXES_CWD", str(project_dir))
    if resume_session_id:
        env[_NEW_TUI_RESUME_ENV] = resume_session_id

    proc = subprocess.run([node, str(entry)], cwd=project_dir, env=env)
    if proc.returncode:
        raise SystemExit(proc.returncode)


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
        await client.spawn()
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


def _run_update_command(argv: list[str]) -> None:
    """Run ``xerxes update``: report status and optionally apply the upgrade."""
    update_parser = argparse.ArgumentParser(
        prog="xerxes update",
        description="Check Xerxes release and git checkout status, then update the installed package.",
    )
    update_parser.add_argument(
        "--check",
        action="store_true",
        help="Only print package/git update status; do not run the package update command.",
    )
    update_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the update command that would run without executing it.",
    )
    update_parser.add_argument(
        "--force",
        action="store_true",
        help="Run the package update command even when no update is detected.",
    )
    update_parser.add_argument(
        "--git",
        action="store_true",
        help="Install from the head of the Xerxes Git repository instead of PyPI or the saved source.",
    )
    update_parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Do not fetch before comparing HEAD to the upstream git ref.",
    )
    args = update_parser.parse_args(argv)

    from .runtime.update import (
        apply_update,
        check_for_update,
        format_git_update_status,
        git_update_status,
        installed_version,
    )

    print(f"Xerxes {installed_version()}")
    package_update = check_for_update()
    if package_update is None:
        print("Package: current or PyPI unavailable")
    else:
        print(f"Package: {package_update.latest_version} available (installed {package_update.installed_version})")

    git_status = git_update_status(fetch=not args.no_fetch, timeout=2.0)
    print(f"Git: {format_git_update_status(git_status)}")

    if args.check:
        return
    if (
        not args.git
        and not args.force
        and not args.dry_run
        and package_update is None
        and git_status.updates_ahead_available == 0
    ):
        print("Already current. Use `xerxes update --force` to reinstall.")
        return

    if args.git:
        result = apply_update(dry_run=args.dry_run, git=True)
    else:
        result = apply_update(dry_run=args.dry_run)
    argv_value = result.get("argv")
    command = shlex.join(str(part) for part in argv_value) if isinstance(argv_value, list) else ""
    if args.dry_run:
        print(f"Would run: {command}")
        return

    if command:
        print(f"Ran: {command}")
    if result.get("stdout"):
        print(str(result["stdout"]).rstrip())
    if result.get("stderr"):
        print(str(result["stderr"]).rstrip(), file=sys.stderr)
    if not result.get("ok"):
        error = result.get("error")
        if error:
            print(str(error), file=sys.stderr)
        raise SystemExit(1)


def _print_export_session_list(sessions: list[dict[str, Any]]) -> None:
    """Print saved sessions in a compact human-readable table."""
    if not sessions:
        print("No saved Xerxes sessions found.")
        return
    for item in sessions:
        title = str(item.get("title") or "").replace("\n", " ").strip()
        if len(title) > 80:
            title = title[:77] + "..."
        project = str(item.get("project_dir") or "")
        print(
            f"{item.get('id')}  {item.get('messages')} message(s), {item.get('turn_count')} turn(s), "
            f"updated {item.get('updated_at')}"
        )
        if title:
            print(f"  title: {title}")
        if project:
            print(f"  project: {project}")


def _run_export_command(argv: list[str]) -> None:
    """Run ``xerxes export`` for persisted daemon sessions."""
    from .runtime.session_export import (
        DEFAULT_EXPORT_FORMAT,
        EXPORT_FORMATS,
        SessionExportError,
        build_session_export,
        format_session_export,
        list_saved_sessions,
        saved_session_summary,
        select_saved_session,
    )

    export_parser = argparse.ArgumentParser(
        prog="xerxes export",
        description="Export a saved Xerxes session trace with full persisted messages.",
    )
    export_parser.add_argument(
        "session",
        nargs="?",
        default="",
        help="Session id, id prefix, key, or title. Defaults to the latest saved session in the project.",
    )
    export_parser.add_argument(
        "--session",
        dest="session_flag",
        default="",
        help="Session id, id prefix, key, or title. Equivalent to the positional session argument.",
    )
    export_parser.add_argument(
        "--project",
        default="",
        help="Project directory to filter sessions by. Defaults to the current project directory.",
    )
    export_parser.add_argument(
        "--all-projects",
        action="store_true",
        help="Do not filter saved sessions by project directory.",
    )
    export_parser.add_argument(
        "--store-dir",
        default="",
        help="Override the daemon session store directory (defaults to ~/.xerxes/sessions).",
    )
    export_parser.add_argument(
        "--format",
        choices=EXPORT_FORMATS,
        default=DEFAULT_EXPORT_FORMAT,
        help="Output format.",
    )
    export_parser.add_argument("-o", "--output", default="", help="Write the export to this file instead of stdout.")
    export_parser.add_argument("--list", action="store_true", help="List matching saved sessions instead of exporting.")
    export_parser.add_argument(
        "--no-archive",
        action="store_true",
        help="Do not include the compacted-message archive sidecar, if one exists.",
    )
    args = export_parser.parse_args(argv)

    query = str(args.session_flag or args.session or "").strip()
    project_dir = None if args.all_projects else (args.project or _resolve_new_tui_project_dir(Path.cwd()))
    store_dir = args.store_dir or None
    try:
        if args.list:
            sessions = [
                saved_session_summary(saved)
                for saved in list_saved_sessions(store_dir=store_dir, project_dir=project_dir)
            ]
            _print_export_session_list(sessions)
            return

        saved = select_saved_session(query, store_dir=store_dir, project_dir=project_dir)
        export = build_session_export(saved, include_archive=not args.no_archive)
        rendered = format_session_export(export, args.format)
    except SessionExportError as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
        print(f"Exported session {export['session']['id']} to {output}")
        return
    sys.stdout.write(rendered)


def _claude_code_cli_path() -> str:
    """Return the configured Claude Code executable path if it is installed."""
    command = os.environ.get("CLAUDE_CODE_CLI", "claude")
    if os.path.sep in command:
        return command if Path(command).exists() else ""
    return shutil.which(command) or ""


def _node_version() -> str:
    version = os.environ.get(_NODE_VERSION_ENV, _MANAGED_NODE_VERSION).strip().removeprefix("v")
    if not version:
        raise RuntimeError(f"{_NODE_VERSION_ENV} cannot be empty")
    return version


def _node_dist_target(version: str) -> tuple[str, str, str]:
    system = platform.system().lower()
    machine = platform.machine().lower()
    os_map = {"darwin": "darwin", "linux": "linux"}
    arch_map = {
        "amd64": "x64",
        "x86_64": "x64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }
    node_os = os_map.get(system)
    node_arch = arch_map.get(machine)
    if node_os is None or node_arch is None:
        raise RuntimeError(f"managed Node.js install is not supported on {platform.system()} {platform.machine()}")
    dist_name = f"node-v{version}-{node_os}-{node_arch}"
    archive_name = f"{dist_name}.tar.gz"
    base_url = os.environ.get(_NODE_DIST_BASE_URL_ENV, "https://nodejs.org/dist").rstrip("/")
    return dist_name, archive_name, f"{base_url}/v{version}/{archive_name}"


def _safe_extract_tar(archive: Path, dest: Path, *, expected_root: str) -> None:
    dest_resolved = dest.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name
            if name != expected_root and not name.startswith(f"{expected_root}/"):
                raise RuntimeError(f"unexpected Node.js archive member: {name}")
            target = (dest / name).resolve()
            if os.path.commonpath([str(dest_resolved), str(target)]) != str(dest_resolved):
                raise RuntimeError(f"unsafe Node.js archive member: {name}")
        tar.extractall(dest)


def _download_file(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _replace_current_node_link(target: Path) -> None:
    current = xerxes_subdir("node", "current")
    if current.is_symlink() or current.is_file():
        current.unlink()
    elif current.exists():
        shutil.rmtree(current)
    current.symlink_to(target, target_is_directory=True)


def _install_managed_node(*, dry_run: bool = False, force: bool = False) -> Path:
    version = _node_version()
    dist_name, archive_name, url = _node_dist_target(version)
    node_root = xerxes_subdir("node")
    target = node_root / dist_name
    node_path = target / "bin" / ("node.exe" if os.name == "nt" else "node")
    if node_path.is_file() and os.access(node_path, os.X_OK) and not force:
        _replace_current_node_link(target)
        print(f"Managed Node.js already installed: {node_path}")
        return node_path

    if dry_run:
        print(f"Would download: {url}")
        print(f"Would install to: {target}")
        print(f"Would set current runtime: {xerxes_subdir('node', 'current')}")
        return node_path

    node_root.mkdir(parents=True, exist_ok=True)
    downloads = node_root / "downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / archive_name
    if force and target.exists():
        shutil.rmtree(target)

    if not target.exists():
        print(f"Downloading Node.js v{version} from {url}")
        _download_file(url, archive)
        print(f"Installing Node.js to {node_root}")
        _safe_extract_tar(archive, node_root, expected_root=dist_name)

    if not node_path.is_file():
        raise RuntimeError(f"Node.js archive did not contain expected executable: {node_path}")
    node_path.chmod(node_path.stat().st_mode | 0o111)
    _replace_current_node_link(target)
    print(f"Managed Node.js installed: {node_path}")
    return node_path


def _run_install_command(argv: list[str]) -> None:
    """Install optional companion CLIs used by Xerxes."""
    install_parser = argparse.ArgumentParser(
        prog="xerxes install",
        description="Install optional Xerxes companion tools.",
    )
    install_parser.add_argument(
        "--node",
        action="store_true",
        help="Install the managed Node.js runtime used by the packaged TypeScript TUI.",
    )
    install_parser.add_argument(
        "--cloud-code",
        "--claude-code",
        action="store_true",
        dest="claude_code",
        help="Install Anthropic Claude Code for the built-in claude-code provider profile.",
    )
    install_parser.add_argument(
        "--force",
        action="store_true",
        help="Reinstall even if the Claude Code executable is already available.",
    )
    install_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the install command without executing it.",
    )
    args = install_parser.parse_args(argv)
    if not args.node and not args.claude_code:
        install_parser.error("choose an install target, for example `xerxes install --node`")

    if args.node:
        _install_managed_node(dry_run=args.dry_run, force=args.force)

    if args.claude_code:
        installed = _claude_code_cli_path()
        if installed and not args.force:
            print(f"Claude Code is already installed: {installed}")
            print("Login or refresh credentials with: claude auth login")
            return

        npm = _resolve_npm_executable()
        if not npm:
            raise SystemExit("npm was not found. Run `xerxes install --node` first, then `xerxes install --cloud-code`.")

        command = [npm, "install", "-g", "@anthropic-ai/claude-code"]
        if args.dry_run:
            print(f"Would run: {shlex.join(command)}")
            return

        proc = subprocess.run(command, text=True)
        if proc.returncode != 0:
            raise SystemExit(proc.returncode)
        print("Claude Code installed.")
        print("Login with: claude auth login")


def _service_slug(value: str) -> str:
    """Return a filesystem-safe service identifier component."""
    clean = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    clean = "-".join(part for part in clean.split("-") if part)
    return clean[:64] or "default"


def _discord_service_name(args: argparse.Namespace) -> str:
    """Derive the stable Discord service name from CLI routing options."""
    explicit = str(getattr(args, "service_name", "") or "").strip()
    if explicit:
        return _service_slug(explicit)
    candidates = (
        list(getattr(args, "allowed_channel_names", []) or [])
        + list(getattr(args, "address_names", []) or [])
        + list(getattr(args, "allowed_channel", []) or [])
    )
    if getattr(args, "instance_name", ""):
        candidates.insert(0, str(args.instance_name))
    for candidate in candidates:
        if str(candidate).strip():
            return f"discord-{_service_slug(str(candidate))}"
    return "discord-default"


def _discord_service_paths(service_name: str) -> dict[str, Path]:
    """Return pid/socket/log paths for one Discord service."""
    base = xerxes_subdir("services", service_name)
    return {
        "base": base,
        "pid_file": base / "service.pid",
        "socket_path": base / "daemon.sock",
        "log_dir": base / "logs",
        "log_file": base / "service.log",
    }


def _read_pid(pid_file: Path) -> int | None:
    """Read a PID file, returning ``None`` when missing or malformed."""
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _pid_running(pid: int | None) -> bool:
    """Return true if ``pid`` appears to be alive."""
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _discord_child_argv(args: argparse.Namespace, service_name: str) -> list[str]:
    """Build the foreground child argv without embedding the Discord token."""
    child = [sys.executable, "-m", "xerxes", "discord", "--foreground", "--service-name", service_name]
    if args.project_dir:
        child.extend(["--project-dir", str(args.project_dir)])
    if args.host:
        child.extend(["--host", str(args.host)])
    if args.port:
        child.extend(["--port", str(args.port)])
    if args.always_reply:
        child.append("--always-reply")
    if args.no_message_content_intent:
        child.append("--no-message-content-intent")
    if args.no_discord_commands:
        child.append("--no-discord-commands")
    for value in args.allowed_channel:
        child.extend(["--allowed-channel", str(value)])
    for value in args.allowed_channel_names:
        child.extend(["--channel-name", str(value)])
    for value in args.allowed_guild:
        child.extend(["--allowed-guild", str(value)])
    if args.instance_name:
        child.extend(["--device-name", str(args.instance_name)])
    for value in args.address_names:
        child.extend(["--address-name", str(value)])
    return child


def _print_discord_service_status(service_name: str) -> bool:
    """Print status for one Discord background service."""
    paths = _discord_service_paths(service_name)
    pid = _read_pid(paths["pid_file"])
    running = _pid_running(pid)
    state = "running" if running else "stopped"
    suffix = f" (pid {pid})" if running else ""
    print(f"Discord service `{service_name}`: {state}{suffix}")
    print(f"PID: {paths['pid_file']}")
    print(f"Log: {paths['log_file']}")
    return running


def _stop_discord_service(service_name: str) -> bool:
    """Stop one Discord background service by pid file."""
    paths = _discord_service_paths(service_name)
    pid = _read_pid(paths["pid_file"])
    if not _pid_running(pid):
        paths["pid_file"].unlink(missing_ok=True)
        print(f"Discord service `{service_name}` is not running.")
        return False
    assert pid is not None
    os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + _DISCORD_SERVICE_STOP_TIMEOUT
    while time.monotonic() < deadline:
        if not _pid_running(pid):
            paths["pid_file"].unlink(missing_ok=True)
            print(f"Stopped Discord service `{service_name}`.")
            return True
        time.sleep(0.1)
    print(f"Discord service `{service_name}` did not stop after SIGTERM (pid {pid}).", file=sys.stderr)
    return False


def _start_discord_service(args: argparse.Namespace, service_name: str) -> None:
    """Start one Discord gateway daemon in the background."""
    paths = _discord_service_paths(service_name)
    pid = _read_pid(paths["pid_file"])
    if _pid_running(pid):
        print(f"Discord service `{service_name}` is already running (pid {pid}).")
        print(f"Log: {paths['log_file']}")
        return

    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.token:
        env["DISCORD_BOT_TOKEN"] = str(args.token)
    env["XERXES_DAEMON_ENABLE_DISCORD"] = "1"
    argv = _discord_child_argv(args, service_name)
    with paths["log_file"].open("ab") as log:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=os.getcwd(),
            env=env,
            start_new_session=True,
        )

    deadline = time.monotonic() + _DISCORD_SERVICE_START_TIMEOUT
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"Discord service `{service_name}` exited during startup. Log: {paths['log_file']}")
        pid = _read_pid(paths["pid_file"])
        if _pid_running(pid):
            break
        time.sleep(0.1)

    print(f"Started Discord service `{service_name}` in background.")
    print(f"PID: {paths['pid_file']}")
    print(f"Log: {paths['log_file']}")


def _configure_discord_daemon_paths(config: Any, service_name: str) -> None:
    """Point a Discord daemon at its service-owned control files."""
    paths = _discord_service_paths(service_name)
    paths["base"].mkdir(parents=True, exist_ok=True)
    paths["log_dir"].mkdir(parents=True, exist_ok=True)
    config.socket_path = str(paths["socket_path"])
    config.pid_file = str(paths["pid_file"])
    config.log_dir = str(paths["log_dir"])


def main(argv: list[str] | None = None) -> None:
    """Parse ``argv`` and dispatch to telegram, one-shot, or TUI mode.

    Imports the legacy TUI lazily so that ``xerxes telegram`` and one-shot
    paths avoid the heavy ``prompt_toolkit`` startup. Honours
    ``KeyboardInterrupt`` quietly.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "install":
        _run_install_command(argv[1:])
        return
    if argv and argv[0] == "update":
        _run_update_command(argv[1:])
        return
    if argv and argv[0] == "export":
        _run_export_command(argv[1:])
        return

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

    if argv and argv[0] == "discord":
        discord_parser = argparse.ArgumentParser(
            prog="xerxes discord", description="Start the daemon with Discord Gateway enabled."
        )
        discord_parser.add_argument(
            "--token", default=os.environ.get("DISCORD_BOT_TOKEN", os.environ.get("DISCORD_TOKEN", ""))
        )
        discord_parser.add_argument("--project-dir", default="")
        discord_parser.add_argument("--host", default="")
        discord_parser.add_argument("--port", type=int, default=0)
        discord_parser.add_argument(
            "--foreground",
            action="store_true",
            help="Run in the current terminal instead of starting the background service.",
        )
        discord_parser.add_argument(
            "--detach",
            action="store_true",
            help="Start in the background (default for Discord).",
        )
        discord_parser.add_argument("--status", action="store_true", help="Show the background Discord service status.")
        discord_parser.add_argument("--stop", action="store_true", help="Stop the background Discord service.")
        discord_parser.add_argument("--restart", action="store_true", help="Restart the background Discord service.")
        discord_parser.add_argument(
            "--service-name",
            default="",
            help="Stable service id for pid/log/socket files. Defaults to the device/channel name.",
        )
        discord_parser.add_argument(
            "--always-reply",
            action="store_true",
            help="Respond to every guild-channel message that passes allowlists instead of requiring a mention.",
        )
        discord_parser.add_argument(
            "--allowed-channel",
            action="append",
            default=[],
            help="Discord channel id to allow. Repeat for multiple channels.",
        )
        discord_parser.add_argument(
            "--channel-name",
            "--allowed-channel-name",
            action="append",
            default=[],
            dest="allowed_channel_names",
            help="Discord channel or thread name to allow. Repeat for multiple names.",
        )
        discord_parser.add_argument(
            "--allowed-guild",
            action="append",
            default=[],
            help="Discord guild id to allow. Repeat for multiple guilds.",
        )
        discord_parser.add_argument(
            "--device-name",
            "--instance-name",
            default="",
            dest="instance_name",
            help="Label replies with this Xerxes instance/device name.",
        )
        discord_parser.add_argument(
            "--address-name",
            "--wake-name",
            action="append",
            default=[],
            dest="address_names",
            help="Only respond to messages that start with this name, like 'm2-max: status'.",
        )
        discord_parser.add_argument(
            "--no-message-content-intent",
            action="store_true",
            help="Do not request Discord's privileged message content intent.",
        )
        discord_parser.add_argument(
            "--no-discord-commands",
            action="store_true",
            help="Do not register Discord slash commands (/ask, /skills, /skill, /status).",
        )
        discord_args = discord_parser.parse_args(argv[1:])
        service_name = _discord_service_name(discord_args)
        if discord_args.status:
            _print_discord_service_status(service_name)
            return
        if discord_args.stop:
            _stop_discord_service(service_name)
            return
        if discord_args.restart:
            _stop_discord_service(service_name)
            _start_discord_service(discord_args, service_name)
            return
        if not discord_args.foreground:
            _start_discord_service(discord_args, service_name)
            return

        if discord_args.token:
            os.environ["DISCORD_BOT_TOKEN"] = discord_args.token
        os.environ["XERXES_DAEMON_ENABLE_DISCORD"] = "1"

        from .daemon.config import load_config
        from .daemon.server import DaemonServer

        config = load_config(project_dir=discord_args.project_dir)
        _configure_discord_daemon_paths(config, service_name)
        if discord_args.host:
            config.ws_host = discord_args.host
        if discord_args.port:
            config.ws_port = discord_args.port
        discord_config = config.channels.setdefault("discord", {"type": "discord", "enabled": True, "settings": {}})
        discord_config["enabled"] = True
        discord_config["type"] = "discord"
        settings = discord_config.setdefault("settings", {})
        settings["transport"] = "gateway"
        settings["require_mention"] = not discord_args.always_reply
        settings["always_reply_in_channels"] = discord_args.always_reply
        settings["message_content_intent"] = not discord_args.no_message_content_intent
        settings["register_commands"] = not discord_args.no_discord_commands
        if discord_args.allowed_channel:
            settings["allowed_channel_ids"] = discord_args.allowed_channel
        if discord_args.allowed_channel_names:
            settings["allowed_channel_names"] = discord_args.allowed_channel_names
        if discord_args.allowed_guild:
            settings["allowed_guild_ids"] = discord_args.allowed_guild
        if discord_args.instance_name:
            settings["instance_name"] = discord_args.instance_name
        if discord_args.address_names:
            settings["address_names"] = discord_args.address_names
        asyncio.run(DaemonServer(config).run())
        return

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
        choices=("code", "researcher", "research", "plan", "objective", "goal"),
        default="code",
        help="Mode for one-shot prompts.",
    )
    parser.add_argument(
        "--yolo",
        action="store_true",
        help="Ignored for one-shot prompts; they always use accept-all permissions.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Kill the running daemon so the next launch starts fresh.",
    )
    parser.add_argument(
        "--legacy-tui",
        action="store_true",
        help="Use the old prompt_toolkit TUI instead of the TypeScript TUI.",
    )
    parser.add_argument(
        "prompt",
        nargs=argparse.REMAINDER,
        help="Run a one-shot prompt instead of opening the TUI.",
    )
    args = parser.parse_args(argv)

    if args.refresh:
        from .tui.engine import BridgeClient

        client = BridgeClient()
        client.restart()
        print("Daemon refreshed. Restart xerxes to connect to the new daemon.")
        return

    prompt, one_shot = _resolve_one_shot_prompt(
        args.prompt,
        stdin_is_tty=sys.stdin.isatty(),
    )

    if one_shot:
        if not prompt:
            parser.error("empty prompt")
        mode = normalize_interaction_mode(args.mode)
        try:
            asyncio.run(_run_one_shot(prompt, resume_session_id=args.resume, mode=mode))
        except KeyboardInterrupt:
            pass
        return

    if not args.legacy_tui:
        try:
            _run_new_tui(resume_session_id=args.resume)
        except KeyboardInterrupt:
            pass
        return

    from .tui import XerxesTUI

    async def _run_legacy() -> None:
        """Open the legacy interactive TUI and await its lifecycle."""
        tui = XerxesTUI(resume_session_id=args.resume)
        async with tui:
            await tui.wait_until_done()

    try:
        asyncio.run(_run_legacy())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
