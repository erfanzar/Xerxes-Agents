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
"""System information and management tools for monitoring system resources and processes.

This module provides tools for gathering system information, managing processes,
file system operations, environment variables, and temporary file management.

Example:
    >>> from xerxes.tools.system_tools import SystemInfo, ProcessManager
    >>> SystemInfo.static_call(info_type="all")
    >>> ProcessManager.static_call(operation="list")
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import psutil

from ..types import AgentBaseFn


class SystemInfo(AgentBaseFn):
    """Gather system information including OS, CPU, memory, disk, and network.

    Provides comprehensive system monitoring capabilities.

    Example:
        >>> SystemInfo.static_call(info_type="cpu")
        >>> SystemInfo.static_call(info_type="all")
    """

    @staticmethod
    def static_call(
        info_type: str = "all",
        **context_variables,
    ) -> dict[str, Any]:
        """Get system information.

        Args:
            info_type: Type of information to retrieve. Options: 'all', 'os', 'cpu',
                'memory', 'disk', 'network'. Defaults to 'all'.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary containing requested system information.
        """
        result: dict[str, Any] = {}

        if info_type in ["all", "os"]:
            result["os"] = {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "hostname": platform.node(),
            }

        if info_type in ["all", "cpu"]:
            result["cpu"] = {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_cores": psutil.cpu_count(logical=True),
                "usage_percent": psutil.cpu_percent(interval=1),
                "frequency": psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            }

        if info_type in ["all", "memory"]:
            mem = psutil.virtual_memory()
            result["memory"] = {
                "total": mem.total,
                "available": mem.available,
                "used": mem.used,
                "percent": mem.percent,
                "total_gb": round(mem.total / (1024**3), 2),
                "available_gb": round(mem.available / (1024**3), 2),
            }

        if info_type in ["all", "disk"]:
            disk = psutil.disk_usage("/")
            result["disk"] = {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": disk.percent,
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
            }

            partitions = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    partitions.append(
                        {
                            "device": partition.device,
                            "mountpoint": partition.mountpoint,
                            "fstype": partition.fstype,
                            "total_gb": round(usage.total / (1024**3), 2),
                            "used_percent": usage.percent,
                        }
                    )
                except (PermissionError, OSError):
                    pass
            result["disk"]["partitions"] = partitions

        if info_type in ["all", "network"]:
            result["network"] = {
                "hostname": platform.node(),
                "interfaces": [],
            }

            for interface, addresses in psutil.net_if_addrs().items():
                iface_info = {"name": interface, "addresses": []}
                for addr in addresses:
                    iface_info["addresses"].append(
                        {
                            "family": str(addr.family),
                            "address": addr.address,
                            "netmask": addr.netmask,
                        }
                    )
                result["network"]["interfaces"].append(iface_info)

        return result


class ProcessManager(AgentBaseFn):
    """Manage system processes including listing, finding, and controlling them.

    Provides process monitoring and control capabilities.

    Example:
        >>> ProcessManager.static_call(operation="list")
        >>> ProcessManager.static_call(operation="kill", pid=1234)
    """

    @staticmethod
    def static_call(
        operation: str,
        process_name: str | None = None,
        pid: int | None = None,
        command: str | None = None,
        limit: int = 20,
        **context_variables,
    ) -> dict[str, Any]:
        """Manage system processes.

        Args:
            operation: Operation to perform. Options: 'list', 'find', 'info', 'run', 'kill'.
            process_name: Process name for 'find' operation.
            pid: Process ID for 'info' and 'kill' operations.
            command: Command to run for 'run' operation.
            limit: Maximum processes to return for 'list'. Defaults to 20.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary containing operation results.
        """
        result: dict[str, Any] = {}

        if operation == "list":
            processes = []
            for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
                try:
                    processes.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "cpu_percent": proc.info["cpu_percent"],
                            "memory_percent": round(proc.info["memory_percent"], 2),
                        }
                    )
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass

            processes.sort(key=lambda x: x["cpu_percent"] or 0, reverse=True)
            result["processes"] = processes[:limit]
            result["total_count"] = len(processes)

        elif operation == "find":
            if not process_name:
                return {"error": "process_name required for find operation"}

            found = []
            for proc in psutil.process_iter(["pid", "name", "cmdline"]):
                try:
                    if process_name.lower() in proc.info["name"].lower():
                        found.append(
                            {
                                "pid": proc.info["pid"],
                                "name": proc.info["name"],
                                "cmdline": " ".join(proc.info["cmdline"]) if proc.info["cmdline"] else "",
                            }
                        )
                except (psutil.AccessDenied, psutil.NoSuchProcess):
                    pass

            result["found"] = found
            result["count"] = len(found)

        elif operation == "info":
            if not pid:
                return {"error": "pid required for info operation"}

            try:
                proc = psutil.Process(pid)
                result["info"] = {
                    "pid": proc.pid,
                    "name": proc.name(),
                    "status": proc.status(),
                    "created": proc.create_time(),
                    "cpu_percent": proc.cpu_percent(),
                    "memory_percent": proc.memory_percent(),
                    "memory_info": proc.memory_info()._asdict(),
                    "num_threads": proc.num_threads(),
                    "cmdline": " ".join(proc.cmdline()),
                }
            except psutil.NoSuchProcess:
                return {"error": f"No process with PID {pid}"}
            except Exception as e:
                return {"error": f"Failed to get process info: {e!s}"}

        elif operation == "run":
            if not command:
                return {"error": "command required for run operation"}

            try:
                process = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                result["completed"] = True
                result["returncode"] = process.returncode
                result["stdout"] = process.stdout[:5000]
                result["stderr"] = process.stderr[:5000]

            except subprocess.TimeoutExpired:
                return {"error": "Command timed out after 30 seconds"}
            except Exception as e:
                return {"error": f"Failed to run command: {e!s}"}

        elif operation == "kill":
            if not pid:
                return {"error": "pid required for kill operation"}

            try:
                proc = psutil.Process(pid)
                proc_name = proc.name()
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    result["status"] = "terminated"
                except psutil.TimeoutExpired:
                    proc.kill()
                    result["status"] = "killed"
                result["pid"] = pid
                result["name"] = proc_name
                result["message"] = f"Process {proc_name} (PID {pid}) has been stopped"
            except psutil.NoSuchProcess:
                return {"error": f"No process with PID {pid}"}
            except psutil.AccessDenied:
                return {"error": f"Access denied to kill process {pid}"}
            except Exception as e:
                return {"error": f"Failed to kill process: {e!s}"}

        else:
            return {"error": f"Unknown operation: {operation}"}

        return result


class FileSystemTools(AgentBaseFn):
    """Perform file system operations including copy, move, delete, search, and tree view.

    Provides comprehensive file system management.

    Example:
        >>> FileSystemTools.static_call(operation="copy", path="src", destination="dst")
        >>> FileSystemTools.static_call(operation="tree", path=".")
    """

    @staticmethod
    def static_call(
        operation: str,
        path: str | None = None,
        destination: str | None = None,
        pattern: str | None = None,
        recursive: bool = False,
        **context_variables,
    ) -> dict[str, Any]:
        """Perform file system operations.

        Args:
            operation: Operation to perform. Options: 'copy', 'move', 'delete', 'search', 'info', 'tree'.
            path: Source path for copy/move/delete operations, or path for search/info/tree.
            destination: Destination path for copy/move operations.
            pattern: Pattern for search operation.
            recursive: Enable recursive operations. Defaults to False.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary containing operation results.
        """
        result: dict[str, Any] = {}

        if operation == "copy":
            if not path or not destination:
                return {"error": "path and destination required for copy operation"}

            try:
                source = Path(path)
                dest = Path(destination)

                if source.is_dir():
                    if recursive:
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                    else:
                        return {"error": "Use recursive=True to copy directories"}
                else:
                    shutil.copy2(source, dest)

                result["success"] = True
                result["source"] = str(source)
                result["destination"] = str(dest)

            except Exception as e:
                return {"error": f"Copy failed: {e!s}"}

        elif operation == "move":
            if not path or not destination:
                return {"error": "path and destination required for move operation"}

            try:
                shutil.move(path, destination)
                result["success"] = True
                result["source"] = path
                result["destination"] = destination

            except Exception as e:
                return {"error": f"Move failed: {e!s}"}

        elif operation == "delete":
            if not path:
                return {"error": "path required for delete operation"}

            try:
                target = Path(path)

                if target.is_dir():
                    if recursive:
                        shutil.rmtree(target)
                    else:
                        target.rmdir()
                else:
                    target.unlink()

                result["success"] = True
                result["deleted"] = str(target)

            except Exception as e:
                return {"error": f"Delete failed: {e!s}"}

        elif operation == "search":
            if not path:
                path = "."

            try:
                search_path = Path(path)
                matches = []

                if pattern:
                    if recursive:
                        matches = list(search_path.rglob(pattern))
                    else:
                        matches = list(search_path.glob(pattern))
                else:
                    if recursive:
                        matches = list(search_path.rglob("*"))
                    else:
                        matches = list(search_path.iterdir())

                result["matches"] = [str(m) for m in matches[:100]]
                result["count"] = len(matches)

            except Exception as e:
                return {"error": f"Search failed: {e!s}"}

        elif operation == "info":
            if not path:
                return {"error": "path required for info operation"}

            try:
                target = Path(path)
                stat = target.stat()

                result["info"] = {
                    "path": str(target.absolute()),
                    "exists": target.exists(),
                    "is_file": target.is_file(),
                    "is_dir": target.is_dir(),
                    "is_symlink": target.is_symlink(),
                    "size": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                    "accessed": stat.st_atime,
                }

                if target.is_dir():
                    items = list(target.iterdir())
                    result["info"]["item_count"] = len(items)
                    result["info"]["subdirs"] = len([i for i in items if i.is_dir()])
                    result["info"]["files"] = len([i for i in items if i.is_file()])

            except Exception as e:
                return {"error": f"Failed to get info: {e!s}"}

        elif operation == "tree":
            if not path:
                path = "."

            try:

                def build_tree(dir_path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0) -> list[str]:
                    """Recursively build directory tree representation."""
                    if current_depth >= max_depth:
                        return []

                    items = []
                    contents = sorted(dir_path.iterdir())

                    for i, item in enumerate(contents[:20]):
                        is_last = i == len(contents) - 1
                        current_prefix = "└── " if is_last else "├── "
                        items.append(prefix + current_prefix + item.name)

                        if item.is_dir():
                            extension = "    " if is_last else "│   "
                            items.extend(build_tree(item, prefix + extension, max_depth, current_depth + 1))

                    return items

                tree_path = Path(path)
                result["tree"] = [str(tree_path), *build_tree(tree_path)]

            except Exception as e:
                return {"error": f"Failed to build tree: {e!s}"}

        else:
            return {"error": f"Unknown operation: {operation}"}

        return result


class EnvironmentManager(AgentBaseFn):
    """Manage environment variables including getting, setting, listing, and removing them.

    Provides environment variable management capabilities.

    Example:
        >>> EnvironmentManager.static_call(operation="get", key="PATH")
        >>> EnvironmentManager.static_call(operation="set", key="DEBUG", value="true")
    """

    @staticmethod
    def static_call(
        operation: str,
        key: str | None = None,
        value: str | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Manage environment variables.

        Args:
            operation: Operation to perform. Options: 'get', 'set', 'list', 'remove'.
            key: Variable name for get/set/remove operations.
            value: Variable value for set operation.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary containing operation results.
        """
        result: dict[str, Any] = {}

        if operation == "get":
            if not key:
                return {"error": "key required for get operation"}

            value = os.environ.get(key)
            result["key"] = key
            result["value"] = value
            result["exists"] = value is not None

        elif operation == "set":
            if not key or value is None:
                return {"error": "key and value required for set operation"}

            os.environ[key] = str(value)
            result["success"] = True
            result["key"] = key
            result["value"] = str(value)

        elif operation == "list":
            env_vars = {}

            if key:
                for k, v in os.environ.items():
                    if k.startswith(key):
                        env_vars[k] = v
            else:
                important_keys = [
                    "PATH",
                    "HOME",
                    "USER",
                    "SHELL",
                    "LANG",
                    "PWD",
                    "PYTHON",
                    "VIRTUAL_ENV",
                    "CONDA_DEFAULT_ENV",
                    "JAVA_HOME",
                    "NODE_ENV",
                    "GOPATH",
                ]
                for k in important_keys:
                    if k in os.environ:
                        env_vars[k] = os.environ[k]

            result["environment"] = env_vars
            result["count"] = len(env_vars)

        elif operation == "remove":
            if not key:
                return {"error": "key required for remove operation"}

            if key in os.environ:
                del os.environ[key]
                result["success"] = True
                result["removed"] = key
            else:
                result["success"] = False
                result["error"] = f"Variable {key} not found"

        else:
            return {"error": f"Unknown operation: {operation}"}

        return result


class TempFileManager(AgentBaseFn):
    """Create and manage temporary files and directories.

    Provides temporary file and directory management.

    Example:
        >>> TempFileManager.static_call(operation="create_file", content="temporary data")
        >>> TempFileManager.static_call(operation="cleanup")
    """

    @staticmethod
    def static_call(
        operation: str,
        content: str | None = None,
        suffix: str | None = None,
        prefix: str | None = None,
        cleanup: bool = True,
        **context_variables,
    ) -> dict[str, Any]:
        """Manage temporary files and directories.

        Args:
            operation: Operation to perform. Options: 'create_file', 'create_dir', 'cleanup'.
            content: File content for 'create_file' operation.
            suffix: File suffix for created files/directories.
            prefix: File prefix for created files/directories. Defaults to "xerxes_".
            cleanup: Whether to auto-cleanup. Defaults to True.
            **context_variables: Additional context passed through to downstream calls.

        Returns:
            Dictionary containing operation results.
        """
        result: dict[str, Any] = {}

        if operation == "create_file":
            try:
                fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix or "xerxes_")

                if content:
                    with os.fdopen(fd, "w") as f:
                        f.write(content)
                else:
                    os.close(fd)

                result["path"] = path
                result["exists"] = os.path.exists(path)
                result["size"] = os.path.getsize(path)

                if not cleanup:
                    result["note"] = "File will persist after program exit"

            except Exception as e:
                return {"error": f"Failed to create temp file: {e!s}"}

        elif operation == "create_dir":
            try:
                path = tempfile.mkdtemp(suffix=suffix, prefix=prefix or "xerxes_")

                result["path"] = path
                result["exists"] = os.path.exists(path)

                if not cleanup:
                    result["note"] = "Directory will persist after program exit"

            except Exception as e:
                return {"error": f"Failed to create temp directory: {e!s}"}

        elif operation == "cleanup":
            import shutil

            temp_dir = tempfile.gettempdir()
            result["temp_dir"] = temp_dir

            xerxes_temps = []
            deleted = []
            failed = []

            for item in Path(temp_dir).iterdir():
                if item.name.startswith("xerxes_"):
                    xerxes_temps.append(str(item))
                    try:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                        deleted.append(str(item))
                    except Exception as e:
                        failed.append({"path": str(item), "error": str(e)})

            result["found"] = xerxes_temps
            result["deleted"] = deleted
            result["failed"] = failed
            result["deleted_count"] = len(deleted)

        else:
            return {"error": f"Unknown operation: {operation}"}

        return result
