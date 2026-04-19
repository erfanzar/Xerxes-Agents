# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0


# distributed under the License is distributed on an "AS IS" BASIS,

# See the License for the specific language governing permissions and
# limitations under the License.


"""System and environment tools for interacting with the operating system.

This module provides a collection of system tools for agents to interact
with the operating system and environment. It includes:
- System information retrieval (OS, CPU, memory, disk, network)
- Process management and monitoring
- File system operations (copy, move, delete, search)
- Environment variable management
- Temporary file and directory handling

All tools provide comprehensive error handling and return structured
dictionaries with operation results and error information.

Example:
    >>> info = SystemInfo.static_call(info_type="cpu")
    >>> print(f"CPU cores: {info['cpu']['logical_cores']}")
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
    """Get system and environment information.

    Provides comprehensive system information retrieval capabilities
    including operating system details, CPU metrics, memory usage,
    disk space, and network interface information.

    Uses the psutil library for cross-platform system monitoring
    and the platform module for OS-level information.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Example:
        >>> result = SystemInfo.static_call(info_type="memory")
        >>> print(f"Available: {result['memory']['available_gb']} GB")
    """

    @staticmethod
    def static_call(
        info_type: str = "all",
        **context_variables,
    ) -> dict[str, Any]:
        """Get system information.

        Retrieves various types of system information based on the
        requested info_type. Can retrieve all categories or specific
        subsets for efficiency.

        Args:
            info_type: Type of information to retrieve. Options:
                - "all": All available system information
                - "os": Operating system and Python version
                - "cpu": CPU cores, usage, and frequency
                - "memory": RAM usage and availability
                - "disk": Disk usage and partitions
                - "network": Network interfaces and addresses
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary containing requested system information:
                - os: System, release, version, machine, processor, python_version
                - cpu: Physical/logical cores, usage_percent, frequency
                - memory: Total, available, used, percent (in bytes and GB)
                - disk: Total, used, free, percent, partitions list
                - network: Hostname, interfaces with addresses
        """
        result = {}

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
    """Manage and monitor system processes.

    Provides process management capabilities including listing,
    finding, inspecting, running commands, and terminating processes.
    Uses psutil for cross-platform process management.

    Supports graceful termination with fallback to force kill,
    and provides detailed process information including resource usage.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Example:
        >>> result = ProcessManager.static_call(
        ...     operation="find",
        ...     process_name="python"
        ... )
        >>> print(f"Found {result['count']} Python processes")
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

        Performs various process management operations including listing,
        finding, getting detailed info, killing, and running processes.

        Args:
            operation: Operation to perform:
                - "list": List top processes by CPU usage
                - "find": Find processes by name
                - "info": Get detailed process information by PID
                - "kill": Terminate a process by PID
                - "run": Execute a shell command
            process_name: Name of process to find (for find operation).
            pid: Process ID (for info and kill operations).
            command: Shell command to run (for run operation).
            limit: Maximum number of processes to return (default: 20).
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary containing operation-specific results:
                - list: processes list with pid, name, cpu_percent, memory_percent
                - find: found list with matching processes
                - info: detailed process info including memory, threads, cmdline
                - kill: status, pid, name, message
                - run: completed, returncode, stdout, stderr
                - error: Error message if operation failed
        """
        result = {}

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
    """Advanced file system operations.

    Provides comprehensive file system operations including copying,
    moving, deleting files and directories, searching with patterns,
    getting detailed file information, and generating directory trees.

    Supports recursive operations and pattern-based file searching
    using glob patterns.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Example:
        >>> result = FileSystemTools.static_call(
        ...     operation="search",
        ...     path="./src",
        ...     pattern="*.py",
        ...     recursive=True
        ... )
        >>> print(f"Found {result['count']} Python files")
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

        Executes various file system operations with support for
        recursive operations and pattern matching.

        Args:
            operation: Operation to perform:
                - "copy": Copy file or directory
                - "move": Move file or directory
                - "delete": Delete file or directory
                - "search": Search for files matching pattern
                - "info": Get detailed file/directory information
                - "tree": Generate directory tree structure
            path: Source path (required for most operations).
            destination: Destination path (for copy/move operations).
            pattern: Glob pattern for search (e.g., "*.py", "**/*.txt").
            recursive: Whether to operate recursively (default: False).
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary containing operation-specific results:
                - copy/move: success, source, destination
                - delete: success, deleted path
                - search: matches list, count
                - info: path, exists, is_file, is_dir, size, timestamps
                - tree: nested directory tree structure
                - error: Error message if operation failed
        """
        result = {}

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

                def build_tree(dir_path: Path, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
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
    """Manage environment variables and settings.

    Provides operations for getting, setting, listing, and removing
    environment variables. Includes filtering capabilities and
    automatic listing of commonly important environment variables.

    Changes to environment variables affect only the current process
    and its child processes.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Example:
        >>> result = EnvironmentManager.static_call(
        ...     operation="get",
        ...     key="PATH"
        ... )
        >>> print(result["value"])
    """

    @staticmethod
    def static_call(
        operation: str,
        key: str | None = None,
        value: str | None = None,
        **context_variables,
    ) -> dict[str, Any]:
        """Manage environment variables.

        Performs operations on environment variables within the
        current process scope.

        Args:
            operation: Operation to perform:
                - "get": Get value of specific variable
                - "set": Set a variable to a value
                - "list": List environment variables (filtered or common)
                - "remove": Remove an environment variable
            key: Environment variable name (required for get/set/remove).
                For list, used as prefix filter.
            value: Value to set (required for set operation).
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary containing operation-specific results:
                - get: key, value, exists
                - set: success, key, value
                - list: environment dict, count
                - remove: success, removed key
                - error: Error message if operation failed

        Note:
            Environment changes are process-scoped and do not persist
            after the program exits.
        """
        result = {}

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

    Provides functionality for creating temporary files and directories
    with optional content, custom prefixes/suffixes, and cleanup
    operations for Xerxes-created temporary files.

    Uses the system's default temporary directory and provides
    options for automatic or manual cleanup.

    Attributes:
        Inherits from AgentBaseFn for agent integration.

    Example:
        >>> result = TempFileManager.static_call(
        ...     operation="create_file",
        ...     content="temporary data",
        ...     suffix=".txt"
        ... )
        >>> print(result["path"])
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

        Creates temporary files or directories in the system temp
        directory, or cleans up previously created Xerxes temp files.

        Args:
            operation: Operation to perform:
                - "create_file": Create a temporary file
                - "create_dir": Create a temporary directory
                - "cleanup": Remove all Xerxes temp files/directories
            content: Content to write to temporary file (optional).
            suffix: File extension/suffix (e.g., ".txt", ".json").
            prefix: Filename prefix (default: "xerxes_").
            cleanup: Whether to mark for automatic cleanup (default: True).
            **context_variables: Additional context passed from the agent.

        Returns:
            Dictionary containing operation-specific results:
                - create_file: path, exists, size
                - create_dir: path, exists
                - cleanup: temp_dir, found, deleted, failed, deleted_count
                - error: Error message if operation failed

        Note:
            Files created with cleanup=False will persist after program exit.
            The cleanup operation only removes files with "xerxes_" prefix.
        """
        result = {}

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
