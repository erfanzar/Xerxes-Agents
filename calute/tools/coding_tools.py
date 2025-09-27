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


"""Comprehensive coding tools for file management, git operations, and code manipulation."""

import difflib
import re
import shutil
import subprocess
from pathlib import Path


def read_file(
    file_path: str, start_line: int = 1, end_line: int | None = None, context_variables: dict | None = None
) -> str:
    """
    Read a file or specific lines from a file.

    Args:
        file_path: Path to the file
        start_line: Starting line number (1-based)
        end_line: Ending line number (inclusive), None for end of file

    Returns:
        File content or error message
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        if start_line < 1:
            start_line = 1

        if end_line is None:
            end_line = len(lines)
        else:
            end_line = min(end_line, len(lines))

        selected_lines = lines[start_line - 1 : end_line]

        result = []
        for i, line in enumerate(selected_lines, start=start_line):
            result.append(f"{i:6d} | {line.rstrip()}")

        return "\n".join(result) if result else "No content in specified range"

    except Exception as e:
        return f"Error reading file: {e!s}"


def write_file(file_path: str, content: str, create_dirs: bool = True, context_variables: dict | None = None) -> str:
    """
    Write content to a file, creating directories if needed.

    Args:
        file_path: Path to the file
        content: Content to write
        create_dirs: Whether to create parent directories

    Returns:
        Success message or error
    """
    try:
        path = Path(file_path)

        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

        lines = content.count("\n") + 1
        return f"Successfully wrote {len(content)} characters ({lines} lines) to {file_path}"

    except Exception as e:
        return f"Error writing file: {e!s}"


def list_directory(
    directory: str = ".",
    pattern: str = "*",
    recursive: bool = False,
    show_hidden: bool = False,
    max_depth: int = 3,
    context_variables: dict | None = None,
) -> str:
    """
    List files and directories with filtering options.

    Args:
        directory: Directory to list
        pattern: Glob pattern for filtering
        recursive: Whether to list recursively
        show_hidden: Whether to show hidden files
        max_depth: Maximum depth for recursive listing

    Returns:
        Directory listing or error message
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"Error: Directory not found: {directory}"

        if not path.is_dir():
            return f"Error: Not a directory: {directory}"

        items = []

        if recursive:

            def list_recursive(p: Path, depth: int = 0, prefix: str = ""):
                if depth > max_depth:
                    return

                try:
                    for item in sorted(p.glob(pattern)):
                        if not show_hidden and item.name.startswith("."):
                            continue

                        rel_path = item.relative_to(path)
                        indent = "  " * depth

                        if item.is_dir():
                            items.append(f"{indent}📁 {rel_path}/")
                            if depth < max_depth:
                                list_recursive(item, depth + 1, prefix + "  ")
                        else:
                            size = item.stat().st_size
                            size_str = format_size(size)
                            items.append(f"{indent}📄 {rel_path} ({size_str})")
                except PermissionError:
                    items.append(f"{indent}❌ Permission denied")

            list_recursive(path)
        else:
            for item in sorted(path.glob(pattern)):
                if not show_hidden and item.name.startswith("."):
                    continue

                if item.is_dir():
                    items.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    size_str = format_size(size)
                    items.append(f"📄 {item.name} ({size_str})")

        if not items:
            return f"No items matching pattern '{pattern}' in {directory}"

        header = f"Directory listing for: {path.absolute()}\n"
        header += f"Pattern: {pattern} | Recursive: {recursive} | Hidden: {show_hidden}\n"
        header += "-" * 60 + "\n"

        return header + "\n".join(items)

    except Exception as e:
        return f"Error listing directory: {e!s}"


def format_size(size: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def copy_file(source: str, destination: str, overwrite: bool = False, context_variables: dict | None = None) -> str:
    """
    Copy a file or directory.

    Args:
        source: Source path
        destination: Destination path
        overwrite: Whether to overwrite existing files

    Returns:
        Success message or error
    """
    try:
        src_path = Path(source)
        dst_path = Path(destination)

        if not src_path.exists():
            return f"Error: Source not found: {source}"

        if dst_path.exists() and not overwrite:
            return f"Error: Destination exists: {destination}. Use overwrite=True to replace."

        if src_path.is_dir():
            shutil.copytree(src_path, dst_path, dirs_exist_ok=overwrite)
            return f"Successfully copied directory {source} to {destination}"
        else:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            return f"Successfully copied file {source} to {destination}"

    except Exception as e:
        return f"Error copying: {e!s}"


def move_file(source: str, destination: str, overwrite: bool = False, context_variables: dict | None = None) -> str:
    """
    Move a file or directory.

    Args:
        source: Source path
        destination: Destination path
        overwrite: Whether to overwrite existing files

    Returns:
        Success message or error
    """
    try:
        src_path = Path(source)
        dst_path = Path(destination)

        if not src_path.exists():
            return f"Error: Source not found: {source}"

        if dst_path.exists() and not overwrite:
            return f"Error: Destination exists: {destination}. Use overwrite=True to replace."

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_path), str(dst_path))
        return f"Successfully moved {source} to {destination}"

    except Exception as e:
        return f"Error moving: {e!s}"


def delete_file(path: str, force: bool = False, context_variables: dict | None = None) -> str:
    """
    Delete a file or directory.

    Args:
        path: Path to delete
        force: Force deletion without confirmation

    Returns:
        Success message or error
    """
    try:
        file_path = Path(path)

        if not file_path.exists():
            return f"Error: Path not found: {path}"

        if not force and file_path.is_dir():
            item_count = sum(1 for _ in file_path.rglob("*"))
            if item_count > 10:
                return f"Error: Directory contains {item_count} items. Use force=True to delete."

        if file_path.is_dir():
            shutil.rmtree(file_path)
            return f"Successfully deleted directory: {path}"
        else:
            file_path.unlink()
            return f"Successfully deleted file: {path}"

    except Exception as e:
        return f"Error deleting: {e!s}"


def git_status(repo_path: str = ".", context_variables: dict | None = None) -> str:
    """
    Get git repository status.

    Args:
        repo_path: Path to git repository

    Returns:
        Git status output or error
    """
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", "-b"], cwd=repo_path, capture_output=True, text=True, timeout=10
        )

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        lines = result.stdout.strip().split("\n")
        if not lines or not lines[0]:
            return "Working directory clean"

        output = []
        for line in lines:
            if line.startswith("##"):
                branch_info = line[3:]
                output.append(f"Branch: {branch_info}")
            elif line:
                status = line[:2]
                file_path = line[3:]

                status_map = {
                    "M ": "Modified (staged)",
                    " M": "Modified (unstaged)",
                    "MM": "Modified (staged and unstaged)",
                    "A ": "Added",
                    "D ": "Deleted",
                    "R ": "Renamed",
                    "C ": "Copied",
                    "??": "Untracked",
                    "!!": "Ignored",
                }

                status_desc = status_map.get(status, status)
                output.append(f"  {status_desc}: {file_path}")

        return "\n".join(output)

    except subprocess.TimeoutExpired:
        return "Error: Git command timed out"
    except Exception as e:
        return f"Error getting git status: {e!s}"


def git_diff(
    repo_path: str = ".",
    file_path: str | None = None,
    staged: bool = False,
    context_lines: int = 3,
    context_variables: dict | None = None,
) -> str:
    """
    Get git diff for changes.

    Args:
        repo_path: Path to git repository
        file_path: Specific file to diff (None for all)
        staged: Whether to show staged changes
        context_lines: Number of context lines

    Returns:
        Git diff output or error
    """
    try:
        cmd = ["git", "diff", f"-U{context_lines}"]

        if staged:
            cmd.append("--staged")

        if file_path:
            cmd.append(file_path)

        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        if not result.stdout:
            return "No changes detected"

        return result.stdout

    except subprocess.TimeoutExpired:
        return "Error: Git diff timed out"
    except Exception as e:
        return f"Error getting git diff: {e!s}"


def git_apply_patch(
    patch_content: str, repo_path: str = ".", check_only: bool = False, context_variables: dict | None = None
) -> str:
    """
    Apply a git patch.

    Args:
        patch_content: The patch content to apply
        repo_path: Path to git repository
        check_only: Only check if patch applies cleanly without applying

    Returns:
        Success message or error
    """
    try:
        cmd = ["git", "apply"]

        if check_only:
            cmd.append("--check")

        result = subprocess.run(cmd, cwd=repo_path, input=patch_content, capture_output=True, text=True, timeout=30)

        if result.returncode != 0:
            return f"Error applying patch: {result.stderr}"

        if check_only:
            return "Patch can be applied cleanly"
        else:
            return "Patch applied successfully"

    except subprocess.TimeoutExpired:
        return "Error: Git apply timed out"
    except Exception as e:
        return f"Error applying patch: {e!s}"


def git_log(
    repo_path: str = ".",
    max_commits: int = 10,
    oneline: bool = True,
    file_path: str | None = None,
    context_variables: dict | None = None,
) -> str:
    """
    Get git commit history.

    Args:
        repo_path: Path to git repository
        max_commits: Maximum number of commits to show
        oneline: Whether to use oneline format
        file_path: Specific file to show history for

    Returns:
        Git log output or error
    """
    try:
        cmd = ["git", "log", f"-{max_commits}"]

        if oneline:
            cmd.append("--oneline")
        else:
            cmd.append("--pretty=format:%h - %an, %ar : %s")

        if file_path:
            cmd.extend(["--", file_path])

        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        if not result.stdout:
            return "No commit history found"

        return result.stdout

    except subprocess.TimeoutExpired:
        return "Error: Git log timed out"
    except Exception as e:
        return f"Error getting git log: {e!s}"


def git_add(files: list[str], repo_path: str = ".", context_variables: dict | None = None) -> str:
    """
    Stage files for commit.

    Args:
        files: List of file paths to stage
        repo_path: Path to git repository

    Returns:
        Success message or error
    """
    try:
        if not files:
            return "Error: No files specified"

        cmd = ["git", "add", *files]

        result = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, timeout=10)

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return f"Successfully staged {len(files)} file(s)"

    except subprocess.TimeoutExpired:
        return "Error: Git add timed out"
    except Exception as e:
        return f"Error staging files: {e!s}"


def create_diff(original: str, modified: str, file_name: str = "file.txt", context_variables: dict | None = None) -> str:
    """
    Create a unified diff between two text contents.

    Args:
        original: Original content
        modified: Modified content
        file_name: Name to use in diff header

    Returns:
        Unified diff string
    """
    try:
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines, modified_lines, fromfile=f"a/{file_name}", tofile=f"b/{file_name}", n=3
        )

        return "".join(diff)

    except Exception as e:
        return f"Error creating diff: {e!s}"


def apply_diff(original: str, diff: str, context_variables: dict | None = None) -> str:
    """
    Apply a unified diff to original content.

    Args:
        original: Original content
        diff: Unified diff to apply

    Returns:
        Modified content or error message
    """
    try:
        lines = original.splitlines()
        diff_lines = diff.splitlines()

        result = []
        i = 0

        for diff_line in diff_lines:
            if diff_line.startswith("+++") or diff_line.startswith("---"):
                continue
            elif diff_line.startswith("@@"):
                match = re.match(r"@@ -(\d+),(\d+) \+(\d+),(\d+) @@", diff_line)
                if match:
                    old_start = int(match.group(1)) - 1
                    i = old_start
            elif diff_line.startswith("+"):
                result.append(diff_line[1:])
            elif diff_line.startswith("-"):
                i += 1
            elif diff_line.startswith(" "):
                if i < len(lines):
                    result.append(lines[i])
                    i += 1

        return "\n".join(result)

    except Exception as e:
        return f"Error applying diff: {e!s}"


def find_and_replace(
    file_path: str,
    search: str,
    replace: str,
    regex: bool = False,
    case_sensitive: bool = True,
    backup: bool = True,
    context_variables: dict | None = None,
) -> str:
    """
    Find and replace text in a file.

    Args:
        file_path: Path to the file
        search: Text or pattern to search for
        replace: Replacement text
        regex: Whether to use regex for search
        case_sensitive: Whether search is case-sensitive
        backup: Whether to create a backup

    Returns:
        Success message with replacement count or error
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if backup:
            backup_path = path.with_suffix(path.suffix + ".bak")
            shutil.copy2(path, backup_path)

        if regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            pattern = re.compile(search, flags)
            new_content, count = pattern.subn(replace, content)
        else:
            if case_sensitive:
                new_content = content.replace(search, replace)
                count = content.count(search)
            else:
                pattern = re.compile(re.escape(search), re.IGNORECASE)
                new_content, count = pattern.subn(replace, content)

        if count > 0:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_content)

        backup_msg = f" (backup saved as {backup_path.name})" if backup else ""
        return f"Replaced {count} occurrence(s) in {file_path}{backup_msg}"

    except Exception as e:
        return f"Error in find and replace: {e!s}"


def analyze_code_structure(file_path: str, context_variables: dict | None = None) -> str:
    """
    Analyze the structure of a code file.

    Args:
        file_path: Path to the code file

    Returns:
        Structural analysis or error
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()

        ext = path.suffix.lower()
        language = detect_language(ext)

        analysis = {
            "file": file_path,
            "language": language,
            "lines": len(lines),
            "characters": len(content),
            "functions": [],
            "classes": [],
            "imports": [],
            "comments": 0,
            "blank_lines": 0,
        }

        if language == "Python":
            analyze_python(lines, analysis)
        elif language == "JavaScript":
            analyze_javascript(lines, analysis)
        elif language == "Java":
            analyze_java(lines, analysis)

        output = [
            f"Code Structure Analysis: {path.name}",
            f"Language: {language}",
            f"Total Lines: {analysis['lines']}",
            f"Blank Lines: {analysis['blank_lines']}",
            f"Comment Lines: {analysis['comments']}",
            f"Code Lines: {analysis['lines'] - analysis['blank_lines'] - analysis['comments']}",
        ]

        if analysis["imports"]:
            output.append(f"\nImports ({len(analysis['imports'])}):")
            for imp in analysis["imports"][:10]:
                output.append(f"  • {imp}")

        if analysis["classes"]:
            output.append(f"\nClasses ({len(analysis['classes'])}):")
            for cls in analysis["classes"]:
                output.append(f"  • {cls}")

        if analysis["functions"]:
            output.append(f"\nFunctions ({len(analysis['functions'])}):")
            for func in analysis["functions"][:20]:
                output.append(f"  • {func}")

        return "\n".join(output)

    except Exception as e:
        return f"Error analyzing code structure: {e!s}"


def detect_language(extension: str) -> str:
    """Detect programming language from file extension."""
    lang_map = {
        ".py": "Python",
        ".js": "JavaScript",
        ".ts": "TypeScript",
        ".java": "Java",
        ".cpp": "C++",
        ".c": "C",
        ".cs": "C#",
        ".go": "Go",
        ".rs": "Rust",
        ".rb": "Ruby",
        ".php": "PHP",
        ".swift": "Swift",
        ".kt": "Kotlin",
        ".scala": "Scala",
        ".r": "R",
        ".m": "MATLAB",
        ".jl": "Julia",
        ".sh": "Shell",
        ".bash": "Bash",
        ".sql": "SQL",
        ".html": "HTML",
        ".css": "CSS",
        ".scss": "SCSS",
        ".yaml": "YAML",
        ".yml": "YAML",
        ".json": "JSON",
        ".xml": "XML",
        ".md": "Markdown",
    }
    return lang_map.get(extension, "Unknown")


def analyze_python(lines: list[str], analysis: dict):
    """Analyze Python code structure."""
    for _i, line in enumerate(lines):
        stripped = line.strip()

        if not stripped:
            analysis["blank_lines"] += 1
        elif stripped.startswith("#"):
            analysis["comments"] += 1
        elif stripped.startswith('"""') or stripped.startswith("'''"):
            analysis["comments"] += 1
        elif stripped.startswith("import ") or stripped.startswith("from "):
            analysis["imports"].append(stripped)
        elif stripped.startswith("def "):
            match = re.match(r"def\s+(\w+)", stripped)
            if match:
                analysis["functions"].append(match.group(1))
        elif stripped.startswith("class "):
            match = re.match(r"class\s+(\w+)", stripped)
            if match:
                analysis["classes"].append(match.group(1))


def analyze_javascript(lines: list[str], analysis: dict):
    """Analyze JavaScript code structure."""
    for line in lines:
        stripped = line.strip()

        if not stripped:
            analysis["blank_lines"] += 1
        elif stripped.startswith("//"):
            analysis["comments"] += 1
        elif stripped.startswith("/*"):
            analysis["comments"] += 1
        elif "import " in stripped or "require(" in stripped:
            analysis["imports"].append(stripped)
        elif "function " in stripped:
            match = re.search(r"function\s+(\w+)", stripped)
            if match:
                analysis["functions"].append(match.group(1))
        elif "class " in stripped:
            match = re.match(r"class\s+(\w+)", stripped)
            if match:
                analysis["classes"].append(match.group(1))
        elif " = function" in stripped or "= () =>" in stripped:
            match = re.search(r"(\w+)\s*=\s*function|\(\w+\)\s*=>", stripped)
            if match:
                analysis["functions"].append(match.group(1) if match.group(1) else "anonymous")


def analyze_java(lines: list[str], analysis: dict):
    """Analyze Java code structure."""
    for line in lines:
        stripped = line.strip()

        if not stripped:
            analysis["blank_lines"] += 1
        elif stripped.startswith("//"):
            analysis["comments"] += 1
        elif stripped.startswith("/*"):
            analysis["comments"] += 1
        elif stripped.startswith("import "):
            analysis["imports"].append(stripped)
        elif "class " in stripped:
            match = re.search(r"class\s+(\w+)", stripped)
            if match:
                analysis["classes"].append(match.group(1))
        elif re.search(r"(public|private|protected).*\s+\w+\s*\(", stripped):
            match = re.search(r"(\w+)\s*\(", stripped)
            if match and match.group(1) not in ["if", "while", "for", "switch", "catch"]:
                analysis["functions"].append(match.group(1))


__all__ = [
    "analyze_code_structure",
    "apply_diff",
    "copy_file",
    "create_diff",
    "delete_file",
    "find_and_replace",
    "git_add",
    "git_apply_patch",
    "git_diff",
    "git_log",
    "git_status",
    "list_directory",
    "move_file",
    "read_file",
    "write_file",
]
