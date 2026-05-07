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
"""File system and code manipulation tools for reading, writing, and analyzing code.

This module provides comprehensive file system operations including reading and writing
files, directory listing, git operations, and code analysis. These tools enable agents
to work with codebases and perform common development tasks.

Example:
    >>> from xerxes.tools.coding_tools import read_file, write_file, git_status
    >>> content = read_file("main.py")
    >>> write_file("test.py", "def test(): pass")
    >>> print(git_status())
"""

import difflib
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


def read_file(
    file_path: str, start_line: int = 1, end_line: int | None = None, context_variables: dict | None = None
) -> str:
    """Read content from a file with optional line range selection.

    Args:
        file_path: Path to the file to read.
        start_line: First line to include (1-indexed). Defaults to 1.
        end_line: Last line to include (inclusive). Defaults to None (end of file).
        context_variables: Additional context for downstream processing.

    Returns:
        File content with line numbers, or error message if file not found.

    Example:
        >>> read_file("main.py", start_line=10, end_line=20)
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
    """Write content to a file, optionally creating parent directories.

    Args:
        file_path: Path to the file to write.
        content: Text content to write.
        create_dirs: Whether to create parent directories if missing. Defaults to True.
        context_variables: Additional context for downstream processing.

    Returns:
        Success message with character and line counts, or error message.

    Example:
        >>> write_file("output.txt", "Hello, World!")
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
    """List directory contents with optional filtering and recursion.

    Args:
        directory: Path to list. Defaults to current directory.
        pattern: Glob pattern to filter files. Defaults to "*" (all files).
        recursive: Whether to list subdirectories recursively. Defaults to False.
        show_hidden: Whether to include hidden files (starting with .). Defaults to False.
        max_depth: Maximum recursion depth for recursive listing. Defaults to 3.
        context_variables: Additional context for downstream processing.

    Returns:
        Formatted directory listing with file sizes, or error message.

    Example:
        >>> list_directory("src", pattern="*.py", recursive=True)
    """
    try:
        path = Path(directory)
        if not path.exists():
            return f"Error: Directory not found: {directory}"

        if not path.is_dir():
            return f"Error: Not a directory: {directory}"

        items = []

        if recursive:

            def list_recursive(p: Path, depth: int = 0, prefix: str = "") -> None:
                """Recursively list directory contents."""
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


def format_size(size: int | float) -> str:
    """Format byte size as human-readable string.

    Args:
        size: Size in bytes.

    Returns:
        Formatted string (e.g., "1.5MB").

    Example:
        >>> format_size(1536000)
        '1.5MB'
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def copy_file(source: str, destination: str, overwrite: bool = False, context_variables: dict | None = None) -> str:
    """Copy a file or directory to a new location.

    Args:
        source: Path to source file or directory.
        destination: Destination path.
        overwrite: Whether to replace existing files. Defaults to False.
        context_variables: Additional context for downstream processing.

    Returns:
        Success message or error description.

    Example:
        >>> copy_file("input.txt", "backup/input.txt")
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
    """Move a file or directory to a new location.

    Args:
        source: Path to source file or directory.
        destination: Destination path.
        overwrite: Whether to replace existing files. Defaults to False.
        context_variables: Additional context for downstream processing.

    Returns:
        Success message or error description.

    Example:
        >>> move_file("old_name.py", "new_name.py")
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
    """Delete a file or directory.

    Args:
        path: Path to delete.
        force: Skip confirmation for non-empty directories. Defaults to False.
        context_variables: Additional context for downstream processing.

    Returns:
        Success message or error description.

    Example:
        >>> delete_file("temp.txt")
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
    """Get the current git repository status.

    Args:
        repo_path: Path to the git repository. Defaults to current directory.
        context_variables: Additional context for downstream processing.

    Returns:
        Formatted status showing branch, modified files, and staging status.

    Example:
        >>> git_status()
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
    """Show changes between commits, working tree, or staged changes.

    Args:
        repo_path: Path to the git repository. Defaults to current directory.
        file_path: Specific file to diff. Defaults to all files.
        staged: Show staged changes instead of working tree. Defaults to False.
        context_lines: Number of context lines around changes. Defaults to 3.
        context_variables: Additional context for downstream processing.

    Returns:
        Unified diff output or message if no changes found.

    Example:
        >>> git_diff(staged=True)
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
    """Apply a patch file to the repository.

    Args:
        patch_content: The patch content to apply.
        repo_path: Path to the git repository. Defaults to current directory.
        check_only: Only check if patch applies cleanly without applying. Defaults to False.
        context_variables: Additional context for downstream processing.

    Returns:
        Success message or error description.

    Example:
        >>> git_apply_patch(patch_content, check_only=True)
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
    """Show commit history.

    Args:
        repo_path: Path to the git repository. Defaults to current directory.
        max_commits: Maximum number of commits to show. Defaults to 10.
        oneline: Show each commit on single line. Defaults to True.
        file_path: Show only commits affecting this file.
        context_variables: Additional context for downstream processing.

    Returns:
        Formatted commit log.

    Example:
        >>> git_log(max_commits=20)
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
    """Stage files for commit.

    Args:
        files: List of file paths to stage.
        repo_path: Path to the git repository. Defaults to current directory.
        context_variables: Additional context for downstream processing.

    Returns:
        Success message or error description.

    Example:
        >>> git_add(["file1.py", "file2.py"])
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
    """Generate a unified diff between two text contents.

    Args:
        original: Original text content.
        modified: Modified text content.
        file_name: Filename for diff headers. Defaults to "file.txt".
        context_variables: Additional context for downstream processing.

    Returns:
        Unified diff string.

    Example:
        >>> diff = create_diff(old_content, new_content, "main.py")
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
    """Apply a unified diff to original content.

    Args:
        original: Original text content.
        diff: Unified diff string to apply.
        context_variables: Additional context for downstream processing.

    Returns:
        Modified text content, or error message.

    Example:
        >>> new_content = apply_diff(original, diff_text)
    """
    try:
        lines = original.splitlines(keepends=True)
        diff_lines = diff.splitlines()

        result = []
        current_line = 0

        for diff_line in diff_lines:
            if diff_line.startswith("+++") or diff_line.startswith("---"):
                continue
            elif diff_line.startswith("@@"):
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff_line)
                if match:
                    old_start = int(match.group(1)) - 1
                    while current_line < old_start and current_line < len(lines):
                        result.append(lines[current_line])
                        current_line += 1
            elif diff_line.startswith("+") and not diff_line.startswith("+++"):
                result.append(diff_line[1:] + "\n")
            elif diff_line.startswith("-") and not diff_line.startswith("---"):
                current_line += 1
            elif diff_line.startswith(" "):
                if current_line < len(lines):
                    result.append(lines[current_line])
                    current_line += 1

        while current_line < len(lines):
            result.append(lines[current_line])
            current_line += 1

        output = "".join(result)
        if output.endswith("\n") and not original.endswith("\n"):
            output = output[:-1]
        elif not output.endswith("\n") and original.endswith("\n"):
            output += "\n"

        return output.rstrip("\n") if not original.endswith("\n") else output

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
    """Find and replace text in a file.

    Args:
        file_path: Path to the file.
        search: Text to find.
        replace: Replacement text.
        regex: Treat search as regular expression. Defaults to False.
        case_sensitive: Case-sensitive search. Defaults to True.
        backup: Create .bak backup file. Defaults to True.
        context_variables: Additional context for downstream processing.

    Returns:
        Success message with replacement count, or error description.

    Example:
        >>> find_and_replace("main.py", "foo", "bar", regex=False)
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
    """Analyze code structure including functions, classes, and imports.

    Args:
        file_path: Path to the code file to analyze.
        context_variables: Additional context for downstream processing.

    Returns:
        Formatted analysis with language, structure metrics, and code elements.

    Example:
        >>> print(analyze_code_structure("main.py"))
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

        analysis: dict[str, Any] = {
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
    """Detect programming language from file extension.

    Args:
        extension: File extension (e.g., ".py", ".js").

    Returns:
        Language name or "Unknown" if not recognized.

    Example:
        >>> detect_language(".py")
        'Python'
    """
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


def analyze_python(lines: list[str], analysis: dict) -> None:
    """Analyze Python code structure.

    Args:
        lines: Source code lines.
        analysis: Dictionary to populate with analysis results.
    """
    for line in lines:
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


def analyze_javascript(lines: list[str], analysis: dict) -> None:
    """Analyze JavaScript/TypeScript code structure.

    Args:
        lines: Source code lines.
        analysis: Dictionary to populate with analysis results.
    """
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


def analyze_java(lines: list[str], analysis: dict) -> None:
    """Analyze Java code structure.

    Args:
        lines: Source code lines.
        analysis: Dictionary to populate with analysis results.
    """
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
