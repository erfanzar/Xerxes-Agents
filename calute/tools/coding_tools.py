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


"""Comprehensive coding tools for file management, git operations, and code manipulation.

This module provides a suite of tools for file system operations, git version control,
and code analysis. It includes functions for reading, writing, copying, and deleting files,
as well as git operations like status, diff, log, and patch management. Additionally,
it offers code structure analysis capabilities for multiple programming languages.
"""

import difflib
import re
import shutil
import subprocess
from pathlib import Path


def read_file(
    file_path: str, start_line: int = 1, end_line: int | None = None, context_variables: dict | None = None
) -> str:
    """Read a file or specific lines from a file.

    Reads the contents of a file and returns it with line numbers.
    Supports reading a specific range of lines for large files.

    Args:
        file_path: Path to the file to read.
        start_line: Starting line number (1-based indexing).
        end_line: Ending line number (inclusive). None reads to end of file.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        The file content with line numbers, or an error message if the file
        cannot be read.

    Example:
        >>> content = read_file("example.py", start_line=1, end_line=10)
        >>> print(content)
             1 | import os
             2 | import sys
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
    """Write content to a file, creating directories if needed.

    Writes the specified content to a file at the given path. Can optionally
    create parent directories if they don't exist.

    Args:
        file_path: Path to the file to write.
        content: Content to write to the file.
        create_dirs: Whether to create parent directories if they don't exist.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A success message with the number of characters and lines written,
        or an error message if the write fails.

    Example:
        >>> result = write_file("output.txt", "Hello, World!")
        >>> print(result)
        Successfully wrote 13 characters (1 lines) to output.txt
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
    """List files and directories with filtering options.

    Lists the contents of a directory with support for glob pattern filtering,
    recursive traversal, and hidden file visibility control.

    Args:
        directory: Directory path to list. Defaults to current directory.
        pattern: Glob pattern for filtering files (e.g., '*.py', '*.txt').
        recursive: Whether to list contents recursively.
        show_hidden: Whether to include hidden files (starting with '.').
        max_depth: Maximum depth for recursive listing.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A formatted directory listing with file sizes, or an error message
        if the directory cannot be accessed.

    Example:
        >>> listing = list_directory("src", pattern="*.py", recursive=True)
        >>> print(listing)
        Directory listing for: /path/to/src
        Pattern: *.py | Recursive: True | Hidden: False
        ------------------------------------------------------------
        📄 main.py (1.2KB)
        📁 utils/
          📄 utils/helpers.py (0.5KB)
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
    """Format file size in human-readable format.

    Converts a byte count to a human-readable string with appropriate
    units (B, KB, MB, GB, TB).

    Args:
        size: File size in bytes.

    Returns:
        A formatted string with the size and appropriate unit.

    Example:
        >>> format_size(1536)
        '1.5KB'
        >>> format_size(1048576)
        '1.0MB'
    """
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{size:.1f}TB"


def copy_file(source: str, destination: str, overwrite: bool = False, context_variables: dict | None = None) -> str:
    """Copy a file or directory to a new location.

    Copies a file or directory from the source path to the destination path.
    Supports both file and directory copying with optional overwrite behavior.

    Args:
        source: Path to the source file or directory.
        destination: Path to the destination location.
        overwrite: Whether to overwrite existing files at the destination.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A success message confirming the copy, or an error message if the
        operation fails.

    Example:
        >>> result = copy_file("config.yaml", "backup/config.yaml")
        >>> print(result)
        Successfully copied file config.yaml to backup/config.yaml
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

    Moves a file or directory from the source path to the destination path.
    This effectively renames or relocates the item.

    Args:
        source: Path to the source file or directory.
        destination: Path to the destination location.
        overwrite: Whether to overwrite existing files at the destination.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A success message confirming the move, or an error message if the
        operation fails.

    Example:
        >>> result = move_file("old_name.py", "new_name.py")
        >>> print(result)
        Successfully moved old_name.py to new_name.py
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

    Removes a file or directory from the filesystem. For directories with
    many items, the force flag must be set to confirm the deletion.

    Args:
        path: Path to the file or directory to delete.
        force: Force deletion of directories with more than 10 items.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A success message confirming the deletion, or an error message if
        the operation fails or requires confirmation.

    Example:
        >>> result = delete_file("temp.txt")
        >>> print(result)
        Successfully deleted file: temp.txt
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
    """Get the current status of a git repository.

    Retrieves the git status including branch information and file
    modifications (staged, unstaged, untracked, etc.).

    Args:
        repo_path: Path to the git repository. Defaults to current directory.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A formatted status output showing branch and file states, or an
        error message if the repository is not accessible.

    Example:
        >>> status = git_status(".")
        >>> print(status)
        Branch: main
          Modified (unstaged): src/main.py
          Untracked: new_file.txt
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
    """Get the git diff showing changes in the repository.

    Retrieves a unified diff of changes in the working directory or
    staging area. Can be filtered to a specific file.

    Args:
        repo_path: Path to the git repository. Defaults to current directory.
        file_path: Specific file to show diff for. None shows all changes.
        staged: Whether to show staged changes instead of unstaged.
        context_lines: Number of context lines to include around changes.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        The unified diff output showing changes, or "No changes detected"
        if there are no modifications.

    Example:
        >>> diff = git_diff(".", file_path="src/main.py", staged=True)
        >>> print(diff)
        --- a/src/main.py
        +++ b/src/main.py
        @@ -1,3 +1,4 @@
         import os
        +import sys
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
    """Apply a git patch to the repository.

    Applies a unified diff patch to the repository. Can optionally check
    if the patch applies cleanly without actually applying it.

    Args:
        patch_content: The unified diff patch content to apply.
        repo_path: Path to the git repository. Defaults to current directory.
        check_only: If True, only verify the patch applies without applying.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A success message if the patch is applied or validated, or an error
        message if the patch cannot be applied.

    Example:
        >>> patch = '''--- a/file.txt
        ... +++ b/file.txt
        ... @@ -1 +1 @@
        ... -old line
        ... +new line'''
        >>> result = git_apply_patch(patch, check_only=True)
        >>> print(result)
        Patch can be applied cleanly
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
    """Get the git commit history.

    Retrieves the commit log for the repository. Can be filtered to show
    only commits affecting a specific file.

    Args:
        repo_path: Path to the git repository. Defaults to current directory.
        max_commits: Maximum number of commits to display.
        oneline: Whether to use compact one-line format per commit.
        file_path: Specific file to show commit history for.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        The commit log output, or an error message if the repository
        is not accessible.

    Example:
        >>> log = git_log(".", max_commits=5)
        >>> print(log)
        abc1234 Add new feature
        def5678 Fix bug in parser
        ghi9012 Update documentation
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

    Adds the specified files to the git staging area in preparation
    for a commit.

    Args:
        files: List of file paths to stage for commit.
        repo_path: Path to the git repository. Defaults to current directory.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A success message with the count of staged files, or an error
        message if the operation fails.

    Example:
        >>> result = git_add(["src/main.py", "README.md"])
        >>> print(result)
        Successfully staged 2 file(s)
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
    """Create a unified diff between two text contents.

    Generates a unified diff format showing the differences between
    the original and modified content.

    Args:
        original: The original text content.
        modified: The modified text content.
        file_name: Name to use in the diff header.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A unified diff string showing the differences.

    Example:
        >>> original = "line1\\nline2\\nline3"
        >>> modified = "line1\\nmodified\\nline3"
        >>> diff = create_diff(original, modified, "example.txt")
        >>> print(diff)
        --- a/example.txt
        +++ b/example.txt
        @@ -1,3 +1,3 @@
         line1
        -line2
        +modified
         line3
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

    Parses a unified diff and applies the changes to the original content
    to produce the modified result.

    Args:
        original: The original text content to modify.
        diff: The unified diff to apply.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        The modified content after applying the diff, or an error message
        if the diff cannot be applied.

    Example:
        >>> original = "line1\\nline2\\nline3"
        >>> diff = '''--- a/file.txt
        ... +++ b/file.txt
        ... @@ -1,3 +1,3 @@
        ...  line1
        ... -line2
        ... +modified
        ...  line3'''
        >>> result = apply_diff(original, diff)
        >>> print(result)
        line1
        modified
        line3
    """
    try:
        lines = original.splitlines(keepends=True)
        diff_lines = diff.splitlines()

        result = []
        current_line = 0  # 0-indexed position in original

        for diff_line in diff_lines:
            if diff_line.startswith("+++") or diff_line.startswith("---"):
                continue
            elif diff_line.startswith("@@"):
                # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff_line)
                if match:
                    old_start = int(match.group(1)) - 1  # Convert to 0-indexed
                    # Copy unchanged lines before this hunk
                    while current_line < old_start and current_line < len(lines):
                        result.append(lines[current_line])
                        current_line += 1
            elif diff_line.startswith("+") and not diff_line.startswith("+++"):
                # Addition - add the new line
                result.append(diff_line[1:] + "\n")
            elif diff_line.startswith("-") and not diff_line.startswith("---"):
                # Deletion - skip the original line
                current_line += 1
            elif diff_line.startswith(" "):
                # Context line - copy from original
                if current_line < len(lines):
                    result.append(lines[current_line])
                    current_line += 1

        # Copy any remaining lines after the last hunk
        while current_line < len(lines):
            result.append(lines[current_line])
            current_line += 1

        # Join and handle trailing newline
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

    Searches for occurrences of a pattern in a file and replaces them
    with the specified replacement text. Supports both literal and
    regex-based search patterns.

    Args:
        file_path: Path to the file to modify.
        search: Text or regex pattern to search for.
        replace: Replacement text.
        regex: Whether to interpret search as a regular expression.
        case_sensitive: Whether the search should be case-sensitive.
        backup: Whether to create a backup file before modifying.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A success message with the count of replacements made, or an
        error message if the operation fails.

    Example:
        >>> result = find_and_replace("config.py", "DEBUG=True", "DEBUG=False")
        >>> print(result)
        Replaced 1 occurrence(s) in config.py (backup saved as config.py.bak)
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
    """Analyze the structure of a code file.

    Performs static analysis on a source code file to extract structural
    information including imports, classes, functions, and code metrics.

    Args:
        file_path: Path to the code file to analyze.
        context_variables: Optional context dictionary for tool execution.

    Returns:
        A formatted analysis report showing the file's structure including
        language detection, line counts, imports, classes, and functions,
        or an error message if the file cannot be analyzed.

    Example:
        >>> analysis = analyze_code_structure("src/main.py")
        >>> print(analysis)
        Code Structure Analysis: main.py
        Language: Python
        Total Lines: 150
        Blank Lines: 20
        Comment Lines: 15
        Code Lines: 115

        Imports (5):
          • import os
          • from typing import Dict

        Classes (2):
          • MyClass
          • AnotherClass

        Functions (8):
          • main
          • helper_function
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
    """Detect programming language from file extension.

    Maps a file extension to its corresponding programming language name.

    Args:
        extension: The file extension including the dot (e.g., '.py', '.js').

    Returns:
        The name of the programming language, or 'Unknown' if the extension
        is not recognized.

    Example:
        >>> detect_language(".py")
        'Python'
        >>> detect_language(".js")
        'JavaScript'
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

    Parses Python source code lines to extract structural information
    including imports, class definitions, function definitions, and
    comment/blank line counts.

    Args:
        lines: List of source code lines to analyze.
        analysis: Dictionary to populate with analysis results. Modified in-place
            with keys: 'imports', 'classes', 'functions', 'comments', 'blank_lines'.
    """
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


def analyze_javascript(lines: list[str], analysis: dict) -> None:
    """Analyze JavaScript code structure.

    Parses JavaScript source code lines to extract structural information
    including imports/requires, class definitions, function definitions,
    and comment/blank line counts.

    Args:
        lines: List of source code lines to analyze.
        analysis: Dictionary to populate with analysis results. Modified in-place
            with keys: 'imports', 'classes', 'functions', 'comments', 'blank_lines'.
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

    Parses Java source code lines to extract structural information
    including imports, class definitions, method definitions, and
    comment/blank line counts.

    Args:
        lines: List of source code lines to analyze.
        analysis: Dictionary to populate with analysis results. Modified in-place
            with keys: 'imports', 'classes', 'functions', 'comments', 'blank_lines'.
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
