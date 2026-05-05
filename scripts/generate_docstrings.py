#!/usr/bin/env python3
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
"""Mechanically generate Google-style docstrings for all Python files."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Any


def humanize(name: str) -> str:
    """Convert snake_case or camelCase to a human-readable phrase."""
    words = name.replace("_", " ").split()
    result = []
    for word in words:
        parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", word).split()
        result.extend(parts)
    return " ".join(result).lower()


def describe_from_name(name: str, is_async: bool = False, is_property: bool = False) -> str:
    """Generate a basic description from a function/method name."""
    prefix = "Asynchronously " if is_async else ""
    if is_property:
        prefix = "Return "

    h = humanize(name)

    patterns = [
        ("get_", "Retrieve the "),
        ("set_", "Set the "),
        ("is_", "Check whether "),
        ("has_", "Check whether "),
        ("can_", "Determine whether "),
        ("should_", "Determine whether "),
        ("find_", "Find "),
        ("search_", "Search for "),
        ("create_", "Create "),
        ("build_", "Build "),
        ("make_", "Make "),
        ("generate_", "Generate "),
        ("compute_", "Compute "),
        ("calculate_", "Calculate "),
        ("process_", "Process "),
        ("handle_", "Handle "),
        ("parse_", "Parse "),
        ("format_", "Format "),
        ("convert_", "Convert "),
        ("validate_", "Validate "),
        ("check_", "Check "),
        ("load_", "Load "),
        ("save_", "Save "),
        ("delete_", "Delete "),
        ("remove_", "Remove "),
        ("add_", "Add "),
        ("update_", "Update "),
        ("clear_", "Clear "),
        ("reset_", "Reset "),
    ]

    for prefix_name, verb in patterns:
        if name.startswith(prefix_name):
            return f"{prefix}{verb}{humanize(name[len(prefix_name) :])}."

    if name.startswith("init") or name == "__init__":
        return f"{prefix}Initialize the instance."
    if name.startswith("__") and name.endswith("__"):
        return f"{prefix}Dunder method for {humanize(name[2:-2])}."
    if name.startswith("_"):
        return f"{prefix}Internal helper to {h}."

    return f"{prefix}{h.capitalize()}."


def type_to_str(annotation: ast.expr | None) -> str:
    """Convert an AST annotation to a string representation."""
    if annotation is None:
        return "Any"

    try:
        return ast.unparse(annotation)
    except Exception:
        if isinstance(annotation, ast.Name):
            return annotation.id
        if isinstance(annotation, ast.Constant):
            return repr(annotation.value)
        return "Any"


def get_function_sig(func: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Extract function signature information."""
    args = func.args
    params = []

    for arg in args.posonlyargs:
        params.append({"name": arg.arg, "annotation": type_to_str(arg.annotation), "kind": "positional-only"})

    for arg in args.args:
        params.append({"name": arg.arg, "annotation": type_to_str(arg.annotation), "kind": "positional"})

    if args.vararg:
        params.append(
            {"name": f"*{args.vararg.arg}", "annotation": type_to_str(args.vararg.annotation), "kind": "vararg"}
        )

    for arg in args.kwonlyargs:
        params.append({"name": arg.arg, "annotation": type_to_str(arg.annotation), "kind": "keyword-only"})

    if args.kwarg:
        params.append({"name": f"**{args.kwarg.arg}", "annotation": type_to_str(args.kwarg.annotation), "kind": "kwarg"})

    # Defaults for positional args
    num_args = len(args.args)
    num_defaults = len(args.defaults)
    for i, param in enumerate(params):
        if param["kind"] == "positional" and i >= num_args - num_defaults and num_args > 0:
            default_idx = i - (num_args - num_defaults)
            if default_idx >= 0 and default_idx < num_defaults:
                try:
                    param["default"] = ast.unparse(args.defaults[default_idx])
                except Exception:
                    param["default"] = "..."

    # Defaults for keyword-only args
    for _i, param in enumerate(params):
        if param["kind"] == "keyword-only":
            kw_idx = args.kwonlyargs.index(next(a for a in args.kwonlyargs if a.arg == param["name"]))
            if kw_idx < len(args.kw_defaults) and args.kw_defaults[kw_idx] is not None:
                try:
                    param["default"] = ast.unparse(args.kw_defaults[kw_idx])
                except Exception:
                    param["default"] = "..."

    return {
        "params": params,
        "returns": type_to_str(func.returns),
    }


def generate_function_docstring(func: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Generate a Google-style docstring for a function."""
    sig = get_function_sig(func)
    is_async = isinstance(func, ast.AsyncFunctionDef)
    is_property = any(isinstance(dec, ast.Name) and dec.id == "property" for dec in func.decorator_list)

    description = describe_from_name(func.name, is_async=is_async, is_property=is_property)

    lines = [description, ""]

    # Args
    if sig["params"]:
        lines.append("Args:")
        for param in sig["params"]:
            name = param["name"]
            ann = param["annotation"]
            default = param.get("default")

            if param["kind"] == "vararg":
                lines.append(
                    f"    {name}: IN: Additional positional arguments. OUT: Passed through to downstream calls."
                )
            elif param["kind"] == "kwarg":
                lines.append(f"    {name}: IN: Additional keyword arguments. OUT: Passed through to downstream calls.")
            elif name == "self":
                lines.append(f"    {name}: IN: The instance. OUT: Used for attribute access.")
            elif name == "cls":
                lines.append(f"    {name}: IN: The class. OUT: Used for class-level operations.")
            else:
                desc = humanize(name)
                if default:
                    lines.append(
                        f"    {name} ({ann}, optional): IN: {desc}. Defaults to {default}. OUT: Consumed during execution."
                    )
                else:
                    lines.append(f"    {name} ({ann}): IN: {desc}. OUT: Consumed during execution.")

    # Returns
    if sig["returns"] and sig["returns"] != "None":
        lines.append("Returns:")
        lines.append(f"    {sig['returns']}: OUT: Result of the operation.")

    return "\n".join(lines)


def generate_class_docstring(cls: ast.ClassDef) -> str:
    """Generate a docstring for a class."""
    name = cls.name
    description = f"{humanize(name).capitalize()}."

    bases = [type_to_str(base) for base in cls.bases]

    lines = [description, ""]

    if bases:
        lines.append(f"Inherits from: {', '.join(bases)}")
        lines.append("")

    # Find class-level attributes
    attrs = []
    for item in cls.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            attrs.append((item.target.id, type_to_str(item.annotation)))
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                if (
                    isinstance(target, ast.Name)
                    and not target.id.startswith("_")
                    and not isinstance(item.value, ast.Constant)
                ):
                    attrs.append((target.id, "Any"))

    if attrs:
        lines.append("Attributes:")
        for attr_name, attr_type in attrs:
            lines.append(f"    {attr_name} ({attr_type}): {humanize(attr_name)}.")

    return "\n".join(lines)


def generate_module_docstring(tree: ast.Module, filename: str) -> str:
    """Generate a module-level docstring."""
    public_names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith("_"):
                public_names.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    public_names.append(target.id)

    name = Path(filename).stem
    description = f"{humanize(name).capitalize()} module for Xerxes."

    lines = [description, ""]
    if public_names:
        lines.append("Exports:")
        for name in public_names[:10]:
            lines.append(f"    - {name}")
        if len(public_names) > 10:
            lines.append(f"    - ... and {len(public_names) - 10} more.")

    return "\n".join(lines)


def process_file(path: Path) -> bool:
    """Process a single file, adding missing docstrings."""
    source = path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"  SKIP (syntax error): {path} — {e}")
        return False

    lines = source.splitlines(keepends=True)

    def find_definition_end(node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> int:
        """Find the 0-based line index where the class/function definition ends (the line with ':')."""
        for i in range(node.lineno - 1, len(lines)):
            stripped = lines[i].split("#")[0].strip()
            if stripped.endswith(":"):
                return i
        return node.lineno - 1

    insertions = []

    # Module docstring
    if not ast.get_docstring(tree):
        mod_doc = generate_module_docstring(tree, path.name)
        # Insert at the very top of the file
        insertions.append((-1, 0, f'"""{mod_doc}"""\n\n'))

    # Collect classes and functions that need docstrings
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if not ast.get_docstring(node):
                doc = generate_class_docstring(node)
                indent = " " * (node.col_offset + 4)
                indented_doc = "\n".join(
                    indent + line if line.strip() else line for line in (f'"""{doc}"""').splitlines()
                )
                def_end = find_definition_end(node)
                insertions.append((def_end + 1, node.col_offset + 4, indented_doc + "\n"))

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not ast.get_docstring(node):
                doc = generate_function_docstring(node)
                indent = " " * (node.col_offset + 4)
                indented_doc = "\n".join(
                    indent + line if line.strip() else line for line in (f'"""{doc}"""').splitlines()
                )
                def_end = find_definition_end(node)
                insertions.append((def_end + 1, node.col_offset + 4, indented_doc + "\n"))

    if not insertions:
        return False

    # Sort insertions by line number in reverse order to preserve line indices
    insertions.sort(key=lambda x: (x[0], -x[1]), reverse=True)

    for line_idx, _col_offset, text in insertions:
        if line_idx == -1:
            # Insert at the very beginning
            lines.insert(0, text)
        elif line_idx < len(lines):
            lines.insert(line_idx, text)
        else:
            lines.append(text)

    new_source = "".join(lines)

    # Verify syntax
    try:
        ast.parse(new_source)
    except SyntaxError as e:
        print(f"  SKIP (would break syntax): {path} — {e}")
        return False

    path.write_text(new_source, encoding="utf-8")
    print(f"  PROCESSED: {path}")
    return True


def main():
    files = sys.argv[1:]
    if not files:
        print("Usage: python generate_docstrings.py <file1.py> [file2.py] ...")
        sys.exit(1)

    total = 0
    for f in files:
        if process_file(Path(f)):
            total += 1

    print(f"\nTotal files modified: {total}")


if __name__ == "__main__":
    main()
