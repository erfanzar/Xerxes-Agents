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
"""Simplify redlines module for Xerxes.

Exports:
    - WORD_NS
    - simplify_redlines
    - get_tracked_change_authors
    - infer_author"""

import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import defusedxml.minidom

WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def simplify_redlines(input_dir: str) -> tuple[int, str]:
    """Simplify redlines.

    Args:
        input_dir (str): IN: input dir. OUT: Consumed during execution.
    Returns:
        tuple[int, str]: OUT: Result of the operation."""
    doc_xml = Path(input_dir) / "word" / "document.xml"

    if not doc_xml.exists():
        return 0, f"Error: {doc_xml} not found"

    try:
        dom = defusedxml.minidom.parseString(doc_xml.read_text(encoding="utf-8"))
        root = dom.documentElement

        merge_count = 0

        containers = _find_elements(root, "p") + _find_elements(root, "tc")

        for container in containers:
            merge_count += _merge_tracked_changes_in(container, "ins")
            merge_count += _merge_tracked_changes_in(container, "del")

        doc_xml.write_bytes(dom.toxml(encoding="UTF-8"))
        return merge_count, f"Simplified {merge_count} tracked changes"

    except Exception as e:
        return 0, f"Error: {e}"


def _merge_tracked_changes_in(container, tag: str) -> int:
    """Internal helper to merge tracked changes in.

    Args:
        container (Any): IN: container. OUT: Consumed during execution.
        tag (str): IN: tag. OUT: Consumed during execution.
    Returns:
        int: OUT: Result of the operation."""
    merge_count = 0

    tracked = [
        child for child in container.childNodes if child.nodeType == child.ELEMENT_NODE and _is_element(child, tag)
    ]

    if len(tracked) < 2:
        return 0

    i = 0
    while i < len(tracked) - 1:
        curr = tracked[i]
        next_elem = tracked[i + 1]

        if _can_merge_tracked(curr, next_elem):
            _merge_tracked_content(curr, next_elem)
            container.removeChild(next_elem)
            tracked.pop(i + 1)
            merge_count += 1
        else:
            i += 1

    return merge_count


def _is_element(node, tag: str) -> bool:
    """Internal helper to is element.

    Args:
        node (Any): IN: node. OUT: Consumed during execution.
        tag (str): IN: tag. OUT: Consumed during execution.
    Returns:
        bool: OUT: Result of the operation."""
    name = node.localName or node.tagName
    return name == tag or name.endswith(f":{tag}")


def _get_author(elem) -> str:
    """Internal helper to get author.

    Args:
        elem (Any): IN: elem. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""
    author = elem.getAttribute("w:author")
    if not author:
        for attr in elem.attributes.values():
            if attr.localName == "author" or attr.name.endswith(":author"):
                return attr.value
    return author


def _can_merge_tracked(elem1, elem2) -> bool:
    """Internal helper to can merge tracked.

    Args:
        elem1 (Any): IN: elem1. OUT: Consumed during execution.
        elem2 (Any): IN: elem2. OUT: Consumed during execution.
    Returns:
        bool: OUT: Result of the operation."""
    if _get_author(elem1) != _get_author(elem2):
        return False

    node = elem1.nextSibling
    while node and node != elem2:
        if node.nodeType == node.ELEMENT_NODE:
            return False
        if node.nodeType == node.TEXT_NODE and node.data.strip():
            return False
        node = node.nextSibling

    return True


def _merge_tracked_content(target, source):
    """Internal helper to merge tracked content.

    Args:
        target (Any): IN: target. OUT: Consumed during execution.
        source (Any): IN: source. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    while source.firstChild:
        child = source.firstChild
        source.removeChild(child)
        target.appendChild(child)


def _find_elements(root, tag: str) -> list:
    """Internal helper to find elements.

    Args:
        root (Any): IN: root. OUT: Consumed during execution.
        tag (str): IN: tag. OUT: Consumed during execution.
    Returns:
        list: OUT: Result of the operation."""
    results = []

    def traverse(node):
        """Traverse.

        Args:
            node (Any): IN: node. OUT: Consumed during execution.
        Returns:
            Any: OUT: Result of the operation."""
        if node.nodeType == node.ELEMENT_NODE:
            name = node.localName or node.tagName
            if name == tag or name.endswith(f":{tag}"):
                results.append(node)
            for child in node.childNodes:
                traverse(child)

    traverse(root)
    return results


def get_tracked_change_authors(doc_xml_path: Path) -> dict[str, int]:
    """Retrieve the tracked change authors.

    Args:
        doc_xml_path (Path): IN: doc xml path. OUT: Consumed during execution.
    Returns:
        dict[str, int]: OUT: Result of the operation."""
    if not doc_xml_path.exists():
        return {}

    try:
        tree = ET.parse(doc_xml_path)
        root = tree.getroot()
    except ET.ParseError:
        return {}

    namespaces = {"w": WORD_NS}
    author_attr = f"{{{WORD_NS}}}author"

    authors: dict[str, int] = {}
    for tag in ["ins", "del"]:
        for elem in root.findall(f".//w:{tag}", namespaces):
            author = elem.get(author_attr)
            if author:
                authors[author] = authors.get(author, 0) + 1

    return authors


def _get_authors_from_docx(docx_path: Path) -> dict[str, int]:
    """Internal helper to get authors from docx.

    Args:
        docx_path (Path): IN: docx path. OUT: Consumed during execution.
    Returns:
        dict[str, int]: OUT: Result of the operation."""
    try:
        with zipfile.ZipFile(docx_path, "r") as zf:
            if "word/document.xml" not in zf.namelist():
                return {}
            with zf.open("word/document.xml") as f:
                tree = ET.parse(f)
                root = tree.getroot()

                namespaces = {"w": WORD_NS}
                author_attr = f"{{{WORD_NS}}}author"

                authors: dict[str, int] = {}
                for tag in ["ins", "del"]:
                    for elem in root.findall(f".//w:{tag}", namespaces):
                        author = elem.get(author_attr)
                        if author:
                            authors[author] = authors.get(author, 0) + 1
                return authors
    except (zipfile.BadZipFile, ET.ParseError):
        return {}


def infer_author(modified_dir: Path, original_docx: Path, default: str = "Claude") -> str:
    """Infer author.

    Args:
        modified_dir (Path): IN: modified dir. OUT: Consumed during execution.
        original_docx (Path): IN: original docx. OUT: Consumed during execution.
        default (str, optional): IN: default. Defaults to 'Claude'. OUT: Consumed during execution.
    Returns:
        str: OUT: Result of the operation."""
    modified_xml = modified_dir / "word" / "document.xml"
    modified_authors = get_tracked_change_authors(modified_xml)

    if not modified_authors:
        return default

    original_authors = _get_authors_from_docx(original_docx)

    new_changes: dict[str, int] = {}
    for author, count in modified_authors.items():
        original_count = original_authors.get(author, 0)
        diff = count - original_count
        if diff > 0:
            new_changes[author] = diff

    if not new_changes:
        return default

    if len(new_changes) == 1:
        return next(iter(new_changes))

    raise ValueError(f"Multiple authors added new changes: {new_changes}. Cannot infer which author to validate.")
