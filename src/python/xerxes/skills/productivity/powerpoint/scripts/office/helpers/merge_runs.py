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
"""Merge runs module for Xerxes.

Exports:
    - merge_runs"""

from pathlib import Path

import defusedxml.minidom


def merge_runs(input_dir: str) -> tuple[int, str]:
    """Merge runs.

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

        _remove_elements(root, "proofErr")
        _strip_run_rsid_attrs(root)

        containers = {run.parentNode for run in _find_elements(root, "r")}

        merge_count = 0
        for container in containers:
            merge_count += _merge_runs_in(container)

        doc_xml.write_bytes(dom.toxml(encoding="UTF-8"))
        return merge_count, f"Merged {merge_count} runs"

    except Exception as e:
        return 0, f"Error: {e}"


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


def _get_child(parent, tag: str):
    """Internal helper to get child.

    Args:
        parent (Any): IN: parent. OUT: Consumed during execution.
        tag (str): IN: tag. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    for child in parent.childNodes:
        if child.nodeType == child.ELEMENT_NODE:
            name = child.localName or child.tagName
            if name == tag or name.endswith(f":{tag}"):
                return child
    return None


def _get_children(parent, tag: str) -> list:
    """Internal helper to get children.

    Args:
        parent (Any): IN: parent. OUT: Consumed during execution.
        tag (str): IN: tag. OUT: Consumed during execution.
    Returns:
        list: OUT: Result of the operation."""
    results = []
    for child in parent.childNodes:
        if child.nodeType == child.ELEMENT_NODE:
            name = child.localName or child.tagName
            if name == tag or name.endswith(f":{tag}"):
                results.append(child)
    return results


def _is_adjacent(elem1, elem2) -> bool:
    """Internal helper to is adjacent.

    Args:
        elem1 (Any): IN: elem1. OUT: Consumed during execution.
        elem2 (Any): IN: elem2. OUT: Consumed during execution.
    Returns:
        bool: OUT: Result of the operation."""
    node = elem1.nextSibling
    while node:
        if node == elem2:
            return True
        if node.nodeType == node.ELEMENT_NODE:
            return False
        if node.nodeType == node.TEXT_NODE and node.data.strip():
            return False
        node = node.nextSibling
    return False


def _remove_elements(root, tag: str):
    """Internal helper to remove elements.

    Args:
        root (Any): IN: root. OUT: Consumed during execution.
        tag (str): IN: tag. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    for elem in _find_elements(root, tag):
        if elem.parentNode:
            elem.parentNode.removeChild(elem)


def _strip_run_rsid_attrs(root):
    """Internal helper to strip run rsid attrs.

    Args:
        root (Any): IN: root. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    for run in _find_elements(root, "r"):
        for attr in list(run.attributes.values()):
            if "rsid" in attr.name.lower():
                run.removeAttribute(attr.name)


def _merge_runs_in(container) -> int:
    """Internal helper to merge runs in.

    Args:
        container (Any): IN: container. OUT: Consumed during execution.
    Returns:
        int: OUT: Result of the operation."""
    merge_count = 0
    run = _first_child_run(container)

    while run:
        while True:
            next_elem = _next_element_sibling(run)
            if next_elem and _is_run(next_elem) and _can_merge(run, next_elem):
                _merge_run_content(run, next_elem)
                container.removeChild(next_elem)
                merge_count += 1
            else:
                break

        _consolidate_text(run)
        run = _next_sibling_run(run)

    return merge_count


def _first_child_run(container):
    """Internal helper to first child run.

    Args:
        container (Any): IN: container. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    for child in container.childNodes:
        if child.nodeType == child.ELEMENT_NODE and _is_run(child):
            return child
    return None


def _next_element_sibling(node):
    """Internal helper to next element sibling.

    Args:
        node (Any): IN: node. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    sibling = node.nextSibling
    while sibling:
        if sibling.nodeType == sibling.ELEMENT_NODE:
            return sibling
        sibling = sibling.nextSibling
    return None


def _next_sibling_run(node):
    """Internal helper to next sibling run.

    Args:
        node (Any): IN: node. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    sibling = node.nextSibling
    while sibling:
        if sibling.nodeType == sibling.ELEMENT_NODE:
            if _is_run(sibling):
                return sibling
        sibling = sibling.nextSibling
    return None


def _is_run(node) -> bool:
    """Internal helper to is run.

    Args:
        node (Any): IN: node. OUT: Consumed during execution.
    Returns:
        bool: OUT: Result of the operation."""
    name = node.localName or node.tagName
    return name == "r" or name.endswith(":r")


def _can_merge(run1, run2) -> bool:
    """Internal helper to can merge.

    Args:
        run1 (Any): IN: run1. OUT: Consumed during execution.
        run2 (Any): IN: run2. OUT: Consumed during execution.
    Returns:
        bool: OUT: Result of the operation."""
    rpr1 = _get_child(run1, "rPr")
    rpr2 = _get_child(run2, "rPr")

    if (rpr1 is None) != (rpr2 is None):
        return False
    if rpr1 is None:
        return True
    return rpr1.toxml() == rpr2.toxml()


def _merge_run_content(target, source):
    """Internal helper to merge run content.

    Args:
        target (Any): IN: target. OUT: Consumed during execution.
        source (Any): IN: source. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    for child in list(source.childNodes):
        if child.nodeType == child.ELEMENT_NODE:
            name = child.localName or child.tagName
            if name != "rPr" and not name.endswith(":rPr"):
                target.appendChild(child)


def _consolidate_text(run):
    """Internal helper to consolidate text.

    Args:
        run (Any): IN: run. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    t_elements = _get_children(run, "t")

    for i in range(len(t_elements) - 1, 0, -1):
        curr, prev = t_elements[i], t_elements[i - 1]

        if _is_adjacent(prev, curr):
            prev_text = prev.firstChild.data if prev.firstChild else ""
            curr_text = curr.firstChild.data if curr.firstChild else ""
            merged = prev_text + curr_text

            if prev.firstChild:
                prev.firstChild.data = merged
            else:
                prev.appendChild(run.ownerDocument.createTextNode(merged))

            if merged.startswith(" ") or merged.endswith(" "):
                prev.setAttribute("xml:space", "preserve")
            elif prev.hasAttribute("xml:space"):
                prev.removeAttribute("xml:space")

            run.removeChild(curr)
