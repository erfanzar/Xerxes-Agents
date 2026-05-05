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
"""Extract pymupdf module for Xerxes.

Exports:
    - extract_text
    - extract_markdown
    - extract_tables
    - extract_images
    - show_metadata"""

import json
import sys


def extract_text(path, pages=None):
    """Extract text.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
        pages (Any, optional): IN: pages. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    import pymupdf

    """Extract text.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
        pages (Any, optional): IN: pages. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    """Extract text.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
        pages (Any, optional): IN: pages. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    doc = pymupdf.open(path)
    page_range = range(len(doc)) if pages is None else pages
    for i in page_range:
        if i < len(doc):
            print(f"\n--- Page {i + 1}/{len(doc)} ---\n")
            print(doc[i].get_text())


def extract_markdown(path, pages=None):
    """Extract markdown.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
        pages (Any, optional): IN: pages. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    import pymupdf4llm

    """Extract markdown.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
        pages (Any, optional): IN: pages. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    """Extract markdown.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
        pages (Any, optional): IN: pages. Defaults to None. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    md = pymupdf4llm.to_markdown(path, pages=pages)
    print(md)


def extract_tables(path):
    """Extract tables.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    import pymupdf

    """Extract tables.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    """Extract tables.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    doc = pymupdf.open(path)
    for i, page in enumerate(doc):
        tables = page.find_tables()
        for j, table in enumerate(tables.tables):
            print(f"\n--- Page {i + 1}, Table {j + 1} ---\n")
            df = table.to_pandas()
            print(df.to_markdown(index=False))


def extract_images(path, output_dir):
    """Extract images.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
        output_dir (Any): IN: output dir. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    from pathlib import Path

    """Extract images.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
        output_dir (Any): IN: output dir. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    """Extract images.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
        output_dir (Any): IN: output dir. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    import pymupdf

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(path)
    count = 0
    for i, page in enumerate(doc):
        for img_idx, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = pymupdf.Pixmap(doc, xref)
            if pix.n >= 5:
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            out_path = f"{output_dir}/page{i + 1}_img{img_idx + 1}.png"
            pix.save(out_path)
            count += 1
    print(f"Extracted {count} images to {output_dir}/")


def show_metadata(path):
    """Show metadata.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    import pymupdf

    """Show metadata.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    """Show metadata.

    Args:
        path (Any): IN: path. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    doc = pymupdf.open(path)
    print(
        json.dumps(
            {
                "pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "format": doc.metadata.get("format", ""),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    path = args[0]
    pages = None

    if "--pages" in args:
        idx = args.index("--pages")
        p = args[idx + 1]
        if "-" in p:
            start, end = p.split("-")
            pages = list(range(int(start), int(end) + 1))
        else:
            pages = [int(p)]

    if "--metadata" in args:
        show_metadata(path)
    elif "--tables" in args:
        extract_tables(path)
    elif "--images" in args:
        idx = args.index("--images")
        output_dir = args[idx + 1] if idx + 1 < len(args) else "./images"
        extract_images(path, output_dir)
    elif "--markdown" in args:
        extract_markdown(path, pages=pages)
    else:
        extract_text(path, pages=pages)
