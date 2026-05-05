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
"""Search arxiv module for Xerxes.

Exports:
    - NS
    - search"""

import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

NS = {"a": "http://www.w3.org/2005/Atom"}


def search(query=None, author=None, category=None, ids=None, max_results=5, sort="relevance"):
    """Search.

    Args:
        query (Any, optional): IN: query. Defaults to None. OUT: Consumed during execution.
        author (Any, optional): IN: author. Defaults to None. OUT: Consumed during execution.
        category (Any, optional): IN: category. Defaults to None. OUT: Consumed during execution.
        ids (Any, optional): IN: ids. Defaults to None. OUT: Consumed during execution.
        max_results (Any, optional): IN: max results. Defaults to 5. OUT: Consumed during execution.
        sort (Any, optional): IN: sort. Defaults to 'relevance'. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    params = {}
    """Search.

    Args:
        query (Any, optional): IN: query. Defaults to None. OUT: Consumed during execution.
        author (Any, optional): IN: author. Defaults to None. OUT: Consumed during execution.
        category (Any, optional): IN: category. Defaults to None. OUT: Consumed during execution.
        ids (Any, optional): IN: ids. Defaults to None. OUT: Consumed during execution.
        max_results (Any, optional): IN: max results. Defaults to 5. OUT: Consumed during execution.
        sort (Any, optional): IN: sort. Defaults to 'relevance'. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""
    """Search.

    Args:
        query (Any, optional): IN: query. Defaults to None. OUT: Consumed during execution.
        author (Any, optional): IN: author. Defaults to None. OUT: Consumed during execution.
        category (Any, optional): IN: category. Defaults to None. OUT: Consumed during execution.
        ids (Any, optional): IN: ids. Defaults to None. OUT: Consumed during execution.
        max_results (Any, optional): IN: max results. Defaults to 5. OUT: Consumed during execution.
        sort (Any, optional): IN: sort. Defaults to 'relevance'. OUT: Consumed during execution.
    Returns:
        Any: OUT: Result of the operation."""

    if ids:
        params["id_list"] = ids
    else:
        parts = []
        if query:
            parts.append(f"all:{urllib.parse.quote(query)}")
        if author:
            parts.append(f"au:{urllib.parse.quote(author)}")
        if category:
            parts.append(f"cat:{category}")
        if not parts:
            print("Error: provide a query, --author, --category, or --id")
            sys.exit(1)
        params["search_query"] = "+AND+".join(parts)

    params["max_results"] = str(max_results)

    sort_map = {"relevance": "relevance", "date": "submittedDate", "updated": "lastUpdatedDate"}
    params["sortBy"] = sort_map.get(sort, sort)
    params["sortOrder"] = "descending"

    url = "https://export.arxiv.org/api/query?" + "&".join(f"{k}={v}" for k, v in params.items())

    req = urllib.request.Request(url, headers={"User-Agent": "HermesAgent/1.0"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = resp.read()

    root = ET.fromstring(data)
    entries = root.findall("a:entry", NS)

    if not entries:
        print("No results found.")
        return

    total = root.find("{http://a9.com/-/spec/opensearch/1.1/}totalResults")
    if total is not None:
        print(f"Found {total.text} results (showing {len(entries)})\n")

    for i, entry in enumerate(entries):
        title = entry.find("a:title", NS).text.strip().replace("\n", " ")
        raw_id = entry.find("a:id", NS).text.strip()
        full_id = raw_id.split("/abs/")[-1] if "/abs/" in raw_id else raw_id
        arxiv_id = full_id.split("v")[0]
        published = entry.find("a:published", NS).text[:10]
        updated = entry.find("a:updated", NS).text[:10]
        authors = ", ".join(a.find("a:name", NS).text for a in entry.findall("a:author", NS))
        summary = entry.find("a:summary", NS).text.strip().replace("\n", " ")
        cats = ", ".join(c.get("term") for c in entry.findall("a:category", NS))

        version = full_id[len(arxiv_id) :] if full_id != arxiv_id else ""
        print(f"{i + 1}. {title}")
        print(f"   ID: {arxiv_id}{version} | Published: {published} | Updated: {updated}")
        print(f"   Authors: {authors}")
        print(f"   Categories: {cats}")
        print(f"   Abstract: {summary[:300]}{'...' if len(summary) > 300 else ''}")
        print(f"   Links: https://arxiv.org/abs/{arxiv_id} | https://arxiv.org/pdf/{arxiv_id}")
        print()


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    query = None
    author = None
    category = None
    ids = None
    max_results = 5
    sort = "relevance"

    i = 0
    positional = []
    while i < len(args):
        if args[i] == "--max" and i + 1 < len(args):
            max_results = int(args[i + 1])
            i += 2
        elif args[i] == "--sort" and i + 1 < len(args):
            sort = args[i + 1]
            i += 2
        elif args[i] == "--author" and i + 1 < len(args):
            author = args[i + 1]
            i += 2
        elif args[i] == "--category" and i + 1 < len(args):
            category = args[i + 1]
            i += 2
        elif args[i] == "--id" and i + 1 < len(args):
            ids = args[i + 1]
            i += 2
        else:
            positional.append(args[i])
            i += 1

    if positional:
        query = " ".join(positional)

    search(query=query, author=author, category=category, ids=ids, max_results=max_results, sort=sort)
