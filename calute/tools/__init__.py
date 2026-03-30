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


"""Calute Tools: A comprehensive collection of tools for AI agents.

This module provides a unified interface to various tool categories that enhance
AI agent capabilities. Tools are organized into logical categories and can be
easily integrated with Calute agents for file operations, web interactions,
data processing, mathematical computations, and more.

Key Features:
    - File System Tools: Read, write, append, and manage files and directories
    - Execution Tools: Execute Python code and shell commands safely
    - Web Tools: Search, scrape, and interact with web content
    - Data Tools: Process JSON, CSV, text, and perform data conversions
    - AI Tools: Text embedding, similarity, classification, and summarization
    - Math Tools: Calculations, statistics, and unit conversions
    - System Tools: System information, process management, and environment handling
    - Memory Tools: Persistent memory storage and retrieval for agents

Example:
    >>> from calute.tools import get_available_tools, ReadFile, Calculator
    >>>
    >>> # Get all available tools by category
    >>> tools = get_available_tools()
    >>> print(tools.keys())
    dict_keys(['file_system', 'execution', 'web', 'data', 'ai', 'math', 'system', 'memory'])
    >>>
    >>> # Use individual tools
    >>> content = ReadFile.static_call(file_path="config.json")
    >>> result = Calculator.static_call(expression="2 + 2")

Note:
    Some tools require additional dependencies. Use ``get_tool_info(tool_name)``
    to check requirements for a specific tool.
"""

from .coding_tools import (
    analyze_code_structure,
    apply_diff,
    copy_file,
    create_diff,
    delete_file,
    find_and_replace,
    git_add,
    git_apply_patch,
    git_diff,
    git_log,
    git_status,
    list_directory,
    move_file,
    read_file,
    write_file,
)
from .duckduckgo_engine import DuckDuckGoSearch
from .standalone import AppendFile, ExecutePythonCode, ExecuteShell, ListDir, ReadFile, WriteFile

try:
    from .web_tools import APIClient, RSSReader, URLAnalyzer, WebScraper

    _WEB_TOOLS_AVAILABLE = True
except ImportError:
    _WEB_TOOLS_AVAILABLE = False


from .data_tools import CSVProcessor, DataConverter, DateTimeProcessor, JSONProcessor, TextProcessor

try:
    from .system_tools import EnvironmentManager, FileSystemTools, ProcessManager, SystemInfo, TempFileManager

    _SYSTEM_TOOLS_AVAILABLE = True
except ImportError:
    _SYSTEM_TOOLS_AVAILABLE = False


from .ai_tools import EntityExtractor, TextClassifier, TextEmbedder, TextSimilarity, TextSummarizer
from .math_tools import Calculator, MathematicalFunctions, NumberTheory, StatisticalAnalyzer, UnitConverter
from .memory_tool import (
    MEMORY_TOOLS,
    consolidate_agent_memories,
    delete_memory,
    get_memory_statistics,
    get_memory_tool_descriptions,
    save_memory,
    search_memory,
)

__all__ = [
    "MEMORY_TOOLS",
    "AppendFile",
    "CSVProcessor",
    "Calculator",
    "DataConverter",
    "DateTimeProcessor",
    "DuckDuckGoSearch",
    "EntityExtractor",
    "ExecutePythonCode",
    "ExecuteShell",
    "JSONProcessor",
    "ListDir",
    "MathematicalFunctions",
    "NumberTheory",
    "ReadFile",
    "StatisticalAnalyzer",
    "TextClassifier",
    "TextEmbedder",
    "TextProcessor",
    "TextSimilarity",
    "TextSummarizer",
    "UnitConverter",
    "WriteFile",
    "analyze_code_structure",
    "apply_diff",
    "consolidate_agent_memories",
    "copy_file",
    "create_diff",
    "delete_file",
    "delete_memory",
    "find_and_replace",
    "get_memory_statistics",
    "get_memory_tool_descriptions",
    "git_add",
    "git_apply_patch",
    "git_diff",
    "git_log",
    "git_status",
    "list_directory",
    "move_file",
    "read_file",
    "save_memory",
    "search_memory",
    "write_file",
]


if _WEB_TOOLS_AVAILABLE:
    __all__.extend(
        [
            "APIClient",
            "RSSReader",
            "URLAnalyzer",
            "WebScraper",
        ]
    )


if _SYSTEM_TOOLS_AVAILABLE:
    __all__.extend(
        [
            "EnvironmentManager",
            "FileSystemTools",
            "ProcessManager",
            "SystemInfo",
            "TempFileManager",
        ]
    )


TOOL_CATEGORIES: dict[str, list[str]] = {
    "file_system": ["ReadFile", "WriteFile", "AppendFile", "ListDir", "FileSystemTools", "TempFileManager"],
    "execution": ["ExecutePythonCode", "ExecuteShell", "ProcessManager"],
    "web": ["DuckDuckGoSearch", "WebScraper", "APIClient", "RSSReader", "URLAnalyzer"],
    "data": ["JSONProcessor", "CSVProcessor", "TextProcessor", "DataConverter", "DateTimeProcessor"],
    "ai": ["TextEmbedder", "TextSimilarity", "TextClassifier", "TextSummarizer", "EntityExtractor"],
    "math": ["Calculator", "StatisticalAnalyzer", "MathematicalFunctions", "NumberTheory", "UnitConverter"],
    "system": ["SystemInfo", "EnvironmentManager", "ProcessManager"],
    "memory": ["save_memory", "search_memory", "consolidate_agent_memories", "delete_memory", "get_memory_statistics"],
}
"""Mapping of tool category names to their constituent tool names.

Categories include:
    - file_system: File and directory operations
    - execution: Code and command execution
    - web: Web scraping, search, and API interaction
    - data: Data format processing and conversion
    - ai: AI-powered text processing tools
    - math: Mathematical and statistical operations
    - system: System information and management
    - memory: Agent memory persistence and retrieval
"""


TOOL_REQUIREMENTS: dict[str, str] = {
    "WebScraper": "beautifulsoup4 (included in core)",
    "APIClient": "httpx (included in core)",
    "RSSReader": "feedparser (included in core)",
    "SystemInfo": "psutil",
    "ProcessManager": "psutil",
    "FileSystemTools": "core",
    "TextEmbedder": "calute[vectors] for advanced methods",
    "TextSimilarity": "calute[vectors] for semantic similarity",
}
"""Mapping of tool names to their installation requirements.

Tools not listed here are available in the core package without
additional dependencies. Use this mapping to determine what extras
need to be installed for specific tool functionality.
"""


def get_available_tools() -> dict[str, list[str]]:
    """Get a dictionary of available tools organized by category.

    This function inspects the module's global namespace to determine
    which tools are currently available (successfully imported) and
    organizes them by their respective categories.

    Returns:
        A dictionary mapping category names to lists of available tool names.
        Only tools that are successfully imported are included.

    Example:
        >>> tools = get_available_tools()
        >>> print(tools["file_system"])
        ['ReadFile', 'WriteFile', 'AppendFile', 'ListDir']
        >>> print(tools["math"])
        ['Calculator', 'StatisticalAnalyzer', 'MathematicalFunctions', 'NumberTheory', 'UnitConverter']
    """
    available = {}

    for category, tools in TOOL_CATEGORIES.items():
        available[category] = []
        for tool in tools:
            if tool in globals():
                available[category].append(tool)

    return available


def get_tool_info(tool_name: str) -> dict[str, str | bool | None]:
    """Get detailed information about a specific tool.

    Retrieves metadata about a tool including its category, availability
    status, installation requirements, and description (if available).

    Args:
        tool_name: The name of the tool to get information about.

    Returns:
        A dictionary containing tool information with the following keys:
            - name: The tool name
            - category: The category the tool belongs to (or None)
            - available: Whether the tool is currently available
            - requirements: Installation requirements for the tool
            - description: First line of the tool's docstring (if available)
            - error: Error message if the tool was not found

    Example:
        >>> info = get_tool_info("Calculator")
        >>> print(info["category"])
        'math'
        >>> print(info["requirements"])
        'core'
        >>>
        >>> info = get_tool_info("NonExistentTool")
        >>> print(info.get("error"))
        'Tool NonExistentTool not found'
    """
    if tool_name not in __all__:
        return {"error": f"Tool {tool_name} not found"}

    tool_class = globals().get(tool_name)
    if not tool_class:
        return {"error": f"Tool {tool_name} not available"}

    category = None
    for cat, tools in TOOL_CATEGORIES.items():
        if tool_name in tools:
            category = cat
            break

    info = {
        "name": tool_name,
        "category": category,
        "available": True,
        "requirements": TOOL_REQUIREMENTS.get(tool_name, "core"),
    }

    if hasattr(tool_class, "static_call"):
        doc = tool_class.static_call.__doc__
        if doc:
            info["description"] = doc.strip().split("\n")[0]

    return info


def list_tools_by_category(category: str | None = None) -> list[str] | dict[str, list[str]]:
    """List available tools, optionally filtered by category.

    This function returns tools that are both defined in TOOL_CATEGORIES
    and exported in __all__. It can either return tools for a specific
    category or all tools organized by category.

    Args:
        category: The category to filter by. If None, returns all tools
            organized by category. Valid categories are: 'file_system',
            'execution', 'web', 'data', 'ai', 'math', 'system', 'memory'.

    Returns:
        If category is specified: A list of tool names in that category.
        If category is None: A dictionary mapping category names to lists
        of tool names.
        Returns an empty list if the specified category does not exist.

    Example:
        >>> # Get tools for a specific category
        >>> math_tools = list_tools_by_category("math")
        >>> print(math_tools)
        ['Calculator', 'StatisticalAnalyzer', 'MathematicalFunctions', 'NumberTheory', 'UnitConverter']
        >>>
        >>> # Get all tools organized by category
        >>> all_tools = list_tools_by_category()
        >>> print(all_tools.keys())
        dict_keys(['file_system', 'execution', 'web', 'data', 'ai', 'math', 'system', 'memory'])
    """
    if category:
        if category not in TOOL_CATEGORIES:
            return []
        return [tool for tool in TOOL_CATEGORIES[category] if tool in __all__]

    result = {}
    for cat in TOOL_CATEGORIES:
        result[cat] = list_tools_by_category(cat)
    return result
