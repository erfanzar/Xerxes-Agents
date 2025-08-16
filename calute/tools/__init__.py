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

"""Calute Tools - A comprehensive collection of tools for AI agents."""

# Core file system tools
# Web search tools
from .duckduckgo_engine import DuckDuckGoSearch
from .standalone import AppendFile, ExecutePythonCode, ExecuteShell, ListDir, ReadFile, WriteFile

# Web tools (requires search extras)
try:
    from .web_tools import APIClient, RSSReader, URLAnalyzer, WebScraper
    _WEB_TOOLS_AVAILABLE = True
except ImportError:
    _WEB_TOOLS_AVAILABLE = False

# Data processing tools
from .data_tools import CSVProcessor, DataConverter, DateTimeProcessor, JSONProcessor, TextProcessor

# System tools (requires psutil)
try:
    from .system_tools import EnvironmentManager, FileSystemTools, ProcessManager, SystemInfo, TempFileManager
    _SYSTEM_TOOLS_AVAILABLE = True
except ImportError:
    _SYSTEM_TOOLS_AVAILABLE = False

# AI tools
from .ai_tools import EntityExtractor, TextClassifier, TextEmbedder, TextSimilarity, TextSummarizer

# Math tools
from .math_tools import Calculator, MathematicalFunctions, NumberTheory, StatisticalAnalyzer, UnitConverter

# Core tools (always available)
__all__ = [
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
]

# Add web tools if available
if _WEB_TOOLS_AVAILABLE:
    __all__.extend([
        "APIClient",
        "RSSReader",
        "URLAnalyzer",
        "WebScraper",
    ])

# Add system tools if available
if _SYSTEM_TOOLS_AVAILABLE:
    __all__.extend([
        "EnvironmentManager",
        "FileSystemTools",
        "ProcessManager",
        "SystemInfo",
        "TempFileManager",
    ])

# Tool categories for easy discovery
TOOL_CATEGORIES = {
    "file_system": [
        "ReadFile", "WriteFile", "AppendFile", "ListDir",
        "FileSystemTools", "TempFileManager"
    ],
    "execution": [
        "ExecutePythonCode", "ExecuteShell", "ProcessManager"
    ],
    "web": [
        "DuckDuckGoSearch", "WebScraper", "APIClient", "RSSReader", "URLAnalyzer"
    ],
    "data": [
        "JSONProcessor", "CSVProcessor", "TextProcessor",
        "DataConverter", "DateTimeProcessor"
    ],
    "ai": [
        "TextEmbedder", "TextSimilarity", "TextClassifier",
        "TextSummarizer", "EntityExtractor"
    ],
    "math": [
        "Calculator", "StatisticalAnalyzer", "MathematicalFunctions",
        "NumberTheory", "UnitConverter"
    ],
    "system": [
        "SystemInfo", "EnvironmentManager", "ProcessManager"
    ]
}

# Requirements for optional tools
TOOL_REQUIREMENTS = {
    "WebScraper": "calute[search]",
    "APIClient": "httpx (included in core)",
    "RSSReader": "feedparser",
    "SystemInfo": "psutil",
    "ProcessManager": "psutil",
    "FileSystemTools": "core",
    "TextEmbedder": "calute[vectors] for advanced methods",
    "TextSimilarity": "calute[vectors] for semantic similarity",
}

def get_available_tools() -> dict:
    """Get list of available tools organized by category."""
    available = {}

    for category, tools in TOOL_CATEGORIES.items():
        available[category] = []
        for tool in tools:
            if tool in globals():
                available[category].append(tool)

    return available

def get_tool_info(tool_name: str) -> dict:
    """Get information about a specific tool."""
    if tool_name not in __all__:
        return {"error": f"Tool {tool_name} not found"}

    tool_class = globals().get(tool_name)
    if not tool_class:
        return {"error": f"Tool {tool_name} not available"}

    # Find category
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

    # Try to get docstring
    if hasattr(tool_class, 'static_call'):
        doc = tool_class.static_call.__doc__
        if doc:
            info["description"] = doc.strip().split('\n')[0]

    return info

def list_tools_by_category(category: str | None = None) -> list:
    """List tools by category."""
    if category:
        if category not in TOOL_CATEGORIES:
            return []
        return [tool for tool in TOOL_CATEGORIES[category] if tool in __all__]

    # Return all categories
    result = {}
    for cat in TOOL_CATEGORIES:
        result[cat] = list_tools_by_category(cat)
    return result
