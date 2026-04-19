# Copyright 2025 The EasyDeL/Xerxes Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# distributed under the License is distributed on an "AS IS" BASIS,
# See the License for the specific language governing permissions and
# limitations under the License.


"""Xerxes Tools: A comprehensive collection of tools for AI agents.

This module provides a unified interface to various tool categories that enhance
AI agent capabilities. Tools are organized into logical categories and can be
easily integrated with Xerxes agents for file operations, web interactions,
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
    >>> from xerxes.tools import get_available_tools, ReadFile, Calculator
    >>>
    >>>
    >>> tools = get_available_tools()
    >>> print(tools.keys())
    dict_keys(['file_system', 'execution', 'web', 'data', 'ai', 'math', 'system', 'memory'])
    >>>
    >>>
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
from .google_search import (
    GoogleSearch,
    GoogleSearchConfig,
    configure_google_search,
    set_google_search_client,
)
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


from .agent_meta_tools import (
    configure_mixture_of_agents,
    mixture_of_agents,
    session_search,
    set_session_searcher,
    set_skill_registry,
    skill_manage,
    skill_view,
    skills_list,
)
from .ai_tools import EntityExtractor, TextClassifier, TextEmbedder, TextSimilarity, TextSummarizer
from .browser_tools import (
    BrowserSession,
    browser_back,
    browser_click,
    browser_console,
    browser_get_images,
    browser_navigate,
    browser_press,
    browser_scroll,
    browser_snapshot,
    browser_type,
    browser_vision,
)
from .claude_tools import (
    AgentTool,
    AskUserQuestionTool,
    EnterPlanModeTool,
    EnterWorktreeTool,
    ExitPlanModeTool,
    ExitWorktreeTool,
    FileEditTool,
    GlobTool,
    GrepTool,
    HandoffTool,
    ListMcpResourcesTool,
    LSPTool,
    MCPTool,
    NotebookEditTool,
    PlanTool,
    ReadMcpResourceTool,
    RemoteTriggerTool,
    ScheduleCronTool,
    SendMessageTool,
    SkillTool,
    SpawnAgents,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskOutputTool,
    TaskStopTool,
    TaskUpdateTool,
    TodoWriteTool,
    ToolSearchTool,
)
from .home_assistant_tools import (
    HomeAssistantClient,
    ha_call_service,
    ha_get_state,
    ha_list_entities,
    ha_list_services,
)
from .math_tools import Calculator, MathematicalFunctions, NumberTheory, StatisticalAnalyzer, UnitConverter
from .media_tools import (
    MediaConfig,
    configure_media,
    image_generate,
    set_media_client,
    text_to_speech,
    vision_analyze,
)
from .memory_tool import (
    MEMORY_TOOLS,
    consolidate_agent_memories,
    delete_memory,
    get_memory_statistics,
    get_memory_tool_descriptions,
    save_memory,
    search_memory,
)
from .rl_tools import (
    InMemoryRLBackend,
    RLBackend,
    get_rl_backend,
    reset_rl_backend,
    rl_check_status,
    rl_edit_config,
    rl_get_current_config,
    rl_get_results,
    rl_list_environments,
    rl_list_runs,
    rl_select_environment,
    rl_start_training,
    rl_stop_training,
    rl_test_inference,
    set_rl_backend,
)

__all__ = [
    "MEMORY_TOOLS",
    "AgentTool",
    "AppendFile",
    "AskUserQuestionTool",
    "BrowserSession",
    "CSVProcessor",
    "Calculator",
    "DataConverter",
    "DateTimeProcessor",
    "DuckDuckGoSearch",
    "EnterPlanModeTool",
    "EnterWorktreeTool",
    "EntityExtractor",
    "ExecutePythonCode",
    "ExecuteShell",
    "ExitPlanModeTool",
    "ExitWorktreeTool",
    "FileEditTool",
    "GlobTool",
    "GoogleSearch",
    "GoogleSearchConfig",
    "GrepTool",
    "HandoffTool",
    "HomeAssistantClient",
    "InMemoryRLBackend",
    "JSONProcessor",
    "LSPTool",
    "ListDir",
    "ListMcpResourcesTool",
    "MCPTool",
    "MathematicalFunctions",
    "MediaConfig",
    "NotebookEditTool",
    "NumberTheory",
    "PlanTool",
    "RLBackend",
    "ReadFile",
    "ReadMcpResourceTool",
    "RemoteTriggerTool",
    "ScheduleCronTool",
    "SendMessageTool",
    "SkillTool",
    "SpawnAgents",
    "StatisticalAnalyzer",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskOutputTool",
    "TaskStopTool",
    "TaskUpdateTool",
    "TextClassifier",
    "TextEmbedder",
    "TextProcessor",
    "TextSimilarity",
    "TextSummarizer",
    "TodoWriteTool",
    "ToolSearchTool",
    "UnitConverter",
    "WriteFile",
    "analyze_code_structure",
    "apply_diff",
    "browser_back",
    "browser_click",
    "browser_console",
    "browser_get_images",
    "browser_navigate",
    "browser_press",
    "browser_scroll",
    "browser_snapshot",
    "browser_type",
    "browser_vision",
    "configure_google_search",
    "configure_media",
    "configure_mixture_of_agents",
    "consolidate_agent_memories",
    "copy_file",
    "create_diff",
    "delete_file",
    "delete_memory",
    "find_and_replace",
    "get_memory_statistics",
    "get_memory_tool_descriptions",
    "get_rl_backend",
    "git_add",
    "git_apply_patch",
    "git_diff",
    "git_log",
    "git_status",
    "ha_call_service",
    "ha_get_state",
    "ha_list_entities",
    "ha_list_services",
    "image_generate",
    "list_directory",
    "mixture_of_agents",
    "move_file",
    "read_file",
    "reset_rl_backend",
    "rl_check_status",
    "rl_edit_config",
    "rl_get_current_config",
    "rl_get_results",
    "rl_list_environments",
    "rl_list_runs",
    "rl_select_environment",
    "rl_start_training",
    "rl_stop_training",
    "rl_test_inference",
    "save_memory",
    "search_memory",
    "session_search",
    "set_google_search_client",
    "set_media_client",
    "set_rl_backend",
    "set_session_searcher",
    "set_skill_registry",
    "skill_manage",
    "skill_view",
    "skills_list",
    "text_to_speech",
    "vision_analyze",
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
    "file_system": [
        "ReadFile",
        "WriteFile",
        "AppendFile",
        "ListDir",
        "FileEditTool",
        "GlobTool",
        "GrepTool",
        "FileSystemTools",
        "TempFileManager",
    ],
    "execution": ["ExecutePythonCode", "ExecuteShell", "ProcessManager"],
    "web": ["GoogleSearch", "DuckDuckGoSearch", "WebScraper", "APIClient", "RSSReader", "URLAnalyzer"],
    "data": ["JSONProcessor", "CSVProcessor", "TextProcessor", "DataConverter", "DateTimeProcessor"],
    "ai": ["TextEmbedder", "TextSimilarity", "TextClassifier", "TextSummarizer", "EntityExtractor"],
    "math": ["Calculator", "StatisticalAnalyzer", "MathematicalFunctions", "NumberTheory", "UnitConverter"],
    "system": ["SystemInfo", "EnvironmentManager", "ProcessManager"],
    "memory": ["save_memory", "search_memory", "consolidate_agent_memories", "delete_memory", "get_memory_statistics"],
    "agent": [
        "AgentTool",
        "HandoffTool",
        "PlanTool",
        "SendMessageTool",
        "SpawnAgents",
        "TaskCreateTool",
        "TaskGetTool",
        "TaskListTool",
        "TaskOutputTool",
        "TaskStopTool",
        "TaskUpdateTool",
    ],
    "workflow": [
        "TodoWriteTool",
        "AskUserQuestionTool",
        "EnterPlanModeTool",
        "ExitPlanModeTool",
        "EnterWorktreeTool",
        "ExitWorktreeTool",
        "ToolSearchTool",
        "SkillTool",
    ],
    "notebook": ["NotebookEditTool"],
    "lsp": ["LSPTool"],
    "mcp": ["MCPTool", "ListMcpResourcesTool", "ReadMcpResourceTool"],
    "remote": ["RemoteTriggerTool", "ScheduleCronTool"],
    "browser": [
        "browser_navigate",
        "browser_back",
        "browser_click",
        "browser_type",
        "browser_press",
        "browser_scroll",
        "browser_snapshot",
        "browser_vision",
        "browser_get_images",
        "browser_console",
    ],
    "home_assistant": [
        "ha_list_entities",
        "ha_list_services",
        "ha_get_state",
        "ha_call_service",
    ],
    "rl": [
        "rl_list_environments",
        "rl_select_environment",
        "rl_get_current_config",
        "rl_edit_config",
        "rl_start_training",
        "rl_stop_training",
        "rl_check_status",
        "rl_get_results",
        "rl_list_runs",
        "rl_test_inference",
    ],
    "media": [
        "image_generate",
        "vision_analyze",
        "text_to_speech",
    ],
    "meta": [
        "mixture_of_agents",
        "session_search",
        "skill_view",
        "skills_list",
        "skill_manage",
    ],
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
    "TextEmbedder": "xerxes[vectors] for advanced methods",
    "TextSimilarity": "xerxes[vectors] for semantic similarity",
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
        >>>
        >>> math_tools = list_tools_by_category("math")
        >>> print(math_tools)
        ['Calculator', 'StatisticalAnalyzer', 'MathematicalFunctions', 'NumberTheory', 'UnitConverter']
        >>>
        >>>
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
