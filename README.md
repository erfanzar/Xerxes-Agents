# Calute

**Agents for intelligence and coordination.**

Calute is an advanced AI agent orchestration framework for building multi-agent systems with intelligent collaboration, memory management, tool integration, and enterprise-grade governance. It supports multiple LLM providers, offers both sync and async execution, and ships with a terminal UI, web UI, and an OpenAI-compatible API server.

---

## Features

- **Multi-Agent Orchestration** &mdash; Register, switch, and coordinate agents with sequential, parallel, hierarchical, consensus, and planned execution strategies.
- **Cortex Framework** &mdash; High-level multi-agent pipelines with tasks, tools, planning, and memory-aware agents.
- **Multi-Provider LLMs** &mdash; Unified interface for OpenAI, Anthropic, Google Gemini, and Ollama/local models.
- **Advanced Memory** &mdash; Five memory types (short-term, long-term, contextual, entity, user) backed by in-memory, SQLite, or vector storage.
- **Comprehensive Tooling** &mdash; 50+ built-in tools across file I/O, web search/scraping, data processing, math, AI text analysis, system utilities, and coding.
- **MCP Integration** &mdash; Model Context Protocol client for connecting to external tool servers via STDIO, SSE, or Streamable HTTP.
- **Security & Sandboxing** &mdash; Tool policies (allow/deny/optional), sandbox execution backends (host, Docker, subprocess).
- **Session Persistence & Replay** &mdash; Record, persist, and replay agent sessions with full audit trails.
- **Plugin & Skill System** &mdash; Extensible architecture with plugin registry, SKILL.md discovery, lifecycle hooks, and dependency resolution.
- **Loop Detection** &mdash; Automatic detection of same-call, ping-pong, and max-iteration tool loops.
- **Structured Audit Events** &mdash; 12 typed event classes with in-memory, JSONL, and composite collectors.
- **Terminal UI** &mdash; Textual-based interactive TUI with the `calute` command.
- **Web UI** &mdash; Optional Chainlit-based web interface.
- **API Server** &mdash; FastAPI server with OpenAI-compatible `/v1/chat/completions` and `/v1/models` endpoints.
- **Context Management** &mdash; Token counting and compaction strategies (summarization, sliding window, priority-based, smart hybrid).

---

## Installation

```bash
pip install calute
```

### Extras

```bash
pip install calute[full]          # Everything
pip install calute[ui]            # Chainlit web UI
pip install calute[monitoring]    # OpenTelemetry, Prometheus, Sentry, Datadog
pip install calute[vectors]       # scikit-learn, sentence-transformers for RAG
pip install calute[mcp]           # Model Context Protocol support
pip install calute[dev]           # Development tools (pytest, ruff, black, mypy)
```

### From source

```bash
git clone https://github.com/erfanzar/Calute.git
cd Calute
pip install -e ".[dev]"
```

---

## Quick Start

### Single Agent

```python
import asyncio
from calute import Agent, Calute, MessagesHistory, UserMessage
import openai

client = openai.OpenAI(api_key="your-key")

def greet(name: str) -> str:
    """Greet a user by name."""
    return f"Hello, {name}!"

agent = Agent(
    id="assistant",
    name="My Assistant",
    model="gpt-4o",
    instructions="You are a helpful assistant.",
    functions=[greet],
)

calute = Calute(client)
calute.register_agent(agent)

async def main():
    messages = MessagesHistory(messages=[])
    messages.messages.append(UserMessage(content="Say hi to Alice"))
    response = await calute.create_response(
        prompt="Say hi to Alice",
        messages=messages,
        agent_id="assistant",
        apply_functions=True,
    )
    print(response.content)

asyncio.run(main())
```

### Cortex Multi-Agent Pipeline

```python
from calute import (
    Cortex, CortexAgent, CortexTask, ProcessType, create_llm,
)

llm = create_llm("openai", api_key="your-key")

researcher = CortexAgent(
    role="Researcher",
    goal="Find accurate information",
    backstory="Expert research analyst",
    llm=llm,
)

writer = CortexAgent(
    role="Writer",
    goal="Produce clear reports",
    backstory="Technical writer",
    llm=llm,
)

research_task = CortexTask(
    description="Research recent advances in AI agents",
    expected_output="A summary of key findings",
    agent=researcher,
)

write_task = CortexTask(
    description="Write a report from the research findings",
    expected_output="A polished report",
    agent=writer,
)

cortex = Cortex(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=ProcessType.SEQUENTIAL,
)

result = cortex.kickoff()
print(result.final_output)
```

### Memory-Aware Agent

```python
from calute import Calute
from calute.memory import MemoryStore, MemoryType

memory = MemoryStore(
    max_short_term=100,
    enable_persistence=True,
    persistence_path="~/.calute/memory",
)

memory.add_memory(
    content="User prefers dark mode",
    memory_type=MemoryType.LONG_TERM,
    agent_id="assistant",
    tags=["preference"],
    importance_score=0.9,
)

calute = Calute(client, enable_memory=True)
calute.memory = memory
```

### Using Different LLM Providers

```python
from calute import create_llm, OpenAILLM, AnthropicLLM, GeminiLLM, OllamaLLM

# OpenAI
llm = create_llm("openai", api_key="sk-...")

# Anthropic Claude (uses httpx directly)
llm = create_llm("anthropic", api_key="sk-ant-...")

# Google Gemini
llm = create_llm("gemini", api_key="...")

# Local via Ollama
llm = create_llm("ollama", model="llama3")

# Custom OpenAI-compatible endpoint
llm = OpenAILLM(api_key="sk-xxx", base_url="http://localhost:8080/v1/")
```

---

## CLI

The package installs a `calute` command that launches the Textual terminal UI:

```bash
calute
```

---

## API Server

Run an OpenAI-compatible API server backed by Calute agents:

```python
from calute.api_server import CaluteAPIServer

server = CaluteAPIServer()
server.run(host="0.0.0.0", port=8000)
```

Endpoints:

- `POST /v1/chat/completions` &mdash; Chat completions (streaming and non-streaming)
- `GET /v1/models` &mdash; List available models

---

## Tools

Built-in tools organized by category:

| Category        | Tools                                                                                                            |
| --------------- | ---------------------------------------------------------------------------------------------------------------- |
| **File System** | `ReadFile`, `WriteFile`, `AppendFile`, `ListDir`, `copy_file`, `move_file`, `delete_file`, `git_diff`, `git_log` |
| **Execution**   | `ExecutePythonCode`, `ExecuteShell`, `ProcessManager`                                                            |
| **Web**         | `DuckDuckGoSearch`, `WebScraper`, `APIClient`, `RSSReader`, `URLAnalyzer`                                        |
| **Data**        | `JSONProcessor`, `CSVProcessor`, `TextProcessor`, `DataConverter`, `DateTimeProcessor`                           |
| **AI**          | `TextEmbedder`, `TextSimilarity`, `TextClassifier`, `TextSummarizer`, `EntityExtractor`                          |
| **Math**        | `Calculator`, `StatisticalAnalyzer`, `MathematicalFunctions`, `NumberTheory`, `UnitConverter`                    |
| **System**      | `SystemInfo`, `EnvironmentManager`, `ProcessManager`                                                             |
| **Memory**      | `save_memory`, `search_memory`, `consolidate_agent_memories`, `delete_memory`                                    |

```python
from calute.tools import get_available_tools, list_tools_by_category

all_tools = get_available_tools()
web_tools = list_tools_by_category("web")
```

---

## Memory System

Five memory types with pluggable storage:

| Memory Type        | Purpose                          |
| ------------------ | -------------------------------- |
| `ShortTermMemory`  | Current conversation context     |
| `LongTermMemory`   | Persistent important information |
| `ContextualMemory` | Situation-aware adaptive memory  |
| `EntityMemory`     | Entity/attribute tracking        |
| `UserMemory`       | Per-user personalization         |

Storage backends: `SimpleStorage` (in-memory), `SQLiteStorage` (persistent), `RAGStorage` (vector semantic search).

---

## Security

### Tool Policies

```python
from calute.security import ToolPolicy, PolicyEngine, PolicyAction

policy = ToolPolicy(
    allowed_tools=["search", "read_file"],
    denied_tools=["execute_shell"],
)
engine = PolicyEngine(policy)
```

### Sandbox Execution

Tools can be routed to sandboxed environments:

- **Host** &mdash; Direct execution (default)
- **Docker** &mdash; Containerized isolation
- **Subprocess** &mdash; Process-level isolation

---

## Session Persistence & Replay

```python
from calute.session import SessionManager, FileSessionStore

store = FileSessionStore(path="./sessions")
manager = SessionManager(store)

# Sessions are automatically recorded
# Replay later:
from calute.session import SessionReplay
replay = SessionReplay(store)
```

---

## Extensions

### Plugins

```python
from calute.extensions import PluginRegistry

registry = PluginRegistry()
registry.discover()  # Auto-discover plugins
```

### Skills

Skills are discovered from `SKILL.md` files with YAML frontmatter:

```python
from calute.extensions import SkillRegistry

skills = SkillRegistry()
skills.discover()
```

### Hooks

Lifecycle hooks for customizing agent behavior at 7 hook points.

---

## Execution Strategies

The Cortex framework supports five execution strategies:

| Strategy       | Description                                        |
| -------------- | -------------------------------------------------- |
| `SEQUENTIAL`   | Tasks run one after another, outputs chain forward |
| `PARALLEL`     | Independent tasks run concurrently                 |
| `HIERARCHICAL` | Manager agent delegates to worker agents           |
| `CONSENSUS`    | Multiple agents vote on outputs                    |
| `PLANNED`      | AI planner creates an optimal execution plan       |

---

## Context Management

Compaction strategies for managing long conversations:

- **Summarization** &mdash; LLM-based summarization of older messages
- **Sliding Window** &mdash; Retain only the N most recent messages
- **Priority-Based** &mdash; Keep messages scored by importance
- **Smart Hybrid** &mdash; Combines multiple strategies
- **Truncation** &mdash; Emergency context reduction

---

## Examples

See the [`examples/`](examples/) directory:

| Example                                      | Description                                        |
| -------------------------------------------- | -------------------------------------------------- |
| `scenario_1_conversational_assistant.py`     | Memory-enhanced conversational agent               |
| `scenario_2_code_analyzer.py`                | Code review and refactoring agent                  |
| `scenario_3_multi_agent_collaboration.py`    | Multi-agent task management system                 |
| `scenario_4_streaming_research_assistant.py` | Streaming research agent                           |
| `cortex_deepsearch_agent.py`                 | Parallel deep-search with configurable researchers |
| `cortex_parallel_benchmark.py`               | Cortex parallel execution benchmarking             |
| `interactive_agent.py`                       | Interactive agent loop                             |
| `textual_tui.py`                             | Terminal UI demonstration                          |

---

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=calute

# Lint
ruff check calute/

# Format
black calute/ tests/

# Type check
mypy calute/
```

---

## Requirements

- Python 3.10, 3.11, 3.12, or 3.13
- An API key for at least one supported LLM provider (OpenAI, Anthropic, Gemini, or a local Ollama instance)

---

## License

[Apache License 2.0](LICENSE)

---

## Author

**Erfan Zare Chavoshi** ([@erfanzar](https://github.com/erfanzar))

- GitHub: [github.com/erfanzar/Calute](https://github.com/erfanzar/Calute)
- Documentation: [erfanzar.github.io/Calute](https://erfanzar.github.io/Calute)
