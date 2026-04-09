# Calute

A coding agent that runs in your terminal. Python runtime, Rust CLI.

```text
╭──────────────────────────────────────────────────────╮
│ >_ Calute (v0.2.0)                                  │
│                                                      │
│ model:     qwen3-8b (custom)   /model to change      │
│ directory: ~/Projects/myapp                          │
╰──────────────────────────────────────────────────────╯

› explain this codebase

• This is a Python web application using FastAPI...

• Read README.md ✓
  └ # MyApp - A REST API for...

• Ran find src -name "*.py" | head -20 ✓
  └ src/main.py
    src/routes/users.py
    src/models/user.py
    … +17 lines

• The project is structured as follows:
  - **src/main.py** — FastAPI application entry point
  - **src/routes/** — API route handlers
  ...
```

## Install

Requires Python 3.10+ and Rust (for the CLI binary).

```bash
# From source
git clone https://github.com/erfanzar/Calute.git
cd Calute
pip install -e .
```

The install automatically compiles the Rust CLI via `cargo build --release` and places the binary in your PATH. If you don't have Rust installed, get it from [rustup.rs](https://rustup.rs).

```bash
# Verify
calute --help
```

## Setup

On first launch, Calute asks you to configure a provider:

```bash
calute
```

```text
• No provider configured. Run /provider to set up a provider profile.

› /provider

Select a provider profile:
  › + New profile

Enter profile name:
  › my-server

Enter base URL (e.g. http://localhost:11434/v1):
  › http://localhost:11434/v1

Enter API key (or press Enter to skip):
  ›

Fetching available models...
Found 3 models. Select one (Up/Down + Enter):
  › llama3-8b
    mistral-7b
    qwen3-8b

Profile 'my-server' saved and activated. Model: llama3-8b
```

Profiles are saved in `~/.calute/profiles.json` and persist across sessions. You can have multiple profiles and switch between them with `/provider`.

### CLI flags

```bash
# Use a specific provider directly (skips profile)
calute --model gpt-4o --base-url https://api.openai.com/v1 --api-key sk-...

# Non-interactive mode — pipe-friendly
calute -p "explain this function" 2>/dev/null

# Custom Python executable
calute --python python3.12

# Auto-approve all tool calls
calute --permission-mode accept-all
```

## Architecture

```text
┌──────────────────┐     JSON-RPC       ┌──────────────────────┐
│   Rust CLI       │ ◄── stdin/stdout ──►│   Python Runtime     │
│   (ratatui)      │                     │   (calute.bridge)    │
│                  │                     │                      │
│ • Inline viewport│                     │ • Agent loop         │
│ • Markdown render│                     │ • Tool execution     │
│ • Input handling │                     │ • LLM streaming      │
│ • Slash commands │                     │ • Provider registry  │
│ • Provider setup │                     │ • Profile management │
└──────────────────┘                     └──────────────────────┘
```

The Rust binary (`calute`) spawns `python -m calute.bridge` as a subprocess. They communicate over newline-delimited JSON. The Rust side handles all rendering; the Python side handles all LLM and tool logic.

## Slash Commands

| Command | Description |
| ------- | ----------- |
| `/provider` | Setup or switch provider profile |
| `/sampling` | View/set sampling params (temperature, top_p, etc.) |
| `/compact` | Summarize conversation using LLM to free context |
| `/model NAME` | Switch model |
| `/cost` | Show token usage and cost |
| `/context` | Show session info |
| `/clear` | Clear conversation |
| `/tools` | List available tools |
| `/thinking` | Toggle thinking display |
| `/help` | Show all commands |
| `/exit` | Exit |

### Sampling

```text
› /sampling temperature 0.7
› /sampling top_p 0.9
› /sampling max_tokens 4096
› /sampling save          # persist to active profile
› /sampling reset         # reset to defaults
```

## Providers

Calute works with any OpenAI-compatible API. Built-in provider detection for:

| Provider | Models | Env Variable |
| -------- | ------ | ------------ |
| OpenAI | gpt-4o, o3, o1 | `OPENAI_API_KEY` |
| Anthropic | claude-opus-4-6, claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| Google | gemini-2.5-pro, gemini-2.0-flash | `GEMINI_API_KEY` |
| DeepSeek | deepseek-chat, deepseek-reasoner | `DEEPSEEK_API_KEY` |
| Qwen | qwen-max, qwq-32b | `DASHSCOPE_API_KEY` |
| Ollama | llama3, mistral, phi4 | (local, no key) |
| LM Studio | any loaded model | (local, no key) |
| Any OpenAI-compatible | custom | via `--base-url` |

## Tools

50+ built-in tools the agent can use:

| Category | Tools |
| -------- | ----- |
| **File** | Read, Write, Edit, Glob, Grep, ListDir, Append |
| **Shell** | Bash execution, Python execution, process management |
| **Web** | DuckDuckGo search, web scraping, API calls, RSS, URL analysis |
| **Data** | JSON, CSV, text processing, data conversion, datetime |
| **Math** | Calculator, statistics, number theory, unit conversion |
| **AI** | Text embedding, similarity, classification, summarization, NER |
| **Agent** | Spawn sub-agents, task management, plan mode, worktrees |
| **Memory** | Save/search/delete persistent memories |
| **MCP** | Model Context Protocol tool integration |

### Permission modes

- **auto** (default) — read-only tools auto-approved, write/execute tools prompt for permission
- **accept-all** — approve everything (use with trusted models)
- **manual** — prompt for every tool call

## Keyboard Shortcuts

| Key | Action |
| --- | ------ |
| Enter | Submit query |
| Up/Down | Input history |
| Ctrl+C | Cancel streaming / quit |
| Ctrl+W | Delete word |
| Ctrl+U | Clear line |
| Ctrl+A/E | Home / End |
| Esc | Cancel provider setup |
| Tab | Autocomplete slash command |
| y/n | Approve/deny permission |

## Python SDK

Calute's Python runtime can also be used as a library:

```python
from calute.streaming.events import AgentState
from calute.streaming.loop import run as run_agent_loop

state = AgentState()

for event in run_agent_loop(
    user_message="What files are in this directory?",
    state=state,
    config={"model": "gpt-4o", "api_key": "sk-..."},
    system_prompt="You are a helpful coding assistant.",
    tool_executor=my_tool_executor,
    tool_schemas=my_tool_schemas,
):
    match event:
        case TextChunk(text=t):
            print(t, end="")
        case ToolStart(name=n):
            print(f"\n[tool] {n}")
        case ToolEnd(name=n, result=r):
            print(f"[done] {r[:80]}")
```

### Cortex — Multi-Agent Orchestration

```python
from calute import Cortex, CortexAgent, CortexTask, ProcessType, create_llm

llm = create_llm("openai", api_key="sk-...")

researcher = CortexAgent(role="Researcher", goal="Find information", llm=llm)
writer = CortexAgent(role="Writer", goal="Write reports", llm=llm)

cortex = Cortex(
    agents=[researcher, writer],
    tasks=[
        CortexTask(description="Research AI agents", agent=researcher),
        CortexTask(description="Write a report", agent=writer),
    ],
    process=ProcessType.SEQUENTIAL,
)

result = cortex.kickoff()
```

### API Server

```python
from calute.api_server import CaluteAPIServer

server = CaluteAPIServer()
server.run(host="0.0.0.0", port=8000)
# POST /v1/chat/completions
# GET /v1/models
```

## Project Structure

```text
src/
├── python/calute/           # Python agent runtime
│   ├── bridge/              # JSON-RPC bridge + provider profiles
│   ├── streaming/           # Event-driven agent loop
│   ├── tools/               # 50+ agent tools
│   ├── llms/                # LLM provider registry
│   ├── runtime/             # Bootstrap, config, execution
│   ├── context/             # Token counting, compaction
│   ├── agents/              # Agent definitions
│   ├── cortex/              # Multi-agent orchestration
│   ├── memory/              # Memory backends
│   ├── security/            # Sandbox, policies
│   ├── session/             # Session persistence
│   ├── api_server/          # FastAPI server
│   └── _bin/                # Rust binary launcher
└── rust/calute-cli/         # Rust CLI frontend
    └── src/
        ├── main.rs          # Entry point, event loop
        ├── render.rs        # Inline viewport rendering
        ├── markdown.rs      # Markdown → terminal
        ├── app.rs           # State machine, cell model
        ├── input.rs         # Keyboard handling
        ├── bridge.rs        # Python subprocess IPC
        ├── events.rs        # Protocol types
        ├── theme.rs         # Colors and styles
        ├── spinner.rs       # Loading animation
        ├── slash.rs         # Command definitions
        └── diff.rs          # Diff rendering
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Build Rust CLI only
cd src/rust && cargo build --release

# Lint
ruff check src/python/calute/

# Format
black src/python/calute/ tests/
```

## Requirements

- Python 3.10+
- Rust toolchain (for building the CLI)
- An LLM provider (cloud API key or local Ollama/LM Studio)

## License

[Apache License 2.0](LICENSE)

## Author

**Erfan Zare Chavoshi** ([@erfanzar](https://github.com/erfanzar))
