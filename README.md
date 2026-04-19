# Xerxes

A coding agent that runs in your terminal. Python runtime, TypeScript/Ink CLI.

```text
 ██╗  ██╗███████╗██████╗ ██╗  ██╗███████╗███████╗  ┌────────────────────────────┐
 ╚██╗██╔╝██╔════╝██╔══██╗╚██╗██╔╝██╔════╝██╔════╝  │ >_ Xerxes (v0.2.0)         │
  ╚███╔╝ █████╗  ██████╔╝ ╚███╔╝ █████╗  ███████╗  ├────────────────────────────┤
  ██╔██╗ ██╔══╝  ██╔══██╗ ██╔██╗ ██╔══╝  ╚════██║  │ model:  qwen3-8b (custom)  │
 ██╔╝ ██╗███████╗██║  ██║██╔╝ ██╗███████╗███████║  │ dir:    ~/Projects/myapp   │
 ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝  └────────────────────────────┘


› explain this codebase

This is a Python web application using FastAPI...

✓ ReadFile README.md ✓
  └ # MyApp - A REST API for...

✓ ExecuteShell find src -name "*.py" | head -20 ✓
  └ src/main.py
    src/routes/users.py
    src/models/user.py
    … +17 lines
```

## Install

Requires Python 3.11+ and Node.js 20+ (for the CLI).

```bash
# One-line install (uv + bun auto-installed if missing)
curl -fsSL https://raw.githubusercontent.com/erfanzar/Xerxes-Agents/main/scripts/install.sh | sh
```

```bash
# From source
git clone https://github.com/erfanzar/Xerxes-Agents.git
cd Xerxes-Agents
pip install -e ".[dev]"
```

The install automatically bundles the TypeScript CLI via `bun build` and places the launcher in your PATH. If you don't have Bun installed, get it from [bun.sh](https://bun.sh).

```bash
# Verify
xerxes --help
```

## Setup

On first launch, Xerxes asks you to configure a provider:

```bash
xerxes
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

Profiles are saved in `~/.xerxes/profiles.json` and persist across sessions. You can have multiple profiles and switch between them with `/provider`.

### CLI flags

```bash
# Use a specific provider directly (skips profile)
xerxes --model gpt-4o --base-url https://api.openai.com/v1 --api-key sk-...

# Non-interactive mode — pipe-friendly
xerxes -p "explain this function" 2>/dev/null

# Custom Python executable
xerxes --python python3.12

# Auto-approve all tool calls
xerxes --permission-mode accept-all
```

## Architecture

```text
┌──────────────────┐     JSON-RPC        ┌──────────────────────┐
│  TypeScript CLI  │ ◄── stdin/stdout ──►│   Python Runtime     │
│   (Ink + React)  │                     │   (xerxes.bridge)    │
│                  │                     │                      │
│ • Inline viewport│                     │ • Agent loop         │
│ • Markdown render│                     │ • Tool execution     │
│ • Input handling │                     │ • LLM streaming      │
│ • Slash commands │                     │ • Provider registry  │
│ • Provider setup │                     │ • Profile management │
│ • Skill registry │                     │ • 98 tools           │
└──────────────────┘                     └──────────────────────┘
```

The TypeScript CLI (`xerxes`) spawns `python -m xerxes.bridge` as a subprocess. They communicate over newline-delimited JSON-RPC. The TypeScript side handles all rendering; the Python side handles all LLM and tool logic.

## Slash Commands

| Command         | Description                                         |
| --------------- | --------------------------------------------------- |
| `/provider`     | Setup or switch provider profile                    |
| `/sampling`     | View/set sampling params (temperature, top_p, etc.) |
| `/compact`      | Summarize conversation using LLM to free context    |
| `/model NAME`   | Switch model                                        |
| `/cost`         | Show token usage and cost                           |
| `/context`      | Show session info                                   |
| `/clear`        | Clear conversation                                  |
| `/tools`        | List available tools                                |
| `/skills`       | List available skills                               |
| `/skill-create` | Create a new skill from current session             |
| `/thinking`     | Toggle thinking display                             |
| `/yolo`         | Toggle accept-all permission mode                   |
| `/permissions`  | View/set permission mode                            |
| `/help`         | Show all commands                                   |
| `/exit`         | Exit                                                |

### Sampling

```text
› /sampling temperature 0.7
› /sampling top_p 0.9
› /sampling max_tokens 4096
› /sampling save          # persist to active profile
› /sampling reset         # reset to defaults
```

## Providers

Xerxes works with any OpenAI-compatible API. Built-in provider detection for:

| Provider              | Models                             | Env Variable        |
| --------------------- | ---------------------------------- | ------------------- |
| OpenAI                | gpt-4o, o3, o1                     | `OPENAI_API_KEY`    |
| Anthropic             | claude-opus-4-6, claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| Google                | gemini-2.5-pro, gemini-2.0-flash   | `GEMINI_API_KEY`    |
| DeepSeek              | deepseek-chat, deepseek-reasoner   | `DEEPSEEK_API_KEY`  |
| Qwen                  | qwen-max, qwq-32b                  | `DASHSCOPE_API_KEY` |
| MiniMax               | minimax-text-01                    | `MINIMAX_API_KEY`   |
| Ollama                | llama3, mistral, phi4              | (local, no key)     |
| LM Studio             | any loaded model                   | (local, no key)     |
| Any OpenAI-compatible | custom                             | via `--base-url`    |

## Tools

98 built-in tools the agent can use:

| Category     | Tools                                                                                                                                              |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **File**     | ReadFile, WriteFile, AppendFile, FileEditTool, GlobTool, GrepTool, ListDir, TempFileManager                                                        |
| **Shell**    | ExecuteShell, PythonExecution, ProcessManager                                                                                                      |
| **Web**      | DuckDuckGoSearch, GoogleSearch, WebScraper, APIClient, RSSReader, URLAnalyzer                                                                      |
| **Data**     | JSONProcessor, CSVProcessor, TextProcessor, DateTimeProcessor                                                                                      |
| **Math**     | Calculator, StatisticalAnalyzer, MathematicalFunctions, NumberTheory, UnitConverter                                                                |
| **AI/ML**    | TextEmbedding, SimilaritySearch, Classifier, Summarizer, NERTagger                                                                                 |
| **Agent**    | SpawnAgents, TaskCreateTool, TaskListTool, TaskGetTool, TaskOutputTool, SendMessageTool, AgentTool                                                 |
| **Memory**   | save_memory, search_memory, delete_memory, get_memory_statistics, consolidate_agent_memories                                                       |
| **Meta**     | configure_mixture_of_agents, session_search, skill_view, skills_list, skill_manage                                                                 |
| **MCP**      | ListMcpResourcesTool, ReadMcpResourceTool                                                                                                          |
| **Planning** | EnterPlanModeTool, ExitPlanModeTool, TodoWriteTool, AskUserQuestionTool                                                                            |
| **RL**       | rl_list_environments, rl_select_environment, rl_start_training, rl_stop_training, rl_check_status, rl_get_results, rl_list_runs, rl_test_inference |

### Permission modes

- **auto** (default) — read-only tools auto-approved, write/execute tools prompt for permission
- **accept-all** — approve everything (use with trusted models)
- **manual** — prompt for every tool call

## Skills

82+ built-in skills covering software development, research, GitHub, productivity, ML/AI training, media, and more. Skills are markdown instruction sets that the agent loads into context when invoked via `/skill-name`.

| Category         | Skills                                                                                                                                                                       |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Software Dev** | plan, test-driven-development, systematic-debugging, subagent-driven-development, requesting-code-review, writing-plans                                                      |
| **Research**     | arxiv, blogwatcher, dspy, llm-wiki, polymarket, research-paper-writing                                                                                                       |
| **GitHub**       | codebase-inspection, github-auth, github-code-review, github-issues, github-pr-workflow, github-repo-management                                                              |
| **ML/Training**  | axolotl, grpo-rl-training, peft-fine-tuning, pytorch-fsdp, fine-tuning-with-trl, unsloth, evaluating-llms-harness, weights-and-biases, huggingface-hub, modal-serverless-gpu |
| **Productivity** | notion, google-workspace, linear, nano-pdf, ocr-and-documents, powerpoint                                                                                                    |
| **Creative**     | ascii-art, ascii-video, excalidraw, architecture-diagram, manim-video, p5js, popular-web-designs, songwriting-and-ai-music                                                   |
| **Other**        | deepscan, obsidian, youtube-content, gif-search, jupyter-live-kernel, webhook-subscriptions, pokemon-player, minecraft-modpack-server                                        |

Create your own with `/skill-create`.

## Keyboard Shortcuts

| Key      | Action                     |
| -------- | -------------------------- |
| Enter    | Submit query               |
| Up/Down  | Input history              |
| Ctrl+C   | Cancel streaming / quit    |
| Ctrl+W   | Delete word                |
| Ctrl+U   | Clear line                 |
| Ctrl+A/E | Home / End                 |
| Esc      | Cancel provider setup      |
| Tab      | Autocomplete slash command |
| y/n      | Approve/deny permission    |

## Python SDK

Xerxes's Python runtime can also be used as a library:

```python
from xerxes.streaming.events import AgentState
from xerxes.streaming.loop import run as run_agent_loop

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
from xerxes import Cortex, CortexAgent, CortexTask, ProcessType, create_llm

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
from xerxes.api_server import XerxesAPIServer

server = XerxesAPIServer()
server.run(host="0.0.0.0", port=8000)
# POST /v1/chat/completions
# GET /v1/models
```

## Project Structure

```text
src/
├── python/xerxes/              # Python agent runtime
│   ├── bridge/                 # JSON-RPC bridge + provider profiles
│   ├── streaming/              # Event-driven agent loop
│   ├── tools/                  # 98 agent tools
│   ├── llms/                   # LLM provider registry
│   ├── runtime/                # Bootstrap, config, execution
│   ├── context/                # Token counting, compaction
│   ├── agents/                 # Agent definitions
│   ├── cortex/                 # Multi-agent orchestration
│   ├── memory/                 # Memory backends
│   ├── security/               # Sandbox, policies
│   ├── session/                # Session persistence
│   ├── api_server/             # FastAPI server
│   └── _bin/                   # CLI launcher + bundled JS
├── typescript/xerxes-cli/      # TypeScript/Ink CLI frontend
│   └── src/
│       ├── index.tsx           # Entry point
│       ├── App.tsx             # Main Ink app
│       ├── components/         # Cell renderers, Markdown, Banner
│       ├── state/              # Cell reducer, bridge events
│       └── bridge/             # Spawn, types
└── tests/                      # pytest suite
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Build TypeScript CLI bundle
cd src/typescript/xerxes-cli && bun build src/index.tsx --target=node --minify --outfile=../../python/xerxes/_bin/xerxes.mjs

# Lint
ruff check src/python/xerxes/

# Format
black src/python/xerxes/ tests/
```

## Requirements

- Python 3.11+
- Node.js 20+ (for CLI runtime)
- Bun (for building the CLI bundle)
- An LLM provider (cloud API key or local Ollama/LM Studio)

## License

[Apache License 2.0](LICENSE)

## Author

**Erfan Zare Chavoshi** ([@erfanzar](https://github.com/erfanzar))
