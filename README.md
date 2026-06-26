# Xerxes

> A multi-agent orchestration framework for building, running, and serving LLM-powered agents.

```text
 ██╗  ██╗███████╗██████╗ ██╗  ██╗███████╗███████╗
 ╚██╗██╔╝██╔════╝██╔══██╗╚██╗██╔╝██╔════╝██╔════╝
  ╚███╔╝ █████╗  ██████╔╝ ╚███╔╝ █████╗  ███████╗
  ██╔██╗ ██╔══╝  ██╔══██╗ ██╔██╗ ██╔══╝  ╚════██║
 ██╔╝ ██╗███████╗██║  ██║██╔╝ ██╗███████╗███████║
 ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝
```

A coding agent that runs in your terminal. Pure-Python runtime with a `prompt_toolkit` TUI.

```text
› xerxes

╭────────────────────────────────────────────────────────╮
│ Xerxes (v0.3.0)                                       │
├────────────────────────────────────────────────────────┤
│ model:  claude-opus-4-7                               │
│ dir:    ~/Projects/myapp                               │
│ session: a1b2c3d4                                     │
╰────────────────────────────────────────────────────────╯

› explain this codebase

This is a Python web application using FastAPI...

✓ ReadFile README.md ✓
✓ exec_command find src -name "*.py" | head -20 ✓
```

## Install

Requires Python 3.11+. The installer uses [uv](https://docs.astral.sh/uv/), installs Xerxes into
`~/.xerxes-venv`, and downloads a managed Node.js runtime under `~/.xerxes/node` for the packaged
TypeScript TUI.

```bash
# One-line install (installs uv if missing, creates ~/.xerxes-venv, then adds a shell alias)
curl -fsSL https://raw.githubusercontent.com/erfanzar/Xerxes-Agents/main/scripts/install.sh | sh
```

```bash
# From source
git clone https://github.com/erfanzar/Xerxes-Agents.git
cd Xerxes-Agents
uv pip install -e ".[dev]"          # or: pip install -e ".[dev]"
```

```bash
# Verify
xerxes --help
```

If you installed from PyPI/editable mode without the shell installer and Node.js is not already on
your machine, run:

```bash
xerxes install --node
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

Profiles are saved in `~/.xerxes/profiles.json` (override with `XERXES_HOME`) and persist across sessions. You can have multiple profiles and switch between them with `/provider`.

### CLI Flags

```bash
# Use a specific provider directly (skips profile)
xerxes --model gpt-4o --base-url https://api.openai.com/v1 --api-key sk-...

# Non-interactive mode — pipe-friendly
xerxes -p "explain this function" 2>/dev/null

# Custom Python executable
xerxes --python python3.12

# Auto-approve all tool calls
xerxes --permission-mode accept-all

# Resume a session
xerxes -r a1b2c3d4
```

## Architecture

```text
┌─────────────────────────────────────────────────────┐
│                Xerxes (single Python process)       │
├─────────────────────┬───────────────────────────────┤
│   TUI layer        │   Runtime layer               │
│   (prompt_toolkit) │                               │
│                     │                               │
│ • Inline viewport   │ • Event-driven agent loop     │
│ • Markdown render   │ • Tool execution + sandbox    │
│ • Input handling    │ • LLM streaming               │
│ • Slash commands    │ • Provider registry            │
│ • Permission prompts│ • Profile management          │
│ • Skill registry    │ • YAML agent specs            │
└─────────────────────┴───────────────────────────────┘
```

Everything runs in one process — no JS, no subprocess bridge. The agent loop streams events that the TUI renders inline.

## Features Overview

| Capability | Details |
|-----------|---------|
| **LLM Providers** | OpenAI, Anthropic, Gemini, DeepSeek, Qwen, Kimi, Zhipu, MiniMax, Ollama, LM Studio |
| **Tools** | 150+ built-in tools across 18 modules |
| **Memory** | Short-term, long-term, entity, user memory with vector embeddings |
| **Skills** | 60+ skill bundles (markdown instruction sets) |
| **Multi-Agent** | Cortex orchestration with 5 process types |
| **Channels** | 14 chat platform adapters |
| **Security** | Sandboxed execution, policy engine, audit logging |
| **Interfaces** | CLI TUI, FastAPI server, WebSocket daemon, Python SDK |

## Slash Commands

| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/provider` | Setup or switch provider profile |
| `/model NAME` | Switch model |
| `/sampling` | View/set sampling params (temperature, top_p, etc.) |
| `/compact` | Summarize conversation using LLM to free context |
| `/plan` | Enter plan mode (read-only research before acting) |
| `/agents` | List / select YAML-defined sub-agents |
| `/skills` | List available skills |
| `/skill NAME` | Invoke a skill by name |
| `/skill-create` | Create a new skill from current session |
| `/tools` | List available tools |
| `/cost` | Show token usage and cost |
| `/context` | Show session info |
| `/clear` | Clear conversation |
| `/history` | Show / search conversation history |
| `/config` | Inspect or edit runtime config |
| `/permissions` | View/set permission mode |
| `/yolo` | Toggle accept-all permission mode |
| `/thinking` | Toggle thinking display |
| `/verbose` | Toggle verbose event logging |
| `/debug` | Toggle debug output |
| `/btw` | Inject side-channel context without breaking the turn |
| `/steer` | Course-correct the agent mid-stream |
| `/cancel` | Cancel the in-flight tool call |
| `/cancel-all` | Cancel all queued tool calls (double-Esc shortcut) |
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

Xerxes works with any OpenAI-compatible API. Built-in provider detection for:

| Provider | Models | Env Variable |
|----------|--------|--------------|
| OpenAI | gpt-4o, o3, o1 | `OPENAI_API_KEY` |
| Anthropic | claude-opus-4-7, claude-sonnet-4-6 | `ANTHROPIC_API_KEY` |
| Google | gemini-2.5-pro, gemini-2.0-flash | `GEMINI_API_KEY` |
| DeepSeek | deepseek-chat, deepseek-reasoner | `DEEPSEEK_API_KEY` |
| Qwen | qwen-max, qwq-32b | `DASHSCOPE_API_KEY` |
| Kimi | moonshot-v1-128k, kimi-latest | `MOONSHOT_API_KEY` |
| Zhipu | glm-4-plus, glm-4 | `ZHIPU_API_KEY` |
| MiniMax | minimax-text-01 | `MINIMAX_API_KEY` |
| Ollama | llama3, mistral, phi4 | (local, no key) |
| LM Studio | any loaded model | (local, no key) |
| Any OpenAI-compatible | custom | via `--base-url` |

## Tools

150+ built-in tools the agent can use, grouped by capability domain:

| Category | Tools |
|----------|-------|
| **File system** | ReadFile, WriteFile, AppendFile, FileEditTool, GlobTool, GrepTool, ListDir, TempFileManager |
| **Execution** | exec_command, write_stdin, list_terminal_sessions, close_terminal_session, ProcessManager |
| **Web** | DuckDuckGoSearch, GoogleSearch, WebScraper, APIClient, RSSReader, URLAnalyzer |
| **Browser** | Playwright-driven page navigation, DOM inspection, screenshotting |
| **Data** | JSONProcessor, CSVProcessor, TextProcessor, DateTimeProcessor |
| **Math** | Calculator, StatisticalAnalyzer, MathematicalFunctions, NumberTheory, UnitConverter |
| **AI/ML** | TextEmbedding, SimilaritySearch, Classifier, Summarizer, NERTagger |
| **Notebook** | Jupyter live-kernel cell exec, notebook read/edit |
| **LSP** | Language-server-driven definitions, references, diagnostics |
| **Agent** | SpawnAgents, TaskCreate, TaskList, TaskGet, SendMessage, AgentTool |
| **Workflow** | Plan/ExitPlan, TodoWrite, AskUserQuestion |
| **Memory** | save_memory, search_memory, delete_memory, consolidate_agent_memories |
| **MCP** | ListMcpResourcesTool, ReadMcpResourceTool |
| **Remote** | RemoteTrigger, PushNotification, webhook subscription |
| **Media** | image / audio / video helpers |
| **System** | Home Assistant, system info, env inspection |
| **Meta** | session_search, skill_view, skills_list, configure_mixture_of_agents |
| **RL** | rl_list_environments, rl_start_training, rl_check_status, rl_get_results |

Run `/tools` in the TUI for the full live list.

### Permission Modes

- **auto** (default) — read-only tools auto-approved, write/execute tools prompt for permission
- **accept-all** — approve everything (use with trusted models)
- **manual** — prompt for every tool call

## Skills

Skills are markdown instruction sets the agent loads into context when invoked via `/skill NAME`. Xerxes ships 60+ skill bundles covering software development, research, GitHub, productivity, ML/AI training, media, and more.

| Category | Skills |
|----------|--------|
| **Software Dev** | plan, test-driven-development, systematic-debugging, subagent-driven-development, requesting-code-review, writing-plans |
| **Research** | arxiv, blogwatcher, dspy, llm-wiki, polymarket, research-paper-writing, autoresearch (debug/fix/learn/plan/predict/scenario/security/ship) |
| **GitHub** | codebase-inspection, github-auth, github-code-review, github-issues, github-pr-workflow, github-repo-management |
| **ML / Training** | axolotl, grpo-rl-training, peft-fine-tuning, pytorch-fsdp, fine-tuning-with-trl, unsloth, evaluating-llms-harness, weights-and-biases, huggingface-hub, modal-serverless-gpu |
| **Inference** | vllm, sglang, tgi, llamacpp, mlx |
| **Productivity** | notion, google-workspace, linear, nano-pdf, ocr-and-documents, powerpoint |
| **Creative / Media** | ascii-art, ascii-video, excalidraw, architecture-diagram, manim-video, p5js, popular-web-designs, songwriting-and-ai-music |
| **Cloud** | aws, gcp, modal, runpod |
| **Other** | deepscan, obsidian, youtube-content, gif-search, jupyter-live-kernel, webhook-subscriptions, pokemon-player, minecraft-modpack-server |

Create your own with `/skill-create`.

## Agents

Sub-agents are defined as YAML specs with inheritance. The defaults live in [src/python/xerxes/agents/default/](src/python/xerxes/agents/default/):

```yaml
# coder.yaml
extend: agent.yaml          # inherit base spec
name: coder
description: Implements code changes against a codebase.
tools: [file_system, execution, lsp, workflow]
system_prompt_file: system.md
```

`agentspec.py` deep-merges the parent and child, so child specs override only the fields they need. Sub-agents are spawned via the `SpawnAgents` tool or selected interactively with `/agents`. Curated tool sets keep each sub-agent focused.

## Memory Architecture

Xerxes implements a four-tier memory system:

| Type | Purpose | Storage |
|------|---------|---------|
| **ShortTermMemory** | Bounded deque of recent interactions | In-memory |
| **LongTermMemory** | Persistent, searchable history | SQLite |
| **EntityMemory** | Graph-based entity relationships | SQLite |
| **UserMemory** | Per-user preferences and context | SQLite |

Features:
- SQLite-backed vector storage with hybrid retrieval (cosine + BM25 + recency)
- Multiple embedder options: SentenceTransformer, OpenAI, Ollama
- ContextualMemory with intelligent promotion threshold
- Automatic memory consolidation

## Cortex — Multi-Agent Orchestration

```python
from xerxes import Cortex, CortexAgent, CortexTask, ProcessType, create_llm

llm = create_llm("openai", api_key="sk-...")

researcher = CortexAgent(
    role="Researcher",
    goal="Find information",
    llm=llm
)
writer = CortexAgent(
    role="Writer",
    goal="Write reports",
    llm=llm
)

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

### Process Types

| ProcessType | Description |
|-------------|-------------|
| `SEQUENTIAL` | Tasks run one after another, context passed forward |
| `PARALLEL` | Tasks run concurrently with ThreadPoolExecutor |
| `HIERARCHICAL` | Manager agent delegates to worker agents with review |
| `CONSENSUS` | All agents contribute, lead synthesizes |
| `PLANNED` | LLM planner builds a DAG, executes in order |

`DynamicCortex` and `CortexPlanner` add runtime task generation and explicit execution planning.

## Channels — Multi-Platform Integration

Xerxes ships adapters for 14 chat platforms:

```
┌─────────┐ ┌──────────┐ ┌─────────┐ ┌────────┐ ┌─────────┐
│  Slack  │ │ Telegram │ │Discord  │ │ Email  │ │ Matrix  │
└─────────┘ └──────────┘ └─────────┘ └────────┘ └─────────┘
┌──────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│ WhatsApp │ │  Signal  │ │ Feishu  │ │ DingTalk│ │ WeCom   │
└──────────┘ └──────────┘ └────────┘ └────────┘ └──────────┘
┌────────────┐ ┌────────────┐ ┌────────────┐
│ Home Assistant│ │ Mattermost │ │ BlueBubbles │
└────────────┘ └────────────┘ └────────────┘
```

All channels use a unified `ChannelMessage` model with `IdentityResolver` for cross-platform identity management.

Discord Gateway mode uses `discord.py`:

```bash
uv run xerxes discord --token "$DISCORD_BOT_TOKEN"
```

By default, `xerxes discord` starts a managed background service and returns to the shell. Use the same routing flags
to check or stop that service:

```bash
uv run xerxes discord --channel-name macbook --status
uv run xerxes discord --channel-name macbook --stop
uv run xerxes discord --channel-name macbook --restart
uv run xerxes discord --token "$DISCORD_BOT_TOKEN" --channel-name macbook --foreground  # debug in this terminal
```

Service pid, socket, and logs live under `~/.xerxes/services/<service-name>/`. The default service name comes from
`--device-name`, `--channel-name`, `--address-name`, or channel id; override it with `--service-name`.

Enable the bot's Message Content Intent in the Discord developer portal. In guild channels, Xerxes responds to DMs
and bot mentions by default; use `--allowed-channel <id>` or `--always-reply` for broader channel behavior.
Invite the bot with both `bot` and `applications.commands` scopes to expose native Discord slash commands:
`/ask`, `/skills`, `/skill`, and `/status`. Disable command registration with `--no-discord-commands`.
For multiple Xerxes daemons in one Discord server, route each device to a named channel or thread and label replies:

```bash
uv run xerxes discord --token "$DISCORD_BOT_TOKEN" --channel-name m2-max --device-name m2-max
```

If several daemons share the exact same text channel, give each one an address name and prefix messages with it:

```bash
uv run xerxes discord --token "$DISCORD_BOT_TOKEN" --address-name m2-max --device-name m2-max
# In Discord: m2-max: status
```

## Security & Sandboxing

Xerxes implements defense-in-depth:

- **Sandbox isolation**: Docker or subprocess backends for untrusted tool execution
- **Pickle escape prevention**: JSON for child→parent IPC (never pickle.loads from child)
- **Policy engine**: Per-agent policy overrides and tool allowlisting
- **Audit logging**: JSONL and OpenTelemetry exporters for compliance
- **Prompt scanner**: Security scanning for skill files

## API Server

An OpenAI-compatible FastAPI server fronts both the standard agent loop and the Cortex orchestrator:

```python
from xerxes.api_server import XerxesAPIServer

server = XerxesAPIServer()
server.run(host="0.0.0.0", port=8000)
# POST /v1/chat/completions
# GET  /v1/models
```

## Python SDK

Xerxes's Python runtime can also be used as a library:

```python
from xerxes.streaming.events import AgentState, TextChunk, ToolStart, ToolEnd
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

### Event Types

```python
from xerxes.streaming.events import (
    TextChunk,        # LLM text output
    ThinkingChunk,    # Model reasoning (think tags)
    ToolStart,        # Tool execution begins
    ToolEnd,          # Tool execution completes
    PermissionRequest, # User approval needed
    TurnDone,         # Turn complete
    Error,            # Something went wrong
)
```

## Examples

See [examples/](examples/):

- `interactive_agent.py` — minimal interactive loop
- `textual_tui.py` — alternative Textual-based TUI
- `cortex_deepsearch_agent.py`, `cortex_parallel_benchmark.py` — Cortex orchestration
- `deepsearch_agent_demo.py`, `openclaw_capabilities_demo.py` — agent capability demos
- `scenario_1_conversational_assistant.py` … `scenario_4_streaming_research_assistant.py` — end-to-end scenarios

## Project Structure

```text
src/python/xerxes/
├── __main__.py              # CLI entry point
├── xerxes.py                # Core Xerxes class
├── tui/                     # prompt_toolkit TUI (app, engine, prompt, console, blocks)
├── bridge/                  # Provider profiles + JSON-RPC bridge server
├── streaming/               # Event-driven agent loop (events, loop, messages)
├── tools/                   # 150+ agent tools (18 modules)
├── llms/                    # LLM provider registry + implementations
├── cortex/                  # Multi-agent orchestration
│   ├── agents/              # CortexAgent, UniversalAgent
│   ├── core/                # Enums, templates, utilities
│   └── orchestration/       # Cortex, DynamicCortex, CortexPlanner
├── memory/                  # Memory backends (short/long/entity/user/contextual)
├── runtime/                 # Bootstrap, config, execution, session, profiles
├── context/                 # Token counting, compaction strategies
├── agents/                  # YAML agent specs + subagent manager
│   └── default/             # Built-in agent.yaml, coder.yaml, planner.yaml
├── extensions/              # Plugins, hooks, skills system
├── security/                # Sandbox, policies, prompt scanner
├── audit/                   # Audit events, collectors, OTEL exporter
├── session/                 # Session persistence (~/.xerxes/sessions)
├── api_server/              # OpenAI-compatible FastAPI server
├── daemon/                  # WebSocket + Unix socket daemon
├── channels/                # 14 chat platform adapters
│   └── adapters/            # Slack, Telegram, Discord, Email, Matrix, etc.
├── operators/               # Operator tools (browser, pty, plans, state)
├── mcp/                     # Model Context Protocol integration
├── logging/                 # Structured logging
└── types/                   # Type definitions
tests/                       # pytest suite (1501 tests)
```

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check src/python/xerxes/

# Format
black src/python/xerxes/ tests/
```

## Requirements

- Python 3.11+
- An LLM provider (cloud API key or local Ollama / LM Studio)

## License

[Apache License 2.0](LICENSE)

## Author

**Erfan Zare Chavoshi** ([@erfanzar](https://github.com/erfanzar))

## Citation

```bibtex
@software{xerxes2026,
  author = {Erfan Zare Chavoshi},
  title = {Xerxes: Multi-Agent Orchestration Framework},
  url = {https://github.com/erfanzar/Xerxes-Agents},
  year = {2026},
  license = {Apache-2.0}
}
```
