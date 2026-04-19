# System Architecture

## 30-second mental model

```mermaid
flowchart TB
    subgraph Interfaces["Interfaces"]
        CLI["Rust TUI<br/>(ratatui)"]
        API["FastAPI Server<br/>(/v1/chat/completions)"]
        Daemon["Daemon<br/>(WebSocket + Unix socket)"]
        Channels["14 Chat Channels<br/>(Slack / TG / Discord / …)"]
        Embed["Direct Python import"]
    end

    subgraph Bridges["Transport"]
        JSONRPC["bridge.server<br/>(JSON-RPC via stdio)"]
        SSE["SSE stream"]
        WS["WebSocket events"]
        ChannelMsg["ChannelMessage<br/>+ IdentityResolver"]
    end

    subgraph Core["Framework core"]
        Xerxes["Xerxes<br/>run / thread_run / athread_run"]
        Orchestrator["AgentOrchestrator<br/>+ FunctionRegistry"]
        Executor["FunctionExecutor"]
        StreamLoop["run_agent_loop<br/>(streaming.loop)"]
        StreamerBuf["StreamerBuffer<br/>(thread-safe queue)"]
    end

    subgraph Cortex["Cortex tier (optional)"]
        CortexOrch["Cortex"]
        Planner["CortexPlanner<br/>(XML DAG)"]
        TaskCreator["TaskCreator<br/>(LLM decomp)"]
    end

    subgraph Plugins["Extensibility"]
        LLMs["BaseLLM × 12<br/>(OpenAI / Anthropic / …)"]
        Tools["AgentBaseFn × 150<br/>(tools/)"]
        Memory["MemoryStore<br/>(short/long/entity/user)"]
        MCP["MCPManager<br/>(stdio/SSE/HTTP)"]
    end

    subgraph Runtime["RuntimeFeaturesConfig (opt-in)"]
        Features["Feature flags"]
        Policy["Policy engine"]
        Sandbox["SandboxRouter<br/>→ Docker / subprocess"]
        Audit["AuditEmitter<br/>→ JSONL / OTEL"]
        Session["SessionStore<br/>(InMemory / File)"]
        Profiles["PromptProfile<br/>(FULL / COMPACT / MINIMAL / NONE)"]
        Cost["CostTracker"]
    end

    CLI --> JSONRPC
    API --> SSE
    Daemon --> WS
    Channels --> ChannelMsg
    Embed --> Xerxes

    JSONRPC --> Xerxes
    SSE --> Xerxes
    WS --> Xerxes
    ChannelMsg --> Xerxes

    Xerxes --> Orchestrator
    Orchestrator --> Executor
    Xerxes --> StreamLoop
    StreamLoop --> StreamerBuf

    Xerxes -.optional.-> CortexOrch
    CortexOrch --> Planner
    CortexOrch --> TaskCreator
    CortexOrch --> Xerxes

    Executor --> Tools
    Executor --> MCP
    StreamLoop --> LLMs
    Xerxes --> Memory

    Xerxes -.enabled.-> Runtime
    Executor -.enabled.-> Sandbox
    Executor -.enabled.-> Policy
    Runtime --> Audit
    Runtime --> Session
    Runtime --> Profiles
    Runtime --> Cost
```

The picture: each interface (TUI, HTTP, daemon, channel, embed) hands a user prompt to the Xerxes core; the core runs a streaming loop through `BaseLLM`, executes tools via `FunctionRegistry`, consults `MemoryStore`, and — if `RuntimeFeaturesConfig.enabled` is true — routes every tool call through the sandbox + policy engine and fires an audit event before and after.

## Component relationships

```mermaid
classDiagram
    class Xerxes {
        +llm_client: BaseLLM
        +orchestrator: AgentOrchestrator
        +executor: FunctionExecutor
        +memory_store: MemoryStore?
        +runtime_features: RuntimeFeaturesConfig?
        +run(prompt, stream)
        +thread_run(prompt)
        +athread_run(prompt)
        +create_response(prompt) async
        +register_agent(agent)
    }

    class AgentOrchestrator {
        +agents: dict[str, Agent]
        +current_agent_id: str?
        +function_registry: FunctionRegistry
        +switch_triggers: dict[Trigger, Handler]
        +register_agent(agent)
        +switch(trigger, reason)
    }

    class Agent {
        +id: str
        +name: str
        +instructions: str
        +model: str
        +functions: list~AgentFunction~
        +capabilities: list~AgentCapability~
        +fallback_agent_id: str?
        +sampling params
    }

    class BaseLLM {
        <<abstract>>
        +generate_completion(...)
        +stream_completion(response, agent)
        +astream_completion(response, agent)
        +parse_tool_calls(raw)
        +extract_content(response)
    }

    class FunctionExecutor {
        +execute(call, agent, state)
        +check_permission(...)
        +apply_policy(...)
        +route_to_sandbox(...)
    }

    class Cortex {
        +agents: list~CortexAgent~
        +tasks: list~CortexTask~
        +process: ProcessType
        +llm: BaseLLM
        +cortex_memory: CortexMemory?
        +kickoff(inputs, use_streaming)
    }

    class CortexAgent {
        +role: str
        +goal: str
        +backstory: str
        +_internal_agent: XerxesAgent
        +xerxes_instance: Xerxes
        +cortex_instance: Cortex
    }

    Xerxes --> AgentOrchestrator
    Xerxes --> BaseLLM
    Xerxes --> FunctionExecutor
    AgentOrchestrator --> Agent
    Cortex --> CortexAgent
    CortexAgent --> Xerxes
    FunctionExecutor ..> "optional" SandboxRouter
```

## Basic execution loop

When a caller invokes `xerxes.run("write a haiku about tensors")`:

```mermaid
sequenceDiagram
    participant Caller
    participant Xerxes
    participant Orchestrator as AgentOrchestrator
    participant Loop as run_agent_loop
    participant LLM as BaseLLM
    participant Executor as FunctionExecutor
    participant Sandbox
    participant Memory
    participant Audit as AuditEmitter

    Caller->>Xerxes: run(prompt)
    Xerxes->>Orchestrator: pick current agent
    Xerxes->>Memory: recall relevant context (opt)
    Xerxes->>Loop: run(messages, agent, llm)

    loop Until no more tool calls
        Loop->>LLM: astream_completion(messages)
        LLM-->>Loop: TextChunk / ToolStart / …
        alt contains tool call
            Loop->>Executor: execute(call, agent)
            Executor->>Audit: emit_tool_call_attempt
            alt RuntimeFeaturesConfig.enabled
                Executor->>Sandbox: decide(tool, agent)
                Sandbox-->>Executor: HOST or SANDBOX
            end
            Executor-->>Loop: ToolEnd(result)
            Executor->>Audit: emit_tool_call_complete
            Loop->>Memory: persist turn (opt)
        else no tool call
            Loop-->>Xerxes: TurnDone
        end
    end

    Xerxes-->>Caller: stream of events
```

Key points:
- The loop is **streaming by default.** Non-streaming = collect events + assemble.
- `FunctionExecutor` is the chokepoint — every tool call passes through it, so audit, policy, sandbox, loop-detection, and retries all live there.
- Memory writes are best-effort and never break the loop (exceptions are swallowed with a log).

## Cortex execution

`Cortex.kickoff()` dispatches on `ProcessType`:

```mermaid
flowchart LR
    Kickoff["cortex.kickoff(inputs)"]
    Kickoff --> PT{ProcessType?}
    PT -->|SEQUENTIAL| S["_run_sequential<br/>task ← task ← task<br/>context passed forward"]
    PT -->|PARALLEL| P["_run_parallel<br/>ThreadPoolExecutor<br/>independent tasks"]
    PT -->|HIERARCHICAL| H["_run_hierarchical<br/>manager delegates<br/>+ reviews"]
    PT -->|CONSENSUS| C["_run_consensus<br/>all agents contribute<br/>lead synthesizes"]
    PT -->|PLANNED| Pl["_run_planned<br/>Planner → ExecutionPlan<br/>steps in DAG order"]

    S --> Out["CortexOutput"]
    P --> Out
    H --> Out
    C --> Out
    Pl --> Out
```

For every process type, individual task execution goes through an internal `Xerxes` instance — i.e. Cortex is a layer on top of the basic loop, not a replacement.

## Streaming data flow

```mermaid
flowchart LR
    LLMResp["LLM response<br/>(async iter)"] --> Converter["provider converter<br/>(anthropic → neutral,<br/>openai → neutral)"]
    Converter --> Events["StreamEvent sequence<br/>TextChunk / ThinkingChunk<br/>ToolStart / ToolEnd<br/>TurnDone"]
    Events --> Buffer["StreamerBuffer<br/>(thread-safe queue)"]
    Buffer --> Sync["Sync caller<br/>(for chunk in buffer)"]
    Buffer --> Async["Async caller<br/>(async for chunk in buffer)"]
    Buffer --> SSE["FastAPI SSE"]
    Buffer --> JSONRPC["Bridge events"]
    Buffer --> WS["Daemon WebSocket"]
```

The same event stream drives four different transports — the only difference is the encoding on the wire.

## Rust TUI ↔ Python bridge

```mermaid
sequenceDiagram
    participant User
    participant TUI as Rust TUI
    participant Bridge as bridge.server
    participant Xerxes
    participant LLM

    User->>TUI: keystroke
    TUI->>Bridge: {"method":"init","params":{...}}
    Bridge->>Xerxes: instantiate
    Bridge-->>TUI: {"event":"ready"}
    User->>TUI: enter prompt
    TUI->>Bridge: {"method":"query","params":{"text":"…"}}
    Bridge->>Xerxes: run_agent_loop
    Xerxes->>LLM: astream_completion

    loop stream events
        LLM-->>Xerxes: chunk
        Xerxes-->>Bridge: StreamEvent
        Bridge-->>TUI: {"event":"text_chunk","data":…}
    end

    alt tool needs permission
        Bridge-->>TUI: {"event":"permission_request"}
        TUI->>User: show approval dialog
        User->>TUI: approve / deny
        TUI->>Bridge: {"method":"permission_response"}
        Bridge->>Xerxes: resume
    end

    Xerxes-->>Bridge: TurnDone
    Bridge-->>TUI: {"event":"turn_done","data":{tokens}}
```

Protocol: newline-delimited JSON over the Python subprocess's stdin/stdout. Trivially observable (`pip install xerxes-agent && python -m xerxes.bridge | tee bridge.jsonl`).

## Channel adapter architecture

```mermaid
flowchart LR
    subgraph Platforms["External platforms"]
        Slack
        Telegram
        Discord
        Email
        Others["(11 more…)"]
    end

    subgraph Adapters["channels/adapters/"]
        SlackAd["SlackChannel"]
        TgAd["TelegramChannel"]
        DisAd["DiscordChannel"]
        EmailAd["EmailChannel"]
        OtherAd["…"]
    end

    subgraph Unified["channels/"]
        Base["Channel ABC<br/>start / send / stop"]
        Msg["ChannelMessage"]
        Identity["IdentityResolver<br/>(channel_user_id → user_id)"]
    end

    subgraph Agent["Agent"]
        Xerxes
    end

    Slack --> SlackAd
    Telegram --> TgAd
    Discord --> DisAd
    Email --> EmailAd
    Others --> OtherAd

    SlackAd --> Msg
    TgAd --> Msg
    DisAd --> Msg
    EmailAd --> Msg
    OtherAd --> Msg

    Msg --> Identity
    Identity --> Xerxes
    Xerxes -->|response| Msg
    Msg -->|outbound| SlackAd
    Msg -->|outbound| TgAd
```

Every platform-specific payload is normalized to `ChannelMessage` (text, channel, channel_user_id, room_id, attachments, metadata). `IdentityResolver` maps `(channel, channel_user_id)` to a stable `user_id` — so the same human talking to the agent via Slack in the morning and Telegram in the evening has one persistent identity.

## Service dependency graph

```mermaid
flowchart TD
    Config["XerxesConfig<br/>(core.config)"]
    Types["types<br/>(Agent, Message, Tool)"]

    LLMs["llms<br/>BaseLLM + registry"]
    Memory["memory<br/>MemoryStore + Storage"]
    Streaming["streaming<br/>events + loop"]

    Executors["executors<br/>FunctionRegistry + Orchestrator"]
    Xerxes["xerxes<br/>Xerxes class"]
    Cortex["cortex<br/>Cortex + tasks"]

    Runtime["runtime<br/>features + bridge + bootstrap"]
    Security["security<br/>sandbox"]
    Audit["audit<br/>events + emitter"]
    Extensions["extensions<br/>plugins + hooks"]
    Session["session<br/>store + replay"]
    Cost["runtime.cost_tracker"]

    APIServer["api_server<br/>FastAPI"]
    Daemon["daemon<br/>WS + socket"]
    Bridge["bridge<br/>JSON-RPC"]
    Channels["channels<br/>14 adapters"]
    RustCLI["Rust CLI<br/>ratatui"]

    Types --> LLMs
    Types --> Memory
    Types --> Streaming
    Types --> Executors
    Config --> Xerxes

    LLMs --> Xerxes
    Memory --> Xerxes
    Streaming --> Xerxes
    Executors --> Xerxes
    Xerxes --> Cortex

    Security --> Executors
    Audit --> Runtime
    Extensions --> Runtime
    Session --> Runtime
    Cost --> Runtime
    Runtime --> Xerxes

    Xerxes --> APIServer
    Xerxes --> Daemon
    Xerxes --> Bridge
    Xerxes --> Channels
    Bridge --> RustCLI
```

Arrows show compile-time Python import dependency (A → B means B imports A). Leaves at the top: `types` and `core.config` are the base layer with no internal imports.

## Runtime feature layering

When `RuntimeFeaturesConfig.enabled = True`, `RuntimeFeaturesState` is instantiated once and wired into every tool call and turn boundary:

```mermaid
flowchart LR
    subgraph State["RuntimeFeaturesState"]
        Plugins["PluginRegistry"]
        Skills["SkillRegistry"]
        Hooks["HookRunner"]
        Policy["PolicyEngine"]
        PromptCtx["PromptContextBuilder"]
        SandboxR["SandboxRouter"]
        AuditE["AuditEmitter"]
        SessionM["SessionManager"]
        OpState["OperatorState"]
    end

    PreTurn["on_turn_start"] --> Hooks
    ToolAttempt["tool call attempted"] --> Policy
    Policy -->|allow| SandboxR
    SandboxR -->|decide| Exec["FunctionExecutor"]
    Exec --> Hooks
    Hooks -.mutation.-> Args["modified args"]
    Exec --> ToolResult
    ToolResult --> Hooks
    Hooks -.mutation.-> PersistedResult["sanitized result"]
    AuditE -.observe every step.-> Sink["InMemory / JSONL / OTEL"]
    SessionM --> Store["SessionStore (File / InMemory)"]
    PromptCtx --> SystemPrompt["built prompt prefix"]
```

All sinks are optional; a missing backend degrades gracefully (structlog / OpenTelemetry / Prometheus are all soft-imported).

## Sandbox isolation boundary

```mermaid
flowchart LR
    subgraph Parent["Parent (trusted)"]
        PCode["Xerxes runtime<br/>FunctionExecutor"]
        PPickle["pickle.dumps<br/>(func, args)"]
    end

    subgraph Boundary["IPC boundary (untrusted child)"]
        B64In["base64 → stdin"]
        B64Out["stdout ← base64"]
    end

    subgraph Child["Child sandbox (docker / subprocess)"]
        CUnpickle["pickle.loads<br/>(trusted parent bytes)"]
        CRun["func(**args)"]
        CJSON["json.dumps(result)"]
    end

    PCode --> PPickle --> B64In --> CUnpickle --> CRun --> CJSON --> B64Out
    B64Out --> PJSON["json.loads<br/>(never pickle.loads)"]
    PJSON --> PCode
```

**Key invariant:** parent→child uses pickle (parent is trusted), child→parent uses JSON (child could be adversarial). The earlier pickle-symmetric design was an arbitrary-code-execution vector via `__reduce__`; it's now closed. See [security fix commit](../debug/260418-1007-fullhunt/findings.md) for the underlying PoC.

## Memory architecture

```mermaid
flowchart TB
    subgraph Types["4 memory types (memory/)"]
        Short["ShortTermMemory<br/>(bounded deque)"]
        Long["LongTermMemory<br/>(persistent)"]
        Entity["EntityMemory<br/>(graph)"]
        User["UserMemory<br/>(per-user)"]
    end

    subgraph Contextual["ContextualMemory"]
        CShort["short-term (20 items)"]
        CLong["long-term (5k items)"]
        Promotion["promotion_threshold<br/>(access count)"]
    end

    subgraph Storage["MemoryStorage protocol"]
        Simple["SimpleStorage<br/>(in-memory dict)"]
        File["FileStorage<br/>(pickle files)"]
        SQLite["SQLiteStorage<br/>(default)"]
        RAG["RAGStorage<br/>(+ semantic)"]
    end

    subgraph Vector["vector_storage.py"]
        VStorage["SQLiteVectorStorage<br/>(JSON embeddings, no pickle)"]
        Embed["Embedder protocol<br/>HashEmbedder / SentenceTransformer /<br/>OpenAIEmbedder / OllamaEmbedder"]
    end

    subgraph Retrieval["retrieval.py"]
        Hybrid["HybridRetriever<br/>cosine + bm25 + recency"]
    end

    Short --> Contextual
    Long --> Contextual
    CShort --> Promotion --> CLong
    Contextual --> Storage
    SQLite --> RAG
    RAG --> Vector
    Vector --> Embed
    Vector --> Hybrid
```

## Runtime profiles

`PromptProfile` controls how much context is injected into the system prompt of each agent:

| Profile | Use case | Includes |
|---------|----------|----------|
| `FULL` (default) | Top-level user-facing agent | Everything: runtime info, workspace, sandbox status, skills, tools, guardrails, bootstrap, memories, user profile |
| `COMPACT` | Sub-agent delegation | Trimmed bootstrap, skill instructions capped at 500 chars, tool list capped at 20 |
| `MINIMAL` | Internal tool agent | Only sandbox, guardrails, 10-entry tool list |
| `NONE` | OpenClaw-style control | Identity only; caller supplies all context explicitly |

## Cross-references

- Per-package file listing: [codebase-summary.md](codebase-summary.md).
- How to run the HTTP server and full endpoint spec: [api-reference.md](api-reference.md).
- Env vars, `XerxesConfig`, runtime flags: [configuration-guide.md](configuration-guide.md).
- How to run and extend tests: [testing-guide.md](testing-guide.md).
