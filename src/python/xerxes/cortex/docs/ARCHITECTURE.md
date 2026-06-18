# Cortex Architecture: Boundary & Direction Decision

## TL;DR

**Cortex is a distinct, higher-level multi-agent workflow orchestration layer.**
It is NOT a duplicate of `agents/` — the two serve fundamentally different
purposes and should remain separate. However, the current coupling is backwards
(cortex defines its own `CortexAgent` instead of composing the core `Agent`),
and the boundary contract should be formalized.

---

## Current State

### What cortex IS

Cortex is a **CrewAI-style workflow engine** for multi-agent task chains.
It provides execution strategies that `agents/` + `streaming/` do not:

| Strategy | What it does |
|---|---|
| `SEQUENTIAL` (`cortex.py:605`) | Tasks run one-after-another, output chains forward |
| `HIERARCHICAL` (`cortex.py:831`) | Manager agent delegates and reviews sub-tasks |
| `PARALLEL` (`cortex.py:744`) | Independent tasks execute concurrently via threading |
| `CONSENSUS` (`cortex.py:988`) | Multiple agents produce outputs that are aggregated |
| `PLANNED` (`cortex.py:1070`) | `CortexPlanner` generates an `ExecutionPlan` of `PlanStep`s |

These are workflow-level abstractions (like LangGraph graphs or CrewAI crews)
that the streaming loop (`streaming/loop.py`) does not provide — the loop runs
a single agent turn with tools, not a multi-task DAG.

### What cortex is NOT

Cortex does **not** use:
- The streaming loop (`streaming/loop.py`)
- The daemon/server (`daemon/`)
- The TUI (`tui/`)
- The subagent manager (`agents/subagent_manager.py`)
- The permission system (`streaming/permissions.py`)
- The tool executor (`executors.py:EnhancedFunctionExecutor`)

### Layering (imports)

Cortex imports from the **core primitives** layer:
- `xerxes.llms.base.BaseLLM` (`agent.py:26`) — calls the LLM directly
- `xerxes.core.streamer_buffer.StreamerBuffer` (`agent.py:25`)
- `xerxes.types.Agent` / `AgentCapability` (`agent.py:32-33`)
- `xerxes.core.utils.function_to_json` (`core/tool.py:20`)

Cortex does NOT import from `agents/`, `runtime/`, `streaming/`, or `operators/`.
The one exception is `agent.py:30` which imports `AutoCompactAgent` from
`agents/auto_compact_agent.py` — this is a compaction helper, not agent spawning.

The core package imports cortex in `xerxes/__init__.py:26` to re-export
`Cortex`, `CortexAgent`, etc. as top-level convenience symbols.

---

## Duplication Inventory

| Concept | cortex/ | agents/ + runtime/ | Duplicate? |
|---|---|---|---|
| Agent class | `CortexAgent` (1420 lines) | `types.Agent` (dataclass) | **Partial** — CortexAgent wraps BaseLLM directly instead of using the streaming loop |
| Task representation | `CortexTask` (924 lines) | No equivalent | **No** — cortex tasks are workflow steps, not tool calls |
| Agent spawning | `delegate_task()` (`agent.py:1171`) | `SubAgentManager` (1103 lines) | **Different** — cortex delegation is within-workflow; SubAgentManager is subprocess-level |
| Orchestration | `Cortex` (1450 lines) — 5 strategies | `EnhancedAgentOrchestrator` — agent switch triggers | **Different** — cortex orchestrates task DAGs; orchestrator routes between agents |
| Memory | `CortexMemory` | `memory/MemoryStore` (4-tier) | **Partial** — CortexMemory is simpler context-scoped store |
| LLM calling | `CortexAgent.execute()` calls `BaseLLM` directly | `streaming/loop.py:run()` streams via provider adapters | **Duplicate** — cortex bypasses the streaming loop entirely |
| Tool execution | `CortexTool` + inline callbacks | `executors.py:EnhancedFunctionExecutor` | **Partial** — cortex has its own tool dispatch |

### Key Finding

The real duplication is NOT between cortex and agents/ — it's that
**`CortexAgent.execute()` reimplements LLM calling, tool dispatch, and streaming
instead of composing the existing `streaming/loop.py:run()` function.**

`CortexAgent.execute()` (`agent.py:678-956`) is a 280-line method that:
1. Builds messages
2. Calls `self.llm.generate_completion()` directly
3. Parses tool calls
4. Executes tools via its own dispatch
5. Feeds results back

This is functionally equivalent to what `streaming/loop.py:run()` does, but:
- No permission gating
- No thinking-tag parsing
- No prompt caching
- No budget enforcement
- No cancellation support
- No audit events

---

## Recommendation: KEEP SEPARATE, but fix the coupling

### Decision: cortex/ stays as a distinct layer

**Rationale:** Cortex provides workflow orchestration (sequential, parallel,
hierarchical, consensus, planned task chains) that is genuinely different from
the streaming loop's single-turn-with-tools model. Deleting cortex would lose:
- Multi-agent task DAGs (examples/cortex_deepsearch_agent.py)
- Parallel agent execution with result aggregation
- Hierarchical manager-worker delegation
- Consensus voting across agents
- Plan-then-execute workflows

These are valuable features for research, benchmarking, and complex task
automation that the streaming loop cannot replace.

### Required Fix: CortexAgent should compose the streaming loop

`CortexAgent.execute()` should delegate single-agent LLM+tool turns to
`streaming/loop.py:run()` instead of reimplementing them. This gives cortex:
- Permission gating (PLAN/AUTO/MANUAL modes)
- Thinking-tag parsing
- Prompt caching (Anthropic)
- Token budget enforcement
- Cancellation support
- Audit events

**Migration path:**
1. `CortexAgent.execute()` calls `streaming.loop.run()` with the agent's
   model, system prompt, tools, and message history
2. Cortex orchestration strategies (`_run_sequential`, `_run_parallel`, etc.)
   call `CortexAgent.execute()` for each task step — unchanged
3. Tool execution flows through the existing `EnhancedFunctionExecutor`
   instead of cortex's inline dispatch

This is a non-breaking refactor: cortex's public API (`Cortex.kickoff()`,
`CortexAgent.execute()`, `CortexTask`) stays the same; only the internal
LLM-calling mechanism changes.

### Contract

```
Layer 4 (Workflow):    cortex/ — task DAGs, parallel/hierarchical/consensus
                         ↓ composes
Layer 3 (Agent Loop):  streaming/loop.py — single-turn streaming + tools
                         ↓ uses
Layer 2 (Primitives):  llms/, tools/, memory/, types/
                         ↑ does NOT use
Layer 1 (Runtime):     runtime/, daemon/, tui/ — session, permissions, audit
```

- `cortex/` composes `streaming/loop.py` for each agent step
- `cortex/` does NOT import from `runtime/`, `daemon/`, `tui/`
- `cortex/` imports `llms/`, `types/`, `core/` primitives directly
- `agents/` provides agent definitions and SubAgentManager for the runtime layer
- `streaming/` is the single agent-turn execution engine used by both cortex and the daemon

---

## Migration Steps

1. **Phase A (non-breaking):** Add `streaming/loop.py:run()` as the execution
   backend inside `CortexAgent.execute()`, behind a feature flag
2. **Phase B (non-breaking):** Route cortex tool dispatch through
   `executors.py:EnhancedFunctionExecutor`
3. **Phase C (breaking, minor):** Remove cortex's inline LLM-calling code
   once the streaming-loop backend is verified
4. **Phase D:** Add cortex workflow examples that demonstrate permission modes
   and budget enforcement flowing through the streaming loop
