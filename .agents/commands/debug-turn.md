---
name: debug-turn
description: Systematic diagnostic checklist for a failing turn, compaction bug, prompt-cache miss, permission denial, or stall from CLI and Daemon through streaming/loop.ts to UI adapters.
tags: [debugging, turn-loop, daemon, compaction, prompt-cache]
---

# Debug a Turn Failure, Stall, Compaction Bug, or Prompt-Cache Miss

Diagnose: `$ARGUMENTS`. Classify the symptom first: stall/timeout, compaction/overflow/pairing error, prompt-cache miss, permission/hook/sandbox/data-boundary denial, transcript loss, or UI event desync.

## 1. Stage-by-Stage Pipeline Checklist

### Stage 1: CLI & Daemon Admission (`cli.ts`, `daemon/{server,runtime,workspaceResources}.ts`)
- **CLI**: Verify `--agent` is not passed on `NON_ONESHOT_COMMANDS` and dash-led args precede `--`.
- **RPC queueing**: Check if a live-turn RPC needs `QUEUE_BYPASS_METHODS`, serialized on `SessionOperationQueue`, or deduplicated (`MAX_ACCEPTED_SUBMISSION_IDS = 4_096`).
- **Lifecycle**: Check `inFlightTurnRunners` across `reload()`, LRU eviction (`DEFAULT_MAX_LOADED_SESSIONS = 12`), `DaemonWorkspaces.release()` (`WORKSPACE_RELEASE_DELAY_MS = 60_000`), and Antigravity `ANTIGRAVITY_COMPOSED_TURN_REFUSAL` (`userAuthored !== true`).

### Stage 2: Turn Assembly & Prompt-Cache Stability (`daemon/turnRunner.ts`, `context/{assembly,promptBudget}.ts`)
- **Cache misses**: Check `AssemblyProvenance` digests and `SessionPromptSnapshots` across rebuilds. Confirm **zero** volatile per-turn data (memory diffs, live goal status, date changes, instruction diffs, approved lessons, context deltas) entered `assembleContextLayers()`; volatile state belongs exclusively in `assembleTurnContext()` (`<turn-context>`).
- **Budgets & lean**: Check `MAX_BOOTSTRAP_INSTRUCTION_FILE_BYTES` (`16 KiB/file`, `32 KiB` total), `.agents/`/`.xerxes/` (`96 KiB` total, `6,000 B/file`), `DEFAULT_PROMPT_BUDGETS` (`24_000` est. tokens), and auto lean profile (`contextLimit <= 32_768` or Bonsai clipping project instructions to `1,500` chars).
- **Routing & gates**: Check `PolicyRouter` (`no_allowed_route`), `outboundGate.ts` (`DataBoundaryError`), `localOnlyActive()`, and `getApiKey` ambient-key withholding on non-vendor `base_url` (unless `XERXES_ALLOW_AMBIENT_KEY=1`).

### Stage 3: Streaming Loop & Recovery (`streaming/loop.ts`, `llms/client.ts`)
- **Watchdogs & failover**: Check connect timeout (`15s`), `firstChunkTimeoutMs`, `DEFAULT_STREAM_INACTIVITY_TIMEOUT_MS` (`120_000`), `retryPolicyForModel`, `refreshLlm`, and local `recoverEngine`.
- **Loop guards**: `MAX_UNCONFIGURED_ONLY_ROUNDS = 3`, `DEFAULT_MAX_IDENTICAL_FAILED_TOOL_ROUNDS = 4` (nudge at 2), `MAX_EMPTY_REPLY_NUDGES = 2`, `MAX_REPLY_BUDGET_STOPS = 2`, `MAX_OUTPUT_LIMIT_ESCALATIONS = 3` (`OUTPUT_LIMIT_RETRY_MAX_TOKENS = 64_000`). Confirm `withoutXerxesAnnotations` stripped internal annotations before provider serialization.

### Stage 4: Compaction & Tool Pairing (`context/{compressor,compactionThreshold,windowUsage,toolPairRepair,sweWorkingSet}.ts`)
- Check `DEFAULT_AUTO_COMPACT_THRESHOLD` (`0.6`; `0` = off; `>= 1024` = absolute tokens) and `windowUsage.ts` calibration.
- Verify pre-pruning (`toolResultPruner.ts`), `protectFirst: 3`, `protectLast: 6`, `<swe-working-set>` (cap `1,600` chars), `xerxes_compaction_summary: true`, and `repairToolMessageSequence`. On cancel/abort, verify `streaming/loop.ts` synthesized interrupted tool results so provider call/result pairing stays valid.

### Stage 5: Tools, Permissions, Hooks & Sandbox (`streaming/permissions.ts`, `executors/toolRegistry.ts`, `extensions/hooks.ts`, `security/`)
- Trace `permissionDisposition`: Static `ToolPolicy` (`deny` > `allow`) -> `DataBoundary` (`deny` > `prompt`) -> `ALWAYS_APPROVAL_TOOLS` -> mode (`accept-all`/`manual`/`auto`/`plan`; `stdin` disqualifies read-only shell).
- Check fail-closed `tool_permission_check` (`PreToolUse`) hooks (throw/timeout/exit-2 denies), `groupToolDecisions()` (`MAX_CONCURRENT_TOOL_CALLS = 8` consecutive `concurrencySafe`), `normalizeToolOutput` (`neutralizeSystemReminders`, empty marker), `ToolResultStorage` (`<persisted-output>`), `CommandSandbox` (`sandbox-exec`/`bwrap` probe), `ToolSearchTool` deferral, and `xerxes/src/agents/default/agent.yaml`.

### Stage 6: Persistence & UI Adapters (`session/{daemonTranscript,resumeRepair}.ts`, `ui/gatewayAdapter.ts`, `desktop/main/ipcEventBatcher.ts`)
- Check `transcriptHasHistory` (skips user-only dead turns), generation CAS + `.jsonl` crash journal, `repairResumedTranscript`, and post-persistence `afterTurn` (reflection 90s timeout, auto-title, goal continuation).
- Check `daemonEventFromStream()` wire mapping (`text_part`, `think_part`, `tool_call`, `tool_result`, `turn_end`), `SNAPSHOT_COVERED_EVENTS`, and `DaemonEventBatcher` FIFO flush before lifecycle events.

## 2. Targeted Diagnostic Tests

```bash
bun test xerxes/test/streamingLoopParity.test.ts xerxes/test/agentTurnRunner.test.ts xerxes/test/contextAssembly.test.ts xerxes/test/contextCompressor.test.ts xerxes/test/sweContextCompaction.test.ts xerxes/test/compactionCalibratedBudget.test.ts xerxes/test/streamingPermissions.test.ts xerxes/test/daemonCrashJournal.test.ts
```
Reproduce offline with an injected `LlmClient` before editing, then re-run targeted tests + `bun run check`.
