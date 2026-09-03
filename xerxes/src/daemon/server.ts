// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { createServer, type Server, type Socket } from "node:net";
import { homedir } from "node:os";
import { basename, dirname, isAbsolute, join, relative, resolve } from "node:path";

import {
  CATEGORIES,
  listCommands,
  resolveCommand,
  type CommandDefinition,
} from "../bridge/commands.js";
import {
  listAgentDefinitions,
  type AgentDefinition,
} from "../agents/definitions.js";
import { persistedSubagentSnapshotValues } from "../agents/subagentPersistence.js";
import { AgentPresetRoster, type AgentPresetEntry } from "../agents/presets.js";
import { CodexSession, fetchCodexModelCatalog } from "../auth/codexAuth.js";
import { collectSubscriptionUsage, formatUsageReport } from "../auth/usage.js";
import { CopilotSession, fetchCopilotModels } from "../auth/copilotAuth.js";
import {
  fallbackReasoningLevels,
  providerReasoningLevels,
  REASONING_OFF,
  catalogReasoningLevels,
  clampEffort,
  reasoningShapeNote,
  resolveEffort,
  selectableEfforts,
  type ReasoningLevelSet,
} from "../llms/reasoningLevels.js";
import {
  ProfileStore,
  resolvedProfileMaxOutputTokens,
  resolvedProfileModelCapabilities,
  SAMPLING_PARAMS,
  type ProviderProfile,
} from "../bridge/profiles.js";
import {
  ChannelManager,
  type ManagedChannelStatus,
} from "../channels/manager.js";
import {
  ChannelWebhookServer,
  type ChannelWebhookServerOptions,
} from "../channels/webhookServer.js";
import { estimateContextTokens } from "../context/windowUsage.js";
import { isNamedPipePath } from "../core/hostPlatform.js";
import {
  createChannelMessage,
  MessageDirection,
  type ChannelMessage,
} from "../channels/types.js";
import { routeOutput } from "../cron/delivery.js";
import { CronJob, JobStore, nextFireAt } from "../cron/jobs.js";
import {
  acquireCronLease,
  readCronLease,
  releaseCronLease,
} from "../cron/lease.js";
import { CronScheduler } from "../cron/scheduler.js";
import { blockGoal, getGoal, pauseGoal } from "../runtime/goalDomain.js";
import { runGoalCommand } from "./goalCommand.js";
import {
  nextGoalRound,
  type AdmittedGoalRound,
} from "../runtime/goalRoundDriver.js";
import {
  defaultSkillDiscoveryRoots,
  skillActivationPrompt,
  skillMatchesPlatform,
  SkillRegistry,
  trustedHashWorkspaceSkills,
} from "../extensions/skills.js";
import { expandSkillInstructions } from "../extensions/skillInjection.js";
import { skillSuggestionValues } from "../extensions/skillSuggestions.js";
import {
  creatorTraceValues,
  DeclarativeToolForge,
  recordCreatorTrace,
  type CreatorTraceRow,
  type DeclarativeForgeDefinition,
  type DeclarativeForgePackage,
} from "../extensions/declarativeForge.js";
import {
  getDefaultSlashPluginRegistry,
  type SlashPluginRegistry,
} from "../extensions/slashPlugins.js";
import { PluginRegistry } from "../extensions/plugins.js";
import {
  JsonRpcParseError,
  daemonEvent,
  jsonRpcFailure,
  jsonRpcSuccess,
  parseJsonRpcRequest,
  type JsonRpcId,
  type JsonRpcPayload,
  type JsonRpcRequest,
} from "../protocol/jsonRpc.js";
import {
  calcCost,
  effectiveContextLimit,
  PROVIDERS,
  resolveProvider,
  type ProviderName,
} from "../llms/providerRegistry.js";
import {
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_K,
} from "../llms/samplingDefaults.js";
import {
  closeLlmClient,
  createLlmClient,
  requireConfiguredModel,
  type LlmClient,
} from "../llms/client.js";
import {
  DEFAULT_RADIUS_GATEWAY,
  getRadiusModelsFromConfig,
  loadRadiusGatewayConfig,
  normalizeRadiusGatewayUrl,
} from "../llms/radiusGateway.js";
import {
  attemptSessionTitle,
  displayTitle,
  generateSessionTitle,
  provisionalTitleFrom,
  type TitleClientFactory,
} from "./titleGenerator.js";
import { formatDoctorReport, runAllDoctorChecks } from "../runtime/doctor.js";
import { formatGitUpdateStatus, gitUpdateStatus } from "../runtime/update.js";
import {
  FILE_READS_METADATA_KEY,
  fileStateTracker,
} from "../tools/fileState.js";
import { WorkspaceMemoryStore } from "../tools/workspaceMemory.js";
import {
  DEFAULT_PERMISSION_MODE,
  type PermissionMode,
} from "../streaming/permissions.js";
import { closeCodexWebSocketSessions } from "../streaming/codexWebSocket.js";
import {
  loadProjectAgentWorkspace,
  projectAgentsDir,
} from "../runtime/projectWorkspace.js";
import {
  AgentMemory,
  CANONICAL_AGENT_MEMORY_FILES,
} from "../memory/agentMemory.js";
import { MCPManager } from "../mcp/manager.js";
import { BrowserManager } from "../operators/browser.js";
import type { TerminalRegistry } from "../runtime/terminalRegistry.js";
import { looksLikeSessionId } from "../session/daemonTranscript.js";
import {
  describeTranscriptRepair,
  summarizeTranscriptRepair,
} from "../session/resumeRepair.js";
import { SnapshotManager, type SnapshotRecord } from "../session/snapshots.js";
import {
  TranscriptSearchIndex,
  type TranscriptSearchHit,
} from "../session/transcriptSearch.js";
import { processAtMentions } from "./atMentions.js";
import {
  compactMessagesIfNeeded,
  compactionCompletionPort,
  lazyCompactionCompletionPort,
  compactionThresholdTokens,
  DEFAULT_AUTO_COMPACT_THRESHOLD,
  normalizeCompactionThreshold,
  precompactArchivePathFor,
} from "./compactionRunner.js";
import {
  MAX_TRANSCRIPT_INLINE_IMAGE_BYTES,
  MAX_TRANSCRIPT_TOTAL_INLINE_IMAGE_BYTES,
  validateTurnImages,
  type TurnImage,
} from "./images.js";
import { DaemonInteractionBoard } from "./interactions.js";
import {
  queueSessionNotification,
  takeSessionNotifications,
} from "./sessionNotifications.js";
import {
  discoverModelCatalog,
  discoverModelIds,
  profileDiscoveryApiKey,
  sanitizeModelDiscoveryError,
  type DiscoveredModel,
} from "./modelDiscovery.js";
import {
  ProviderProfileFlow,
  type ProviderFlowPrompt,
  type ProviderFlowTransition,
  type ProviderModelDiscoveryPort,
} from "./providerFlow.js";
import { SkillCreateFlow, type SkillCreateTransition } from "./skillCreate.js";
import {
  BUN_DAEMON_BUILD_ID,
  DAEMON_PROTOCOL_VERSION,
  XERXES_VERSION,
  type DaemonEvent,
  type DaemonRuntime,
  type DaemonSession,
  type DaemonTranscriptMessage,
  type SavedDaemonSession,
  type SubmitTurnOptions,
  InMemoryDaemonRuntime,
} from "./runtime.js";
import { resolveProjectDirectory, xerxesHome } from "./paths.js";
import { formatBytes, wipeHistoryStores, wipeMemoryStores } from "./wipe.js";
import { searchProjectFileMentions } from "./projectFileMentions.js";
import type { DaemonTransportConnection } from "./transport.js";
import {
  DaemonWebSocketGateway,
  type DaemonWebSocketGatewayOptions,
} from "./websocketGateway.js";

export const MIGRATED_ERROR =
  "Old daemon task API was removed; use session.open, turn.submit, turn.cancel, session.list, and runtime.status.";

/** Matches the WebSocket gateway default so both transports cap inbound frames. */
const DEFAULT_MAX_SOCKET_FRAME_BYTES = 16 * 1024 * 1024;
const DEFAULT_MAX_PENDING_SOCKET_REQUESTS = 1_024;
const DEFAULT_MAX_PENDING_SOCKET_BYTES = 16 * 1024 * 1024;
const DEFAULT_MAX_SOCKET_OUTPUT_BYTES = 16 * 1024 * 1024;

/**
 * Accepted client submission ids retained per session for reconnect-retry
 * idempotency. A long-lived daemon otherwise grows the set with every submit,
 * so it is FIFO-bounded and cleared when a session is evicted; the bound is
 * far above any realistic retry window.
 */
export const MAX_ACCEPTED_SUBMISSION_IDS = 4_096;

/**
 * Compaction mechanics — thresholds, summary budgets, retry policy and the
 * pre-compaction archive — live in `compactionRunner` so delegated children
 * run the identical routine. `compactionCompletionPort` is re-exported
 * because hosts import it from this module.
 */
export { compactionCompletionPort } from "./compactionRunner.js";

/**
 * Consecutive auto-compaction failures after which the session stops trying.
 *
 * Nothing about a failed compaction is persisted, so the next turn re-evaluates
 * the identical condition and pays for another full-window summarization call —
 * every turn, forever. Several of those failures surface as "Nothing to
 * compact", which reads as benign.
 */
const MAX_AUTO_COMPACT_FAILURES = 3;

/**
 * Fraction of the prompt budget at which a daemon with auto-compaction turned
 * off says so. Disabling the threshold is a valid choice; walking into a
 * provider 400 without one word of warning is not.
 */
const AUTO_COMPACT_DISABLED_WARNING_FRACTION = 0.9;

/**
 * How long shutdown waits for in-flight turns before persisting anyway. A
 * generator that never settles is a bug; losing the transcript because of it
 * is a worse one.
 */
const TURN_DRAIN_TIMEOUT_MS = 2_000;

/**
 * How many completed turns a still-untitled session stays eligible for
 * automatic naming.
 *
 * The title is always generated from the opening exchange, so the prompt is
 * identical whether it runs on turn 1 or turn 3. The window exists purely so
 * a transient provider failure on the very first turn does not orphan a new
 * session forever. It is intentionally short: this is a retry for new
 * sessions, not a backfill for long-lived history.
 */
const TITLE_RETRY_TURN_WINDOW = 3;

/**
 * Methods dispatched WITHOUT holding the per-connection serialization queue.
 *
 * That queue exists so handlers cannot race on shared session state, and every
 * handler pays for it: one slow request stalls every later request from the
 * same client. Model discovery is the pathological case — it reads the profile
 * store, then waits on a provider's network endpoint, which for a dead
 * self-hosted URL means the full `MODEL_DISCOVERY_TIMEOUT_MS`. Measured on a
 * real profile set: nine providers took 16.2s end to end, 8.4s of it one
 * unreachable endpoint, and everything queued behind it — including whichever
 * provider the user was actually looking at, which is why the model picker sat
 * on "discovering models…" while the daemon was perfectly healthy.
 *
 * Safe to admit here because these handlers touch no session: they read the
 * profile store and write only `discoveredContextLimits`, keyed by profile, so
 * two concurrent runs either address different keys or write identical values.
 * Do NOT add a method that mutates a session, its transcript, or its metadata.
 */
const CONCURRENT_DISPATCH_METHODS = new Set([
  "creator_trace",
  "fetch_models",
  "forge.inspect",
  "forge.list",
  "provider_models",
]);

/**
 * Snapshots retained per workspace. A per-turn snapshot makes the shadow repo
 * grow with the conversation, so the daemon prunes it once on start rather
 * than letting a long-lived project accumulate history forever.
 */
const DEFAULT_SNAPSHOT_RETENTION = 200;

/** Transcripts read from disk the first time a cross-session search runs. */
const SEARCH_HYDRATION_SESSION_LIMIT = 200;
/** Rows one `/search` renders; the RPC returns the same bounded set. */
const SEARCH_RESULT_LIMIT = 20;

/** Resolve when `work` settles or the timer fires, whichever comes first. */
async function raceWithTimeout(
  work: Promise<unknown>,
  timeoutMs: number,
): Promise<void> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      work.then(
        () => undefined,
        () => undefined,
      ),
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, timeoutMs);
        // The timer must never be the reason the process stays alive.
        timer.unref?.();
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

interface DaemonSlashCommand {
  readonly aliases: readonly string[];
  readonly category: string;
  readonly description: string;
  readonly name: string;
}

/** Canonical commands exposed by concrete native daemon handlers. */
const HANDLED_CANONICAL_COMMANDS: ReadonlySet<string> = new Set([
  "agents",
  "background",
  "branch",
  "branches",
  "browser",
  "budget",
  "btw",
  "cancel-all",
  "clear",
  "commands",
  "compact",
  "config",
  "cost",
  "cron",
  "context",
  "debug",
  "doctor",
  "fast",
  "feedback",
  "history",
  "help",
  "image",
  "init",
  "insights",
  "memory",
  "model",
  "new",
  "nudge",
  "paste",
  "personality",
  "permissions",
  "platforms",
  "plugins",
  "provider",
  "queue",
  "reload",
  "reload-mcp",
  "remove-history",
  "remove-memory",
  "reasoning",
  "restart",
  "resume",
  "retry",
  "retry-connection",
  "save",
  "sampling",
  "skill",
  "skill-create",
  "skills",
  "skin",
  "snapshot",
  "snapshots",
  "status",
  "statusbar",
  "steer",
  "stop",
  "soul",
  "title",
  "toolsets",
  "tools",
  "undo",
  "update",
  "usage",
  "verbose",
  "voice",
  "workspace",
  "yolo",
  "rollback",
  "exit",
]);

/** Descriptions for handlers which intentionally implement a narrower daemon operation. */
const DAEMON_DESCRIPTION_OVERRIDES: Readonly<Record<string, string>> =
  Object.freeze({
    agents: "List native agent definitions",
    clear: "Acknowledge TUI scrollback clear",
    config: "Show effective native runtime configuration",
    context: "Show session token usage",
    platforms: "List configured messaging platforms",
    save: "Persist the active session",
    sampling: "Show or set next-turn native sampling options",
    title: "Show or set the session title",
    tools: "Show native tool count",
    usage: "Show session and subscription usage",
  });

/** These controls are daemon protocol extensions rather than registry commands. */
const DAEMON_EXTENSION_COMMANDS: readonly DaemonSlashCommand[] = Object.freeze([
  Object.freeze({
    name: "mode",
    aliases: Object.freeze([]),
    category: "daemon",
    description: "Set the session interaction mode",
  }),
  Object.freeze({
    name: "plan",
    aliases: Object.freeze([]),
    category: "daemon",
    description: "Toggle plan mode",
  }),
  Object.freeze({
    name: "ultra",
    aliases: Object.freeze([]),
    category: "daemon",
    description: "Toggle ultra mode",
  }),
  Object.freeze({
    name: "search",
    aliases: Object.freeze([]),
    category: "daemon",
    description: "Search every saved transcript",
  }),
]);

const DAEMON_SLASH_COMMANDS: readonly DaemonSlashCommand[] = Object.freeze([
  ...listCommands()
    .filter((commandDefinition) =>
      HANDLED_CANONICAL_COMMANDS.has(commandDefinition.name),
    )
    .map((commandDefinition) => daemonSlashCommand(commandDefinition)),
  ...DAEMON_EXTENSION_COMMANDS,
]);

function daemonSlashCommand(
  commandDefinition: CommandDefinition,
): DaemonSlashCommand {
  const legacyAliases = commandDefinition.name === "help" ? ["h"] : [];
  return Object.freeze({
    name: commandDefinition.name,
    aliases: Object.freeze([...commandDefinition.aliases, ...legacyAliases]),
    category: commandDefinition.category,
    description:
      DAEMON_DESCRIPTION_OVERRIDES[commandDefinition.name] ??
      commandDefinition.description,
  });
}

function daemonCommandPairs(
  commands: readonly DaemonSlashCommand[],
): Array<[string, string]> {
  return commands.map((command) => [`/${command.name}`, command.description]);
}

function daemonCommandCategories(): Array<{
  name: string;
  pairs: Array<[string, string]>;
}> {
  const categories = CATEGORIES.flatMap((category) => {
    const pairs = daemonCommandPairs(
      DAEMON_SLASH_COMMANDS.filter((command) => command.category === category),
    );
    return pairs.length ? [{ name: category, pairs }] : [];
  });
  const extensions = daemonCommandPairs(DAEMON_EXTENSION_COMMANDS);
  return extensions.length
    ? [...categories, { name: "daemon", pairs: extensions }]
    : categories;
}

function slashCompletionPrefix(text: string): string {
  const withoutSlash = text.trim().replace(/^\/+/, "");
  return (withoutSlash.split("@", 1)[0] ?? "").toLowerCase();
}

const RUNTIME_OVERRIDE_KEYS = new Set([
  "api_key",
  "auto_title",
  "base_url",
  "context_limit",
  "frequency_penalty",
  "max_context",
  "max_context_tokens",
  "max_tokens",
  "min_p",
  "model",
  "permission_mode",
  "presence_penalty",
  "provider",
  "reasoning_effort",
  "repetition_penalty",
  "responses_api",
  "service_tier",
  "temperature",
  "thinking",
  "thinking_budget",
  "top_k",
  "top_p",
]);

const DISPLAYED_RUNTIME_CONFIG_KEYS = [
  "base_url",
  "max_tokens",
  "model",
  "permission_mode",
  "provider",
  "responses_api",
  "temperature",
  "top_k",
  "top_p",
] as const;

const NATIVE_SAMPLING_KEYS = Object.freeze([...SAMPLING_PARAMS]);

type NativeSamplingKey = (typeof NATIVE_SAMPLING_KEYS)[number];

interface CronAddArguments {
  readonly at?: string;
  readonly deliver?: string;
  readonly prompt: string;
  readonly recipient?: string;
  readonly schedule?: string;
  readonly workspaceId?: string;
}

type ParsedCronAddArguments = CronAddArguments | { readonly error: string };

/** A host-owned UI action exposed through a daemon slash command. */
export type DaemonUiAction = "paste" | "queue" | "skin" | "statusbar" | "voice";

export interface DaemonUiControlInput {
  readonly action: DaemonUiAction;
  readonly argument: string;
  readonly sessionKey: string;
}

/**
 * Optional TUI boundary for controls which cannot be performed by a headless
 * daemon. The command is always emitted as a typed daemon event, so a native
 * client can handle it without coupling the daemon to a particular TUI.
 */
export interface DaemonUiControlPort {
  execute(input: DaemonUiControlInput):
    | void
    | { readonly message?: string; readonly payload?: JsonRpcPayload }
    | Promise<void | {
        readonly message?: string;
        readonly payload?: JsonRpcPayload;
      }>;
}

/** Native tool inventory supplied by an embedding runtime when it owns tools. */
export interface DaemonToolCatalogPort {
  listTools():
    | readonly { readonly description?: string; readonly name: string }[]
    | Promise<
        readonly { readonly description?: string; readonly name: string }[]
      >;
}

export interface DaemonServerOptions {
  /** Resolve real native agent definitions for `/agents`; injectable for embedding hosts. */
  readonly agentDefinitionLoader?: (cwd: string) => readonly AgentDefinition[];
  /**
   * The project this daemon owns (`--project-dir`). When a client binds a
   * session without an explicit `project_dir`, this — not the daemon
   * process's launch directory — is the workspace the session lives in;
   * otherwise a daemon spawned from another cwd silently misfiles every
   * session it creates.
   */
  readonly projectDirectory?: string;
  /**
   * Capture a workspace snapshot before every turn, so an agent-made mess is
   * recoverable without the user having remembered to run `/snapshot`.
   *
   * Opt-in: each snapshot spawns git over the whole workspace, which a host
   * embedding the daemon in a large or non-git tree may not want per turn.
   */
  readonly autoSnapshotTurns?: boolean;
  /**
   * Context-usage fraction that triggers provider-backed auto-compaction
   * before a turn is submitted. Defaults to 0.8; a runtime setting of
   * `auto_compact_threshold` overrides it per daemon, and 0 disables it.
   */
  readonly autoCompactThreshold?: number;
  /**
   * Generate a model-written session title after the first exchange. Defaults
   * on; a runtime setting of `auto_title` set to false disables it. Generation
   * is background and silent: a failure leaves the session untitled and never
   * surfaces in the turn.
   */
  readonly autoTitle?: boolean;
  /** Test seam for title generation; production uses the real client factory. */
  readonly titleClientFactory?: TitleClientFactory;
  /** Browser state shared with native operator tools; `/browser` never invents a browser backend. */
  readonly browserManager?: BrowserManager;
  /**
   * Terminals the agent is driving, shared with the tool registry that runs them.
   *
   * Absent, `terminal.*` reports an empty list rather than inventing a registry:
   * a registry the tools do not write into would show nothing while claiming to
   * be complete.
   */
  readonly terminalRegistry?: TerminalRegistry;
  /** Host-owned adapter registry. No channel transport is synthesized when absent. */
  readonly channelManager?: ChannelManager;
  /** Optional Bun HTTP listener that delivers provider webhooks to configured channel adapters. */
  readonly channelWebhook?: Omit<ChannelWebhookServerOptions, "manager">;
  /** Directory used to archive every automatic and manually-run cron result. */
  readonly cronArchiveDirectory?: string;
  /**
   * Exclusive-lease file deciding which daemon fires cron jobs. Every project's
   * daemon shares one job store, so without the lease each of them runs every
   * job. Defaults to a path under the Xerxes home.
   */
  readonly cronLeasePath?: string;
  /** How often a daemon refused the cron lease re-probes for it. Defaults to a minute. */
  readonly cronLeaseRetryInterval?: number;
  /** Testable native scheduler cadence; production defaults to 30 seconds. */
  readonly cronPollInterval?: number;
  /**
   * Install process-level `uncaughtException` / `unhandledRejection` handlers
   * that flush sessions before exiting. Off by default: only a host that owns
   * the whole process may claim those handlers.
   */
  readonly crashHandlers?: boolean;
  /** Shared approval/question state passed to the native agent turn runner. */
  readonly interactions?: DaemonInteractionBoard;
  /** Legacy declarative text-template store retained for persisted packages. */
  readonly declarativeForge?: DeclarativeToolForge;
  /** DSH-style live agent-preset roster used by session selection and authoring RPCs. */
  readonly agentPresetRoster?: AgentPresetRoster;
  /** Opens the persistent native cron job store used by `/cron list`. */
  readonly cronStoreFactory?: () => JobStore;
  /** Native MCP lifecycle owner used by `/reload-mcp`. */
  readonly mcpManager?: MCPManager;
  /** Optional persistent-memory factory; defaults to native global + project memory. */
  readonly memoryFactory?: (session: DaemonSession | undefined) => AgentMemory;
  /** Called for `/restart`; without it the daemon performs a graceful native shutdown. */
  readonly onRestart?: () => void | Promise<void>;
  /** Called for RPC `shutdown`; the process host remains responsible for final cleanup. */
  readonly onShutdown?: () => void | Promise<void>;
  readonly pidPath?: string;
  /** Native extension registry used by `/plugins`. */
  readonly pluginRegistry?: PluginRegistry;
  /** Persistent native provider profile store. */
  readonly profileStore?: ProfileStore;
  /** Refresh provider-reported model capabilities after initialize. Host opt-in avoids ambient network calls. */
  readonly autoDiscoverModelCapabilities?: boolean;
  /** Optional host-owned model catalogue lookup for interactive `/provider` setup. */
  readonly providerModelDiscovery?: ProviderModelDiscoveryPort;
  readonly runtime?: DaemonRuntime;
  /**
   * Directory holding persisted transcripts, where compaction archives the
   * history it is about to replace. Must match the runtime's transcript
   * directory for archives to land beside their session; defaults to the
   * daemon's own session directory, which is what the production host uses.
   */
  readonly sessionArchiveDirectory?: string;
  /** Directories re-scanned by `/skills` and `/reload`; defaults to all native discovery roots. */
  readonly skillDirectories?: readonly string[];
  /** Writable user-owned root used by the interactive `/skill-create` flow. */
  readonly skillDirectory?: string;
  /** Native skill registry used by `/skills`, `/skill`, and skill shorthand commands. */
  readonly skillRegistry?: SkillRegistry;
  /** Plugin slash commands share the daemon dispatch path rather than a Python fallback. */
  readonly slashPluginRegistry?: SlashPluginRegistry;
  /** Creates filesystem snapshots for the active session workspace. */
  readonly snapshotManagerFactory?: (
    workspaceDirectory: string,
  ) => SnapshotManager;
  /** Snapshots retained per workspace when the store is pruned on start. */
  readonly snapshotRetention?: number;
  readonly socketPath: string;
  /** Max bytes in one inbound NDJSON frame before the Unix client is dropped. */
  readonly maxSocketFrameBytes?: number;
  /** Max parsed requests waiting for serial dispatch on one Unix connection. */
  readonly maxPendingSocketRequests?: number;
  /** Max aggregate bytes of parsed requests waiting for serial dispatch. */
  readonly maxPendingSocketBytes?: number;
  /** Max queued outbound bytes before a slow Unix client is dropped. */
  readonly maxSocketOutputBytes?: number;
  /** Tool inventory port for `/tools`; omit only when the runtime owns no visible tool registry. */
  readonly toolCatalog?: DaemonToolCatalogPort;
  /** Typed bridge for UI-only slash commands such as `/skin` and `/paste`. */
  readonly uiControl?: DaemonUiControlPort;
  /** Optional remote JSON-RPC transport; omitted by default to avoid network exposure. */
  readonly websocket?: DaemonWebSocketGatewayOptions;
}

interface Connection extends DaemonTransportConnection {
  activeSessionKey: string;
  buffer: string;
  pendingRequestBytes: number;
  pendingRequestCount: number;
  queuedOutputBytes: number;
  readonly outputQueue: string[];
  outputBlocked: boolean;
  /** Serializes request dispatch so interleaved handlers cannot race on shared state. */
  queue: Promise<void>;
  readonly socket: Socket;
}

interface ChannelStatusData {
  readonly available: boolean;
  readonly channels: JsonRpcPayload[];
  readonly configured: boolean;
}

/** NDJSON JSON-RPC v35 Unix socket server consumed by the OpenTUI client and native hosts. */
/**
 * Event types that prove a turn actually did something.
 *
 * Deliberately narrow: a turn that emits only status and lifecycle events has
 * produced nothing an objective can be advanced by, however successful it looks
 * from the outside.
 */
const PRODUCTIVE_TURN_EVENTS: ReadonlySet<string> = new Set([
  "text_part",
  "thinking_part",
  "tool_call",
  "tool_result",
]);

export class DaemonServer {
  private readonly agentDefinitionLoader: (
    cwd: string,
  ) => readonly AgentDefinition[];
  private readonly approvalOwners = new Map<
    string,
    DaemonTransportConnection
  >();
  /** Serializes compaction and turn admission for each session. */
  private readonly sessionOperations = new Map<string, Promise<void>>();
  /** Consecutive auto-compaction failures per session; reset by any deliberate history change. */
  private readonly autoCompactFailures = new Map<string, number>();
  /** Sessions already told that auto-compaction is off while their window fills. */
  private readonly autoCompactDisabledWarned = new Set<string>();
  private readonly autoCompactThreshold: number;
  private readonly autoTitle: boolean;
  private readonly titleClientFactory: TitleClientFactory | undefined;
  private readonly browserManager: BrowserManager;
  private readonly channelManager: ChannelManager | undefined;
  private readonly channelWebhookServer: ChannelWebhookServer | undefined;
  private readonly connections = new Set<Connection>();
  private readonly cronArchiveDirectory: string;
  private cronLeaseProbe: ReturnType<typeof setInterval> | undefined;
  private readonly cronLeaseOwnerKey: string;
  private readonly cronLeasePath: string;
  private cronLeaseRefusalLogged = false;
  private readonly cronLeaseRetryInterval: number;
  private readonly cronScheduler: CronScheduler;
  private cronSchedulerStarted = false;
  private readonly cronStore: JobStore;
  private readonly cronStoreFactory: () => JobStore;
  private crashHandler: ((error: unknown) => void) | undefined;
  private readonly crashHandlersEnabled: boolean;
  private readonly interactions: DaemonInteractionBoard;
  private readonly declarativeForge: DeclarativeToolForge;
  private readonly agentPresetRoster: AgentPresetRoster;
  private readonly agentPresetSwitches = new Map<string, Promise<void>>();
  private readonly inFlightTurns = new Set<Promise<void>>();
  private readonly mcpManager: MCPManager | undefined;
  private readonly maxSocketFrameBytes: number;
  private readonly maxPendingSocketRequests: number;
  private readonly maxPendingSocketBytes: number;
  private readonly maxSocketOutputBytes: number;
  private readonly memoryFactory: (
    session: DaemonSession | undefined,
  ) => AgentMemory;
  private readonly onRestart: (() => void | Promise<void>) | undefined;
  private readonly onShutdown: (() => void | Promise<void>) | undefined;
  private readonly pidPath: string | undefined;
  private readonly pluginRegistry: PluginRegistry;
  private readonly providerFlows = new Map<
    DaemonTransportConnection,
    ProviderProfileFlow
  >();
  private readonly providerModelDiscovery:
    ProviderModelDiscoveryPort | undefined;
  private readonly discoveredContextLimits = new Map<string, number>();
  private readonly autoDiscoverModelCapabilities: boolean;
  private readonly modelCapabilityRefreshes = new Map<string, Promise<void>>();
  /** Per-model reasoning-level sets, so the picker does not refetch each open. */
  private readonly reasoningLevelCache = new Map<string, ReasoningLevelSet>();
  private readonly profileStore: ProfileStore;
  private readonly questionOwners = new Map<
    string,
    DaemonTransportConnection
  >();
  private readonly runtime: DaemonRuntime;
  private runtimeShutdown = false;
  private readonly projectDirectory: string | undefined;
  private readonly sessionArchiveDirectory: string;
  /** True when a host named the transcript directory, so archives are unconditional. */
  private readonly sessionArchiveDirectoryConfigured: boolean;
  private readonly skillDirectories: readonly string[] | undefined;
  private readonly skillRegistry: SkillRegistry;
  private readonly skillCreates = new Map<
    DaemonTransportConnection,
    SkillCreateFlow
  >();
  private readonly skillDirectory: string;
  private readonly slashPluginRegistry: SlashPluginRegistry;
  private readonly snapshotManagerFactory: (
    workspaceDirectory: string,
  ) => SnapshotManager;
  private readonly autoSnapshotTurns: boolean;
  private readonly snapshotRetention: number;
  private server: Server | undefined;
  private readonly socketPath: string;
  private readonly terminalRegistry: TerminalRegistry | undefined;
  private readonly transcriptSearch = new TranscriptSearchIndex();
  /**
   * Accepted client submission ids, keyed by `<session-key>\u0000<submission-id>`
   * and FIFO-bounded, so reconnect retries stay idempotent without growing a
   * daemon-lifetime set. Entries for an evicted session are dropped with it.
   */
  private readonly acceptedSubmissionIds = new Set<string>();
  /** Session-scoped signals bounding background work to the session's life. */
  private readonly sessionLifetimeSignals = new Map<string, AbortController>();
  /** One cold read of the transcript directory, shared by concurrent searches. */
  private transcriptSearchHydration: Promise<void> | undefined;
  private readonly toolCatalog: DaemonToolCatalogPort | undefined;
  private readonly turnOwners = new Map<string, DaemonTransportConnection>();
  private readonly uiControl: DaemonUiControlPort | undefined;
  private readonly websocketOptions: DaemonWebSocketGatewayOptions | undefined;
  private websocketGateway: DaemonWebSocketGateway | undefined;

  constructor(options: DaemonServerOptions) {
    this.socketPath = options.socketPath;
    this.pidPath = options.pidPath;
    this.projectDirectory = options.projectDirectory
      ? resolveProjectDirectory(options.projectDirectory)
      : undefined;
    this.autoCompactThreshold = normalizeCompactionThreshold(
      options.autoCompactThreshold ?? DEFAULT_AUTO_COMPACT_THRESHOLD,
    );
    this.autoTitle = options.autoTitle ?? true;
    this.titleClientFactory = options.titleClientFactory;
    this.channelManager = options.channelManager;
    this.channelWebhookServer =
      options.channelManager && options.channelWebhook
        ? new ChannelWebhookServer({
            ...options.channelWebhook,
            manager: options.channelManager,
          })
        : undefined;
    this.runtime = options.runtime ?? new InMemoryDaemonRuntime();
    this.sessionArchiveDirectory =
      options.sessionArchiveDirectory ?? join(xerxesHome(), "sessions");
    this.sessionArchiveDirectoryConfigured =
      options.sessionArchiveDirectory !== undefined;
    this.agentDefinitionLoader =
      options.agentDefinitionLoader ?? ((cwd) => listAgentDefinitions({ cwd }));
    this.interactions = options.interactions ?? new DaemonInteractionBoard();
    this.declarativeForge = options.declarativeForge ?? new DeclarativeToolForge();
    this.agentPresetRoster = options.agentPresetRoster ?? new AgentPresetRoster({
      ...(this.projectDirectory ? { projectDirectory: this.projectDirectory } : {}),
    });
    this.maxSocketFrameBytes =
      options.maxSocketFrameBytes ?? DEFAULT_MAX_SOCKET_FRAME_BYTES;
    this.maxPendingSocketRequests =
      options.maxPendingSocketRequests ?? DEFAULT_MAX_PENDING_SOCKET_REQUESTS;
    this.maxPendingSocketBytes =
      options.maxPendingSocketBytes ?? DEFAULT_MAX_PENDING_SOCKET_BYTES;
    this.maxSocketOutputBytes =
      options.maxSocketOutputBytes ?? DEFAULT_MAX_SOCKET_OUTPUT_BYTES;
    this.crashHandlersEnabled = options.crashHandlers === true;
    this.cronLeaseOwnerKey = resolveProjectDirectory(process.cwd());
    this.cronLeasePath = resolve(
      options.cronLeasePath ?? join(xerxesHome(), "cron", "scheduler.lease"),
    );
    this.cronLeaseRetryInterval = options.cronLeaseRetryInterval ?? 60_000;
    this.cronStoreFactory =
      options.cronStoreFactory ??
      // Stamp the owning project onto every job this store creates: the file is
      // shared across projects, so a job that does not name its repo can never
      // be attributed to one afterwards.
      (() =>
        new JobStore(join(xerxesHome(), "cron", "jobs.json"), {
          projectRoot: this.cronLeaseOwnerKey,
        }));
    this.cronStore = this.cronStoreFactory();
    this.cronArchiveDirectory = resolve(
      options.cronArchiveDirectory ?? join(xerxesHome(), "cron", "archive"),
    );
    this.cronScheduler = new CronScheduler(
      this.cronStore,
      (job) => this.runScheduledCronJob(job),
      {
        // The lease is re-checked on every tick, not just at start: a daemon
        // that loses or releases it mid-run must stop firing immediately.
        holdsLease: () => this.cronSchedulerStarted && this.holdsCronLease(),
        onComplete: async (job, output) => {
          await this.deliverCronOutput(job, output);
        },
        ...(options.cronPollInterval === undefined
          ? {}
          : { pollInterval: options.cronPollInterval }),
      },
    );
    this.profileStore = options.profileStore ?? new ProfileStore();
    // Only the process-owning host opts in; embeddings remain network-silent.
    this.autoDiscoverModelCapabilities = options.autoDiscoverModelCapabilities ?? false;
    this.providerModelDiscovery =
      options.providerModelDiscovery ??
      {
        discover: (input) =>
          discoverModelIds({
            allowPrivateEndpoint: true,
            apiKey: input.apiKey,
            baseUrl: input.baseUrl,
            provider: input.provider,
            resolveProviderCredential: false,
          }),
      };
    this.mcpManager = options.mcpManager;
    this.memoryFactory =
      options.memoryFactory ??
      ((session) =>
        new AgentMemory(session?.cwd ? { projectRoot: session.cwd } : {}));
    this.onRestart = options.onRestart;
    this.onShutdown = options.onShutdown;
    this.pluginRegistry = options.pluginRegistry ?? new PluginRegistry();
    this.skillDirectory = resolve(
      options.skillDirectory ?? join(homedir(), ".xerxes", "skills"),
    );
    this.skillDirectories = options.skillDirectories;
    this.skillRegistry =
      options.skillRegistry ??
      new SkillRegistry({ workspaceTrust: trustedHashWorkspaceSkills() });
    this.slashPluginRegistry =
      options.slashPluginRegistry ?? getDefaultSlashPluginRegistry();
    this.snapshotManagerFactory =
      options.snapshotManagerFactory ??
      ((workspaceDirectory) => new SnapshotManager(workspaceDirectory));
    this.autoSnapshotTurns = options.autoSnapshotTurns === true;
    this.snapshotRetention =
      Number.isInteger(options.snapshotRetention) &&
      (options.snapshotRetention ?? -1) >= 0
        ? (options.snapshotRetention ?? DEFAULT_SNAPSHOT_RETENTION)
        : DEFAULT_SNAPSHOT_RETENTION;
    this.websocketOptions = options.websocket;
    this.browserManager = options.browserManager ?? new BrowserManager();
    this.terminalRegistry = options.terminalRegistry;
    this.toolCatalog = options.toolCatalog;
    this.uiControl = options.uiControl;
  }

  /** Public remote WebSocket endpoint after start, including an OS-assigned port. */
  get websocketUrl(): URL | undefined {
    return this.websocketGateway?.url;
  }

  /** Public channel webhook base URL after start when the host configured one. */
  get channelWebhookUrl(): URL | undefined {
    return this.channelWebhookServer?.url;
  }

  async start(): Promise<void> {
    if (this.server) {
      return;
    }
    if (!isNamedPipePath(this.socketPath)) {
      await mkdir(dirname(this.socketPath), { recursive: true });
    }
    await this.unlinkSocketPath();
    this.server = createServer((socket) => this.attach(socket));
    await new Promise<void>((resolve, reject) => {
      const server = this.server;
      if (!server) {
        reject(new Error("Daemon server was not initialized"));
        return;
      }
      server.once("error", reject);
      server.listen(this.socketPath, () => {
        server.off("error", reject);
        // A listening server without an "error" listener crashes the process
        // on any asynchronous transport failure; log it instead.
        server.on("error", (error) => {
          console.error("Xerxes daemon socket server error:", error);
        });
        resolve();
      });
    });
    try {
      this.startWebSocketGateway();
      this.channelWebhookServer?.start();
      if (this.pidPath) {
        await mkdir(dirname(this.pidPath), { recursive: true });
        await writeFile(this.pidPath, `${process.pid}\n`, "utf8");
      }
      this.installCrashHandlers();
      this.startCronSchedulerIfOwned();
      this.pruneSnapshotStore();
    } catch (error) {
      this.stopCronScheduler();
      this.removeCrashHandlers();
      await this.channelWebhookServer?.stop();
      await this.websocketGateway?.stop();
      this.websocketGateway = undefined;
      await closeServer(this.server);
      this.server = undefined;
      await this.unlinkSocketPath();
      await this.shutdownRuntime();
      throw error;
    }
  }

  /**
   * Drop a stale Unix socket file; a no-op for a Windows named pipe.
   *
   * A leftover socket file from a crashed daemon makes bind() fail with
   * EADDRINUSE, so removing it before listen() is load-bearing on POSIX. A named
   * pipe is a kernel object with no filesystem entry: unlink cannot address it,
   * and the resulting rejection used to abort daemon startup on Windows before
   * it reached listen() at all. The pipe disappears with its last handle, so
   * there is nothing to clean up.
   */
  private async unlinkSocketPath(): Promise<void> {
    if (isNamedPipePath(this.socketPath)) return;
    await rm(this.socketPath, { force: true });
  }

  /**
   * Trim this project's shadow snapshot repository once per daemon start.
   *
   * Pruning re-anchors the retained history and garbage-collects the rest,
   * which is git work: it is fire-and-forget so it can never delay the socket
   * becoming available, and a failure is not worth a daemon that refuses to
   * start. A workspace that was never snapshotted has no record log, so this
   * is a no-op that creates nothing.
   */
  private pruneSnapshotStore(): void {
    const workspaceDirectory = resolveProjectDirectory(process.cwd());
    void (async () => {
      const removed = await this.snapshotManagerFactory(workspaceDirectory).prune({
        keep: this.snapshotRetention,
      });
      if (removed > 0) {
        console.info(
          `Pruned ${removed} old workspace snapshot${removed === 1 ? "" : "s"} for ${workspaceDirectory}`,
        );
      }
    })().catch((error: unknown) => {
      console.warn(`Could not prune workspace snapshots: ${errorMessage(error)}`);
    });
  }

  /**
   * Capture the workspace as it stands before a turn runs.
   *
   * Fire-and-forget with the rejection swallowed: a snapshot is a safety net,
   * and a net that can fail the turn it protects is worse than no net. The
   * record carries the session id and the index of the turn it precedes, which
   * is what makes "take me back to before turn 7" answerable at all.
   */
  private captureTurnSnapshot(sessionKey: string): void {
    if (!this.autoSnapshotTurns) {
      return;
    }
    const session = this.runtime.sessionStatus(sessionKey);
    if (!session) {
      return;
    }
    const { cwd, id, turnCount } = session;
    // The factory itself can throw, and it runs on the submit path: without
    // this the turn would fail before it ever reached the runtime.
    void (async () =>
      this.snapshotManagerFactory(cwd).snapshot(`turn-${turnCount}`, {
        sessionId: id,
        turnIndex: turnCount,
      }))().catch((error: unknown) => {
      console.warn(`Could not snapshot the workspace before a turn: ${errorMessage(error)}`);
    });
  }

  /**
   * Start cron only while this daemon holds the lease.
   *
   * The job store is one file shared by every project's daemon, so an
   * unconditional start had each open project firing the same job as its own
   * agent turn. A refusal is not an error — another daemon owns cron — but it
   * has to be recoverable without a restart, hence the unref'd probe: it can
   * never be the reason the process stays alive.
   */
  private startCronSchedulerIfOwned(): void {
    if (this.cronSchedulerStarted) {
      return;
    }
    if (this.acquireCronLease()) {
      this.cronScheduler.start();
      this.cronSchedulerStarted = true;
      return;
    }
    if (this.cronLeaseProbe) {
      return;
    }
    this.cronLeaseProbe = setInterval(() => {
      if (this.cronSchedulerStarted) return;
      this.startCronSchedulerIfOwned();
    }, this.cronLeaseRetryInterval);
    this.cronLeaseProbe.unref?.();
  }

  private holdsCronLease(): boolean {
    const holder = readCronLease(this.cronLeasePath);
    return (
      holder !== undefined &&
      holder.pid === process.pid &&
      holder.ownerKey === this.cronLeaseOwnerKey
    );
  }

  private acquireCronLease(): boolean {
    try {
      const outcome = acquireCronLease(this.cronLeasePath, {
        ownerKey: this.cronLeaseOwnerKey,
      });
      if (!outcome.held && !this.cronLeaseRefusalLogged) {
        // Once per daemon: a refused lease is the normal state for every
        // project but the one that owns cron, and logging it on every probe
        // would bury the daemon log.
        this.cronLeaseRefusalLogged = true;
        console.error(
          `Cron scheduling is owned by pid ${outcome.holder?.pid ?? "unknown"}`
            + ` (${outcome.holder?.ownerKey ?? "unknown project"}); this daemon will not fire jobs.`,
        );
      }
      return outcome.held;
    } catch (error) {
      console.error(`Could not take the cron lease: ${errorMessage(error)}`);
      return false;
    }
  }

  private releaseCronLease(): void {
    if (this.cronLeaseProbe !== undefined) {
      clearInterval(this.cronLeaseProbe);
      this.cronLeaseProbe = undefined;
    }
    try {
      releaseCronLease(this.cronLeasePath, { ownerKey: this.cronLeaseOwnerKey });
    } catch (error) {
      console.error(`Could not release the cron lease: ${errorMessage(error)}`);
    }
  }

  private stopCronScheduler(): void {
    this.cronScheduler.stop();
    this.cronSchedulerStarted = false;
    this.releaseCronLease();
  }

  /**
   * Turn an unhandled failure into a saved transcript.
   *
   * Sessions are written once per turn, in the turn's `finally`, so a crash
   * anywhere else discards everything since the last boundary. Opt-in because
   * installing process-global handlers from a constructor-owned object would
   * change the semantics of every host that embeds a DaemonServer.
   */
  private installCrashHandlers(): void {
    if (!this.crashHandlersEnabled || this.crashHandler) {
      return;
    }
    const handler = (error: unknown): void => {
      console.error("Xerxes daemon crashed:", error);
      void this.flushBeforeExit();
    };
    this.crashHandler = handler;
    process.on("uncaughtException", handler);
    process.on("unhandledRejection", handler);
  }

  private removeCrashHandlers(): void {
    const handler = this.crashHandler;
    if (!handler) {
      return;
    }
    this.crashHandler = undefined;
    process.off("uncaughtException", handler);
    process.off("unhandledRejection", handler);
  }

  private async flushBeforeExit(): Promise<void> {
    await raceWithTimeout(
      this.runtime.flushSessions().catch((error: unknown) => {
        console.error(`Could not flush sessions while crashing: ${errorMessage(error)}`);
      }),
      TURN_DRAIN_TIMEOUT_MS,
    );
    process.exit(1);
  }

  private startWebSocketGateway(): void {
    if (!this.websocketOptions) {
      return;
    }
    const gateway = new DaemonWebSocketGateway(
      this.websocketOptions,
      (connection, line) => this.handleLine(connection, line),
      (connection) => this.disconnect(connection),
    );
    try {
      gateway.start();
      this.websocketGateway = gateway;
    } catch {
      // The Unix socket is the primary local control plane. An unavailable
      // optional remote bind must not make the local daemon unusable.
      void gateway.stop();
      this.websocketGateway = undefined;
    }
  }

  async stop(): Promise<void> {
    const server = this.server;
    const gateway = this.websocketGateway;
    const channelWebhook = this.channelWebhookServer;
    if (!server && !gateway && !channelWebhook && !this.cronSchedulerStarted) {
      // Still ours to give back: a start() that failed after taking the lease,
      // or a probe armed while cron was refused, both land here.
      this.releaseCronLease();
      this.removeCrashHandlers();
      await this.shutdownRuntime();
      return;
    }
    // An operator who sends a second SIGTERM has decided this daemon is stuck
    // and wants it gone. Node's default handler is gone once the first signal
    // was consumed with `process.once`, so without this the second signal is
    // swallowed and the only way out is SIGKILL.
    const hardExit = (): void => {
      console.error("Second SIGTERM during shutdown — exiting immediately.");
      process.exit(143);
    };
    process.once("SIGTERM", hardExit);
    try {
      this.stopCronScheduler();
      this.runtime.cancelAllTurns();
      // Let cancelled turns land their final state sync and saveSession, but
      // never wait on them forever: one generator that fails to settle used to
      // park the daemon here with the transcript still unwritten, because the
      // only flush sat behind this await.
      await raceWithTimeout(
        Promise.all([...this.inFlightTurns]),
        TURN_DRAIN_TIMEOUT_MS,
      );
      // Persist before any transport teardown can fail: a channel that hangs
      // on stop must not be able to cost the user their session history.
      await this.runtime.flushSessions();
      await channelWebhook?.stop();
      await this.channelManager?.stopAll();
      for (const connection of this.connections) {
        connection.socket.destroy();
      }
      await gateway?.stop();
      this.websocketGateway = undefined;
      if (server) {
        await new Promise<void>((resolve, reject) =>
          server.close((error) => (error ? reject(error) : resolve())),
        );
      }
      this.server = undefined;
      await this.unlinkSocketPath();
      if (this.pidPath) {
        await rm(this.pidPath, { force: true });
      }
    } finally {
      process.off("SIGTERM", hardExit);
      // Background work bound to a session must not outlive the daemon.
      this.endSessionLifetime([...this.sessionLifetimeSignals.keys()]);
      this.removeCrashHandlers();
      await this.shutdownRuntime();
    }
  }

  private async shutdownRuntime(): Promise<void> {
    if (this.runtimeShutdown) return;
    this.runtimeShutdown = true;
    // Pooled Codex WebSocket sessions outlive individual turns; drop them so
    // the daemon exits without waiting on an idle-timeout close.
    await closeCodexWebSocketSessions();
    await this.runtime.shutdown?.();
  }

  private attach(socket: Socket): void {
    socket.setEncoding("utf8");
    const connection: Connection = {
      socket,
      buffer: "",
      pendingRequestBytes: 0,
      pendingRequestCount: 0,
      queuedOutputBytes: 0,
      outputQueue: [],
      outputBlocked: false,
      queue: Promise.resolve(),
      activeSessionKey: `tui:${newConnectionKey()}`,
      send: (frame) => this.sendSocketFrame(connection, frame),
    };
    this.connections.add(connection);
    socket.on("data", (chunk) => this.receive(connection, chunk));
    socket.on("drain", () => this.flushSocketOutput(connection));
    socket.on("error", () => socket.destroy());
    socket.on("close", () => {
      connection.outputQueue.length = 0;
      connection.queuedOutputBytes = 0;
      this.connections.delete(connection);
      this.disconnect(connection);
    });
  }

  private sendSocketFrame(connection: Connection, frame: object): void {
    if (connection.socket.destroyed) return;
    const encoded = `${JSON.stringify(frame)}\n`;
    const bytes = Buffer.byteLength(encoded, "utf8");
    if (bytes > this.maxSocketOutputBytes) {
      console.error("Xerxes daemon dropping slow client: response exceeds the socket output limit");
      this.destroyWithOutputErrorFrame(connection, frame);
      return;
    }
    if (connection.outputBlocked || connection.outputQueue.length > 0) {
      if (connection.queuedOutputBytes + bytes > this.maxSocketOutputBytes) {
        console.error("Xerxes daemon dropping slow client: queued output exceeds the socket output limit");
        this.destroyWithOutputErrorFrame(connection, frame);
        return;
      }
      connection.outputQueue.push(encoded);
      connection.queuedOutputBytes += bytes;
      return;
    }
    connection.outputBlocked = !connection.socket.write(encoded);
  }

  /**
   * Destroy an over-limit connection, but tell the client why first.
   *
   * A silent destroy leaves the request's author hanging on a response that
   * will never arrive. When the oversized frame carries a JSON-RPC id, deliver
   * a minimal correlated error frame before closing; `end` flushes the write
   * before FIN so the error is not discarded with the socket buffer.
   */
  private destroyWithOutputErrorFrame(connection: Connection, frame: object): void {
    const id = (frame as { id?: unknown }).id;
    const routable = typeof id === "string" ? id.length > 0 : typeof id === "number";
    const socket = connection.socket;
    if (!routable || socket.destroyed) {
      socket.destroy();
      return;
    }
    try {
      const failure = jsonRpcFailure(id as JsonRpcId, -32000, "response exceeds socket output limit");
      socket.end(`${JSON.stringify(failure)}\n`, () => socket.destroy());
    } catch {
      socket.destroy();
    }
  }

  private flushSocketOutput(connection: Connection): void {
    connection.outputBlocked = false;
    while (!connection.socket.destroyed && connection.outputQueue.length > 0) {
      const encoded = connection.outputQueue.shift();
      if (encoded === undefined) return;
      connection.queuedOutputBytes -= Buffer.byteLength(encoded, "utf8");
      if (!connection.socket.write(encoded)) {
        connection.outputBlocked = true;
        return;
      }
    }
  }

  private receive(connection: Connection, chunk: string | Uint8Array): void {
    connection.buffer +=
      typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk);
    let newline = connection.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = connection.buffer.slice(0, newline);
      connection.buffer = connection.buffer.slice(newline + 1);
      const frameBytes = Buffer.byteLength(line, "utf8");
      if (frameBytes > this.maxSocketFrameBytes) {
        console.error("Xerxes daemon dropping client: request exceeds the socket frame limit");
        connection.socket.destroy();
        return;
      }
      if (line.trim()) {
        const pendingBytes = frameBytes + 1;
        if (
          connection.pendingRequestCount + 1 > this.maxPendingSocketRequests ||
          connection.pendingRequestBytes + pendingBytes > this.maxPendingSocketBytes
        ) {
          console.error("Xerxes daemon dropping client: pending requests exceed the socket queue limit");
          connection.socket.destroy();
          return;
        }
        connection.pendingRequestCount += 1;
        connection.pendingRequestBytes += pendingBytes;
        // Resolved when this request hands the queue back: at settlement for
        // an ordinary handler, right after parsing for a concurrent-safe one.
        let releaseQueue!: () => void;
        const queueHandback = new Promise<void>((resolve) => {
          releaseQueue = resolve;
        });
        const handle = async (): Promise<void> => {
          // Work still STARTS in arrival order — only the handback moves.
          const settled = (async () => {
            try {
              if (!connection.socket.destroyed) {
                await this.handleLine(connection, line, releaseQueue);
              }
            } finally {
              releaseQueue();
              // Backpressure accounting tracks real completion, never the
              // handback, so a client cannot queue unbounded concurrent work
              // by choosing a method that releases the queue early.
              connection.pendingRequestCount -= 1;
              connection.pendingRequestBytes -= pendingBytes;
            }
          })();
          void settled.catch(() => undefined);
          await queueHandback;
        };
        // Serialize dispatch per connection so handlers cannot race on shared state.
        connection.queue = connection.queue.then(handle, handle);
      }
      newline = connection.buffer.indexOf("\n");
    }
    if (Buffer.byteLength(connection.buffer, "utf8") > this.maxSocketFrameBytes) {
      console.error("Xerxes daemon dropping client: request exceeds the socket frame limit");
      connection.socket.destroy();
    }
  }

  /** Broadcast a global event to every local and remote daemon client. */
  broadcast(type: string, payload: JsonRpcPayload): void {
    for (const connection of this.connections) {
      this.emit(connection, type, payload);
    }
    this.websocketGateway?.broadcast(type, payload);
  }

  private async handleLine(
    connection: DaemonTransportConnection,
    line: string,
    releaseQueue: () => void = () => undefined,
  ): Promise<void> {
    let request: JsonRpcRequest;
    try {
      request = parseJsonRpcRequest(line);
    } catch (error) {
      connection.send(
        jsonRpcFailure(
          null,
          -32700,
          error instanceof JsonRpcParseError ? error.message : "Invalid JSON",
        ),
      );
      return;
    }
    // Hand the queue back before the slow part: a concurrent-safe handler
    // touches no session, so a request that waits on a provider endpoint must
    // not hold up everything else this client asked for.
    if (CONCURRENT_DISPATCH_METHODS.has(request.method)) {
      releaseQueue();
    }
    try {
      const result = await this.dispatch(connection, request);
      connection.send(jsonRpcSuccess(request.id, result));
    } catch (error) {
      connection.send(jsonRpcFailure(request.id, -32000, errorMessage(error)));
    }
  }

  private async dispatch(
    connection: DaemonTransportConnection,
    request: JsonRpcRequest,
  ): Promise<JsonRpcPayload> {
    const { method, params } = request;
    if (
      method.startsWith("task.") ||
      method === "submit" ||
      method === "list" ||
      method === "status"
    ) {
      return { ok: false, error: MIGRATED_ERROR };
    }
    if (method === "initialize") {
      return this.initialize(connection, params);
    }
    if (method === "session.open") {
      const key = requestedSessionKey(params, "default");
      const activeSession = this.runtime.sessionStatus(
        connection.activeSessionKey,
      );
      const cwd = resolveProjectDirectory(
        optionalString(params.project_dir) ||
          optionalString(activeSession?.metadata.project_root) ||
          activeSession?.cwd ||
          process.cwd(),
      );
      const requestedAgent = optionalString(params.agent_id)
        ?? this.runtime.sessionStatus(key)?.agentId
        ?? this.agentPresetRoster.defaultId;
      let preset: AgentPresetEntry;
      try {
        preset = this.agentPresetRoster.resolve(requestedAgent, cwd);
      } catch (error) {
        return { ok: false, code: "agent-preset-not-found", error: errorMessage(error) };
      }
      if (preset.broken) return { ok: false, code: "agent-preset-broken", error: preset.broken };
      const session = await this.runtime.openSession(key, preset.id, { cwd });
      connection.activeSessionKey = key;
      // Drain notices that settled while no client was attached (background
      // tasks above). At-most-once: the attaching client receives them here,
      // never again.
      for (const notice of takeSessionNotifications(session.metadata)) {
        this.emit(connection, "notification", {
          level: notice.level,
          message: notice.message,
        });
      }
      return {
        ok: true,
        session: sessionPayload(session, this.contextLimit(session.model), this.mcpStatusRecord()),
      };
    }
    if (method === "session.active_list") {
      return {
        ok: true,
        sessions: this.runtime
          .listSessions()
          .map((session) =>
            sessionPayload(session, this.contextLimit(session.model), this.mcpStatusRecord()),
          ),
      };
    }
    if (method === "session.list") {
      const limit = integerValue(params.limit);
      const kind = savedSessionKind(params.kind);
      if (params.kind !== undefined && kind === undefined) {
        return {
          ok: false,
          error: "session kind must be main, subagent, or all",
        };
      }
      const globalScope = optionalString(params.scope)?.toLowerCase() === "global";
      const projectScoped = globalScope
        ? false
        : booleanValue(params.project_scoped, true);
      const activeSession = this.runtime.sessionStatus(
        connection.activeSessionKey,
      );
      const activeProject = optionalString(activeSession?.metadata.project_root) || activeSession?.cwd;
      const projectDirectory = projectScoped
        ? optionalString(params.project_dir) || activeProject
        : undefined;
      if (projectScoped && !projectDirectory) {
        // Falling back to the daemon's cwd would silently scope history to an
        // unrelated project; say so instead of guessing.
        return {
          ok: false,
          error:
            "project-scoped session.list needs an active session or project_dir; pass scope \"global\" to list every project",
        };
      }
      const sessions = await this.runtime.listSavedSessions(limit, {
        ...(typeof params.include_subagents === "boolean"
          ? { includeSubagents: params.include_subagents }
          : {}),
        ...(kind ? { kind } : {}),
        ...(projectDirectory ? { projectDirectory } : {}),
      });
      return { ok: true, sessions: sessions.map(savedSessionPayload) };
    }
    if (method === "session.status") {
      const session = this.runtime.sessionStatus(
        sessionKey(connection, params),
      );
      return {
        ok: Boolean(session),
        session: session
          ? {
              ...sessionPayload(session, this.contextLimit(session.model), this.mcpStatusRecord()),
              // This is intentionally an identity only. The picker can use it
              // to select the exact stored profile without receiving the live
              // endpoint or credential that proved the match.
              profile_name: this.activeRuntimeProfileName(),
            }
          : null,
      };
    }
    if (method === "session.usage") {
      const session = this.runtime.sessionStatus(
        sessionKey(connection, params),
      );
      return session
        ? {
            ok: true,
            ...sessionUsagePayload(session, this.contextLimit(session.model)),
          }
        : { ok: false, error: "no active session" };
    }
    if (method === "session.title") {
      const key = sessionKey(connection, params);
      connection.activeSessionKey = key;
      return this.setSessionTitle(
        connection,
        this.runtime.sessionStatus(key),
        optionalString(params.title) ?? optionalString(params.value) ?? "",
        false,
      );
    }
    if (method === "changes.undo") {
      const key = sessionKey(connection, params);
      connection.activeSessionKey = key;
      return this.undoChanges(
        this.runtime.sessionStatus(key),
        optionalString(params.path) ?? "",
      );
    }
    if (method === "workspace.worktree") {
      const key = sessionKey(connection, params);
      connection.activeSessionKey = key;
      if (optionalString(params.action) !== "create") {
        return { ok: false, error: "unsupported workspace.worktree action" };
      }
      return this.createWorktree(
        this.runtime.sessionStatus(key),
        optionalString(params.name) ?? "",
      );
    }
    if (method === "session.goal") {
      const key = sessionKey(connection, params);
      connection.activeSessionKey = key;
      const session = this.runtime.sessionStatus(key);
      if (!session) {
        return { ok: false, error: "no active session" };
      }
      const result = runGoalCommand(
        session.metadata,
        session.id,
        optionalString(params.input) ?? optionalString(params.text) ?? "",
      );
      // A goal edit is durable state a crash must not lose, and it is the
      // thing that decides whether this session keeps working on its own.
      // Persist before answering.
      await this.runtime.flushSessions();
      return { ok: result.ok, text: result.text };
    }
    if (method === "session.compress") {
      connection.activeSessionKey = sessionKey(connection, params);
      return this.compactSession(connection, false);
    }
    if (method === "session.search") {
      const query = optionalString(params.query) ?? optionalString(params.text) ?? "";
      if (!query.trim()) {
        return { ok: false, error: "search query is required" };
      }
      await this.hydrateTranscriptSearch();
      const scopedSessionId = optionalString(params.session_id);
      const results = this.transcriptSearch.search(query, {
        limit: integerOption(params.limit) ?? SEARCH_RESULT_LIMIT,
        ...(scopedSessionId ? { sessionId: scopedSessionId } : {}),
      });
      return {
        ok: true,
        results: results.map(searchHitPayload),
        stats: searchStatsPayload(this.transcriptSearch.stats()),
      };
    }
    if (method === "session.save") {
      const key = sessionKey(connection, params);
      connection.activeSessionKey = key;
      return this.saveActiveSession(
        connection,
        this.runtime.sessionStatus(key),
        optionalString(params.title) ?? "",
        false,
      );
    }
    if (method === "session.undo") {
      const key = sessionKey(connection, params);
      connection.activeSessionKey = key;
      return this.undoLastTurn(
        connection,
        this.runtime.sessionStatus(key),
        false,
      );
    }
    if (method === "session.most_recent") {
      const activeSession = this.runtime.sessionStatus(connection.activeSessionKey);
      const projectDirectory =
        optionalString(params.project_dir) ||
        optionalString(activeSession?.metadata.project_root) ||
        activeSession?.cwd ||
        process.cwd();
      const mostRecent = (
        await this.runtime.listSavedSessions(1, {
          kind: "main",
          projectDirectory,
        })
      )[0];
      return {
        ok: true,
        session: mostRecent ? savedSessionPayload(mostRecent) : null,
      };
    }
    if (method === "session.delete") {
      const requested =
        optionalString(params.session_id) ??
        optionalString(params.id) ??
        optionalString(params.key);
      const active = requested
        ? this.runtime
            .listSessions()
            .find(
              (session) =>
                session.id === requested || session.sessionKey === requested,
            )
        : this.runtime.sessionStatus(sessionKey(connection, params));
      const sessionId = active?.id ?? requested;
      if (!sessionId) {
        return { ok: false, error: "session id is required" };
      }
      if (active?.activeTurnId) {
        return {
          ok: false,
          error: "cannot delete a session with an active turn",
        };
      }
      const remove = this.runtime.deleteSavedSession;
      if (!remove) {
        return {
          ok: false,
          error:
            "This native runtime does not expose persistent session deletion.",
        };
      }
      try {
        const deleted = await remove.call(this.runtime, sessionId);
        if (deleted) {
          this.forgetAcceptedSubmissions([sessionId, active?.sessionKey ?? ""]);
          this.endSessionLifetime(
            [sessionId, active?.sessionKey].filter(
              (value): value is string => Boolean(value),
            ),
          );
        }
        return deleted
          ? { ok: true, deleted: true, session_id: sessionId }
          : { ok: false, deleted: false, error: "saved session not found" };
      } catch (error) {
        return { ok: false, error: errorMessage(error) };
      }
    }
    if (method === "runtime.status") {
      return this.runtimeStatusPayload();
    }
    if (method === "runtime.update_status") {
      return this.updateStatus(connection, params);
    }
    if (method === "browser.manage") {
      return this.manageBrowser(params);
    }
    if (method === "terminal.list") {
      const ownerSessionId = this.terminalOwnerSessionId(connection, params);
      return { ok: true, terminals: this.terminalRegistry?.list(ownerSessionId) ?? [] };
    }
    if (method === "terminal.inspect") {
      return this.inspectTerminal(connection, params);
    }
    if (method === "terminal.control") {
      return this.controlTerminal(connection, params);
    }
    if (method === "channel.list") {
      return this.listChannels();
    }
    if (method === "channel.enable") {
      return this.enableChannel(params);
    }
    if (method === "channel.disable") {
      return this.disableChannel(params);
    }
    if (method === "runtime.reload") {
      this.runtime.reload(runtimeOverrides(params));
      const status = this.runtimeStatusWithChannels();
      const session = this.runtime.sessionStatus(
        sessionKey(connection, params),
      );
      if (session) {
        this.emitStatus(connection, session);
      }
      return { ...status, ok: true };
    }
    if (method === "turn.submit" || method === "prompt") {
      const rawText =
        typeof params.text === "string"
          ? params.text
          : typeof params.user_input === "string"
            ? params.user_input
            : "";
      const key = sessionKey(connection, params);
      connection.activeSessionKey = key;
      await this.runtime.openSession(key);
      const intercepted = await this.consumeSkillCreateInput(
        connection,
        key,
        rawText,
      );
      if (intercepted) {
        return intercepted;
      }
      const text = rawText.trim();
      if (!text) {
        return { ok: false, error: "text is required" };
      }
      // Optional v35-compatible image attachments. Validation failures reject
      // the submit outright instead of silently dropping or truncating data.
      let images: readonly TurnImage[];
      try {
        images = validateTurnImages(params.images);
      } catch (error) {
        return { ok: false, error: errorMessage(error) };
      }
      const session = this.runtime.sessionStatus(key);
      requireConfiguredModel(
        session?.model || stringValue(this.runtime.status().model),
      );
      const displayText =
        (typeof params.display_text === "string"
          ? params.display_text.trim()
          : "") || text;
      const submissionId = optionalString(params.submission_id)?.trim();
      const submissionKey = submissionId ? `${key}\u0000${submissionId}` : undefined;
      if (submissionKey && this.acceptedSubmissionIds.has(submissionKey)) {
        return { ok: true, duplicate: true };
      }
      if (submissionKey) {
        this.rememberAcceptedSubmission(submissionKey);
      }
      void this.submitTrackedTurn(
        key,
        text,
        (event) => this.emit(connection, event.type, event.payload),
        connection,
        { displayText, ...(images.length ? { images } : {}) },
      ).catch((error) =>
        this.emit(connection, "notification", {
          level: "error",
          message: errorMessage(error),
        }),
      );
      return { ok: true };
    }
    if (method === "turn.background") {
      const rawText =
        typeof params.text === "string"
          ? params.text
          : typeof params.user_input === "string"
            ? params.user_input
            : "";
      const text = rawText.trim();
      if (!text) {
        return { ok: false, error: "text is required" };
      }
      // A background prompt runs in its own session so the foreground stays
      // usable while it works. Deliberately not touching activeSessionKey:
      // that is what makes this different from turn.submit, which would park
      // the user inside the background conversation.
      const parentKey = sessionKey(connection, params);
      const backgroundKey = `bg-${newConnectionKey()}`;
      const parent = this.runtime.sessionStatus(parentKey);
      await this.runtime.openSession(backgroundKey, undefined, {
        // Inherit the parent's model so a background prompt is answered by
        // the model the user is actually working with, not the daemon default.
        ...(parent?.model ? { model: parent.model } : {}),
        ...(parent?.cwd ? { cwd: parent.cwd } : {}),
      });
      const background = this.runtime.sessionStatus(backgroundKey);
      requireConfiguredModel(
        background?.model || stringValue(this.runtime.status().model),
      );
      const taskId = background?.id ?? backgroundKey;
      void this.submitTrackedTurn(
        backgroundKey,
        text,
        (event) =>
          this.emit(connection, event.type, {
            ...event.payload,
            background_task_id: taskId,
            // Route every background delta to its own live session. The TUI's
            // active-session filter then prevents it from mutating foreground
            // streaming, tools, usage, or approval state.
            session_id: taskId,
          }),
        connection,
        { displayText: text },
      )
        .then(() => {
          // Queue before the live emit: the notice must survive a disconnect
          // even if this connection is already gone when the task settles.
          const parentSession = this.runtime.sessionStatus(parentKey);
          if (parentSession) {
            queueSessionNotification(parentSession.metadata, {
              at: Date.now(),
              level: "info",
              message: `Background task ${taskId} finished.`,
            });
          }
          this.emit(connection, "background.complete", {
            task_id: taskId,
            text: "finished",
          });
          this.emit(connection, "notification", {
            level: "info",
            message: `Background task ${taskId} finished.`,
          });
        })
        .catch((error) => {
          const message = errorMessage(error);
          const parentSession = this.runtime.sessionStatus(parentKey);
          if (parentSession) {
            queueSessionNotification(parentSession.metadata, {
              at: Date.now(),
              level: "error",
              message: `Background task ${taskId} failed: ${message}`,
            });
          }
          this.emit(connection, "background.complete", {
            task_id: taskId,
            text: `failed: ${message}`,
          });
          this.emit(connection, "notification", {
            level: "error",
            message: `Background task ${taskId} failed: ${message}`,
          });
        });
      return { ok: true, task_id: taskId, session_key: backgroundKey };
    }
    if (method === "turn.cancel" || method === "cancel") {
      return { ok: this.cancelTrackedTurn(sessionKey(connection, params)) };
    }
    if (method === "cancel_all") {
      return { ok: true, cancelled: this.runtime.cancelAllTurns() };
    }
    if (method === "subagent.retry") {
      const task =
        optionalString(params.task) ??
        optionalString(params.agent) ??
        optionalString(params.name);
      if (!task) {
        return {
          ok: false,
          error: "subagent.retry requires a task id or stable name",
        };
      }
      if (!this.runtime.retrySubagent) {
        return {
          ok: false,
          error: "subagent retry is not available on this daemon runtime",
        };
      }
      const message = optionalString(params.message);
      return this.runtime.retrySubagent({
        sessionKey: sessionKey(connection, params),
        task,
        ...(message ? { message } : {}),
      });
    }
    if (method === "turn.steer" || method === "steer") {
      const content =
        optionalString(params.content) ?? optionalString(params.text) ?? "";
      const key = sessionKey(connection, params);
      const session = this.runtime.sessionStatus(key);
      const processed = session
        ? await processAtMentions(content, session.cwd)
        : { enhancedMessage: content, mentionedFiles: [] };
      const ok = this.runtime.steerTurn(key, processed.enhancedMessage);
      if (ok) {
        this.emit(connection, "steer_input", {
          content,
          ...(processed.mentionedFiles.length
            ? { mentioned_files: processed.mentionedFiles }
            : {}),
        });
      }
      return ok
        ? { ok: true }
        : { ok: false, error: "No session or steering text to apply" };
    }
    if (method === "slash") {
      return this.handleSlash(connection, optionalString(params.command) ?? "");
    }
    if (method === "commands.catalog") {
      return this.commandCatalog();
    }
    if (method === "complete") {
      return this.complete(connection, params);
    }
    if (method === "set_plan_mode") {
      const enabled = booleanValue(
        params.enabled,
        booleanValue(params.plan_mode, false),
      );
      const mode = optionalString(params.mode) ?? (enabled ? "plan" : "code");
      return this.setMode(connection, mode, enabled, sessionKey(connection, params));
    }
    if (method === "set_mode") {
      return this.setMode(
        connection,
        optionalString(params.mode) ?? "code",
        undefined,
        sessionKey(connection, params),
      );
    }
    if (method === "set_model") {
      const model = optionalString(params.model);
      if (!model) {
        return { ok: false, error: "model id is required" };
      }
      return this.setModel(connection, model, sessionKey(connection, params));
    }
    if (method === "set_reasoning") {
      const effort = optionalString(params.reasoning_effort)
        ?? optionalString(params.effort);
      if (!effort) {
        return { ok: false, error: "reasoning effort is required" };
      }
      return this.setReasoning(
        connection,
        effort,
        sessionKey(connection, params),
      );
    }
    if (method === "permission_response") {
      return this.permissionResponse(connection, params);
    }
    if (method === "question_response") {
      return this.questionResponse(connection, params);
    }
    if (method === "fetch_models") {
      return this.fetchModels(params);
    }
    if (method === "provider_model_override") {
      return this.updateProviderModelOverride(connection, params);
    }
    if (method === "provider_models") {
      const profileName = optionalString(params.profile_name) ?? optionalString(params.name);
      if (!profileName) {
        return { ok: false, error: "provider_models requires profile_name", models: [] };
      }
      const profile = this.profileStore.get(profileName);
      if (!profile) {
        return { ok: false, error: `No provider profile named ${profileName}`, models: [] };
      }
      const result = await this.fetchModels({ profile_name: profileName });
      return {
        ...result,
        profile: profileName,
        provider: profile.provider,
        configured_model: profile.model,
      };
    }
    if (method === "context_breakdown") {
      return this.contextBreakdown(connection, params);
    }
    if (method === "reasoning_levels") {
      // Resolve the ladder against the requested session's model. The daemon
      // default can differ after another tab/provider changes configuration.
      const activeSession = this.runtime.sessionStatus(sessionKey(connection, params));
      const set = await this.reasoningLevels(activeSession?.model);
      const selectable = selectableEfforts(set);
      // Session-first, like configureReasoning: /thinking pins the effort per
      // session, so reading only the daemon-wide value would report an effort
      // this session is not running at (the picker looked "stuck on off").
      return {
        ok: true,
        current:
          activeSession?.reasoningEffort
          || stringValue(this.runtime.status().reasoning_effort)
          || REASONING_OFF,
        default: set.defaultEffort ?? null,
        // An `inherent` provider yields no selectable efforts at all, so the
        // panel shows the note rather than a menu that cannot change anything.
        levels: selectable.map((effort) =>
          effort === REASONING_OFF
            ? {
                effort,
                description: "No extended reasoning; fastest replies",
              }
            : {
                effort,
                ...(set.levels.find((level) => level.effort === effort)
                  ?.description === undefined
                  ? {}
                  : {
                      description: set.levels.find(
                        (level) => level.effort === effort,
                      )?.description,
                    }),
              },
        ),
        note: reasoningShapeNote(set),
        shape: set.shape,
        source: set.source,
      };
    }
    if (method === "skill_suggestions") {
      const session = this.runtime.sessionStatus(sessionKey(connection, params));
      if (!session) return { ok: false, error: "no active session", suggestions: [] };
      return {
        ok: true,
        suggestions: skillSuggestionValues(session.metadata).map((suggestion) => ({
          skill_name: suggestion.skillName,
          description: suggestion.description,
          version: suggestion.version,
          source_path: suggestion.sourcePath,
          tool_count: suggestion.toolCount,
          unique_tools: [...suggestion.uniqueTools],
        })),
      };
    }
    if (method === "provider_list") {
      return {
        ok: true,
        profiles: this.profileStore.list().map(profilePayload),
      };
    }
    if (method === "provider_types") {
      // The registry IS the adapter list an add/edit form may offer — names,
      // default endpoints, and the env var each type falls back to. Catalog
      // facts only; keys never cross the wire.
      return { ok: true, types: providerTypePayloads() };
    }
    if (method === "provider_save") {
      return this.saveProvider(connection, params);
    }
    if (method === "provider_select") {
      return this.selectProvider(connection, optionalString(params.name) ?? "");
    }
    if (method === "provider_delete") {
      return this.deleteProvider(connection, optionalString(params.name) ?? "");
    }
    if (method.startsWith("agentPreset.")) {
      return this.agentPresetRpc(connection, method, params);
    }
    if (method.startsWith("forge.")) {
      return this.forgeRpc(connection, method, params);
    }
    if (method === "creator_trace") {
      const session = this.runtime.sessionStatus(sessionKey(connection, params));
      return session
        ? { ok: true, trace: creatorTraceValues(session.metadata).map(creatorTracePayload) }
        : { ok: false, error: "no active session", trace: [] };
    }
    if (method === "shutdown") {
      queueMicrotask(() => {
        const shutdown = this.onShutdown ? this.onShutdown() : this.stop();
        void Promise.resolve(shutdown).catch((error) =>
          this.broadcast("notification", {
            level: "error",
            message: `Native daemon shutdown failed: ${errorMessage(error)}`,
          }),
        );
      });
      return { ok: true };
    }
    if (method === "daemon.wipe_memory") {
      return this.wipeMemory(connection);
    }
    if (method === "daemon.wipe_history") {
      return this.wipeHistory(connection);
    }
    return { ok: false, error: `Unknown method: ${method}` };
  }

  private listChannels(): JsonRpcPayload {
    const manager = this.channelManager;
    if (!manager) {
      return {
        ok: false,
        error: "channel manager is not configured",
        channels: [],
        channels_available: false,
        channels_configured: false,
      };
    }
    const data = this.channelStatusData();
    return {
      ok: true,
      channels: data.channels,
      channels_available: data.available,
      channels_configured: data.configured,
    };
  }

  private async enableChannel(params: JsonRpcPayload): Promise<JsonRpcPayload> {
    const name =
      optionalString(params.name) ?? optionalString(params.channel) ?? "";
    if (!name) {
      return { ok: false, error: "channel name is required" };
    }
    const manager = this.channelManager;
    if (!manager) {
      return { ok: false, error: "channel manager is not configured" };
    }
    try {
      const channel = await manager.enable(name);
      const data = this.channelStatusData();
      this.broadcast("channel_status", channelStatusEventPayload(data));
      return {
        ok: true,
        channel: channelStatusPayload(channel),
        channels: data.channels,
      };
    } catch (error) {
      return { ok: false, error: errorMessage(error) };
    }
  }

  private async disableChannel(
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    const name =
      optionalString(params.name) ?? optionalString(params.channel) ?? "";
    if (!name) {
      return { ok: false, error: "channel name is required" };
    }
    const manager = this.channelManager;
    if (!manager) {
      return { ok: false, error: "channel manager is not configured" };
    }
    try {
      const channel = await manager.disable(name);
      const data = this.channelStatusData();
      this.broadcast("channel_status", channelStatusEventPayload(data));
      return {
        ok: true,
        channel: channelStatusPayload(channel),
        channels: data.channels,
      };
    } catch (error) {
      return { ok: false, error: errorMessage(error) };
    }
  }

  private channelStatusData(): ChannelStatusData {
    const manager = this.channelManager;
    if (!manager) {
      return { available: false, configured: false, channels: [] };
    }
    return {
      available: true,
      configured: manager.hasConfiguredChannels,
      channels: manager.list().map(channelStatusPayload),
    };
  }

  private async complete(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    const text = stringValue(params.text);
    const stripped = text.trim();
    // `/skill <partial>` completes native skill names — and `name:subcommand`
    // references — from the same registry `/skills` lists, so every client
    // hints skills without re-implementing discovery. Matched against the raw
    // text: a trailing space is the signal the user left the command word and
    // wants argument completion, while a bare `/skill` still completes as a
    // command.
    const skillArg = /^\/skill\s+(\S*)$/.exec(text);
    if (skillArg) {
      const session = this.runtime.sessionStatus(sessionKey(connection, params));
      return {
        ok: true,
        kind: "slash",
        completions: await this.completeSkillReference(
          skillArg[1] ?? "",
          session,
        ),
      };
    }
    if (stripped.startsWith("/") && !/\s/.test(stripped)) {
      // A single token is both a canonical command and a skill shorthand —
      // `/review` invokes the review skill without spelling `/skill review`.
      const session = this.runtime.sessionStatus(sessionKey(connection, params));
      return {
        ok: true,
        kind: "slash",
        completions: await this.completeSlashAndSkills(stripped, session),
      };
    }
    const session = this.runtime.sessionStatus(sessionKey(connection, params));
    const cwd = session?.cwd ?? process.cwd();
    return {
      ok: true,
      kind: "path",
      completions: await completePath(text, cwd),
    };
  }

  /**
   * Single-token completions: canonical commands first (they own their
   * names), then skill shorthands — `/review` is the same invocation as
   * `/skill review` — ranked prefix, then name substring, then description
   * substring, so `/bug` and even `/bounty` find `bug-bounty-hunter`.
   */
  private async completeSlashAndSkills(
    stripped: string,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload[]> {
    const commands = this.completeSlash(stripped);
    const reserved = new Set<string>();
    const prefix = slashCompletionPrefix(stripped);
    for (const command of DAEMON_SLASH_COMMANDS) {
      if (
        command.name.startsWith(prefix) ||
        command.aliases.some((alias) => alias.startsWith(prefix))
      ) {
        reserved.add(command.name);
        for (const alias of command.aliases) reserved.add(alias);
      }
    }
    // First keystroke of a fresh daemon must already hint skills: refresh
    // before ranking instead of trusting whatever a previous call loaded.
    await this.refreshSkills(session);
    const skills = (await this.completeSkillEntries(stripped.slice(1)))
      .filter((entry) => !reserved.has(String(entry.label ?? "").split(":", 1)[0] ?? ""));
    return [...commands, ...skills].slice(0, 200);
  }

  /**
   * Skill-reference completions for `/skill <prefix>` — trusted sources only.
   * Matching is ranked: exact prefix, then substring in the reference, then
   * substring in the description — so `/skill bounty` still finds a skill
   * named `read-project-and-hunt-bugs` whose description says "bug bounty".
   */
  private async completeSkillReference(
    prefix: string,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload[]> {
    await this.refreshSkills(session);
    return this.rankSkillReferences(prefix).map((entry) => ({
      value: `/skill ${entry.reference} `,
      label: entry.reference,
      meta: entry.description,
    }));
  }

  /** Skill shorthands as bare `/<reference> ` completions. */
  private async completeSkillEntries(
    prefix: string,
  ): Promise<JsonRpcPayload[]> {
    return this.rankSkillReferences(prefix).map((entry) => ({
      value: `/${entry.reference} `,
      label: entry.reference,
      meta: entry.description,
    }));
  }

  /**
   * Rank trusted, platform-valid skill references (name and
   * `name:subcommand`) for a partial token: exact prefix first, then
   * substring in the reference, then substring in the description —
   * `/bounty` finds `bug-bounty-hunter` through its description.
   */
  private rankSkillReferences(
    prefix: string,
  ): Array<{ reference: string; description: string }> {
    const wanted = prefix.toLowerCase();
    const starts: Array<{ reference: string; description: string }> = [];
    const nameHits: Array<{ reference: string; description: string }> = [];
    const descriptionHits: Array<{ reference: string; description: string }> = [];
    for (const skill of this.skillRegistry.all()) {
      if (!skillMatchesPlatform(skill)) continue;
      const description = skill.metadata.description || "No description";
      const references = [
        skill.metadata.name,
        ...skill.metadata.subcommands.map(
          (subcommand) => `${skill.metadata.name}:${subcommand}`,
        ),
      ];
      for (const reference of references) {
        const item = { reference, description };
        const lower = reference.toLowerCase();
        if (!wanted) starts.push(item);
        else if (lower.startsWith(wanted)) starts.push(item);
        else if (lower.includes(wanted)) nameHits.push(item);
        else if (
          reference === skill.metadata.name &&
          description.toLowerCase().includes(wanted)
        ) {
          descriptionHits.push(item);
        }
      }
    }
    const byReference = (
      left: { reference: string },
      right: { reference: string },
    ): number => left.reference.localeCompare(right.reference);
    return [...starts.sort(byReference), ...nameHits, ...descriptionHits].slice(
      0,
      200,
    );
  }

  private completeSlash(text: string): JsonRpcPayload[] {
    const prefix = slashCompletionPrefix(text);
    const pluginCommands: DaemonSlashCommand[] = this.slashPluginRegistry
      .list()
      .map((plugin) => ({
        name: plugin.command.name,
        aliases: plugin.command.aliases,
        category: plugin.command.category,
        description: plugin.command.description,
      }));
    return [...DAEMON_SLASH_COMMANDS, ...pluginCommands]
      .filter(
        (command) =>
          command.name.startsWith(prefix) ||
          command.aliases.some((alias) => alias.startsWith(prefix)),
      )
      .slice(0, 50)
      .map((command) => ({
        value: `/${command.name}`,
        label: command.name,
        meta: command.description,
        // The UI ranks the bare-slash menu by category so a plain "/" surfaces
        // the commands people reach for instead of an alphabetical wall.
        category: command.category,
      }));
  }

  private commandCatalog(): JsonRpcPayload {
    const pluginCommands = this.slashPluginRegistry
      .list()
      .map((plugin) => plugin.command);
    const pairs = [
      ...daemonCommandPairs(DAEMON_SLASH_COMMANDS),
      ...pluginCommands.map(
        (command) =>
          [`/${command.name}`, command.description] as [string, string],
      ),
    ];
    const canon: Record<string, string> = {};
    for (const command of DAEMON_SLASH_COMMANDS) {
      canon[`/${command.name}`] = `/${command.name}`;
      for (const alias of command.aliases) {
        canon[`/${alias}`] = `/${command.name}`;
      }
    }
    for (const command of pluginCommands) {
      canon[`/${command.name}`] = `/${command.name}`;
      for (const alias of command.aliases) {
        canon[`/${alias}`] = `/${command.name}`;
      }
    }
    return {
      ok: true,
      canon,
      categories: daemonCommandCategories(),
      pairs,
      skill_count: this.skillRegistry.all().length,
      sub: {},
    };
  }

  private async openProviderFlow(
    connection: DaemonTransportConnection,
  ): Promise<JsonRpcPayload> {
    this.cancelSkillCreate(connection);
    this.cancelProviderFlow(connection);
    const flow = new ProviderProfileFlow({
      profileStore: this.profileStore,
      ...(this.providerModelDiscovery
        ? { modelDiscovery: this.providerModelDiscovery }
        : {}),
    });
    this.providerFlows.set(connection, flow);
    return this.applyProviderFlowTransition(
      connection,
      flow,
      await flow.start(),
    );
  }

  private async applyProviderFlowTransition(
    connection: DaemonTransportConnection,
    flow: ProviderProfileFlow,
    transition: ProviderFlowTransition,
  ): Promise<JsonRpcPayload> {
    if (transition.notice) {
      this.emitSlash(
        connection,
        transition.notice.body,
        transition.notice.severity,
      );
    }
    if (transition.reload) {
      this.runtime.reload(profileOverrides(this.profileStore.active()));
      await this.emitProviderInit(connection);
    }
    if (transition.prompt) {
      this.emitProviderFlowPrompt(connection, transition.prompt);
    }
    if (transition.finished && this.providerFlows.get(connection) === flow) {
      this.providerFlows.delete(connection);
    }
    return {
      ok: true,
      ...(transition.finished ? { completed: true } : {}),
    };
  }

  private emitProviderFlowPrompt(
    connection: DaemonTransportConnection,
    prompt: ProviderFlowPrompt,
  ): void {
    const question = prompt.question;
    const placeholder = question.placeholder?.trim();
    this.emit(connection, "question_request", {
      flow: "provider",
      id: prompt.requestId,
      tool_call_id: question.toolCallId ?? "",
      questions: [
        {
          id: question.questionId ?? "answer",
          question: question.question,
          options: [...(question.options ?? [])],
          allow_free_form: question.allowFreeform ?? true,
          ...(placeholder ? { placeholder } : {}),
        },
      ],
    });
  }

  private cancelProviderFlow(connection: DaemonTransportConnection): void {
    const flow = this.providerFlows.get(connection);
    if (!flow) {
      return;
    }
    const requestId = flow.activeRequestId;
    if (requestId) {
      this.questionOwners.delete(requestId);
    }
    flow.cancel();
    this.providerFlows.delete(connection);
  }

  private async openSkillCreate(
    connection: DaemonTransportConnection,
    rawName: string,
  ): Promise<JsonRpcPayload> {
    this.cancelSkillCreate(connection);
    const flow = new SkillCreateFlow({ skillsDirectory: this.skillDirectory });
    this.skillCreates.set(connection, flow);
    return this.applySkillCreateTransition(
      connection,
      await flow.start(rawName, connection.activeSessionKey),
      false,
    );
  }

  private async consumeSkillCreateInput(
    connection: DaemonTransportConnection,
    sessionKey: string,
    rawText: string,
  ): Promise<JsonRpcPayload | undefined> {
    const flow = this.skillCreates.get(connection);
    if (!flow || !flow.ownsSession(sessionKey)) {
      return undefined;
    }
    const transition = await flow.answer(sessionKey, rawText);
    if (!transition) {
      return undefined;
    }
    return this.applySkillCreateTransition(connection, transition, true);
  }

  private applySkillCreateTransition(
    connection: DaemonTransportConnection,
    transition: SkillCreateTransition,
    consumedPrompt: boolean,
  ): JsonRpcPayload {
    if (transition.kind === "prompt") {
      this.emitSlash(connection, transition.message);
    } else if (transition.kind === "cancelled") {
      this.skillCreates.delete(connection);
      this.emitSlash(connection, transition.message);
    } else {
      this.skillCreates.delete(connection);
      this.emitSlash(connection, transition.draft.announcement);
      const sessionKey = connection.activeSessionKey;
      queueMicrotask(() => {
        void this.submitTrackedTurn(
          sessionKey,
          transition.draft.prompt,
          (event) => this.emit(connection, event.type, event.payload),
          connection,
        ).catch((error) =>
          this.emit(connection, "notification", {
            level: "error",
            message: errorMessage(error),
          }),
        );
      });
    }
    if (consumedPrompt) {
      this.emit(connection, "turn_begin", {});
      this.emit(connection, "turn_end", {});
    }
    return {
      ok: true,
      ...(consumedPrompt ? { consumed_for: "skill-create" } : {}),
      ...(transition.kind === "cancelled" ? { cancelled: true } : {}),
    };
  }

  private cancelSkillCreate(connection: DaemonTransportConnection): void {
    this.skillCreates.delete(connection);
  }

  private async deleteProvider(
    connection: DaemonTransportConnection,
    name: string,
  ): Promise<JsonRpcPayload> {
    this.cancelProviderFlow(connection);
    if (!name) {
      return { ok: false, error: "provider name is required" };
    }
    const removed = this.profileStore.delete(name);
    if (!removed) {
      return { ok: false, error: `No provider profile named ${name}` };
    }
    const active = this.profileStore.active();
    this.runtime.reload(profileOverrides(active));
    await this.emitProviderInit(connection);
    return { ok: true };
  }

  private async emitProviderInit(
    connection: DaemonTransportConnection,
  ): Promise<void> {
    const session = this.runtime.sessionStatus(connection.activeSessionKey);
    if (!session) {
      return;
    }
    this.emitInitDone(connection, session);
    this.emitStatus(connection, session);
  }

  private emitInitDone(
    connection: DaemonTransportConnection,
    session: DaemonSession,
  ): void {
    const model = session.model || stringValue(this.runtime.status().model);
    this.emit(
      connection,
      "init_done",
      initPayload(
        session,
        model,
        session.reasoningEffort
          || stringValue(this.runtime.status().reasoning_effort)
          || "off",
        runtimePermissionMode(
          session.permissionMode || this.runtime.status().permission_mode,
        ),
        this.contextLimit(model),
      ),
    );
  }

  private emitSlash(
    connection: DaemonTransportConnection,
    body: string,
    severity: "error" | "info" | "warning" = "info",
  ): void {
    this.emit(connection, "notification", {
      id: newConnectionKey(),
      category: "slash",
      type: "result",
      severity,
      title: "",
      body,
      payload: {},
    });
  }

  /**
   * `!<cmd>` shell mode. Runs the command in the user's login shell with the
   * project directory as cwd, mirroring Claude Code's shell mode: bounded
   * runtime and output, non-zero exits surface the code, and the raw
   * stdout/stderr ride the slash notification body so every client renders
   * the same thing. This is deliberately NOT permission-gated — the user
   * typed the command themselves; it is exactly what they'd get in their own
   * terminal.
   */
  private async handleShellCommand(
    connection: DaemonTransportConnection,
    shellCommand: string,
  ): Promise<JsonRpcPayload> {
    if (!shellCommand) {
      this.emitSlash(connection, "usage: !<command>", "warning");
      return { ok: false, error: "empty shell command" };
    }

    const SHELL_TIMEOUT_MS = 120_000;
    const SHELL_OUTPUT_CAP = 30_000;
    const cwd = this.projectDirectory ?? process.cwd();
    const shell = process.platform === "win32" ? "cmd.exe" : "/bin/sh";
    const shellArgs = process.platform === "win32" ? ["/d", "/s", "/c", shellCommand] : ["-c", shellCommand];

    let code = 0;
    let stdout = "";
    let stderr = "";
    let timedOut = false;
    try {
      const proc = Bun.spawn([shell, ...shellArgs], {
        cwd,
        stdin: "ignore",
        stdout: "pipe",
        stderr: "pipe",
      });
      const killer = setTimeout(() => {
        timedOut = true;
        proc.kill();
      }, SHELL_TIMEOUT_MS);
      const [out, err, exit] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
      ]);
      clearTimeout(killer);
      code = timedOut ? 124 : exit;
      stdout = out;
      stderr = err;
    } catch (error) {
      this.emitSlash(connection, `shell failed: ${errorMessage(error)}`, "error");
      return { ok: false, error: errorMessage(error) };
    }

    const clip = (value: string) =>
      value.length > SHELL_OUTPUT_CAP
        ? value.slice(0, SHELL_OUTPUT_CAP) + `\n… (truncated, ${value.length} chars total)`
        : value;
    const combined = [clip(stdout), clip(stderr)].filter(Boolean).join("\n").trimEnd();
    const suffix = timedOut ? `\n(exited: timed out after ${SHELL_TIMEOUT_MS / 1000}s)` : code !== 0 ? `\n(exit ${code})` : "";
    const body = combined ? combined + suffix : suffix.trim() || "(no output)";
    this.emitSlash(connection, body, code === 0 ? "info" : "warning");
    return { code, ok: code === 0, stderr: clip(stderr), stdout: clip(stdout) };
  }

  /**
   * `#<note>` quick memory (Claude Code's `#` prefix): appends one line to
   * the project MEMORY.md through the same store the memory tools use, so
   * notes are immediately visible to future turns without a model round.
   */
  private async handleMemoryNote(
    connection: DaemonTransportConnection,
    note: string,
  ): Promise<JsonRpcPayload> {
    if (!note) {
      this.emitSlash(connection, "usage: #<note to remember>", "warning");
      return { ok: false, error: "empty memory note" };
    }
    try {
      const store = new WorkspaceMemoryStore(
        this.projectDirectory ? { workspaceRoot: this.projectDirectory } : {},
      );
      const result = await store.add("memory", note);
      if ("ok" in result && result.ok) {
        this.emitSlash(connection, `remembered (MEMORY.md #${result.id}): ${result.content}`);
        return { id: result.id, ok: true };
      }
      const message = "error" in result ? String(result.error) : "memory write failed";
      this.emitSlash(connection, `memory note failed: ${message}`, "error");
      return { ok: false, error: message };
    } catch (error) {
      this.emitSlash(connection, `memory note failed: ${errorMessage(error)}`, "error");
      return { ok: false, error: errorMessage(error) };
    }
  }

  private emitCompactionLog(
    connection: DaemonTransportConnection,
    body: string,
    tokensBefore: number,
    tokensAfter: number,
    automatic: boolean,
  ): void {
    this.emit(connection, "notification", {
      id: newConnectionKey(),
      category: "history",
      type: "compaction",
      severity: "info",
      title: "Context compacted",
      body,
      payload: {
        automatic,
        tokens_before: tokensBefore,
        tokens_after: tokensAfter,
      },
    });
  }

  private emitStatus(
    connection: DaemonTransportConnection,
    session: DaemonSession,
  ): void {
    const model = session.model || stringValue(this.runtime.status().model);
    this.emit(
      connection,
      "status_update",
      statusUpdatePayload(
        session,
        model,
        this.contextLimit(model),
        this.channelStatusData(),
        session.reasoningEffort
        || stringValue(this.runtime.status().reasoning_effort)
        || "off",
        runtimePermissionMode(
          session.permissionMode ?? this.runtime.status().permission_mode,
        ),
        this.mcpStatusRecord(),
      ),
    );
  }

  /**
   * Announce a mode change that the MODEL made, not the user.
   *
   * The human path (`set_mode` / `set_plan_mode`) already ends in emitStatus,
   * which is how the TUI learns its footer changed. A transition driven by
   * SetInteractionModeTool went through the runtime instead and told nobody,
   * so the session really did leave plan mode while every client kept
   * rendering — and gating on — the old one.
   *
   * Scoped to the connections actually attached to that session rather than
   * broadcast: a background session changing mode must not repaint the mode
   * of whatever session the user happens to be looking at. A client that
   * attaches later reads the current mode from the session payload anyway.
   */
  notifySessionModeChanged(sessionId: string): void {
    const target = sessionId.trim();
    if (!target) return;
    for (const connection of this.connections) {
      const session = this.runtime.sessionStatus(connection.activeSessionKey);
      if (!session || session.id !== target) continue;
      this.emitStatus(connection, session);
    }
  }

  private async agentPresetRpc(
    connection: DaemonTransportConnection,
    method: string,
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    const key = sessionKey(connection, params);
    const session = this.runtime.sessionStatus(key);
    const cwd = session?.cwd ?? this.projectDirectory ?? process.cwd();
    const id = optionalString(params.agent_preset) ?? optionalString(params.id) ?? "";
    try {
      if (method === "agentPreset.list") {
        return {
          ok: true,
          presets: this.agentPresetRoster.list(cwd).map(agentPresetPayload),
          default_id: this.agentPresetRoster.defaultId,
          authorable: true,
          has_document: true,
        };
      }
      if (method === "agentPreset.read") {
        const preset = this.agentPresetRoster.read(id, cwd);
        return { ok: true, preset: agentPresetPayload(preset), content: preset.content };
      }
      if (method === "agentPreset.copy") {
        const from = optionalString(params.from) ?? "";
        const preset = this.agentPresetRoster.copy(from, id, optionalString(params.name), cwd);
        this.runtime.reload({});
        return { ok: true, preset: agentPresetPayload(preset), path: preset.path ?? "" };
      }
      if (method === "agentPreset.write") {
        const content = optionalString(params.content) ?? "";
        const preset = this.agentPresetRoster.write(id, content, cwd);
        this.runtime.reload({});
        return { ok: true, preset: agentPresetPayload(preset) };
      }
      if (method === "agentPreset.remove") {
        this.agentPresetRoster.remove(id, cwd);
        this.runtime.reload({});
        return { ok: true, removed: id, default_id: this.agentPresetRoster.defaultId };
      }
      if (method === "agentPreset.setDefault") {
        const preset = this.agentPresetRoster.setDefault(id, cwd);
        return { ok: true, preset: agentPresetPayload(preset), default_id: preset.id };
      }
      if (method === "agentPreset.openDocument") {
        const preset = this.agentPresetRoster.resolve(id, cwd);
        if (!preset.manageable || !preset.path) {
          return { ok: false, code: "agent-preset-not-writable", error: "shipped agent presets are read-only" };
        }
        return { ok: true, opened: false, path: dirname(preset.path) };
      }
      if (method === "agentPreset.select") {
        const select = async (): Promise<JsonRpcPayload> => {
          const current = this.runtime.sessionStatus(key);
          if (!current) return { ok: false, code: "session-not-found", error: "no active session" };
          const preset = this.agentPresetRoster.resolve(id, current.cwd);
          if (preset.broken) {
            return { ok: false, code: "agent-preset-broken", error: preset.broken };
          }
          const selected = this.runtime.selectSessionAgent
            ? await this.runtime.selectSessionAgent(key, preset.id)
            : await this.runtime.openSession(key, preset.id, { cwd: current.cwd });
          if (!selected) return { ok: false, code: "session-not-found", error: "no active session" };
          this.emitStatus(connection, selected);
          this.emit(connection, "agent_preset_selected", {
            session_id: selected.id,
            agent_preset: selected.agentId,
          });
          return { ok: true, agent_preset: selected.agentId };
        };
        const prior = this.agentPresetSwitches.get(key) ?? Promise.resolve();
        const operation = prior.then(select);
        const settled = operation.then(() => undefined, () => undefined);
        this.agentPresetSwitches.set(key, settled);
        try {
          return await operation;
        } finally {
          if (this.agentPresetSwitches.get(key) === settled) this.agentPresetSwitches.delete(key);
        }
      }
      return { ok: false, code: "method-not-found", error: `Unknown agent preset method: ${method}` };
    } catch (error) {
      const message = errorMessage(error);
      return {
        ok: false,
        code: message.includes("already started") ? "agent-preset-locked" : "agent-preset-error",
        error: message,
      };
    }
  }

  private async forgeRpc(
    connection: DaemonTransportConnection,
    method: string,
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    const session = this.runtime.sessionStatus(sessionKey(connection, params));
    const name = optionalString(params.name) ?? "";
    const version = optionalString(params.version);
    let result: JsonRpcPayload;
    try {
      if (method === "forge.list") {
        result = { ok: true, packages: this.declarativeForge.list().map(pkg => forgePackagePayload(pkg)) };
      } else if (method === "forge.inspect") {
        const pkg = this.declarativeForge.inspect(name, version);
        result = pkg
          ? { ok: true, package: forgePackagePayload(pkg, true) }
          : { ok: false, error: "forged package not found" };
      } else if (method === "forge.run") {
        result = {
          ok: true,
          ...this.declarativeForge.run(
            name,
            version,
            isRecord(params.input) ? params.input : {},
          ),
        };
      } else if (method === "forge.define") {
        if (params.confirm !== true) {
          result = {
            ok: false,
            error: "forge.define requires confirm: true; definitions are persistent and immutable",
          };
        } else {
          const definition: DeclarativeForgeDefinition = {
            name,
            version: version ?? "",
            description: optionalString(params.description) ?? "",
            template: typeof params.template === "string" ? params.template : "",
            parameters: Array.isArray(params.parameters)
              ? params.parameters.filter(isRecord)
              : [],
          };
          result = {
            ok: true,
            package: forgePackagePayload(this.declarativeForge.define(definition)),
          };
        }
      } else if (method === "forge.undefine") {
        result = params.confirm !== true
          ? { ok: false, error: "forge.undefine requires confirm: true" }
          : this.declarativeForge.undefine(name, version ?? "")
            ? { ok: true, removed: `${name}@${version ?? ""}` }
            : { ok: false, error: "forged package not found" };
      } else if (method === "forge.stop") {
        result = { ok: false, error: "declarative forge runs are synchronous; no run is active" };
      } else {
        result = { ok: false, error: `Unknown method: ${method}` };
      }
    } catch (error) {
      result = { ok: false, error: errorMessage(error) };
    }
    if (session && method !== "forge.list" && method !== "forge.inspect") {
      recordCreatorTrace(session.metadata, {
        action: method.slice("forge.".length),
        name,
        version: version ?? optionalString(result.version) ?? "",
        status: result.ok === false ? "error" : "ok",
        detail: optionalString(result.error) ?? optionalString(result.output) ?? "",
      });
      await this.runtime.flushSessions();
    }
    return result;
  }

  private refreshActiveModelCapabilities(connection: DaemonTransportConnection): void {
    if (!this.autoDiscoverModelCapabilities) return;
    const profileName = this.activeRuntimeProfileName();
    if (!profileName) return;
    let flight = this.modelCapabilityRefreshes.get(profileName);
    if (!flight) {
      flight = this.fetchModels({ profile_name: profileName }).then(() => undefined);
      this.modelCapabilityRefreshes.set(profileName, flight);
      void flight.then(
        () => this.modelCapabilityRefreshes.delete(profileName),
        () => this.modelCapabilityRefreshes.delete(profileName),
      );
    }
    void flight.then(() => {
      const session = this.runtime.sessionStatus(connection.activeSessionKey);
      if (session) this.emitStatus(connection, session);
    }).catch((error) => {
      this.emit(connection, "notification", {
        level: "warning",
        message: `Model capability discovery failed: ${errorMessage(error)}`,
      });
    });
  }

  private async fetchModels(params: JsonRpcPayload): Promise<JsonRpcPayload> {
    const profileName =
      optionalString(params.profile_name) ??
      optionalString(params.profile) ??
      optionalString(params.name);
    const baseUrl = optionalString(params.base_url);
    const requestedProvider = optionalString(params.provider);
    const hasExplicitConnection =
      baseUrl !== undefined ||
      params.api_key !== undefined ||
      requestedProvider !== undefined;

    if (hasExplicitConnection) {
      return {
        ok: false,
        error:
          "model discovery only accepts a stored profile name; save the provider profile first",
        models: [],
      };
    }

    const profile = profileName
      ? this.profileStore.get(profileName)
      : this.profileStore.active();
    if (!profile) {
      return {
        ok: false,
        error: profileName
          ? `No provider profile named ${profileName}`
          : "No active provider profile is configured",
        models: [],
      };
    }
    const fallbackModels = [...new Set([
      ...(profile.model.trim() ? [profile.model.trim()] : []),
      ...Object.keys(profile.model_capabilities ?? {}),
    ])];
    if (
      profile.provider === "claude-code" ||
      profile.base_url.startsWith("claude-code://")
    ) {
      return {
        ok: true,
        models: fallbackModels,
        catalog: fallbackModels.map(model => modelCapabilityPayload(profile, model)),
        profile: profile.name,
        source: "profile",
      };
    }

    if (
      profile.provider === "openai-codex" ||
      profile.base_url.includes("/backend-api/codex")
    ) {
      return this.fetchCodexModels(profile, fallbackModels);
    }
    if (profile.provider === "github-copilot") {
      return this.fetchCopilotModels(profile, fallbackModels);
    }
    if (profile.provider === "radius") {
      return this.fetchRadiusModels(profile, fallbackModels);
    }

    const apiKey = profileDiscoveryApiKey(profile);
    try {
      const catalog = await discoverModelCatalog({
        allowPrivateEndpoint: true,
        apiKey,
        baseUrl: profile.base_url,
        provider: profile.provider,
      });
      const models = catalog.map((model) => model.id);
      if (models.length > 0) this.rememberDiscoveredContextLimits(profile, catalog);
      const cachedProfile = this.profileStore.get(profile.name) ?? profile;
      return models.length
        ? {
            ok: true,
            models,
            catalog: models.map(model => modelCapabilityPayload(cachedProfile, model)),
            profile: profile.name,
            source: "remote",
          }
        : {
            ok: true,
            models: fallbackModels,
            catalog: fallbackModels.map(model => modelCapabilityPayload(cachedProfile, model)),
            profile: profile.name,
            source: "profile",
            warning: "provider returned no model ids",
          };
    } catch (error) {
      const warning = sanitizeModelDiscoveryError(error, {
        apiKey,
        baseUrl: profile.base_url,
      });
      return fallbackModels.length
        ? {
            ok: true,
            models: fallbackModels,
            catalog: fallbackModels.map(model => modelCapabilityPayload(profile, model)),
            profile: profile.name,
            source: "profile",
            warning,
          }
        : { ok: false, error: warning, models: [] };
    }
  }

  /**
   * Map the live runtime connection back to the selected stored profile.
   *
   * Runtime configuration can override a profile's provider or endpoint. In
   * that case returning the store's active name would make the TUI discover
   * and switch through the wrong connection, so report an explicit null
   * identity instead. No endpoint or credential leaves this method.
   */
  private activeRuntimeProfileName(): string | null {
    const profile = this.profileStore.active();
    if (!profile) {
      return null;
    }
    const status = this.runtime.status();
    const provider = optionalString(status.provider);
    const baseUrl = optionalString(status.base_url);
    if (!provider && !baseUrl) {
      return null;
    }
    if (
      provider &&
      normalizeProviderIdentity(provider) !==
        normalizeProviderIdentity(profile.provider)
    ) {
      return null;
    }
    if (
      baseUrl &&
      normalizeBaseUrlIdentity(baseUrl) !==
        normalizeBaseUrlIdentity(profile.base_url)
    ) {
      return null;
    }
    return profile.name;
  }

  /**
   * Discover the Codex catalog through the ChatGPT OAuth session.
   *
   * The generic discovery path cannot serve this provider: it authenticates
   * with an API key the subscription backend does not accept, and the catalog
   * lives behind a `client_version`-gated route rather than `/models`. The
   * list is plan-scoped, so it is fetched live instead of hard-coded.
   */
  private async fetchCodexModels(
    profile: ProviderProfile,
    fallbackModels: readonly string[],
  ): Promise<JsonRpcPayload> {
    try {
      const credential = await new CodexSession().credential();
      const catalog = await fetchCodexModelCatalog(credential, {
        ...(profile.base_url.trim() ? { baseUrl: profile.base_url.trim() } : {}),
      });
      const models = catalog.map((model) => model.id);
      if (models.length > 0) {
        this.rememberDiscoveredContextLimits(
          profile,
          catalog.map((model) => ({
            id: model.id,
            ...(model.contextLimit === undefined
              ? {}
              : { contextLimit: model.contextLimit }),
          })),
        );
      }
      const cachedProfile = this.profileStore.get(profile.name) ?? profile;
      return models.length
        ? {
            ok: true,
            models,
            catalog: models.map(model => modelCapabilityPayload(cachedProfile, model)),
            profile: profile.name,
            source: "remote",
          }
        : {
            ok: true,
            models: [...fallbackModels],
            catalog: fallbackModels.map(model => modelCapabilityPayload(cachedProfile, model)),
            profile: profile.name,
            source: "profile",
            warning: "ChatGPT plan returned no Codex models",
          };
    } catch (error) {
      // Falling back to the configured model keeps the picker usable when the
      // session has lapsed; the warning is what tells the user to sign in
      // rather than leaving an unexplained one-entry list.
      return {
        ok: true,
        models: [...fallbackModels],
        catalog: fallbackModels.map(model => modelCapabilityPayload(profile, model)),
        profile: profile.name,
        source: "profile",
        warning: errorMessage(error),
      };
    }
  }

  /** Copilot lists models through the exchanged proxy token, never the GitHub token. */
  private async fetchCopilotModels(
    profile: ProviderProfile,
    fallbackModels: readonly string[],
  ): Promise<JsonRpcPayload> {
    try {
      const credential = await new CopilotSession().credential();
      const models = await fetchCopilotModels(credential);
      const cachedProfile = this.profileStore.get(profile.name) ?? profile;
      return models.length
        ? {
            ok: true,
            models,
            catalog: models.map(model => modelCapabilityPayload(cachedProfile, model)),
            profile: profile.name,
            source: "remote",
          }
        : {
            ok: true,
            models: [...fallbackModels],
            catalog: fallbackModels.map(model => modelCapabilityPayload(cachedProfile, model)),
            profile: profile.name,
            source: "profile",
            warning: "GitHub Copilot returned no models",
          };
    } catch (error) {
      return {
        ok: true,
        models: [...fallbackModels],
        catalog: fallbackModels.map(model => modelCapabilityPayload(profile, model)),
        profile: profile.name,
        source: "profile",
        warning: errorMessage(error),
      };
    }
  }

  /** Radius's catalog is live gateway configuration, not a static list. */
  private async fetchRadiusModels(
    profile: ProviderProfile,
    fallbackModels: readonly string[],
  ): Promise<JsonRpcPayload> {
    try {
      const gateway = normalizeRadiusGatewayUrl(
        profile.base_url.trim() || DEFAULT_RADIUS_GATEWAY,
      );
      const apiKey = profileDiscoveryApiKey(profile);
      const config = await loadRadiusGatewayConfig(gateway, apiKey || undefined);
      const models = getRadiusModelsFromConfig("radius", config).map((model) => model.id);
      const cachedProfile = this.profileStore.get(profile.name) ?? profile;
      return models.length
        ? {
            ok: true,
            models,
            catalog: models.map(model => modelCapabilityPayload(cachedProfile, model)),
            profile: profile.name,
            source: "remote",
          }
        : {
            ok: true,
            models: [...fallbackModels],
            catalog: fallbackModels.map(model => modelCapabilityPayload(cachedProfile, model)),
            profile: profile.name,
            source: "profile",
            warning: "Radius gateway returned no models",
          };
    } catch (error) {
      return {
        ok: true,
        models: [...fallbackModels],
        catalog: fallbackModels.map(model => modelCapabilityPayload(profile, model)),
        profile: profile.name,
        source: "profile",
        warning: errorMessage(error),
      };
    }
  }

  private rememberDiscoveredContextLimits(
    profile: ProviderProfile,
    models: readonly DiscoveredModel[],
  ): void {
    const profilePrefix = discoveredContextProfilePrefix(profile);
    for (const key of this.discoveredContextLimits.keys()) {
      if (key.startsWith(profilePrefix)) this.discoveredContextLimits.delete(key);
    }
    const capabilities: Record<string, {
      readonly context_limit?: number;
      readonly max_output_tokens?: number;
    }> = Object.create(null);
    for (const model of models) {
      if (model.contextLimit !== undefined) {
        this.discoveredContextLimits.set(
          discoveredContextKey(profile, model.id),
          model.contextLimit,
        );
      }
      capabilities[model.id] = {
        ...(model.contextLimit === undefined ? {} : { context_limit: model.contextLimit }),
        ...(model.maxOutputTokens === undefined ? {} : { max_output_tokens: model.maxOutputTokens }),
      };
    }
    this.profileStore.replaceModelCapabilities(profile.name, capabilities);
  }

  private contextLimit(model: string): number {
    const activeName = this.activeRuntimeProfileName();
    const direct = this.contextLimitForProfile(activeName, model);
    if (direct > 0) return direct;
    // Cross-profile fallback: one attached client switching the daemon-wide
    // provider must not blind the sessions still running on another profile.
    // Their model keeps resolving against the profile that actually serves it;
    // without this the status bar reported "ctx unknown" for every session
    // left behind by a provider_select from a second TUI.
    for (const profile of this.profileStore.list()) {
      if (profile.name === activeName) continue;
      const candidate = this.contextLimitForProfile(profile.name, model);
      if (candidate > 0) return candidate;
    }
    return 0;
  }

  private contextLimitForProfile(profileName: string | null, model: string): number {
    const profile = profileName ? this.profileStore.get(profileName) : undefined;
    const resolved = resolvedProfileModelCapabilities(profile, model);
    if (resolved.contextSource === "override") return resolved.contextLimit ?? 0;
    const discovered = profile
      ? this.discoveredContextLimits.get(discoveredContextKey(profile, model))
      : undefined;
    return discovered ?? resolved.contextLimit ?? 0;
  }

  private maxOutputTokens(model: string): number | undefined {
    const activeName = this.activeRuntimeProfileName();
    const direct = resolvedProfileMaxOutputTokens(
      activeName ? this.profileStore.get(activeName) : undefined,
      model,
    );
    if (direct !== undefined) return direct;
    for (const profile of this.profileStore.list()) {
      if (profile.name === activeName) continue;
      const candidate = resolvedProfileMaxOutputTokens(profile, model);
      if (candidate !== undefined) return candidate;
    }
    return undefined;
  }

  /**
   * Tokens a prompt may actually occupy: the window less the reply the
   * provider is still allowed to emit. Both the auto-compaction trigger and
   * the `/budget` display read it here so they cannot drift apart, and so
   * neither of them measures a prompt against a ceiling the request as a whole
   * has to fit under.
   */
  private promptBudget(model: string): number {
    if (!model.trim()) return 0;
    const status = this.runtime.status();
    const requestedOutputTokens = typeof status.max_tokens === "number"
      ? status.max_tokens
      : this.maxOutputTokens(model);
    return effectiveContextLimit({
      contextLimit: this.contextLimit(model),
      ...(requestedOutputTokens === undefined ? {} : { requestedOutputTokens }),
    });
  }

  private async handleSlash(
    connection: DaemonTransportConnection,
    raw: string,
  ): Promise<JsonRpcPayload> {
    const command = raw.trim();
    // `!<cmd>` shell mode (Claude Code parity): run the rest in the project
    // shell and report output as a slash notification. The TUI already
    // routes `!` input here via the slash RPC; without this branch the
    // startsWith('/') guard rejected it and shell mode was dead on the wire.
    if (command.startsWith("!")) {
      return this.handleShellCommand(connection, command.slice(1).trim());
    }
    // `#<note>` quick memory (Claude Code parity): one line appended to the
    // project MEMORY.md — the same store the memory_add tool writes.
    if (command.startsWith("#")) {
      return this.handleMemoryNote(connection, command.slice(1).trim());
    }
    if (!command.startsWith("/")) {
      this.emitSlash(
        connection,
        "Slash commands must start with `/`.",
        "warning",
      );
      return { ok: false, error: "slash command must start with /" };
    }
    const [typed, ...argumentParts] = command.slice(1).split(/\s+/);
    const token = typed?.toLowerCase() ?? "";
    const entry = DAEMON_SLASH_COMMANDS.find(
      (candidate) =>
        candidate.name === token || candidate.aliases.includes(token),
    );
    const canonical = resolveCommand(command);
    const name = entry?.name ?? canonical?.name ?? token;
    const args = argumentParts.join(" ").trim();
    const key = connection.activeSessionKey;
    const session = this.runtime.sessionStatus(key);
    const plugin = this.slashPluginRegistry.resolve(command);
    if (plugin) {
      try {
        const result = await plugin.handler();
        const body =
          typeof result === "string" && result.trim()
            ? result
            : `Plugin command /${plugin.command.name} completed.`;
        this.emitSlash(connection, body);
        return { ok: true, plugin: plugin.command.name };
      } catch (error) {
        const message = errorMessage(error);
        this.emitSlash(
          connection,
          `Plugin command /${plugin.command.name} failed: \`${message}\`.`,
          "error",
        );
        return { ok: false, error: message };
      }
    }

    switch (name) {
      case "help":
      case "commands":
        this.emitSlash(
          connection,
          [
            "Available Bun daemon commands:",
            ...DAEMON_SLASH_COMMANDS.map(
              (item) => `  /${item.name} — ${item.description}`,
            ),
          ].join("\n"),
        );
        return { ok: true };
      case "status":
        this.emitSlash(
          connection,
          JSON.stringify(this.runtimeStatusPayload(), null, 2),
        );
        return { ok: true };
      case "config":
        return this.showRuntimeConfig(connection);
      case "sampling":
        return this.configureSampling(connection, args);
      case "reasoning":
        return this.configureReasoning(connection, args);
      case "fast":
        return this.configureRuntimeToggle(
          connection,
          "fast_mode",
          args,
          "Fast mode",
        );
      case "nudge":
        return this.configureRuntimeToggle(connection, "nudge", args, "Nudge");
      case "verbose":
        return this.configureRuntimeToggle(
          connection,
          "verbose",
          args,
          "Verbose logging",
        );
      case "debug":
        return this.configureRuntimeToggle(
          connection,
          "debug",
          args,
          "Debug logging",
        );
      case "agents":
        return this.listAgents(connection, session);
      case "toolsets":
        return this.listToolsets(connection, session);
      case "platforms":
        return this.listPlatforms(connection);
      case "plugins":
        return this.listPlugins(connection);
      case "skills":
        return this.listSkills(connection, session);
      case "skill":
        return this.invokeSkill(connection, args, session);
      case "soul":
        return this.showSoul(connection, session);
      case "memory":
        return this.showMemory(connection, session);
      case "personality":
        return this.showPersonality(connection, session);
      case "context":
      case "usage": {
        if (!session) {
          this.emitSlash(connection, "No active session yet.", "warning");
          return { ok: false, error: "no active session" };
        }
        const section = formatSessionUsage(
          session,
          this.contextLimit(session.model),
        );
        // Subscription quota joins the session block only when a provider
        // answers; a fetch failure or missing login must never hide the
        // local usage the command already had.
        let subscriptionSection = "";
        try {
          const collection = await collectSubscriptionUsage();
          if (collection.reports.length) {
            subscriptionSection = [
              "",
              "Subscription usage:",
              ...collection.reports.map((report) => `  ${formatUsageReport(report)}`),
            ].join("\n");
          }
        } catch {
          // Keep the session report usable when the network is unavailable.
        }
        this.emitSlash(connection, `${section}${subscriptionSection}`);
        return { ok: true };
      }
      case "history":
        if (!session) {
          this.emitSlash(connection, "No active session yet.", "warning");
          return { ok: false, error: "no active session" };
        }
        this.emitSlash(connection, formatSessionHistory(session));
        return { ok: true, history: sessionHistoryPayload(session) };
      case "cron":
        return this.manageCronJobs(connection, args);
      case "background":
        return this.showBackgroundTasks(connection);
      case "browser":
        return this.manageBrowserSlash(connection, args);
      case "clear":
        this.clearAutoCompactFailures(key);
        this.emitSlash(connection, "Cleared. Scrollback is owned by the TUI.");
        return { ok: true };
      case "feedback":
        this.emitSlash(
          connection,
          "Feedback / issues:\n  • GitHub: https://github.com/erfanzar/Xerxes/issues\n  • Native daemon logs: `~/.xerxes/daemon.log`.",
        );
        return { ok: true };
      case "new": {
        // Flush before evicting so unpersisted edits survive the reset.
        await this.runtime.flushSessions();
        // A resumed connection holds the persisted hex id as its session key,
        // and openSession re-adopts that transcript — announcing a new
        // session while actually continuing the old one. Mint a fresh
        // non-hex slot key exactly like a new attach does, so /new always
        // opens an empty conversation; the old transcript is left untouched.
        const previousKey = key;
        this.forgetAcceptedSubmissions([previousKey]);
        this.endSessionLifetime([previousKey]);
        this.runtime.evictSession(previousKey);
        this.clearAutoCompactFailures(previousKey);
        const freshKey = `tui:${newConnectionKey()}`;
        connection.activeSessionKey = freshKey;
        const fresh = await this.runtime.openSession(freshKey);
        this.emitSlash(connection, `New session \`${fresh.id}\` started.`);
        this.emitInitDone(connection, fresh);
        this.emitStatus(connection, fresh);
        return {
          ok: true,
          session: sessionPayload(fresh, this.contextLimit(fresh.model), this.mcpStatusRecord()),
        };
      }
      case "stop": {
        const cancelled = this.cancelTrackedTurn(key);
        this.emitSlash(
          connection,
          cancelled ? "Cancelled." : "Nothing running to cancel.",
        );
        return { ok: cancelled };
      }
      case "cancel-all": {
        const cancelled = this.runtime.cancelAllTurns();
        this.emitSlash(
          connection,
          `Cancelled ${cancelled} running turn${cancelled === 1 ? "" : "s"}.`,
        );
        return { ok: true, cancelled };
      }
      case "btw":
      case "steer": {
        if (!args) {
          this.emitSlash(connection, "Usage: `/steer <hint>`.", "warning");
          return { ok: false, error: "steer text is required" };
        }
        const processed = session
          ? await processAtMentions(args, session.cwd)
          : { enhancedMessage: args, mentionedFiles: [] };
        const steered = this.runtime.steerTurn(
          key,
          processed.enhancedMessage,
        );
        if (steered) {
          this.emit(connection, "steer_input", {
            content: args,
            ...(processed.mentionedFiles.length
              ? { mentioned_files: processed.mentionedFiles }
              : {}),
          });
        }
        this.emitSlash(
          connection,
          steered ? "Steer accepted." : "No active session to steer.",
          steered ? "info" : "warning",
        );
        return { ok: steered };
      }
      case "model": {
        const active = this.runtime.sessionStatus(connection.activeSessionKey);
        if (!args) {
          // The session's own model, not the daemon-wide one: with two
          // sessions open those differ, and reporting the global value would
          // name a model this session is not using.
          const current = active?.model || stringValue(this.runtime.status().model);
          this.emitSlash(
            connection,
            `Active model: \`${current || "(not configured)"}\`.`,
          );
          return { ok: true, model: current };
        }
        // Scoped to this session so a second open session keeps its own model.
        // Only when the host cannot do that does this fall back to the global
        // reload, which moves every unpinned session at once.
        const pinned = await this.runtime.setSessionModel?.(
          connection.activeSessionKey,
          args,
        );
        if (!pinned) {
          this.runtime.reload({ model: args });
        }
        // Persist as the default for sessions opened later; it no longer
        // retargets sessions that already picked for themselves.
        try {
          this.profileStore?.updateActiveModel(args);
        } catch {
          // Profile persistence is best-effort; the in-memory model applies regardless.
        }
        this.emitSlash(connection, `Model set to \`${args}\`.`);
        await this.emitProviderInit(connection);
        const session = this.runtime.sessionStatus(connection.activeSessionKey);
        if (session) this.emitStatus(connection, session);
        return { ok: true, model: args };
      }
      case "provider":
        if (!args) {
          return this.openProviderFlow(connection);
        }
        this.cancelSkillCreate(connection);
        this.cancelProviderFlow(connection);
        return this.selectProvider(connection, args);
      case "skill-create":
        return this.openSkillCreate(connection, args);
      case "permissions": {
        const current = runtimePermissionMode(
          this.runtime.status().permission_mode,
        );
        if (!args) {
          this.emitSlash(connection, `Permission mode: \`${current}\`.`);
          return { ok: true, permission_mode: current };
        }
        if (!isPermissionMode(args)) {
          this.emitSlash(
            connection,
            "Permission mode must be `accept-all`, `auto`, `manual`, or `plan`.",
            "warning",
          );
          return { ok: false, error: "invalid permission mode" };
        }
        // Scoped to this session so a second one keeps its own trust level.
        const pinnedPermission = await this.runtime.setSessionPermissionMode?.(
          connection.activeSessionKey,
          args,
        );
        if (!pinnedPermission) {
          this.runtime.reload({ permission_mode: args });
        }
        const session = this.runtime.sessionStatus(connection.activeSessionKey);
        if (session) {
          this.emitStatus(connection, session);
        }
        this.emitSlash(connection, `Permission mode: \`${args}\`.`);
        return { ok: true, permission_mode: args };
      }
      case "yolo": {
        // Toggles relative to this session's own mode: keyed off the global
        // one it would flip based on a value another session had set.
        const active = this.runtime.sessionStatus(connection.activeSessionKey);
        const current = runtimePermissionMode(
          active?.permissionMode ?? this.runtime.status().permission_mode,
        );
        const next = current === "accept-all" ? "auto" : "accept-all";
        const pinnedYolo = await this.runtime.setSessionPermissionMode?.(
          connection.activeSessionKey,
          next,
        );
        if (!pinnedYolo) {
          this.runtime.reload({ permission_mode: next });
        }
        const session = this.runtime.sessionStatus(connection.activeSessionKey);
        if (session) {
          this.emitStatus(connection, session);
        }
        this.emitSlash(
          connection,
          `YOLO mode ${next === "accept-all" ? "ON" : "OFF"}.`,
        );
        return { ok: true, permission_mode: next };
      }
      case "mode":
        return this.setMode(connection, args || "code");
      case "plan":
        return this.setMode(
          connection,
          args === "off" ? "code" : "plan",
          args !== "off",
        );
      // /ultra [off] toggles session-scoped ultra mode; bare "/ultra" turns
      // it on, only the explicit "off" argument disables it.
      case "ultra":
        return this.setUltra(connection, args.trim().toLowerCase() !== "off");
      case "compact":
        return this.compactSession(connection);
      case "budget":
        return this.showSessionBudget(connection, session);
      case "cost":
        return this.showSessionCost(connection, session);
      case "doctor":
        return this.runDoctor(connection);
      case "insights":
        return this.showSessionInsights(connection, session);
      case "reload":
        return this.reloadRuntime(connection, session);
      case "reload-mcp":
        return this.reloadMcp(connection);
      case "restart":
        return this.restartDaemon(connection);
      case "update":
        return this.showUpdate(connection, session);
      case "resume":
        return this.resumeSavedSession(connection, args);
      case "search":
        return this.searchTranscripts(connection, args);
      case "branches":
        return this.listSavedSessionBranches(connection);
      case "branch":
        return this.branchSession(connection, session, args);
      case "undo":
        return this.undoLastTurn(connection, session);
      case "retry":
        return this.retryLastTurn(connection, session);
      case "retry-connection":
        return this.retryConnection(connection, session);
      case "title":
        return this.setSessionTitle(connection, session, args);
      case "save":
        return this.saveActiveSession(connection, session, args);
      case "snapshot":
        return this.createSnapshot(connection, session, args);
      case "snapshots":
        return this.listSnapshots(connection, session);
      case "rollback":
        return this.rollbackSnapshot(connection, session, args);
      case "tools":
        return this.listTools(connection);
      case "init":
        return this.initializeProject(connection, session, args);
      case "workspace":
        return this.showWorkspace(connection, session, args);
      case "image":
        return this.generateImage(connection, args);
      case "paste":
      case "queue":
      case "skin":
      case "statusbar":
      case "voice":
        return this.forwardUiControl(connection, name, args);
      case "exit":
        this.emitSlash(
          connection,
          "Close this TUI or send the `shutdown` JSON-RPC method to stop the daemon.",
        );
        return { ok: true };
      default:
        return this.tryInvokeSkillShorthand(connection, token, args, session);
    }
  }

  private showRuntimeConfig(
    connection: DaemonTransportConnection,
  ): JsonRpcPayload {
    const config = displayedRuntimeConfig(this.runtime.status());
    const entries = Object.entries(config);
    const body = entries.length
      ? [
          "Effective native runtime config:",
          ...entries.map(
            ([name, value]) => `  \`${name}\` = \`${String(value)}\``,
          ),
        ].join("\n")
      : "No native runtime configuration is active.";
    this.emitSlash(connection, body);
    return { ok: true, config };
  }

  private async configureReasoning(
    connection: DaemonTransportConnection,
    raw: string,
  ): Promise<JsonRpcPayload> {
    // This session's own effort, not the daemon-wide one: with two sessions
    // open those differ, and naming the global value would report an effort
    // this session is not running at.
    const active = this.runtime.sessionStatus(connection.activeSessionKey);
    const current =
      active?.reasoningEffort
      || stringValue(this.runtime.status().reasoning_effort)
      || REASONING_OFF;
    const levels = await this.reasoningLevels(active?.model);
    const offered = selectableEfforts(levels);
    const requested = raw.trim();
    if (!requested) {
      this.emitSlash(
        connection,
        `Thinking: \`${current}\`\nLevels: ${offered.join(" | ")}\nSet with \`/thinking <level>\`.`,
      );
      return { ok: true, reasoning_effort: current, levels: offered };
    }
    // Validated against what this model actually accepts. The efforts differ
    // per model — some publish `ultra`, others stop at `xhigh` — so a fixed
    // list would both reject valid levels and accept ones the backend 400s on.
    // Validated against what this model actually accepts. The efforts differ
    // per model — some publish `ultra`, others stop at `xhigh` — so a fixed
    // list would both reject valid levels and accept ones the backend 400s on.
    // A known ladder word the model lacks clamps to its nearest rung
    // (pi-ai clampThinkingLevel); an unknown word stays a usage error.
    const resolved = resolveEffort(levels, requested) ?? clampEffort(levels, requested);
    if (!resolved) {
      this.emitSlash(
        connection,
        `Thinking level must be one of: ${offered.join(", ")}.`,
        "warning",
      );
      return { ok: false, error: "invalid reasoning effort", levels: offered };
    }
    // Scoped to this session so a second open session keeps its own effort;
    // only a host without the session-level setter falls back to the global
    // reload, which moves every unpinned session at once.
    const pinned = await this.runtime.setSessionReasoning?.(
      connection.activeSessionKey,
      resolved,
    );
    if (!pinned) {
      this.runtime.reload({
        reasoning_effort: resolved,
        thinking: resolved !== REASONING_OFF,
      });
    }
    // Still recorded as the default for sessions opened later; it no longer
    // retargets sessions already running.
    const profile = this.profileStore.active();
    if (profile) {
      this.profileStore.updateSampling(profile.name, {
        reasoning_effort: resolved,
        thinking: resolved !== REASONING_OFF,
      });
    }
    const session = this.runtime.sessionStatus(connection.activeSessionKey);
    if (session) {
      this.emitStatus(connection, session);
    }
    this.emitSlash(connection, `Thinking: \`${resolved}\`.`);
    return { ok: true, reasoning_effort: resolved, levels: offered };
  }

  /**
   * Reasoning efforts the active model accepts.
   *
   * Asked of the provider whenever it can answer, because the set is a
   * property of the model rather than of Xerxes: the Codex catalog alone
   * ranges from four efforts to six, with three different defaults. Providers
   * with no capability endpoint fall back to a per-provider table.
   */
  /**
   * Estimated token budget split for the active session's next request:
   * system-prompt scaffold, tool schemas, and transcript messages. These are
   * the same counter estimates that drive auto-compaction — never provider
   * telemetry, and rendered with a `~` by clients for that reason.
   */
  private contextBreakdown(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): JsonRpcPayload {
    const key = sessionKey(connection, params);
    const session = this.runtime.sessionStatus(key);
    if (!session) {
      return { ok: false, error: "no active session" };
    }
    const model = session.model || stringValue(this.runtime.status().model) || "";
    const scaffold = sessionContextScaffold(session);
    const systemPromptTokens = scaffold.systemPrompt
      ? estimateContextTokens([], { model, systemPrompt: scaffold.systemPrompt })
      : 0;
    const toolsTokens = scaffold.toolSchemas?.length
      ? estimateContextTokens([], { model, toolSchemas: scaffold.toolSchemas })
      : 0;
    const messagesTokens = estimateContextTokens(session.messages, { model });
    return {
      ok: true,
      model,
      system_prompt_tokens: systemPromptTokens,
      tools_tokens: toolsTokens,
      messages_tokens: messagesTokens,
      total_tokens: sessionContextTokens(session, model),
      context_limit: this.contextLimit(model),
    };
  }

  private async reasoningLevels(modelOverride?: string): Promise<ReasoningLevelSet> {
    const status = this.runtime.status();
    const model = modelOverride?.trim() || stringValue(status.model) || "";
    const profile = this.profileStore.active();
    const providerName = resolveProviderSafely(model, profile);
    // The generated Pi catalog knows each model's real ladder
    // (thinking_level_map); the static provider table is only the last resort
    // for models the catalog does not carry.
    const catalog = catalogReasoningLevels(model, providerName);

    if (providerName !== "openai-codex") {
      return catalog ?? fallbackReasoningLevels(providerName);
    }

    const cached = this.reasoningLevelCache.get(model);
    if (cached) {
      return cached;
    }
    try {
      const credential = await new CodexSession().credential();
      const liveCatalog = await fetchCodexModelCatalog(credential, {
        ...(profile?.base_url.trim() ? { baseUrl: profile.base_url.trim() } : {}),
      });
      const bare = model.includes("/") ? model.slice(model.indexOf("/") + 1) : model;
      const entry = liveCatalog.find((candidate) => candidate.id === bare);
      if (!entry?.reasoningLevels.length) {
        return catalog ?? fallbackReasoningLevels(providerName);
      }
      const resolved = providerReasoningLevels(
        entry.reasoningLevels.map((level) => ({
          effort: level.effort,
          ...(level.description === undefined
            ? {}
            : { description: level.description }),
        })),
        entry.defaultReasoningLevel,
      );
      this.reasoningLevelCache.set(model, resolved);
      return resolved;
    } catch {
      // A lapsed session or offline host must not make the level list
      // unusable; the catalog still describes the model's real ladder.
      return catalog ?? fallbackReasoningLevels(providerName);
    }
  }

  private configureRuntimeToggle(
    connection: DaemonTransportConnection,
    key: "debug" | "fast_mode" | "nudge" | "verbose",
    raw: string,
    label: string,
  ): JsonRpcPayload {
    const current = this.runtime.status()[key] === true;
    const action = raw.trim().toLowerCase();
    if (action && action !== "on" && action !== "off") {
      this.emitSlash(
        connection,
        `Usage: \`/${key === "fast_mode" ? "fast" : key} [on|off]\`.`,
        "warning",
      );
      return { ok: false, error: "invalid toggle value" };
    }
    const enabled =
      action === "on" ? true : action === "off" ? false : !current;
    this.runtime.reload({ [key]: enabled });
    this.emitSlash(connection, `${label}: ${enabled ? "ON" : "OFF"}.`);
    return { ok: true, [key]: enabled };
  }

  private listToolsets(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): JsonRpcPayload {
    const definitions = this.agentDefinitionLoader(
      session?.cwd ?? process.cwd(),
    );
    const toolsets = definitions.map(agentDefinitionPayload);
    if (!toolsets.length) {
      this.emitSlash(connection, "No native agent toolsets configured.");
      return { ok: true, toolsets: [] };
    }
    this.emitSlash(
      connection,
      [
        `Native agent toolsets (${toolsets.length}):`,
        ...toolsets.map(
          (toolset) =>
            `  \`${String(toolset.name)}\` — ${String(toolset.description) || "No description"}`,
        ),
      ].join("\n"),
    );
    return { ok: true, toolsets };
  }

  private listPlugins(connection: DaemonTransportConnection): JsonRpcPayload {
    const plugins = this.pluginRegistry.pluginNames.sort();
    const slashCommands = this.slashPluginRegistry.list();
    const lines = ["Native plugins:"];
    lines.push(
      ...(plugins.length
        ? plugins.map((name) => `  \`${name}\``)
        : ["  (no plugins loaded)"]),
    );
    if (slashCommands.length) {
      lines.push("", "Plugin slash commands:");
      lines.push(
        ...slashCommands.map(
          (plugin) =>
            `  \`/${plugin.command.name}\` — ${plugin.command.description}`,
        ),
      );
    }
    this.emitSlash(connection, lines.join("\n"));
    return {
      ok: true,
      plugins,
      slash_commands: slashCommands.map((plugin) => ({
        name: plugin.command.name,
        description: plugin.command.description,
      })),
    };
  }

  private async refreshSkills(
    session: DaemonSession | undefined,
  ): Promise<void> {
    const roots =
      this.skillDirectories ??
      defaultSkillDiscoveryRoots({
        cwd: session?.cwd ?? process.cwd(),
        userSkillsDirectory: this.skillDirectory,
      });
    await this.skillRegistry.refresh(...roots);
  }

  private async listSkills(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    await this.refreshSkills(session);
    const skills = this.skillRegistry
      .all()
      .filter((skill) => skillMatchesPlatform(skill));
    if (!skills.length) {
      this.emitSlash(connection, "No native skills discovered.");
      return { ok: true, skills: [] };
    }
    this.emitSlash(
      connection,
      [
        `Native skills (${skills.length}):`,
        ...skills.map(
          (skill) =>
            `  \`/${skill.metadata.name}\` — ${skill.metadata.description || "No description"}`,
        ),
      ].join("\n"),
    );
    return {
      ok: true,
      skills: skills.map((skill) => ({
        name: skill.metadata.name,
        description: skill.metadata.description,
        source: skill.sourcePath,
        subcommands: [...skill.metadata.subcommands],
      })),
    };
  }

  private async invokeSkill(
    connection: DaemonTransportConnection,
    raw: string,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    const [reference = "", ...argumentParts] = raw.trim().split(/\s+/);
    if (!reference) {
      this.emitSlash(
        connection,
        "Usage: `/skill <name[:subcommand]> [arguments]`.",
        "warning",
      );
      return { ok: false, error: "skill name is required" };
    }
    await this.refreshSkills(session);
    const [name, subcommand] = reference.split(":", 2);
    const skill = name ? this.skillRegistry.get(name) : undefined;
    if (!skill || !skillMatchesPlatform(skill)) {
      this.emitSlash(
        connection,
        `No native skill named \`${reference}\`.`,
        "warning",
      );
      return { ok: false, error: "skill not found" };
    }
    if (subcommand && !skill.metadata.subcommands.includes(subcommand)) {
      this.emitSlash(
        connection,
        `Skill \`${name}\` has no \`${subcommand}\` subcommand.`,
        "warning",
      );
      return { ok: false, error: "skill subcommand not found" };
    }
    const sessionKey = connection.activeSessionKey;
    const openedSession = await this.runtime.openSession(sessionKey);
    const argumentsText = argumentParts.join(" ").trim();
    // Claude Code custom-command parity: expand $ARGUMENTS/$N and !`cmd`
    // injections before the activation prompt is built. Expansion output is
    // untrusted — the scan in skillPromptSection still applies.
    const expandedInstructions = await expandSkillInstructions(skill.instructions, {
      ...(argumentsText ? { args: argumentsText } : {}),
      cwd: openedSession.cwd,
    });
    const prompt = skillActivationPrompt({ ...skill, instructions: expandedInstructions }, {
      ...(subcommand ? { subcommand } : {}),
      ...(argumentsText ? { request: argumentsText } : {}),
    });
    void this.submitTrackedTurn(
      sessionKey,
      prompt,
      (event) => this.emit(connection, event.type, event.payload),
      connection,
      // Preserve what the user actually typed for mid-turn tab reattachment.
      // The expanded [Skill … activated] prompt is private runtime context and
      // is intentionally filtered from the transcript.
      { displayText: `/skill ${raw.trim()}` },
    ).catch((error) =>
      this.emit(connection, "notification", {
        level: "error",
        message: errorMessage(error),
      }),
    );
    return {
      ok: true,
      queued: true,
      skill: skill.metadata.name,
      ...(subcommand ? { subcommand } : {}),
    };
  }

  private async showSoul(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    const memory = this.memoryFactory(session);
    await memory.ensure();
    const path = join(memory.scopeDirectory("global"), "SOUL.md");
    this.emitSlash(
      connection,
      `Soul / values file: \`${path}\`\nEdit it, then run \`/reload\` to refresh native skill and runtime state.`,
    );
    return { ok: true, path };
  }

  private async showMemory(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    const memory = this.memoryFactory(session);
    await memory.ensure();
    const files = await memory.listFiles();
    const lines = [
      "Native memory:",
      `  Global scope: \`${memory.globalDirectory}\``,
    ];
    for (const name of CANONICAL_AGENT_MEMORY_FILES) {
      const item = files.find(
        (file) => file.scope === "global" && file.path === name,
      );
      lines.push(`    \`${name}\` — ${item?.bytes ?? 0} bytes`);
    }
    if (memory.projectDirectory) {
      lines.push(`  Project scope: \`${memory.projectDirectory}\``);
    }
    this.emitSlash(connection, lines.join("\n"));
    return {
      ok: true,
      global_directory: memory.globalDirectory,
      ...(memory.projectDirectory
        ? { project_directory: memory.projectDirectory }
        : {}),
      files: files.map((file) => ({ ...file })),
    };
  }

  /**
   * Global memory wipe. Every memory scope the daemon can reach is removed:
   * the cross-project store, per-agent self-memory, per-project stores, and
   * the SQLite tiers. A running session re-creates empty stores on its next
   * memory read, so wiping never leaves a session unable to write memory.
   */
  private async wipeMemory(
    connection: DaemonTransportConnection,
  ): Promise<JsonRpcPayload> {
    const active = this.runtime
      .listSessions()
      .find((session) => session.activeTurnId);
    if (active) {
      return {
        ok: false,
        error:
          "a turn may be using memory; wait for it to finish or cancel it before wiping memory",
      };
    }
    try {
      const session = this.runtime.sessionStatus(connection.activeSessionKey);
      const result = await wipeMemoryStores(xerxesHome(), session?.cwd);
      this.emitSlash(
        connection,
        `Memory wiped globally: ${result.removed.files} file(s), ${formatBytes(result.removed.bytes)} removed.`,
      );
      return { ...result, removed: { ...result.removed } };
    } catch (error) {
      return { ok: false, error: errorMessage(error) };
    }
  }

  /**
   * Global history wipe. Removes the persisted transcript store and snapshot
   * shadows and drops the search index entries. Live sessions keep running;
   * each re-saves its transcript on the next turn, so this clears saved and
   * resumable history without killing open work.
   */
  private async wipeHistory(
    connection: DaemonTransportConnection,
  ): Promise<JsonRpcPayload> {
    const active = this.runtime
      .listSessions()
      .find((session) => session.activeTurnId);
    if (active) {
      return {
        ok: false,
        error:
          "a turn is mid-write; wait for it to finish or cancel it before wiping history",
      };
    }
    try {
      const result = await wipeHistoryStores(
        this.sessionArchiveDirectory,
        join(xerxesHome(), "snapshots"),
      );
      // The files no longer carry the optimistic generation/message prefix
      // each live session was based on. Reset those baselines so the next turn
      // can recreate its retained in-memory transcript from generation zero.
      this.runtime.resetSavedTranscriptState?.();
      this.transcriptSearch.clear();
      this.transcriptSearchHydration = undefined;
      this.emitSlash(
        connection,
        `History wiped globally: ${result.removed.files} file(s), ${formatBytes(result.removed.bytes)} removed.`,
      );
      return { ...result, removed: { ...result.removed } };
    } catch (error) {
      return { ok: false, error: errorMessage(error) };
    }
  }

  private showPersonality(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): JsonRpcPayload {
    if (!session) {
      this.emitSlash(connection, "No active session yet.", "warning");
      return { ok: false, error: "no active session" };
    }
    const path = join(session.workspace, "AGENTS.md");
    this.emitSlash(
      connection,
      `Persona / instructions file: \`${path}\`\nEdit it, then run \`/reload\`.`,
    );
    return { ok: true, path };
  }

  private showBackgroundTasks(
    connection: DaemonTransportConnection,
  ): JsonRpcPayload {
    const sessions = this.runtime
      .listSessions()
      .filter(
        (session) =>
          session.status === "starting" ||
          session.status === "waiting" ||
          session.status === "working",
      )
      .map((session) => sessionPayload(session, this.contextLimit(session.model)));
    if (!sessions.length) {
      this.emitSlash(connection, "No native background turns running.");
      return { ok: true, sessions: [] };
    }
    this.emitSlash(
      connection,
      [
        `Native background turns (${sessions.length}):`,
        ...sessions.map(
          (item) =>
            `  \`${String(item.key)}\` — ${String(item.status)} (${String(item.active_turn_id)})`,
        ),
      ].join("\n"),
    );
    return { ok: true, sessions };
  }

  private terminalOwnerSessionId(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): string {
    const key = sessionKey(connection, params);
    return this.runtime.sessionStatus(key)?.id ?? key;
  }

  /** One terminal with the retained tail of its output; never drains the model's copy. */
  private inspectTerminal(connection: DaemonTransportConnection, params: JsonRpcPayload): JsonRpcPayload {
    const id = optionalString(params.terminal_id) ?? optionalString(params.id);
    if (!id) return { ok: false, error: "terminal_id is required" };
    const requested = integerOption(params.max_output_chars);
    const maxChars =
      requested === undefined ? undefined : Math.min(requested, 200_000);
    const terminal = this.terminalRegistry?.inspect(
      this.terminalOwnerSessionId(connection, params),
      id,
      ...(maxChars === undefined ? [] : ([maxChars] as const)),
    );
    return terminal
      ? { ok: true, terminal }
      : { ok: false, error: "unknown terminal" };
  }

  /**
   * Send input to, interrupt, or kill one live terminal.
   *
   * Kept behind an explicit action rather than three methods so a client can
   * discover the whole control surface from one signature, and so an
   * unsupported action fails with the reason instead of "unknown method".
   */
  private async controlTerminal(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    const registry = this.terminalRegistry;
    if (!registry) {
      return { ok: false, error: "this daemon tracks no terminals" };
    }
    const id = optionalString(params.terminal_id) ?? optionalString(params.id);
    if (!id) return { ok: false, error: "terminal_id is required" };
    const action = (optionalString(params.action) ?? "").toLowerCase();
    const ownerSessionId = this.terminalOwnerSessionId(connection, params);
    try {
      if (action === "write") {
        // Deliberately not `optionalString`, which trims: the trailing newline
        // is what submits the line, and trimming it would send a command the
        // shell then sits on waiting for Enter.
        await registry.write(
          ownerSessionId,
          id,
          typeof params.chars === "string" ? params.chars : "",
        );
      } else if (action === "interrupt") {
        await registry.interrupt(ownerSessionId, id);
      } else if (action === "kill") {
        const force = params.signal === "SIGKILL" || params.force === true;
        await registry.kill(ownerSessionId, id, force ? "SIGKILL" : "SIGTERM");
      } else {
        return {
          ok: false,
          error: "terminal action must be write, interrupt, or kill",
        };
      }
    } catch (error) {
      return { ok: false, error: errorMessage(error) };
    }
    const terminal = registry.inspect(ownerSessionId, id);
    return { ok: true, ...(terminal ? { terminal } : {}) };
  }

  private async manageBrowser(params: JsonRpcPayload): Promise<JsonRpcPayload> {
    const action = (optionalString(params.action) ?? "status").toLowerCase();
    if (action === "status" || action === "pages") {
      return this.browserStatusPayload();
    }
    if (action === "connect") {
      const endpoint =
        optionalString(params.endpoint) ??
        optionalString(params.cdp_url) ??
        optionalString(params.url);
      if (!endpoint) {
        return { ok: false, error: "browser CDP endpoint is required" };
      }
      try {
        const status = await this.browserManager.connectCdp(endpoint);
        return { ok: true, status, pages: this.browserManager.listPages() };
      } catch (error) {
        return { ok: false, error: errorMessage(error) };
      }
    }
    if (action === "disconnect") {
      await this.browserManager.disconnect();
      return this.browserStatusPayload();
    }
    return {
      ok: false,
      error: "browser action must be status, pages, connect, or disconnect",
    };
  }

  private async manageBrowserSlash(
    connection: DaemonTransportConnection,
    raw: string,
  ): Promise<JsonRpcPayload> {
    const [typedAction, ...argumentParts] = raw.trim().split(/\s+/);
    const action = typedAction?.toLowerCase() || "status";
    const endpoint = argumentParts.join(" ").trim();
    const result = await this.manageBrowser({
      action,
      ...(endpoint ? { endpoint } : {}),
    });
    if (result.ok !== true) {
      this.emitSlash(
        connection,
        `Browser command failed: \`${String(result.error)}\`.`,
        "warning",
      );
      return result;
    }
    const status = isRecord(result.status) ? result.status : {};
    const pages = Array.isArray(result.pages) ? result.pages : [];
    const actionName = action;
    if (actionName === "connect") {
      this.emitSlash(
        connection,
        `Connected native browser (${String(status.kind ?? "unknown")}).`,
      );
      return result;
    }
    if (actionName === "disconnect") {
      this.emitSlash(connection, "Disconnected native browser.");
      return result;
    }
    const lines = [
      `Native browser: ${status.connected === true ? "connected" : "not connected"} (${String(status.kind ?? "none")})`,
      ...(typeof status.endpoint === "string"
        ? [`Endpoint: \`${status.endpoint}\``]
        : []),
    ];
    if (pages.length) {
      lines.push(
        "Pages:",
        ...pages.map((page) => {
          const item = isRecord(page) ? page : {};
          return `  \`${String(item.refId ?? "?")}\` — ${String(item.title ?? "")} (${String(item.url ?? "")})`;
        }),
      );
    } else {
      lines.push("No browser pages are open.");
    }
    lines.push(
      "Use `/browser connect <http(s) CDP endpoint>` to attach Chromium, or `/browser disconnect` to detach.",
    );
    this.emitSlash(connection, lines.join("\n"));
    return result;
  }

  private browserStatusPayload(): JsonRpcPayload {
    return {
      ok: true,
      status: this.browserManager.connectionStatus(),
      pages: this.browserManager.listPages(),
    };
  }

  /** Redacted per-server MCP status for the wire; empty without a manager. */
  private mcpStatusRecord(): Record<string, unknown> {
    const statuses = this.mcpManager?.listStatus() ?? [];
    return Object.fromEntries(
      statuses.map((entry) => [
        entry.name,
        {
          connected: entry.connected,
          tools: entry.tools,
          resources: entry.resources,
          prompts: entry.prompts,
          ...(entry.lastError ? { lastError: entry.lastError } : {}),
        },
      ]),
    );
  }

  private async reloadMcp(
    connection: DaemonTransportConnection,
  ): Promise<JsonRpcPayload> {
    const manager = this.mcpManager;
    if (!manager) {
      this.emitSlash(
        connection,
        "No native MCP manager is configured. Inject `mcpManager` into DaemonServer to enable `/reload-mcp`.",
        "warning",
      );
      return { ok: true, configured: false, servers: [] };
    }
    const servers = manager.listServers();
    if (!servers.length) {
      this.emitSlash(connection, "No native MCP servers are connected.");
      return { ok: true, configured: true, servers: [] };
    }
    const results: Array<{
      readonly name: string;
      readonly reconnected: boolean;
    }> = [];
    for (const name of servers) {
      results.push({ name, reconnected: await manager.reconnect(name) });
    }
    const failed = results.filter((result) => !result.reconnected);
    this.emitSlash(
      connection,
      failed.length
        ? `Reloaded ${results.length - failed.length}/${results.length} native MCP server(s).`
        : `Reloaded ${results.length} native MCP server(s).`,
      failed.length ? "warning" : "info",
    );
    return { ok: !failed.length, configured: true, servers: results };
  }

  private restartDaemon(connection: DaemonTransportConnection): JsonRpcPayload {
    this.emitSlash(
      connection,
      "Restarting native daemon — re-run `xerxes` after it shuts down.",
    );
    queueMicrotask(() => {
      const restart = this.onRestart ? this.onRestart() : this.stop();
      void Promise.resolve(restart).catch((error) =>
        this.broadcast("notification", {
          level: "error",
          message: `Native daemon restart failed: ${errorMessage(error)}`,
        }),
      );
    });
    return { ok: true };
  }

  private async showUpdate(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    const git = await gitUpdateStatus({ cwd: session?.cwd ?? process.cwd() });
    const summary = formatGitUpdateStatus(git);
    this.emitSlash(
      connection,
      `Xerxes Bun runtime \`${BUN_DAEMON_BUILD_ID}\`\nGit: ${summary}\nRun: \`bun run xerxes update\`.`,
    );
    return { ok: true, git, summary };
  }

  private retryConnection(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    this.emitSlash(
      connection,
      "Retrying the latest native provider turn for this session.",
    );
    return this.retryLastTurn(connection, session);
  }

  private async listTools(
    connection: DaemonTransportConnection,
  ): Promise<JsonRpcPayload> {
    const tools = this.toolCatalog ? await this.toolCatalog.listTools() : [];
    if (tools.length) {
      this.emitSlash(
        connection,
        [
          `Native tools (${tools.length}):`,
          ...tools.map(
            (tool) =>
              `  \`${tool.name}\`${tool.description ? ` — ${tool.description}` : ""}`,
          ),
        ].join("\n"),
      );
      return { ok: true, tools: tools.map((tool) => ({ ...tool })) };
    }
    const count = numberValue(this.runtime.status().tools);
    this.emitSlash(
      connection,
      count
        ? `Native tool count: ${count}.`
        : "No native tool catalogue is attached to this daemon runtime.",
    );
    return { ok: true, tools: [], count };
  }

  private async initializeProject(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    args: string,
  ): Promise<JsonRpcPayload> {
    const projectDirectory = session?.cwd ?? process.cwd();
    const key = connection.activeSessionKey;
    await this.runtime.openSession(key);
    this.emitSlash(
      connection,
      `Starting native project initialization for \`${projectDirectory}\`.`,
    );
    const turn = this.submitTrackedTurn(
      key,
      projectInitializationPrompt(projectDirectory, args),
      (event) => this.emit(connection, event.type, event.payload),
      connection,
    );
    void turn
      .then(async () => {
        const active = this.runtime.sessionStatus(key);
        await this.refreshSkills(active);
        const workspace = await loadProjectAgentWorkspace(projectDirectory);
        this.emitSlash(
          connection,
          `Project initialization turn finished. Loaded ${workspace.loadedFiles.length} project workspace file(s) and ${this.skillRegistry.all().length} native skill(s).`,
        );
      })
      .catch((error) =>
        this.emit(connection, "notification", {
          level: "error",
          message: errorMessage(error),
        }),
      );
    return {
      ok: true,
      queued: true,
      project_directory: projectDirectory,
      agents_directory: projectAgentsDir(projectDirectory),
    };
  }

  private async showWorkspace(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    raw: string,
  ): Promise<JsonRpcPayload> {
    const action = raw.trim().toLowerCase();
    if (action === "init") {
      return this.initializeProject(connection, session, "");
    }
    if (action && action !== "status") {
      this.emitSlash(
        connection,
        "Usage: `/workspace [status|init]`.",
        "warning",
      );
      return { ok: false, error: "invalid workspace command" };
    }
    const projectDirectory = session?.cwd ?? process.cwd();
    const workspace = await loadProjectAgentWorkspace(projectDirectory);
    const lines = [
      `Project dir:    \`${projectDirectory}\``,
      `Agent workspace: \`${session?.workspace ?? "(no session)"}\``,
      `Agent id:        \`${session?.agentId ?? "default"}\``,
      `Project .agents: \`${projectAgentsDir(projectDirectory)}\` (${workspace.prompt ? "ready" : "not initialized"})`,
    ];
    if (workspace.loadedFiles.length) {
      lines.push(
        "Loaded project context:",
        ...workspace.loadedFiles.map((path) => `  \`${path}\``),
      );
    }
    this.emitSlash(connection, lines.join("\n"));
    return {
      ok: true,
      project_directory: projectDirectory,
      workspace_directory: session?.workspace ?? "",
      agents_directory: workspace.agentsDir,
      loaded_files: workspace.loadedFiles,
    };
  }

  private async generateImage(
    connection: DaemonTransportConnection,
    raw: string,
  ): Promise<JsonRpcPayload> {
    const prompt = raw.trim();
    if (!prompt) {
      this.emitSlash(connection, "Usage: `/image <prompt>`.", "warning");
      return { ok: false, error: "image prompt is required" };
    }
    const synthetic = [
      "Generate an image matching this brief and report the saved path.",
      "Use the native image-generation tool if it is attached to this runtime.",
      "",
      prompt,
    ].join("\n");
    void this.submitTrackedTurn(
      connection.activeSessionKey,
      synthetic,
      (event) => this.emit(connection, event.type, event.payload),
      connection,
    ).catch((error) =>
      this.emit(connection, "notification", {
        level: "error",
        message: errorMessage(error),
      }),
    );
    return { ok: true, queued: true };
  }

  private async forwardUiControl(
    connection: DaemonTransportConnection,
    action: string,
    argument: string,
  ): Promise<JsonRpcPayload> {
    if (!isDaemonUiAction(action)) {
      return { ok: false, error: `unsupported UI action: ${action}` };
    }
    const input: DaemonUiControlInput = {
      action,
      argument,
      sessionKey: connection.activeSessionKey,
    };
    this.emit(connection, "ui_command", {
      action,
      argument,
      session_key: connection.activeSessionKey,
    });
    const result = await this.uiControl?.execute(input);
    this.emitSlash(
      connection,
      result?.message ??
        `Sent native UI command \`/${action}\` to the connected client.`,
    );
    return {
      ok: true,
      action,
      ...(result?.payload ? { result: result.payload } : {}),
    };
  }

  private async tryInvokeSkillShorthand(
    connection: DaemonTransportConnection,
    token: string,
    args: string,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    await this.refreshSkills(session);
    const [name] = token.split(":", 1);
    const skill = name ? this.skillRegistry.get(name) : undefined;
    if (skill && skillMatchesPlatform(skill)) {
      return this.invokeSkill(
        connection,
        `${token}${args ? ` ${args}` : ""}`,
        session,
      );
    }
    const canonical = resolveCommand(`/${token}`);
    if (canonical) {
      this.emitSlash(
        connection,
        `Native handler coverage defect for /${canonical.name}; this command is registered but not routed.`,
        "error",
      );
      return {
        ok: false,
        error: `unrouted native slash command: /${canonical.name}`,
      };
    }
    this.emitSlash(
      connection,
      `Unknown command: /${token} (type /help).`,
      "warning",
    );
    return { ok: false, error: `Unknown slash command: /${token}` };
  }

  private async configureSampling(
    connection: DaemonTransportConnection,
    raw: string,
  ): Promise<JsonRpcPayload> {
    const input = raw.trim();
    if (!input) {
      const sampling = samplingConfig(this.runtime.status());
      const body = [
        "Native next-turn sampling:",
        ...NATIVE_SAMPLING_KEYS.map(
          (name) =>
            `  \`${name}\` = \`${sampling[name] ?? "(provider default)"}\``,
        ),
        "",
        "Use `/sampling <key> <value>` or `/sampling reset`.",
      ].join("\n");
      this.emitSlash(connection, body);
      return { ok: true, sampling };
    }

    if (input.toLowerCase() === "reset") {
      const cleared = Object.fromEntries(
        NATIVE_SAMPLING_KEYS.map((key) => [key, null]),
      );
      this.runtime.reload({
        ...cleared,
        temperature: DEFAULT_TEMPERATURE,
        top_k: DEFAULT_TOP_K,
      });
      const active = this.profileStore.active();
      if (active) {
        this.profileStore.updateSampling(active.name, cleared);
      }
      this.emitSlash(
        connection,
        `Restored native sampling defaults (temperature ${DEFAULT_TEMPERATURE}, top_k ${DEFAULT_TOP_K}).`,
      );
      return { ok: true, sampling: samplingConfig(this.runtime.status()) };
    }

    const [rawName, rawValue, ...extra] = input.split(/\s+/);
    const name = rawName?.toLowerCase() ?? "";
    if (!rawValue || extra.length || !isNativeSamplingKey(name)) {
      this.emitSlash(
        connection,
        "Usage: `/sampling <key> <value>` or `/sampling reset`.",
        "warning",
      );
      return { ok: false, error: "invalid sampling command" };
    }
    const efforts = selectableEfforts(await this.reasoningLevels());
    const value = parseNativeSamplingValue(name, rawValue, efforts);
    if (value === undefined) {
      this.emitSlash(connection, invalidSamplingMessage(name, efforts), "warning");
      return { ok: false, error: `invalid ${name}` };
    }

    this.runtime.reload({ [name]: value });
    const active = this.profileStore.active();
    if (active) {
      this.profileStore.updateSampling(active.name, { [name]: value });
    }
    const sampling = {
      ...samplingConfig(this.runtime.status()),
      [name]: value,
    };
    this.emitSlash(
      connection,
      `Native next-turn sampling \`${name}\` = \`${value}\`.`,
    );
    return { ok: true, sampling };
  }

  private listAgents(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): JsonRpcPayload {
    try {
      const agents = this.agentDefinitionLoader(session?.cwd ?? process.cwd());
      const payload = agents.map(agentDefinitionPayload);
      if (!payload.length) {
        this.emitSlash(
          connection,
          "No native agent definitions are available.",
        );
        return { ok: true, agents: [] };
      }
      const lines = [
        `Native agent definitions (${payload.length}):`,
        ...payload.map(
          (agent) =>
            `  \`${String(agent.name)}\`${agent.source === "built-in" ? "" : ` [${String(agent.source)}]`} — ${String(agent.description) || "No description"}`,
        ),
      ];
      this.emitSlash(connection, lines.join("\n"));
      return { ok: true, agents: payload };
    } catch (error) {
      const message = errorMessage(error);
      this.emitSlash(
        connection,
        `Agent definition discovery failed: \`${message}\``,
        "error",
      );
      return { ok: false, error: message };
    }
  }

  private listPlatforms(connection: DaemonTransportConnection): JsonRpcPayload {
    const data = this.channelStatusData();
    if (!data.available) {
      this.emitSlash(connection, "No channel platform manager is configured.");
      return {
        ok: true,
        platforms: [],
        channels_available: false,
        channels_configured: false,
      };
    }
    if (!data.channels.length) {
      this.emitSlash(connection, "No messaging platforms are configured.");
      return {
        ok: true,
        platforms: [],
        channels_available: true,
        channels_configured: false,
      };
    }
    const lines = [
      `Messaging platforms (${data.channels.length}):`,
      ...data.channels.map(
        (platform) =>
          `  \`${String(platform.name)}\` — ${platform.enabled === true ? "enabled" : "disabled"}`,
      ),
    ];
    this.emitSlash(connection, lines.join("\n"));
    return {
      ok: true,
      platforms: data.channels,
      channels_available: true,
      channels_configured: data.configured,
    };
  }

  private async compactSession(
    connection: DaemonTransportConnection,
    notify = true,
  ): Promise<JsonRpcPayload> {
    // Preserve the command's immediate refusal semantics instead of queueing a
    // manual compaction behind a live turn.
    const active = this.runtime.sessionStatus(connection.activeSessionKey);
    const result = active?.activeTurnId
      ? await this.compactSessionByKeyUnlocked(
        connection.activeSessionKey,
        notify ? connection : undefined,
      )
      : await this.compactSessionByKey(
        connection.activeSessionKey,
        notify ? connection : undefined,
      );
    if (result.ok === true && result.compacted === true) {
      const session = this.runtime.sessionStatus(connection.activeSessionKey);
      if (session) {
        this.emitStatus(connection, session);
      }
    }
    return result;
  }

  /**
   * Provider-backed compaction of one session's transcript. Notifications go
   * to `notify` when a connection is supplied; cron and auto-compaction paths
   * without an owning connection run silently. Messages appended while the
   * summary is generated (for example an idle steer) are preserved.
   */
  private compactSessionByKey(
    sessionKey: string,
    notify: DaemonTransportConnection | undefined,
    verb = "Compacted",
    reason = "compact",
  ): Promise<JsonRpcPayload> {
    return this.withSessionOperation(sessionKey, () =>
      this.compactSessionByKeyUnlocked(sessionKey, notify, verb, reason)
    );
  }

  private async compactSessionByKeyUnlocked(
    sessionKey: string,
    notify: DaemonTransportConnection | undefined,
    verb = "Compacted",
    /** Recorded in the archive and the metadata stamp; who asked for this pass. */
    reason = "compact",
  ): Promise<JsonRpcPayload> {
    const session = this.runtime.sessionStatus(sessionKey);
    if (!session) {
      if (notify) {
        this.emitSlash(notify, "No active session to compact.", "warning");
      }
      return { ok: false, error: "no active session" };
    }
    if (session.activeTurnId) {
      if (notify) {
        this.emitSlash(
          notify,
          "Cannot compact while a turn is running. Use `/stop` first.",
          "warning",
        );
      }
      return { ok: false, error: "turn is running" };
    }
    const model = session.model || stringValue(this.runtime.status().model);
    if (!model) {
      const error = "model is not configured; select a provider model before compacting";
      if (notify) this.emitSlash(notify, error, "warning");
      return { ok: false, error };
    }
    // Deferred on purpose: compaction often answers "nothing to compact" without
    // consulting a provider at all, and building the client eagerly made that
    // no-op require a constructible one. See lazyCompactionCompletionPort.
    const completion = lazyCompactionCompletionPort(
      () => createCompactionClient(model, this.profileStore?.active(), this.runtime.status()),
      model,
    );
    // Compaction is one long provider call with nothing between the command
    // and its result, so the screen sat dead for as long as the summary took.
    // Announcing the work and marking the session busy gives the same feedback
    // a turn gets; the status is restored in `finally` so a failure cannot
    // strand the session as permanently working.
    const previousStatus = session.status;
    if (notify) {
      this.emitSlash(
        notify,
        `Compacting ${session.messages.length} message(s) with \`${model}\`…`,
      );
      session.status = "working";
      this.emitStatus(notify, session);
    }
    try {
      const archivePath = await this.precompactArchivePath(session.id);
      const outcome = await compactMessagesIfNeeded({
        ...(archivePath === undefined ? {} : { archivePath }),
        completion: completion.port,
        messages: session.messages,
        model,
        reason,
      });
      if (!outcome.compacted) {
        if (outcome.reason === "unchanged") {
          if (notify) {
            this.emitSlash(notify, "Nothing to compact.");
          }
          return { ok: true, compacted: false };
        }
        const failure = outcome.error ?? outcome.reason;
        if (notify) {
          this.emitSlash(notify, `Compaction failed: ${failure}`, "error");
        }
        return { ok: false, error: failure };
      }
      // Anything appended while the summary was in flight is newer than the
      // compacted window and must survive the swap.
      const appended = session.messages.slice(outcome.originalCount);
      session.messages = [
        ...outcome.messages,
        ...appended,
      ] as DaemonSession["messages"];
      session.metadata.last_compaction = outcome.stamp;
      // Compaction dropped full file contents out of the model's context, so
      // its belief about what a file looks like is no longer trustworthy:
      // retire the read-guard state and force fresh reads before the next
      // edit. Covers /compact and auto-compact — both funnel through here.
      fileStateTracker.clearSession(session.id);
      session.metadata[FILE_READS_METADATA_KEY] = [];
      // A compaction that worked — by hand or automatically — retires the
      // failure evidence, so `/compact` is a way back from the bail-out.
      this.clearAutoCompactFailures(sessionKey);
      await this.runtime.flushSessions("rewrite");
      if (notify) {
        const replaced = outcome.originalCount - outcome.messages.length;
        const body = `${verb} ${replaced} message(s): ${outcome.stamp.tokens_before} → ${outcome.stamp.tokens_after} tokens.`;
        this.emitCompactionLog(
          notify,
          body,
          outcome.stamp.tokens_before,
          outcome.stamp.tokens_after,
          reason === "auto-compact",
        );
        if (outcome.stamp.archive_error !== undefined) {
          // The user never asked for auto-compaction, so a silently
          // unrecoverable transcript is not an acceptable outcome of it.
          this.emitSlash(
            notify,
            `Pre-compaction transcript could not be archived: ${outcome.stamp.archive_error}`,
            "warning",
          );
        }
      }
      return {
        ok: true,
        compacted: true,
        tokens_before: outcome.stamp.tokens_before,
        tokens_after: outcome.stamp.tokens_after,
        ...(outcome.stamp.archive_path === undefined
          ? {}
          : { archive_path: outcome.stamp.archive_path }),
      };
    } catch (error) {
      if (notify) {
        this.emitSlash(
          notify,
          `Compaction failed: ${errorMessage(error)}`,
          "error",
        );
      }
      return { ok: false, error: errorMessage(error) };
    } finally {
      // Do not overwrite state established by work that began independently
      // while the provider call was in flight.
      if (!session.activeTurnId) {
        session.status = previousStatus;
      }
      if (notify) {
        this.emitStatus(notify, session);
      }
      await completion.close();
    }
  }

  private withSessionOperation<T>(
    sessionKey: string,
    operation: () => Promise<T>,
  ): Promise<T> {
    const previous = this.sessionOperations.get(sessionKey) ?? Promise.resolve();
    const result = previous.catch(() => undefined).then(operation);
    const tail = result.then(
      () => undefined,
      () => undefined,
    );
    this.sessionOperations.set(sessionKey, tail);
    void tail.then(() => {
      if (this.sessionOperations.get(sessionKey) === tail) {
        this.sessionOperations.delete(sessionKey);
      }
    });
    return result;
  }

  /**
   * Where this session's pre-compaction transcript is archived.
   *
   * Compaction replaces `session.messages` and the very next flush overwrites
   * the single per-session JSON, so without this sidecar the original history
   * leaves memory and disk on the same tick — for an auto-compaction the user
   * never asked for.
   *
   * An archive is only written beside a transcript that is actually there:
   * a host whose transcripts live in a directory this server was not told
   * about (it is configured on the runtime, not here) would otherwise
   * accumulate orphan archives under the default home, next to nothing.
   */
  private async precompactArchivePath(
    sessionId: string,
  ): Promise<string | undefined> {
    if (!looksLikeSessionId(sessionId)) return undefined;
    const directory = this.sessionArchiveDirectory;
    if (!this.sessionArchiveDirectoryConfigured) {
      const transcript = join(directory, `${sessionId}.json`);
      const found = await stat(transcript).then(
        (entry) => entry.isFile(),
        () => false,
      );
      if (!found) return undefined;
    }
    return precompactArchivePathFor(directory, sessionId);
  }

  private resolvedAutoCompactThreshold(): number {
    const status = this.runtime.status();
    if (status.auto_compact_threshold !== undefined) {
      return normalizeCompactionThreshold(
        numberValue(status.auto_compact_threshold),
      );
    }
    return this.autoCompactThreshold;
  }

  /**
   * Compact the session before a turn when the estimated context usage has
   * reached the configured threshold. Concurrent submissions join the same
   * compaction, and a compaction failure only warns — the turn still runs.
   */
  private async autoCompactIfDue(
    sessionKey: string,
    owner: DaemonTransportConnection | undefined,
  ): Promise<void> {
    const session = this.runtime.sessionStatus(sessionKey);
    if (!session || session.activeTurnId || session.messages.length < 2) {
      return Promise.resolve();
    }
    const model = session.model || stringValue(this.runtime.status().model);
    if (!model) {
      return Promise.resolve();
    }
    // The prompt budget, not the raw window: a prompt that fills the window
    // leaves the reply nowhere to go, and the request fails as a 400 that no
    // local meter predicted.
    const limit = this.promptBudget(model);
    if (!limit) {
      return Promise.resolve();
    }
    const used = sessionContextTokens(session, model);
    // One threshold source for main sessions and delegated children alike.
    const due = compactionThresholdTokens(
      limit,
      this.resolvedAutoCompactThreshold(),
    );
    if (due <= 0) {
      this.warnAutoCompactDisabled(sessionKey, owner, used, limit);
      return Promise.resolve();
    }
    if (used < due) {
      return Promise.resolve();
    }
    const failures = this.autoCompactFailures.get(sessionKey) ?? 0;
    if (failures >= MAX_AUTO_COMPACT_FAILURES) {
      // Silent from here on: the actionable line was emitted on the attempt
      // that reached the limit, and repeating it every turn is the same noise
      // the retry loop was.
      return Promise.resolve();
    }
    if (owner) {
      this.emitSlash(
        owner,
        `Context at ${((used / limit) * 100).toFixed(0)}% — auto-compacting before this turn…`,
      );
    }
    try {
      const result = await this.compactSessionByKeyUnlocked(
        sessionKey,
        owner,
        "Auto-compacted",
        "auto-compact",
      );
      // `compacted: false` counts as a failure. It leaves the window exactly
      // as full as it was, so the next turn would re-run the same
      // full-window summarization call and reach the same conclusion.
      if (result.ok !== true || result.compacted !== true) {
        const failure = stringValue(result.error) || "nothing to compact";
        this.recordAutoCompactFailure(sessionKey, owner, failure);
      }
    } catch (error) {
      this.recordAutoCompactFailure(sessionKey, owner, errorMessage(error));
    }
  }

  private recordAutoCompactFailure(
    sessionKey: string,
    owner: DaemonTransportConnection | undefined,
    reason: string,
  ): void {
    const failures = (this.autoCompactFailures.get(sessionKey) ?? 0) + 1;
    this.autoCompactFailures.set(sessionKey, failures);
    if (!owner) {
      return;
    }
    if (failures < MAX_AUTO_COMPACT_FAILURES) {
      this.emitSlash(owner, `Auto-compaction skipped: ${reason}.`, "warning");
      return;
    }
    this.emitSlash(
      owner,
      `Auto-compaction failed ${failures} times in a row (${reason}) and is now off for this session. `
        + "Run `/compact` to see the error, or `/new` to start a fresh session.",
      "error",
    );
  }

  /** Reset after a deliberate history change so the session gets a clean slate. */
  private clearAutoCompactFailures(sessionKey: string): void {
    this.autoCompactFailures.delete(sessionKey);
    this.autoCompactDisabledWarned.delete(sessionKey);
  }

  /**
   * Name a session from its opening prompt the moment work starts.
   *
   * The model-written title only exists once the first exchange *ends*, so
   * every chat spent its entire first turn — minutes, for a dispatched
   * background chat — rendering as an anonymous `—` in Agent View, and a
   * session whose three title attempts all failed wore that dash forever.
   *
   * The placeholder is written with `title_derived`, the flag
   * `maybeGenerateTitle` already reads as "replaceable", so the generated
   * title still wins the moment it lands and an explicit `/title` wins over
   * both. Sourced from the session's FIRST user message rather than this
   * turn's text, so an old untitled chat is backfilled with the prompt that
   * actually opened it.
   */
  private seedProvisionalTitle(sessionKey: string, text: string): void {
    if (!this.resolvedAutoTitle()) return;
    const session = this.runtime.sessionStatus(sessionKey);
    if (!session) return;
    // Any title at all wins — explicit ones must never be clobbered, and a
    // provisional one is already the opening prompt.
    if (stringValue(session.metadata.title)) return;
    const title = provisionalTitleFrom(
      firstExchangeText(session, "user") || text,
    );
    if (!title) return;
    session.metadata.title = title;
    session.metadata.title_derived = true;
    this.broadcast("session_title", { session_id: session.id, title });
  }

  /**
   * Generate a model-written title after a session's first exchange.
   *
   * Fires on every turn-end edge but spends a provider call exactly once per
   * session (`attemptSessionTitle`). Everything about it is best-effort:
   * a session the user already titled, a disabled setting, a missing provider,
   * or a failed generation all leave the existing title state untouched.
   */
  private maybeGenerateTitle(sessionKey: string): void {
    const session = this.runtime.sessionStatus(sessionKey);
    if (!session) return;
    if (!this.resolvedAutoTitle()) return;
    // Explicit titles — set through /title or session.title — always win.
    // `title_derived` is retained only as a migration seam for transcripts
    // written by older builds that used the first message as a fallback.
    const existing = stringValue(session.metadata.title);
    if (existing && session.metadata.title_derived !== true) return;
    // Titles are generated from the opening exchange, so only a session whose
    // whole history is that first exchange qualifies. A session that already
    // holds several turns (resumed or seeded history) keeps its fallback
    // rather than paying a provider call for a name it has lived without.
    const user = firstExchangeText(session, "user");
    const assistant = firstExchangeText(session, "assistant");
    if (!user || !assistant) return;
    // Tool-using opening exchanges contain more than two transcript rows; the
    // completed-turn count is the stable definition of "first exchange".
    //
    // A short window rather than exactly turn 1: the title is always built
    // from the FIRST exchange (see `firstExchangeText` above), so it is
    // byte-identical at turn 1 and turn 3. Insisting on turn 1 meant a single
    // transient provider failure orphaned a brand-new session permanently,
    // with no way back. The window is deliberately small — this is a retry
    // for new sessions, not a backfill of long-lived history.
    if (session.turnCount < 1 || session.turnCount > TITLE_RETRY_TURN_WINDOW) return;

    const attempt = attemptSessionTitle(session.id, () =>
      generateSessionTitle({
        userText: user,
        assistantText: assistant,
        sessionModel: session.model,
        profile: this.profileStore.active(),
        // Bound the background call to the session's lifetime: a title
        // request for an evicted or reset session must not outlive it.
        signal: this.sessionSignal(sessionKey),
        ...(this.titleClientFactory ? { clientFactory: this.titleClientFactory } : {}),
      }));
    if (!attempt) return;
    void attempt.then((title) => {
      if (!title) return;
      // Persist through the session-operation queue: a flush outside it can
      // land mid-compaction and lose the generation race, failing the
      // compaction rewrite that owns the transcript.
      void this.withSessionOperation(sessionKey, async () => {
        // Re-read: a later turn or an explicit /title may have landed while
        // the provider answered, and neither may be overwritten.
        const current = this.runtime.sessionStatus(sessionKey);
        if (!current) return;
        const currentTitle = stringValue(current.metadata.title);
        if (currentTitle && current.metadata.title_derived !== true) return;
        current.metadata.title = title;
        delete current.metadata.title_derived;
        try {
          await this.runtime.flushSessions();
        } catch {
          return;
        }
        this.broadcast("session_title", {
          session_id: current.id,
          title,
        });
      });
    });
  }

  private resolvedAutoTitle(): boolean {
    const setting = this.runtime.status().auto_title;
    if (typeof setting === "boolean") return setting;
    return this.autoTitle;
  }

  private warnAutoCompactDisabled(
    sessionKey: string,
    owner: DaemonTransportConnection | undefined,
    used: number,
    limit: number,
  ): void {
    if (used < Math.floor(limit * AUTO_COMPACT_DISABLED_WARNING_FRACTION)) {
      // Drop the latch once the window is comfortable again, so a session that
      // is manually compacted and then refills is warned a second time.
      this.autoCompactDisabledWarned.delete(sessionKey);
      return;
    }
    if (!owner || this.autoCompactDisabledWarned.has(sessionKey)) {
      return;
    }
    this.autoCompactDisabledWarned.add(sessionKey);
    this.emitSlash(
      owner,
      `Context at ${((used / limit) * 100).toFixed(0)}% of the ${limit.toLocaleString()}-token prompt budget `
        + "and auto-compaction is disabled (`auto_compact_threshold` is 0). Run `/compact` before the provider "
        + "rejects the next request.",
      "warning",
    );
  }

  private showSessionBudget(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): JsonRpcPayload {
    if (!session) {
      this.emitSlash(connection, "No active session yet.", "warning");
      return { ok: false, error: "no active session" };
    }
    const model = session.model || stringValue(this.runtime.status().model);
    const contextLimit = this.contextLimit(model);
    // The same measurement and the same limit the auto-compaction trigger
    // uses. Reading them from two different estimates is how `/context` and
    // the status bar came to disagree about one session.
    const promptBudget = this.promptBudget(model);
    const used = sessionContextTokens(session, model);
    const remaining = Math.max(0, promptBudget - used);
    const percent = promptBudget ? (used / promptBudget) * 100 : 0;
    this.emitSlash(
      connection,
      [
        contextLimit > 0
          ? `Context window: ${contextLimit.toLocaleString()} tokens for \`${model}\``
          : model
            ? `Context window: unknown (provider reported no capacity for \`${model}\`)`
            : "Context window: unknown (model not configured)",
        promptBudget && promptBudget < contextLimit
          ? `Prompt budget: ${promptBudget.toLocaleString()} (window minus the reply this model may emit)`
          : "",
        promptBudget
          ? `Used: ${used.toLocaleString()} (${percent.toFixed(1)}%) · Remaining: ${remaining.toLocaleString()}`
          : `Used: ${used.toLocaleString()} · Remaining: unknown`,
      ].filter(Boolean).join("\n"),
    );
    return {
      ok: true,
      context_limit: contextLimit,
      prompt_budget: promptBudget,
      ...(promptBudget > 0 ? { remaining_tokens: remaining } : {}),
      used_tokens: used,
    };
  }

  private showSessionCost(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): JsonRpcPayload {
    if (!session) {
      this.emitSlash(connection, "No active session yet.", "warning");
      return { ok: false, error: "no active session" };
    }
    const model = session.model || stringValue(this.runtime.status().model);
    const cost = calcCost(
      model,
      session.totalInputTokens,
      session.totalOutputTokens,
    );
    this.emitSlash(
      connection,
      `Estimated cost: \`$${cost.toFixed(4)}\` (model: \`${model || "(not configured)"}\`).`,
    );
    return {
      ok: true,
      cost_usd: cost,
      model,
      input_tokens: session.totalInputTokens,
      output_tokens: session.totalOutputTokens,
    };
  }

  private runDoctor(connection: DaemonTransportConnection): JsonRpcPayload {
    const diagnostics = runAllDoctorChecks();
    this.emitSlash(
      connection,
      `Diagnostics:\n${formatDoctorReport(diagnostics)}`,
    );
    return {
      ok: true,
      diagnostics: diagnostics.map((diagnosis) => ({ ...diagnosis })),
    };
  }

  private showSessionInsights(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): JsonRpcPayload {
    if (!session) {
      this.emitSlash(connection, "No active session yet.", "warning");
      return { ok: false, error: "no active session" };
    }
    const counts = new Map<string, number>();
    for (const execution of session.toolExecutions) {
      const name = toolExecutionName(execution);
      if (name) {
        counts.set(name, (counts.get(name) ?? 0) + 1);
      }
    }
    if (!counts.size) {
      this.emitSlash(connection, "No tools invoked in this session yet.");
      return { ok: true, tools: [] };
    }
    const tools = [...counts.entries()]
      .sort(
        ([leftName, leftCount], [rightName, rightCount]) =>
          rightCount - leftCount || leftName.localeCompare(rightName),
      )
      .slice(0, 10)
      .map(([name, count]) => ({ name, count }));
    this.emitSlash(
      connection,
      [
        "Top tools this session:",
        ...tools.map(
          (tool) =>
            `  \`${tool.name}\` — ${tool.count} call${tool.count === 1 ? "" : "s"}`,
        ),
      ].join("\n"),
    );
    return { ok: true, tools };
  }

  private async reloadRuntime(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    this.runtime.reload({});
    const active =
      session ?? this.runtime.sessionStatus(connection.activeSessionKey);
    if (active) {
      this.emitInitDone(connection, active);
      this.emitStatus(connection, active);
    }
    await this.refreshSkills(active);
    this.emitSlash(
      connection,
      `Reloaded native runtime configuration and ${this.skillRegistry.all().length} discovered skill(s).`,
    );
    return {
      ok: true,
      runtime: this.runtimeStatusWithChannels(),
      skills: this.skillRegistry.all().length,
    };
  }

  private async saveActiveSession(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    title: string,
    notify = true,
  ): Promise<JsonRpcPayload> {
    if (!session) {
      if (notify) {
        this.emitSlash(connection, "No active session to save.", "warning");
      }
      return { ok: false, error: "no active session" };
    }
    if (!sessionHasHistory(session)) {
      if (notify) {
        this.emitSlash(
          connection,
          "Nothing to save yet — this session has no messages.",
          "warning",
        );
      }
      return { ok: false, error: "session has no history" };
    }
    if (title) {
      session.metadata.title = title;
      delete session.metadata.title_derived;
    }
    try {
      await this.runtime.flushSessions();
      const persisted = (await this.runtime.listSavedSessions()).find(
        (candidate) => candidate.id === session.id,
      );
      if (!persisted) {
        throw new Error(
          "session persistence did not produce a saved transcript",
        );
      }
      const named = title ? ` as \`${persisted.title || title}\`` : "";
      if (notify) {
        this.emitSlash(
          connection,
          `Saved session \`${persisted.id}\`${named} to \`${persisted.path}\`.`,
        );
      }
      return {
        ok: true,
        session: savedSessionPayload(persisted),
        ...(title ? { title: persisted.title || title } : {}),
      };
    } catch (error) {
      const message = errorMessage(error);
      if (notify) {
        this.emitSlash(
          connection,
          `Session save failed: \`${message}\``,
          "error",
        );
      }
      return { ok: false, error: message };
    }
  }

  /**
   * Undo recorded FileEditTool changes — one path, or every recorded path.
   * The daemon's own execution record is the only source of truth: edits
   * are reverse-applied strictly newest-first and only while each inserted
   * span is still present verbatim. Any drift refuses the undo instead of
   * corrupting the file.
   */
  private async undoChanges(
    session: DaemonSession | undefined,
    requestedPath: string,
  ): Promise<JsonRpcPayload> {
    if (!session) return { ok: false, error: "no active session" };
    type Edit = { readonly path: string; readonly oldString: string; readonly newString: string };
    const edits: Edit[] = [];
    for (const exec of session.toolExecutions) {
      if (!exec || typeof exec !== "object") continue;
      const record = exec as Record<string, unknown>;
      if (record.name !== "FileEditTool") continue;
      const args = record.inputs ?? record.arguments;
      if (!args || typeof args !== "object" || Array.isArray(args)) continue;
      const path = stringValue((args as Record<string, unknown>).file_path);
      if (!path || (requestedPath && path !== requestedPath)) continue;
      edits.push({
        path,
        oldString: stringValue((args as Record<string, unknown>).old_string),
        newString: stringValue((args as Record<string, unknown>).new_string),
      });
    }
    if (!edits.length) {
      return {
        ok: false,
        error: `no reversible recorded edits${requestedPath ? ` for ${requestedPath}` : ""}`,
      };
    }
    const byPath = new Map<string, Edit[]>();
    for (const edit of edits) {
      const list = byPath.get(edit.path) ?? [];
      list.push(edit);
      byPath.set(edit.path, list);
    }
    const results: Array<{
      readonly path: string;
      readonly ok: boolean;
      readonly reverted?: number;
      readonly error?: string;
    }> = [];
    for (const [path, pathEdits] of byPath) {
      try {
        const file = Bun.file(path);
        if (!(await file.exists())) {
          results.push({ path, ok: false, error: "file is gone — nothing to undo" });
          continue;
        }
        let content = await file.text();
        let reverted = 0;
        let refused = false;
        for (let index = pathEdits.length - 1; index >= 0; index--) {
          const edit = pathEdits[index];
          if (!edit) break;
          const { oldString, newString } = edit;
          // A deleted span (empty new_string) cannot be re-located safely.
          if (!newString || !content.includes(newString)) {
            results.push({
              path,
              ok: false,
              error: `file changed since edit ${pathEdits.length - index} of ${pathEdits.length} — refusing to undo blindly`,
            });
            refused = true;
            break;
          }
          content = content.replace(newString, oldString);
          reverted += 1;
        }
        if (refused) continue;
        await Bun.write(path, content);
        results.push({ path, ok: true, reverted });
      } catch (error) {
        results.push({ path, ok: false, error: errorMessage(error) });
      }
    }
    return {
      ok: results.every((result) => result.ok),
      results,
      reverted: results.reduce((sum, result) => sum + (result.reverted ?? 0), 0),
    };
  }

  /**
   * Create a git worktree beside the project (`<project>-<name>` on a branch
   * named after it) so a task can start in an isolated checkout. Refuses
   * with a typed error outside a repo or on an existing path — never
   * guesses a fallback directory.
   */
  private async createWorktree(
    session: DaemonSession | undefined,
    rawName: string,
  ): Promise<JsonRpcPayload> {
    if (!session) return { ok: false, error: "no active session" };
    const name = rawName.trim().replace(/[^a-zA-Z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48);
    if (!name) return { ok: false, error: "worktree name is required" };
    const cwd = session.cwd;
    const inside = Bun.spawnSync(["git", "-C", cwd, "rev-parse", "--is-inside-work-tree"], {
      stdin: "ignore", stdout: "pipe", stderr: "pipe",
    });
    if (inside.exitCode !== 0 || new TextDecoder().decode(inside.stdout).trim() !== "true") {
      return { ok: false, error: `not a git work tree: ${cwd}` };
    }
    const path = join(dirname(cwd), `${basename(cwd)}-${name}`);
    if (existsSync(path)) return { ok: false, error: `worktree path already exists: ${path}` };
    const run = (args: string[]): { code: number; stderr: string } => {
      const proc = Bun.spawnSync(["git", "-C", cwd, ...args], { stdin: "ignore", stdout: "pipe", stderr: "pipe" });
      return { code: proc.exitCode ?? 1, stderr: new TextDecoder().decode(proc.stderr).trim() };
    };
    // Fresh branch when the name is free; attach when it already exists.
    let created = run(["worktree", "add", path, "-b", name]);
    if (created.code !== 0 && !created.stderr.includes("already exists")) {
      return { ok: false, error: created.stderr || "git worktree add failed" };
    }
    if (created.code !== 0) created = run(["worktree", "add", path, name]);
    if (created.code !== 0) return { ok: false, error: created.stderr || "git worktree add failed" };
    return { ok: true, path, branch: name };
  }

  private async setSessionTitle(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    title: string,
    notify = true,
  ): Promise<JsonRpcPayload> {
    if (!session) {
      if (notify) {
        this.emitSlash(connection, "No active session yet.", "warning");
      }
      return { ok: false, error: "no active session" };
    }
    if (title) {
      session.metadata.title = title;
      delete session.metadata.title_derived;
      await this.runtime.flushSessions();
      // Same broadcast the auto-titler emits: every surface showing this
      // session (the renamer's own header, other clients' sidebars) learns
      // the new title instead of waiting for the next full refresh.
      this.broadcast("session_title", { session_id: session.id, title });
    }
    const current = stringValue(session.metadata.title);
    if (notify) {
      this.emitSlash(connection, `Session title: \`${current || "(unset)"}\`.`);
    }
    return { ok: true, title: current };
  }

  private async resumeSavedSession(
    connection: DaemonTransportConnection,
    query: string,
  ): Promise<JsonRpcPayload> {
    const saved = await this.runtime.listSavedSessions();
    const needle = query.trim().toLowerCase();
    if (!needle) {
      if (!saved.length) {
        this.emitSlash(connection, "No saved sessions found.");
        return { ok: true, sessions: [] };
      }
      const sessions = saved.slice(0, 20).map(savedSessionPayload);
      this.emitSlash(
        connection,
        [
          `Saved sessions (${saved.length}):`,
          ...saved
            .slice(0, 20)
            .map(
              (candidate) =>
                `  \`${candidate.id}\` — ${candidate.turnCount} turn${candidate.turnCount === 1 ? "" : "s"}, updated ${candidate.updatedAt}`,
            ),
          "Use `/resume <id>` to switch.",
        ].join("\n"),
      );
      return { ok: true, sessions };
    }
    const matches = saved.filter((candidate) => {
      const title = candidate.title.toLowerCase();
      const key = candidate.key.toLowerCase();
      return (
        candidate.id.toLowerCase().startsWith(needle) ||
        key === needle ||
        title === needle
      );
    });
    if (!matches.length) {
      this.emitSlash(
        connection,
        `No saved session matches \`${query}\`. Run \`/resume\` to list sessions.`,
        "warning",
      );
      return { ok: false, error: "saved session not found" };
    }
    if (matches.length > 1) {
      this.emitSlash(
        connection,
        [
          `Multiple sessions match \`${query}\`:`,
          ...matches
            .slice(0, 20)
            .map(
              (candidate) =>
                `  \`${candidate.id}\` — ${candidate.title || "(untitled)"}`,
            ),
          "Use a longer id prefix.",
        ].join("\n"),
        "warning",
      );
      return {
        ok: false,
        error: "multiple saved sessions match",
        sessions: matches.slice(0, 20).map(savedSessionPayload),
      };
    }
    const target = matches[0];
    if (!target) {
      return { ok: false, error: "saved session not found" };
    }
    await this.runtime.flushSessions();
    // Open the resume target first: a failed resume must leave the
    // connection and every live session untouched.
    const session = await this.runtime.openSession(target.id, undefined, {
      resume: true,
    });
    // Live sessions are keyed by sessionKey, not session id; evict a stale
    // duplicate registered under another key like deleteSavedSession does.
    const activeKey = this.runtime
      .listSessions()
      .find(
        (candidate) =>
          candidate.id === target.id && candidate.sessionKey !== target.id,
      )?.sessionKey;
    if (activeKey) {
      this.forgetAcceptedSubmissions([activeKey]);
      this.endSessionLifetime([activeKey]);
      this.runtime.evictSession(activeKey);
    }
    connection.activeSessionKey = target.id;
    this.emitInitDone(connection, session);
    this.emitStatus(connection, session);
    this.replaySessionHistory(connection, session);
    this.emitSlash(connection, `Resumed session \`${session.id}\`.`);
    await this.reportResumeRepair(connection, session.id);
    this.indexSessionForSearch(target.id);
    return {
      ok: true,
      session: sessionPayload(session, this.contextLimit(session.model), this.mcpStatusRecord()),
    };
  }

  private async listSavedSessionBranches(
    connection: DaemonTransportConnection,
  ): Promise<JsonRpcPayload> {
    const saved = await this.runtime.listSavedSessions();
    if (!saved.length) {
      this.emitSlash(connection, "No branches / saved sessions.");
      return { ok: true, sessions: [] };
    }
    this.emitSlash(
      connection,
      [
        `Branches / saved sessions (${saved.length}):`,
        ...saved
          .slice(0, 20)
          .map(
            (candidate) =>
              `  \`${candidate.id}\` — ${candidate.turnCount} turn${candidate.turnCount === 1 ? "" : "s"}, updated ${candidate.updatedAt}`,
          ),
      ].join("\n"),
    );
    return { ok: true, sessions: saved.slice(0, 20).map(savedSessionPayload) };
  }

  private async branchSession(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    title: string,
  ): Promise<JsonRpcPayload> {
    if (!session) {
      this.emitSlash(connection, "No active session to branch.", "warning");
      return { ok: false, error: "no active session" };
    }
    if (!sessionHasHistory(session)) {
      this.emitSlash(
        connection,
        "Nothing to branch yet — this session has no messages.",
        "warning",
      );
      return { ok: false, error: "session has no history" };
    }
    const id = newConnectionKey();
    const branch = await this.runtime.openSession(id, session.agentId, {
      cwd: session.cwd,
      model: session.model,
    });
    branch.messages = session.messages.map((message) =>
      structuredClone(message),
    );
    branch.metadata = {
      ...session.metadata,
      forked_from: session.id,
      parent_session_id: session.id,
      ...(title ? { title } : {}),
    };
    branch.extra = {
      ...session.extra,
      parent_session_id: session.id,
    };
    branch.interactionMode = session.interactionMode;
    branch.planMode = session.planMode;
    branch.thinkingContent = structuredClone(session.thinkingContent);
    branch.toolExecutions = structuredClone(session.toolExecutions);
    branch.totalInputTokens = session.totalInputTokens;
    branch.totalOutputTokens = session.totalOutputTokens;
    branch.turnCount = session.turnCount;
    await this.runtime.flushSessions();
    const persisted = (await this.runtime.listSavedSessions()).find(
      (candidate) => candidate.id === branch.id,
    );
    this.emitSlash(
      connection,
      `Branched to new session \`${branch.id}\` (${branch.messages.length} messages).`,
    );
    return {
      ok: true,
      session: persisted
        ? savedSessionPayload(persisted)
        : sessionPayload(branch, this.contextLimit(branch.model), this.mcpStatusRecord()),
    };
  }

  private undoLastTurn(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    notify = true,
  ): Promise<JsonRpcPayload> {
    if (session && this.sessionOperations.has(session.sessionKey)) {
      if (notify) {
        this.emitSlash(
          connection,
          "Cannot undo while another session operation is in progress.",
          "warning",
        );
      }
      return Promise.resolve({
        ok: false,
        error: "session operation in progress",
      });
    }
    return session
      ? this.withSessionOperation(session.sessionKey, () =>
          this.undoLastTurnUnlocked(connection, session, notify)
        )
      : this.undoLastTurnUnlocked(connection, session, notify);
  }

  private async undoLastTurnUnlocked(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    notify: boolean,
  ): Promise<JsonRpcPayload> {
    if (!session || !session.messages.length) {
      if (notify) {
        this.emitSlash(connection, "Nothing to undo.");
      }
      return { ok: true, dropped: 0 };
    }
    if (session.activeTurnId) {
      if (notify) {
        this.emitSlash(
          connection,
          "Cannot undo while a turn is running. Use `/stop` first.",
          "warning",
        );
      }
      return { ok: false, error: "turn is running" };
    }
    const dropped = discardLastUserTurn(session.messages);
    if (!dropped) {
      if (notify) {
        this.emitSlash(connection, "Nothing to undo.");
      }
      return { ok: true, dropped: 0 };
    }
    session.turnCount = Math.max(0, session.turnCount - 1);
    // The window just shrank, so the condition that kept failing is no longer
    // the one the counter was recording.
    this.clearAutoCompactFailures(session.sessionKey);
    await this.runtime.flushSessions("rewrite");
    if (session.messages.length === 0 && session.turnCount === 0) {
      // The store's empty-save path used to delete the transcript here
      // implicitly. Routine saves never delete anymore, so removing the
      // last remaining turn removes the persisted record explicitly while
      // the live (now empty) session stays usable.
      await this.runtime.removeSavedTranscript?.(session.id);
      // The persisted record is gone, so the in-memory session no longer has
      // a generation or message boundary to be authorized against. Resetting
      // both keeps the next turn's append from conflicting with the deleted
      // transcript's absence.
      session.transcriptGeneration = 0;
      session.persistedMessageCount = 0;
    }
    if (notify) {
      this.emitSlash(
        connection,
        `Undone — dropped ${dropped} message${dropped === 1 ? "" : "s"} from the conversation.`,
      );
    }
    return { ok: true, dropped };
  }

  private retryLastTurn(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    if (session && this.sessionOperations.has(session.sessionKey)) {
      this.emitSlash(
        connection,
        "Cannot retry while another session operation is in progress.",
        "warning",
      );
      return Promise.resolve({
        ok: false,
        error: "session operation in progress",
      });
    }
    return session
      ? this.withSessionOperation(session.sessionKey, () =>
          this.retryLastTurnUnlocked(connection, session)
        )
      : this.retryLastTurnUnlocked(connection, session);
  }

  private async retryLastTurnUnlocked(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): Promise<JsonRpcPayload> {
    if (!session || !session.messages.length) {
      this.emitSlash(connection, "Nothing to retry.");
      return { ok: true, retried: false };
    }
    if (session.activeTurnId) {
      this.emitSlash(
        connection,
        "A turn is already running. Use `/stop` before retrying.",
        "warning",
      );
      return { ok: false, error: "turn is running" };
    }
    const prompt = lastUserMessage(session.messages);
    if (!prompt) {
      this.emitSlash(connection, "No prior user message to retry.");
      return { ok: true, retried: false };
    }
    // Capture the discarded turn so a failed resubmit can restore it instead
    // of permanently losing the user's prompt.
    const priorMessages = session.messages.slice();
    const priorTurnCount = session.turnCount;
    discardLastUserTurn(session.messages);
    session.turnCount = Math.max(0, session.turnCount - 1);
    this.emitSlash(connection, "Retrying the last prompt…");
    const key = connection.activeSessionKey;
    void this.submitTrackedTurn(
      key,
      prompt,
      (event) => this.emit(connection, event.type, event.payload),
      connection,
    ).catch((error) => {
      void this.withSessionOperation(key, async () => {
        session.messages.splice(0, session.messages.length, ...priorMessages);
        session.turnCount = priorTurnCount;
        this.emitSlash(connection, `Retry failed: ${errorMessage(error)}`, "error");
      });
    });
    return { ok: true, retried: true };
  }

  private async manageCronJobs(
    connection: DaemonTransportConnection,
    args: string,
  ): Promise<JsonRpcPayload> {
    const tokens = tokenizeSlashArguments(args);
    if (!tokens) {
      this.emitSlash(
        connection,
        "Cron command has an unclosed quote.",
        "warning",
      );
      return { ok: false, error: "invalid cron arguments" };
    }
    const [rawAction = "list", ...rest] = tokens;
    const action = rawAction.toLowerCase();
    if (action === "list") {
      return this.listCronJobs(connection);
    }
    if (action === "add") {
      return this.addCronJob(connection, rest);
    }
    if (action === "remove") {
      return this.removeCronJob(connection, rest);
    }
    if (action === "pause") {
      return this.setCronPaused(connection, rest, true);
    }
    if (action === "resume") {
      return this.setCronPaused(connection, rest, false);
    }
    if (action === "run") {
      return this.runCronJob(connection, rest);
    }
    this.emitSlash(connection, cronUsage(), "warning");
    return { ok: false, error: `unknown cron action: ${action}` };
  }

  private listCronJobs(connection: DaemonTransportConnection): JsonRpcPayload {
    try {
      const jobs = this.cronStore.listJobs();
      if (!jobs.length) {
        this.emitSlash(connection, "No cron jobs scheduled.");
        return { ok: true, jobs: [] };
      }
      const lines = [
        `Cron jobs (${jobs.length}):`,
        ...jobs.map(
          (job) =>
            `  \`${job.id}\` — \`${job.schedule}\` (${job.paused ? "paused" : "active"})`,
        ),
      ];
      this.emitSlash(connection, lines.join("\n"));
      return { ok: true, jobs: jobs.map(cronJobPayload) };
    } catch (error) {
      const message = errorMessage(error);
      this.emitSlash(connection, `Cron list failed: \`${message}\``, "error");
      return { ok: false, error: message };
    }
  }

  private addCronJob(
    connection: DaemonTransportConnection,
    tokens: readonly string[],
  ): JsonRpcPayload {
    const parsed = parseCronAddArguments(tokens);
    if ("error" in parsed) {
      this.emitSlash(connection, `${parsed.error}\n${cronUsage()}`, "warning");
      return { ok: false, error: parsed.error };
    }
    try {
      const nextRunAt = parsed.at
        ? parsed.at
        : nextFireAt(parsed.schedule ?? "").toISOString();
      const store = this.cronStore;
      const job = store.add(
        new CronJob({
          id: store.newId(),
          prompt: parsed.prompt,
          schedule: parsed.schedule ?? "",
          nextRunAt,
          oneshot: Boolean(parsed.at),
          // The project the job was created from, so a listing can say which
          // repo owns it instead of leaving every daemon to assume it is theirs.
          projectRoot: this.cronProjectRoot(connection),
          ...(parsed.deliver ? { deliver: parsed.deliver } : {}),
          ...(parsed.recipient ? { recipient: parsed.recipient } : {}),
          ...(parsed.workspaceId ? { workspaceId: parsed.workspaceId } : {}),
        }),
      );
      this.emitSlash(
        connection,
        `Scheduled cron job \`${job.id}\` for \`${job.nextRunAt}\`.`,
      );
      return { ok: true, job: cronJobPayload(job) };
    } catch (error) {
      const message = errorMessage(error);
      this.emitSlash(connection, `Cron add failed: \`${message}\``, "error");
      return { ok: false, error: message };
    }
  }

  /** The repo a `/cron add` belongs to: the caller's session, not the daemon's cwd. */
  private cronProjectRoot(connection: DaemonTransportConnection): string {
    const session = this.runtime.sessionStatus(connection.activeSessionKey);
    return resolveProjectDirectory(
      optionalString(session?.metadata.project_root) ||
        session?.cwd ||
        this.cronLeaseOwnerKey,
    );
  }

  private removeCronJob(
    connection: DaemonTransportConnection,
    tokens: readonly string[],
  ): JsonRpcPayload {
    const id = singleCronJobId(tokens);
    if (!id) {
      this.emitSlash(connection, "Usage: `/cron remove <job-id>`.", "warning");
      return { ok: false, error: "cron job id is required" };
    }
    const removed = this.cronStore.remove(id);
    if (!removed) {
      this.emitSlash(connection, `No cron job named \`${id}\`.`, "warning");
      return { ok: false, error: "cron job not found" };
    }
    this.emitSlash(connection, `Removed cron job \`${id}\`.`);
    return { ok: true, id };
  }

  private setCronPaused(
    connection: DaemonTransportConnection,
    tokens: readonly string[],
    paused: boolean,
  ): JsonRpcPayload {
    const id = singleCronJobId(tokens);
    if (!id) {
      this.emitSlash(
        connection,
        `Usage: \`/cron ${paused ? "pause" : "resume"} <job-id>\`.`,
        "warning",
      );
      return { ok: false, error: "cron job id is required" };
    }
    const store = this.cronStore;
    const current = store.get(id);
    if (!current) {
      this.emitSlash(connection, `No cron job named \`${id}\`.`, "warning");
      return { ok: false, error: "cron job not found" };
    }
    try {
      const nextRunAt =
        !paused && !current.oneshot && current.schedule
          ? nextFireAt(current.schedule).toISOString()
          : current.nextRunAt;
      const job = store.update(id, { paused, nextRunAt });
      if (!job) {
        return { ok: false, error: "cron job not found" };
      }
      this.emitSlash(
        connection,
        `${paused ? "Paused" : "Resumed"} cron job \`${job.id}\`.`,
      );
      return { ok: true, job: cronJobPayload(job) };
    } catch (error) {
      const message = errorMessage(error);
      this.emitSlash(connection, `Cron update failed: \`${message}\``, "error");
      return { ok: false, error: message };
    }
  }

  private async runCronJob(
    connection: DaemonTransportConnection,
    tokens: readonly string[],
  ): Promise<JsonRpcPayload> {
    const id = singleCronJobId(tokens);
    if (!id) {
      this.emitSlash(connection, "Usage: `/cron run <job-id>`.", "warning");
      return { ok: false, error: "cron job id is required" };
    }
    const job = this.cronStore.get(id);
    if (!job) {
      this.emitSlash(connection, `No cron job named \`${id}\`.`, "warning");
      return { ok: false, error: "cron job not found" };
    }
    this.emitSlash(connection, `Running cron job \`${job.id}\`.`);
    const result = await this.runCronJobTurn(
      job,
      connection.activeSessionKey,
      (event) => this.emit(connection, event.type, event.payload),
    );
    const archivePath = await this.deliverCronOutput(job, result.output);
    const updated = this.cronStore.update(job.id, {
      lastRunAt: new Date().toISOString(),
    });
    this.emitSlash(
      connection,
      `Cron job \`${job.id}\` finished; archived to \`${archivePath}\`.`,
    );
    return {
      ok: true,
      job: cronJobPayload(updated ?? job),
      output: result.output,
      session_key: result.sessionKey,
      archive_path: archivePath,
    };
  }

  private async runScheduledCronJob(job: CronJob): Promise<string> {
    const result = await this.runCronJobTurn(job, `cron:${job.id}`, (event) => {
      this.broadcast("cron_event", {
        job_id: job.id,
        event_type: event.type,
        payload: event.payload,
      });
    });
    this.broadcast("cron_run", {
      job_id: job.id,
      session_key: result.sessionKey,
    });
    return result.output;
  }

  private async runCronJobTurn(
    job: CronJob,
    fallbackSessionKey: string,
    emit: (event: {
      readonly payload: JsonRpcPayload;
      readonly type: string;
    }) => void,
  ): Promise<{ readonly output: string; readonly sessionKey: string }> {
    const sessionKey = job.workspaceId || fallbackSessionKey;
    await this.runtime.openSession(sessionKey);
    const parts: string[] = [];
    // Cron turns have no owning connection, but they are still tracked in
    // inFlightTurns so stop() awaits them before flushing sessions.
    await this.submitTrackedTurn(sessionKey, job.prompt, (event) => {
      if (event.type === "text_part") {
        const text = optionalString(event.payload.text);
        if (text) {
          parts.push(text);
        }
      }
      emit(event);
    }, undefined);
    return {
      sessionKey,
      output: parts.join("").trim() || "(No text response was produced.)",
    };
  }

  private async deliverCronOutput(
    job: CronJob,
    output: string,
  ): Promise<string> {
    const archivePath = await routeOutput(
      { platform: job.deliver, recipient: job.recipient },
      output,
      {
        archiveDirectory: this.cronArchiveDirectory,
        jobId: job.id,
        sender: async (platform, recipient, content) => {
          const manager = this.channelManager;
          if (!manager) {
            throw new Error(
              "Cron delivery requested but no native channel manager is configured.",
            );
          }
          const message: ChannelMessage = createChannelMessage({
            channel: platform,
            direction: MessageDirection.OUTBOUND,
            text: content,
            ...(recipient
              ? { channelUserId: recipient, roomId: recipient }
              : {}),
            metadata: { cron_job_id: job.id },
          });
          await manager.send(message);
        },
      },
    );
    this.broadcast("cron_complete", {
      job_id: job.id,
      deliver: job.deliver,
      recipient: job.recipient,
      archive_path: archivePath,
    });
    return archivePath;
  }

  private async createSnapshot(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    label: string,
  ): Promise<JsonRpcPayload> {
    if (!session) {
      this.emitSlash(connection, "No active session yet.", "warning");
      return { ok: false, error: "no active session" };
    }
    try {
      const snapshot = await this.snapshotManagerFactory(session.cwd).snapshot(
        label || "manual",
      );
      this.emitSlash(connection, `Snapshot \`${snapshot.id}\` saved.`);
      return { ok: true, snapshot: snapshotPayload(snapshot) };
    } catch (error) {
      const message = errorMessage(error);
      this.emitSlash(connection, `Snapshot failed: \`${message}\``, "error");
      return { ok: false, error: message };
    }
  }

  /** Feed one live session's transcript into the cross-session search index. */
  private indexSessionForSearch(sessionKey: string): void {
    const session = this.runtime.sessionStatus(sessionKey);
    if (!session || !session.messages.length) {
      return;
    }
    const title = stringValue(session.metadata.title);
    this.transcriptSearch.index({
      messages: session.messages,
      sessionId: session.id,
      ...(title ? { title } : {}),
      updatedAt: new Date().toISOString(),
    });
  }

  /**
   * Load persisted transcripts into the search index once per daemon.
   *
   * Sessions this daemon has already opened or run a turn for are fed
   * incrementally and skipped here; the cold read exists only so a search can
   * reach conversations from earlier runs. Bounded to the most recent
   * transcripts because the point is to answer a query, not to page every
   * session a project has ever had into memory.
   */
  private hydrateTranscriptSearch(): Promise<void> {
    const existing = this.transcriptSearchHydration;
    if (existing) {
      return existing;
    }
    const hydration = (async () => {
      const saved = await this.runtime.listSavedSessions(
        SEARCH_HYDRATION_SESSION_LIMIT,
      );
      for (const candidate of saved) {
        if (this.transcriptSearch.has(candidate.id)) {
          continue;
        }
        let raw: unknown;
        try {
          raw = JSON.parse(await readFile(candidate.path, "utf8")) as unknown;
        } catch {
          continue;
        }
        if (!isRecord(raw) || !Array.isArray(raw.messages)) {
          continue;
        }
        this.transcriptSearch.index({
          messages: raw.messages,
          sessionId: candidate.id,
          title: candidate.title,
          updatedAt: candidate.updatedAt,
        });
      }
    })();
    // A failed cold read must not poison every later search with the same
    // rejected promise; drop the memo so the next search can retry.
    this.transcriptSearchHydration = hydration.catch((error: unknown) => {
      this.transcriptSearchHydration = undefined;
      console.warn(`Could not hydrate transcript search: ${errorMessage(error)}`);
    });
    return this.transcriptSearchHydration;
  }

  private async searchTranscripts(
    connection: DaemonTransportConnection,
    query: string,
  ): Promise<JsonRpcPayload> {
    const needle = query.trim();
    if (!needle) {
      this.emitSlash(
        connection,
        "Usage: `/search <text>` — searches every saved transcript.",
        "warning",
      );
      return { ok: false, error: "search query is required" };
    }
    await this.hydrateTranscriptSearch();
    const hits = this.transcriptSearch.search(needle, {
      limit: SEARCH_RESULT_LIMIT,
    });
    const stats = this.transcriptSearch.stats();
    if (!hits.length) {
      // An empty answer is exactly where a silent under-count does the most
      // damage: it reads as "not in any transcript".
      const blindSpot =
        stats.unrecognizedMessages > 0
          ? ` (${stats.unrecognizedMessages} message${stats.unrecognizedMessages === 1 ? "" : "s"} could not be indexed and were searched as empty)`
          : "";
      this.emitSlash(
        connection,
        `No transcript matches \`${needle}\` across ${stats.sessions} session${stats.sessions === 1 ? "" : "s"}.${blindSpot}`,
      );
      return { ok: true, results: [], stats: searchStatsPayload(stats) };
    }
    const lines = [
      `Transcript matches for \`${needle}\` (${hits.length}):`,
      ...hits.map(
        (hit) =>
          `  \`${hit.sessionId}\` #${hit.messageIndex} ${hit.role || "?"} — ${hit.excerpt}`,
      ),
    ];
    // Say out loud how much of the corpus the index could not read. A silent
    // under-count reads exactly like "your text is not in any transcript".
    if (stats.unrecognizedMessages > 0) {
      lines.push(
        `  (${stats.unrecognizedMessages} message${stats.unrecognizedMessages === 1 ? "" : "s"} could not be indexed and were searched as empty)`,
      );
    }
    this.emitSlash(connection, lines.join("\n"));
    return {
      ok: true,
      results: hits.map(searchHitPayload),
      stats: searchStatsPayload(stats),
    };
  }

  /**
   * Report what loading this transcript changed before the user starts typing
   * into it. Repair is cheap — a parse and one linear pass — so the resume
   * path re-derives the counts from the file rather than staying silent about
   * messages the load dropped.
   */
  private async reportResumeRepair(
    connection: DaemonTransportConnection,
    sessionId: string,
  ): Promise<void> {
    // A resume that already succeeded must not fail because its diagnostic
    // could not be computed.
    await this.emitResumeRepairNotice(connection, sessionId).catch(
      (error: unknown) => {
        console.warn(`Could not summarize resume repair: ${errorMessage(error)}`);
      },
    );
  }

  private async emitResumeRepairNotice(
    connection: DaemonTransportConnection,
    sessionId: string,
  ): Promise<void> {
    const saved = await this.runtime.listSavedSessions();
    const path = saved.find((candidate) => candidate.id === sessionId)?.path;
    if (!path) {
      return;
    }
    const raw: unknown = JSON.parse(await readFile(path, "utf8")) as unknown;
    if (!isRecord(raw) || !Array.isArray(raw.messages)) {
      return;
    }
    const line = describeTranscriptRepair(summarizeTranscriptRepair(raw.messages));
    if (line) {
      this.emitSlash(connection, line, "warning");
    }
  }

  private listSnapshots(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
  ): JsonRpcPayload {
    if (!session) {
      this.emitSlash(connection, "No active session yet.", "warning");
      return { ok: false, error: "no active session" };
    }
    try {
      const snapshots = this.snapshotManagerFactory(session.cwd).list();
      if (!snapshots.length) {
        this.emitSlash(
          connection,
          "No snapshots yet. Take one with `/snapshot [label]`.",
        );
        return { ok: true, snapshots: [] };
      }
      const lines = [
        `Snapshots (${snapshots.length}):`,
        ...snapshots
          .slice(0, 20)
          .map(
            (snapshot) =>
              `  \`${snapshot.id}\` — \`${snapshot.label}\` @ ${snapshot.createdAt}`,
          ),
      ];
      this.emitSlash(connection, lines.join("\n"));
      return { ok: true, snapshots: snapshots.map(snapshotPayload) };
    } catch (error) {
      const message = errorMessage(error);
      this.emitSlash(
        connection,
        `Snapshot list failed: \`${message}\``,
        "error",
      );
      return { ok: false, error: message };
    }
  }

  /**
   * Roll the workspace back to a snapshot, or — with a path — only that one
   * file. Restoring a whole tree to recover a single damaged file also throws
   * away every unrelated edit made since, which is rarely what was meant.
   */
  private async rollbackSnapshot(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    argument: string,
  ): Promise<JsonRpcPayload> {
    if (!session) {
      this.emitSlash(connection, "No active session yet.", "warning");
      return { ok: false, error: "no active session" };
    }
    const [ref = "", ...pathParts] = argument.split(/\s+/).filter(Boolean);
    const filePath = pathParts.join(" ");
    if (!ref) {
      this.emitSlash(
        connection,
        "Usage: `/rollback <snapshot-id> [path]` — list with `/snapshots`.",
        "warning",
      );
      return { ok: false, error: "snapshot reference is required" };
    }
    try {
      if (filePath) {
        const restored = await this.snapshotManagerFactory(session.cwd).restoreFile(
          ref,
          filePath,
        );
        this.emitSlash(
          connection,
          `Restored \`${restored.path}\` from snapshot \`${ref}\` (undo with \`/rollback ${restored.previous.id}\`).`,
        );
        return {
          ok: true,
          path: restored.path,
          snapshot: snapshotPayload(restored.snapshot),
          previous: snapshotPayload(restored.previous),
        };
      }
      const snapshot = await this.snapshotManagerFactory(session.cwd).rollback(ref);
      this.emitSlash(connection, `Rolled back to snapshot \`${ref}\`.`);
      return { ok: true, snapshot: snapshotPayload(snapshot) };
    } catch (error) {
      const message = errorMessage(error);
      this.emitSlash(connection, `Rollback failed: \`${message}\``, "error");
      return { ok: false, error: message };
    }
  }

  private permissionResponse(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): JsonRpcPayload {
    const requestId = optionalString(params.request_id) ?? "";
    const owner = this.approvalOwners.get(requestId);
    if (owner && owner !== connection) {
      return { ok: false, error: "approval owned by another connection" };
    }
    const response = optionalString(params.response) ?? "reject";
    const ok = this.interactions.respondPermission(requestId, response);
    if (ok) {
      this.approvalOwners.delete(requestId);
      this.emit(connection, "approval_response", {
        request_id: requestId,
        response,
      });
    }
    return { ok };
  }

  private async questionResponse(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    const requestId = optionalString(params.request_id) ?? "";
    const owner = this.questionOwners.get(requestId);
    if (owner && owner !== connection) {
      return { ok: false, error: "question owned by another connection" };
    }
    const answers = stringRecord(params.answers);
    const providerFlow = this.providerFlows.get(connection);
    if (providerFlow?.activeRequestId === requestId) {
      const transition = await providerFlow.answer(requestId, answers);
      if (!transition) {
        return { ok: false, error: "invalid provider setup response" };
      }
      if (this.providerFlows.get(connection) !== providerFlow) {
        return { ok: false, error: "provider setup was cancelled" };
      }
      this.questionOwners.delete(requestId);
      this.emit(connection, "question_response", { id: requestId, answers });
      return this.applyProviderFlowTransition(
        connection,
        providerFlow,
        transition,
      );
    }
    const ok = this.interactions.respondQuestion(requestId, answers);
    if (ok) {
      this.questionOwners.delete(requestId);
      this.emit(connection, "question_response", { id: requestId, answers });
    }
    return { ok };
  }

  private updateProviderModelOverride(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): JsonRpcPayload {
    const profileName = optionalString(params.profile_name) ?? optionalString(params.name);
    const model = optionalString(params.model);
    if (!profileName || !model) {
      return { ok: false, error: "profile_name and model are required" };
    }
    const profile = this.profileStore.get(profileName);
    if (!profile) return { ok: false, error: `No provider profile named ${profileName}` };
    const cached = Object.prototype.hasOwnProperty.call(profile.model_capabilities ?? {}, model);
    if (!cached && profile.model.trim() !== model) {
      return { ok: false, error: `No cached model named ${model} for profile ${profileName}` };
    }
    const hasContext = Object.prototype.hasOwnProperty.call(params, "context_limit");
    const hasOutput = Object.prototype.hasOwnProperty.call(params, "max_output_tokens");
    if (!hasContext && !hasOutput) {
      return { ok: false, error: "context_limit or max_output_tokens is required" };
    }
    const contextLimit = nullablePositiveSafeInteger(params.context_limit);
    const maxOutputTokens = nullablePositiveSafeInteger(params.max_output_tokens);
    if (hasContext && contextLimit === undefined) {
      return { ok: false, error: "context_limit must be a positive safe integer or null" };
    }
    if (hasOutput && maxOutputTokens === undefined) {
      return { ok: false, error: "max_output_tokens must be a positive safe integer or null" };
    }
    const updated = this.profileStore.updateModelCapabilities(profileName, model, {
      ...(hasContext ? { contextLimit: contextLimit as number | null } : {}),
      ...(hasOutput ? { maxOutputTokens: maxOutputTokens as number | null } : {}),
    });
    if (!updated) return { ok: false, error: "model capability update was rejected" };
    if (this.activeRuntimeProfileName() === profileName) {
      this.runtime.reload({});
      const session = this.runtime.sessionStatus(connection.activeSessionKey);
      if (session) this.emitStatus(connection, session);
    }
    return {
      ok: true,
      model: modelCapabilityPayload(updated, model),
    };
  }

  private async saveProvider(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    this.cancelProviderFlow(connection);
    const name = optionalString(params.name);
    const provider = optionalString(params.provider)
      ?.trim()
      .toLowerCase()
      .replaceAll("_", "-");
    // A known provider type carries its default endpoint in the registry —
    // "Provider default" in the form means exactly this, not an empty string.
    const known =
      provider !== undefined &&
      Object.prototype.hasOwnProperty.call(PROVIDERS, provider)
        ? PROVIDERS[provider as keyof typeof PROVIDERS]
        : undefined;
    const baseUrl = optionalString(params.base_url) ?? known?.baseUrl;
    const model = optionalString(params.model);
    if (!name || !baseUrl || !model) {
      return {
        ok: false,
        error: known
          ? "name and model are required"
          : "name, base_url, and model are required",
      };
    }
    // An absent (or blank) api_key keeps the stored one — an edit that only
    // changes the model must not wipe the credential it never re-typed.
    // (The desktop client omits the field unless the user typed a
    // replacement; `stringValue` maps absent to "", which ?? cannot catch.)
    const existing = this.profileStore.list().find(p => p.name === name);
    const typedKey =
      typeof params.api_key === "string" && params.api_key.trim()
        ? params.api_key
        : undefined;
    const apiKey = typedKey ?? existing?.api_key ?? "";
    const profile = this.profileStore.save({
      name,
      baseUrl,
      apiKey,
      model,
      ...(provider === undefined ? {} : { provider }),
    });
    this.runtime.reload(profileOverrides(profile));
    await this.emitProviderInit(connection);
    return { ok: true, profile: profilePayload({ ...profile, active: true }) };
  }

  private async selectProvider(
    connection: DaemonTransportConnection,
    name: string,
  ): Promise<JsonRpcPayload> {
    this.cancelProviderFlow(connection);
    if (!name || !this.profileStore.setActive(name)) {
      return { ok: false, error: `No provider profile named ${name}` };
    }
    const active = this.profileStore.active();
    this.runtime.reload(profileOverrides(active));
    await this.emitProviderInit(connection);
    this.emitSlash(connection, `Switched to provider profile \`${name}\`.`);
    return { ok: true };
  }

  private async setMode(
    connection: DaemonTransportConnection,
    mode: string,
    planMode?: boolean,
    // Callers pass the session the request named. The TUI has always sent one;
    // this used to drop it and act on whatever session the connection happened
    // to be attached to, so a mode change aimed at one tab could land on another.
    targetSessionKey = connection.activeSessionKey,
  ): Promise<JsonRpcPayload> {
    const session = await this.runtime.setSessionMode(
      targetSessionKey,
      mode,
      planMode,
    );
    if (!session) {
      return { ok: false, error: "no active session" };
    }
    this.emitStatus(connection, session);
    return {
      ok: true,
      mode: session.interactionMode,
      plan_mode: session.planMode,
    };
  }

  /** Pin a picker-selected model to the session named by the RPC. */
  private async setModel(
    connection: DaemonTransportConnection,
    model: string,
    targetSessionKey = connection.activeSessionKey,
  ): Promise<JsonRpcPayload> {
    if (this.runtime.setSessionModel === undefined) {
      return { ok: false, error: "this runtime does not support session model selection" };
    }
    const session = await this.runtime.setSessionModel(targetSessionKey, model);
    if (!session) {
      return { ok: false, error: "no active session" };
    }
    // Keep the selected profile's default aligned for sessions opened later;
    // existing sessions retain their own pins.
    try {
      this.profileStore?.updateActiveModel(session.model);
    } catch {
      // Profile persistence is best-effort; the session pin already applies.
    }
    if (targetSessionKey === connection.activeSessionKey) {
      await this.emitProviderInit(connection);
    }
    this.emitStatus(connection, session);
    return { ok: true, model: session.model };
  }

  /** Pin a validated reasoning effort to the session named by the RPC. */
  private async setReasoning(
    connection: DaemonTransportConnection,
    requested: string,
    targetSessionKey = connection.activeSessionKey,
  ): Promise<JsonRpcPayload> {
    const active = this.runtime.sessionStatus(targetSessionKey);
    if (!active) {
      return { ok: false, error: "no active session" };
    }
    const levels = await this.reasoningLevels(active.model);
    const offered = selectableEfforts(levels);
    const resolved = resolveEffort(levels, requested)
      ?? clampEffort(levels, requested);
    if (!resolved) {
      return {
        ok: false,
        error: `Thinking level must be one of: ${offered.join(", ")}.`,
        levels: offered,
      };
    }
    if (this.runtime.setSessionReasoning === undefined) {
      return { ok: false, error: "this runtime does not support session reasoning selection" };
    }
    const session = await this.runtime.setSessionReasoning(
      targetSessionKey,
      resolved,
    );
    if (!session) {
      return { ok: false, error: "no active session" };
    }
    const profile = this.profileStore.active();
    if (profile) {
      this.profileStore.updateSampling(profile.name, {
        reasoning_effort: resolved,
        thinking: resolved !== REASONING_OFF,
      });
    }
    this.emitStatus(connection, session);
    return { ok: true, reasoning_effort: resolved, levels: offered };
  }

  /**
   * /ultra handler. Guards on the optional DaemonRuntime.setSessionUltra so
   * runtimes without ultra support receive a typed error instead of a crash,
   * and echoes the resolved flag in the payload so clients can render the
   * new state without a second round trip.
   */
  private async setUltra(
    connection: DaemonTransportConnection,
    enabled: boolean,
  ): Promise<JsonRpcPayload> {
    if (this.runtime.setSessionUltra === undefined) {
      return { ok: false, error: "this runtime does not support ultra mode" };
    }
    const session = await this.runtime.setSessionUltra(
      connection.activeSessionKey,
      enabled,
    );
    if (!session) {
      return { ok: false, error: "no active session" };
    }
    this.emitStatus(connection, session);
    return {
      ok: true,
      ultra_mode: session.ultraMode === true,
    };
  }

  private async initialize(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    const resumeId = optionalString(params.resume_session_id);
    const requestedKey = optionalString(params.session_key);
    const key = resumeId || requestedKey || `tui:${newConnectionKey()}`;
    const cwd = resolveProjectDirectory(
      optionalString(params.project_dir) ||
        this.projectDirectory ||
        process.cwd(),
    );
    connection.activeSessionKey = key;
    const runtimeOverrides = Object.fromEntries(
      ["model", "base_url", "api_key", "provider", "permission_mode"].flatMap(
        (name) => (params[name] === undefined ? [] : [[name, params[name]]]),
      ),
    );
    if (Object.keys(runtimeOverrides).length) {
      this.runtime.reload(runtimeOverrides);
    }
    if (!resumeId) {
      // Evicting a session with an active turn would hijack work another
      // connection may still own; adopt the live session instead of
      // resetting it.
      const live = this.runtime.sessionStatus(key);
      if (!live?.activeTurnId) {
        // Eviction drops every mutation not yet persisted (idle steers,
        // title and mode edits); flush first so a reconnect cannot silently
        // lose them.
        await this.runtime.flushSessions();
        // Re-check across the flush await: a turn admitted while it was in
        // flight registers its controller and session state, and evicting
        // now would abort just-admitted work. With no further yield between
        // this check and eviction, the decision is atomic.
        if (!this.runtime.sessionStatus(key)?.activeTurnId) {
          this.forgetAcceptedSubmissions([key]);
          this.endSessionLifetime([key]);
          this.runtime.evictSession(key);
        }
      }
    }
    const modelOverride = optionalString(params.model);
    const openOptions = {
      cwd,
      resume: Boolean(resumeId),
      ...(modelOverride ? { model: modelOverride } : {}),
    };
    let requestedAgent = optionalString(params.agent_id)
      ?? this.runtime.sessionStatus(key)?.agentId
      ?? (resumeId ? undefined : this.agentPresetRoster.defaultId);
    if (requestedAgent) {
      let preset: AgentPresetEntry;
      try {
        preset = this.agentPresetRoster.resolve(requestedAgent, cwd);
      } catch (error) {
        return { ok: false, code: "agent-preset-not-found", error: errorMessage(error) };
      }
      if (preset.broken) return { ok: false, code: "agent-preset-broken", error: preset.broken };
      requestedAgent = preset.id;
    }
    const session = await this.runtime.openSession(key, requestedAgent, openOptions);
    await this.refreshSkills(session);
    const skills = this.skillRegistry
      .all()
      .filter((skill) => skillMatchesPlatform(skill));
    const model = session.model || stringValue(this.runtime.status().model);
    const contextLimit = this.contextLimit(model);
    // One cheap git call per initialize: the shell shows the branch the work
    // is happening on. Null outside a repo — never fabricated.
    const branch = await gitBranch(cwd);
    const initPayload: JsonRpcPayload = {
      session_id: session.id,
      model,
      cwd: session.cwd,
      branch: branch ?? "",
      // JSON-RPC v35 keeps this numeric field; zero means provider metadata is
      // unavailable and clients must render an unknown capacity.
      context_limit: contextLimit,
      agent_name: session.agentId,
      mode: session.interactionMode,
      plan_mode: session.planMode,
      ultra_mode: session.ultraMode === true,
      // Session-first: with two sessions open the daemon-wide value names an
      // effort this session may not be running at.
      reasoning_effort: session.reasoningEffort
        || stringValue(this.runtime.status().reasoning_effort)
        || "off",
      permission_mode: runtimePermissionMode(
        session.permissionMode ?? this.runtime.status().permission_mode,
      ),
      skills: skills.map((skill) => skill.metadata.name),
      skill_descriptions: Object.fromEntries(
        skills.map((skill) => [
          skill.metadata.name,
          skill.metadata.description,
        ]),
      ),
      head_hash: "",
      // `version` is retained for existing clients; daemon_version makes the
      // handshake role explicit for desktop/app compatibility checks.
      version: XERXES_VERSION,
      daemon_version: XERXES_VERSION,
      daemon_protocol: DAEMON_PROTOCOL_VERSION,
      daemon_build_id: this.daemonBuildId(),
    };
    this.emit(connection, "init_done", initPayload);
    this.emit(
      connection,
      "status_update",
      statusUpdatePayload(
        session,
        model,
        contextLimit,
        this.channelStatusData(),
        stringValue(this.runtime.status().reasoning_effort) || "off",
        runtimePermissionMode(this.runtime.status().permission_mode),
        this.mcpStatusRecord(),
      ),
    );
    if (session.messages.length) {
      this.replaySessionHistory(connection, session);
      this.indexSessionForSearch(key);
    }
    if (resumeId && session.messages.length) {
      await this.reportResumeRepair(connection, session.id);
    }
    // Populate the live capability cache after the initial frame. Until the
    // provider answers, every context surface remains explicitly unknown.
    this.refreshActiveModelCapabilities(connection);
    return {
      ...this.runtimeStatusWithChannels(),
      ...initPayload,
      ok: true,
      session: sessionPayload(session, contextLimit, this.mcpStatusRecord()),
      daemon_protocol: DAEMON_PROTOCOL_VERSION,
      daemon_build_id: this.daemonBuildId(),
    };
  }

  private replaySessionHistory(
    connection: DaemonTransportConnection,
    session: DaemonSession,
  ): void {
    // Persisted tool executions form a flat list keyed by tool_call_id while
    // assistant messages carry the matching tool_calls, so replay each tool
    // row right after the assistant turn that requested it. Executions whose
    // call is no longer present in the retained messages (e.g. trimmed
    // history) flush afterwards in recorded order.
    const executionsByToolCallId = new Map<string, Record<string, unknown>>();
    for (const execution of session.toolExecutions) {
      if (!isRecord(execution)) {
        continue;
      }
      const toolCallId = toolExecutionCallId(execution);
      if (toolCallId && !executionsByToolCallId.has(toolCallId)) {
        executionsByToolCallId.set(toolCallId, execution);
      }
    }
    const replayedToolCallIds = new Set<string>();
    let count = 0;
    for (const message of session.messages) {
      const role = message.role.toLowerCase();
      if (role !== "user" && role !== "assistant") {
        continue;
      }
      const text = messageText(message);
      if (text && !(role === "user" && looksLikeInternalReplayMessage(text))) {
        // Persisted thinking traces ride the replay payload so a reopened TUI
        // can render them exactly like live thinking instead of dropping them.
        const thinking =
          role === "assistant" && typeof message.thinking === "string" && message.thinking.trim()
            ? message.thinking
            : undefined;
        this.emit(connection, "notification", {
          id: newConnectionKey(),
          category: "history",
          type: `replay_${role}`,
          severity: "info",
          title: "",
          body: role === "user" ? `✨ ${text}` : text,
          payload: thinking === undefined ? {} : { thinking },
        });
        count += 1;
      }
      if (role !== "assistant" || !Array.isArray(message.tool_calls)) {
        continue;
      }
      for (const call of message.tool_calls) {
        if (!isRecord(call)) {
          continue;
        }
        const toolCallId = stringValue(call.id);
        const functionRecord = isRecord(call.function) ? call.function : {};
        this.emitToolReplay(connection, {
          argumentsPreview: replayPreviewText(
            functionRecord.arguments,
            REPLAY_ARGUMENTS_PREVIEW_CHARS,
          ),
          execution: toolCallId ? executionsByToolCallId.get(toolCallId) : undefined,
          fallbackName: stringValue(functionRecord.name),
        });
        if (toolCallId) {
          replayedToolCallIds.add(toolCallId);
        }
      }
    }
    for (const execution of session.toolExecutions) {
      if (!isRecord(execution)) {
        continue;
      }
      const toolCallId = toolExecutionCallId(execution);
      if (toolCallId && replayedToolCallIds.has(toolCallId)) {
        continue;
      }
      this.emitToolReplay(connection, { argumentsPreview: "", execution, fallbackName: "" });
    }
    this.emit(connection, "notification", {
      id: newConnectionKey(),
      category: "history",
      type: "resumed",
      severity: "info",
      title: "",
      body: `── resumed session ${session.id} (${count} message${count === 1 ? "" : "s"}) ──`,
      payload: {},
    });
  }

  /** Emit one bounded replay_tool row so a resumed transcript shows tool calls like a live session. */
  private emitToolReplay(
    connection: DaemonTransportConnection,
    input: {
      argumentsPreview: string;
      execution?: Record<string, unknown> | undefined;
      fallbackName: string;
    },
  ): void {
    const { execution } = input;
    const name =
      input.fallbackName || (execution ? toolExecutionName(execution) : "") || "tool";
    const ok = execution ? execution.permitted !== false : true;
    const durationMs = execution ? toolExecutionDurationMs(execution) : undefined;
    const context =
      input.argumentsPreview ||
      (execution ? replayPreviewText(execution.inputs, REPLAY_ARGUMENTS_PREVIEW_CHARS) : "");
    // Failed calls keep one compact diagnostic line; successful calls settle
    // to a single semantic row exactly like the live transcript.
    const note = ok
      ? ""
      : replayPreviewText(
          stringValue(execution?.result) || stringValue(execution?.return_value),
          REPLAY_RESULT_PREVIEW_CHARS,
        );
    this.emit(connection, "notification", {
      id: newConnectionKey(),
      category: "history",
      type: "replay_tool",
      severity: ok ? "info" : "warning",
      title: "",
      body: `${ok ? "✓" : "✗"} ${name}`,
      payload: {
        name,
        ok,
        ...(context ? { context } : {}),
        ...(durationMs === undefined ? {} : { duration_ms: durationMs }),
        ...(note ? { preview: note } : {}),
      },
    });
  }

  private async updateStatus(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    const session = this.runtime.sessionStatus(sessionKey(connection, params));
    const git = await gitUpdateStatus({ cwd: session?.cwd ?? process.cwd() });
    return {
      ok: true,
      applied: false,
      command: "bun run xerxes update",
      git,
      summary: formatGitUpdateStatus(git),
      next_steps: [
        "bun run xerxes update --dry-run --spec <package-or-source-spec>",
        "bun run xerxes update --apply --spec <package-or-source-spec>",
      ],
    };
  }

  private runtimeStatusPayload(): JsonRpcPayload {
    const status = this.runtimeStatusWithChannels();
    return {
      ...status,
      // `ok` reports JSON-RPC endpoint success. Runtime configuration readiness
      // is independent: an unconfigured daemon must still be probeable by TUI
      // startup and provider setup flows.
      runtime_ready: status.ok === true,
      ok: true,
      pid: typeof status.pid === "number" ? status.pid : process.pid,
      daemon_protocol: DAEMON_PROTOCOL_VERSION,
      daemon_build_id: this.daemonBuildId(),
      channels: Array.isArray(status.channels) ? status.channels : [],
      channels_available: status.channels_available === true,
      channels_configured: status.channels_configured === true,
    };
  }

  private runtimeStatusWithChannels(): JsonRpcPayload {
    const data = this.channelStatusData();
    return {
      ...this.runtime.status(),
      channels: data.channels,
      channels_available: data.available,
      channels_configured: data.configured,
    };
  }

  private daemonBuildId(): string {
    return (
      optionalString(this.runtime.status().daemon_build_id) ||
      BUN_DAEMON_BUILD_ID
    );
  }

  private emit(
    connection: DaemonTransportConnection,
    type: string,
    payload: JsonRpcPayload,
  ): void {
    if (type === "approval_request") {
      const requestId =
        optionalString(payload.id) ?? optionalString(payload.request_id);
      if (requestId) this.approvalOwners.set(requestId, connection);
    }
    if (type === "question_request") {
      const requestId = optionalString(payload.id);
      if (requestId) this.questionOwners.set(requestId, connection);
    }
    if (type === "status_update") {
      const session = this.runtime.sessionStatus(connection.activeSessionKey);
      const model = optionalString(payload.model) || session?.model || "";
      if (model) {
        const normalized = { ...payload };
        delete normalized.max_context;
        const contextLimit = this.contextLimit(model);
        connection.send(
          daemonEvent(type, {
            ...normalized,
            // Zero is the explicit unknown sentinel and clears any previous
            // profile/model window held by a connected client.
            max_context: contextLimit,
          }),
        );
        return;
      }
    }
    connection.send(daemonEvent(type, payload));
  }

  /**
   * Submit a turn with the same tracking as the turn.submit RPC branch:
   * every runtime turn is registered in inFlightTurns so stop() drains it
   * before flushing sessions, and — when an owning connection is supplied —
   * in turnOwners so disconnect() cancels it. The returned promise is the
   * raw submitTurn promise for caller-specific error handling; the tracked
   * view never rejects.
   */
  private cancelTrackedTurn(sessionKey: string): boolean {
    const owner = this.turnOwners.get(sessionKey);
    if (!owner) {
      return this.runtime.cancelTurn(sessionKey);
    }
    // Retain the stop intent while server-side setup (notably compaction) is
    // still awaiting and no runtime controller exists yet. The admission
    // check in submitTrackedTurn consumes the removed ownership and skips
    // launch after setup settles.
    this.turnOwners.delete(sessionKey);
    if (!this.runtime.cancelTurn(sessionKey)) {
      const session = this.runtime.sessionStatus(sessionKey);
      if (session) session.cancelRequested = true;
    }
    return true;
  }

  /**
   * Ask the goal driver whether this session has earned another round.
   *
   * Returns undefined for every refusal — no goal, paused, blocked, complete,
   * disarmed, out of capacity, or a human message already waiting — because
   * from the caller's side they all mean the same thing: stop here and hand the
   * session back to the person.
   */
  /** Record a durable blocker on the live goal, if there is still one to block. */
  private blockGoalForFailure(
    sessionKey: string,
    code: string,
    message: string,
  ): void {
    const session = this.runtime.sessionStatus(sessionKey);
    if (!session) return;
    const goal = getGoal(session.metadata, session.id);
    if (!goal || goal.phase !== "active") return;
    try {
      blockGoal(
        session.metadata,
        session.id,
        { id: goal.id, revision: goal.revision },
        { code, message },
        Date.now(),
      );
    } catch (error) {
      // Losing the blocker record must not mask the failure that caused it.
      console.error(`Could not record goal blocker: ${errorMessage(error)}`);
    }
  }

  /** Pause a goal whose round the user interrupted, leaving it resumable. */
  private pauseGoalAfterInterrupt(sessionKey: string): void {
    const session = this.runtime.sessionStatus(sessionKey);
    if (!session) return;
    const goal = getGoal(session.metadata, session.id);
    if (!goal || goal.phase !== "active") return;
    try {
      pauseGoal(session.metadata, session.id, { id: goal.id, revision: goal.revision }, Date.now());
    } catch (error) {
      console.error(`Could not pause interrupted goal: ${errorMessage(error)}`);
    }
  }

  private admitGoalRound(
    sessionKey: string,
  ): AdmittedGoalRound | undefined {
    const session = this.runtime.sessionStatus(sessionKey);
    if (!session || session.cancelRequested) return undefined;
    const outcome = nextGoalRound(session.metadata, session.id, {
      humanWorkPending: this.runtime.hasPendingSteer?.(sessionKey) === true,
    });
    return "admitted" in outcome ? outcome.admitted : undefined;
  }

  private submitTrackedTurn(
    sessionKey: string,
    text: string,
    emit: (event: DaemonEvent) => void,
    owner: DaemonTransportConnection | undefined,
    options: SubmitTurnOptions = {},
  ): Promise<void> {
    // Reserve ownership before any asynchronous compaction. A second submit
    // cannot join that wait and later become a surprise turn, nor can its
    // cleanup release the first turn's disconnect ownership.
    const ownsCancellation = Boolean(owner && !this.turnOwners.has(sessionKey));
    if (owner && !ownsCancellation) {
      return Promise.reject(new Error("a turn is already active for this session"));
    }
    if (owner) {
      this.turnOwners.set(sessionKey, owner);
    }
    const interactionIds = new Set<string>();
    // Named before the first token streams, not after the turn ends.
    this.seedProvisionalTitle(sessionKey, options.displayText ?? text);
    // Before compaction, so the capture reflects the tree the user is looking
    // at rather than one an auto-compaction turn may already have edited.
    this.captureTurnSnapshot(sessionKey);
    const turnPromise = this.withSessionOperation(sessionKey, async () => {
      await this.autoCompactIfDue(sessionKey, owner);
      // Disconnect may happen while pre-turn compaction is awaiting a provider.
      // Its cancellation removes this ownership entry before a runtime turn
      // exists, so do not launch work that no connection can cancel or observe.
      if (owner && this.turnOwners.get(sessionKey) !== owner) {
        // The submit was already acknowledged to the client, so the suppressed
        // turn still owes it the terminal event a launched one would produce.
        // Without this the client waits forever, exactly as if the daemon had
        // died mid-turn. `unstarted` marks that no turn_begin or assistant
        // content ever existed for this submission (additive wire vocabulary).
        emit({
          type: "turn_end",
          payload: {
            cancelled: true,
            unstarted: true,
            session_id: this.runtime.sessionStatus(sessionKey)?.id ?? sessionKey,
          },
        });
        return;
      }
      // Watched per round: a round that produced nothing must not be allowed to
      // spend the whole budget in a hot loop (see `unproductiveRound` below).
      const round_ = { productive: false, error: undefined as string | undefined };
      const forward = (event: DaemonEvent): void => {
        if (PRODUCTIVE_TURN_EVENTS.has(event.type)) round_.productive = true;
        if (event.type === "notification" && event.payload?.level === "error") {
          round_.error = String(event.payload.message ?? "");
        }
        this.rememberTurnInteraction(event, interactionIds);
        emit(event);
      };
      await this.runtime.submitTurn(sessionKey, text, forward, options);
      // Goal continuation. Each round is a real turn — its own turn_begin and
      // turn_end, its own auto-compaction check, its own place in the
      // transcript — rather than another lap inside one physical turn. That is
      // what lets a person read the run, steer between rounds, and lose only
      // the current round to a crash.
      for (;;) {
        const round = this.admitGoalRound(sessionKey);
        if (!round) break;
        // Rounds ride the same edges a human turn does: a session named after
        // its first exchange should not wait for the whole objective to end,
        // and search must see each round as it lands.
        this.indexSessionForSearch(sessionKey);
        this.maybeGenerateTitle(sessionKey);
        await this.autoCompactIfDue(sessionKey, owner);
        if (owner && this.turnOwners.get(sessionKey) !== owner) return;
        round_.productive = false;
        round_.error = undefined;
        try {
          await this.runtime.submitTurn(sessionKey, round.prompt, forward, {
            displayText: round.displayText,
            goalRound: round.source.round,
          });
        } catch (error) {
          // The round was already reserved in the durable log, so failing to
          // run it must be recorded rather than retried: a silent retry loop
          // against a persistently failing submit is exactly the runaway this
          // subsystem exists to bound.
          this.blockGoalForFailure(
            sessionKey,
            "round-failed",
            `Goal round ${round.source.round} could not run: ${errorMessage(error)}`,
          );
          throw error;
        }
        // An interrupt during an automatic round is the person taking the
        // session back. Pausing is durable and visible — /goal reports paused
        // and offers resume — where merely dropping authority would leave the
        // goal reading "active" while nothing advanced it.
        if (this.runtime.sessionStatus(sessionKey)?.cancelRequested) {
          this.pauseGoalAfterInterrupt(sessionKey);
          break;
        }
        // A round that failed, or produced no work at all, did not advance the
        // objective — and the next round fails the same way. Left alone this is
        // a hot loop: against an out-of-quota provider a live run burned all 24
        // rounds in nine seconds and wrote nothing but its own prompts into the
        // transcript. Stop on the first one and record why.
        //
        // The error notification is the decisive signal, not the absence of
        // output. That same live run showed why: the runtime renders a failure
        // as assistant text, so "did any text arrive" reported a productive
        // round for every single 403.
        if (round_.error !== undefined || !round_.productive) {
          this.blockGoalForFailure(
            sessionKey,
            round_.error === undefined ? "round-produced-nothing" : "round-failed",
            round_.error === undefined
              ? `Goal round ${round.source.round} produced no work.`
              : `Goal round ${round.source.round} failed: ${round_.error}`,
          );
          break;
        }
      }
    });
    const tracked = turnPromise.catch(() => undefined);
    this.inFlightTurns.add(tracked);
    void tracked.then(() => {
      this.inFlightTurns.delete(tracked);
      if (owner && this.turnOwners.get(sessionKey) === owner) {
        this.turnOwners.delete(sessionKey);
      }
      // Final full status for the turn that just settled: token totals,
      // context, and the cumulative telemetry row (turns/steps/timings) all
      // changed during it, and mid-turn ticks carry only deltas. Without
      // this, clients show init-time counters until the next slash command.
      const settled = this.runtime.sessionStatus(sessionKey);
      if (owner && settled) {
        this.emitStatus(owner, settled);
      }
      // A turn that ends or is cancelled without an answer must not leak its
      // approval/question ownership entries into later requests.
      this.releaseTurnInteractions(interactionIds);
      // The runtime persists the session as the turn ends, so this is the
      // incremental feed: the index tracks the transcript that was just saved.
      this.indexSessionForSearch(sessionKey);
      // Title generation rides the same edge: the first exchange just landed.
      this.maybeGenerateTitle(sessionKey);
    });
    return turnPromise;
  }

  private rememberTurnInteraction(
    event: DaemonEvent,
    ids: Set<string>,
  ): void {
    if (event.type === "approval_request") {
      const requestId =
        optionalString(event.payload.id) ??
        optionalString(event.payload.request_id);
      if (requestId) {
        ids.add(requestId);
      }
    } else if (event.type === "question_request") {
      const requestId = optionalString(event.payload.id);
      if (requestId) {
        ids.add(requestId);
      }
    }
  }

  private releaseTurnInteractions(ids: Set<string>): void {
    for (const requestId of ids) {
      this.approvalOwners.delete(requestId);
      this.questionOwners.delete(requestId);
    }
  }

  private dropConnectionRequests(connection: DaemonTransportConnection): void {
    this.providerFlows.delete(connection);
    this.skillCreates.delete(connection);
    for (const [requestId, owner] of this.approvalOwners) {
      if (owner === connection) this.approvalOwners.delete(requestId);
    }
    for (const [requestId, owner] of this.questionOwners) {
      if (owner === connection) this.questionOwners.delete(requestId);
    }
  }

  /**
   * Record an accepted submission id, evicting the oldest entries once the
   * FIFO cap is reached so the set stays bounded for the daemon's lifetime.
   */
  private rememberAcceptedSubmission(submissionKey: string): void {
    while (this.acceptedSubmissionIds.size >= MAX_ACCEPTED_SUBMISSION_IDS) {
      const oldest = this.acceptedSubmissionIds.values().next().value;
      if (oldest === undefined) break;
      this.acceptedSubmissionIds.delete(oldest);
    }
    this.acceptedSubmissionIds.add(submissionKey);
  }

  /**
   * Forget the submissions recorded under any of these session keys or ids.
   *
   * Called when a session is evicted or deleted: its retry window is over,
   * and keeping the entries only delays cap turnover for live sessions.
   */
  private forgetAcceptedSubmissions(sessionKeys: readonly string[]): void {
    if (this.acceptedSubmissionIds.size === 0) return;
    const dropped = new Set(sessionKeys);
    for (const entry of this.acceptedSubmissionIds) {
      const separator = entry.indexOf("\u0000");
      if (separator > 0 && dropped.has(entry.slice(0, separator))) {
        this.acceptedSubmissionIds.delete(entry);
      }
    }
  }

  /** Signal bounding background work (title generation) to this session's life. */
  private sessionSignal(sessionKey: string): AbortSignal {
    let controller = this.sessionLifetimeSignals.get(sessionKey);
    if (!controller) {
      controller = new AbortController();
      this.sessionLifetimeSignals.set(sessionKey, controller);
    }
    return controller.signal;
  }

  /** Abort background work bound to these sessions and forget their signals. */
  private endSessionLifetime(sessionKeys: readonly string[]): void {
    for (const sessionKey of sessionKeys) {
      this.sessionLifetimeSignals.get(sessionKey)?.abort(
        new Error("Session closed"),
      );
      this.sessionLifetimeSignals.delete(sessionKey);
    }
  }

  private disconnect(connection: DaemonTransportConnection): void {
    // Only cancel turns this connection actually submitted: on a shared
    // session key, another client's disconnect must not kill a live turn.
    for (const [key, owner] of this.turnOwners) {
      if (owner !== connection) {
        continue;
      }
      this.turnOwners.delete(key);
      if (this.runtime.sessionStatus(key)?.activeTurnId) {
        this.runtime.cancelTurn(key);
      }
    }
    // Slot keys are minted per connection, so without this the compaction
    // bookkeeping grows for the lifetime of the daemon. A session that outlives
    // its client and refills simply re-earns the count.
    this.clearAutoCompactFailures(connection.activeSessionKey);
    this.dropConnectionRequests(connection);
    // Exchange-less sessions no client is bound to anymore are empty shells:
    // nothing to resume, nothing to persist (the store skips them), nothing to
    // show. Reap them so dead app launches do not litter active_list — and
    // the GUI sidebar — with 0-turn ghosts. Sessions with history, sessions
    // with a live turn, and sessions still pinned by a connected client all
    // survive. (The disconnecting connection is already out of this.connections.)
    const attached = new Set(
      [...this.connections].map((other) => other.activeSessionKey),
    );
    for (const session of this.runtime.listSessions()) {
      if (sessionHasHistory(session) || session.activeTurnId) continue;
      if (attached.has(session.sessionKey)) continue;
      this.runtime.evictSession(session.sessionKey);
    }
  }
}

function sessionHasHistory(session: DaemonSession): boolean {
  return session.messages.length > 0 || session.turnCount > 0;
}

function lastUserMessage(
  messages: readonly DaemonSession["messages"][number][],
): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role.toLowerCase() !== "user") {
      continue;
    }
    const text = messageText(message).trim();
    if (text) {
      return text;
    }
  }
  return "";
}

function discardLastUserTurn(messages: DaemonSession["messages"]): number {
  let dropped = 0;
  while (messages.length) {
    const message = messages.pop();
    if (!message) {
      break;
    }
    dropped += 1;
    if (message.role.toLowerCase() === "user") {
      return dropped;
    }
  }
  return dropped;
}

function toolExecutionName(value: unknown): string {
  if (!isRecord(value)) {
    return "";
  }
  const direct = optionalString(value.name);
  if (direct) {
    return direct;
  }
  const functionValue = value.function;
  return isRecord(functionValue)
    ? (optionalString(functionValue.name) ?? "")
    : "";
}

/** Bounds for replay_tool previews: enough context to recognize a call, never a raw dump. */
const REPLAY_ARGUMENTS_PREVIEW_CHARS = 200;
const REPLAY_RESULT_PREVIEW_CHARS = 160;

/** Executions persist either wire spelling (tool_call_id or legacy-normalized toolCallId). */
function toolExecutionCallId(value: Record<string, unknown>): string {
  return optionalString(value.tool_call_id) ?? optionalString(value.toolCallId) ?? "";
}

/** Executions persist either duration spelling (duration_ms or legacy-normalized durationMs). */
function toolExecutionDurationMs(value: Record<string, unknown>): number | undefined {
  const raw = value.duration_ms ?? value.durationMs;
  return typeof raw === "number" && Number.isFinite(raw) ? raw : undefined;
}

/** Compact an arguments/result payload into one bounded single-line preview. */
function replayPreviewText(value: unknown, limit: number): string {
  const raw =
    typeof value === "string"
      ? value
      : value === undefined || value === null
        ? ""
        : safeJsonStringify(value);
  const compact = raw.replace(/\s+/g, " ").trim();
  return compact.length > limit ? `${compact.slice(0, limit - 1)}…` : compact;
}

function safeJsonStringify(value: unknown): string {
  try {
    return JSON.stringify(value) ?? "";
  } catch {
    return "";
  }
}

function tokenizeSlashArguments(input: string): string[] | undefined {
  const tokens: string[] = [];
  let token = "";
  let quote = "";
  let escaped = false;
  for (const character of input.trim()) {
    if (escaped) {
      token += character;
      escaped = false;
      continue;
    }
    if (character === "\\") {
      escaped = true;
      continue;
    }
    if (quote) {
      if (character === quote) {
        quote = "";
      } else {
        token += character;
      }
      continue;
    }
    if (character === "'" || character === '"') {
      quote = character;
      continue;
    }
    if (/\s/u.test(character)) {
      if (token) {
        tokens.push(token);
        token = "";
      }
      continue;
    }
    token += character;
  }
  if (quote) {
    return undefined;
  }
  if (escaped) {
    token += "\\";
  }
  if (token) {
    tokens.push(token);
  }
  return tokens;
}

function parseCronAddArguments(
  tokens: readonly string[],
): ParsedCronAddArguments {
  const values: Record<string, string> = {};
  const allowed = new Set([
    "at",
    "deliver",
    "prompt",
    "recipient",
    "schedule",
    "workspace",
  ]);
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index] ?? "";
    if (!token.startsWith("--")) {
      return { error: `Unexpected cron argument: \`${token}\`.` };
    }
    const option = token.slice(2);
    const separator = option.indexOf("=");
    const name = (separator < 0 ? option : option.slice(0, separator)).trim();
    const inlineValue = separator < 0 ? undefined : option.slice(separator + 1);
    if (!allowed.has(name)) {
      return { error: `Unknown cron option: \`--${name}\`.` };
    }
    if (values[name] !== undefined) {
      return {
        error: `Cron option \`--${name}\` was provided more than once.`,
      };
    }
    const value = inlineValue ?? tokens[++index];
    if (!value?.trim()) {
      return { error: `Cron option \`--${name}\` requires a value.` };
    }
    values[name] = value.trim();
  }
  const schedule = values.schedule;
  const rawAt = values.at;
  if (Boolean(schedule) === Boolean(rawAt)) {
    return {
      error:
        "Provide exactly one of \`--schedule <five-field-cron>\` or \`--at <ISO-8601-time>\`.",
    };
  }
  const prompt = values.prompt;
  if (!prompt) {
    return { error: "Cron jobs require \`--prompt <text>\`." };
  }
  let at: string | undefined;
  if (rawAt) {
    const parsed = new Date(rawAt);
    if (Number.isNaN(parsed.valueOf())) {
      return { error: "\`--at\` must be a valid ISO-8601 timestamp." };
    }
    at = parsed.toISOString();
  }
  return {
    prompt,
    ...(schedule ? { schedule } : {}),
    ...(at ? { at } : {}),
    ...(values.deliver ? { deliver: values.deliver } : {}),
    ...(values.recipient ? { recipient: values.recipient } : {}),
    ...(values.workspace ? { workspaceId: values.workspace } : {}),
  };
}

function singleCronJobId(tokens: readonly string[]): string | undefined {
  if (tokens.length !== 1) {
    return undefined;
  }
  return optionalString(tokens[0]);
}

function cronUsage(): string {
  return [
    "Usage:",
    "  `/cron list`",
    '  `/cron add --schedule "0 9 * * 1" --prompt "Summarize my PRs"`',
    '  `/cron add --at "2026-07-15T09:00:00Z" --prompt "Send the report"`',
    "  `/cron pause|resume|remove|run <job-id>`",
  ].join("\n");
}

function sessionKey(
  connection: DaemonTransportConnection,
  params: JsonRpcPayload,
): string {
  return requestedSessionKey(params, connection.activeSessionKey);
}

function requestedSessionKey(params: JsonRpcPayload, fallback: string): string {
  return (
    optionalString(params.session_key) || optionalString(params.key) || fallback
  );
}

/** A content part carrying an inline base64 image payload. */
interface InlineImagePart {
  readonly image_url: { readonly url: string };
  readonly type: "image_url";
}

function isInlineImagePart(part: unknown): part is InlineImagePart & Record<string, unknown> {
  return (
    isRecord(part) &&
    part.type === "image_url" &&
    isRecord(part.image_url) &&
    typeof part.image_url.url === "string"
  );
}

/** Placeholder text standing in for an omitted transcript image. */
function transcriptImagePlaceholder(dataUrlBytes: number): string {
  const kilobytes = Math.max(1, Math.round(dataUrlBytes / 1024));
  return `[image omitted: ${kilobytes} KB]`;
}

/** One data-URL image part surviving the inner per-message rule. */
interface TranscriptImageSlot {
  readonly part: InlineImagePart & Record<string, unknown>;
  /** UTF-8 byte length of the full data URL — the cost of keeping it inline. */
  readonly urlBytes: number;
  /** Flipped false when the whole-projection ceiling omits this image. */
  inline: boolean;
}

/** Per-part projection decision built during the classification pass. */
type TranscriptSlot =
  | { readonly kind: "part"; readonly part: Record<string, unknown> }
  | { readonly kind: "image"; readonly image: TranscriptImageSlot };

/**
 * Echo a session's transcript for wire payloads with inline image payloads
 * bounded twice.
 *
 * Inner rule (per message): data-URL image parts stay verbatim until one
 * message's cumulative base64 bytes exceed `MAX_TRANSCRIPT_INLINE_IMAGE_BYTES`.
 * Outer rule (whole projection): surviving images then draw on a shared
 * ceiling, `MAX_TRANSCRIPT_TOTAL_INLINE_IMAGE_BYTES`, spent newest first so
 * the most recent context keeps its real pixels while the OLDEST inline
 * images are omitted first once the ceiling hits. Every omitted image becomes
 * `{ type: "text", text: "[image omitted: N KB]" }`.
 *
 * Frame-safety arithmetic: one outbound frame is capped at 16 MiB on both
 * transports (the Unix socket's `DEFAULT_MAX_SOCKET_OUTPUT_BYTES` and the
 * WebSocket gateway's `DEFAULT_MAX_MESSAGE_BYTES`). Images therefore
 * contribute at most ~2 MiB — an eighth of the cap — plus a few dozen bytes
 * per placeholder to any echoed payload, so even the worst case where every
 * historical turn carries an individually legal ~250 KB screenshot can no
 * longer wedge initialize/open/status from the image side. Non-image text was
 * never bounded here and stays untouched by this projection.
 *
 * Only this wire projection is compacted: provider-facing requests and the
 * live session keep the full images. Part count and ordering are preserved so
 * clients can still align the surrounding content parts, and the total number
 * of omissions is reported alongside the transcript. Remote (http) image URLs
 * are small and never touched.
 */
function projectTranscriptForPayload(
  messages: readonly DaemonTranscriptMessage[],
): { readonly imagesOmitted: number; readonly messages: DaemonTranscriptMessage[] } {
  let imagesOmitted = 0;

  // Pass 1 — classify every part under the inner per-message rule. Data-URL
  // images that survive become shared slot objects so the budgeting pass can
  // flip them without re-walking the messages.
  const messageSlots: Array<TranscriptSlot[] | undefined> = [];
  const candidates: TranscriptImageSlot[] = [];
  for (const message of messages) {
    const { content } = message;
    if (!Array.isArray(content)) {
      messageSlots.push(undefined);
      continue;
    }
    let inlineBytes = 0;
    const slots: TranscriptSlot[] = [];
    for (const part of content) {
      if (
        isInlineImagePart(part) &&
        part.image_url.url.startsWith("data:")
      ) {
        const urlBytes = Buffer.byteLength(part.image_url.url, "utf8");
        if (inlineBytes + urlBytes > MAX_TRANSCRIPT_INLINE_IMAGE_BYTES) {
          slots.push({
            kind: "image",
            image: { part, urlBytes, inline: false },
          });
          continue;
        }
        const image: TranscriptImageSlot = { part, urlBytes, inline: true };
        slots.push({ kind: "image", image });
        candidates.push(image);
        inlineBytes += urlBytes;
        continue;
      }
      slots.push({ kind: "part", part });
    }
    messageSlots.push(slots);
  }

  // Pass 2 — spend the whole-projection ceiling newest first. Walking the
  // candidates in reverse document order makes the oldest inline images drop
  // off first; a later-but-smaller image may still fit after a huge old one
  // was skipped, which only ever keeps more recent context inside the same
  // fixed ceiling.
  let totalInlineBytes = 0;
  for (let index = candidates.length - 1; index >= 0; index -= 1) {
    const image = candidates[index];
    if (image === undefined) continue;
    if (totalInlineBytes + image.urlBytes > MAX_TRANSCRIPT_TOTAL_INLINE_IMAGE_BYTES) {
      image.inline = false;
      continue;
    }
    totalInlineBytes += image.urlBytes;
  }

  // Pass 3 — materialize the projected messages in document order.
  const projected = messages.map((message, index) => {
    const slots = messageSlots[index];
    if (slots === undefined) {
      return structuredClone(message);
    }
    const { content, ...rest } = message;
    const parts = slots.map((slot) => {
      if (slot.kind === "image") {
        if (!slot.image.inline) {
          imagesOmitted += 1;
          return {
            type: "text" as const,
            text: transcriptImagePlaceholder(slot.image.urlBytes),
          };
        }
        return structuredClone(slot.image.part);
      }
      return structuredClone(slot.part);
    });
    // Clone the small remainder so the payload shares no state with the live
    // session, matching what a full structuredClone used to guarantee.
    return { ...structuredClone(rest), content: parts };
  });

  return { imagesOmitted, messages: projected };
}

/**
 * The wire twin of a stored tool execution, for transcript replay: the call's
 * identity and timing without the result body. Clients rebuilding a reopened
 * transcript need verb/args/duration; full results would bloat the frame the
 * same way inline images did before `projectTranscriptForPayload`.
 */
function replayExecutionPayload(exec: unknown): unknown {
  if (!exec || typeof exec !== "object" || Array.isArray(exec)) return exec;
  const { result: _result, permitted: _permitted, ...rest } =
    exec as Record<string, unknown>;
  void _result;
  void _permitted;
  return rest;
}

/**
 * Current git branch of `dir` — null outside a work tree, on a detached
 * HEAD, or whenever git fails/times out. Never fabricates a name.
 */
export async function gitBranch(dir: string): Promise<string | null> {
  try {
    const proc = Bun.spawn(
      ["git", "-C", dir, "rev-parse", "--abbrev-ref", "HEAD"],
      { stdout: "pipe", stderr: "pipe", stdin: "ignore" },
    );
    const timer = setTimeout(() => proc.kill(), 2000);
    try {
      const out = await new Response(proc.stdout).text();
      await proc.exited;
      const branch = out.trim();
      return proc.exitCode === 0 && branch && branch !== "HEAD"
        ? branch
        : null;
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return null;
  }
}

function sessionPayload(
  session: DaemonSession,
  contextLimit: number,
  mcpStatus: Record<string, unknown> = {},
): JsonRpcPayload {
  const model = session.model;
  const contextTokens = sessionContextTokens(session, model);
  const calls = exactSessionApiCalls(session);
  const hierarchy = sessionHierarchyPayload(session.metadata);
  // Derived titles ride the wire too. They used to be blanked here, which is
  // why a chat dispatched from Agent View rendered as `—` for the whole time
  // it worked: the model-written title only exists after the first exchange
  // ENDS. `title_derived` still marks the value as replaceable — it just no
  // longer means "invisible".
  const title = displayTitle(optionalString(session.metadata.title) ?? "") ||
    undefined;
  const subagentSnapshots = subagentSnapshotPanelPayloads(session.metadata);
  // The transcript echo must fit a socket frame even when turns carried
  // multi-megabyte image attachments, so inline data URLs are bounded twice:
  // a small per-message budget, then a whole-projection ceiling spent newest
  // first. Only this wire projection is compacted: session.messages and every
  // provider-facing request keep the full images.
  const transcript = projectTranscriptForPayload(session.messages);
  return {
    id: session.id,
    key: session.sessionKey,
    ...hierarchy,
    ...(title ? { title } : {}),
    ...(subagentSnapshots.length
      ? { subagent_snapshots: subagentSnapshots }
      : {}),
    agent_id: session.agentId,
    workspace: session.workspace,
    cwd: session.cwd,
    active_turn_id: session.activeTurnId,
    mode: session.interactionMode,
    plan_mode: session.planMode,
    model: session.model,
    ...(session.reasoningEffort
      ? { reasoning_effort: session.reasoningEffort }
      : {}),
    messages: session.messages.length,
    message_count: session.messages.length,
    transcript: transcript.messages,
    // Additive replay fields: the stored twins of the streamed tool calls and
    // per-turn reasoning, so a reopened transcript renders the same
    // think → tool rows the live stream did instead of dropping the activity.
    tool_executions: session.toolExecutions.slice(-200).map(replayExecutionPayload),
    thinking_content: session.thinkingContent.slice(-32),
    ...(transcript.imagesOmitted > 0
      ? { transcript_images_omitted: transcript.imagesOmitted }
      : {}),
    ...(session.activeTurnId
      ? {
          inflight: {
            user: session.inflightUser ?? "",
            assistant: session.inflightAssistant ?? "",
            streaming: true,
            // Additive reattach fields: turn-start continuity plus the work
            // so far, since runner-managed sessions only synchronize
            // session.messages at turn end.
            ...(session.inflightStartedAt
              ? { started_at: session.inflightStartedAt / 1000 }
              : {}),
            ...(session.inflightThinking
              ? { thinking: session.inflightThinking }
              : {}),
            ...(session.inflightTools?.length
              ? { tools: session.inflightTools.map((tool) => ({ ...tool })) }
              : {}),
          },
        }
      : {}),
    turn_count: session.turnCount,
    input_tokens: session.totalInputTokens,
    output_tokens: session.totalOutputTokens,
    total_tokens: session.totalInputTokens + session.totalOutputTokens,
    ...sessionRuntimeTelemetryPayload(session.extra.runtime_telemetry),
    // Same estimate the /cost slash reports, now on the session wire: an
    // unknown model prices at 0, and a session without a model omits the
    // field rather than implying a free run.
    ...(session.model
      ? {
          cost_usd: calcCost(
            session.model,
            session.totalInputTokens,
            session.totalOutputTokens,
          ),
        }
      : {}),
    mcp_status: mcpStatus,
    context_tokens: contextTokens,
    context_limit: contextLimit,
    max_context: contextLimit,
    ...(calls === undefined ? {} : { calls }),
    calls_complete: calls !== undefined,
    ...(calls === undefined && session.totalApiCalls !== undefined
      ? { observed_calls: session.totalApiCalls }
      : {}),
    usage_complete: session.usageComplete ?? session.turnCount === 0,
    cancel_requested: session.cancelRequested,
    // Epoch seconds of the latest conversation message. This is not session
    // creation time and does not move for metadata-only rewrites/compaction.
    last_active: session.lastActive / 1000,
    status: session.status,
  };
}

function sessionRuntimeTelemetryPayload(value: unknown): JsonRpcPayload {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return {}
  const record = value as Record<string, unknown>
  const metric = (key: string): number => {
    const candidate = record[key]
    return typeof candidate === 'number' && Number.isFinite(candidate) && candidate >= 0 ? candidate : 0
  }
  const cacheHitRate = record.cacheTelemetryKnown === true ? Math.min(1, metric('cacheHitRate')) : null
  const llmDurationMs = metric('llmDurationMs')
  const llmSteps = Math.trunc(metric('llmSteps'))
  const toolDurationMs = metric('toolDurationMs')
  const toolSteps = Math.trunc(metric('toolSteps'))
  const measuredTokensPerSecond = metric('tokensPerSecond')
  // Older Codex tool-only rounds measured a sub-millisecond terminal-event
  // burst and persisted multi-million-token rates. Such a sample describes
  // event batching, not model decode throughput; suppress it on reattach.
  const tokensPerSecond = measuredTokensPerSecond <= 100_000 ? measuredTokensPerSecond : 0
  const ttftSamples = Math.trunc(metric('ttftSamples'))
  const ttftTotalMs = metric('ttftTotalMs')
  return {
    ...(cacheHitRate === null ? {} : { cache_hit_rate: cacheHitRate }),
    llm_duration_ms: llmDurationMs,
    llm_steps: llmSteps,
    tool_duration_ms: toolDurationMs,
    tool_steps: toolSteps,
    tokens_per_second: tokensPerSecond,
    ttft_samples: ttftSamples,
    ttft_total_ms: ttftTotalMs,
    ...(ttftSamples > 0 ? { ttft_avg_ms: ttftTotalMs / ttftSamples } : {}),
  }
}

function sessionHierarchyPayload(
  metadata: Readonly<Record<string, unknown>>,
): JsonRpcPayload {
  const parentSessionId = optionalString(metadata.parent_session_id);
  const subagentId = optionalString(metadata.subagent_id);
  const declaredKind = optionalString(metadata.session_kind)?.toLowerCase();
  const kind =
    declaredKind === "subagent" || subagentId
      ? "subagent"
      : "main";
  const rootSessionId =
    optionalString(metadata.root_session_id) || parentSessionId;
  return {
    kind,
    session_kind: kind,
    ...(parentSessionId ? { parent_session_id: parentSessionId } : {}),
    ...(rootSessionId ? { root_session_id: rootSessionId } : {}),
    ...(subagentId ? { subagent_id: subagentId } : {}),
  };
}

/**
 * Project the persisted subagent manifest into bounded panel rows. Identity,
 * hierarchy, status, and persisted usage metadata only: `last_input` and
 * `last_output` are child conversation content and stay out of the parent
 * session payload so a resumed transcript never receives a subagent dump.
 */
function subagentSnapshotPanelPayloads(
  metadata: Readonly<Record<string, unknown>>,
): JsonRpcPayload[] {
  const rows: JsonRpcPayload[] = [];
  for (const value of persistedSubagentSnapshotValues(metadata)) {
    const id = optionalString(value.id);
    const status = optionalString(value.status);
    if (!id || !status) continue;
    const row: JsonRpcPayload = { id, status };
    for (const key of [
      "name",
      "title",
      "agent_id",
      "creator_id",
      "parent_id",
      "source_agent_id",
      "model",
      "prompt_profile",
      "summary",
      "error",
      "history_session_id",
      "created_at",
      "updated_at",
    ] as const) {
      const field = value[key];
      if (typeof field === "string" && field) row[key] = field;
      else if (field === null) row[key] = null;
    }
    for (const key of [
      "api_calls",
      "tool_count",
      "input_tokens",
      "output_tokens",
      "reasoning_tokens",
      "queue_size",
    ] as const) {
      const field = value[key];
      if (typeof field === "number" && Number.isFinite(field)) row[key] = field;
    }
    for (const key of ["files_read", "files_written", "rules", "toolsets"] as const) {
      const field = value[key];
      if (Array.isArray(field)) {
        row[key] = field.filter((item): item is string => typeof item === "string");
      }
    }
    if (value.closed === true) row.closed = true;
    rows.push(row);
  }
  return rows;
}

function sessionUsagePayload(
  session: DaemonSession,
  contextMax: number,
): JsonRpcPayload {
  const model = session.model;
  const contextUsed = sessionContextTokens(session, model);
  const calls = exactSessionApiCalls(session);
  const total = session.totalInputTokens + session.totalOutputTokens;
  return {
    model,
    input: session.totalInputTokens,
    output: session.totalOutputTokens,
    total,
    context_used: contextUsed,
    context_max: contextMax,
    context_percent: contextMax > 0 ? (contextUsed / contextMax) * 100 : 0,
    ...(calls === undefined ? {} : { calls }),
    calls_complete: calls !== undefined,
    ...(calls === undefined && session.totalApiCalls !== undefined
      ? { observed_calls: session.totalApiCalls }
      : {}),
    usage_complete: session.usageComplete ?? session.turnCount === 0,
  };
}

function exactSessionApiCalls(session: DaemonSession): number | undefined {
  if (session.apiCallsComplete === true) return session.totalApiCalls ?? 0;
  if (session.turnCount === 0 && session.totalApiCalls === undefined) return 0;
  return undefined;
}

function sessionHistoryPayload(session: DaemonSession): JsonRpcPayload {
  return {
    message_count: session.messages.length,
    turn_count: session.turnCount,
    input_tokens: session.totalInputTokens,
    output_tokens: session.totalOutputTokens,
  };
}

function savedSessionPayload(session: SavedDaemonSession): JsonRpcPayload {
  return {
    id: session.id,
    session_id: session.id,
    key: session.key,
    kind: session.kind,
    session_kind: session.kind,
    resumable: session.resumable,
    title: session.title,
    agent_id: session.agentId,
    // The workspace grouping key: which project folder the chat ran in.
    cwd: session.cwd,
    ...(session.model ? { model: session.model } : {}),
    ...(session.parentSessionId
      ? { parent_session_id: session.parentSessionId }
      : {}),
    ...(session.rootSessionId
      ? { root_session_id: session.rootSessionId }
      : {}),
    ...(session.status ? { status: session.status } : {}),
    ...(session.subagentId ? { subagent_id: session.subagentId } : {}),
    updated_at: session.updatedAt,
    turn_count: session.turnCount,
    messages: session.messageCount,
    message_count: session.messageCount,
    path: session.path,
  };
}

function savedSessionKind(
  value: unknown,
): "all" | "main" | "subagent" | undefined {
  const normalized = optionalString(value)?.toLowerCase();
  return normalized === "all" ||
    normalized === "main" ||
    normalized === "subagent"
    ? normalized
    : undefined;
}

function cronJobPayload(job: CronJob): JsonRpcPayload {
  return {
    id: job.id,
    prompt: job.prompt,
    schedule: job.schedule,
    deliver: job.deliver,
    recipient: job.recipient,
    paused: job.paused,
    oneshot: job.oneshot,
    last_run_at: job.lastRunAt ?? null,
    next_run_at: job.nextRunAt ?? null,
    workspace_id: job.workspaceId ?? null,
  };
}

function snapshotPayload(snapshot: SnapshotRecord): JsonRpcPayload {
  return {
    id: snapshot.id,
    label: snapshot.label,
    commit_sha: snapshot.commitSha,
    created_at: snapshot.createdAt,
    workspace_dir: snapshot.workspaceDir,
    ...(snapshot.sessionId === undefined ? {} : { session_id: snapshot.sessionId }),
    ...(snapshot.turnIndex === undefined ? {} : { turn_index: snapshot.turnIndex }),
  };
}

function searchHitPayload(hit: TranscriptSearchHit): JsonRpcPayload {
  return {
    session_id: hit.sessionId,
    message_index: hit.messageIndex,
    role: hit.role,
    excerpt: hit.excerpt,
    title: hit.title,
    updated_at: hit.updatedAt,
  };
}

function searchStatsPayload(
  stats: ReturnType<TranscriptSearchIndex["stats"]>,
): JsonRpcPayload {
  return {
    sessions: stats.sessions,
    indexed_messages: stats.indexedMessages,
    searchable_messages: stats.searchableMessages,
    truncated_messages: stats.truncatedMessages,
    unrecognized_messages: stats.unrecognizedMessages,
  };
}

function initPayload(
  session: DaemonSession,
  model: string,
  reasoningEffort = "off",
  permissionMode = DEFAULT_PERMISSION_MODE,
  contextLimit = 0,
): JsonRpcPayload {
  return {
    session_id: session.id,
    model,
    cwd: session.cwd,
    context_limit: contextLimit,
    agent_name: session.agentId,
    mode: session.interactionMode,
    plan_mode: session.planMode,
    ultra_mode: session.ultraMode === true,
    reasoning_effort: reasoningEffort,
    permission_mode: permissionMode,
    skills: [],
    skill_descriptions: {},
    head_hash: "",
    version: XERXES_VERSION,
  };
}

function statusUpdatePayload(
  session: DaemonSession,
  model: string,
  contextLimit: number,
  channelData: ChannelStatusData,
  reasoningEffort = "off",
  permissionMode = DEFAULT_PERMISSION_MODE,
  mcpStatus: Record<string, unknown> = {},
): JsonRpcPayload {
  const calls = exactSessionApiCalls(session);
  return {
    model,
    context_tokens: sessionContextTokens(session, model),
    max_context: contextLimit,
    input_tokens: session.totalInputTokens,
    output_tokens: session.totalOutputTokens,
    ...(model
      ? {
          cost_usd: calcCost(
            model,
            session.totalInputTokens,
            session.totalOutputTokens,
          ),
        }
      : {}),
    ...(calls === undefined ? {} : { calls }),
    calls_complete: calls !== undefined,
    ...(calls === undefined && session.totalApiCalls !== undefined
      ? { observed_calls: session.totalApiCalls }
      : {}),
    usage_complete: session.usageComplete ?? session.turnCount === 0,
    // Cumulative session telemetry, additive: turns, step counts, LLM/tool
    // wall time, TTFT, throughput, cache hit rate. The desktop reads the same
    // counters from the session payload; status_update now carries them so
    // the TUI can show the identical live bar without an extra RPC.
    turn_count: session.turnCount,
    ...sessionRuntimeTelemetryPayload(session.extra.runtime_telemetry),
    plan_mode: session.planMode,
    ultra_mode: session.ultraMode === true,
    mode: session.interactionMode,
    reasoning_effort: reasoningEffort,
    permission_mode: permissionMode,
    mcp_status: mcpStatus,
    channels: channelData.channels,
    channels_available: channelData.available,
    channels_configured: channelData.configured,
  };
}

/**
 * Price the live provider request, not a lossy copy of it.
 *
 * The messages go through as they are: mapping them to `{role, content}` threw
 * away `tool_calls`, whose serialized arguments are usually the largest thing
 * in a tool-heavy window — and the summarizer prompt puts them back, so a
 * session could pass the compaction threshold and then overflow the window on
 * the very call meant to shrink it. The system prompt and tool schemas ride
 * every request without ever appearing in the transcript, so they are priced
 * too whenever the turn runner has cached them on the session.
 */
function sessionContextScaffold(session: DaemonSession): {
  systemPrompt: string | undefined;
  toolSchemas: readonly Record<string, unknown>[] | undefined;
} {
  return {
    systemPrompt:
      session.requestScaffold?.systemPrompt ?? session.systemPromptAddendum,
    toolSchemas: session.requestScaffold?.toolSchemas,
  };
}

function sessionContextTokens(session: DaemonSession, model: string): number {
  const scaffold = sessionContextScaffold(session);
  return estimateContextTokens(session.messages, {
    model,
    ...(scaffold.systemPrompt ? { systemPrompt: scaffold.systemPrompt } : {}),
    ...(scaffold.toolSchemas?.length
      ? { toolSchemas: scaffold.toolSchemas }
      : {}),
  });
}

function channelStatusPayload(status: ManagedChannelStatus): JsonRpcPayload {
  return {
    name: status.name,
    adapter_name: status.adapterName,
    enabled: status.enabled,
    ...(status.lastOperation === undefined
      ? {}
      : { last_operation: status.lastOperation }),
    ...(status.lastError === undefined ? {} : { last_error: status.lastError }),
  };
}

function channelStatusEventPayload(data: ChannelStatusData): JsonRpcPayload {
  return {
    channels: data.channels,
    channels_available: data.available,
    channels_configured: data.configured,
  };
}

function looksLikeInternalReplayMessage(text: string): boolean {
  const head = text.trimStart().slice(0, 64);
  if (head.startsWith("[Skill") && head.includes("activated")) {
    return true;
  }
  if (
    [
      "[sub-agent events]",
      "[mid-turn steer from user]",
      "[steer from user]",
      "[steer from user saved for next turn]",
      "[Workspace guard]",
      "[Objective gate]",
      "[Previous conversation summary",
    ].some((prefix) => head.startsWith(prefix))
  ) {
    return true;
  }
  return [
    "Please compact this conversation:",
    "Write a reusable agent skill called",
    "Generate an image matching this brief",
  ].some((prefix) => text.trimStart().startsWith(prefix));
}

/** First message text of a role in the session, used as title-generation input. */
function firstExchangeText(
  session: DaemonSession,
  role: "assistant" | "user",
): string {
  for (const message of session.messages) {
    if (message.role.toLowerCase() !== role) continue;
    const text = messageText(message).trim();
    if (text) return text;
  }
  return "";
}

function messageText(message: DaemonSession["messages"][number]): string {  if (typeof message.text === "string") {
    return message.text.trim();
  }
  const content = message.content;
  if (typeof content === "string") {
    return content.trim();
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        return isRecord(part)
          ? stringValue(part.text) || stringValue(part.content)
          : "";
      })
      .filter(Boolean)
      .join("\n")
      .trim();
  }
  return isRecord(content)
    ? stringValue(content.text) || stringValue(content.content)
    : "";
}

function createCompactionClient(
  model: string,
  profile: ProviderProfile | undefined,
  status: JsonRpcPayload,
): LlmClient {
  return createLlmClient(model, {
    ...(profile?.api_key ? { api_key: profile.api_key } : {}),
    ...(profile?.base_url ? { base_url: profile.base_url } : {}),
    ...(profile?.provider ? { provider: profile.provider } : {}),
    ...(typeof status.base_url === "string" && status.base_url
      ? { base_url: status.base_url }
      : {}),
    ...(typeof status.provider === "string" && status.provider
      ? { provider: status.provider }
      : {}),
  });
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function integerValue(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return 0;
  }
  return Math.max(0, Math.trunc(value));
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function newConnectionKey(): string {
  return crypto.randomUUID().replaceAll("-", "").slice(0, 12);
}

function optionalString(value: unknown): string | undefined {
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

function nullablePositiveSafeInteger(value: unknown): number | null | undefined {
  if (value === null) return null;
  return typeof value === "number" && Number.isSafeInteger(value) && value > 0
    ? value
    : undefined;
}

/** A positive whole count from the wire, or undefined so a default applies. */
function integerOption(value: unknown): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value > 0
    ? value
    : undefined;
}

/**
 * Resolve the routed provider without letting an unroutable model throw.
 *
 * `resolveProvider` rejects an unknown explicit prefix by design, but asking
 * "which reasoning levels apply" must never break a session over a model the
 * registry cannot place — the caller falls back to a generic set instead.
 */
function resolveProviderSafely(
  model: string,
  profile: ProviderProfile | undefined,
): ProviderName | undefined {
  if (!model.trim()) {
    return undefined;
  }
  try {
    return resolveProvider(model, {
      ...(profile?.provider ? { provider: profile.provider } : {}),
      ...(profile?.base_url ? { base_url: profile.base_url } : {}),
    });
  } catch {
    return undefined;
  }
}

function normalizeProviderIdentity(value: string): string {
  return value.trim().toLowerCase().replaceAll("_", "-");
}

function normalizeBaseUrlIdentity(value: string): string {
  return value.trim().replace(/\/+$/u, "");
}

function discoveredContextKey(
  profile: ProviderProfile,
  model: string,
): string {
  return [
    profile.name,
    normalizeProviderIdentity(profile.provider),
    normalizeBaseUrlIdentity(profile.base_url),
    model.trim(),
  ].join("\u0000");
}

function discoveredContextProfilePrefix(profile: ProviderProfile): string {
  return `${profile.name}\u0000`;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  return typeof value === "boolean" ? value : fallback;
}

async function completePath(
  text: string,
  cwd: string,
): Promise<JsonRpcPayload[]> {
  const token = text.trim().split(/\s+/).at(-1) ?? "";
  const mention = token.startsWith("@");
  const raw = mention ? token.slice(1) : token;
  if (mention && !externalMentionPath(raw)) {
    const query = raw.replace(/^"/, "").replace(/^\.\//, "");
    if (!query) {
      return [];
    }
    const result = await searchProjectFileMentions(cwd, query);
    const projectMatches = result.matches.slice(0, 8).map((match) => {
      const relativePath = relative(cwd, match.absolutePath).replaceAll(
        "\\",
        "/",
      );
      const displayPath = relativePath || match.basename;
      return {
        value: mentionCompletionValue(displayPath),
        label: match.relativePath,
        meta: "file",
      };
    });
    if (projectMatches.length) {
      return projectMatches;
    }
    // An explicit path may intentionally target a Git-ignored workspace
    // artifact (for example generated audit output). Preserve direct path
    // navigation when the ranked project index has no eligible file match.
  }
  if (
    !mention &&
    (!raw ||
      (raw[0] !== "/" &&
        raw[0] !== "." &&
        raw[0] !== "~" &&
        !raw.includes("/")))
  ) {
    return [];
  }
  const slash = raw.lastIndexOf("/");
  const prefix = slash >= 0 ? raw.slice(0, slash + 1) : "";
  const base = slash >= 0 ? raw.slice(slash + 1) : raw;
  const directory = completionDirectory(prefix || ".", cwd);
  try {
    const entries = await readdir(directory, { withFileTypes: true });
    return entries
      .filter((entry) => base.startsWith(".") || !entry.name.startsWith("."))
      .filter(
        (entry) =>
          !base || entry.name.toLowerCase().startsWith(base.toLowerCase()),
      )
      .sort((left, right) => left.name.localeCompare(right.name))
      .slice(0, 50)
      .map((entry) => {
        const directorySuffix = entry.isDirectory() ? "/" : "";
        const label = `${entry.name}${directorySuffix}`;
        return {
          value: `${mention ? "@" : ""}${prefix}${label}`,
          label,
          meta: entry.isDirectory() ? "dir" : "file",
        };
      });
  } catch {
    return [];
  }
}

function externalMentionPath(raw: string): boolean {
  return (
    raw.startsWith("/") ||
    raw.startsWith("~/") ||
    raw === "~" ||
    raw.startsWith("../") ||
    /^[A-Za-z]:[\\/]/.test(raw)
  );
}

function mentionCompletionValue(path: string): string {
  return /\s/.test(path) ? `@"${path.replaceAll('"', '\\"')}"` : `@${path}`;
}

function completionDirectory(prefix: string, cwd: string): string {
  const expanded =
    prefix === "~" || prefix === "~/"
      ? homedir()
      : prefix.startsWith("~/")
        ? resolve(homedir(), prefix.slice(2))
        : prefix;
  return isAbsolute(expanded) ? resolve(expanded) : resolve(cwd, expanded);
}

function formatSessionUsage(
  session: DaemonSession,
  contextLimit: number,
): string {
  const total = session.totalInputTokens + session.totalOutputTokens;
  const calls = exactSessionApiCalls(session);
  const contextUsed = sessionContextTokens(session, session.model);
  return [
    `Model: ${session.model || "(not configured)"}`,
    `Messages: ${session.messages.length}`,
    `Turns: ${session.turnCount}`,
    `Input tokens: ${session.totalInputTokens}`,
    `Output tokens: ${session.totalOutputTokens}`,
    `Total tokens: ${total}`,
    `API calls: ${calls === undefined ? "unknown (imported session)" : calls}`,
    `Context used: ${contextUsed}`,
    `Context window: ${contextLimit || "unknown"}`,
  ].join("\n");
}

function formatSessionHistory(session: DaemonSession): string {
  return [
    `Messages: ${session.messages.length}`,
    `Turns: ${session.turnCount}`,
    `Input tokens: ${session.totalInputTokens}`,
    `Output tokens: ${session.totalOutputTokens}`,
  ].join("\n");
}

function isPermissionMode(value: string): value is PermissionMode {
  return (
    value === "accept-all" ||
    value === "auto" ||
    value === "manual" ||
    value === "plan"
  );
}

function runtimePermissionMode(value: unknown): PermissionMode {
  const mode = stringValue(value);
  return isPermissionMode(mode) ? mode : DEFAULT_PERMISSION_MODE;
}

function isDaemonUiAction(value: string): value is DaemonUiAction {
  return (
    value === "paste" ||
    value === "queue" ||
    value === "skin" ||
    value === "statusbar" ||
    value === "voice"
  );
}

function projectInitializationPrompt(
  projectDirectory: string,
  args: string,
): string {
  const request = args.trim();
  return [
    "Initialize this repository for Xerxes using evidence from the current workspace.",
    `Project root: \`${projectDirectory}\`.`,
    ...(request ? ["", `Additional request: ${request}`] : []),
    "",
    "Inspect the repository before changing files. Produce project-specific agent context in `XERXES.md` and `.agents/` only when the current runtime exposes the needed file and sub-agent tools.",
    "Capture real build/test commands, architecture, conventions, and risks. Do not invent a generic template when tooling is unavailable; report the blocker instead.",
  ].join("\n");
}

function numberValue(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function displayedRuntimeConfig(status: JsonRpcPayload): JsonRpcPayload {
  const config: JsonRpcPayload = {};
  for (const key of DISPLAYED_RUNTIME_CONFIG_KEYS) {
    const value = status[key];
    if (
      typeof value === "boolean" ||
      typeof value === "number" ||
      typeof value === "string"
    ) {
      config[key] = value;
    }
  }
  return config;
}

function samplingConfig(
  status: JsonRpcPayload,
): Record<string, boolean | number | string | undefined> {
  return Object.fromEntries(
    NATIVE_SAMPLING_KEYS.map((key) => {
      const value = status[key];
      const configured =
        typeof value === "boolean" ||
        typeof value === "number" ||
        typeof value === "string"
          ? value
          : undefined;
      return [
        key,
        configured ??
          (key === "temperature"
            ? DEFAULT_TEMPERATURE
            : key === "top_k"
              ? DEFAULT_TOP_K
              : undefined),
      ];
    }),
  );
}

function isNativeSamplingKey(value: string): value is NativeSamplingKey {
  return (NATIVE_SAMPLING_KEYS as readonly string[]).includes(value);
}

function parseNativeSamplingValue(
  key: NativeSamplingKey,
  raw: string,
  reasoningEfforts: readonly string[],
): boolean | number | string | undefined {
  if (key === "thinking") {
    if (["on", "true", "1"].includes(raw.toLowerCase())) return true;
    if (["off", "false", "0"].includes(raw.toLowerCase())) return false;
    return undefined;
  }
  if (key === "reasoning_effort") {
    // Validated against the efforts this model publishes; a fixed list would
    // reject `xhigh`/`ultra` on models that accept them and accept `high` on
    // models that do not.
    return reasoningEfforts.find(
      (effort) => effort.toLowerCase() === raw.trim().toLowerCase(),
    );
  }
  if (key === "service_tier") {
    const tier = raw.trim().toLowerCase();
    return ["auto", "default", "flex", "priority"].includes(tier) ? tier : undefined;
  }
  const value = Number(raw);
  if (!Number.isFinite(value)) return undefined;
  if (key === "temperature") {
    return value >= 0 && value <= 2 ? value : undefined;
  }
  if (key === "top_p") {
    return value >= 0 && value <= 1 ? value : undefined;
  }
  if (key === "max_tokens" || key === "top_k" || key === "thinking_budget") {
    return Number.isInteger(value) && value >= 0 && value <= 100_000
      ? value
      : undefined;
  }
  return value >= -2 && value <= 2 ? value : undefined;
}

function invalidSamplingMessage(
  key: NativeSamplingKey,
  reasoningEfforts: readonly string[],
): string {
  if (key === "temperature") {
    return "`temperature` must be a finite number from 0 to 2.";
  }
  if (key === "top_p") {
    return "`top_p` must be a finite number from 0 to 1.";
  }
  if (key === "thinking") {
    return "`thinking` must be `on` or `off`.";
  }
  if (key === "reasoning_effort") {
    return reasoningEfforts.length
      ? `\`reasoning_effort\` must be one of: ${reasoningEfforts.join(", ")}.`
      : "`reasoning_effort` is not available for this model.";
  }
  if (key === "service_tier") {
    return "`service_tier` must be one of: auto, default, flex, priority.";
  }
  return `\`${key}\` must be a valid finite numeric value.`;
}

function agentDefinitionPayload(definition: AgentDefinition): JsonRpcPayload {
  return {
    name: definition.name,
    description: definition.description,
    source: definition.source,
    model: definition.model,
    tools: [...definition.tools],
    allowed_tools:
      definition.allowedTools === null ? null : [...definition.allowedTools],
    exclude_tools: [...definition.excludeTools],
    max_depth: definition.maxDepth,
    isolation: definition.isolation,
  };
}

function profileOverrides(
  profile: ProviderProfile | undefined,
): JsonRpcPayload {
  const clearedSampling = Object.fromEntries(
    NATIVE_SAMPLING_KEYS.map((key) => [key, null]),
  );
  if (!profile) {
    return {
      ...clearedSampling,
      temperature: DEFAULT_TEMPERATURE,
      top_k: DEFAULT_TOP_K,
    };
  }
  return {
    ...clearedSampling,
    temperature: DEFAULT_TEMPERATURE,
    top_k: DEFAULT_TOP_K,
    ...profile.sampling,
    model: profile.model,
    base_url: profile.base_url,
    api_key: profile.api_key,
    // A profile configured by base URL alone carries no provider NAME, and
    // `RuntimeService.reload` skips empty values — so selecting such a
    // profile wrote no provider at all and every later resolution fell back
    // to sniffing the model id. For an OpenRouter catalogue that means
    // reading the vendor in `stealth/ox-alpha` as a routing prefix and
    // throwing `unknown provider prefix 'stealth'`.
    //
    // Resolve it from the profile itself (its base URL already identifies
    // the vendor) so the selected profile's provider always travels with it.
    provider: profile.provider,
  };
}

function profilePayload(
  profile: ProviderProfile & { readonly active: boolean },
): JsonRpcPayload {
  return {
    name: profile.name,
    base_url: profile.base_url,
    model: profile.model,
    provider: profile.provider,
    sampling: { ...profile.sampling },
    active: profile.active,
  };
}

/**
 * The registry's adapter catalog for the provider add/edit form: type name,
 * transport, default endpoint, and the environment variable a blank API key
 * falls back to. Metadata only — `defaultApiKey` and stored keys are never
 * part of this payload.
 */
function providerTypePayloads(): JsonRpcPayload[] {
  return Object.values(PROVIDERS).map(config => ({
    name: config.name,
    transport: config.transport,
    base_url: config.baseUrl ?? null,
    api_key_env: config.apiKeyEnv ?? null,
  }));
}

function modelCapabilityPayload(profile: ProviderProfile, model: string): JsonRpcPayload {
  const resolved = resolvedProfileModelCapabilities(profile, model);
  return {
    id: model,
    ...(resolved.contextLimit === undefined ? {} : { context_limit: resolved.contextLimit }),
    ...(resolved.contextSource === "unknown" ? {} : { context_source: resolved.contextSource }),
    ...(resolved.maxOutputTokens === undefined ? {} : { max_output_tokens: resolved.maxOutputTokens }),
    ...(resolved.outputSource === "unknown" ? {} : { output_source: resolved.outputSource }),
    ...(resolved.contextSource === "override" || resolved.outputSource === "override"
      ? { overridden: true }
      : {}),
  };
}

function agentPresetPayload(preset: AgentPresetEntry): JsonRpcPayload {
  return {
    id: preset.id,
    name: preset.name,
    description: preset.description,
    trust: preset.trust,
    is_default: preset.isDefault,
    manageable: preset.manageable,
    ...(preset.broken ? { broken: preset.broken } : {}),
  };
}

function forgePackagePayload(
  pkg: DeclarativeForgePackage,
  includeTemplate = false,
): JsonRpcPayload {
  return {
    name: pkg.name,
    version: pkg.version,
    description: pkg.description,
    parameters: pkg.parameters.map(parameter => ({
      name: parameter.name,
      description: parameter.description,
      required: parameter.required,
      ...(parameter.defaultValue === undefined ? {} : { default: parameter.defaultValue }),
    })),
    ...(includeTemplate ? { template: pkg.template } : {}),
    created_at: pkg.createdAt,
  };
}

function creatorTracePayload(row: CreatorTraceRow): JsonRpcPayload {
  return {
    action: row.action,
    name: row.name,
    version: row.version,
    status: row.status,
    detail: row.detail,
    at: row.at,
  };
}

function runtimeOverrides(params: JsonRpcPayload): JsonRpcPayload {
  return Object.fromEntries(
    Object.entries(params).filter(([key]) => RUNTIME_OVERRIDE_KEYS.has(key)),
  );
}

function stringRecord(value: unknown): Record<string, string> {
  if (!isRecord(value)) return {};
  return Object.fromEntries(
    Object.entries(value).filter(
      (entry): entry is [string, string] => typeof entry[1] === "string",
    ),
  );
}

async function closeServer(server: Server | undefined): Promise<void> {
  if (!server) {
    return;
  }
  await new Promise<void>((resolve) => server.close(() => resolve()));
}
