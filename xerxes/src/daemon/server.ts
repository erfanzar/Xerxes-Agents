// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, readdir, rm, writeFile } from "node:fs/promises";
import { createServer, type Server, type Socket } from "node:net";
import { homedir } from "node:os";
import { dirname, isAbsolute, join, relative, resolve } from "node:path";

import {
  CATEGORIES,
  listCommands,
  resolveCommand,
  type CommandDefinition,
} from "../bridge/commands.js";
import {
  createCompactionAgent,
  type CompactionCompletionPort,
} from "../agents/compactionAgent.js";
import {
  listAgentDefinitions,
  type AgentDefinition,
} from "../agents/definitions.js";
import {
  ProfileStore,
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
import {
  createChannelMessage,
  MessageDirection,
  type ChannelMessage,
} from "../channels/types.js";
import { routeOutput } from "../cron/delivery.js";
import { CronJob, JobStore, nextFireAt } from "../cron/jobs.js";
import { CronScheduler } from "../cron/scheduler.js";
import {
  defaultSkillDiscoveryDirectories,
  skillMatchesPlatform,
  skillPromptSection,
  SkillRegistry,
} from "../extensions/skills.js";
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
  type JsonRpcPayload,
  type JsonRpcRequest,
} from "../protocol/jsonRpc.js";
import {
  calcCost,
  getContextLimit,
} from "../llms/providerRegistry.js";
import {
  DEFAULT_TEMPERATURE,
  DEFAULT_TOP_K,
} from "../llms/samplingDefaults.js";
import {
  closeLlmClient,
  completeLlm,
  createLlmClient,
  requireConfiguredModel,
  type LlmClient,
} from "../llms/client.js";
import { formatDoctorReport, runAllDoctorChecks } from "../runtime/doctor.js";
import { formatGitUpdateStatus, gitUpdateStatus } from "../runtime/update.js";
import {
  DEFAULT_PERMISSION_MODE,
  type PermissionMode,
} from "../streaming/permissions.js";
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
import { SnapshotManager, type SnapshotRecord } from "../session/snapshots.js";
import { processAtMentions } from "./atMentions.js";
import { DaemonInteractionBoard } from "./interactions.js";
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
  type DaemonEvent,
  type DaemonRuntime,
  type DaemonSession,
  type SavedDaemonSession,
  type SubmitTurnOptions,
  InMemoryDaemonRuntime,
} from "./runtime.js";
import { resolveProjectDirectory, xerxesHome } from "./paths.js";
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
    usage: "Show session token usage",
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
  /** Browser state shared with native operator tools; `/browser` never invents a browser backend. */
  readonly browserManager?: BrowserManager;
  /** Host-owned adapter registry. No channel transport is synthesized when absent. */
  readonly channelManager?: ChannelManager;
  /** Optional Bun HTTP listener that delivers provider webhooks to configured channel adapters. */
  readonly channelWebhook?: Omit<ChannelWebhookServerOptions, "manager">;
  /** Directory used to archive every automatic and manually-run cron result. */
  readonly cronArchiveDirectory?: string;
  /** Testable native scheduler cadence; production defaults to 30 seconds. */
  readonly cronPollInterval?: number;
  /** Shared approval/question state passed to the native agent turn runner. */
  readonly interactions?: DaemonInteractionBoard;
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
  /** Optional host-owned model catalogue lookup for interactive `/provider` setup. */
  readonly providerModelDiscovery?: ProviderModelDiscoveryPort;
  readonly runtime?: DaemonRuntime;
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
  readonly socketPath: string;
  /** Max buffered inbound bytes per Unix connection before the client is dropped. */
  readonly maxSocketFrameBytes?: number;
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
export class DaemonServer {
  private readonly agentDefinitionLoader: (
    cwd: string,
  ) => readonly AgentDefinition[];
  private readonly approvalOwners = new Map<
    string,
    DaemonTransportConnection
  >();
  private readonly browserManager: BrowserManager;
  private readonly channelManager: ChannelManager | undefined;
  private readonly channelWebhookServer: ChannelWebhookServer | undefined;
  private readonly connections = new Set<Connection>();
  private readonly cronArchiveDirectory: string;
  private readonly cronScheduler: CronScheduler;
  private cronSchedulerStarted = false;
  private readonly cronStore: JobStore;
  private readonly cronStoreFactory: () => JobStore;
  private readonly interactions: DaemonInteractionBoard;
  private readonly inFlightTurns = new Set<Promise<void>>();
  private readonly mcpManager: MCPManager | undefined;
  private readonly maxSocketFrameBytes: number;
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
  private readonly profileStore: ProfileStore;
  private readonly questionOwners = new Map<
    string,
    DaemonTransportConnection
  >();
  private readonly runtime: DaemonRuntime;
  private runtimeShutdown = false;
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
  private server: Server | undefined;
  private readonly socketPath: string;
  private readonly toolCatalog: DaemonToolCatalogPort | undefined;
  private readonly turnOwners = new Map<string, DaemonTransportConnection>();
  private readonly uiControl: DaemonUiControlPort | undefined;
  private readonly websocketOptions: DaemonWebSocketGatewayOptions | undefined;
  private websocketGateway: DaemonWebSocketGateway | undefined;

  constructor(options: DaemonServerOptions) {
    this.socketPath = options.socketPath;
    this.pidPath = options.pidPath;
    this.channelManager = options.channelManager;
    this.channelWebhookServer =
      options.channelManager && options.channelWebhook
        ? new ChannelWebhookServer({
            ...options.channelWebhook,
            manager: options.channelManager,
          })
        : undefined;
    this.runtime = options.runtime ?? new InMemoryDaemonRuntime();
    this.agentDefinitionLoader =
      options.agentDefinitionLoader ?? ((cwd) => listAgentDefinitions({ cwd }));
    this.interactions = options.interactions ?? new DaemonInteractionBoard();
    this.maxSocketFrameBytes =
      options.maxSocketFrameBytes ?? DEFAULT_MAX_SOCKET_FRAME_BYTES;
    this.cronStoreFactory =
      options.cronStoreFactory ??
      (() => new JobStore(join(xerxesHome(), "cron", "jobs.json")));
    this.cronStore = this.cronStoreFactory();
    this.cronArchiveDirectory = resolve(
      options.cronArchiveDirectory ?? join(xerxesHome(), "cron", "archive"),
    );
    this.cronScheduler = new CronScheduler(
      this.cronStore,
      (job) => this.runScheduledCronJob(job),
      {
        onComplete: async (job, output) => {
          await this.deliverCronOutput(job, output);
        },
        ...(options.cronPollInterval === undefined
          ? {}
          : { pollInterval: options.cronPollInterval }),
      },
    );
    this.profileStore = options.profileStore ?? new ProfileStore();
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
    this.skillRegistry = options.skillRegistry ?? new SkillRegistry();
    this.slashPluginRegistry =
      options.slashPluginRegistry ?? getDefaultSlashPluginRegistry();
    this.snapshotManagerFactory =
      options.snapshotManagerFactory ??
      ((workspaceDirectory) => new SnapshotManager(workspaceDirectory));
    this.websocketOptions = options.websocket;
    this.browserManager = options.browserManager ?? new BrowserManager();
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
    await mkdir(dirname(this.socketPath), { recursive: true });
    await rm(this.socketPath, { force: true });
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
      this.cronScheduler.start();
      this.cronSchedulerStarted = true;
    } catch (error) {
      this.cronScheduler.stop();
      this.cronSchedulerStarted = false;
      await this.channelWebhookServer?.stop();
      await this.websocketGateway?.stop();
      this.websocketGateway = undefined;
      await closeServer(this.server);
      this.server = undefined;
      await rm(this.socketPath, { force: true });
      await this.shutdownRuntime();
      throw error;
    }
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
      await this.shutdownRuntime();
      return;
    }
    try {
      this.cronScheduler.stop();
      this.cronSchedulerStarted = false;
      await channelWebhook?.stop();
      await this.channelManager?.stopAll();
      this.runtime.cancelAllTurns();
      // Let cancelled turns land their final state sync and saveSession before
      // the flush below persists the last known state for every session.
      await Promise.all([...this.inFlightTurns]);
      await this.runtime.flushSessions();
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
      await rm(this.socketPath, { force: true });
      if (this.pidPath) {
        await rm(this.pidPath, { force: true });
      }
    } finally {
      await this.shutdownRuntime();
    }
  }

  private async shutdownRuntime(): Promise<void> {
    if (this.runtimeShutdown) return;
    this.runtimeShutdown = true;
    await this.runtime.shutdown?.();
  }

  private attach(socket: Socket): void {
    socket.setEncoding("utf8");
    const connection: Connection = {
      socket,
      buffer: "",
      queue: Promise.resolve(),
      activeSessionKey: `tui:${newConnectionKey()}`,
      send: (frame) => {
        if (!socket.destroyed) {
          socket.write(`${JSON.stringify(frame)}\n`);
        }
      },
    };
    this.connections.add(connection);
    socket.on("data", (chunk) => this.receive(connection, chunk));
    socket.on("error", () => socket.destroy());
    socket.on("close", () => {
      this.connections.delete(connection);
      this.disconnect(connection);
    });
  }

  private receive(connection: Connection, chunk: string | Uint8Array): void {
    connection.buffer +=
      typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk);
    if (Buffer.byteLength(connection.buffer, "utf8") > this.maxSocketFrameBytes) {
      // Matches the WebSocket frame cap: an uncapped buffer lets one client
      // exhaust daemon memory, so drop the offending connection.
      console.error(
        "Xerxes daemon dropping client: request exceeds the socket frame limit",
      );
      connection.socket.destroy();
      return;
    }
    let newline = connection.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = connection.buffer.slice(0, newline);
      connection.buffer = connection.buffer.slice(newline + 1);
      if (line.trim()) {
        // Serialize dispatch per connection: concurrent handlers would
        // interleave at awaits and race on shared mutable state such as
        // activeSessionKey. turn.submit returns while its turn runs in the
        // background, so a queued permission_response is never blocked
        // behind the turn it answers.
        connection.queue = connection.queue.then(
          () => this.handleLine(connection, line),
          () => this.handleLine(connection, line),
        );
      }
      newline = connection.buffer.indexOf("\n");
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
      const session = await this.runtime.openSession(
        key,
        optionalString(params.agent_id),
        { cwd },
      );
      connection.activeSessionKey = key;
      return {
        ok: true,
        session: sessionPayload(session, this.contextLimit(session.model)),
      };
    }
    if (method === "session.active_list") {
      return {
        ok: true,
        sessions: this.runtime
          .listSessions()
          .map((session) =>
            sessionPayload(session, this.contextLimit(session.model)),
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
              ...sessionPayload(session, this.contextLimit(session.model)),
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
    if (method === "session.compress") {
      connection.activeSessionKey = sessionKey(connection, params);
      return this.compactSession(connection, false);
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
      const session = this.runtime.sessionStatus(key);
      requireConfiguredModel(
        session?.model || stringValue(this.runtime.status().model),
      );
      const displayText =
        (typeof params.display_text === "string"
          ? params.display_text.trim()
          : "") || text;
      void this.submitTrackedTurn(
        key,
        text,
        (event) => this.emit(connection, event.type, event.payload),
        connection,
        { displayText },
      ).catch((error) =>
        this.emit(connection, "notification", {
          level: "error",
          message: errorMessage(error),
        }),
      );
      return { ok: true };
    }
    if (method === "turn.cancel" || method === "cancel") {
      return { ok: this.runtime.cancelTurn(sessionKey(connection, params)) };
    }
    if (method === "cancel_all") {
      return { ok: true, cancelled: this.runtime.cancelAllTurns() };
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
      return this.setMode(connection, mode, enabled);
    }
    if (method === "set_mode") {
      return this.setMode(connection, optionalString(params.mode) ?? "code");
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
    if (method === "provider_list") {
      return {
        ok: true,
        profiles: this.profileStore.list().map(profilePayload),
      };
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
    if (stripped.startsWith("/") && !/\s/.test(stripped)) {
      return {
        ok: true,
        kind: "slash",
        completions: this.completeSlash(stripped),
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
        stringValue(this.runtime.status().reasoning_effort) || "off",
        runtimePermissionMode(this.runtime.status().permission_mode),
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
        stringValue(this.runtime.status().reasoning_effort) || "off",
        runtimePermissionMode(this.runtime.status().permission_mode),
      ),
    );
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
    const fallbackModels = profile.model.trim() ? [profile.model.trim()] : [];
    if (
      profile.provider === "claude-code" ||
      profile.base_url.startsWith("claude-code://")
    ) {
      return {
        ok: true,
        models: fallbackModels,
        profile: profile.name,
        source: "profile",
      };
    }

    const apiKey = profileDiscoveryApiKey(profile);
    try {
      const catalog = await discoverModelCatalog({
        allowPrivateEndpoint: true,
        apiKey,
        baseUrl: profile.base_url,
        provider: profile.provider,
      });
      this.rememberDiscoveredContextLimits(profile, catalog);
      const models = catalog.map((model) => model.id);
      return models.length
        ? {
            ok: true,
            models,
            profile: profile.name,
            source: "remote",
          }
        : {
            ok: true,
            models: fallbackModels,
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

  private rememberDiscoveredContextLimits(
    profile: ProviderProfile,
    models: readonly DiscoveredModel[],
  ): void {
    const profilePrefix = discoveredContextProfilePrefix(profile);
    for (const key of this.discoveredContextLimits.keys()) {
      if (key.startsWith(profilePrefix)) {
        this.discoveredContextLimits.delete(key);
      }
    }
    for (const model of models) {
      if (model.contextLimit !== undefined) {
        this.discoveredContextLimits.set(
          discoveredContextKey(profile, model.id),
          model.contextLimit,
        );
      }
    }
  }

  private contextLimit(model: string): number {
    const profileName = this.activeRuntimeProfileName();
    const profile = profileName ? this.profileStore.get(profileName) : undefined;
    const discovered = profile
      ? this.discoveredContextLimits.get(discoveredContextKey(profile, model))
      : undefined;
    if (discovered !== undefined) {
      return discovered;
    }
    const status = this.runtime.status();
    return configuredContextLimit(model, {
      provider: status.provider,
      base_url: status.base_url,
    });
  }

  private async handleSlash(
    connection: DaemonTransportConnection,
    raw: string,
  ): Promise<JsonRpcPayload> {
    const command = raw.trim();
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
      case "usage":
        if (!session) {
          this.emitSlash(connection, "No active session yet.", "warning");
          return { ok: false, error: "no active session" };
        }
        this.emitSlash(
          connection,
          formatSessionUsage(
            session,
            this.contextLimit(session.model),
          ),
        );
        return { ok: true };
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
        this.emitSlash(connection, "Cleared. Scrollback is owned by the TUI.");
        return { ok: true };
      case "feedback":
        this.emitSlash(
          connection,
          "Feedback / issues:\n  • GitHub: https://github.com/erfanzar/Xerxes/issues\n  • Native daemon logs: `~/.xerxes/daemon.log`.",
        );
        return { ok: true };
      case "new": {
        this.runtime.evictSession(key);
        const fresh = await this.runtime.openSession(key);
        this.emitSlash(connection, `New session \`${fresh.id}\` started.`);
        this.emitInitDone(connection, fresh);
        this.emitStatus(connection, fresh);
        return {
          ok: true,
          session: sessionPayload(fresh, this.contextLimit(fresh.model)),
        };
      }
      case "stop": {
        const cancelled = this.runtime.cancelTurn(key);
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
      case "model":
        if (!args) {
          this.emitSlash(
            connection,
            `Active model: \`${stringValue(this.runtime.status().model) || "(not configured)"}\`.`,
          );
          return { ok: true };
        }
        this.runtime.reload({ model: args });
        // Persist the choice so a TUI/daemon restart keeps it instead of
        // falling back to the profile's stored default model.
        try {
          this.profileStore?.updateActiveModel(args);
        } catch {
          // Profile persistence is best-effort; the in-memory model applies regardless.
        }
        this.emitSlash(connection, `Model set to \`${args}\`.`);
        await this.emitProviderInit(connection);
        return { ok: true, model: args };
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
        this.runtime.reload({ permission_mode: args });
        const session = this.runtime.sessionStatus(connection.activeSessionKey);
        if (session) {
          this.emitStatus(connection, session);
        }
        this.emitSlash(connection, `Permission mode: \`${args}\`.`);
        return { ok: true, permission_mode: args };
      }
      case "yolo": {
        const current = runtimePermissionMode(
          this.runtime.status().permission_mode,
        );
        const next = current === "accept-all" ? "auto" : "accept-all";
        this.runtime.reload({ permission_mode: next });
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

  private configureReasoning(
    connection: DaemonTransportConnection,
    raw: string,
  ): JsonRpcPayload {
    const current =
      stringValue(this.runtime.status().reasoning_effort) || "off";
    const requested = raw.trim().toLowerCase();
    if (!requested) {
      this.emitSlash(
        connection,
        `Thinking: \`${current}\`\nLevels: off | low | medium | high\nSet with \`/thinking <level>\`.`,
      );
      return { ok: true, reasoning_effort: current };
    }
    if (!["off", "low", "medium", "high"].includes(requested)) {
      this.emitSlash(
        connection,
        "Thinking level must be `off`, `low`, `medium`, or `high`.",
        "warning",
      );
      return { ok: false, error: "invalid reasoning effort" };
    }
    this.runtime.reload({
      reasoning_effort: requested,
      thinking: requested !== "off",
    });
    const active = this.profileStore.active();
    if (active) {
      this.profileStore.updateSampling(active.name, {
        reasoning_effort: requested,
        thinking: requested !== "off",
      });
    }
    const session = this.runtime.sessionStatus(connection.activeSessionKey);
    if (session) {
      this.emitStatus(connection, session);
    }
    this.emitSlash(connection, `Thinking: \`${requested}\`.`);
    return { ok: true, reasoning_effort: requested };
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
    const directories =
      this.skillDirectories ??
      defaultSkillDiscoveryDirectories({
        cwd: session?.cwd ?? process.cwd(),
        userSkillsDirectory: this.skillDirectory,
      });
    await this.skillRegistry.refresh(...directories);
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
    await this.runtime.openSession(sessionKey);
    const argumentsText = argumentParts.join(" ").trim();
    const prompt = [
      `[Skill ${skill.metadata.name}${subcommand ? `:${subcommand}` : ""} activated]`,
      "",
      skillPromptSection(skill),
      ...(argumentsText ? ["", "## User request", argumentsText] : []),
    ].join("\n");
    void this.submitTrackedTurn(
      sessionKey,
      prompt,
      (event) => this.emit(connection, event.type, event.payload),
      connection,
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
      .map(sessionPayload);
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

  private configureSampling(
    connection: DaemonTransportConnection,
    raw: string,
  ): JsonRpcPayload {
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
    const value = parseNativeSamplingValue(name, rawValue);
    if (value === undefined) {
      this.emitSlash(connection, invalidSamplingMessage(name), "warning");
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
    const session = this.runtime.sessionStatus(connection.activeSessionKey);
    if (!session) {
      if (notify) {
        this.emitSlash(connection, "No active session to compact.", "warning");
      }
      return { ok: false, error: "no active session" };
    }
    if (session.activeTurnId) {
      if (notify) {
        this.emitSlash(
          connection,
          "Cannot compact while a turn is running. Use `/stop` first.",
          "warning",
        );
      }
      return { ok: false, error: "turn is running" };
    }
    const model = session.model || stringValue(this.runtime.status().model);
    if (!model) {
      const error = "model is not configured; select a provider model before compacting";
      if (notify) this.emitSlash(connection, error, "warning");
      return { ok: false, error };
    }
    let client: LlmClient | undefined;
    try {
      const profile = this.profileStore?.active();
      client = createCompactionClient(model, profile, this.runtime.status());
      const agent = createCompactionAgent({
        model,
        completion: compactionCompletionPort(client, model),
      });
      const originalCount = session.messages.length;
      const compacted = await agent.summarizeMessages(session.messages);
      const unchanged =
        compacted.length === originalCount &&
        compacted.every(
          (message, index) =>
            JSON.stringify(message) === JSON.stringify(session.messages[index]),
        );
      if (unchanged) {
        if (notify) {
          this.emitSlash(connection, "Nothing to compact.");
        }
        return { ok: true, compacted: false };
      }
      const tokensBefore = estimateContextTokens(session.messages, { model });
      const tokensAfter = estimateContextTokens(compacted, { model });
      session.messages = compacted as DaemonSession["messages"];
      session.metadata.last_compaction = {
        tokens_before: tokensBefore,
        tokens_after: tokensAfter,
      };
      await this.runtime.flushSessions();
      if (notify) {
        this.emitSlash(
          connection,
          `Compacted ${originalCount - compacted.length} message(s): ${tokensBefore} → ${tokensAfter} tokens.`,
        );
      }
      this.emitStatus(connection, session);
      return {
        ok: true,
        compacted: true,
        tokens_before: tokensBefore,
        tokens_after: tokensAfter,
      };
    } catch (error) {
      if (notify) {
        this.emitSlash(
          connection,
          `Compaction failed: ${errorMessage(error)}`,
          "error",
        );
      }
      return { ok: false, error: errorMessage(error) };
    } finally {
      if (client !== undefined) await closeLlmClient(client);
    }
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
    const used = estimateContextTokens(session.messages, {
      model,
    });
    const remaining = Math.max(0, contextLimit - used);
    const percent = contextLimit ? (used / contextLimit) * 100 : 0;
    this.emitSlash(
      connection,
      [
        model
          ? `Context window: ${contextLimit.toLocaleString()} tokens for \`${model}\``
          : "Context window: unknown (model not configured)",
        contextLimit
          ? `Used: ${used.toLocaleString()} (${percent.toFixed(1)}%) · Remaining: ${remaining.toLocaleString()}`
          : `Used: ${used.toLocaleString()} · Remaining: unknown`,
      ].join("\n"),
    );
    return {
      ok: true,
      context_limit: contextLimit,
      used_tokens: used,
      remaining_tokens: remaining,
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
      await this.runtime.flushSessions();
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
      this.runtime.evictSession(activeKey);
    }
    connection.activeSessionKey = target.id;
    this.emitInitDone(connection, session);
    this.emitStatus(connection, session);
    this.replaySessionHistory(connection, session);
    this.emitSlash(connection, `Resumed session \`${session.id}\`.`);
    return {
      ok: true,
      session: sessionPayload(session, this.contextLimit(session.model)),
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
        : sessionPayload(branch, this.contextLimit(branch.model)),
    };
  }

  private async undoLastTurn(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    notify = true,
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
    await this.runtime.flushSessions();
    if (notify) {
      this.emitSlash(
        connection,
        `Undone — dropped ${dropped} message${dropped === 1 ? "" : "s"} from the conversation.`,
      );
    }
    return { ok: true, dropped };
  }

  private async retryLastTurn(
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
      session.messages.splice(0, session.messages.length, ...priorMessages);
      session.turnCount = priorTurnCount;
      this.emitSlash(connection, `Retry failed: ${errorMessage(error)}`, "error");
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

  private async rollbackSnapshot(
    connection: DaemonTransportConnection,
    session: DaemonSession | undefined,
    ref: string,
  ): Promise<JsonRpcPayload> {
    if (!session) {
      this.emitSlash(connection, "No active session yet.", "warning");
      return { ok: false, error: "no active session" };
    }
    if (!ref) {
      this.emitSlash(
        connection,
        "Usage: `/rollback <snapshot-id>` — list with `/snapshots`.",
        "warning",
      );
      return { ok: false, error: "snapshot reference is required" };
    }
    try {
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

  private async saveProvider(
    connection: DaemonTransportConnection,
    params: JsonRpcPayload,
  ): Promise<JsonRpcPayload> {
    this.cancelProviderFlow(connection);
    const name = optionalString(params.name);
    const baseUrl = optionalString(params.base_url);
    const model = optionalString(params.model);
    if (!name || !baseUrl || !model) {
      return { ok: false, error: "name, base_url, and model are required" };
    }
    const provider = optionalString(params.provider);
    const profile = this.profileStore.save({
      name,
      baseUrl,
      apiKey: stringValue(params.api_key),
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
  ): Promise<JsonRpcPayload> {
    const session = await this.runtime.setSessionMode(
      connection.activeSessionKey,
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
      optionalString(params.project_dir) || process.cwd(),
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
        this.runtime.evictSession(key);
      }
    }
    const modelOverride = optionalString(params.model);
    const openOptions = {
      cwd,
      resume: Boolean(resumeId),
      ...(modelOverride ? { model: modelOverride } : {}),
    };
    const session = await this.runtime.openSession(
      key,
      optionalString(params.agent_id),
      openOptions,
    );
    await this.refreshSkills(session);
    const skills = this.skillRegistry
      .all()
      .filter((skill) => skillMatchesPlatform(skill));
    const model = session.model || stringValue(this.runtime.status().model);
    const contextLimit = this.contextLimit(model);
    const initPayload: JsonRpcPayload = {
      session_id: session.id,
      model,
      cwd: session.cwd,
      context_limit: contextLimit,
      agent_name: session.agentId,
      mode: session.interactionMode,
      plan_mode: session.planMode,
      ultra_mode: session.ultraMode === true,
      reasoning_effort:
        stringValue(this.runtime.status().reasoning_effort) || "off",
      permission_mode: runtimePermissionMode(
        this.runtime.status().permission_mode,
      ),
      skills: skills.map((skill) => skill.metadata.name),
      skill_descriptions: Object.fromEntries(
        skills.map((skill) => [
          skill.metadata.name,
          skill.metadata.description,
        ]),
      ),
      head_hash: "",
      version: "0.3.0",
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
      ),
    );
    if (session.messages.length) {
      this.replaySessionHistory(connection, session);
    }
    return {
      ...this.runtimeStatusWithChannels(),
      ...initPayload,
      ok: true,
      session: sessionPayload(session, contextLimit),
      daemon_protocol: DAEMON_PROTOCOL_VERSION,
      daemon_build_id: this.daemonBuildId(),
    };
  }

  private replaySessionHistory(
    connection: DaemonTransportConnection,
    session: DaemonSession,
  ): void {
    let count = 0;
    for (const message of session.messages) {
      const role = message.role.toLowerCase();
      if (role !== "user" && role !== "assistant") {
        continue;
      }
      const text = messageText(message);
      if (!text || (role === "user" && looksLikeInternalReplayMessage(text))) {
        continue;
      }
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
        connection.send(
          daemonEvent(type, {
            ...payload,
            max_context: this.contextLimit(model),
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
  private submitTrackedTurn(
    sessionKey: string,
    text: string,
    emit: (event: DaemonEvent) => void,
    owner: DaemonTransportConnection | undefined,
    options: SubmitTurnOptions = {},
  ): Promise<void> {
    // A duplicate submit while a turn is active fails in submitTurn; keep
    // the running turn's owner so its submitter stays the cancellation tie.
    if (owner && !this.turnOwners.has(sessionKey)) {
      this.turnOwners.set(sessionKey, owner);
    }
    const interactionIds = new Set<string>();
    const turnPromise = this.runtime.submitTurn(
      sessionKey,
      text,
      (event) => {
        this.rememberTurnInteraction(event, interactionIds);
        emit(event);
      },
      options,
    );
    const tracked = turnPromise.catch(() => undefined);
    this.inFlightTurns.add(tracked);
    void tracked.then(() => {
      this.inFlightTurns.delete(tracked);
      if (owner && this.turnOwners.get(sessionKey) === owner) {
        this.turnOwners.delete(sessionKey);
      }
      // A turn that ends or is cancelled without an answer must not leak its
      // approval/question ownership entries into later requests.
      this.releaseTurnInteractions(interactionIds);
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
    this.dropConnectionRequests(connection);
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

function sessionPayload(
  session: DaemonSession,
  contextLimit = configuredContextLimit(session.model),
): JsonRpcPayload {
  const model = session.model;
  const contextTokens = sessionContextTokens(session, model);
  const calls = exactSessionApiCalls(session);
  const hierarchy = sessionHierarchyPayload(session.metadata);
  const title = optionalString(session.metadata.title);
  return {
    id: session.id,
    key: session.sessionKey,
    ...hierarchy,
    ...(title ? { title } : {}),
    agent_id: session.agentId,
    workspace: session.workspace,
    cwd: session.cwd,
    active_turn_id: session.activeTurnId,
    mode: session.interactionMode,
    plan_mode: session.planMode,
    model: session.model,
    messages: session.messages.length,
    message_count: session.messages.length,
    turn_count: session.turnCount,
    input_tokens: session.totalInputTokens,
    output_tokens: session.totalOutputTokens,
    total_tokens: session.totalInputTokens + session.totalOutputTokens,
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
    status: session.status,
  };
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

function sessionUsagePayload(
  session: DaemonSession,
  contextMax = configuredContextLimit(session.model),
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
    context_percent: contextMax ? (contextUsed / contextMax) * 100 : 0,
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
  };
}

function initPayload(
  session: DaemonSession,
  model: string,
  reasoningEffort = "off",
  permissionMode = DEFAULT_PERMISSION_MODE,
  contextLimit = configuredContextLimit(model),
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
    version: "0.3.0",
  };
}

function statusUpdatePayload(
  session: DaemonSession,
  model: string,
  contextLimit: number,
  channelData: ChannelStatusData,
  reasoningEffort = "off",
  permissionMode = DEFAULT_PERMISSION_MODE,
): JsonRpcPayload {
  const calls = exactSessionApiCalls(session);
  return {
    model,
    context_tokens: sessionContextTokens(session, model),
    max_context: contextLimit,
    input_tokens: session.totalInputTokens,
    output_tokens: session.totalOutputTokens,
    ...(calls === undefined ? {} : { calls }),
    calls_complete: calls !== undefined,
    ...(calls === undefined && session.totalApiCalls !== undefined
      ? { observed_calls: session.totalApiCalls }
      : {}),
    usage_complete: session.usageComplete ?? session.turnCount === 0,
    plan_mode: session.planMode,
    ultra_mode: session.ultraMode === true,
    mode: session.interactionMode,
    reasoning_effort: reasoningEffort,
    permission_mode: permissionMode,
    mcp_status: {},
    channels: channelData.channels,
    channels_available: channelData.available,
    channels_configured: channelData.configured,
  };
}

function sessionContextTokens(session: DaemonSession, model: string): number {
  return estimateContextTokens(
    session.messages.map((message) => ({
      role: message.role,
      content: message.content ?? message.text ?? "",
    })),
    { model },
  );
}

function configuredContextLimit(
  model: string,
  overrides: Readonly<Record<string, unknown>> = {},
): number {
  const configured = model.trim();
  return configured ? getContextLimit(configured, overrides) : 0;
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

function messageText(message: DaemonSession["messages"][number]): string {
  if (typeof message.text === "string") {
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

function compactionCompletionPort(
  client: LlmClient,
  model: string,
): CompactionCompletionPort {
  return async (request) => {
    const result = await completeLlm(client, {
      model,
      messages: [{ role: "user", content: request.prompt }],
      maxTokens: request.maxTokens,
      temperature: request.temperature,
    });
    return result.content;
  };
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
): boolean | number | string | undefined {
  if (key === "thinking") {
    if (["on", "true", "1"].includes(raw.toLowerCase())) return true;
    if (["off", "false", "0"].includes(raw.toLowerCase())) return false;
    return undefined;
  }
  if (key === "reasoning_effort") {
    return ["off", "low", "medium", "high"].includes(raw.toLowerCase())
      ? raw.toLowerCase()
      : undefined;
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

function invalidSamplingMessage(key: NativeSamplingKey): string {
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
    return "`reasoning_effort` must be `off`, `low`, `medium`, or `high`.";
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
