// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { version } from "../package.json" with { type: "json" };
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { AcpAgentRunner } from "./acp/runner.js";
import {
  ACP_HELP,
  parseAcpCommandOptions,
  type AcpPermissionMode,
} from "./acp/command.js";
import { writeAcpRegistryFile } from "./acp/registry.js";
import { AcpServer } from "./acp/server.js";
import { serveACPStdio } from "./acp/transport.js";
import { loadAgentDefinitions } from "./agents/definitions.js";
import {
  BunDiscordGatewayWebSocketPort,
  FetchDiscordGatewayRestPort,
} from "./channels/discordGateway.js";
import { FetchDiscordApplicationRestPort } from "./channels/discordApplications.js";
import { ProfileStore } from "./bridge/profiles.js";
import {
  createDaemonChannelManager,
  daemonChannelWebhookOptions,
} from "./daemon/channels.js";
import { loadSystemDaemonConfig, type DaemonConfig } from "./daemon/config.js";
import type { DaemonInteractionBoard } from "./daemon/interactions.js";
import { daemonPaths, xerxesHome } from "./daemon/paths.js";
import { createProductionInteractionBoard } from "./daemon/productionInteractions.js";
import { runtimeConnection } from "./daemon/runtimeConnection.js";
import { InMemoryDaemonRuntime } from "./daemon/runtime.js";
import { daemonBuildIdForEntry } from "./daemon/sourceBuild.js";
import { DaemonSubagentEventBus } from "./daemon/subagentEvents.js";
import { createNativeSubagentHost } from "./daemon/subagentHost.js";
import { AgentTurnRunner, formatSubagentResults } from "./daemon/turnRunner.js";
import {
  defaultSkillDiscoveryDirectories,
  SkillRegistry,
} from "./extensions/skills.js";
import {
  ToolRegistry,
  type ToolExecutionContext,
} from "./executors/toolRegistry.js";
import { createLlmClient } from "./llms/client.js";
import { AgentMemory } from "./memory/agentMemory.js";
import { getAgentSelfMemory } from "./memory/agentSelfMemory.js";
import { ContextualMemory } from "./memory/contextualMemory.js";
import {
  BrowserManager,
  registerBrowserManagerTools,
} from "./operators/browser.js";
import { INSTALL_HELP, runInstallCommand } from "./runtime/companionInstall.js";
import { bootstrap, bootstrapSubagentsForAgent } from "./runtime/bootstrap.js";
import {
  formatDoctorReport,
  hasDoctorFailures,
  runAllDoctorChecks,
} from "./runtime/doctor.js";
import { resolveTuiEntry } from "./runtime/distribution.js";
import { registerInteractionModeTool } from "./runtime/interactionModeTool.js";
import { DaemonTranscriptStore } from "./session/daemonTranscript.js";
import { UPDATE_HELP, runUpdateCommand } from "./runtime/update.js";
import { withTerminalWatchdog } from "./ui/lib/terminalModes.js";
import {
  DEFAULT_EXPORT_FORMAT,
  EXPORT_FORMATS,
  buildSessionExport,
  formatSessionExport,
  listSavedSessions,
  savedSessionSummary,
  selectSavedSession,
} from "./runtime/sessionExport.js";
import {
  registerClaudeAgentTools,
  registerClaudeSkillTool,
  registerCoreTools,
} from "./tools/index.js";
import type { MemoryToolContext } from "./tools/memoryTools.js";
import { createAgentState } from "./streaming/events.js";
import { runTurn } from "./streaming/loop.js";
import { runBundledSkillCli } from "./skills/cli.js";

const HELP = `Xerxes ${version} (Bun runtime)

Usage:
  xerxes [prompt]
  xerxes --resume <session_id> [prompt]
  xerxes acp
  xerxes daemon
  xerxes telegram --token <token>
  xerxes install --cloud-code [--force] [--dry-run]
  xerxes doctor
  xerxes update [--check] [--dry-run] [--apply]
  xerxes export [session]
  xerxes skill <skill> [arguments]
  xerxes --help
  xerxes --version

One-shot, daemon, ACP, API, and the interactive TypeScript TUI run on Bun.
Browser tools attach only to an explicitly supplied Chromium CDP endpoint; use /browser in the TUI or the daemon browser command to connect.`;

const [argument, ...argumentsAfterCommand] = Bun.argv.slice(2);

if (argument === "--help" || argument === "-h") {
  console.log(HELP);
  process.exit(0);
} else if (argument === "--version" || argument === "-V") {
  console.log(version);
  process.exit(0);
} else if (argument === "skill") {
  process.exit(await runBundledSkillCli(argumentsAfterCommand));
} else if (argument === "doctor") {
  const report = runAllDoctorChecks();
  console.log(formatDoctorReport(report));
  process.exit(hasDoctorFailures(report) ? 1 : 0);
} else if (argument === "install") {
  if (
    argumentsAfterCommand.includes("--help") ||
    argumentsAfterCommand.includes("-h")
  ) {
    console.log(INSTALL_HELP);
  } else {
    await runInstallCommand(argumentsAfterCommand);
  }
} else if (argument === "update") {
  if (
    argumentsAfterCommand.includes("--help") ||
    argumentsAfterCommand.includes("-h")
  ) {
    console.log(UPDATE_HELP);
  } else {
    await runUpdateCommand(argumentsAfterCommand);
  }
} else if (argument === "export") {
  await runExport(argumentsAfterCommand);
} else if (argument === "daemon") {
  const projectDirectory = optionValue(argumentsAfterCommand, "--project-dir");
  const config = loadSystemDaemonConfig({
    ...(projectDirectory ? { projectDirectory } : {}),
  });
  const socketPath =
    optionValue(argumentsAfterCommand, "--socket") ??
    daemonPaths(projectDirectory).socketPath;
  const pidPath = optionValue(argumentsAfterCommand, "--pid-file");
  await runDaemon(config, projectDirectory, socketPath, pidPath);
  process.exit(0);
} else if (argument === "telegram") {
  const token =
    optionValue(argumentsAfterCommand, "--token") ??
    process.env.TELEGRAM_BOT_TOKEN?.trim();
  if (!token)
    throw new Error("telegram requires --token or TELEGRAM_BOT_TOKEN");
  const projectDirectory = optionValue(argumentsAfterCommand, "--project-dir");
  const config = telegramDaemonConfig(
    loadSystemDaemonConfig({
      ...(projectDirectory ? { projectDirectory } : {}),
    }),
    token,
    optionValue(argumentsAfterCommand, "--host"),
    optionValue(argumentsAfterCommand, "--port"),
  );
  const socketPath =
    optionValue(argumentsAfterCommand, "--socket") ??
    daemonPaths(projectDirectory).socketPath;
  const pidPath = optionValue(argumentsAfterCommand, "--pid-file");
  await runDaemon(config, projectDirectory, socketPath, pidPath);
  process.exit(0);
} else if (argument === "acp") {
  await runAcp(argumentsAfterCommand);
} else if (argument === "-r" || argument === "--resume") {
  const sessionId = argumentsAfterCommand[0]?.trim();
  if (!sessionId) throw new Error("The --resume option requires a session ID");
  const prompt = argumentsAfterCommand.slice(1).join(" ").trim();
  if (prompt) {
    await runResumedOneShot(sessionId, prompt);
  } else {
    await runTui(sessionId);
  }
} else if (argument === undefined) {
  const prompt = process.stdin.isTTY ? "" : await readStandardInput();
  if (prompt) {
    await runOneShot(prompt);
  } else if (process.stdin.isTTY) {
    await runTui();
  } else {
    throw new Error("No prompt was provided on standard input");
  }
} else {
  await runOneShot([argument, ...argumentsAfterCommand].join(" "));
}

function optionValue(
  args: readonly string[],
  flag: string,
): string | undefined {
  const index = args.indexOf(flag);
  const value = index >= 0 ? args[index + 1] : undefined;
  return value && !value.startsWith("-") ? value : undefined;
}

async function runDaemon(
  config: DaemonConfig,
  projectDirectory: string | undefined,
  socketPath: string,
  pidPath: string | undefined,
): Promise<void> {
  const { DaemonServer } = await import("./daemon/server.js");
  const profileStore = new ProfileStore();
  const interactions = createProductionInteractionBoard({
    onApprovalStoreError: (error) => {
      console.error(`Could not persist approval decision: ${errorMessage(error)}`);
    },
  });
  const browserManager = new BrowserManager();
  const skillRegistry = new SkillRegistry();
  const buildId = await daemonBuildIdForEntry(
    import.meta.dir,
    fileURLToPath(import.meta.url),
  );
  const runtime = daemonRuntime(
    config,
    projectDirectory,
    profileStore,
    interactions,
    browserManager,
    { ...(buildId ? { buildId } : {}), skillRegistry },
  );
  const channelManager = createDaemonChannelManager(config, runtime, {
    discordApplicationRest: new FetchDiscordApplicationRestPort(),
    discordGatewayPorts: {
      rest: new FetchDiscordGatewayRestPort(),
      webSocket: new BunDiscordGatewayWebSocketPort(),
    },
    environment: process.env,
    ...(projectDirectory === undefined ? {} : { projectDirectory }),
  });
  let finishDaemon: (() => void) | undefined;
  const daemonLifetime = new Promise<void>((resolveLifetime) => {
    finishDaemon = resolveLifetime;
  });
  const finish = () => finishDaemon?.();
  const daemon = new DaemonServer({
    socketPath,
    runtime,
    interactions,
    browserManager,
    profileStore,
    skillRegistry,
    onRestart: finish,
    onShutdown: finish,
    websocket: websocketOptions(config),
    ...(channelManager.hasConfiguredChannels ? { channelManager } : {}),
    ...(channelManager.hasWebhookChannels()
      ? { channelWebhook: daemonChannelWebhookOptions(config) }
      : {}),
    ...(pidPath ? { pidPath } : {}),
  });
  try {
    await daemon.start();
    await channelManager.startConfigured();
  } catch (error) {
    await channelManager.stopAll();
    await daemon.stop();
    throw error;
  }
  console.error("Xerxes Bun daemon listening on " + socketPath);
  process.once("SIGINT", finish);
  process.once("SIGTERM", finish);
  try {
    await daemonLifetime;
  } finally {
    process.off("SIGINT", finish);
    process.off("SIGTERM", finish);
    await daemon.stop();
  }
}

function telegramDaemonConfig(
  config: DaemonConfig,
  token: string,
  host: string | undefined,
  port: string | undefined,
): DaemonConfig {
  const existing = config.channels.telegram ?? {};
  const settings = isRecord(existing.settings) ? existing.settings : {};
  const normalizedHost = host?.trim();
  const normalizedPort = telegramPort(port);
  return {
    ...config,
    control: {
      ...config.control,
      ...(normalizedHost ? { websocket_host: normalizedHost } : {}),
      ...(normalizedPort === undefined
        ? {}
        : { websocket_port: normalizedPort }),
    },
    channels: {
      ...config.channels,
      telegram: {
        ...existing,
        type: "telegram",
        enabled: true,
        settings: { ...settings, token },
      },
    },
  };
}

function telegramPort(value: string | undefined): number | undefined {
  if (value === undefined) return undefined;
  if (!/^\d+$/.test(value))
    throw new Error("telegram --port must be an integer between 0 and 65535");
  const port = Number.parseInt(value, 10);
  if (port < 0 || port > 65_535)
    throw new Error("telegram --port must be an integer between 0 and 65535");
  return port;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

interface ExportCommandOptions {
  readonly allProjects: boolean;
  readonly format: string;
  readonly includeArchive: boolean;
  readonly list: boolean;
  readonly output: string | undefined;
  readonly projectDirectory: string | undefined;
  readonly session: string;
  readonly storeDirectory: string | undefined;
}

async function runExport(args: readonly string[]): Promise<void> {
  try {
    const options = parseExportOptions(args);
    const scope = {
      ...(options.storeDirectory === undefined
        ? {}
        : { storeDir: options.storeDirectory }),
      ...(options.projectDirectory === undefined
        ? {}
        : { projectDir: options.projectDirectory }),
    };
    if (options.list) {
      printExportSessionList(
        (await listSavedSessions(scope)).map(savedSessionSummary),
      );
      return;
    }
    const saved = await selectSavedSession(options.session, scope);
    const exportRecord = await buildSessionExport(saved, {
      includeArchive: options.includeArchive,
    });
    const rendered = formatSessionExport(exportRecord, options.format);
    if (options.output === undefined) {
      process.stdout.write(rendered);
      return;
    }
    const output = resolve(options.output);
    await mkdir(dirname(output), { recursive: true });
    await writeFile(output, rendered, "utf8");
    console.log(
      "Exported session " + exportRecord.session.id + " to " + output,
    );
  } catch (error) {
    console.error("Export failed: " + errorMessage(error));
    process.exitCode = 1;
  }
}

function parseExportOptions(args: readonly string[]): ExportCommandOptions {
  let allProjects = false;
  let format = DEFAULT_EXPORT_FORMAT;
  let includeArchive = true;
  let list = false;
  let output: string | undefined;
  let projectDirectory: string | undefined;
  let session = "";
  let storeDirectory: string | undefined;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === undefined) continue;
    if (argument === "--all-projects") {
      allProjects = true;
      continue;
    }
    if (argument === "--list") {
      list = true;
      continue;
    }
    if (argument === "--no-archive") {
      includeArchive = false;
      continue;
    }
    if (argument === "--format") {
      format = requiredCommandValue(args, ++index, argument);
      continue;
    }
    if (argument === "--project") {
      projectDirectory = requiredCommandValue(args, ++index, argument);
      continue;
    }
    if (argument === "--session") {
      session = requiredCommandValue(args, ++index, argument);
      continue;
    }
    if (argument === "--store-dir") {
      storeDirectory = requiredCommandValue(args, ++index, argument);
      continue;
    }
    if (argument === "--output" || argument === "-o") {
      output = requiredCommandValue(args, ++index, argument);
      continue;
    }
    if (argument.startsWith("-")) {
      throw new Error("Unknown export option: " + argument);
    }
    if (session) {
      throw new Error("Only one session selector may be provided");
    }
    session = argument;
  }
  if (!EXPORT_FORMATS.includes(format as (typeof EXPORT_FORMATS)[number])) {
    throw new Error("Unsupported export format: " + format);
  }
  return {
    allProjects,
    format,
    includeArchive,
    list,
    output,
    projectDirectory: allProjects
      ? undefined
      : (projectDirectory ?? process.cwd()),
    session,
    storeDirectory,
  };
}

function requiredCommandValue(
  args: readonly string[],
  index: number,
  flag: string,
): string {
  const value = args[index]?.trim();
  if (!value || value.startsWith("-")) {
    throw new Error(flag + " requires a value");
  }
  return value;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function printExportSessionList(
  sessions: readonly ReturnType<typeof savedSessionSummary>[],
): void {
  if (!sessions.length) {
    console.log("No saved Xerxes sessions found.");
    return;
  }
  for (const session of sessions) {
    console.log(
      session.id +
        "  " +
        session.messages +
        " message(s), " +
        session.turn_count +
        " turn(s), updated " +
        session.updated_at,
    );
    const title = session.title.replace(/\n/g, " ").trim();
    if (title) console.log("  title: " + title);
    if (session.project_dir) console.log("  project: " + session.project_dir);
  }
}

async function readStandardInput(): Promise<string> {
  return (await new Response(Bun.stdin.stream()).text()).trim();
}

async function runTui(resumeSessionId = ""): Promise<void> {
  if (!process.stdin.isTTY) {
    throw new Error("The interactive TUI requires a terminal");
  }
  const entry = resolveTuiEntry(import.meta.dir);
  if (!entry) {
    throw new Error(
      "The OpenTUI bundle is missing. Run bun run build:ui or reinstall Xerxes.",
    );
  }
  const projectDirectory = resolve(process.cwd());
  const environment: Record<string, string | undefined> = {
    ...process.env,
    XERXES_CWD: projectDirectory,
    XERXES_PROJECT_DIR: projectDirectory,
  };
  // Build identity belongs to this executable. Never let a stale exported
  // environment value make a newly installed TUI accept an older daemon.
  const buildId = await daemonBuildIdForEntry(
    import.meta.dir,
    fileURLToPath(import.meta.url),
  );
  if (buildId) {
    environment.XERXES_DAEMON_BUILD_ID = buildId;
    environment.XERXES_EXPECTED_DAEMON_BUILD_ID = buildId;
  }
  if (resumeSessionId) environment.XERXES_TUI_RESUME = resumeSessionId;
  if (
    !environment.XERXES_TUI_BUN_DAEMON?.trim() &&
    !environment.XERXES_BUN_DAEMON?.trim()
  ) {
    environment.XERXES_TUI_BUN_DAEMON = fileURLToPath(import.meta.url);
  }
  const exitCode = await withTerminalWatchdog(async () => {
    const child = Bun.spawn([process.execPath, entry], {
      cwd: projectDirectory,
      env: environment,
      stderr: "inherit",
      stdin: "inherit",
      stdout: "inherit",
    });

    return child.exited;
  });
  if (exitCode !== 0) {
    console.error(
      `Xerxes TUI exited unexpectedly (code ${exitCode}); terminal state was restored.`,
    );
    process.exitCode = exitCode;
  }
}

function daemonRuntime(
  config: DaemonConfig,
  projectDirectory: string | undefined,
  profileStore: ProfileStore,
  interactions?: DaemonInteractionBoard,
  browserManager?: BrowserManager,
  host: {
    readonly buildId?: string;
    readonly skillRegistry?: SkillRegistry;
  } = {},
): InMemoryDaemonRuntime {
  const workspaceRoot = projectDirectory ?? config.projectDirectory;
  const home = xerxesHome();
  const transcriptStore = new DaemonTranscriptStore({
    currentProjectDirectory: workspaceRoot,
    directory: join(home, "sessions"),
    workspaceRoot: join(home, "agents"),
  });
  const agentMemories = new Map<string, AgentMemory>();
  const memoryToolContext = memoryToolContextResolver();
  const memoryForProject = (root: string): AgentMemory => {
    const normalizedRoot = resolve(root);
    const existing = agentMemories.get(normalizedRoot);
    if (existing) return existing;
    const memory = new AgentMemory({ projectRoot: normalizedRoot });
    agentMemories.set(normalizedRoot, memory);
    return memory;
  };
  const initialConnection = runtimeConnection(config, profileStore.active());
  const initialSettings: Record<string, unknown> = {
    ...config.runtime,
    ...(initialConnection
      ? {
          model: initialConnection.model,
          permission_mode: initialConnection.permissionMode,
          ...(initialConnection.apiKey
            ? { api_key: initialConnection.apiKey }
            : {}),
          ...(initialConnection.baseUrl
            ? { base_url: initialConnection.baseUrl }
            : {}),
          ...(initialConnection.provider
            ? { provider: initialConnection.provider }
            : {}),
          ...(initialConnection.maxTokens === undefined
            ? {}
            : { max_tokens: initialConnection.maxTokens }),
          ...(initialConnection.temperature === undefined
            ? {}
            : { temperature: initialConnection.temperature }),
          ...(initialConnection.topK === undefined
            ? {}
            : { top_k: initialConnection.topK }),
          ...(initialConnection.topP === undefined
            ? {}
            : { top_p: initialConnection.topP }),
          ...(initialConnection.responsesApi === undefined
            ? {}
            : { responses_api: initialConnection.responsesApi }),
        }
      : {}),
  };
  const subagentEvents = new DaemonSubagentEventBus();
  let subagentHost: ReturnType<typeof createNativeSubagentHost> | undefined;
  let runtime: InMemoryDaemonRuntime | undefined;
  let activeToolCount = 0;
  const runnerFactory = (settings: Readonly<Record<string, unknown>>) => {
    const connection = runtimeConnection(
      { ...config, runtime: { ...config.runtime, ...settings } },
      profileStore.active(),
    );
    if (!connection || connection.provider === "claude-code") {
      subagentHost?.invalidateAll();
      activeToolCount = 0;
      return undefined;
    }
    const tools = new ToolRegistry();
    registerCoreTools(tools, {
      workspaceRoot,
      agentMemoryTools: {
        resolveMemory: (context) => {
          const projectRoot = context.metadata.project_root;
          return memoryForProject(
            typeof projectRoot === "string" ? projectRoot : workspaceRoot,
          );
        },
        resolveSelfMemory: (context) =>
          getAgentSelfMemory(context.agentId ?? "default"),
      },
      memoryTools: { resolveContext: memoryToolContext.resolve },
    });
    if (browserManager) {
      registerBrowserManagerTools(tools, browserManager);
    }
    if (interactions) {
      registerDaemonQuestionTool(tools);
    }
    if (host.skillRegistry) {
      registerClaudeSkillTool(tools, host.skillRegistry);
    }
    registerInteractionModeTool(tools, {
      async setMode({ context, mode }) {
        const activeRuntime = runtime;
        const session = activeRuntime
          ?.listSessions()
          .find((candidate) => candidate.id === context.sessionId);
        if (!activeRuntime || !session) {
          throw new Error("SetInteractionModeTool requires an active daemon session");
        }
        const changed = await activeRuntime.setSessionMode(
          session.sessionKey,
          mode,
        );
        if (!changed) {
          throw new Error("SetInteractionModeTool could not update the active daemon session");
        }
        return {
          mode,
          planMode: changed.planMode,
        };
      },
    });
    const agentDefinitions = loadAgentDefinitions({ cwd: workspaceRoot });
    const llm = createLlmClient(connection.model, {
      ...(connection.apiKey ? { api_key: connection.apiKey } : {}),
      ...(connection.baseUrl ? { base_url: connection.baseUrl } : {}),
      ...(connection.provider ? { provider: connection.provider } : {}),
      ...(connection.responsesApi ? { responsesApi: true } : {}),
    });
    const subagentOptions = {
      agentDefinitions,
      cwd: workspaceRoot,
      eventBus: subagentEvents,
      ...(host.skillRegistry?.markdownIndex()
        ? { extraContext: host.skillRegistry.markdownIndex() }
        : {}),
      llm,
      ...(connection.maxTokens === undefined
        ? {}
        : { maxTokens: connection.maxTokens }),
      model: connection.model,
      permissionMode: connection.permissionMode,
      ...(connection.temperature === undefined
        ? {}
        : { temperature: connection.temperature }),
      ...(connection.topK === undefined ? {} : { topK: connection.topK }),
      toolExecutor: tools,
      tools: tools.definitions(),
      ...(connection.topP === undefined ? {} : { topP: connection.topP }),
      transcriptStore,
    };
    if (subagentHost) {
      subagentHost.reconfigure(subagentOptions);
    } else {
      subagentHost = createNativeSubagentHost(subagentOptions);
    }
    registerClaudeAgentTools(tools, {
      backgroundAgents: subagentHost.turnCoordinator,
      manager: subagentHost.managerPort,
    });
    activeToolCount = tools.definitions().length;
    return new AgentTurnRunner({
      agentDefinitions,
      agentMemory: (session) => memoryForProject(
        session.metadata.session_kind === "subagent" &&
        typeof session.metadata.project_root === "string" &&
        session.metadata.project_root.trim()
          ? session.metadata.project_root
          : session.cwd,
      ),
      agentSelfMemory: (session) => getAgentSelfMemory(session.agentId),
      bootstrapSystemPrompt: ({ agentId, session, model, tools: runnerTools }) =>
        bootstrap({
          cwd: session.cwd,
          ...(host.skillRegistry?.markdownIndex()
            ? { extraContext: host.skillRegistry.markdownIndex() }
            : {}),
          model,
          subagents: bootstrapSubagentsForAgent(agentDefinitions, agentId),
          ...(runnerTools === undefined ? {} : { tools: runnerTools }),
        }).then((result) => result.systemPrompt),
      llm,
      ...(connection.maxTokens !== undefined
        ? { maxTokens: connection.maxTokens }
        : {}),
      model: connection.model,
      permissionMode: connection.permissionMode,
      subagentCoordinator: subagentHost.turnCoordinator,
      subagentEvents,
      ...(connection.temperature !== undefined
        ? { temperature: connection.temperature }
        : {}),
      ...(connection.topK !== undefined ? { topK: connection.topK } : {}),
      tools: tools.definitions(),
      toolExecutor: tools,
      ...(connection.topP !== undefined ? { topP: connection.topP } : {}),
      ...(interactions ? { interactions } : {}),
    });
  };
  runtime = new InMemoryDaemonRuntime(undefined, {
    ...(host.buildId ? { buildId: host.buildId } : {}),
    currentProjectDirectory: workspaceRoot,
    runtimeSettings: initialSettings,
    transcriptStore,
    statusInventory: () => ({
      activeSubagents:
        subagentHost?.manager
          .listTasks()
          .filter(
            (task) => task.status === "pending" || task.status === "running",
          ).length ?? 0,
      skills: host.skillRegistry?.all().length ?? 0,
      tools: activeToolCount,
    }),
    shutdown: () => subagentHost?.manager.shutdown(),
    onSessionEvict: sessionId => {
      subagentHost?.cancelSource(sessionId);
      memoryToolContext.prune(sessionId);
    },
    onSessionModeChange: (sessionId, mode) => {
      if (mode === "plan" || mode === "researcher") {
        subagentHost?.cancelSource(sessionId);
      }
    },
    turnRunnerFactory: runnerFactory,
    ...(interactions ? { interactions } : {}),
  });
  return runtime;
}

function websocketOptions(
  config: DaemonConfig,
): import("./daemon/websocketGateway.js").DaemonWebSocketGatewayOptions {
  const port = numericSetting(config.control.websocket_port, 11996);
  return {
    host: stringSetting(config.control.websocket_host) || "127.0.0.1",
    port,
    ...(stringSetting(config.control.auth_token)
      ? { authToken: stringSetting(config.control.auth_token) }
      : {}),
  };
}

function numericSetting(value: unknown, fallback: number): number {
  if (
    typeof value === "number" &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= 65_535
  )
    return value;
  if (typeof value === "string" && /^\d+$/.test(value)) {
    const parsed = Number.parseInt(value, 10);
    return parsed <= 65_535 ? parsed : fallback;
  }
  return fallback;
}

function stringSetting(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

/** The runner intercepts this schema and routes it through the daemon reply board. */
function registerDaemonQuestionTool(registry: ToolRegistry): void {
  registry.replace(
    {
      type: "function",
      function: {
        name: "AskUserQuestionTool",
        description:
          "Ask the connected user a blocking clarification question.",
        parameters: {
          type: "object",
          properties: {
            question: {
              type: "string",
              description: "Question shown to the user.",
            },
          },
          required: ["question"],
        },
      },
    },
    () => {
      throw new Error(
        "AskUserQuestionTool requires a daemon interaction board",
      );
    },
  );
}

async function acpServer(
  config: DaemonConfig,
  projectDirectory: string | undefined,
  defaultPermissionMode: AcpPermissionMode,
): Promise<{
  readonly server: AcpServer;
  readonly shutdown: () => Promise<void>;
}> {
  const connection = runtimeConnection(config, new ProfileStore().active());
  if (!connection) {
    throw new Error(
      "ACP requires a configured runtime connection or active provider profile",
    );
  }
  const workspaceRoot = projectDirectory ?? config.projectDirectory;
  const tools = new ToolRegistry();
  const skillRegistry = new SkillRegistry();
  await skillRegistry.refresh(...defaultSkillDiscoveryDirectories({ cwd: workspaceRoot }));
  const memoryToolContext = memoryToolContextResolver();
  registerCoreTools(tools, {
    workspaceRoot,
    agentMemoryTools: {
      memory: new AgentMemory({ projectRoot: workspaceRoot }),
      resolveSelfMemory: (context) =>
        getAgentSelfMemory(context.agentId ?? "default"),
    },
    memoryTools: { resolveContext: memoryToolContext.resolve },
  });
  registerClaudeSkillTool(tools, skillRegistry);
  const definitions = loadAgentDefinitions({ cwd: workspaceRoot });
  const agent = definitions.get("default");
  const agentId = agent?.name ?? "default";
  const selfMemory = getAgentSelfMemory(agentId);
  const model = agent?.model || connection.model;
  const llm = createLlmClient(connection.model, {
    ...(connection.apiKey ? { api_key: connection.apiKey } : {}),
    ...(connection.baseUrl ? { base_url: connection.baseUrl } : {}),
    ...(connection.provider ? { provider: connection.provider } : {}),
    ...(connection.responsesApi ? { responsesApi: true } : {}),
  });
  const subagentHost = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: workspaceRoot,
    eventBus: new DaemonSubagentEventBus(),
    ...(skillRegistry.markdownIndex() ? { extraContext: skillRegistry.markdownIndex() } : {}),
    llm,
    ...(connection.maxTokens === undefined
      ? {}
      : { maxTokens: connection.maxTokens }),
    model,
    permissionMode: defaultPermissionMode,
    ...(connection.temperature === undefined
      ? {}
      : { temperature: connection.temperature }),
    ...(connection.topK === undefined ? {} : { topK: connection.topK }),
    toolExecutor: tools,
    tools: tools.definitions(),
    ...(connection.topP === undefined ? {} : { topP: connection.topP }),
  });
  registerClaudeAgentTools(tools, {
    backgroundAgents: subagentHost.turnCoordinator,
    manager: subagentHost.managerPort,
  });
  const selectedTools = agentToolDefinitions(tools.definitions(), agent);
  const boot = await bootstrap({
    cwd: workspaceRoot,
    ...(skillRegistry.markdownIndex() ? { extraContext: skillRegistry.markdownIndex() } : {}),
    model,
    subagents: bootstrapSubagentsForAgent(definitions, agentId),
    tools: selectedTools,
  });
  const systemPrompt = joinSystemPrompts(
    boot.systemPrompt,
    agent?.systemPrompt,
    await selfMemory.systemPromptAddendum(),
  );
  const runner = new AcpAgentRunner({
    llm,
    model,
    agentId,
    ...(systemPrompt ? { systemPrompt } : {}),
    ...(connection.maxTokens !== undefined
      ? { maxTokens: connection.maxTokens }
      : {}),
    defaultPermissionMode,
    subagentCoordinator: subagentHost.turnCoordinator,
    ...(connection.temperature !== undefined
      ? { temperature: connection.temperature }
      : {}),
    ...(connection.topK !== undefined ? { topK: connection.topK } : {}),
    tools: selectedTools,
    toolExecutor: tools,
    ...(connection.topP !== undefined ? { topP: connection.topP } : {}),
  });
  return {
    server: new AcpServer({ runner, onSessionClose: sessionId => subagentHost.cancelSource(sessionId) }),
    shutdown: () => subagentHost.manager.shutdown(),
  };
}

async function runAcp(args: readonly string[]): Promise<void> {
  const options = parseAcpCommandOptions(args);
  if (options.help) {
    console.log(ACP_HELP);
    return;
  }
  if (options.writeRegistry) {
    const path = await writeAcpRegistryFile();
    console.log(`Wrote ACP registry manifest: ${path}`);
    return;
  }
  const config = loadSystemDaemonConfig({
    ...(options.projectDirectory
      ? { projectDirectory: options.projectDirectory }
      : {}),
  });
  const runtime = await acpServer(
    config,
    options.projectDirectory,
    options.permissionMode,
  );
  try {
    await serveACPStdio(runtime.server, Bun.stdin.stream(), (line) => {
      process.stdout.write(line);
    });
  } finally {
    await runtime.shutdown();
  }
}

async function runOneShot(prompt: string): Promise<void> {
  const config = loadSystemDaemonConfig();
  const connection = runtimeConnection(config, new ProfileStore().active());
  if (!connection) {
    throw new Error(
      "One-shot execution requires a configured runtime connection or active provider profile",
    );
  }
  const workspaceRoot = config.projectDirectory;
  const tools = new ToolRegistry();
  const skillRegistry = new SkillRegistry();
  await skillRegistry.refresh(...defaultSkillDiscoveryDirectories({ cwd: workspaceRoot }));
  const memoryToolContext = memoryToolContextResolver();
  const agentMemory = new AgentMemory({ projectRoot: workspaceRoot });
  registerCoreTools(tools, {
    workspaceRoot,
    agentMemoryTools: {
      memory: agentMemory,
      resolveSelfMemory: (context) =>
        getAgentSelfMemory(context.agentId ?? "default"),
    },
    memoryTools: { resolveContext: memoryToolContext.resolve },
  });
  registerClaudeSkillTool(tools, skillRegistry);
  const definitions = loadAgentDefinitions({ cwd: workspaceRoot });
  const agent = definitions.get("default");
  const selfMemory = getAgentSelfMemory(agent?.name ?? "default");
  const model = agent?.model || connection.model;
  const llm = createLlmClient(model, {
    ...(connection.apiKey ? { api_key: connection.apiKey } : {}),
    ...(connection.baseUrl ? { base_url: connection.baseUrl } : {}),
    ...(connection.provider ? { provider: connection.provider } : {}),
    ...(connection.responsesApi ? { responsesApi: true } : {}),
  });
  const subagentHost = createNativeSubagentHost({
    agentDefinitions: definitions,
    cwd: workspaceRoot,
    eventBus: new DaemonSubagentEventBus(),
    ...(skillRegistry.markdownIndex() ? { extraContext: skillRegistry.markdownIndex() } : {}),
    llm,
    ...(connection.maxTokens === undefined
      ? {}
      : { maxTokens: connection.maxTokens }),
    model,
    permissionMode: "accept-all",
    ...(connection.temperature === undefined
      ? {}
      : { temperature: connection.temperature }),
    ...(connection.topK === undefined ? {} : { topK: connection.topK }),
    toolExecutor: tools,
    tools: tools.definitions(),
    ...(connection.topP === undefined ? {} : { topP: connection.topP }),
  });
  registerClaudeAgentTools(tools, {
    backgroundAgents: subagentHost.turnCoordinator,
    manager: subagentHost.managerPort,
  });
  const selectedTools = agentToolDefinitions(tools.definitions(), agent);
  const boot = await bootstrap({
    cwd: workspaceRoot,
    ...(skillRegistry.markdownIndex() ? { extraContext: skillRegistry.markdownIndex() } : {}),
    model,
    subagents: bootstrapSubagentsForAgent(definitions, agent?.name ?? "default"),
    tools: selectedTools,
  });
  const systemPrompt = joinSystemPrompts(
    boot.systemPrompt,
    agent?.systemPrompt,
    await agentMemory.toPromptSection(),
    await selfMemory.systemPromptAddendum(),
  );
  const sessionId = `oneshot-${crypto.randomUUID()}`;
  const state = createAgentState();
  const subagentCohort = subagentHost.turnCoordinator.begin(sessionId);
  let wroteText = false;
  try {
    for await (const event of runTurn(
      {
        model,
        state,
        userMessage: prompt,
        permissionMode: "accept-all",
        sessionId,
        ...(agent?.name ? { agentId: agent.name } : {}),
        ...(systemPrompt ? { systemPrompt } : {}),
        ...(connection.maxTokens !== undefined
          ? { maxTokens: connection.maxTokens }
          : {}),
        ...(connection.temperature !== undefined
          ? { temperature: connection.temperature }
          : {}),
        ...(connection.topK !== undefined ? { topK: connection.topK } : {}),
        tools: selectedTools,
        ...(connection.topP !== undefined ? { topP: connection.topP } : {}),
      },
      {
        awaitAgentEvents: async (signal) =>
          formatSubagentResults(await subagentCohort.waitForResults(signal)),
        llm,
        toolExecutor: tools,
      },
    )) {
      if (event.type === "text") {
        wroteText = true;
        process.stdout.write(event.text);
      } else if (event.type === "provider_retry" && event.final) {
        console.error(`Provider error: ${event.error}`);
      }
    }
  } finally {
    subagentCohort.close();
    await subagentHost.manager.shutdown();
  }
  if (wroteText) process.stdout.write("\n");
}

/**
 * Submit a non-interactive turn against an explicitly persisted daemon session.
 *
 * This intentionally creates no interaction board: approvals are set to accept-all,
 * and question tools are not advertised, so a piped CLI invocation can never wait
 * for a TUI/daemon client to answer it.
 */
async function runResumedOneShot(
  sessionId: string,
  prompt: string,
): Promise<void> {
  const projectDirectory = resolve(process.cwd());
  const config = loadSystemDaemonConfig({ projectDirectory });
  const runtime = daemonRuntime(config, projectDirectory, new ProfileStore());
  let wroteText = false;
  try {
    runtime.reload({ permission_mode: "accept-all" });
    const session = await runtime.openSession(sessionId, undefined, {
      cwd: projectDirectory,
      resume: true,
    });
    await runtime.submitTurn(session.sessionKey, prompt, (event) => {
      if (event.type === "text_part") {
        // Provider deltas are byte-for-byte text fragments. Trimming each one
        // removes meaningful leading spaces and joins adjacent streamed words.
        const text =
          typeof event.payload.text === "string" ? event.payload.text : "";
        if (text) {
          wroteText = true;
          process.stdout.write(text);
        }
        return;
      }
      if (event.type === "notification" && event.payload.level === "error") {
        const message = stringSetting(event.payload.message);
        if (message) {
          console.error(`Provider error: ${message}`);
        }
      }
    });
  } finally {
    await runtime.shutdown();
  }
  if (wroteText) process.stdout.write("\n");
}

function agentToolDefinitions(
  definitions: readonly import("./types/toolCalls.js").ToolDefinition[],
  agent: import("./agents/definitions.js").AgentDefinition | undefined,
): readonly import("./types/toolCalls.js").ToolDefinition[] {
  if (!agent) return definitions;
  const listed = new Set(agent.tools);
  const allowed =
    agent.allowedTools === null ? undefined : new Set(agent.allowedTools);
  const excluded = new Set(agent.excludeTools);
  return definitions.filter((definition) => {
    const name = definition.function.name;
    return (
      !excluded.has(name) &&
      (!allowed || allowed.has(name)) &&
      (listed.size === 0 || listed.has(name))
    );
  });
}

function joinSystemPrompts(
  ...sections: Array<string | undefined>
): string | undefined {
  const prompt = sections
    .filter((section): section is string => Boolean(section?.trim()))
    .join("\n\n");
  return prompt || undefined;
}

function memoryToolContextResolver(): {
  readonly prune: (sessionId: string) => void;
  readonly resolve: (context: ToolExecutionContext) => MemoryToolContext;
} {
  const memories = new Map<string, ContextualMemory>();
  return {
    prune(sessionId) {
      const prefix = `${sessionId}:`;
      for (const key of memories.keys()) {
        if (key.startsWith(prefix)) memories.delete(key);
      }
    },
    resolve(context) {
      const agentId = context.agentId ?? "default";
      const key = (context.sessionId ?? "sessionless") + ":" + agentId;
      let memory = memories.get(key);
      if (!memory) {
        memory = new ContextualMemory();
        memories.set(key, memory);
      }
      return { agentId, memory };
    },
  };
}
