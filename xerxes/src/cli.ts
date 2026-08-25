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
import { loadAgentDefinitions, resolveAgentDefinition } from "./agents/definitions.js";
import { ConfigurationError } from "./core/errors.js";
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
import { compactionCompletionPort } from "./daemon/server.js";
import { DaemonSubagentEventBus } from "./daemon/subagentEvents.js";
import { createNativeSubagentHost, subagentRetryWirePayload } from "./daemon/subagentHost.js";
import { AgentTurnRunner, formatSubagentResults } from "./daemon/turnRunner.js";
import {
  defaultSkillDiscoveryRoots,
  SkillRegistry,
  trustedHashWorkspaceSkills,
} from "./extensions/skills.js";
import {
  ToolRegistry,
  type ToolExecutionContext,
} from "./executors/toolRegistry.js";
import { createCompactionAgent } from "./agents/compactionAgent.js";
import { AuditEmitter } from "./audit/emitter.js";
import { JSONLSinkCollector } from "./audit/collector.js";
import { estimateContextTokens } from "./context/windowUsage.js";
import { getContextLimit } from "./llms/providerRegistry.js";
import { createLlmClient } from "./llms/client.js";
import type { ChatMessage } from "./types/messages.js";
import type { SpawnedAgentSnapshot } from "./operators/subagents.js";
import { AgentMemory } from "./memory/agentMemory.js";
import { getAgentSelfMemory } from "./memory/agentSelfMemory.js";
import { ContextualMemory } from "./memory/contextualMemory.js";
import {
  BrowserManager,
  registerBrowserManagerTools,
} from "./operators/browser.js";
import {
  INSTALL_HELP,
  InstallCommandError,
  runInstallCommand,
} from "./runtime/companionInstall.js";
import { bootstrap, bootstrapSubagentsForAgent } from "./runtime/bootstrap.js";
import {
  hasDoctorFailures,
  printDoctorReport,
  runAllDoctorChecks,
} from "./runtime/doctor.js";
import { CliWriter, createCliStyle, detectColorDepth } from "./runtime/cliStyle.js";
import { resolveTuiEntry } from "./runtime/distribution.js";
import { registerInteractionModeTool } from "./runtime/interactionModeTool.js";
import { extractAgentOption, parseValueOptions } from "./runtime/commandOptions.js";
import { ProcessRegistry } from "./runtime/processRegistry.js";
import { TerminalRegistry } from "./runtime/terminalRegistry.js";
import { BackgroundCommandManager } from "./tools/backgroundCommands.js";
import { DaemonTranscriptStore } from "./session/daemonTranscript.js";
import {
  UPDATE_HELP,
  UpdateCommandError,
  runUpdateCommand,
} from "./runtime/update.js";
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
import { createMacOSComputerUseToolOptions } from "./tools/computerUse/macosPort.js";
import {
  createLlmPlanGenerator,
  registerClaudeAgentTools,
  registerClaudeSkillTool,
  registerClaudeWorkflowTools,
  registerCoreTools,
} from "./tools/index.js";
import type { MemoryToolContext } from "./tools/memoryTools.js";
import { createAgentState } from "./streaming/events.js";
import { runTurn } from "./streaming/loop.js";
import { runBundledSkillCli } from "./skills/cli.js";
import { AuthCommandError, runAuthCommand } from "./auth/command.js";

/**
 * Command list for `--help`, grouped by what the reader is trying to do.
 *
 * Kept as data rather than a formatted blob so the renderer can align the
 * description column and colour the invocations; a single template string could
 * do neither, which is why the old help was a flat wall of usage lines.
 */
const HELP_GROUPS: readonly {
  readonly commands: readonly (readonly [invocation: string, description: string])[];
  readonly title: string;
}[] = [
  {
    title: "Ask",
    commands: [
      ["xerxes", "open the interactive terminal interface"],
      ["xerxes [prompt]", "run one turn and exit"],
      ["xerxes --agent <name|path> [prompt]", "run one turn as a named or file-defined agent"],
      ["xerxes --resume <session_id> [prompt]", "continue an earlier session"],
    ],
  },
  {
    title: "Serve",
    commands: [
      ["xerxes daemon", "run the local project daemon"],
      ["xerxes acp", "speak the Agent Client Protocol over stdio"],
      ["xerxes telegram --token <token>", "run the Telegram channel gateway"],
    ],
  },
  {
    title: "Maintain",
    commands: [
      ["xerxes auth login codex", "sign in to ChatGPT and use its Codex plan"],
      ["xerxes doctor", "check the host, providers, and configuration"],
      ["xerxes update [--check] [--git] [--dry-run] [--apply]", "report or apply an update"],
      ["xerxes install --cloud-code [--force] [--dry-run]", "install a companion integration"],
      ["xerxes export [session]", "write a session transcript to disk"],
      ["xerxes skill <skill> [arguments]", "run a bundled skill"],
    ],
  },
];

/** Render `--help` with aligned descriptions and coloured invocations. */
function renderHelp(writer: CliWriter): void {
  writer.heading(`Xerxes ${version}`);
  writer.hint("Bun-native coding agent and multi-agent runtime.");
  const widest = Math.max(
    ...HELP_GROUPS.flatMap((group) => group.commands.map(([invocation]) => invocation.length)),
  );
  for (const group of HELP_GROUPS) {
    writer.line();
    writer.line(writer.style.bold(group.title));
    for (const [invocation, description] of group.commands) {
      // Pad before colouring: escape sequences have no width, so padding a
      // styled string would align by byte count and leave the column ragged.
      writer.line(`  ${writer.command(invocation.padEnd(widest))}  ${writer.style.dim(description)}`);
    }
  }
  writer.line();
  writer.line(writer.style.bold("Also"));
  writer.line(`  ${writer.command("--help".padEnd(widest))}  ${writer.style.dim("show this message")}`);
  writer.line(`  ${writer.command("-v, --version".padEnd(widest))}  ${writer.style.dim("print the version and exit")}`);
  writer.line(`  ${writer.command("--".padEnd(widest))}  ${writer.style.dim("send everything after this marker verbatim as the prompt")}`);
  writer.line();
  writer.hint(
    "Browser tools attach only to an explicitly supplied Chromium CDP endpoint; use /browser in the TUI\n"
      + "or the daemon browser command to connect.",
  );
}

/** Summary budget for a mid-turn overflow rescue: small, because the window is already full. */
const OVERFLOW_SUMMARY_MAX_TOKENS = 2_048;

const { agent: cliAgentReference, rest: cliArguments } = extractAgentOption(
  Bun.argv.slice(2),
);
const [argument, ...argumentsAfterCommand] = cliArguments;

/**
 * Commands that own their runtime (or never start a turn) cannot adopt a
 * one-shot agent; accepting the flag there would silently ignore it.
 */
const NON_ONESHOT_COMMANDS: ReadonlySet<string> = new Set([
  "skill",
  "auth",
  "doctor",
  "install",
  "update",
  "export",
  "daemon",
  "telegram",
  "acp",
  "-r",
  "--resume",
]);
if (
  cliAgentReference !== undefined &&
  argument !== undefined &&
  NON_ONESHOT_COMMANDS.has(argument)
) {
  throw new Error(
    `The --agent option is only supported for one-shot prompts, not '${argument}'`,
  );
}

/**
 * A bare token that looks like a flag (`-x`, `--model`) rather than prompt text.
 *
 * Negative numbers and dash-led words without a letter (e.g. `-42`, `---`) stay
 * prompt-eligible; only single- or double-dash letter-initial tokens read as
 * flags the dispatcher never recognized.
 */
function isFlagLikeToken(word: string): boolean {
  return /^-{1,2}[A-Za-z]/.test(word);
}

/**
 * Join free-form prompt words into one prompt, rejecting unrecognized flags.
 *
 * Dash-led tokens before any `--` separator are treated as mistyped flags and
 * rendered through the standard usage-error reporter instead of being sent to
 * the provider as prompt text; everything after the separator joins verbatim.
 *
 * `command` is the invocation prefix shown in the escape-hatch hint. For the
 * bare one-shot form the bun launcher itself consumes one leading `--`, so the
 * leading-flag hint there tells writers to repeat the separator; every other
 * form (e.g. after `--resume <id>`) keeps its first argument slot occupied and
 * therefore passes a mid-position separator straight through.
 */
function parsePromptArguments(words: readonly string[], command: string): string {
  const separator = words.indexOf("--");
  const flagged = (separator === -1 ? words : words.slice(0, separator))
    .find(isFlagLikeToken);
  if (flagged !== undefined) {
    const doubledNote = command === "xerxes" && flagged === words[0]
      ? " — write the separator twice, because bun consumes the first one"
      : "";
    reportCommandUsageError(
      new Error(
        `Unknown option '${flagged}'. To send dash-led text as a prompt, put it after '--'${doubledNote}: ${command} -- ${flagged}`,
      ),
      "xerxes --help",
    );
  }
  return (
    separator === -1 ? [...words] : [...words.slice(0, separator), ...words.slice(separator + 1)]
  ).join(" ").trim();
}

/**
 * A turn cannot start because no provider connection is configured.
 *
 * A missing profile is a setup problem, not a crash, so callers render this
 * through {@link reportCommandUsageError} rather than letting it surface as an
 * unhandled stack trace.
 */
class RuntimeConnectionRequiredError extends Error {}

/** Render a typed command error as two clean stderr lines and exit; never dump a stack. */
function reportCommandUsageError(error: Error, helpCommand: string): never {
  // Errors go to stderr, so the styler is built against stderr's TTY state
  // rather than stdout's: `xerxes update > log` should still colour the error a
  // human is watching, and `2>&1 | cat` should still be plain.
  const errorWriter = new CliWriter({
    style: createCliStyle(detectColorDepth({ isTTY: Boolean(process.stderr.isTTY) })),
    write: (line) => console.error(line),
  });
  // The literal "error" label is kept alongside the glyph: it is the
  // conventional, greppable prefix, and a glyph alone would be invisible to
  // anything filtering a log.
  errorWriter.status("fail", "error", error.message);
  errorWriter.hint(`run '${helpCommand}' for usage.`);
  process.exit(1);
}

if (argument === "--help" || argument === "-h") {
  renderHelp(new CliWriter());
  process.exit(0);
} else if (argument === "--version" || argument === "-v" || argument === "-V") {
  console.log(version);
  process.exit(0);
} else if (argument === "skill") {
  process.exit(await runBundledSkillCli(argumentsAfterCommand));
} else if (argument === "auth") {
  try {
    process.exit(await runAuthCommand(argumentsAfterCommand));
  } catch (error) {
    if (error instanceof AuthCommandError) {
      reportCommandUsageError(error, "xerxes auth --help");
    }
    throw error;
  }
} else if (argument === "doctor") {
  const report = runAllDoctorChecks();
  printDoctorReport(report);
  process.exit(hasDoctorFailures(report) ? 1 : 0);
} else if (argument === "install") {
  if (
    argumentsAfterCommand.includes("--help") ||
    argumentsAfterCommand.includes("-h")
  ) {
    console.log(INSTALL_HELP);
  } else {
    try {
      await runInstallCommand(argumentsAfterCommand);
    } catch (error) {
      if (error instanceof InstallCommandError) {
        reportCommandUsageError(error, "xerxes install --help");
      }
      throw error;
    }
  }
} else if (argument === "update") {
  if (
    argumentsAfterCommand.includes("--help") ||
    argumentsAfterCommand.includes("-h")
  ) {
    console.log(UPDATE_HELP);
  } else {
    try {
      await runUpdateCommand(argumentsAfterCommand);
    } catch (error) {
      if (error instanceof UpdateCommandError) {
        reportCommandUsageError(error, "xerxes update --help");
      }
      throw error;
    }
  }
} else if (argument === "export") {
  await runExport(argumentsAfterCommand);
} else if (argument === "daemon") {
  const options = parseValueOptions(argumentsAfterCommand, "daemon", [
    "--pid-file",
    "--project-dir",
    "--socket",
  ]);
  const projectDirectory = options.get("--project-dir");
  const config = loadSystemDaemonConfig({
    ...(projectDirectory ? { projectDirectory } : {}),
  });
  const socketPath =
    options.get("--socket") ?? daemonPaths(projectDirectory).socketPath;
  const pidPath = options.get("--pid-file");
  await runDaemon(config, projectDirectory, socketPath, pidPath);
  process.exit(0);
} else if (argument === "telegram") {
  const options = parseValueOptions(argumentsAfterCommand, "telegram", [
    "--host",
    "--pid-file",
    "--port",
    "--project-dir",
    "--socket",
    "--token",
  ]);
  const token =
    options.get("--token") ?? process.env.TELEGRAM_BOT_TOKEN?.trim();
  if (!token)
    throw new Error("telegram requires --token or TELEGRAM_BOT_TOKEN");
  const projectDirectory = options.get("--project-dir");
  const config = telegramDaemonConfig(
    loadSystemDaemonConfig({
      ...(projectDirectory ? { projectDirectory } : {}),
    }),
    token,
    options.get("--host"),
    options.get("--port"),
  );
  const socketPath =
    options.get("--socket") ?? daemonPaths(projectDirectory).socketPath;
  const pidPath = options.get("--pid-file");
  await runDaemon(config, projectDirectory, socketPath, pidPath);
  process.exit(0);
} else if (argument === "acp") {
  await runAcp(argumentsAfterCommand);
} else if (argument === "-r" || argument === "--resume") {
  const sessionId = argumentsAfterCommand[0]?.trim();
  if (!sessionId || sessionId.startsWith("-")) {
    throw new Error(
      sessionId
        ? `The --resume option requires a session ID, not the flag ${sessionId}`
        : "The --resume option requires a session ID",
    );
  }
  // Same contract as the one-shot catch-all: dash-led words here are
  // unrecognized flags, not prompt text — `--resume abc --model gpt-4 hi`
  // must not silently resume with '--model gpt-4 hi' as the turn's prompt.
  // Unlike that form, this one keeps '--resume' in argv's first slot, so a
  // single separator survives and needs no doubling.
  const prompt = parsePromptArguments(
    argumentsAfterCommand.slice(1),
    "xerxes --resume <session_id>",
  );
  if (prompt) {
    await runResumedOneShot(sessionId, prompt);
  } else {
    await runTui(sessionId);
  }
} else if (argument === undefined) {
  const prompt = process.stdin.isTTY ? "" : await readStandardInput();
  if (prompt) {
    await runOneShotOrUsageError(prompt, cliAgentReference);
  } else if (cliAgentReference !== undefined) {
    throw new Error(
      "The --agent option requires a prompt: xerxes --agent <name|path> <prompt>",
    );
  } else if (process.stdin.isTTY) {
    await runTui();
  } else {
    throw new Error("No prompt was provided on standard input");
  }
} else {
  // The catch-all owns free-form prompts, so a dash-prefixed token here is
  // almost certainly a mistyped or unsupported flag — sending it to the
  // provider as prompt text used to silently produce an answer to nobody.
  const prompt = parsePromptArguments([argument, ...argumentsAfterCommand], "xerxes");
  if (!prompt) {
    throw new Error(
      "No prompt was provided. Put prompt text after '--' when it begins with a dash",
    );
  }
  await runOneShotOrUsageError(prompt, cliAgentReference);
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
  const skillRegistry = new SkillRegistry({ workspaceTrust: trustedHashWorkspaceSkills() });
  // Shared by the tool registry that starts the processes and the RPC surface
  // that lists them. One instance is the whole point: a second registry would
  // be a second, permanently empty view of the same shells.
  const terminals = new TerminalRegistry();
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
    { ...(buildId ? { buildId } : {}), skillRegistry, terminals },
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
    terminalRegistry: terminals,
    profileStore,
    skillRegistry,
    onRestart: finish,
    onShutdown: finish,
    // Only a process-owning host may claim uncaughtException/unhandledRejection,
    // which is why the server leaves them off by default. This IS that host, and
    // without them a crash loses the whole in-flight turn.
    crashHandlers: true,
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
  // `once` dropped every signal after the first, so an operator whose first
  // SIGTERM caught a turn that would not settle had no second chance short of
  // SIGKILL. `finish` is already idempotent; the hard exit lives in stop().
  process.on("SIGINT", finish);
  process.on("SIGTERM", finish);
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
    readonly terminals?: TerminalRegistry;
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
  // Off by default: an always-on JSONL sink is a surprise disk writer. Every
  // downstream call is optional-chained, so enabling it is purely additive —
  // but until something constructs one the daemon emits no audit record at all.
  // The sink buffers writes, so the runtime `shutdown` hook below must close it:
  // without that barrier a fast exit could drop queued records (policy denials
  // included) after the turn already reported them.
  const auditEmitter = process.env.XERXES_AUDIT?.trim()
    ? new AuditEmitter({
      collector: new JSONLSinkCollector(join(xerxesHome(), "audit", "events.jsonl")),
    })
    : undefined;
  const subagentEvents = new DaemonSubagentEventBus();
  // Built once, outside the runner factory. Every settings change rebuilds the
  // registry, and a per-registry manager took every running background process
  // with it — the `proc_id` the model was polling stopped resolving mid-build.
  const backgroundCommands = new BackgroundCommandManager(
    new ProcessRegistry(),
    host.terminals,
  );
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
    const computerUseTool = createMacOSComputerUseToolOptions({
      ...config.runtime,
      ...settings,
    });
    registerCoreTools(tools, {
      workspaceRoot,
      backgroundCommands,
      ...(host.terminals === undefined ? {} : { terminals: host.terminals }),
      ...(computerUseTool === undefined ? {} : { computerUseTool }),
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
    // Deterministic multi-agent orchestration: PlanTool decomposes an explicit
    // objective into dependency-ordered steps and executes them through the
    // same managed subagent pool (depth caps, cohort join, persistence).
    registerClaudeWorkflowTools(tools, {
      ...(agentDefinitions.size ? { agentDefinitions: [...agentDefinitions.values()] } : {}),
      ...(host.skillRegistry ? { skillRegistry: host.skillRegistry } : {}),
      planGenerator: createLlmPlanGenerator(llm, { model: connection.model }),
      subagentManager: subagentHost.managerPort,
    });
    activeToolCount = tools.definitions().length;
    return new AgentTurnRunner({
      // The profile's provider, carried explicitly so nothing downstream has
      // to infer it from the model id. An OpenRouter id like
      // `stealth/ox-alpha` has a vendor before the slash, not a routing
      // prefix, and inferring one threw on every turn.
      providerOverrides: {
        ...(connection.provider ? { provider: connection.provider } : {}),
        ...(connection.baseUrl ? { base_url: connection.baseUrl } : {}),
      },
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
      ...(connection.reasoningEffort === undefined
        ? {}
        : { reasoningEffort: connection.reasoningEffort }),
      subagentCoordinator: subagentHost.turnCoordinator,
      subagentEvents,
      ...(connection.temperature !== undefined
        ? { temperature: connection.temperature }
        : {}),
      ...(connection.thinking === undefined ? {} : { thinking: connection.thinking }),
      ...(connection.thinkingBudget === undefined
        ? {}
        : { thinkingBudget: connection.thinkingBudget }),
      ...(connection.topK !== undefined ? { topK: connection.topK } : {}),
      tools: tools.definitions(),
      toolExecutor: tools,
      // Per-tool usage-policy sections ride with the visible tool surface.
      toolRegistry: tools,
      // The call's arguments ride along so invocation-scoped refinement can
      // widen one axis (a read-only exec_command may run concurrently) without
      // ever loosening the permissioned axes.
      toolCapabilities: (name, agentId, args) => tools.capabilities(name, agentId, args),
      // Spill oversized tool results outside the user's repo. The bootstrap
      // prompt has always claimed this happens; supplying the root is what
      // makes the claim true.
      toolResultDirectory: join(xerxesHome(), "tool-results"),
      // The daemon is the one host where a per-edit typecheck earns its cost:
      // it turns "the change compiles" from a claim into a reported fact.
      editDiagnostics: process.env.XERXES_EDIT_DIAGNOSTICS?.trim() !== "0",
      ...(auditEmitter ? { auditEmitter } : {}),
      // Compaction as a mid-turn recovery, not only a between-turn chore: the
      // loop can retry an overflowed round once, but only if something is
      // willing to shrink the history for it.
      reduceContext: async (messages) => {
        const priced = messages as unknown as Readonly<Record<string, unknown>>[];
        const before = estimateContextTokens(priced, { model: connection.model });
        const agent = createCompactionAgent({
          model: connection.model,
          completion: compactionCompletionPort(llm, connection.model),
          summaryMaxTokens: OVERFLOW_SUMMARY_MAX_TOKENS,
        });
        // `ContextMessage` is deliberately `Record<string, unknown>` so
        // compaction survives provider-specific fields it does not model.
        // It only drops or replaces whole messages, so the typed shape the
        // loop handed in is preserved through the round trip.
        const reduced = await agent.summarizeMessages(priced) as unknown as ChatMessage[];
        return {
          messages: reduced,
          tokensFreed: Math.max(
            0,
            before - estimateContextTokens(
              reduced as unknown as Readonly<Record<string, unknown>>[],
              { model: connection.model },
            ),
          ),
        };
      },
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
    backgroundCommands,
    shutdown: async () => {
      // Children first: their final events still belong in this session's
      // audit log, so the audit sink is only closed after they settle.
      await subagentHost?.manager.shutdown();
      // Durability barrier for queued audit records: holds process shutdown
      // (daemon stop, SIGINT/SIGTERM finish, resumed one-shot finally) until
      // every buffered record reached the JSONL sink.
      await auditEmitter?.close();
    },
    onSessionEvict: sessionId => {
      subagentHost?.cancelSource(sessionId);
      memoryToolContext.prune(sessionId);
    },
    // Esc/Ctrl+C stops the whole delegation tree the turn started, not just
    // the parent's provider stream. Children stay retryable because the user
    // asked to pause, not to discard the work.
    onTurnCancel: sessionId => subagentHost?.interruptSource(sessionId) ?? 0,
    // First-class retry of a dead subagent under its stable identity
    // (`subagent.retry` RPC, `/agents retry`, agents-panel `r` key). The host
    // continues the persisted conversation when one survives; without an
    // active provider connection there is no runner to resume with.
    subagentRetry: async ({ task, message }) => {
      const host = subagentHost;
      if (!host) {
        return {
          ok: false,
          error:
            "subagent retry requires an active provider connection; configure a profile and try again",
        };
      }
      try {
        const snapshot = await host.retry(task, message ? { message } : {});
        return { ok: true, agent: subagentRetryWirePayload(snapshot) };
      } catch (error) {
        return { ok: false, error: errorMessage(error) };
      }
    },
    // An interaction-mode change (set_mode / set_plan_mode /
    // SetInteractionModeTool) must never cancel this session's running
    // subagents: mode only re-scopes the parent turn's next tool surface,
    // while delegated children keep the permission ceiling they were
    // spawned with. Children are reclaimed only when their owning session
    // is evicted (above) or the daemon shuts down.
    turnRunnerFactory: runnerFactory,
    ...(interactions ? { interactions } : {}),
  });
  return runtime;
}

function websocketOptions(
  config: DaemonConfig,
): import("./daemon/websocketGateway.js").DaemonWebSocketGatewayOptions {
  const port = websocketPortSetting(config.control.websocket_port);
  return {
    host: stringSetting(config.control.websocket_host) || "127.0.0.1",
    port,
    ...(stringSetting(config.control.auth_token)
      ? { authToken: stringSetting(config.control.auth_token) }
      : {}),
  };
}

/** Port served when configuration stays silent about the WebSocket control channel. */
const DEFAULT_WEBSOCKET_PORT = 11_996;

/**
 * Resolve the WebSocket control-channel port.
 *
 * Absent settings fall back to {@link DEFAULT_WEBSOCKET_PORT}; an explicit but
 * invalid value is a configuration error rather than a silent fallback onto the
 * default port, which would strand every client that was told a different port
 * and make the misconfiguration undiagnosable. Digit strings stay accepted so
 * historically valid string configs keep working; their values are range-checked
 * like numbers. Wording mirrors the core config's integer field parser.
 */
function websocketPortSetting(value: unknown): number {
  if (value === undefined) return DEFAULT_WEBSOCKET_PORT;
  const parsed = typeof value === "number"
    ? value
    : typeof value === "string" && /^\d+$/.test(value)
      ? Number.parseInt(value, 10)
      : Number.NaN;
  if (!Number.isInteger(parsed)) {
    throw new ConfigurationError("control.websocket_port", "must be a finite integer");
  }
  if (parsed < 0 || parsed > 65_535) {
    throw new ConfigurationError("control.websocket_port", "must be between 0 and 65535");
  }
  return parsed;
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
  const skillRegistry = new SkillRegistry({ workspaceTrust: trustedHashWorkspaceSkills() });
  await skillRegistry.refresh(...defaultSkillDiscoveryRoots({ cwd: workspaceRoot }));
  const memoryToolContext = memoryToolContextResolver();
  const acpComputerUseTool = createMacOSComputerUseToolOptions(config.runtime);
  registerCoreTools(tools, {
    workspaceRoot,
    ...(acpComputerUseTool === undefined ? {} : { computerUseTool: acpComputerUseTool }),
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

/**
 * Run a one-shot turn, rendering a missing provider configuration as a clean
 * usage error.
 *
 * Without this, `xerxes "hi"` with no configured profile died with an unhandled
 * stack trace; that is setup guidance, not a crash report. Every other failure
 * still propagates untouched.
 */
async function runOneShotOrUsageError(
  prompt: string,
  agentReference?: string,
): Promise<void> {
  try {
    await runOneShot(prompt, agentReference);
  } catch (error) {
    if (error instanceof RuntimeConnectionRequiredError) {
      reportCommandUsageError(error, "xerxes --help");
    }
    throw error;
  }
}

async function runOneShot(
  prompt: string,
  agentReference?: string,
): Promise<void> {
  const config = loadSystemDaemonConfig();
  const connection = runtimeConnection(config, new ProfileStore().active());
  if (!connection) {
    throw new RuntimeConnectionRequiredError(
      "One-shot execution requires a configured runtime connection or active provider profile",
    );
  }
  const workspaceRoot = config.projectDirectory;
  const tools = new ToolRegistry();
  const skillRegistry = new SkillRegistry({ workspaceTrust: trustedHashWorkspaceSkills() });
  await skillRegistry.refresh(...defaultSkillDiscoveryRoots({ cwd: workspaceRoot }));
  const memoryToolContext = memoryToolContextResolver();
  const agentMemory = new AgentMemory({ projectRoot: workspaceRoot });
  const computerUseTool = createMacOSComputerUseToolOptions(config.runtime);
  registerCoreTools(tools, {
    workspaceRoot,
    ...(computerUseTool === undefined ? {} : { computerUseTool }),
    agentMemoryTools: {
      memory: agentMemory,
      resolveSelfMemory: (context) =>
        getAgentSelfMemory(context.agentId ?? "default"),
    },
    memoryTools: { resolveContext: memoryToolContext.resolve },
  });
  registerClaudeSkillTool(tools, skillRegistry);
  const definitions = loadAgentDefinitions({ cwd: workspaceRoot });
  // An explicit --agent reference swaps the session's persona, tool surface,
  // and model for the named catalog entry or the referenced YAML/Markdown file;
  // without one the catalog's "default" agent keeps its historical role.
  const agent =
    agentReference === undefined
      ? definitions.get("default")
      : resolveAgentDefinition(agentReference, {
          builtinDefinitions: definitions,
          cwd: workspaceRoot,
        });
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
  let pendingAgentEventSnapshots: readonly SpawnedAgentSnapshot[] = [];
  let wroteText = false;
  let terminalProviderError: string | undefined;
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
        awaitAgentEvents: async (signal) => {
          pendingAgentEventSnapshots = await subagentCohort.waitForResults(signal);
          return formatSubagentResults(pendingAgentEventSnapshots);
        },
        acknowledgeAgentEvents: () => {
          if (!pendingAgentEventSnapshots.length) return;
          subagentHost.turnCoordinator.consume(pendingAgentEventSnapshots);
          pendingAgentEventSnapshots = [];
        },
        llm,
        toolExecutor: tools,
      },
    )) {
      if (event.type === "text") {
        wroteText = true;
        process.stdout.write(event.text);
      } else if (event.type === "provider_retry" && event.final) {
        terminalProviderError = event.error;
        console.error(`Provider error: ${event.error}`);
      }
    }
  } finally {
    subagentCohort.close();
    await subagentHost.manager.shutdown();
  }
  if (wroteText) process.stdout.write("\n");
  // A terminal provider failure still yields a text event, so scripts and CI
  // must learn about it through the exit code rather than stdout alone.
  if (terminalProviderError !== undefined) process.exitCode = 1;
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
  let turnFailed = false;
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
        turnFailed = true;
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
  // Turn failures surface as error-level notification events only; without a
  // failing exit code a broken resumed run looks successful to scripts.
  if (turnFailed) process.exitCode = 1;
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
