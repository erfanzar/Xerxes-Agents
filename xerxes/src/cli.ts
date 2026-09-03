// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { version } from "../package.json" with { type: "json" };
import { mkdir, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
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
import { AgentPresetRoster } from "./agents/presets.js";
import { ConfigurationError } from "./core/errors.js";
import {
  BunDiscordGatewayWebSocketPort,
  FetchDiscordGatewayRestPort,
} from "./channels/discordGateway.js";
import { FetchDiscordApplicationRestPort } from "./channels/discordApplications.js";
import {
  resolvedProfileContextLimit,
  resolvedProfileMaxOutputTokens,
  ProfileStore,
} from "./bridge/profiles.js";
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
import { DeclarativeToolForge } from "./extensions/declarativeForge.js";
import { HookRunner } from "./extensions/hooks.js";
import { loadShellHookConfigSync, registerShellHooks } from "./extensions/shellHooks.js";
import {
  ToolRegistry,
  type ToolExecutionContext,
} from "./executors/toolRegistry.js";
import { createCompactionAgent } from "./agents/compactionAgent.js";
import { AuditEmitter } from "./audit/emitter.js";
import { JSONLSinkCollector } from "./audit/collector.js";
import { estimateContextTokens } from "./context/windowUsage.js";
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
import { goalPolicyPrompt, registerGoalTools } from "./runtime/goalTools.js";
import { extractAgentOption, extractOutputFormatOption, parseValueOptions, type OutputFormat } from "./runtime/commandOptions.js";
import { ProcessRegistry } from "./runtime/processRegistry.js";
import { TerminalRegistry } from "./runtime/terminalRegistry.js";
import { PtySessionManager } from "./operators/pty.js";
import { BackgroundCommandManager } from "./tools/backgroundCommands.js";
import { DaemonTranscriptStore } from "./session/daemonTranscript.js";
import {
  UPDATE_HELP,
  UpdateCommandError,
  runUpdateCommand,
} from "./runtime/update.js";
import { runSetupCommand } from "./runtime/setupCommand.js";
import { runWorkspaceCommand } from "./runtime/workspaceCommand.js";
import { runScheduleCommand } from "./runtime/scheduleCommand.js";
import { runMemoryCommand, type MemoryCommandOptions } from "./runtime/memoryCommand.js";
import { runCapabilityCommand } from "./runtime/capabilityCommand.js";
import { runTelemetryCommand, type TelemetryCommandOptions } from "./runtime/telemetryCommand.js";
import { runStatusCommand } from "./runtime/statusCommand.js";
import { runChannelsCommand } from "./runtime/channelsCommand.js";
import { runSandboxCommand } from "./runtime/sandboxCommand.js";
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
import { registerCreatorForgeTool } from "./tools/creatorForge.js";
import { registerAgentPresetTools } from "./tools/agentPresets.js";
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
import { bridgeDurableTaskLifecycle } from "./tasks/durableTaskBridge.js";
import { DurableTaskRuntime } from "./tasks/durableTaskRuntime.js";

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
      ["xerxes discord --token <token> --application-id <id>", "run the Discord channel gateway"],
      ["xerxes slack --token <token> [--signing-secret <secret>]", "run the Slack channel gateway"],
      ["xerxes whatsapp --access-token <token> --phone-number-id <id>", "run the WhatsApp channel gateway"],
      ["xerxes email --smtp-user <u> --smtp-password <p> [--smtp-host <h>] [--smtp-port <p>]", "run the email channel gateway"],
      ["xerxes signal --rest-url <url> --sender-number <number>", "run the Signal channel gateway"],
    ],
  },
  {
    title: "Maintain",
    commands: [
      ["xerxes auth login codex", "sign in to ChatGPT and use its Codex plan"],
      ["xerxes auth login copilot", "sign in to GitHub and use Copilot models"],
      ["xerxes auth login anthropic|kimi|openrouter|xai|radius", "authorize subscription or gateway OAuth sessions"],
      ["xerxes doctor", "check the host, providers, and configuration"],
      ["xerxes update [--check] [--git] [--dry-run] [--apply]", "report or apply an update"],
      ["xerxes install --cloud-code [--force] [--dry-run]", "install a companion integration"],
      ["xerxes export [session]", "write a session transcript to disk"],
      ["xerxes skill <skill> [arguments]", "run a bundled skill"],
      ["xerxes setup [--provider <p>] [--model <m>] [--api-key <k>] [--permission-mode <mode>]", "create an initial provider configuration"],
      ["xerxes workspace create|exec|read|write|destroy --id <id>", "manage a local sandbox workspace"],
      ["xerxes schedule create|fire|list --id <id> --schedule <spec>", "manage durable scheduled triggers"],
      ["xerxes memory record|review|classify|correct|expire|list", "manage governed memory records"],
      ["xerxes capability register|unregister|list|diff --id <id>", "manage capability manifests"],
      ["xerxes telemetry record|list|benchmark|inject", "record events, benchmark, or inject failures"],
      ["xerxes status [--directory <dir>]", "show subsystem health snapshot"],
      ["xerxes channels list", "show supported channel gateways"],
      ["xerxes sandbox status", "report local sandbox availability"],
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

const { agent: cliAgentReference, rest: agentlessArguments } = extractAgentOption(
  Bun.argv.slice(2),
);
const { format: cliOutputFormat, rest: cliArguments } = extractOutputFormatOption(agentlessArguments);
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
  "setup",
  "workspace",
  "schedule",
  "memory",
  "capability",
  "telemetry",
  "status",
  "channels",
  "sandbox",
  "daemon",
  "telegram",
  "discord",
  "slack",
  "whatsapp",
  "email",
  "signal",
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

/** Image generation through OpenRouter's chat-modality surface, when a key exists. */
function generateImageToolOptions(workspaceRoot: string): {
  resolveApiKey: () => string;
  workspaceRoot: string;
} {
  return {
    resolveApiKey: () => process.env.OPENROUTER_API_KEY?.trim() ?? "",
    workspaceRoot,
  };
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
  const json = argumentsAfterCommand.includes("--json")
  const report = runAllDoctorChecks();
  if (json) {
    console.log(JSON.stringify(report, null, 2))
  } else {
    printDoctorReport(report);
  }
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
} else if (argument === "setup") {
  const options = parseValueOptions(argumentsAfterCommand, "setup", [
    "--provider",
    "--model",
    "--api-key",
    "--permission-mode",
    "--profile",
    "--target",
  ]);
  const target = options.get("--target") ?? `${xerxesHome()}/config/setup.yaml`;
  const answers: Record<string, unknown> = {}
  for (const [flag, key] of [
    ["--provider", "provider"],
    ["--model", "model"],
    ["--api-key", "api_key"],
    ["--permission-mode", "permission_mode"],
  ] as const) {
    const value = options.get(flag)
    if (value !== undefined) answers[key] = value
  }
  const profile = options.get("--profile")
  try {
    await runSetupCommand({ targetPath: target, answers, ...(profile === undefined ? {} : { profile }) })
    console.log(`Wrote setup configuration to ${target}`)
    process.exit(0)
  } catch (error) {
    reportCommandUsageError(error instanceof Error ? error : new Error(String(error)), "xerxes setup --help")
  }
} else if (argument === "workspace") {
  const action = argumentsAfterCommand[0]
  if (
    action !== "create" &&
    action !== "exec" &&
    action !== "read" &&
    action !== "write" &&
    action !== "destroy"
  ) {
    reportCommandUsageError(new Error("workspace requires action: create|exec|read|write|destroy"), "xerxes workspace --help")
  }
  const knownFlags = new Set(["--id", "--working-dir", "--path", "--content"]);
  const options = new Map<string, string>()
  const command: string[] = []
  // `--` ends option parsing, so the command being run inside the workspace can
  // carry its own flags. Without it every `-`-prefixed token was claimed as a
  // workspace option and `xerxes workspace exec --id w ls -la` died on `-la`.
  let parsingOptions = true
  for (let index = 1; index < argumentsAfterCommand.length; index += 1) {
    const flag = argumentsAfterCommand[index]
    if (flag === undefined) continue
    if (parsingOptions && flag === "--") {
      parsingOptions = false
      continue
    }
    if (!parsingOptions || !flag.startsWith("-")) {
      command.push(flag)
      continue
    }
    if (!knownFlags.has(flag)) {
      reportCommandUsageError(new Error(`Unknown workspace option: ${flag}`), "xerxes workspace --help")
    }
    const value = argumentsAfterCommand[index + 1]
    // Only a MISSING value is an error. Rejecting anything starting with `-`
    // also rejected legitimate values: negative numbers, and paths or content
    // that begin with a dash.
    if (value === undefined) {
      reportCommandUsageError(new Error(`workspace option ${flag} requires a value`), "xerxes workspace --help")
    }
    options.set(flag, value)
    index += 1
  }
  const id = options.get("--id")
  if (typeof id !== "string" || id.length === 0) {
    reportCommandUsageError(new Error("workspace requires --id"), "xerxes workspace --help")
  }
  const result = await runWorkspaceCommand({
    action,
    id,
    ...optionalOptions(options, {
      workingDir: "--working-dir",
      path: "--path",
      content: "--content",
    }),
    command,
  })
  if (!result.ok) {
    reportCommandUsageError(new Error(result.error ?? "workspace command failed"), "xerxes workspace --help")
  }
  if (result.stdout) console.log(result.stdout)
  if (result.stderr) console.error(result.stderr)
  if (result.content !== undefined) console.log(result.content)
  process.exit(result.exitCode ?? 0)
} else if (argument === "schedule") {
  const action = argumentsAfterCommand[0]
  if (
    action !== "create" &&
    action !== "disable" &&
    action !== "enable" &&
    action !== "remove" &&
    action !== "fire" &&
    action !== "list"
  ) {
    reportCommandUsageError(new Error("schedule requires action: create|disable|enable|remove|fire|list"), "xerxes schedule --help")
  }
  const options = parseValueOptions(argumentsAfterCommand.slice(1), "schedule", [
    "--id",
    "--owner",
    "--schedule",
    "--objective",
    "--delivery-id",
    "--directory",
  ]);
  const result = await runScheduleCommand({
    action,
    ...optionalOptions(options, {
      id: "--id",
      owner: "--owner",
      schedule: "--schedule",
      objective: "--objective",
      deliveryId: "--delivery-id",
    }),
    ...optionalOptions(options, { directory: "--directory" }),
  })
  if (!result.ok) {
    reportCommandUsageError(new Error(result.error ?? "schedule command failed"), "xerxes schedule --help")
  }
  if (result.message) console.log(result.message)
  process.exit(0)
} else if (argument === "memory") {
  const action = argumentsAfterCommand[0]
  if (
    action !== "record" &&
    action !== "review" &&
    action !== "classify" &&
    action !== "correct" &&
    action !== "expire" &&
    action !== "list"
  ) {
    reportCommandUsageError(new Error("memory requires action: record|review|classify|correct|expire|list"), "xerxes memory --help")
  }
  const options = parseValueOptions(argumentsAfterCommand.slice(1), "memory", [
    "--id",
    "--content",
    "--source",
    "--agent-id",
    "--sensitivity",
    "--original-id",
    "--new-id",
    "--reason",
    "--directory",
  ]);
  const sensitivity = options.get("--sensitivity") as MemoryCommandOptions["sensitivity"];
  const result = await runMemoryCommand({
    action,
    ...optionalOptions(options, {
      id: "--id",
      content: "--content",
      source: "--source",
      agentId: "--agent-id",
      originalId: "--original-id",
      newId: "--new-id",
      reason: "--reason",
    }),
    ...(sensitivity === undefined ? {} : { sensitivity }),
    ...optionalOptions(options, { directory: "--directory" }),
  })
  if (!result.ok) {
    reportCommandUsageError(new Error(result.error ?? "memory command failed"), "xerxes memory --help")
  }
  if (result.message) console.log(result.message)
  process.exit(0)
} else if (argument === "capability") {
  const action = argumentsAfterCommand[0]
  if (
    action !== "register" &&
    action !== "unregister" &&
    action !== "list" &&
    action !== "diff"
  ) {
    reportCommandUsageError(new Error("capability requires action: register|unregister|list|diff"), "xerxes capability --help")
  }
  const options = parseValueOptions(argumentsAfterCommand.slice(1), "capability", [
    "--id",
    "--file",
    "--manifest-json",
    "--from-snapshot",
    "--to-snapshot",
    "--directory",
  ]);
  const result = await runCapabilityCommand({
    action,
    id: options.get("--id"),
    file: options.get("--file"),
    manifestJson: options.get("--manifest-json"),
    fromSnapshot: options.get("--from-snapshot"),
    toSnapshot: options.get("--to-snapshot"),
    directory: options.get("--directory"),
  })
  if (!result.ok) {
    reportCommandUsageError(new Error(result.error ?? "capability command failed"), "xerxes capability --help")
  }
  if (result.message) console.log(result.message)
  process.exit(0)
} else if (argument === "telemetry") {
  const action = argumentsAfterCommand[0]
  if (
    action !== "record" &&
    action !== "list" &&
    action !== "benchmark" &&
    action !== "inject"
  ) {
    reportCommandUsageError(new Error("telemetry requires action: record|list|benchmark|inject"), "xerxes telemetry --help")
  }
  const options = parseValueOptions(argumentsAfterCommand.slice(1), "telemetry", [
    "--event",
    "--data",
    "--name",
    "--iterations",
    "--target",
    "--operation",
    "--mode",
    "--probability",
    "--latency-ms",
    "--error-message",
    "--directory",
  ]);
  const result = await runTelemetryCommand({
    action,
    event: options.get("--event"),
    data: options.get("--data"),
    name: options.get("--name"),
    iterations: options.get("--iterations") !== undefined ? Number(options.get("--iterations")) : undefined,
    target: options.get("--target") as TelemetryCommandOptions['target'],
    operation: options.get("--operation"),
    mode: options.get("--mode") as TelemetryCommandOptions['mode'],
    probability: options.get("--probability") !== undefined ? Number(options.get("--probability")) : undefined,
    latencyMs: options.get("--latency-ms") !== undefined ? Number(options.get("--latency-ms")) : undefined,
    errorMessage: options.get("--error-message"),
    directory: options.get("--directory"),
  })
  if (!result.ok) {
    reportCommandUsageError(new Error(result.error ?? "telemetry command failed"), "xerxes telemetry --help")
  }
  if (result.message) console.log(result.message)
  process.exit(0)
} else if (argument === "status") {
  const options = parseValueOptions(argumentsAfterCommand, "status", ["--directory"]);
  const result = await runStatusCommand({ directory: options.get("--directory") })
  if (!result.ok) {
    reportCommandUsageError(new Error(result.error ?? "status command failed"), "xerxes status --help")
  }
  if (result.message) console.log(result.message)
  process.exit(0)
} else if (argument === "channels") {
  const action = argumentsAfterCommand[0]
  if (action !== "list") {
    reportCommandUsageError(new Error("channels requires action: list"), "xerxes channels --help")
  }
  const result = await runChannelsCommand(action)
  if (!result.ok) {
    reportCommandUsageError(new Error(result.error ?? "channels command failed"), "xerxes channels --help")
  }
  if (result.message) console.log(result.message)
  process.exit(0)
} else if (argument === "sandbox") {
  const action = argumentsAfterCommand[0]
  if (action !== "status") {
    reportCommandUsageError(new Error("sandbox requires action: status"), "xerxes sandbox --help")
  }
  const result = await runSandboxCommand({ action })
  if (!result.ok) {
    reportCommandUsageError(new Error(result.error ?? "sandbox command failed"), "xerxes sandbox --help")
  }
  if (result.message) console.log(result.message)
  process.exit(0)
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
} else if (argument === "discord") {
  const options = parseValueOptions(argumentsAfterCommand, "discord", [
    "--application-id",
    "--pid-file",
    "--project-dir",
    "--public-key",
    "--socket",
    "--token",
    "--transport",
  ]);
  const token =
    options.get("--token") ?? process.env.DISCORD_BOT_TOKEN?.trim();
  if (!token)
    throw new Error("discord requires --token or DISCORD_BOT_TOKEN");
  const applicationId = options.get("--application-id");
  if (!applicationId)
    throw new Error("discord requires --application-id");
  const transport = options.get("--transport") ?? "gateway";
  if (transport !== "gateway" && transport !== "webhook")
    throw new Error('discord --transport must be "gateway" or "webhook"');
  const projectDirectory = options.get("--project-dir");
  const config = discordDaemonConfig(
    loadSystemDaemonConfig({
      ...(projectDirectory ? { projectDirectory } : {}),
    }),
    token,
    applicationId,
    options.get("--public-key"),
    transport,
  );
  const socketPath =
    options.get("--socket") ?? daemonPaths(projectDirectory).socketPath;
  const pidPath = options.get("--pid-file");
  await runDaemon(config, projectDirectory, socketPath, pidPath);
  process.exit(0);
} else if (argument === "slack") {
  const options = parseValueOptions(argumentsAfterCommand, "slack", [
    "--pid-file",
    "--project-dir",
    "--signing-secret",
    "--socket",
    "--token",
  ]);
  const token =
    options.get("--token") ?? process.env.SLACK_BOT_TOKEN?.trim();
  if (!token)
    throw new Error("slack requires --token or SLACK_BOT_TOKEN");
  const projectDirectory = options.get("--project-dir");
  const config = slackDaemonConfig(
    loadSystemDaemonConfig({
      ...(projectDirectory ? { projectDirectory } : {}),
    }),
    token,
    options.get("--signing-secret"),
  );
  const socketPath =
    options.get("--socket") ?? daemonPaths(projectDirectory).socketPath;
  const pidPath = options.get("--pid-file");
  await runDaemon(config, projectDirectory, socketPath, pidPath);
  process.exit(0);
} else if (argument === "whatsapp") {
  const options = parseValueOptions(argumentsAfterCommand, "whatsapp", [
    "--access-token",
    "--app-secret",
    "--phone-number-id",
    "--pid-file",
    "--project-dir",
    "--socket",
  ]);
  const accessToken =
    options.get("--access-token") ?? process.env.WHATSAPP_ACCESS_TOKEN?.trim();
  if (!accessToken)
    throw new Error("whatsapp requires --access-token or WHATSAPP_ACCESS_TOKEN");
  const phoneNumberId = options.get("--phone-number-id");
  if (!phoneNumberId)
    throw new Error("whatsapp requires --phone-number-id");
  const projectDirectory = options.get("--project-dir");
  const config = whatsappDaemonConfig(
    loadSystemDaemonConfig({
      ...(projectDirectory ? { projectDirectory } : {}),
    }),
    accessToken,
    phoneNumberId,
    options.get("--app-secret"),
  );
  const socketPath =
    options.get("--socket") ?? daemonPaths(projectDirectory).socketPath;
  const pidPath = options.get("--pid-file");
  await runDaemon(config, projectDirectory, socketPath, pidPath);
  process.exit(0);
} else if (argument === "email") {
  const options = parseValueOptions(argumentsAfterCommand, "email", [
    "--from",
    "--pid-file",
    "--project-dir",
    "--smtp-host",
    "--smtp-password",
    "--smtp-port",
    "--smtp-user",
    "--socket",
  ]);
  const smtpHost = options.get("--smtp-host") ?? "localhost";
  const smtpPortRaw = options.get("--smtp-port");
  const smtpPort = smtpPortRaw === undefined ? 25 : Number(smtpPortRaw);
  if (!Number.isFinite(smtpPort) || smtpPort < 1 || smtpPort > 65_535)
    throw new Error("email --smtp-port must be an integer between 1 and 65535");
  const smtpUser = options.get("--smtp-user");
  if (!smtpUser)
    throw new Error("email requires --smtp-user");
  const smtpPassword = options.get("--smtp-password");
  if (!smtpPassword)
    throw new Error("email requires --smtp-password");
  const projectDirectory = options.get("--project-dir");
  const config = emailDaemonConfig(
    loadSystemDaemonConfig({
      ...(projectDirectory ? { projectDirectory } : {}),
    }),
    smtpHost,
    smtpPort,
    smtpUser,
    smtpPassword,
    options.get("--from"),
  );
  const socketPath =
    options.get("--socket") ?? daemonPaths(projectDirectory).socketPath;
  const pidPath = options.get("--pid-file");
  await runDaemon(config, projectDirectory, socketPath, pidPath);
  process.exit(0);
} else if (argument === "signal") {
  const options = parseValueOptions(argumentsAfterCommand, "signal", [
    "--pid-file",
    "--project-dir",
    "--rest-url",
    "--sender-number",
    "--socket",
  ]);
  const restBaseUrl = options.get("--rest-url") ?? process.env.SIGNAL_REST_URL?.trim();
  if (!restBaseUrl)
    throw new Error("signal requires --rest-url or SIGNAL_REST_URL");
  const senderNumber = options.get("--sender-number");
  if (!senderNumber)
    throw new Error("signal requires --sender-number");
  const projectDirectory = options.get("--project-dir");
  const config = signalDaemonConfig(
    loadSystemDaemonConfig({
      ...(projectDirectory ? { projectDirectory } : {}),
    }),
    restBaseUrl,
    senderNumber,
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
  await skillRegistry.refresh(...defaultSkillDiscoveryRoots({
    cwd: projectDirectory ?? config.projectDirectory,
  }));
  const declarativeForge = new DeclarativeToolForge();
  const agentPresetRoster = new AgentPresetRoster({
    projectDirectory: projectDirectory ?? config.projectDirectory,
  });
  // Shared by the tool registry that starts the processes and the RPC surface
  // that lists them. One instance is the whole point: a second registry would
  // be a second, permanently empty view of the same shells.
  const terminals = new TerminalRegistry();
  const buildId = await daemonBuildIdForEntry(
    import.meta.dir,
    fileURLToPath(import.meta.url),
  );
  // Forward reference: the runtime is built before the server that announces
  // its events, and the server needs the runtime. Same shape as `finishDaemon`
  // below.
  let announceModeChange: ((sessionId: string) => void) | undefined;
  const runtime = daemonRuntime(
    config,
    projectDirectory,
    profileStore,
    interactions,
    browserManager,
    {
      ...(buildId ? { buildId } : {}),
      onSessionModeChange: (sessionId) => announceModeChange?.(sessionId),
      skillRegistry,
      declarativeForge,
      agentPresetRoster,
      terminals,
    },
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
  // MCP: connect the servers configured in ~/.xerxes/mcp.json so their
  // status is real (session.status.mcp_status) and /reload-mcp works.
  // Per-server failures are recorded on the manager and logged — one broken
  // server must not stop the daemon.
  const { MCPManager } = await import("./mcp/manager.js");
  const { loadMcpConfig } = await import("./mcp/config.js");
  const mcpConfig = loadMcpConfig(join(xerxesHome(), "mcp.json"));
  for (const warning of mcpConfig.warnings) console.error(`mcp: ${warning}`);
  // Project-scoped MCP (Claude Code `.mcp.json` parity): a repo can ship its
  // own servers. They execute arbitrary commands, so they load ONLY behind
  // the workspace-config trust opt-in — otherwise the file is noted and
  // ignored. Project servers never shadow user-configured names.
  const mcpServers = [...mcpConfig.servers];
  const workspaceTrusted = process.env.XERXES_ALLOW_WORKSPACE_CONFIG === "1"
    || /^true|yes|on$/i.test(process.env.XERXES_ALLOW_WORKSPACE_CONFIG ?? "");
  const projectRoot = projectDirectory ?? config.projectDirectory;
  const projectMcpPath = join(projectRoot, ".mcp.json");
  if (existsSync(projectMcpPath)) {
    if (!workspaceTrusted) {
      console.error("mcp: project .mcp.json found but workspace config is not trusted — ignored (set XERXES_ALLOW_WORKSPACE_CONFIG=1 to enable)");
    } else {
      const projectMcp = loadMcpConfig(projectMcpPath);
      for (const warning of projectMcp.warnings) console.error(`mcp: ${warning}`);
      const known = new Set(mcpServers.map((server) => server.name));
      for (const server of projectMcp.servers) {
        if (known.has(server.name)) {
          console.error(`mcp: project server '${server.name}' ignored — a user server with that name exists`);
          continue;
        }
        mcpServers.push(server);
      }
    }
  }
  const mcpManager = new MCPManager();
  for (const server of mcpServers) {
    void mcpManager
      .addServer(server)
      .then((connected) => {
        if (!connected) console.error(`mcp: server '${server.name}' not connected (disabled or duplicate)`);
      })
      .catch((error) => console.error(`mcp: server '${server.name}' failed: ${errorMessage(error)}`));
  }
  const daemon = new DaemonServer({
    socketPath,
    runtime,
    // Sessions created without an explicit project_dir belong to THIS
    // project, not to wherever the daemon process happened to start.
    ...(projectDirectory ? { projectDirectory } : {}),
    interactions,
    browserManager,
    terminalRegistry: terminals,
    profileStore,
    autoDiscoverModelCapabilities: true,
    skillRegistry,
    declarativeForge,
    agentPresetRoster,
    ...(mcpServers.length ? { mcpManager } : {}),
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
  announceModeChange = (sessionId) => daemon.notifySessionModeChanged(sessionId);
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

function discordDaemonConfig(
  config: DaemonConfig,
  token: string,
  applicationId: string,
  publicKey: string | undefined,
  transport: 'gateway' | 'webhook',
): DaemonConfig {
  const existing = config.channels.discord ?? {};
  const settings = isRecord(existing.settings) ? existing.settings : {};
  return {
    ...config,
    channels: {
      ...config.channels,
      discord: {
        ...existing,
        type: "discord",
        enabled: true,
        settings: {
          ...settings,
          token,
          applicationId,
          ...(publicKey === undefined ? {} : { publicKey }),
          transport,
        },
      },
    },
  };
}

function slackDaemonConfig(
  config: DaemonConfig,
  botToken: string,
  signingSecret: string | undefined,
): DaemonConfig {
  const existing = config.channels.slack ?? {};
  const settings = isRecord(existing.settings) ? existing.settings : {};
  return {
    ...config,
    channels: {
      ...config.channels,
      slack: {
        ...existing,
        type: "slack",
        enabled: true,
        settings: {
          ...settings,
          botToken,
          ...(signingSecret === undefined ? {} : { signingSecret }),
        },
      },
    },
  };
}

function whatsappDaemonConfig(
  config: DaemonConfig,
  accessToken: string,
  phoneNumberId: string,
  appSecret: string | undefined,
): DaemonConfig {
  const existing = config.channels.whatsapp ?? {};
  const settings = isRecord(existing.settings) ? existing.settings : {};
  return {
    ...config,
    channels: {
      ...config.channels,
      whatsapp: {
        ...existing,
        type: "whatsapp",
        enabled: true,
        settings: {
          ...settings,
          accessToken,
          phoneNumberId,
          ...(appSecret === undefined ? {} : { appSecret }),
        },
      },
    },
  };
}

function emailDaemonConfig(
  config: DaemonConfig,
  smtpHost: string,
  smtpPort: number,
  smtpUser: string,
  smtpPassword: string,
  fromAddress: string | undefined,
): DaemonConfig {
  const existing = config.channels.email ?? {};
  const settings = isRecord(existing.settings) ? existing.settings : {};
  return {
    ...config,
    channels: {
      ...config.channels,
      email: {
        ...existing,
        type: "email",
        enabled: true,
        settings: {
          ...settings,
          smtpHost,
          smtpPort,
          smtpUser,
          smtpPassword,
          ...(fromAddress === undefined ? {} : { fromAddress }),
          requireImapTransport: true,
        },
      },
    },
  };
}

function signalDaemonConfig(
  config: DaemonConfig,
  restBaseUrl: string,
  senderNumber: string,
): DaemonConfig {
  const existing = config.channels.signal ?? {};
  const settings = isRecord(existing.settings) ? existing.settings : {};
  return {
    ...config,
    channels: {
      ...config.channels,
      signal: {
        ...existing,
        type: "signal",
        enabled: true,
        settings: {
          ...settings,
          restBaseUrl,
          senderNumber,
        },
      },
    },
  };
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

/**
 * Whether the model is shown the deferred tool surface rather than all of it.
 *
 * OFF by default, on the evidence. Deferral shrinks a request from 76 schemas
 * to ~16, which is the right direction — but measured against a real provider
 * it broke delegation outright. The same prompt, same model, same daemon:
 *
 *   deferral off  — 18.4s, 1 tool call, 20 subagent events, correct answer
 *   deferral on   — 300s, 122 tool calls, 0 subagent events, no answer at all
 *
 * Told about the hidden tools through the catalog layer, the model still
 * thrashed instead of searching once and proceeding. A surface that is large
 * but works beats a small one that does not, so the machinery and its catalog
 * stay — reachable with XERXES_DEFERRED_TOOL_LOADING=1 — and the default goes
 * back to the behaviour that demonstrably completes work.
 */
function deferredToolLoadingEnabled(
  settings: Readonly<Record<string, unknown>>,
): boolean {
  const configured = settings.deferred_tool_loading ?? settings.deferredToolLoading;
  if (typeof configured === "boolean") return configured;
  const override = (process.env.XERXES_DEFERRED_TOOL_LOADING ?? "").trim().toLowerCase();
  if (override === "0" || override === "false" || override === "off") return false;
  if (override === "1" || override === "true" || override === "on") return true;
  return false;
}

/**
 * Build an options object that OMITS absent flags rather than setting them to
 * undefined.
 *
 * With exactOptionalPropertyTypes on, `{ id: undefined }` is not the same as
 * `{}` — and the difference is real at runtime too, wherever a callee uses
 * `'id' in options` or Object.keys. Spreading this keeps every CLI command's
 * option object honest without repeating the ternary per field.
 */
function optionalOptions<K extends string>(
  parsed: ReadonlyMap<string, string | undefined>,
  fields: Readonly<Record<K, string>>,
): Partial<Record<K, string>> {
  const result: Partial<Record<K, string>> = {};
  for (const [key, flag] of Object.entries(fields) as [K, string][]) {
    const value = parsed.get(flag);
    if (value !== undefined) result[key] = value;
  }
  return result;
}

function daemonRuntime(
  config: DaemonConfig,
  projectDirectory: string | undefined,
  profileStore: ProfileStore,
  interactions?: DaemonInteractionBoard,
  browserManager?: BrowserManager,
  host: {
    readonly buildId?: string;
    /** Announce a model-driven interaction-mode change to attached clients. */
    readonly onSessionModeChange?: (sessionId: string, mode: string) => void;
    readonly skillRegistry?: SkillRegistry;
    readonly declarativeForge?: DeclarativeToolForge;
    readonly agentPresetRoster?: AgentPresetRoster;
    readonly terminals?: TerminalRegistry;
  } = {},
): InMemoryDaemonRuntime {
  const workspaceRoot = projectDirectory ?? config.projectDirectory;
  // User shell hooks (Claude Code settings.json parity): one runner for the
  // whole daemon, shared by every turn. Without this the loop's hook points
  // never fired in production — nothing passed a HookRunner down before.
  // Workspace hooks load only behind the workspace-config trust opt-in.
  const hookRunner = new HookRunner();
  {
    const hookLoad = loadShellHookConfigSync({
      allowWorkspace: process.env.XERXES_ALLOW_WORKSPACE_CONFIG === "1"
        || /^true|yes|on$/i.test(process.env.XERXES_ALLOW_WORKSPACE_CONFIG ?? ""),
      home: xerxesHome(),
      workspaceRoot,
    });
    for (const error of hookLoad.errors) console.error(`hooks: ${error}`);
    registerShellHooks(hookRunner, hookLoad.hooks, { cwd: workspaceRoot });
  }
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
  // Persistent interactive PTYs (pty_open/pty_write/...). Built once for the
  // same reason as backgroundCommands: a per-registry manager would orphan
  // live terminals on every settings rebuild. Mirrors into the terminals
  // panel so the user can watch and type.
  const ptySessions = new PtySessionManager({
    ...(host.terminals === undefined ? {} : { terminals: host.terminals }),
    workspaceRoot,
  });
  // Both process-owning managers share the daemon teardown contract; the
  // runtime accepts one lifecycle object, so compose them here.
  const processLifecycle = {
    disposeAll: async () => {
      await Promise.all([backgroundCommands.disposeAll(), ptySessions.disposeAll()]);
    },
    disposeOwner: async (owner: string) => {
      await Promise.all([
        backgroundCommands.disposeOwner(owner),
        ptySessions.disposeOwner(owner),
      ]);
    },
  };
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
    const maxTokens = connection.maxTokens;
    const maxOutputTokens = (candidate: string): number | undefined =>
      resolvedProfileMaxOutputTokens(profileStore.active(), candidate);
    // Deferred schema loading. The model gets the always-loaded core plus
    // whatever ToolSearchTool has already revealed in this transcript, instead
    // of the entire surface on every request — measured at 76 schemas, which is
    // well past where models start confusing neighbouring tools and borrowing
    // one tool's argument shape for another.
    //
    // Escape hatch rather than a hard-coded truth: discovery becomes
    // load-bearing when this is on, so a host that hits a gap can put the full
    // surface back without a rebuild.
    const tools = new ToolRegistry({
      deferredToolLoading: deferredToolLoadingEnabled({
        ...config.runtime,
        ...settings,
      }),
    });
    const computerUseTool = createMacOSComputerUseToolOptions({
      ...config.runtime,
      ...settings,
    });
    registerCoreTools(tools, {
      workspaceRoot,
      backgroundCommands,
      ptySessions,
      generateImageTool: generateImageToolOptions(workspaceRoot),
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
    if (host.declarativeForge) {
      registerCreatorForgeTool(tools, host.declarativeForge);
    }
    if (host.agentPresetRoster) {
      registerAgentPresetTools(tools, host.agentPresetRoster, {
        onChanged: () => { runtime?.reload({}); },
      });
    }
    if (browserManager) {
      registerBrowserManagerTools(tools, browserManager);
    }
    if (interactions) {
      registerDaemonQuestionTool(tools);
    }
    if (host.skillRegistry) {
      registerClaudeSkillTool(tools, host.skillRegistry);
    }
    // Same-session goals: the model states lifecycle through typed calls
    // instead of the runtime inferring it from English phrases in the prose.
    registerGoalTools(tools, {
      sessionId: (context) => String(context.sessionId ?? ""),
      metadata: (context) => context.metadata,
      // Authority is a property of the turn, written by the turn runner.
      isHumanTurn: (context) => context.metadata.goal_turn_human !== false,
      currentRound: (context) =>
        typeof context.metadata.goal_turn_round === "number"
          ? context.metadata.goal_turn_round
          : undefined,
    });
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
          contextDeltaRecorded: true,
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
    // Fallback model chain (Claude Code fallback-model parity): configured via
    // XERXES_FALLBACK_MODEL or runtime.fallback_model; the fallback client
    // reuses the primary connection's credentials and routing.
    const fallbackSetting = settings.fallback_model ?? config.runtime.fallback_model;
    const fallbackModel = process.env.XERXES_FALLBACK_MODEL?.trim()
      || (typeof fallbackSetting === "string" ? fallbackSetting.trim() : "")
      || undefined;
    const subagentOptions = {
      agentDefinitions,
      contextLimit: (model: string) => resolvedProfileContextLimit(profileStore.active(), model),
      cwd: workspaceRoot,
      // The durable attempt log. Its runtime, bridge, and every consumer branch
      // were written and tested, and nothing ever constructed one — so in
      // production `durableTaskBridge` was always undefined and every guarded
      // recording branch in the manager and the Cortex orchestrator was dead.
      durableTaskBridge: bridgeDurableTaskLifecycle(
        new DurableTaskRuntime({ directory: join(xerxesHome(), "tasks") }),
      ),
      eventBus: subagentEvents,
      ...(host.skillRegistry?.markdownIndex()
        ? { extraContext: host.skillRegistry.markdownIndex() }
        : {}),
      llm,
      ...(maxTokens === undefined ? {} : { maxTokens }),
      maxOutputTokens,
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
    // Report what a request actually carries, not what is registered. With
    // deferred loading on those differ by design, and the status line claiming
    // the full registry would hide the very thing this setting changes.
    activeToolCount = tools.definitionsForTranscript([]).length;
    const contextLimit = resolvedProfileContextLimit(profileStore.active(), connection.model);
    return new AgentTurnRunner({
      ...(contextLimit === undefined ? {} : { contextLimit }),
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
      hookRunner,
      ...(fallbackModel === undefined ? {} : {
        fallbackModel,
        createLlmForModel: (candidate: string) => createLlmClient(candidate, {
          ...(connection.apiKey ? { api_key: connection.apiKey } : {}),
          ...(connection.baseUrl ? { base_url: connection.baseUrl } : {}),
          ...(connection.provider ? { provider: connection.provider } : {}),
          ...(connection.responsesApi ? { responsesApi: true } : {}),
        }),
      }),
      ...(maxTokens === undefined ? {} : { maxTokens }),
      maxOutputTokens,
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
        // PreCompact hook before any message is dropped (Claude Code parity).
        if (hookRunner.hasHooks("on_compact")) {
          await hookRunner.run("on_compact", {
            message_count: messages.length,
            model: connection.model,
            trigger: "context_overflow",
          });
        }
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
    backgroundCommands: processLifecycle,
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
    // A mode change the MODEL made has to reach the clients too. Without this
    // the session left plan mode while every TUI kept rendering — and gating
    // on — the old mode, which is indistinguishable from the switch failing.
    ...(host.onSessionModeChange
      ? { onSessionModeChange: host.onSessionModeChange }
      : {}),
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
    hookRunner,
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
  const profileStore = new ProfileStore();
  const connection = runtimeConnection(config, profileStore.active());
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
    generateImageTool: generateImageToolOptions(workspaceRoot),
    ...(acpComputerUseTool === undefined ? {} : { computerUseTool: acpComputerUseTool }),
    agentMemoryTools: {
      memory: new AgentMemory({ projectRoot: workspaceRoot }),
      resolveSelfMemory: (context) =>
        getAgentSelfMemory(context.agentId ?? "default"),
    },
    memoryTools: { resolveContext: memoryToolContext.resolve },
  });
  registerCreatorForgeTool(tools, new DeclarativeToolForge());
  registerAgentPresetTools(tools, new AgentPresetRoster({ projectDirectory: workspaceRoot }));
  registerClaudeSkillTool(tools, skillRegistry);
  const definitions = loadAgentDefinitions({ cwd: workspaceRoot });
  const agent = definitions.get("default");
  const agentId = agent?.name ?? "default";
  const selfMemory = getAgentSelfMemory(agentId);
  const model = agent?.model || connection.model;
  const maxTokens = connection.maxTokens;
  const maxOutputTokens = (candidate: string): number | undefined =>
    resolvedProfileMaxOutputTokens(profileStore.active(), candidate);
  const effectiveMaxTokens = maxTokens ?? maxOutputTokens(model);
  const llm = createLlmClient(connection.model, {
    ...(connection.apiKey ? { api_key: connection.apiKey } : {}),
    ...(connection.baseUrl ? { base_url: connection.baseUrl } : {}),
    ...(connection.provider ? { provider: connection.provider } : {}),
    ...(connection.responsesApi ? { responsesApi: true } : {}),
  });
  const subagentHost = createNativeSubagentHost({
    agentDefinitions: definitions,
    contextLimit: candidate => resolvedProfileContextLimit(profileStore.active(), candidate),
    cwd: workspaceRoot,
    eventBus: new DaemonSubagentEventBus(),
    ...(skillRegistry.markdownIndex() ? { extraContext: skillRegistry.markdownIndex() } : {}),
    llm,
    ...(maxTokens === undefined ? {} : { maxTokens }),
    maxOutputTokens,
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
    ...(effectiveMaxTokens === undefined ? {} : { maxTokens: effectiveMaxTokens }),
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
    await runOneShot(prompt, agentReference, { outputFormat: cliOutputFormat });
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
  options: { readonly outputFormat?: OutputFormat } = {},
): Promise<void> {
  const config = loadSystemDaemonConfig();
  const profileStore = new ProfileStore();
  const connection = runtimeConnection(config, profileStore.active());
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
    generateImageTool: generateImageToolOptions(workspaceRoot),
    ...(computerUseTool === undefined ? {} : { computerUseTool }),
    agentMemoryTools: {
      memory: agentMemory,
      resolveSelfMemory: (context) =>
        getAgentSelfMemory(context.agentId ?? "default"),
    },
    memoryTools: { resolveContext: memoryToolContext.resolve },
  });
  registerCreatorForgeTool(tools, new DeclarativeToolForge());
  registerAgentPresetTools(tools, new AgentPresetRoster({ projectDirectory: workspaceRoot }));
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
  const maxTokens = connection.maxTokens;
  const maxOutputTokens = (candidate: string): number | undefined =>
    resolvedProfileMaxOutputTokens(profileStore.active(), candidate);
  const effectiveMaxTokens = maxTokens ?? maxOutputTokens(model);
  const llm = createLlmClient(model, {
    ...(connection.apiKey ? { api_key: connection.apiKey } : {}),
    ...(connection.baseUrl ? { base_url: connection.baseUrl } : {}),
    ...(connection.provider ? { provider: connection.provider } : {}),
    ...(connection.responsesApi ? { responsesApi: true } : {}),
  });
  const subagentHost = createNativeSubagentHost({
    agentDefinitions: definitions,
    contextLimit: candidate => resolvedProfileContextLimit(profileStore.active(), candidate),
    cwd: workspaceRoot,
    eventBus: new DaemonSubagentEventBus(),
    ...(skillRegistry.markdownIndex() ? { extraContext: skillRegistry.markdownIndex() } : {}),
    llm,
    ...(maxTokens === undefined ? {} : { maxTokens }),
    maxOutputTokens,
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
  const outputFormat: OutputFormat = options.outputFormat ?? "text";
  let pendingAgentEventSnapshots: readonly SpawnedAgentSnapshot[] = [];
  let wroteText = false;
  let terminalProviderError: string | undefined;
  // json buffers the whole reply; stream-json mirrors every event as NDJSON.
  let bufferedText = "";
  let lastUsage: Record<string, unknown> | undefined;
  let stopReason: string | undefined;
  const writeStreamJson = (record: Record<string, unknown>) => {
    process.stdout.write(`${JSON.stringify(record)}\n`);
  };
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
        ...(effectiveMaxTokens === undefined ? {} : { maxTokens: effectiveMaxTokens }),
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
        if (outputFormat === "stream-json") {
          writeStreamJson({ text: event.text, type: "text" });
        } else if (outputFormat === "json") {
          bufferedText += event.text;
        } else {
          process.stdout.write(event.text);
        }
      } else if (outputFormat === "stream-json" && (event.type === "tool_start" || event.type === "tool_end")) {
        writeStreamJson({ ...(event as unknown as Record<string, unknown>), type: event.type });
      } else if (event.type === "usage_update") {
        lastUsage = { ...(event as unknown as Record<string, unknown>) };
        delete lastUsage.type;
      } else if (event.type === "turn_done") {
        stopReason = (event as unknown as Record<string, unknown>).stopReason as string | undefined;
      } else if (event.type === "provider_retry" && event.final) {
        terminalProviderError = event.error;
        console.error(`Provider error: ${event.error}`);
      }
    }
  } finally {
    subagentCohort.close();
    await subagentHost.manager.shutdown();
  }
  if (outputFormat === "json") {
    writeStreamJson({
      is_error: terminalProviderError !== undefined,
      model,
      response: bufferedText,
      session_id: sessionId,
      type: "result",
      ...(stopReason === undefined ? {} : { stop_reason: stopReason }),
      ...(lastUsage === undefined ? {} : { usage: lastUsage }),
    });
  } else if (outputFormat === "stream-json") {
    writeStreamJson({
      is_error: terminalProviderError !== undefined,
      session_id: sessionId,
      type: "result",
      ...(stopReason === undefined ? {} : { stop_reason: stopReason }),
      ...(lastUsage === undefined ? {} : { usage: lastUsage }),
    });
  } else if (wroteText) {
    process.stdout.write("\n");
  }
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
  const runtime = daemonRuntime(
    config,
    projectDirectory,
    new ProfileStore(),
    undefined,
    undefined,
    {
      declarativeForge: new DeclarativeToolForge(),
      agentPresetRoster: new AgentPresetRoster({ projectDirectory }),
    },
  );
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
