// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import {
  GoogleWorkspaceBridge,
  type GoogleWorkspaceBridgeCommand,
} from "./bridge.js";
import {
  GoogleWorkspaceClient,
  type GoogleSheetValues,
  type GoogleWorkspaceEndpoints,
} from "./client.js";
import {
  GoogleWorkspaceOAuthClient,
  type GoogleAuthorizationBrowser,
  type GoogleWorkspaceAuthorizationStorage,
  type GoogleWorkspaceOAuthConfig,
} from "./oauth.js";
import { googleWorkspaceSetupGuidance } from "./setup.js";
import type { SkillFetch } from "../http.js";

/** Help for the native Google Workspace CLI. */
export const GOOGLE_WORKSPACE_CLI_USAGE = `Usage: xerxes skill google-workspace [--adapter <module>] <service> <action> [arguments]

Services and actions:
  setup [guidance|status|begin|complete|revoke]
  gmail search <query> [--max <count>]
  gmail get <message-id>
  gmail send --to <email> --subject <text> --body <text> [--cc <email>] [--from <header>] [--html] [--thread-id <id>]
  gmail reply <message-id> --body <text> [--from <header>]
  gmail labels
  gmail modify <message-id> [--add-labels <id,id>] [--remove-labels <id,id>]
  calendar list [--start <iso> --end <iso> --max <count> --calendar <id>]
  calendar create --summary <text> --start <iso> --end <iso> [--location <text> --description <text> --attendees <email,email> --calendar <id>]
  calendar delete <event-id> [--calendar <id>]
  drive search <query> [--max <count>] [--raw-query]
  contacts list [--max <count>]
  sheets get <spreadsheet-id> <range>
  sheets update <spreadsheet-id> <range> --values <json-array-of-arrays>
  sheets append <spreadsheet-id> <range> --values <json-array-of-arrays>
  docs get <document-id>

Setup requires an explicit adapter that supplies OAuth configuration, credential
storage, and fetch (plus a browser adapter only when --open is requested):
  xerxes skill google-workspace --adapter ./google-workspace.adapter.ts setup status

The adapter is explicitly selected by the caller. Xerxes does not discover
credentials, token files, environment variables, a gws binary, or Python.`;

/** Explicit host-owned dependencies accepted by the native CLI. */
export interface GoogleWorkspaceCliAdapter {
  readonly browser?: GoogleAuthorizationBrowser;
  readonly client?: GoogleWorkspaceClient;
  readonly endpoints?: Partial<GoogleWorkspaceEndpoints>;
  readonly fetchImplementation?: SkillFetch;
  readonly oauth?: GoogleWorkspaceOAuthClient;
  readonly oauthConfig?: GoogleWorkspaceOAuthConfig;
  readonly storage?: GoogleWorkspaceAuthorizationStorage;
}

export interface GoogleWorkspaceCliDependencies extends GoogleWorkspaceCliAdapter {
  /** Load an adapter only when the caller explicitly supplied `--adapter`. */
  readonly loadAdapter?: (
    path: string,
  ) => GoogleWorkspaceCliAdapter | Promise<GoogleWorkspaceCliAdapter>;
  readonly writeLine?: (line: string) => void | Promise<void>;
}

interface ParsedArguments {
  readonly flags: ReadonlyMap<string, readonly string[]>;
  readonly positionals: readonly string[];
}

interface FlagParserOptions {
  readonly booleanFlags?: readonly string[];
  readonly valueFlags?: readonly string[];
}

interface AdapterArguments {
  readonly adapterPath?: string;
  readonly command: readonly string[];
}

class GoogleWorkspaceCliUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GoogleWorkspaceCliUsageError";
  }
}

class GoogleWorkspaceCliConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "GoogleWorkspaceCliConfigurationError";
  }
}

/**
 * Run the documented native Google Workspace workflow with caller-owned ports.
 *
 * No ambient credential path, environment variable, browser, or network transport
 * is ever selected here. A caller may pass dependencies directly or choose an
 * adapter module explicitly with `--adapter`.
 */
export async function runGoogleWorkspaceCli(
  args: readonly string[],
  dependencies: GoogleWorkspaceCliDependencies = {},
): Promise<number> {
  const writeLine = dependencies.writeLine ?? ((line) => console.log(line));
  try {
    const adapterArguments = extractAdapterArguments(args);
    const activeDependencies = await resolveDependencies(
      adapterArguments,
      dependencies,
    );
    const [service, ...rest] = adapterArguments.command;
    if (service === undefined || isHelp(service)) {
      await writeLine(GOOGLE_WORKSPACE_CLI_USAGE);
      return 0;
    }
    const result = await dispatchGoogleWorkspaceCommand(
      service,
      rest,
      activeDependencies,
    );
    await writeLine(JSON.stringify(result, null, 2));
    return 0;
  } catch (error) {
    const message = errorMessage(error);
    if (error instanceof GoogleWorkspaceCliUsageError) {
      await writeLine(`${message}\n\n${GOOGLE_WORKSPACE_CLI_USAGE}`);
      return 2;
    }
    if (error instanceof GoogleWorkspaceCliConfigurationError) {
      await writeLine(
        `${message}\n\nRun \`xerxes skill google-workspace setup guidance\` for the explicit adapter contract.`,
      );
      return 2;
    }
    await writeLine(`Google Workspace command failed: ${message}`);
    return 1;
  }
}

/** Direct Bun entry point. Operational commands still require an explicit adapter. */
export async function main(
  args: readonly string[] = process.argv.slice(2),
): Promise<number> {
  return runGoogleWorkspaceCli(args);
}

/**
 * Load a caller-selected local Bun adapter module.
 *
 * The path is never inferred. The module must use either a default export or a
 * named `googleWorkspaceCliAdapter` export and is responsible for supplying its
 * own credential storage and configuration.
 */
export async function loadGoogleWorkspaceCliAdapter(
  path: string,
): Promise<GoogleWorkspaceCliAdapter> {
  const text = requiredText(path, "--adapter");
  const url = text.startsWith("file:")
    ? new URL(text)
    : pathToFileURL(resolve(text));
  if (url.protocol !== "file:")
    throw new GoogleWorkspaceCliConfigurationError(
      "--adapter must be a local file path or file: URL",
    );
  let loaded: Record<string, unknown>;
  try {
    loaded = (await import(url.href)) as Record<string, unknown>;
  } catch (error) {
    throw new GoogleWorkspaceCliConfigurationError(
      `could not load explicit Google Workspace adapter: ${errorMessage(error)}`,
    );
  }
  const candidate = loaded.default ?? loaded.googleWorkspaceCliAdapter;
  if (!isRecord(candidate)) {
    throw new GoogleWorkspaceCliConfigurationError(
      "the explicit adapter must export an object as default or googleWorkspaceCliAdapter",
    );
  }
  validateAdapter(candidate);
  return candidate as GoogleWorkspaceCliAdapter;
}

async function dispatchGoogleWorkspaceCommand(
  service: string,
  args: readonly string[],
  dependencies: GoogleWorkspaceCliDependencies,
): Promise<unknown> {
  switch (service) {
    case "setup":
      return dispatchSetup(args, dependencies);
    case "gmail":
    case "calendar":
    case "drive":
    case "contacts":
    case "sheets":
    case "docs": {
      const command = parseServiceCommand(service, args);
      return createBridge(dependencies).dispatch(command);
    }
    default:
      throw new GoogleWorkspaceCliUsageError(
        `Unknown Google Workspace service: ${service}`,
      );
  }
}

async function dispatchSetup(
  args: readonly string[],
  dependencies: GoogleWorkspaceCliDependencies,
): Promise<unknown> {
  const [action = "guidance", ...rest] = args;
  if (isHelp(action)) return { usage: GOOGLE_WORKSPACE_CLI_USAGE };
  switch (action) {
    case "guidance":
      requireNoArguments(rest, "setup guidance");
      return googleWorkspaceSetupGuidance();
    case "status":
      requireNoArguments(rest, "setup status");
      return oauthFor(dependencies).authorizationStatus();
    case "begin": {
      const parsed = parseFlags(rest, { booleanFlags: ["--help", "--open"] });
      if (hasFlag(parsed, "--help"))
        return {
          usage: "Usage: xerxes skill google-workspace setup begin [--open]",
        };
      requireNoPositionals(parsed, "setup begin");
      if (
        hasFlag(parsed, "--open") &&
        dependencies.browser === undefined &&
        dependencies.oauth === undefined
      ) {
        throw new GoogleWorkspaceCliConfigurationError(
          "setup begin --open requires a caller-provided browser adapter",
        );
      }
      const request = await oauthFor(dependencies).beginAuthorization({
        openBrowser: hasFlag(parsed, "--open"),
      });
      return { authorization_url: request.url };
    }
    case "complete": {
      const callbackOrCode = exactlyOnePositional(
        parseFlags(rest, { booleanFlags: ["--help"] }),
        "setup complete requires an OAuth callback URL or authorization code",
      );
      await oauthFor(dependencies).completeAuthorization(callbackOrCode);
      return { status: await oauthFor(dependencies).authorizationStatus() };
    }
    case "revoke":
      requireNoArguments(rest, "setup revoke");
      return { revoked: await oauthFor(dependencies).revoke() };
    default:
      throw new GoogleWorkspaceCliUsageError(
        `Unknown Google Workspace setup action: ${action}`,
      );
  }
}

function parseServiceCommand(
  service: string,
  args: readonly string[],
): GoogleWorkspaceBridgeCommand {
  const [action, ...rest] = args;
  if (action === undefined || isHelp(action)) {
    throw new GoogleWorkspaceCliUsageError(
      `A Google Workspace ${service} action is required`,
    );
  }
  switch (service) {
    case "gmail":
      return parseGmailCommand(action, rest);
    case "calendar":
      return parseCalendarCommand(action, rest);
    case "drive":
      return parseDriveCommand(action, rest);
    case "contacts":
      return parseContactsCommand(action, rest);
    case "sheets":
      return parseSheetsCommand(action, rest);
    case "docs":
      return parseDocsCommand(action, rest);
    default:
      throw new GoogleWorkspaceCliUsageError(
        `Unknown Google Workspace service: ${service}`,
      );
  }
}

function parseGmailCommand(
  action: string,
  args: readonly string[],
): GoogleWorkspaceBridgeCommand {
  switch (action) {
    case "search": {
      const parsed = parseFlags(args, {
        booleanFlags: ["--help"],
        valueFlags: ["--max"],
      });
      if (hasFlag(parsed, "--help"))
        throw new GoogleWorkspaceCliUsageError(
          "Usage: google-workspace gmail search <query> [--max <count>]",
        );
      return {
        action,
        ...(singleValue(parsed, "--max") === undefined
          ? {}
          : {
              maxResults: positiveInteger(
                singleValue(parsed, "--max")!,
                "--max",
              ),
            }),
        query: exactlyOneOrMorePositionals(
          parsed,
          "gmail search requires a query",
        ).join(" "),
        service: "gmail",
      };
    }
    case "get":
      return {
        action,
        messageId: exactlyOnePositional(
          parseFlags(args, { booleanFlags: ["--help"] }),
          "gmail get requires a message ID",
        ),
        service: "gmail",
      };
    case "send": {
      const parsed = parseFlags(args, {
        booleanFlags: ["--help", "--html"],
        valueFlags: [
          "--to",
          "--subject",
          "--body",
          "--cc",
          "--from",
          "--thread-id",
        ],
      });
      if (hasFlag(parsed, "--help"))
        throw new GoogleWorkspaceCliUsageError(
          "Usage: google-workspace gmail send --to <email> --subject <text> --body <text> [--cc <email>] [--from <header>] [--html] [--thread-id <id>]",
        );
      requireNoPositionals(parsed, "gmail send");
      const cc = singleValue(parsed, "--cc");
      const from = singleValue(parsed, "--from");
      const threadId = singleValue(parsed, "--thread-id");
      return {
        action,
        options: {
          body: requiredFlag(parsed, "--body"),
          ...(cc === undefined ? {} : { cc }),
          ...(from === undefined ? {} : { from }),
          ...(hasFlag(parsed, "--html") ? { html: true } : {}),
          subject: requiredFlag(parsed, "--subject"),
          ...(threadId === undefined ? {} : { threadId }),
          to: requiredFlag(parsed, "--to"),
        },
        service: "gmail",
      };
    }
    case "reply": {
      const parsed = parseFlags(args, {
        booleanFlags: ["--help"],
        valueFlags: ["--body", "--from"],
      });
      if (hasFlag(parsed, "--help"))
        throw new GoogleWorkspaceCliUsageError(
          "Usage: google-workspace gmail reply <message-id> --body <text> [--from <header>]",
        );
      const from = singleValue(parsed, "--from");
      return {
        action,
        messageId: exactlyOnePositional(
          parsed,
          "gmail reply requires a message ID",
        ),
        options: {
          body: requiredFlag(parsed, "--body"),
          ...(from === undefined ? {} : { from }),
        },
        service: "gmail",
      };
    }
    case "labels": {
      const parsed = parseFlags(args, { booleanFlags: ["--help"] });
      if (hasFlag(parsed, "--help"))
        throw new GoogleWorkspaceCliUsageError(
          "Usage: google-workspace gmail labels",
        );
      requireNoPositionals(parsed, "gmail labels");
      return { action, service: "gmail" };
    }
    case "modify": {
      const parsed = parseFlags(args, {
        booleanFlags: ["--help"],
        valueFlags: ["--add-labels", "--remove-labels"],
      });
      if (hasFlag(parsed, "--help"))
        throw new GoogleWorkspaceCliUsageError(
          "Usage: google-workspace gmail modify <message-id> [--add-labels <id,id>] [--remove-labels <id,id>]",
        );
      return {
        action,
        messageId: exactlyOnePositional(
          parsed,
          "gmail modify requires a message ID",
        ),
        options: {
          ...(singleValue(parsed, "--add-labels") === undefined
            ? {}
            : {
                addLabelIds: commaSeparated(
                  singleValue(parsed, "--add-labels")!,
                  "--add-labels",
                ),
              }),
          ...(singleValue(parsed, "--remove-labels") === undefined
            ? {}
            : {
                removeLabelIds: commaSeparated(
                  singleValue(parsed, "--remove-labels")!,
                  "--remove-labels",
                ),
              }),
        },
        service: "gmail",
      };
    }
    default:
      throw new GoogleWorkspaceCliUsageError(`Unknown Gmail action: ${action}`);
  }
}

function parseCalendarCommand(
  action: string,
  args: readonly string[],
): GoogleWorkspaceBridgeCommand {
  switch (action) {
    case "list": {
      const parsed = parseFlags(args, {
        booleanFlags: ["--help"],
        valueFlags: ["--start", "--end", "--max", "--calendar"],
      });
      if (hasFlag(parsed, "--help"))
        throw new GoogleWorkspaceCliUsageError(
          "Usage: google-workspace calendar list [--start <iso> --end <iso> --max <count> --calendar <id>]",
        );
      requireNoPositionals(parsed, "calendar list");
      const calendarId = singleValue(parsed, "--calendar");
      const end = singleValue(parsed, "--end");
      const max = singleValue(parsed, "--max");
      const start = singleValue(parsed, "--start");
      return {
        action,
        options: {
          ...(calendarId === undefined ? {} : { calendarId }),
          ...(end === undefined ? {} : { end }),
          ...(max === undefined
            ? {}
            : { maxResults: positiveInteger(max, "--max") }),
          ...(start === undefined ? {} : { start }),
        },
        service: "calendar",
      };
    }
    case "create": {
      const parsed = parseFlags(args, {
        booleanFlags: ["--help"],
        valueFlags: [
          "--summary",
          "--start",
          "--end",
          "--location",
          "--description",
          "--attendees",
          "--calendar",
        ],
      });
      if (hasFlag(parsed, "--help"))
        throw new GoogleWorkspaceCliUsageError(
          "Usage: google-workspace calendar create --summary <text> --start <iso> --end <iso> [--location <text> --description <text> --attendees <email,email> --calendar <id>]",
        );
      requireNoPositionals(parsed, "calendar create");
      const attendees = singleValue(parsed, "--attendees");
      const calendarId = singleValue(parsed, "--calendar");
      const description = singleValue(parsed, "--description");
      const location = singleValue(parsed, "--location");
      return {
        action,
        options: {
          ...(attendees === undefined
            ? {}
            : { attendees: commaSeparated(attendees, "--attendees") }),
          ...(calendarId === undefined ? {} : { calendarId }),
          ...(description === undefined ? {} : { description }),
          end: requiredFlag(parsed, "--end"),
          ...(location === undefined ? {} : { location }),
          start: requiredFlag(parsed, "--start"),
          summary: requiredFlag(parsed, "--summary"),
        },
        service: "calendar",
      };
    }
    case "delete": {
      const parsed = parseFlags(args, {
        booleanFlags: ["--help"],
        valueFlags: ["--calendar"],
      });
      if (hasFlag(parsed, "--help"))
        throw new GoogleWorkspaceCliUsageError(
          "Usage: google-workspace calendar delete <event-id> [--calendar <id>]",
        );
      const calendarId = singleValue(parsed, "--calendar");
      return {
        action,
        eventId: exactlyOnePositional(
          parsed,
          "calendar delete requires an event ID",
        ),
        options: calendarId === undefined ? {} : { calendarId },
        service: "calendar",
      };
    }
    default:
      throw new GoogleWorkspaceCliUsageError(
        `Unknown Calendar action: ${action}`,
      );
  }
}

function parseDriveCommand(
  action: string,
  args: readonly string[],
): GoogleWorkspaceBridgeCommand {
  if (action !== "search")
    throw new GoogleWorkspaceCliUsageError(`Unknown Drive action: ${action}`);
  const parsed = parseFlags(args, {
    booleanFlags: ["--help", "--raw-query"],
    valueFlags: ["--max"],
  });
  if (hasFlag(parsed, "--help"))
    throw new GoogleWorkspaceCliUsageError(
      "Usage: google-workspace drive search <query> [--max <count>] [--raw-query]",
    );
  return {
    action,
    ...(singleValue(parsed, "--max") === undefined
      ? {}
      : {
          maxResults: positiveInteger(singleValue(parsed, "--max")!, "--max"),
        }),
    query: exactlyOneOrMorePositionals(
      parsed,
      "drive search requires a query",
    ).join(" "),
    ...(hasFlag(parsed, "--raw-query") ? { rawQuery: true } : {}),
    service: "drive",
  };
}

function parseContactsCommand(
  action: string,
  args: readonly string[],
): GoogleWorkspaceBridgeCommand {
  if (action !== "list")
    throw new GoogleWorkspaceCliUsageError(
      `Unknown Contacts action: ${action}`,
    );
  const parsed = parseFlags(args, {
    booleanFlags: ["--help"],
    valueFlags: ["--max"],
  });
  if (hasFlag(parsed, "--help"))
    throw new GoogleWorkspaceCliUsageError(
      "Usage: google-workspace contacts list [--max <count>]",
    );
  requireNoPositionals(parsed, "contacts list");
  return {
    action,
    ...(singleValue(parsed, "--max") === undefined
      ? {}
      : { pageSize: positiveInteger(singleValue(parsed, "--max")!, "--max") }),
    service: "contacts",
  };
}

function parseSheetsCommand(
  action: string,
  args: readonly string[],
): GoogleWorkspaceBridgeCommand {
  const parsed = parseFlags(args, {
    booleanFlags: ["--help"],
    valueFlags: ["--values"],
  });
  if (hasFlag(parsed, "--help"))
    throw new GoogleWorkspaceCliUsageError(
      `Usage: google-workspace sheets ${action} <spreadsheet-id> <range>${action === "get" ? "" : " --values <json-array-of-arrays>"}`,
    );
  const positionals = exactlyPositionals(
    parsed,
    2,
    `sheets ${action} requires a spreadsheet ID and range`,
  );
  const spreadsheetId = positionals[0] ?? "";
  const range = positionals[1] ?? "";
  switch (action) {
    case "get":
      return { action, range, service: "sheets", spreadsheetId };
    case "update":
    case "append":
      return {
        action,
        range,
        service: "sheets",
        spreadsheetId,
        values: parseSheetValues(requiredFlag(parsed, "--values")),
      };
    default:
      throw new GoogleWorkspaceCliUsageError(
        `Unknown Sheets action: ${action}`,
      );
  }
}

function parseDocsCommand(
  action: string,
  args: readonly string[],
): GoogleWorkspaceBridgeCommand {
  if (action !== "get")
    throw new GoogleWorkspaceCliUsageError(`Unknown Docs action: ${action}`);
  return {
    action,
    documentId: exactlyOnePositional(
      parseFlags(args, { booleanFlags: ["--help"] }),
      "docs get requires a document ID",
    ),
    service: "docs",
  };
}

function createBridge(
  dependencies: GoogleWorkspaceCliDependencies,
): GoogleWorkspaceBridge {
  const client = clientFor(dependencies);
  return new GoogleWorkspaceBridge({
    client,
    ...(dependencies.oauth === undefined ? {} : { oauth: dependencies.oauth }),
  });
}

function clientFor(
  dependencies: GoogleWorkspaceCliDependencies,
): GoogleWorkspaceClient {
  if (dependencies.client !== undefined) return dependencies.client;
  const fetchImplementation = fetchFor(dependencies);
  return new GoogleWorkspaceClient({
    ...(dependencies.endpoints === undefined
      ? {}
      : { endpoints: dependencies.endpoints }),
    fetchImplementation,
    tokenProvider: oauthFor(dependencies),
  });
}

function oauthFor(
  dependencies: GoogleWorkspaceCliDependencies,
): GoogleWorkspaceOAuthClient {
  if (dependencies.oauth !== undefined) return dependencies.oauth;
  if (dependencies.oauthConfig === undefined) {
    throw new GoogleWorkspaceCliConfigurationError(
      "Google Workspace requires caller-provided OAuth configuration or an OAuth client",
    );
  }
  if (dependencies.storage === undefined) {
    throw new GoogleWorkspaceCliConfigurationError(
      "Google Workspace requires caller-provided OAuth authorization storage or an OAuth client",
    );
  }
  return new GoogleWorkspaceOAuthClient(dependencies.oauthConfig, {
    ...(dependencies.browser === undefined
      ? {}
      : { browser: dependencies.browser }),
    fetchImplementation: fetchFor(dependencies),
    storage: dependencies.storage,
  });
}

function fetchFor(dependencies: GoogleWorkspaceCliDependencies): SkillFetch {
  if (dependencies.fetchImplementation === undefined) {
    throw new GoogleWorkspaceCliConfigurationError(
      "Google Workspace requires a caller-provided fetch implementation or a fully configured native client",
    );
  }
  return dependencies.fetchImplementation;
}

async function resolveDependencies(
  argumentsWithAdapter: AdapterArguments,
  dependencies: GoogleWorkspaceCliDependencies,
): Promise<GoogleWorkspaceCliDependencies> {
  if (argumentsWithAdapter.adapterPath === undefined) return dependencies;
  const loadAdapter = dependencies.loadAdapter ?? loadGoogleWorkspaceCliAdapter;
  const adapter = await loadAdapter(argumentsWithAdapter.adapterPath);
  return {
    ...dependencies,
    ...adapter,
    ...(dependencies.loadAdapter === undefined
      ? {}
      : { loadAdapter: dependencies.loadAdapter }),
    ...(dependencies.writeLine === undefined
      ? {}
      : { writeLine: dependencies.writeLine }),
  };
}

function extractAdapterArguments(args: readonly string[]): AdapterArguments {
  const command: string[] = [];
  let adapterPath: string | undefined;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--adapter") {
      if (adapterPath !== undefined)
        throw new GoogleWorkspaceCliUsageError(
          "--adapter may only be provided once",
        );
      adapterPath = requiredText(args[index + 1], "--adapter");
      index += 1;
      continue;
    }
    if (argument === undefined) continue;
    command.push(argument);
  }
  return { ...(adapterPath === undefined ? {} : { adapterPath }), command };
}

function parseFlags(
  args: readonly string[],
  options: FlagParserOptions,
): ParsedArguments {
  const booleanFlags = new Set(options.booleanFlags ?? []);
  const valueFlags = new Set(options.valueFlags ?? []);
  const flags = new Map<string, string[]>();
  const positionals: string[] = [];
  let positionalOnly = false;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === undefined) continue;
    if (argument === "--") {
      positionalOnly = true;
      continue;
    }
    if (!positionalOnly && booleanFlags.has(argument)) {
      appendFlag(flags, argument, "true");
      continue;
    }
    if (!positionalOnly && valueFlags.has(argument)) {
      appendFlag(flags, argument, requiredText(args[index + 1], argument));
      index += 1;
      continue;
    }
    if (!positionalOnly && argument.startsWith("-"))
      throw new GoogleWorkspaceCliUsageError(`Unknown option: ${argument}`);
    positionals.push(argument);
  }
  return { flags, positionals };
}

function appendFlag(
  flags: Map<string, string[]>,
  name: string,
  value: string,
): void {
  const values = flags.get(name) ?? [];
  values.push(value);
  flags.set(name, values);
}

function hasFlag(parsed: ParsedArguments, name: string): boolean {
  return parsed.flags.has(name);
}

function singleValue(
  parsed: ParsedArguments,
  name: string,
): string | undefined {
  const values = parsed.flags.get(name) ?? [];
  if (values.length > 1)
    throw new GoogleWorkspaceCliUsageError(`${name} may only be provided once`);
  return values[0];
}

function requiredFlag(parsed: ParsedArguments, name: string): string {
  const value = singleValue(parsed, name);
  if (value === undefined)
    throw new GoogleWorkspaceCliUsageError(`${name} is required`);
  return value;
}

function exactlyOnePositional(
  parsed: ParsedArguments,
  message: string,
): string {
  if (parsed.positionals.length !== 1)
    throw new GoogleWorkspaceCliUsageError(message);
  return parsed.positionals[0] ?? "";
}

function exactlyOneOrMorePositionals(
  parsed: ParsedArguments,
  message: string,
): readonly string[] {
  if (!parsed.positionals.length)
    throw new GoogleWorkspaceCliUsageError(message);
  return parsed.positionals;
}

function exactlyPositionals(
  parsed: ParsedArguments,
  count: number,
  message: string,
): readonly string[] {
  if (parsed.positionals.length !== count)
    throw new GoogleWorkspaceCliUsageError(message);
  return parsed.positionals;
}

function requireNoPositionals(parsed: ParsedArguments, command: string): void {
  if (parsed.positionals.length)
    throw new GoogleWorkspaceCliUsageError(
      `${command} does not accept positional arguments`,
    );
}

function requireNoArguments(args: readonly string[], command: string): void {
  if (args.length)
    throw new GoogleWorkspaceCliUsageError(
      `${command} does not accept arguments`,
    );
}

function positiveInteger(value: string, name: string): number {
  if (!/^\d+$/.test(value))
    throw new GoogleWorkspaceCliUsageError(
      `${name} must be a positive integer`,
    );
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 1)
    throw new GoogleWorkspaceCliUsageError(
      `${name} must be a positive integer`,
    );
  return parsed;
}

function commaSeparated(value: string, name: string): readonly string[] {
  const values = value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  if (!values.length)
    throw new GoogleWorkspaceCliUsageError(
      `${name} requires at least one non-empty value`,
    );
  return values;
}

function parseSheetValues(value: string): GoogleSheetValues {
  let parsed: unknown;
  try {
    parsed = JSON.parse(value) as unknown;
  } catch (error) {
    throw new GoogleWorkspaceCliUsageError(
      `--values must be valid JSON: ${errorMessage(error)}`,
    );
  }
  if (!Array.isArray(parsed) || !parsed.every((row) => Array.isArray(row))) {
    throw new GoogleWorkspaceCliUsageError(
      "--values must be a JSON array of arrays",
    );
  }
  return parsed as unknown as GoogleSheetValues;
}

function validateAdapter(adapter: Readonly<Record<string, unknown>>): void {
  if (
    adapter.fetchImplementation !== undefined &&
    typeof adapter.fetchImplementation !== "function"
  ) {
    throw new GoogleWorkspaceCliConfigurationError(
      "adapter.fetchImplementation must be a function",
    );
  }
  if (adapter.storage !== undefined && !isRecord(adapter.storage)) {
    throw new GoogleWorkspaceCliConfigurationError(
      "adapter.storage must implement GoogleWorkspaceAuthorizationStorage",
    );
  }
  if (adapter.oauthConfig !== undefined && !isRecord(adapter.oauthConfig)) {
    throw new GoogleWorkspaceCliConfigurationError(
      "adapter.oauthConfig must be a Google OAuth configuration object",
    );
  }
  if (adapter.client !== undefined && !isRecord(adapter.client)) {
    throw new GoogleWorkspaceCliConfigurationError(
      "adapter.client must be a GoogleWorkspaceClient instance",
    );
  }
  if (adapter.oauth !== undefined && !isRecord(adapter.oauth)) {
    throw new GoogleWorkspaceCliConfigurationError(
      "adapter.oauth must be a GoogleWorkspaceOAuthClient instance",
    );
  }
}

function requiredText(value: string | undefined, name: string): string {
  if (typeof value !== "string" || !value.trim())
    throw new GoogleWorkspaceCliUsageError(
      `${name} requires a non-empty value`,
    );
  return value.trim();
}

function isHelp(value: string): boolean {
  return value === "--help" || value === "-h" || value === "help";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

if (import.meta.main) {
  process.exitCode = await main();
}
