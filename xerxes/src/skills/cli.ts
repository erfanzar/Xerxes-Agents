// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile } from "node:fs/promises";

import { ArxivClient, formatArxivResult } from "./arxiv.js";
import { uploadExcalidrawDocument } from "./excalidraw.js";
import type { SkillFetch } from "./http.js";
import {
  FindNearbyClient,
  formatNearbyPlaces,
  type GeographicPoint,
  type NearbyPlace,
} from "./nearby.js";
import {
  runOcrDocumentsCli,
  type OcrDocumentsCliDependencies,
} from "./ocrDocuments/cli.js";
import {
  formatPolymarketMarket,
  formatPolymarketPercentage,
  formatPolymarketVolume,
  PolymarketClient,
  type PolymarketEvent,
  type PolymarketHistoryPoint,
  type PolymarketMarket,
  type PolymarketOrderBook,
  type PolymarketSearchResult,
  type PolymarketTrade,
} from "./polymarket.js";
import {
  runGoogleWorkspaceCli,
  type GoogleWorkspaceCliDependencies,
} from "./googleWorkspace/cli.js";
import {
  runPowerPointCli,
  type PowerPointCliDependencies,
} from "./powerpoint/cli.js";
import {
  runGrpoTemplateCli,
  type GrpoTemplateCliDependencies,
} from "./grpoTraining/cli.js";
import { YoutubeTranscriptClient } from "./youtubeTranscript.js";

/** Help for the native bundled-skill CLI surface. */
export const BUNDLED_SKILL_CLI_USAGE = `Usage: xerxes skill <skill> [arguments]

Native bundled skills:
  excalidraw upload <file.excalidraw>
  find-nearby (--near <place> | --lat <latitude> --lon <longitude>) [--type <amenity>] [--radius <meters>] [--limit <count>] [--json]
  youtube-transcript <url-or-id> [--language <codes>] [--timestamps] [--text-only]
  arxiv [query] [--author <name>] [--category <category>] [--id <ids>] [--max <count>] [--sort relevance|date|updated]
  polymarket <search|trending|market|event|price|book|history|trades> [arguments]
  google-workspace [--adapter <module>] <service> <action> [arguments]
  powerpoint <add-slide|clean|pack|merge-runs|simplify-redlines> [arguments]
  ocr-and-documents [--adapter <module>] <pymupdf|marker> [arguments]
  grpo-rl-training --dry-run --dataset <gsm8k.jsonl>

These skill commands use native Bun/TypeScript implementations. Google Workspace
requires an explicit caller-selected adapter for OAuth and credential storage.
OCR/document extraction requires an explicit caller-selected Bun-native converter.
GRPO execution requires explicitly injected accelerator and storage ports.
No Python executable, Python package, or Python skill script is used.`;

export interface BundledSkillCliDependencies {
  /** Optional fetch seam for tests and constrained hosts. */
  readonly fetchImplementation?: SkillFetch;
  /** Optional file-reading seam for Excalidraw uploads. */
  readonly readTextFile?: (path: string) => Promise<string> | string;
  /** Optional output seam for tests and embedding hosts. */
  readonly writeLine?: (line: string) => Promise<void> | void;
  /** Explicit host ports for the Google Workspace native CLI. */
  readonly googleWorkspace?: Omit<GoogleWorkspaceCliDependencies, "writeLine">;
  /** Explicit host ports for the native PowerPoint CLI. */
  readonly powerpoint?: Omit<PowerPointCliDependencies, "writeLine">;
  /** Explicit host ports for the OCR/document native CLI. */
  readonly ocrDocuments?: Omit<
    OcrDocumentsCliDependencies,
    "writeDiagnostic" | "writeLine"
  >;
  /** Explicit host ports for the native GRPO training workflow. */
  readonly grpoTraining?: Omit<GrpoTemplateCliDependencies, "writeLine">;
}

interface ParsedArguments {
  readonly flags: ReadonlyMap<string, readonly string[]>;
  readonly positionals: readonly string[];
}

interface FlagParserOptions {
  readonly aliases?: Readonly<Record<string, string>>;
  readonly booleanFlags?: readonly string[];
  readonly valueFlags?: readonly string[];
}

class SkillCliUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "SkillCliUsageError";
  }
}

/** Dispatch a documented bundled-skill command without a Python subprocess fallback. */
export async function runBundledSkillCli(
  args: readonly string[],
  dependencies: BundledSkillCliDependencies = {},
): Promise<number> {
  const writeLine = dependencies.writeLine ?? ((line) => console.log(line));
  const [skill, ...rest] = args;
  if (skill === undefined || isHelp(skill)) {
    await writeLine(BUNDLED_SKILL_CLI_USAGE);
    return 0;
  }

  try {
    switch (skill) {
      case "excalidraw":
        await runExcalidrawCli(rest, dependencies, writeLine);
        return 0;
      case "find-nearby":
        await runNearbyCli(rest, dependencies, writeLine);
        return 0;
      case "youtube-transcript":
        await runYoutubeTranscriptCli(rest, dependencies, writeLine);
        return 0;
      case "arxiv":
        await runArxivCli(rest, dependencies, writeLine);
        return 0;
      case "polymarket":
        await runPolymarketCli(rest, dependencies, writeLine);
        return 0;
      case "google-workspace":
        return runGoogleWorkspaceCli(rest, {
          ...(dependencies.googleWorkspace ?? {}),
          writeLine,
        });
      case "powerpoint":
        return runPowerPointCli(rest, {
          ...(dependencies.powerpoint ?? {}),
          writeLine,
        });
      case "ocr-and-documents":
        return runOcrDocumentsCli(rest, {
          ...(dependencies.ocrDocuments ?? {}),
          writeLine,
        });
      case "grpo-rl-training": {
        const grpoTraining = dependencies.grpoTraining ?? {};
        return runGrpoTemplateCli(rest, {
          ...grpoTraining,
          readTextFile:
            grpoTraining.readTextFile ??
            (async (path) => await readFile(path, "utf8")),
          writeLine,
        });
      }
      default:
        throw new SkillCliUsageError(`Unknown native bundled skill: ${skill}`);
    }
  } catch (error) {
    await writeLine(errorMessage(error));
    return error instanceof SkillCliUsageError ? 2 : 1;
  }
}

/** Standalone Bun entry point for direct execution during development. */
export async function main(
  args: readonly string[] = process.argv.slice(2),
): Promise<number> {
  return runBundledSkillCli(args);
}

async function runExcalidrawCli(
  args: readonly string[],
  dependencies: BundledSkillCliDependencies,
  writeLine: (line: string) => Promise<void> | void,
): Promise<void> {
  if (args.some(isHelp)) {
    await writeLine("Usage: xerxes skill excalidraw upload <file.excalidraw>");
    return;
  }
  const [action, path, ...extra] = args;
  if (action !== "upload" || path === undefined || extra.length) {
    throw new SkillCliUsageError(
      "Usage: xerxes skill excalidraw upload <file.excalidraw>",
    );
  }
  const contents = await (
    dependencies.readTextFile ?? (async (file) => await readFile(file, "utf8"))
  )(path);
  const url = await uploadExcalidrawDocument(contents, {
    ...(dependencies.fetchImplementation === undefined
      ? {}
      : { fetchImplementation: dependencies.fetchImplementation }),
  });
  await writeLine(url);
}

async function runNearbyCli(
  args: readonly string[],
  dependencies: BundledSkillCliDependencies,
  writeLine: (line: string) => Promise<void> | void,
): Promise<void> {
  const parsed = parseFlags(args, {
    booleanFlags: ["--help", "--json"],
    valueFlags: ["--lat", "--lon", "--near", "--type", "--radius", "--limit"],
  });
  if (hasFlag(parsed, "--help")) {
    await writeLine(
      "Usage: xerxes skill find-nearby (--near <place> | --lat <latitude> --lon <longitude>) [--type <amenity>] [--radius <meters>] [--limit <count>] [--json]",
    );
    return;
  }
  requireNoPositionals(parsed, "find-nearby");
  const near = singleValue(parsed, "--near");
  const lat = singleValue(parsed, "--lat");
  const lon = singleValue(parsed, "--lon");
  if (near !== undefined && (lat !== undefined || lon !== undefined)) {
    throw new SkillCliUsageError(
      "find-nearby accepts either --near or --lat with --lon, not both",
    );
  }
  const types = values(parsed, "--type");
  const client = new FindNearbyClient({
    ...(dependencies.fetchImplementation === undefined
      ? {}
      : { fetchImplementation: dependencies.fetchImplementation }),
  });
  const origin =
    near !== undefined
      ? await client.geocode(near)
      : coordinatesFromFlags(lat, lon);
  const radius = positiveInteger(
    singleValue(parsed, "--radius") ?? "1500",
    "--radius",
  );
  const limit = nonNegativeInteger(
    singleValue(parsed, "--limit") ?? "15",
    "--limit",
  );
  const amenities = types.length ? types : ["restaurant"];
  const places = await client.findNearby(origin, {
    limit,
    radius,
    types: amenities,
  });
  if (hasFlag(parsed, "--json")) {
    await writeLine(
      JSON.stringify(
        {
          count: places.length,
          origin,
          results: places.map(nearbyJsonResult),
        },
        null,
        2,
      ),
    );
    return;
  }
  await writeLine(formatNearbyPlaces(places, amenities, radius));
}

async function runYoutubeTranscriptCli(
  args: readonly string[],
  dependencies: BundledSkillCliDependencies,
  writeLine: (line: string) => Promise<void> | void,
): Promise<void> {
  const parsed = parseFlags(args, {
    aliases: { "-h": "--help", "-l": "--language", "-t": "--timestamps" },
    booleanFlags: ["--help", "--timestamps", "--text-only"],
    valueFlags: ["--language"],
  });
  if (hasFlag(parsed, "--help")) {
    await writeLine(
      "Usage: xerxes skill youtube-transcript <url-or-id> [--language en,tr] [--timestamps] [--text-only]",
    );
    return;
  }
  const video = exactlyOnePositional(
    parsed,
    "youtube-transcript requires one YouTube URL or video ID",
  );
  const language = singleValue(parsed, "--language");
  const languages =
    language === undefined
      ? undefined
      : language
          .split(",")
          .map((value) => value.trim())
          .filter(Boolean);
  const summary = await new YoutubeTranscriptClient({
    ...(dependencies.fetchImplementation === undefined
      ? {}
      : { fetchImplementation: dependencies.fetchImplementation }),
  }).summarize(video, {
    ...(languages === undefined ? {} : { languages }),
    ...(hasFlag(parsed, "--timestamps") ? { timestamps: true } : {}),
  });
  if (hasFlag(parsed, "--text-only")) {
    await writeLine(
      hasFlag(parsed, "--timestamps")
        ? (summary.timestampedText ?? "")
        : summary.fullText,
    );
    return;
  }
  await writeLine(
    JSON.stringify(
      {
        duration: summary.duration,
        full_text: summary.fullText,
        segment_count: summary.segmentCount,
        ...(summary.timestampedText === undefined
          ? {}
          : { timestamped_text: summary.timestampedText }),
        video_id: summary.videoId,
      },
      null,
      2,
    ),
  );
}

async function runArxivCli(
  args: readonly string[],
  dependencies: BundledSkillCliDependencies,
  writeLine: (line: string) => Promise<void> | void,
): Promise<void> {
  const parsed = parseFlags(args, {
    booleanFlags: ["--help"],
    valueFlags: ["--author", "--category", "--id", "--max", "--sort"],
  });
  if (hasFlag(parsed, "--help")) {
    await writeLine(
      "Usage: xerxes skill arxiv [query] [--author <name>] [--category <category>] [--id <ids>] [--max <count>] [--sort relevance|date|updated]",
    );
    return;
  }
  const author = singleValue(parsed, "--author");
  const category = singleValue(parsed, "--category");
  const ids = singleValue(parsed, "--id");
  const maximum = singleValue(parsed, "--max");
  const sort = singleValue(parsed, "--sort");
  const result = await new ArxivClient({
    ...(dependencies.fetchImplementation === undefined
      ? {}
      : { fetchImplementation: dependencies.fetchImplementation }),
  }).search({
    ...(parsed.positionals.length
      ? { query: parsed.positionals.join(" ") }
      : {}),
    ...(author === undefined ? {} : { author }),
    ...(category === undefined ? {} : { category }),
    ...(ids === undefined ? {} : { ids }),
    ...(maximum === undefined
      ? {}
      : {
          maxResults: nonNegativeInteger(maximum, "--max"),
        }),
    ...(sort === undefined ? {} : { sort }),
  });
  await writeLine(formatArxivResult(result));
}

async function runPolymarketCli(
  args: readonly string[],
  dependencies: BundledSkillCliDependencies,
  writeLine: (line: string) => Promise<void> | void,
): Promise<void> {
  const [action, ...rest] = args;
  if (action === undefined || isHelp(action)) {
    await writeLine(polymarketUsage());
    return;
  }
  const client = new PolymarketClient({
    ...(dependencies.fetchImplementation === undefined
      ? {}
      : { fetchImplementation: dependencies.fetchImplementation }),
  });
  switch (action) {
    case "search": {
      const parsed = parseFlags(rest, { booleanFlags: ["--help"] });
      if (hasFlag(parsed, "--help")) {
        await writeLine("Usage: xerxes skill polymarket search <query>");
        return;
      }
      const query = exactlyOneOrMorePositionals(
        parsed,
        "polymarket search requires a query",
      ).join(" ");
      await writeLine(formatPolymarketSearch(await client.search(query)));
      return;
    }
    case "trending": {
      const parsed = parseFlags(rest, {
        booleanFlags: ["--help"],
        valueFlags: ["--limit"],
      });
      if (hasFlag(parsed, "--help")) {
        await writeLine(
          "Usage: xerxes skill polymarket trending [--limit <count>]",
        );
        return;
      }
      requireNoPositionals(parsed, "polymarket trending");
      const events = await client.trending(
        positiveInteger(singleValue(parsed, "--limit") ?? "10", "--limit"),
      );
      await writeLine(formatPolymarketTrending(events));
      return;
    }
    case "market": {
      const parsed = parseFlags(rest, { booleanFlags: ["--help"] });
      if (hasFlag(parsed, "--help")) {
        await writeLine("Usage: xerxes skill polymarket market <slug>");
        return;
      }
      const market = await client.market(
        exactlyOnePositional(parsed, "polymarket market requires a slug"),
      );
      await writeLine(
        market === undefined
          ? "No market found."
          : formatPolymarketDetail(market),
      );
      return;
    }
    case "event": {
      const parsed = parseFlags(rest, { booleanFlags: ["--help"] });
      if (hasFlag(parsed, "--help")) {
        await writeLine("Usage: xerxes skill polymarket event <slug>");
        return;
      }
      const event = await client.event(
        exactlyOnePositional(parsed, "polymarket event requires a slug"),
      );
      await writeLine(
        event === undefined ? "No event found." : formatPolymarketEvent(event),
      );
      return;
    }
    case "price": {
      const parsed = parseFlags(rest, { booleanFlags: ["--help"] });
      if (hasFlag(parsed, "--help")) {
        await writeLine("Usage: xerxes skill polymarket price <token-id>");
        return;
      }
      const price = await client.price(
        exactlyOnePositional(parsed, "polymarket price requires a token ID"),
      );
      await writeLine(
        [
          `Token: ${shortToken(price.tokenId)}`,
          `  Buy price: ${formatPolymarketPercentage(price.buy)}`,
          `  Midpoint:  ${formatPolymarketPercentage(price.midpoint)}`,
          `  Spread:    ${price.spread}`,
        ].join("\n"),
      );
      return;
    }
    case "book": {
      const parsed = parseFlags(rest, { booleanFlags: ["--help"] });
      if (hasFlag(parsed, "--help")) {
        await writeLine("Usage: xerxes skill polymarket book <token-id>");
        return;
      }
      await writeLine(
        formatPolymarketBook(
          await client.book(
            exactlyOnePositional(parsed, "polymarket book requires a token ID"),
          ),
        ),
      );
      return;
    }
    case "history": {
      const parsed = parseFlags(rest, {
        booleanFlags: ["--help"],
        valueFlags: ["--fidelity", "--interval"],
      });
      if (hasFlag(parsed, "--help")) {
        await writeLine(
          "Usage: xerxes skill polymarket history <condition-id> [--interval <interval>] [--fidelity <count>]",
        );
        return;
      }
      const conditionId = exactlyOnePositional(
        parsed,
        "polymarket history requires a condition ID",
      );
      const history = await client.history(conditionId, {
        fidelity: positiveInteger(
          singleValue(parsed, "--fidelity") ?? "50",
          "--fidelity",
        ),
        interval: singleValue(parsed, "--interval") ?? "all",
      });
      await writeLine(
        formatPolymarketHistory(
          history,
          singleValue(parsed, "--interval") ?? "all",
        ),
      );
      return;
    }
    case "trades": {
      const parsed = parseFlags(rest, {
        booleanFlags: ["--help"],
        valueFlags: ["--limit", "--market"],
      });
      if (hasFlag(parsed, "--help")) {
        await writeLine(
          "Usage: xerxes skill polymarket trades [--limit <count>] [--market <condition-id>]",
        );
        return;
      }
      requireNoPositionals(parsed, "polymarket trades");
      const market = singleValue(parsed, "--market");
      const trades = await client.trades({
        limit: positiveInteger(
          singleValue(parsed, "--limit") ?? "10",
          "--limit",
        ),
        ...(market === undefined ? {} : { market }),
      });
      await writeLine(formatPolymarketTrades(trades));
      return;
    }
    default:
      throw new SkillCliUsageError(
        `Unknown Polymarket command: ${action}\n${polymarketUsage()}`,
      );
  }
}

function parseFlags(
  args: readonly string[],
  options: FlagParserOptions,
): ParsedArguments {
  const aliases = options.aliases ?? {};
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
    const name = aliases[argument] ?? argument;
    if (!positionalOnly && booleanFlags.has(name)) {
      appendFlag(flags, name, "true");
      continue;
    }
    if (!positionalOnly && valueFlags.has(name)) {
      const value = args[index + 1];
      if (value === undefined)
        throw new SkillCliUsageError(`${name} requires a value`);
      appendFlag(flags, name, value);
      index += 1;
      continue;
    }
    if (!positionalOnly && argument.startsWith("-")) {
      throw new SkillCliUsageError(`Unknown option: ${argument}`);
    }
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

function values(parsed: ParsedArguments, name: string): readonly string[] {
  return parsed.flags.get(name) ?? [];
}

function singleValue(
  parsed: ParsedArguments,
  name: string,
): string | undefined {
  const found = values(parsed, name);
  if (found.length > 1)
    throw new SkillCliUsageError(`${name} may only be provided once`);
  return found[0];
}

function requireNoPositionals(parsed: ParsedArguments, command: string): void {
  if (parsed.positionals.length)
    throw new SkillCliUsageError(
      `${command} does not accept positional arguments`,
    );
}

function exactlyOnePositional(
  parsed: ParsedArguments,
  message: string,
): string {
  if (parsed.positionals.length !== 1) throw new SkillCliUsageError(message);
  return parsed.positionals[0] ?? "";
}

function exactlyOneOrMorePositionals(
  parsed: ParsedArguments,
  message: string,
): readonly string[] {
  if (!parsed.positionals.length) throw new SkillCliUsageError(message);
  return parsed.positionals;
}

function coordinatesFromFlags(
  lat: string | undefined,
  lon: string | undefined,
): GeographicPoint {
  if (lat === undefined || lon === undefined) {
    throw new SkillCliUsageError(
      "find-nearby requires --near or both --lat and --lon",
    );
  }
  const point = {
    lat: finiteNumber(lat, "--lat"),
    lon: finiteNumber(lon, "--lon"),
  };
  if (point.lat < -90 || point.lat > 90)
    throw new SkillCliUsageError("--lat must be between -90 and 90");
  if (point.lon < -180 || point.lon > 180)
    throw new SkillCliUsageError("--lon must be between -180 and 180");
  return point;
}

function nearbyJsonResult(
  place: NearbyPlace,
): Readonly<Record<string, unknown>> {
  return {
    ...(place.address === undefined ? {} : { address: place.address }),
    ...(place.cuisine === undefined ? {} : { cuisine: place.cuisine }),
    directions_url: place.directionsUrl,
    distance_m: place.distanceMeters,
    ...(place.hours === undefined ? {} : { hours: place.hours }),
    lat: place.lat,
    lon: place.lon,
    maps_url: place.mapsUrl,
    name: place.name,
    ...(place.phone === undefined ? {} : { phone: place.phone }),
    type: place.type,
    ...(place.website === undefined ? {} : { website: place.website }),
  };
}

function formatPolymarketSearch(result: PolymarketSearchResult): string {
  if (!result.events.length) return `Found 0 results.`;
  const lines = [`Found ${result.totalResults} results:`, ""];
  for (const event of result.events.slice(0, 10)) {
    lines.push(
      `=== ${event.title} ===`,
      `  Volume: ${formatPolymarketVolume(event.volume)}  |  slug: ${event.slug}`,
    );
    for (const market of event.markets.slice(0, 5))
      lines.push(formatPolymarketMarket(market, "  "));
    if (event.markets.length > 5)
      lines.push(`  ... and ${event.markets.length - 5} more markets`);
    lines.push("");
  }
  return lines.join("\n").trimEnd();
}

function formatPolymarketTrending(events: readonly PolymarketEvent[]): string {
  const lines = [`Top ${events.length} trending events:`, ""];
  for (const [index, event] of events.entries()) {
    lines.push(
      `${index + 1}. ${event.title}`,
      `   Volume: ${formatPolymarketVolume(event.volume)}  |  Markets: ${event.markets.length}`,
      `   slug: ${event.slug}`,
    );
    for (const market of event.markets.slice(0, 3))
      lines.push(formatPolymarketMarket(market, "   "));
    if (event.markets.length > 3)
      lines.push(`   ... and ${event.markets.length - 3} more markets`);
    lines.push("");
  }
  return lines.join("\n").trimEnd();
}

function formatPolymarketDetail(market: PolymarketMarket): string {
  const lines = [
    `Market: ${market.question}`,
    `Status: ${market.closed ? "CLOSED" : "ACTIVE"}`,
    formatPolymarketMarket(market),
    "",
    `  conditionId: ${market.conditionId ?? "N/A"}`,
  ];
  for (const [index, token] of market.clobTokenIds.entries()) {
    lines.push(
      `  token (${market.outcomes[index] ?? `Outcome ${index}`}): ${token}`,
    );
  }
  if (market.description)
    lines.push("", `  Description: ${market.description.slice(0, 500)}`);
  return lines.join("\n");
}

function formatPolymarketEvent(event: PolymarketEvent): string {
  const lines = [
    `Event: ${event.title}`,
    `Volume: ${formatPolymarketVolume(event.volume)}`,
    `Status: ${event.closed ? "CLOSED" : "ACTIVE"}`,
    `Markets: ${event.markets.length}`,
    "",
  ];
  for (const market of event.markets)
    lines.push(formatPolymarketMarket(market, "  "), "");
  return lines.join("\n").trimEnd();
}

function formatPolymarketBook(book: PolymarketOrderBook): string {
  const lines = [
    `Orderbook for ${shortToken(book.tokenId)}`,
    `Last trade: ${formatPolymarketPercentage(book.lastTradePrice)}  |  Tick size: ${book.tickSize}`,
    "",
    `  Top bids (${book.bids.length} total):`,
    ...book.bids
      .slice(0, 10)
      .map(
        (order) =>
          `    ${formatPolymarketPercentage(order.price).padStart(7)}  |  Size: ${finiteNumericText(order.size).padStart(10)}`,
      ),
    "",
    `  Top asks (${book.asks.length} total):`,
    ...book.asks
      .slice(0, 10)
      .map(
        (order) =>
          `    ${formatPolymarketPercentage(order.price).padStart(7)}  |  Size: ${finiteNumericText(order.size).padStart(10)}`,
      ),
  ];
  return lines.join("\n");
}

function formatPolymarketHistory(
  history: readonly PolymarketHistoryPoint[],
  interval: string,
): string {
  if (!history.length) return "No price history available for this market.";
  const lines = [
    `Price history (${history.length} points, interval=${interval}):`,
    "",
  ];
  for (const point of history) {
    const date = new Date(point.timestamp * 1_000);
    const timestamp = Number.isNaN(date.valueOf())
      ? String(point.timestamp)
      : date.toISOString().slice(0, 16).replace("T", " ");
    lines.push(
      `  ${timestamp}  ${formatPolymarketPercentage(point.price).padStart(7)}  ${"█".repeat(Math.max(0, Math.floor(point.price * 40)))}`,
    );
  }
  return lines.join("\n");
}

function formatPolymarketTrades(trades: readonly PolymarketTrade[]): string {
  const lines = [`Recent trades (${trades.length}):`, ""];
  for (const trade of trades) {
    lines.push(
      `  ${trade.side.padEnd(4)}  ${formatPolymarketPercentage(trade.price).padStart(7)}  x${finiteNumericText(trade.size).padStart(8)}  [${trade.outcome}]  ${trade.title.slice(0, 50)}`,
    );
  }
  return lines.join("\n");
}

function polymarketUsage(): string {
  return "Usage: xerxes skill polymarket <search|trending|market|event|price|book|history|trades> [arguments]";
}

function finiteNumericText(value: string): string {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(2) : value;
}

function shortToken(value: string): string {
  return value.length > 30 ? `${value.slice(0, 30)}...` : value;
}

function positiveInteger(value: string, name: string): number {
  const parsed = integer(value, name);
  if (parsed < 1)
    throw new SkillCliUsageError(`${name} must be a positive integer`);
  return parsed;
}

function nonNegativeInteger(value: string, name: string): number {
  const parsed = integer(value, name);
  if (parsed < 0)
    throw new SkillCliUsageError(`${name} must be a non-negative integer`);
  return parsed;
}

function integer(value: string, name: string): number {
  if (!/^[-+]?\d+$/.test(value))
    throw new SkillCliUsageError(`${name} must be an integer`);
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed))
    throw new SkillCliUsageError(`${name} must be a safe integer`);
  return parsed;
}

function finiteNumber(value: string, name: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed))
    throw new SkillCliUsageError(`${name} must be a finite number`);
  return parsed;
}

function isHelp(value: string): boolean {
  return value === "--help" || value === "-h" || value === "help";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

if (import.meta.main) {
  process.exitCode = await main();
}
