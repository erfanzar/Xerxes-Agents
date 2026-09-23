// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * `ui/PROTOCOL.md` calls itself the frozen contract between the daemon and its
 * clients, and AGENTS.md requires the daemon handler, the client and the docs to
 * move together. Nothing enforced that: the document had drifted 25 live RPC
 * methods and 8 emitted events behind the dispatcher, and it still listed 17
 * events that only the optional native bridge can produce.
 *
 * These assertions read the sources rather than a hand-kept list, so a new
 * method or event fails here instead of silently leaving the contract stale.
 */

import { describe, expect, test } from "bun:test";
import { Glob } from "bun";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const SOURCE_ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "src");

const read = (relative: string): Promise<string> =>
  Bun.file(join(SOURCE_ROOT, relative)).text();

async function readAll(prefixes: readonly string[]): Promise<{ path: string; text: string }[]> {
  const paths = (await Array.fromAsync(new Glob("**/*.ts").scan({ cwd: SOURCE_ROOT })))
    .filter(path => prefixes.some(prefix => path.startsWith(prefix)))
    .map(path => path.replaceAll("\\", "/"));
  return Promise.all(paths.map(async path => ({ path, text: await read(path) })));
}

/** Method names the dispatcher actually branches on, in all three spellings it uses. */
function dispatchableMethods(server: string): Set<string> {
  const methods = new Set<string>();
  for (const match of server.matchAll(/method === (["'])([a-zA-Z._]+)\1/g)) {
    methods.add(match[2]!);
  }
  // `["a", "b"].includes(method)` — whole families are routed this way.
  for (const match of server.matchAll(/\[((?:\s*(["'])[a-zA-Z._]+\2\s*,?\s*)+)\]\.includes\(method\)/g)) {
    for (const name of match[1]!.matchAll(/(["'])([a-zA-Z._]+)\1/g)) methods.add(name[2]!);
  }
  return methods;
}

/** Event names the daemon hands to a transport, as opposed to display-block or content-part kinds. */
function emittedEventNames(sources: readonly { text: string }[]): Set<string> {
  const names = new Set<string>();
  for (const { text } of sources) {
    for (const match of text.matchAll(
      /(?:this\.)?(?:emit|broadcast)\(\s*(?:[A-Za-z_][A-Za-z0-9_.]*\s*,\s*)?(["'])([a-z_][a-z_.]*)\1/g,
    )) {
      names.add(match[2]!);
    }
  }
  return names;
}

/** snake_case column of the event-name map, with the rows flagged as bridge-only. */
function documentedEvents(doc: string): { all: Set<string>; bridgeOnly: Set<string> } {
  const all = new Set<string>();
  const bridgeOnly = new Set<string>();
  let inTable = false;
  for (const line of doc.split("\n")) {
    if (line.includes("PascalCase (bridge alias)")) {
      inTable = true;
      continue;
    }
    if (inTable && !line.startsWith("|")) break;
    if (!inTable) continue;
    const cell = line.split("|")[2] ?? "";
    const name = cell.match(/`([a-z_]+)`/)?.[1];
    if (!name) continue;
    all.add(name);
    if (cell.includes("bridge only")) bridgeOnly.add(name);
  }
  return { all, bridgeOnly };
}

/**
 * A plain `includes` would accept a renamed row: `session.usage` is a substring
 * of `session.usageZZZ`. Require identifier boundaries so only the exact name
 * counts as documentation.
 */
function mentions(doc: string, name: string): boolean {
  const escaped = name.replaceAll(".", "\\.");
  return new RegExp(`(?<![A-Za-z0-9_.])${escaped}(?![A-Za-z0-9_.])`).test(doc);
}

describe("daemon ⇄ client protocol contract", () => {
  test("every dispatchable RPC method is documented", async () => {
    const [server, doc] = await Promise.all([read("daemon/server.ts"), read("ui/PROTOCOL.md")]);
    const methods = dispatchableMethods(server);

    // Guard the extraction itself: a refactor that replaces the if-chain must
    // update this test rather than silently assert over an empty set.
    expect(methods.size).toBeGreaterThan(100);

    const undocumented = [...methods].filter(method => !mentions(doc, method)).sort();
    expect(undocumented).toEqual([]);
  });

  test("no method table row documents a handler that no longer exists", async () => {
    const [server, doc] = await Promise.all([read("daemon/server.ts"), read("ui/PROTOCOL.md")]);
    const documented = new Set<string>();
    for (const line of doc.split("\n")) {
      if (!line.startsWith("|")) continue;
      for (const match of (line.split("|")[1] ?? "").matchAll(/`([a-z][a-zA-Z._]*)`/g)) {
        documented.add(match[1]!);
      }
    }
    expect(documented.size).toBeGreaterThan(50);

    const phantom = [...documented]
      .filter(method => !server.includes(`"${method}"`) && !server.includes(`'${method}'`))
      .sort();
    expect(phantom).toEqual([]);
  });

  test("every event the daemon emits appears in the event vocabulary", async () => {
    const [doc, daemonSources] = await Promise.all([read("ui/PROTOCOL.md"), readAll(["daemon/"])]);
    const emitted = emittedEventNames(daemonSources);

    expect(emitted.has("notification")).toBe(true);
    expect(emitted.has("session_title")).toBe(true);

    const undocumented = [...emitted].filter(name => !mentions(doc, name)).sort();
    expect(undocumented).toEqual([]);
  });

  test("rows flagged bridge-only have no daemon or streaming producer", async () => {
    const [doc, sources] = await Promise.all([
      read("ui/PROTOCOL.md"),
      readAll(["daemon/", "streaming/", "bridge/"]),
    ]);
    const { all, bridgeOnly } = documentedEvents(doc);
    expect(all.size).toBeGreaterThan(25);
    expect(bridgeOnly.size).toBeGreaterThan(0);

    const producedBy = (name: string): string[] => {
      const pattern = new RegExp(`\\b${name}\\b`);
      return [
        ...new Set(
          sources.filter(source => pattern.test(source.text)).map(source => source.path.split("/")[0]!),
        ),
      ];
    };

    // A flagged row that grew a live producer is now reachable, and the flag
    // would tell a client author to ignore an event it will actually receive.
    const nowLive = [...bridgeOnly].filter(name => producedBy(name).some(area => area !== "bridge")).sort();
    expect(nowLive).toEqual([]);

    // The converse: an unflagged row with no live producer is a handler a
    // socket client would write and never exercise.
    const unreachable = [...all]
      .filter(name => !bridgeOnly.has(name))
      .filter(name => {
        const areas = producedBy(name);
        return areas.length > 0 && !areas.some(area => area !== "bridge");
      })
      .sort();
    expect(unreachable).toEqual([]);
  });

  test("the reasoning wire name is the one the daemon actually emits", async () => {
    const [server, turnRunner] = await Promise.all([
      read("daemon/server.ts"),
      read("daemon/turnRunner.ts"),
    ]);

    // `PRODUCTIVE_TURN_EVENTS` gates goal-round productivity. It previously
    // listed the internal `thinking_part`, which no producer emits, so a round
    // that only reasoned was recorded as having produced nothing.
    const productive = server
      .match(/PRODUCTIVE_TURN_EVENTS[^=]*=\s*new Set\(\[([\s\S]*?)\]\)/)?.[1]
      ?.match(/"([a-z_]+)"/g)
      ?.map(entry => entry.replaceAll('"', ""));

    expect(productive).toBeDefined();
    expect(productive).toContain("think_part");
    expect(productive).not.toContain("thinking_part");
    expect(turnRunner).toContain("'think_part'");
  });

  test("queue-bypass methods stay a subset of the concurrency-safe methods", async () => {
    const server = await read("daemon/server.ts");
    const names = (label: string): string[] =>
      (server.match(new RegExp(`${label}[^=]*=[^[]*\\[([\\s\\S]*?)\\]`))?.[1]?.match(/"([a-zA-Z._]+)"/g) ?? [])
        .map(entry => entry.replaceAll('"', ""));

    const bypass = names("QUEUE_BYPASS_METHODS");
    expect(bypass.length).toBeGreaterThan(0);

    // The two tiers were independent hand-kept lists with different members.
    // Concurrency-safety is now derived from the bypass set; this keeps it so.
    expect(server).toContain("...QUEUE_BYPASS_METHODS");
    for (const method of bypass) {
      expect(server.includes(`"${method}"`) || server.includes(`'${method}'`)).toBe(true);
    }
  });
});
