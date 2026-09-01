// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * `~/.xerxes/mcp.json` loader. One bad entry must not take down the daemon's
 * whole MCP setup: invalid entries become collected warnings (observable in
 * the daemon log), valid siblings still load, and a missing file simply means
 * no servers. Secrets stay in the file — the loader never logs entry contents.
 */

import { existsSync, readFileSync } from "node:fs";

import type { MCPServerConfig } from "./types.js";

export interface McpLoadedConfig {
  readonly servers: readonly MCPServerConfig[];
  readonly warnings: readonly string[];
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value);

const text = (value: unknown): string => (typeof value === "string" ? value.trim() : "");

export function loadMcpConfig(path: string): McpLoadedConfig {
  if (!existsSync(path)) return { servers: [], warnings: [] };
  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(path, "utf8"));
  } catch (error) {
    return {
      servers: [],
      warnings: [`mcp.json at ${path} is not valid JSON: ${String(error)}`],
    };
  }
  const warnings: string[] = [];
  let entries: unknown[] = [];
  if (Array.isArray(parsed)) {
    entries = parsed;
  } else if (isRecord(parsed) && Array.isArray(parsed.servers)) {
    entries = parsed.servers;
  } else if (isRecord(parsed)) {
    // Loose map shape: { "name": { ...serverConfig } }.
    entries = Object.entries(parsed).map(([name, config]) => ({
      ...(isRecord(config) ? config : { __invalid: config }),
      name,
    }));
  } else {
    return { servers: [], warnings: [`mcp.json at ${path} is neither a server list nor a map`] };
  }
  const servers: MCPServerConfig[] = [];
  for (let index = 0; index < entries.length; index++) {
    const entry = entries[index];
    const label = isRecord(entry) && text(entry.name) ? `'${text(entry.name)}'` : `#${index + 1}`;
    if (!isRecord(entry)) {
      warnings.push(`mcp.json server ${label} is not an object — skipped`);
      continue;
    }
    const name = text(entry.name);
    if (!name) {
      warnings.push(`mcp.json server #${index + 1} has no name — skipped`);
      continue;
    }
    if (entry.enabled === false) continue; // documented opt-out, not a warning
    const hasCommand = text(entry.command) !== "";
    const hasUrl = text(entry.url) !== "";
    if (!hasCommand && !hasUrl) {
      warnings.push(`mcp.json server '${name}' needs a command or url — skipped`);
      continue;
    }
    servers.push(entry as unknown as MCPServerConfig);
  }
  return { servers, warnings };
}
