// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readdir } from "node:fs/promises";
import { join } from "node:path";

import { computeDaemonBuildId } from "./fingerprint.js";

/**
 * Hash the complete Bun runtime source tree when Xerxes is running from a
 * checkout. Built distributions intentionally return undefined: their caller
 * supplies the release build identity instead of hashing absent TypeScript.
 */
export async function sourceDaemonBuildId(
  sourceRoot: string,
): Promise<string | undefined> {
  const marker = Bun.file(join(sourceRoot, "daemon", "server.ts"));
  if (!(await marker.exists())) return undefined;

  const files = await sourceFiles(sourceRoot);
  if (!files.length) return undefined;

  return computeDaemonBuildId({
    files,
    sourceRoot,
    sourceReader: {
      async readFile(root, relativePath) {
        const file = Bun.file(join(root, relativePath));
        if (!(await file.exists())) return undefined;
        return new Uint8Array(await file.arrayBuffer());
      },
    },
  });
}

/**
 * Resolve the identity used by a running CLI. Source checkouts hash their full
 * runtime tree; bundled releases hash the actual executable module bytes.
 * A missing entry returns undefined rather than fabricating an identity.
 */
export async function daemonBuildIdForEntry(
  sourceRoot: string,
  entryPath: string,
): Promise<string | undefined> {
  const sourceBuild = await sourceDaemonBuildId(sourceRoot);
  if (sourceBuild) return sourceBuild;

  const entry = Bun.file(entryPath);
  if (!(await entry.exists())) return undefined;
  const digest = new Bun.CryptoHasher("sha256");
  digest.update(new TextEncoder().encode("xerxes-bun-entry\0"));
  digest.update(new Uint8Array(await entry.arrayBuffer()));
  return digest.digest("hex").slice(0, 16);
}

async function sourceFiles(
  sourceRoot: string,
  relativeDirectory = "",
): Promise<string[]> {
  const directory = join(sourceRoot, relativeDirectory);
  const entries = await readdir(directory, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries.sort((left, right) =>
    left.name.localeCompare(right.name),
  )) {
    const relativePath = relativeDirectory
      ? `${relativeDirectory}/${entry.name}`
      : entry.name;
    if (entry.isDirectory()) {
      files.push(...(await sourceFiles(sourceRoot, relativePath)));
    } else if (
      entry.isFile() &&
      [".md", ".ts", ".tsx", ".yaml", ".yml"].some((extension) =>
        entry.name.endsWith(extension),
      )
    ) {
      files.push(relativePath);
    }
  }

  return files;
}
