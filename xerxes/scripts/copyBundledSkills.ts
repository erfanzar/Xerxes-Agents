// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { cp, lstat, readdir, rm } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const packageDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const BUNDLED_SKILLS_SOURCE_DIRECTORY = resolve(
  packageDirectory,
  "skills",
);
export const BUNDLED_SKILLS_DESTINATION_DIRECTORY = resolve(
  packageDirectory,
  "dist",
  "skills",
);
export const BUNDLED_AGENTS_SOURCE_DIRECTORY = resolve(
  packageDirectory,
  "src",
  "agents",
  "default",
);
export const BUNDLED_AGENTS_DESTINATION_DIRECTORY = resolve(
  packageDirectory,
  "dist",
  "default",
);

export interface CopyBundledSkillsOptions {
  readonly destinationDirectory?: string;
  readonly sourceDirectory?: string;
}

/** Copy the native skill asset tree into a distributable runtime directory. */
export async function copyBundledSkills(
  options: CopyBundledSkillsOptions = {},
): Promise<void> {
  const sourceDirectory =
    options.sourceDirectory ?? BUNDLED_SKILLS_SOURCE_DIRECTORY;
  const destinationDirectory =
    options.destinationDirectory ?? BUNDLED_SKILLS_DESTINATION_DIRECTORY;

  await assertSafeSkillAssets(sourceDirectory);
  await rm(destinationDirectory, { force: true, recursive: true });
  await cp(sourceDirectory, destinationDirectory, { recursive: true });
}

/** Copy built-in YAML agents beside the bundled CLI so discovery is identical after install. */
export async function copyBundledAgents(
  sourceDirectory = BUNDLED_AGENTS_SOURCE_DIRECTORY,
  destinationDirectory = BUNDLED_AGENTS_DESTINATION_DIRECTORY,
): Promise<void> {
  await assertSafeSkillAssets(sourceDirectory);
  await rm(destinationDirectory, { force: true, recursive: true });
  await cp(sourceDirectory, destinationDirectory, { recursive: true });
}

async function assertSafeSkillAssets(
  directory: string,
  prefix = "",
): Promise<void> {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
    const path = resolve(directory, entry.name);
    const metadata = await lstat(path);
    if (metadata.isSymbolicLink()) {
      throw new Error(
        `Bundled skill asset must not be a symlink: ${relativePath}`,
      );
    }
    if (metadata.isDirectory()) {
      await assertSafeSkillAssets(path, relativePath);
      continue;
    }
    if (!metadata.isFile()) {
      throw new Error(
        `Bundled skill asset must be a regular file: ${relativePath}`,
      );
    }
    if ((metadata.mode & 0o111) !== 0) {
      throw new Error(
        `Bundled skill asset must not be executable: ${relativePath}`,
      );
    }
  }
}

if (import.meta.main) {
  await copyBundledSkills();
  await copyBundledAgents();
}
